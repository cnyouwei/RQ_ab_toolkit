#include "wck/wck_table/solver.hpp"

#include "wck/common/parallel.hpp"
#include "wck/wck_table/tridiag.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace wck {

namespace {

constexpr double kClampEps = 1e-8;
constexpr double kTiny = 1e-15;
constexpr double kMinTailCheckTime = 1.0;

struct PreparedSpace {
    int nx = 0;
    int interior = 0;
    double dx = 0.0;
    double x_max = 0.0;
    std::vector<double> x{};
    std::vector<double> pi{};
    std::vector<double> psi_lower{};
    std::vector<double> psi_diag{};
    std::vector<double> psi_upper{};
    std::vector<double> psi_force{};
    std::vector<double> h_lower{};
    std::vector<double> h_diag{};
    std::vector<double> h_upper{};
    std::vector<double> h_force{};
};

struct HMoments {
    double mean = 0.0;
    double variance = 0.0;
};

double int_pow(double base, int exp) {
    if (exp < 0) {
        throw std::invalid_argument("int_pow expects nonnegative exponent");
    }
    double result = 1.0;
    double x = base;
    int e = exp;
    while (e > 0) {
        if ((e & 1) != 0) {
            result *= x;
        }
        x *= x;
        e >>= 1;
    }
    return result;
}

double log_pi_unnormalized(double x, double c, int k) {
    const int kp1 = k + 1;
    return c * x - int_pow(x, kp1) / static_cast<double>(kp1);
}

double clamp_nonnegative(double value) {
    if (value < 0.0 && value > -kClampEps) {
        return 0.0;
    }
    return value;
}

double clamp_w(double value) {
    double w = clamp_nonnegative(value);
    if (w < 0.0) {
        w = 0.0;
    }
    return w;
}

double clamp_psi(double value) {
    double psi = clamp_nonnegative(value);
    if (psi < 0.0) {
        psi = 0.0;
    }
    if (psi > 1.0 + kClampEps) {
        psi = 1.0 + kClampEps;
    }
    return psi;
}

void validate_params(const WckParams& params, const SolverConfig& config) {
    if (params.k < 1) {
        throw std::invalid_argument("k must be >= 1");
    }
    if (!(config.time.t_min > 0.0)) {
        throw std::invalid_argument("t_min must be > 0");
    }
    if (!(config.time.t_max > config.time.t_min)) {
        throw std::invalid_argument("t_max must be > t_min");
    }
    if (config.time.n_points < 2) {
        throw std::invalid_argument("n_points must be >= 2");
    }
    if (config.space.nx < 100) {
        throw std::invalid_argument("nx must be >= 100");
    }
    if (!config.space.auto_x_max && !(config.space.x_max > 0.0)) {
        throw std::invalid_argument("x_max must be > 0 when auto_x_max is false");
    }
    if (!(config.space.tail_tol > 0.0 && config.space.tail_tol < 1.0)) {
        throw std::invalid_argument("tail_tol must be in (0,1)");
    }
    if (!(config.rel_dt_max > 0.0)) {
        throw std::invalid_argument("rel_dt_max must be > 0");
    }
    if (!(config.steady_tol > 0.0)) {
        throw std::invalid_argument("steady_tol must be > 0");
    }
    if (config.steady_window < 5) {
        throw std::invalid_argument("steady_window must be >= 5");
    }
}

std::vector<double> build_log_time_grid(const TimeGridConfig& config) {
    const double log_min = std::log(config.t_min);
    const double log_max = std::log(config.t_max);
    const int n = config.n_points;

    std::vector<double> grid(static_cast<std::size_t>(n), 0.0);
    const double step = (log_max - log_min) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        grid[static_cast<std::size_t>(i)] = std::exp(log_min + step * static_cast<double>(i));
    }
    grid.back() = config.t_max;
    return grid;
}

double choose_auto_x_max(const WckParams& params, const SpaceGridConfig& config) {
    const double c = params.c;
    const int k = params.k;

    double x_mode = 0.0;
    if (c > 0.0) {
        x_mode = std::exp(std::log(c) / static_cast<double>(k));
    }

    const double log_mode = log_pi_unnormalized(x_mode, c, k);
    const double log_target = std::log(config.tail_tol);

    double x_max = std::max(8.0, x_mode + 8.0);
    constexpr int kMaxGrowIters = 120;
    for (int it = 0; it < kMaxGrowIters; ++it) {
        const double ratio_log = log_pi_unnormalized(x_max, c, k) - log_mode;
        if (ratio_log <= log_target) {
            return x_max;
        }
        x_max *= 1.35;
    }

    throw std::runtime_error("failed to choose x_max: tail criterion not met");
}

double trapz(const std::vector<double>& values, double dx) {
    if (values.size() < 2U) {
        return 0.0;
    }
    double sum = 0.5 * values.front() + 0.5 * values.back();
    for (std::size_t i = 1; i + 1 < values.size(); ++i) {
        sum += values[i];
    }
    return sum * dx;
}

PreparedSpace prepare_space(const WckParams& params, const SpaceGridConfig& config) {
    PreparedSpace prepared{};
    prepared.nx = config.nx;
    prepared.interior = config.nx - 1;
    prepared.x_max = config.auto_x_max ? choose_auto_x_max(params, config) : config.x_max;
    prepared.dx = prepared.x_max / static_cast<double>(prepared.nx);

    const int nx = prepared.nx;
    const int k = params.k;
    const double c = params.c;
    const int interior = prepared.interior;
    const double dx = prepared.dx;
    const double inv_dx = 1.0 / dx;
    const double inv_dx2 = inv_dx * inv_dx;

    prepared.x.assign(static_cast<std::size_t>(nx + 1), 0.0);
    for (int i = 0; i <= nx; ++i) {
        prepared.x[static_cast<std::size_t>(i)] = static_cast<double>(i) * dx;
    }

    std::vector<double> log_pi(static_cast<std::size_t>(nx + 1), 0.0);
    double max_log = -std::numeric_limits<double>::infinity();
    for (int i = 0; i <= nx; ++i) {
        const double lp = log_pi_unnormalized(prepared.x[static_cast<std::size_t>(i)], c, k);
        log_pi[static_cast<std::size_t>(i)] = lp;
        max_log = std::max(max_log, lp);
    }

    std::vector<double> scaled(static_cast<std::size_t>(nx + 1), 0.0);
    for (int i = 0; i <= nx; ++i) {
        scaled[static_cast<std::size_t>(i)] =
            std::exp(log_pi[static_cast<std::size_t>(i)] - max_log);
    }
    const double integral_scaled = trapz(scaled, dx);
    if (!(integral_scaled > 0.0)) {
        throw std::runtime_error("stationary density normalization integral is nonpositive");
    }
    const double log_norm = max_log + std::log(integral_scaled);

    prepared.pi.assign(static_cast<std::size_t>(nx + 1), 0.0);
    for (int i = 0; i <= nx; ++i) {
        prepared.pi[static_cast<std::size_t>(i)] =
            std::exp(log_pi[static_cast<std::size_t>(i)] - log_norm);
    }
    const double pi_mass = trapz(prepared.pi, dx);
    if (!(pi_mass > 0.0)) {
        throw std::runtime_error("stationary density mass is nonpositive");
    }
    for (double& v : prepared.pi) {
        v /= pi_mass;
    }

    prepared.psi_lower.assign(static_cast<std::size_t>(interior), 0.0);
    prepared.psi_diag.assign(static_cast<std::size_t>(interior), 0.0);
    prepared.psi_upper.assign(static_cast<std::size_t>(interior), 0.0);
    prepared.psi_force.assign(static_cast<std::size_t>(interior), 0.0);
    prepared.h_lower.assign(static_cast<std::size_t>(interior), 0.0);
    prepared.h_diag.assign(static_cast<std::size_t>(interior), 0.0);
    prepared.h_upper.assign(static_cast<std::size_t>(interior), 0.0);
    prepared.h_force.assign(static_cast<std::size_t>(interior), 0.0);

    for (int i = 1; i <= nx - 1; ++i) {
        const std::size_t j = static_cast<std::size_t>(i - 1);
        const double x_i = prepared.x[static_cast<std::size_t>(i)];
        const double b_i = c - int_pow(x_i, k);
        const double q_i = static_cast<double>(k) * int_pow(x_i, k - 1);
        const double xk_i = int_pow(x_i, k);
        const bool use_central = (std::abs(b_i) * dx <= 2.0);

        double base_lower = 0.0;
        double base_diag = 0.0;
        double base_upper = 0.0;

        if (i == nx - 1) {
            base_lower += inv_dx2;
            base_diag += -inv_dx2;
        } else {
            base_lower += inv_dx2;
            base_diag += -2.0 * inv_dx2;
            base_upper += inv_dx2;
        }

        if (use_central) {
            if (i == nx - 1) {
                base_lower += -0.5 * b_i * inv_dx;
                base_diag += 0.5 * b_i * inv_dx;
            } else {
                base_lower += -0.5 * b_i * inv_dx;
                base_upper += 0.5 * b_i * inv_dx;
            }
        } else if (b_i >= 0.0) {
            // Generator-form upwinding differences along the drift so the
            // off-diagonal stays nonnegative (Metzler); the reversed choice
            // is anti-diffusive and unstable exactly when |b|dx > 2. For
            // b >= 0 that is the forward difference; at i == nx-1 the
            // zero-gradient boundary value makes the term vanish.
            if (i < nx - 1) {
                base_diag += -b_i * inv_dx;
                base_upper += b_i * inv_dx;
            }
        } else {
            base_lower += -b_i * inv_dx;
            base_diag += b_i * inv_dx;
        }

        double psi_lower = base_lower;
        double psi_diag = base_diag - q_i;
        double psi_upper = base_upper;
        double psi_force = 0.0;
        if (i - 1 >= 1) {
            prepared.psi_lower[j] = psi_lower;
        } else {
            psi_force += psi_lower;
        }
        prepared.psi_diag[j] = psi_diag;
        if (i + 1 <= nx - 1) {
            prepared.psi_upper[j] = psi_upper;
        }
        prepared.psi_force[j] = psi_force;

        double h_lower = base_lower;
        double h_diag = base_diag;
        double h_upper = base_upper;
        if (i - 1 >= 1) {
            prepared.h_lower[j] = h_lower;
        } else {
            // Neumann BC at x=0: h(t,0) = h(t,dx), fold boundary coefficient into diagonal.
            h_diag += h_lower;
        }
        prepared.h_diag[j] = h_diag;
        if (i + 1 <= nx - 1) {
            prepared.h_upper[j] = h_upper;
        }
        prepared.h_force[j] = xk_i;
    }

    return prepared;
}

double compute_m(
    const std::vector<double>& psi,
    const std::vector<double>& pi,
    double dx) {
    if (psi.size() != pi.size()) {
        throw std::invalid_argument("psi and pi size mismatch");
    }
    std::vector<double> weighted(psi.size(), 0.0);
    for (std::size_t i = 0; i < psi.size(); ++i) {
        weighted[i] = psi[i] * psi[i] * pi[i];
    }
    double m = trapz(weighted, dx);
    m = clamp_nonnegative(m);
    if (m < 0.0) {
        m = 0.0;
    }
    return m;
}

HMoments compute_h_moments(
    const std::vector<double>& h,
    const std::vector<double>& pi,
    double dx) {
    if (h.size() != pi.size()) {
        throw std::invalid_argument("h and pi size mismatch");
    }
    std::vector<double> weighted(h.size(), 0.0);
    for (std::size_t i = 0; i < h.size(); ++i) {
        weighted[i] = h[i] * pi[i];
    }
    HMoments out{};
    out.mean = trapz(weighted, dx);
    // h grows like E_pi[x^k] * t, so E[h^2] - E[h]^2 cancels catastrophically
    // at large t (absolute noise ~ (mean)^2 * eps swamps the O(1) variance and
    // keeps the steady-state window from ever triggering). Center first: the
    // integrand stays O(variance) and the result keeps full relative precision.
    for (std::size_t i = 0; i < h.size(); ++i) {
        const double dev = h[i] - out.mean;
        weighted[i] = dev * dev * pi[i];
    }
    const double raw_var = trapz(weighted, dx);
    if (!std::isfinite(out.mean) || !std::isfinite(raw_var)) {
        throw std::runtime_error("h moments became non-finite (solver instability)");
    }
    out.variance = raw_var;
    return out;
}

void fill_full_psi_from_interior(
    const std::vector<double>& interior,
    std::vector<double>& full) {
    const std::size_t nx = full.size() - 1U;
    full[0] = 1.0;
    for (std::size_t j = 0; j < interior.size(); ++j) {
        full[j + 1U] = interior[j];
    }
    full[nx] = interior.back();
}

void fill_full_h_from_interior(
    const std::vector<double>& interior,
    std::vector<double>& full) {
    const std::size_t nx = full.size() - 1U;
    full[0] = interior.front();
    for (std::size_t j = 0; j < interior.size(); ++j) {
        full[j + 1U] = interior[j];
    }
    full[nx] = interior.back();
}

void crank_nicolson_step(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& force,
    double dt,
    std::vector<double>& state,
    std::vector<double>& lower_a,
    std::vector<double>& diag_a,
    std::vector<double>& upper_a,
    std::vector<double>& rhs) {
    const int m = static_cast<int>(state.size());
    for (int j = 0; j < m; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        lower_a[idx] = -0.5 * dt * lower[idx];
        diag_a[idx] = 1.0 - 0.5 * dt * diag[idx];
        upper_a[idx] = -0.5 * dt * upper[idx];

        double rhs_j = state[idx];
        rhs_j += 0.5 * dt * diag[idx] * state[idx];
        rhs_j += dt * force[idx];
        if (j > 0) {
            rhs_j += 0.5 * dt * lower[idx] * state[idx - 1U];
        }
        if (j + 1 < m) {
            rhs_j += 0.5 * dt * upper[idx] * state[idx + 1U];
        }
        rhs[idx] = rhs_j;
    }
    solve_tridiagonal(lower_a, diag_a, upper_a, rhs);
    state.swap(rhs);
}

}  // namespace

double estimate_stationary_mass(
    const WckParams& params,
    const SpaceGridConfig& config,
    double* x_max_used) {
    SolverConfig default_config{};
    default_config.space = config;
    validate_params(params, default_config);

    const PreparedSpace prepared = prepare_space(params, config);
    if (x_max_used != nullptr) {
        *x_max_used = prepared.x_max;
    }
    return trapz(prepared.pi, prepared.dx);
}

SolveResult solve_w_grid(const WckParams& params, const SolverConfig& config) {
    validate_params(params, config);

    const std::vector<double> time_grid = build_log_time_grid(config.time);
    const PreparedSpace prepared = prepare_space(params, config.space);

    const int m = prepared.interior;
    std::vector<double> psi_interior(static_cast<std::size_t>(m), 1.0);
    std::vector<double> h_interior(static_cast<std::size_t>(m), 0.0);
    std::vector<double> psi_full(static_cast<std::size_t>(prepared.nx + 1), 1.0);
    std::vector<double> h_full(static_cast<std::size_t>(prepared.nx + 1), 0.0);
    fill_full_psi_from_interior(psi_interior, psi_full);
    fill_full_h_from_interior(h_interior, h_full);

    double t = 0.0;
    double m_prev = compute_m(psi_full, prepared.pi, prepared.dx);
    double var_h_prev = compute_h_moments(h_full, prepared.pi, prepared.dx).variance;
    double integral_m = 0.0;
    int steps = 0;

    std::vector<double> lower_a(static_cast<std::size_t>(m), 0.0);
    std::vector<double> diag_a(static_cast<std::size_t>(m), 0.0);
    std::vector<double> upper_a(static_cast<std::size_t>(m), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(m), 0.0);

    std::deque<double> m_window{};
    std::deque<double> var_h_window{};
    const int window = config.steady_window;
    const double min_tail_time = std::max(kMinTailCheckTime, 1000.0 * config.time.t_min);
    bool use_tail = false;
    double t_ss = 0.0;
    double integral_ss = 0.0;
    double m_inf = m_prev;
    double var_h_inf = var_h_prev;

    SolveResult result{};
    result.points.reserve(static_cast<std::size_t>(config.time.n_points + (config.time.include_t0 ? 1 : 0)));
    if (config.time.include_t0) {
        result.points.push_back(GridPoint{0.0, 1.0, 1.0, 0.0});
    }

    for (const double t_target : time_grid) {
        if (use_tail) {
            const double integral_target = integral_ss + m_inf * (t_target - t_ss);
            const double m_term = integral_target / t_target;
            const double var_h_term = var_h_inf / (2.0 * t_target);
            result.points.push_back(GridPoint{
                t_target,
                clamp_w(m_term + var_h_term),
                m_term,
                var_h_term});
            continue;
        }

        while (t + kTiny < t_target) {
            const double dt_limit = config.rel_dt_max * std::max(t, config.time.t_min);
            if (!(dt_limit > 0.0)) {
                throw std::runtime_error("nonpositive dt encountered");
            }
            const double dt = std::min(dt_limit, t_target - t);

            crank_nicolson_step(
                prepared.psi_lower,
                prepared.psi_diag,
                prepared.psi_upper,
                prepared.psi_force,
                dt,
                psi_interior,
                lower_a,
                diag_a,
                upper_a,
                rhs);
            for (double& v : psi_interior) {
                v = clamp_psi(v);
            }
            fill_full_psi_from_interior(psi_interior, psi_full);
            const double m_cur = compute_m(psi_full, prepared.pi, prepared.dx);

            crank_nicolson_step(
                prepared.h_lower,
                prepared.h_diag,
                prepared.h_upper,
                prepared.h_force,
                dt,
                h_interior,
                lower_a,
                diag_a,
                upper_a,
                rhs);
            fill_full_h_from_interior(h_interior, h_full);
            const double var_h_cur = compute_h_moments(h_full, prepared.pi, prepared.dx).variance;

            integral_m += 0.5 * (m_prev + m_cur) * dt;
            t += dt;
            m_prev = m_cur;
            var_h_prev = var_h_cur;
            ++steps;

            m_window.push_back(m_cur);
            var_h_window.push_back(var_h_cur);
            if (static_cast<int>(m_window.size()) > window) {
                m_window.pop_front();
            }
            if (static_cast<int>(var_h_window.size()) > window) {
                var_h_window.pop_front();
            }
            if (static_cast<int>(m_window.size()) == window
                && static_cast<int>(var_h_window.size()) == window
                && t >= min_tail_time) {
                const auto [m_min_it, m_max_it] = std::minmax_element(m_window.begin(), m_window.end());
                const auto [v_min_it, v_max_it] = std::minmax_element(var_h_window.begin(), var_h_window.end());
                if ((*m_max_it - *m_min_it) <= config.steady_tol
                    && (*v_max_it - *v_min_it) <= config.steady_tol) {
                    use_tail = true;
                    t_ss = t;
                    integral_ss = integral_m;
                    m_inf = m_cur;
                    var_h_inf = var_h_cur;
                    break;
                }
            }
        }

        double integral_target = integral_m;
        double var_h_target = var_h_prev;
        if (use_tail && t < t_target) {
            integral_target = integral_ss + m_inf * (t_target - t_ss);
            var_h_target = var_h_inf;
        }
        const double m_term = integral_target / t_target;
        const double var_h_term = var_h_target / (2.0 * t_target);
        result.points.push_back(GridPoint{
            t_target,
            clamp_w(m_term + var_h_term),
            m_term,
            var_h_term});
    }

    if (!use_tail) {
        m_inf = m_prev;
    }
    result.w_infty_est = clamp_w(m_inf);
    result.x_max_used = prepared.x_max;
    result.steps = steps;
    return result;
}

std::vector<SolveResult> solve_many(
    const std::vector<WckParams>& params_list,
    const SolverConfig& config) {
    if (params_list.empty()) {
        return {};
    }

    std::size_t threads = config.thread_count;
    if (threads == 0U) {
        threads = static_cast<std::size_t>(std::thread::hardware_concurrency());
    }
    if (threads == 0U) {
        threads = 1U;
    }

    const std::size_t workers = std::min<std::size_t>(threads, params_list.size());
    std::vector<SolveResult> results(params_list.size());

    parallel_for_index(params_list.size(), workers, [&](std::size_t idx) {
        results[idx] = solve_w_grid(params_list[idx], config);
    });
    return results;
}

}  // namespace wck
