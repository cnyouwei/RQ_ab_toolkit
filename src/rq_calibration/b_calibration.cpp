#include "wck/rq_calibration/b_calibration.hpp"

#include "wck/common/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace wck {

namespace {

constexpr double kSqrt2 = 1.4142135623730950488;
constexpr double kFeasibilityThreshold = -1e-12;
constexpr double kCapTolerance = 1e-12;
constexpr int kCoarseGridPoints = 4001;
constexpr double kLogUMin = -16.0;
constexpr double kLogUMax = 18.0;
constexpr double kPsiLogTailCutoff = 80.0;
constexpr double kPsiSimpsonEps = 1e-14;
constexpr int kPsiSimpsonMaxDepth = 40;

long double inverse_factorial(int n) {
    if (n < 0) {
        throw std::invalid_argument("inverse_factorial expects n >= 0");
    }
    long double inv = 1.0L;
    for (int i = 2; i <= n; ++i) {
        inv /= static_cast<long double>(i);
    }
    return inv;
}

long double pow_integer(long double base, int exponent) {
    if (exponent < 0) {
        throw std::invalid_argument("pow_integer expects exponent >= 0");
    }
    long double out = 1.0L;
    for (int i = 0; i < exponent; ++i) {
        out *= base;
    }
    return out;
}

long double canonical_fk0(int k) {
    if (k < 1) {
        throw std::invalid_argument("k must be >= 1");
    }
    if (k == 1) {
        return 1.0L;
    }
    return pow_integer(static_cast<long double>(k), k);
}

double beta_from_k(int k) {
    const long double fk0 = canonical_fk0(k);
    const long double inv_k_fact = inverse_factorial(k);
    const long double beta = fk0 * inv_k_fact;
    const double beta_d = static_cast<double>(beta);
    if (!(beta_d > 0.0) || !std::isfinite(beta_d)) {
        throw std::runtime_error("computed beta is invalid");
    }
    return beta_d;
}

template <typename Fn>
double adaptive_simpson(Fn&& f, double a, double b, double eps, int max_depth) {
    const double fa = f(a);
    const double fb = f(b);
    const double c = 0.5 * (a + b);
    const double fc = f(c);
    const double s = (b - a) * (fa + 4.0 * fc + fb) / 6.0;

    auto recurse = [&](auto&& self, double l, double r, double f_l, double f_r, double f_m, double s_lr, int depth)
        -> double {
        const double m = 0.5 * (l + r);
        const double lm = 0.5 * (l + m);
        const double mr = 0.5 * (m + r);

        const double f_lm = f(lm);
        const double f_mr = f(mr);

        const double s_left = (m - l) * (f_l + 4.0 * f_lm + f_m) / 6.0;
        const double s_right = (r - m) * (f_m + 4.0 * f_mr + f_r) / 6.0;
        const double s2 = s_left + s_right;

        if (depth <= 0 || std::abs(s2 - s_lr) <= 15.0 * eps) {
            return s2 + (s2 - s_lr) / 15.0;
        }

        return self(self, l, m, f_l, f_m, f_lm, s_left, depth - 1)
            + self(self, m, r, f_m, f_r, f_mr, s_right, depth - 1);
    };

    return recurse(recurse, a, b, fa, fb, fc, s, max_depth);
}

double exact_psi_target_impl(int k, double c) {
    if (k < 1) {
        throw std::invalid_argument("k must be >= 1");
    }
    if (!std::isfinite(c)) {
        throw std::invalid_argument("c must be finite");
    }

    const long double fk0_ld = canonical_fk0(k);
    const long double inv_kp1_fact = inverse_factorial(k + 1);
    const long double a_ld = fk0_ld * inv_kp1_fact;
    const double a = static_cast<double>(a_ld);
    if (!(a > 0.0) || !std::isfinite(a)) {
        throw std::runtime_error("invalid a coefficient in psi integral");
    }

    const double kp1 = static_cast<double>(k + 1);
    const double beta = a * kp1;

    auto g = [&](double x) -> double {
        return c * x - a * std::pow(x, kp1);
    };

    // g'(x) = c - beta*x^k: interior mode for c > 0, boundary mode at 0 otherwise.
    const double x_mode = (c > 0.0) ? std::pow(c / beta, 1.0 / static_cast<double>(k)) : 0.0;
    const double g_max = g(x_mode);

    auto g_shift = [&](double x) -> double {
        return g(x) - g_max;
    };

    // Localize the integration window: for large |c| the integrand is a
    // narrow spike at x_mode that global quadrature over [0,inf) misses.
    // Right endpoint: expand past the mode, then bisect to where the
    // log-integrand drops kPsiLogTailCutoff below its maximum.
    const double step = std::max({x_mode, 1.0, std::pow(beta, -1.0 / kp1)});
    double hi = x_mode + step;
    bool right_localized = false;
    for (int i = 0; i < 200; ++i) {
        if (g_shift(hi) < -kPsiLogTailCutoff) {
            right_localized = true;
            break;
        }
        hi = x_mode + (hi - x_mode) * 2.0;
    }
    if (!right_localized) {
        throw std::runtime_error("failed to localize right tail of psi integrand");
    }
    double lo_r = x_mode;
    double hi_r = hi;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo_r + hi_r);
        if (g_shift(mid) < -kPsiLogTailCutoff) {
            hi_r = mid;
        } else {
            lo_r = mid;
        }
        if (hi_r - lo_r <= 1e-12 * std::max(1.0, hi_r)) {
            break;
        }
    }
    const double x_hi = hi_r;

    // Left endpoint: 0 unless the mode is interior and the integrand at 0 is
    // already negligible.
    double x_lo = 0.0;
    if (x_mode > 0.0 && g_shift(0.0) < -kPsiLogTailCutoff) {
        double lo_l = 0.0;
        double hi_l = x_mode;
        for (int i = 0; i < 200; ++i) {
            const double mid = 0.5 * (lo_l + hi_l);
            if (g_shift(mid) < -kPsiLogTailCutoff) {
                lo_l = mid;
            } else {
                hi_l = mid;
            }
            if (hi_l - lo_l <= 1e-12 * std::max(1.0, hi_l)) {
                break;
            }
        }
        x_lo = lo_l;
    }

    if (!(x_hi > x_lo)) {
        throw std::runtime_error("degenerate psi integration window");
    }

    auto exp_term = [&](double x) -> double {
        const double exponent = g_shift(x);
        if (exponent < -745.0) {
            return 0.0;
        }
        return std::exp(exponent);
    };

    const double width = x_hi - x_lo;
    const double eps0 = kPsiSimpsonEps * width;
    const double i0 = adaptive_simpson(exp_term, x_lo, x_hi, eps0, kPsiSimpsonMaxDepth);
    // First moment centered at x_mode, so psi = x_mode + m1/i0. Centering
    // keeps the quadrature error of psi independent of the magnitude of
    // x_mode; calibrate_b_table compares a_psi = c - beta*psi^k against a
    // 1e-12-scale threshold, which an uncentered first moment cannot resolve
    // for large c.
    const double m1 = adaptive_simpson(
        [&](double x) { return (x - x_mode) * exp_term(x); }, x_lo, x_hi, eps0, kPsiSimpsonMaxDepth);
    if (!(i0 > 0.0) || !std::isfinite(i0) || !std::isfinite(m1)) {
        throw std::runtime_error("failed to compute finite psi integrals");
    }
    const double psi = x_mode + m1 / i0;
    if (!std::isfinite(psi) || !(psi >= 0.0)) {
        throw std::runtime_error("computed psi is invalid");
    }
    return psi;
}

double objective_for_exact_calibration(
    double log_u,
    double psi,
    double a_psi,
    double tilde_c,
    double tau,
    const WTableInterpolator& w_table) {
    const double u = std::exp(log_u);
    if (!(u > 0.0) || !std::isfinite(u)) {
        return std::numeric_limits<double>::infinity();
    }

    const double w_val = w_table.w(tilde_c, tau * u);
    if (!(w_val > 0.0) || !std::isfinite(w_val)) {
        return std::numeric_limits<double>::infinity();
    }

    const double numerator = psi - a_psi * u;
    const double denominator = 2.0 * w_val * u;
    if (!(denominator > 0.0) || !std::isfinite(denominator) || !std::isfinite(numerator)) {
        return std::numeric_limits<double>::infinity();
    }
    return numerator / std::sqrt(denominator);
}

std::pair<double, double> compute_exact_b_and_u_star(
    double psi,
    double a_psi,
    double tilde_c,
    double tau,
    const WTableInterpolator& w_table) {
    auto objective = [&](double log_u) {
        return objective_for_exact_calibration(log_u, psi, a_psi, tilde_c, tau, w_table);
    };

    double best_log_u = kLogUMin;
    double best_val = std::numeric_limits<double>::infinity();

    for (int i = 0; i < kCoarseGridPoints; ++i) {
        const double ratio = static_cast<double>(i) / static_cast<double>(kCoarseGridPoints - 1);
        const double log_u = kLogUMin + (kLogUMax - kLogUMin) * ratio;
        const double val = objective(log_u);
        if (val < best_val) {
            best_val = val;
            best_log_u = log_u;
        }
    }

    if (!std::isfinite(best_val)) {
        throw std::runtime_error("failed to find finite coarse objective value for exact b calibration");
    }

    double left = std::max(kLogUMin, best_log_u - 1.0);
    double right = std::min(kLogUMax, best_log_u + 1.0);
    if (!(right > left)) {
        right = std::min(kLogUMax, left + 1.0);
    }

    const double inv_phi = (std::sqrt(5.0) - 1.0) / 2.0;
    double c1 = right - inv_phi * (right - left);
    double c2 = left + inv_phi * (right - left);
    double f1 = objective(c1);
    double f2 = objective(c2);

    for (int iter = 0; iter < 120; ++iter) {
        if (f1 < f2) {
            right = c2;
            c2 = c1;
            f2 = f1;
            c1 = right - inv_phi * (right - left);
            f1 = objective(c1);
        } else {
            left = c1;
            c1 = c2;
            f1 = f2;
            c2 = left + inv_phi * (right - left);
            f2 = objective(c2);
        }
    }

    const double log_u_star = 0.5 * (left + right);
    const double u_star = std::exp(log_u_star);
    const double b_value = objective(log_u_star);
    if (!(b_value >= 0.0) || !std::isfinite(b_value)) {
        throw std::runtime_error("calibrated b is invalid");
    }
    return {b_value, u_star};
}

}  // namespace

std::vector<double> build_c_grid(double c_min, double c_max, double dc) {
    if (!(dc > 0.0)) {
        throw std::invalid_argument("dc must be > 0");
    }
    if (!(c_max >= c_min)) {
        throw std::invalid_argument("c_max must be >= c_min");
    }

    const double n_float = (c_max - c_min) / dc;
    const long long n = std::llround(n_float);
    if (std::abs(n_float - static_cast<double>(n)) > 1e-9) {
        throw std::invalid_argument("(c_max - c_min) must be an integer multiple of dc");
    }

    std::vector<double> grid{};
    grid.reserve(static_cast<std::size_t>(n + 1LL));
    for (long long i = 0LL; i <= n; ++i) {
        double c = c_min + static_cast<double>(i) * dc;
        c = std::round(c * 1e12) / 1e12;
        if (std::abs(c) < 5e-13) {
            c = 0.0;
        }
        grid.push_back(c);
    }
    return grid;
}

double exact_psi_target(int k, double c) {
    return exact_psi_target_impl(k, c);
}

BCalibrationResult calibrate_b_table(
    int k,
    const std::vector<double>& c_grid,
    const WTableInterpolator& w_table,
    std::size_t thread_count) {
    if (k < 1) {
        throw std::invalid_argument("k must be >= 1");
    }
    if (c_grid.empty()) {
        throw std::invalid_argument("c-grid is empty");
    }
    for (std::size_t i = 0U; i + 1U < c_grid.size(); ++i) {
        if (!(c_grid[i + 1U] > c_grid[i])) {
            throw std::invalid_argument("c-grid must be strictly increasing");
        }
    }

    const double beta = beta_from_k(k);
    const double tau = std::pow(beta, 2.0 / static_cast<double>(k + 1));
    const double c_scale = std::pow(beta, -1.0 / static_cast<double>(k + 1));

    BCalibrationResult out{};
    out.k = k;
    out.beta = beta;
    out.tau = tau;
    out.rows.resize(c_grid.size());

    auto compute_one = [&](std::size_t idx) {
        const double c = c_grid[idx];
        const double psi = exact_psi_target(k, c);
        const double a_psi = c - beta * std::pow(psi, static_cast<double>(k));

        BCalibrationRow row{};
        row.c = c;
        row.psi = psi;
        row.a_psi = a_psi;

        if (a_psi < kFeasibilityThreshold) {
            const double tilde_c = c * c_scale;
            const auto [b_value, u_star] = compute_exact_b_and_u_star(psi, a_psi, tilde_c, tau, w_table);
            if (b_value <= kSqrt2 + kCapTolerance) {
                row.b = b_value;
                row.u_star = u_star;
            } else {
                // Explicit cap policy: stored calibration values are capped at sqrt(2).
                row.b = kSqrt2;
                row.u_star = std::numeric_limits<double>::quiet_NaN();
            }
            row.z_model = std::numeric_limits<double>::quiet_NaN();
            row.abs_error = 0.0;
        } else {
            // If exact matching is infeasible, use k-dependent fallback:
            // k>1 -> b=0, k=1 -> b=sqrt(2).
            row.b = (k > 1) ? 0.0 : kSqrt2;
            if (c > 0.0) {
                row.z_model = std::pow(c / beta, 1.0 / static_cast<double>(k));
                row.abs_error = std::abs(row.z_model - psi);
            } else {
                row.z_model = std::numeric_limits<double>::quiet_NaN();
                row.abs_error = std::numeric_limits<double>::quiet_NaN();
            }
            row.u_star = std::numeric_limits<double>::quiet_NaN();
        }

        out.rows[idx] = row;
    };

    std::size_t workers = thread_count;
    if (workers == 0U) {
        workers = static_cast<std::size_t>(std::thread::hardware_concurrency());
    }
    if (workers == 0U) {
        workers = 1U;
    }
    workers = std::min(workers, c_grid.size());

    parallel_for_index(c_grid.size(), workers, compute_one);
    return out;
}

void write_b_calibration_table_csv(const std::filesystem::path& path, const BCalibrationResult& result) {
    if (result.rows.empty()) {
        throw std::invalid_argument("calibration result has no rows");
    }

    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }

    out << std::setprecision(17) << std::scientific;
    out << "c,b,psi,z_model,abs_error,a_psi,u_star\n";
    for (const auto& row : result.rows) {
        out << row.c << ','
            << row.b << ','
            << row.psi << ','
            << row.z_model << ','
            << row.abs_error << ','
            << row.a_psi << ','
            << row.u_star << '\n';
    }
    out.flush();
    if (!out.good()) {
        throw std::runtime_error("failed while writing output file: " + path.string());
    }
}

}  // namespace wck
