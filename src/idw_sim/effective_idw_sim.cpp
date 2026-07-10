#include "wck/idw_sim/effective_idw_sim.hpp"
#include "wck/common/distributions.hpp"
#include "wck/common/hash.hpp"
#include "wck/common/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace wck {

namespace {

struct RunningStats {
    std::size_t n = 0U;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double x) {
        ++n;
        const double delta = x - mean;
        mean += delta / static_cast<double>(n);
        const double delta2 = x - mean;
        m2 += delta * delta2;
    }

    void merge(const RunningStats& other) {
        if (other.n == 0U) {
            return;
        }
        if (n == 0U) {
            *this = other;
            return;
        }
        const double total = static_cast<double>(n + other.n);
        const double delta = other.mean - mean;
        m2 += other.m2 + delta * delta * static_cast<double>(n) * static_cast<double>(other.n) / total;
        mean += delta * (static_cast<double>(other.n) / total);
        n += other.n;
    }

    double sample_variance() const {
        if (n < 2U) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double v = m2 / static_cast<double>(n - 1U);
        if (v < 0.0 && v > -1e-12) {
            v = 0.0;
        }
        return v;
    }
};

void append_drops(
    int start_level,
    int max_level,
    double tau_shift,
    const std::string& reason,
    std::vector<double>* dropped_horizons,
    std::vector<std::string>* dropped_reasons) {
    for (int level = start_level; level <= max_level; ++level) {
        const double t = std::ldexp(tau_shift, level);
        if (dropped_horizons != nullptr) {
            dropped_horizons->push_back(t);
        }
        if (dropped_reasons != nullptr) {
            dropped_reasons->push_back(reason);
        }
    }
}

}  // namespace

std::vector<EffectiveIdwEstimate> estimate_effective_idw_from_bins(
    std::vector<double> bins,
    const EstimatorConfig& config,
    std::vector<double>* dropped_horizons,
    std::vector<std::string>* dropped_reasons,
    const std::function<void(double)>& progress_callback) {
    if (bins.empty()) {
        throw std::invalid_argument("bins cannot be empty");
    }
    if (!(config.tau_base > 0.0)) {
        throw std::invalid_argument("estimator tau_base must be > 0");
    }
    if (!(config.tau_shift > 0.0)) {
        throw std::invalid_argument("estimator tau_shift must be > 0");
    }
    if (config.max_level < 0) {
        throw std::invalid_argument("estimator max_level must be >= 0");
    }
    if (config.max_level > 30) {
        throw std::invalid_argument("estimator max_level must be <= 30");
    }
    if (config.min_windows_per_t < 2) {
        throw std::invalid_argument("estimator min_windows_per_t must be >= 2");
    }
    if (!(config.ev_hat > 0.0) || !std::isfinite(config.ev_hat)) {
        throw std::invalid_argument("estimator ev_hat must be finite and > 0");
    }

    auto report = [&](double p) {
        if (!progress_callback) {
            return;
        }
        progress_callback(std::clamp(p, 0.0, 1.0));
    };
    report(0.0);

    std::vector<EffectiveIdwEstimate> rows{};
    rows.reserve(static_cast<std::size_t>(config.max_level + 1));

    // bins doubles as the working buffer H^(level): each level's pairwise sums
    // are written into its front half in place, so the estimator allocates
    // nothing beyond the path it was handed.

    for (int level = 0; level <= config.max_level; ++level) {
        const std::int64_t r_bins_i64 = static_cast<std::int64_t>(1) << level;
        if (r_bins_i64 > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("r_bins overflow at requested level");
        }

        RunningStats stats_even{};
        RunningStats stats_odd{};

        if (level == 0) {
            for (double x : bins) {
                stats_even.add(x);
            }
        } else {
            const std::size_t m = bins.size();
            const std::size_t n_even = m / 2U;
            if (n_even == 0U) {
                append_drops(
                    level,
                    config.max_level,
                    config.tau_shift,
                    "horizon_exceeds_sample_bins",
                    dropped_horizons,
                    dropped_reasons);
                report(1.0);
                break;
            }

            // Odd-offset windows must be read before the in-place even
            // reduction below overwrites the front of the buffer.
            if (m >= 3U) {
                const std::size_t n_odd = (m - 1U) / 2U;
                for (std::size_t k = 0U; k < n_odd; ++k) {
                    const double value = bins[2U * k + 1U] + bins[2U * k + 2U];
                    stats_odd.add(value);
                }
            }

            for (std::size_t k = 0U; k < n_even; ++k) {
                const double value = bins[2U * k] + bins[2U * k + 1U];
                stats_even.add(value);
                bins[k] = value;
            }
            bins.resize(n_even);
        }

        RunningStats stats_all = stats_even;
        stats_all.merge(stats_odd);
        const std::size_t n_windows = stats_all.n;

        if (n_windows < static_cast<std::size_t>(config.min_windows_per_t)) {
            append_drops(
                level,
                config.max_level,
                config.tau_shift,
                "insufficient_windows",
                dropped_horizons,
                dropped_reasons);
            report(1.0);
            break;
        }

        const double ey_hat = stats_all.mean;
        const double vary_hat = stats_all.sample_variance();
        if (!(ey_hat > 0.0) || !std::isfinite(ey_hat)) {
            append_drops(
                level,
                config.max_level,
                config.tau_shift,
                "nonpositive_mean_window_work",
                dropped_horizons,
                dropped_reasons);
            report(1.0);
            break;
        }
        if (!std::isfinite(vary_hat)) {
            append_drops(
                level,
                config.max_level,
                config.tau_shift,
                "nonfinite_variance",
                dropped_horizons,
                dropped_reasons);
            report(1.0);
            break;
        }

        EffectiveIdwEstimate row{};
        row.tau = config.tau_base;
        row.tau_shift = config.tau_shift;
        row.tau_shift_index = config.tau_shift_index;
        row.level = level;
        row.t = std::ldexp(config.tau_shift, level);
        row.r_bins = static_cast<int>(r_bins_i64);
        row.n_windows = n_windows;
        row.ey_hat = ey_hat;
        row.vary_hat = vary_hat;
        row.ev_hat = config.ev_hat;
        row.idw_hat = vary_hat / (config.ev_hat * ey_hat);
        rows.push_back(row);

        const double frac = static_cast<double>(level + 1) / static_cast<double>(config.max_level + 1);
        report(frac);
    }

    report(1.0);
    return rows;
}

SimulationRunResult simulate_effective_idw(
    const RunParameters& params,
    const std::filesystem::path* event_trace_path,
    const std::function<void(double)>& progress_callback) {
    if (!(params.lambda > 0.0)) {
        throw std::invalid_argument("lambda must be > 0");
    }
    if (!(params.alpha > 0.0)) {
        throw std::invalid_argument("alpha must be > 0");
    }
    if (params.scaling.k < 1) {
        throw std::invalid_argument("scaling.k must be >= 1");
    }
    if (!(params.scaling.beta_patience > 0.0)) {
        throw std::invalid_argument("scaling.beta_patience must be > 0");
    }

    validate_distribution_spec(params.arrival, "arrival distribution");
    validate_distribution_spec(params.service, "service distribution");
    validate_distribution_spec(params.patience, "patience distribution");

    const DistributionMoments arrival_base_moments = distribution_moments(params.arrival);
    const double arrival_base_rate = 1.0 / arrival_base_moments.mean;
    if (!(arrival_base_rate > 0.0) || !std::isfinite(arrival_base_rate)) {
        throw std::invalid_argument("arrival base rate must be finite and > 0");
    }
    const double arrival_rate_scale = params.lambda / arrival_base_rate;
    const DistributionSpec arrival_runtime = scale_distribution_rates(params.arrival, arrival_rate_scale);
    const DistributionSpec service_runtime = params.service;
    const DistributionSpec patience_runtime = scale_distribution_rates(params.patience, params.alpha);

    const auto start_time = std::chrono::steady_clock::now();

    const std::vector<double> tau_shifts =
        build_tau_shift_grid(params.simulation.tau, params.simulation.n_tau_shifts);

    struct ShiftBins {
        double tau_shift = 0.0;
        std::size_t n_bins = 0U;
        double observed_time = 0.0;
        std::vector<double> bins{};
    };

    std::vector<ShiftBins> shifts{};
    shifts.reserve(tau_shifts.size());
    for (double tau_shift : tau_shifts) {
        const double raw_n_bins = std::floor(params.simulation.sample_time / tau_shift);
        if (!(raw_n_bins >= 2.0)) {
            throw std::invalid_argument("sample_time / tau_shift must be >= 2 for every generated shift");
        }
        if (raw_n_bins > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
            throw std::invalid_argument("sample_time / tau_shift exceeds SIZE_MAX");
        }

        ShiftBins s{};
        s.tau_shift = tau_shift;
        s.n_bins = static_cast<std::size_t>(raw_n_bins);
        s.observed_time = static_cast<double>(s.n_bins) * tau_shift;
        s.bins.assign(s.n_bins, 0.0);
        shifts.push_back(std::move(s));
    }

    const double warmup_end = params.simulation.warmup_time;
    const double sample_end = warmup_end + params.simulation.sample_time;

    std::ofstream trace;
    if (event_trace_path != nullptr) {
        trace.open(*event_trace_path);
        if (!trace.is_open()) {
            throw std::runtime_error("failed to open event trace file: " + event_trace_path->string());
        }
        trace << std::setprecision(17) << std::scientific;
        trace << "arrival_index,time,interarrival,workload_before,service,patience,accepted,workload_after,sample_bin_tau0\n";
    }

    std::mt19937_64 rng(params.seed);
    DistributionSampler arrival_sampler(arrival_runtime, &rng);
    DistributionSampler service_sampler(service_runtime, &rng);
    DistributionSampler patience_sampler(patience_runtime, &rng);

    double t = 0.0;
    double workload = 0.0;
    std::uint64_t arrival_idx = 0U;

    SimulationStats stats{};
    stats.sample_observed_time = params.simulation.sample_time;

    double last_reported_progress = -1.0;
    auto last_report_time = std::chrono::steady_clock::now();
    auto report_progress = [&](double progress, bool force) {
        if (!progress_callback) {
            return;
        }
        progress = std::clamp(progress, 0.0, 1.0);
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report_time).count();
        if (!force) {
            if (progress - last_reported_progress < 0.0008 && elapsed_ms < 120) {
                return;
            }
        }
        progress_callback(progress);
        last_reported_progress = progress;
        last_report_time = now;
    };
    report_progress(0.0, true);

    while (true) {
        const double interarrival = arrival_sampler.sample();

        t += interarrival;
        if (t > sample_end) {
            break;
        }

        const double workload_before = std::max(workload - interarrival, 0.0);
        workload = workload_before;

        const double service = service_sampler.sample();
        const double patience = patience_sampler.sample();
        // FCFS+abandonment recursion: accepted iff D_k > W_k at arrival.
        const bool accepted = (patience > workload_before);
        if (accepted) {
            workload += service;
        }

        ++arrival_idx;
        ++stats.arrivals_total;
        if (accepted) {
            ++stats.accepted_total;
        } else {
            ++stats.abandoned_total;
        }

        std::int64_t sample_bin_tau0 = -1;
        const bool in_sample = (t >= warmup_end) && (t < sample_end);
        if (in_sample) {
            ++stats.arrivals_sample;
            stats.sample_service_sum += service;
            ++stats.sample_service_count;

            const double rel_time = t - warmup_end;
            std::size_t sample_bin_tau0_idx = static_cast<std::size_t>(
                std::floor(rel_time / shifts[0].tau_shift));
            if (sample_bin_tau0_idx >= shifts[0].n_bins) {
                sample_bin_tau0_idx = shifts[0].n_bins - 1U;
            }
            if (sample_bin_tau0_idx > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
                sample_bin_tau0 = std::numeric_limits<std::int64_t>::max();
            } else {
                sample_bin_tau0 = static_cast<std::int64_t>(sample_bin_tau0_idx);
            }

            if (accepted) {
                ++stats.accepted_sample;
                stats.sample_effective_work += service;
                for (auto& shift : shifts) {
                    if (rel_time >= shift.observed_time) {
                        continue;
                    }
                    std::size_t idx = static_cast<std::size_t>(std::floor(rel_time / shift.tau_shift));
                    if (idx >= shift.n_bins) {
                        idx = shift.n_bins - 1U;
                    }
                    shift.bins[idx] += service;
                }
            } else {
                ++stats.abandoned_sample;
            }
        }

        if (trace.is_open()) {
            trace << arrival_idx << ',' << t << ',' << interarrival << ',' << workload_before
                  << ',' << service << ',' << patience << ',' << (accepted ? 1 : 0)
                  << ',' << workload << ',' << sample_bin_tau0 << '\n';
        }

        report_progress(0.75 * (t / sample_end), false);
    }

    if (stats.sample_service_count == 0U) {
        throw std::runtime_error("no arrivals in sample window; cannot estimate E[V]");
    }
    const double ev_hat = stats.sample_service_sum / static_cast<double>(stats.sample_service_count);

    SimulationRunResult result{};
    result.params = params;
    result.stats = stats;

    const auto estimation_start = std::chrono::steady_clock::now();

    const std::size_t n_shifts = shifts.size();
    std::vector<std::vector<EffectiveIdwEstimate>> rows_by_shift(n_shifts);
    std::vector<std::vector<double>> dropped_h_by_shift(n_shifts);
    std::vector<std::vector<std::string>> dropped_r_by_shift(n_shifts);

    std::size_t resolved_threads = 1U;
    if (params.simulation.threads == 0) {
        resolved_threads = static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (resolved_threads == 0U) {
            resolved_threads = 1U;
        }
    } else {
        resolved_threads = static_cast<std::size_t>(params.simulation.threads);
        if (resolved_threads == 0U) {
            resolved_threads = 1U;
        }
    }

    const std::size_t workers = std::min(resolved_threads, n_shifts);
    result.estimator_threads_used = static_cast<int>(workers);

    auto estimate_shift = [&](std::size_t idx) {
        EstimatorConfig cfg{};
        cfg.tau_base = params.simulation.tau;
        cfg.tau_shift = shifts[idx].tau_shift;
        cfg.tau_shift_index = static_cast<int>(idx);
        cfg.max_level = params.simulation.max_level;
        cfg.min_windows_per_t = params.simulation.min_windows_per_t;
        cfg.ev_hat = ev_hat;

        rows_by_shift[idx] = estimate_effective_idw_from_bins(
            std::move(shifts[idx].bins),
            cfg,
            &dropped_h_by_shift[idx],
            &dropped_r_by_shift[idx]);
    };

    if (workers == 1U) {
        for (std::size_t idx = 0U; idx < n_shifts; ++idx) {
            estimate_shift(idx);

            const double frac = static_cast<double>(idx + 1U) / static_cast<double>(n_shifts);
            report_progress(0.75 + 0.25 * frac, false);
        }
    } else {
        std::atomic<std::size_t> completed{0U};
        std::mutex progress_mutex{};
        double last_estimation_progress = 0.75;
        auto report_estimation_progress = [&](std::size_t done) {
            const double frac = static_cast<double>(done) / static_cast<double>(n_shifts);
            const double p = 0.75 + 0.25 * frac;
            std::lock_guard<std::mutex> lock(progress_mutex);
            if (p > last_estimation_progress) {
                report_progress(p, false);
                last_estimation_progress = p;
            }
        };

        parallel_for_index(n_shifts, workers, [&](std::size_t idx) {
            estimate_shift(idx);

            const std::size_t done = completed.fetch_add(1U, std::memory_order_relaxed) + 1U;
            report_estimation_progress(done);
        });
    }

    result.estimates.clear();
    for (std::size_t idx = 0U; idx < n_shifts; ++idx) {
        for (const auto& row : rows_by_shift[idx]) {
            result.estimates.push_back(row);
        }

        for (std::size_t j = 0U; j < dropped_h_by_shift[idx].size(); ++j) {
            result.dropped_horizons.push_back(dropped_h_by_shift[idx][j]);
            std::string reason = "tau_shift_index=" + std::to_string(idx)
                + ",tau_shift=" + std::to_string(shifts[idx].tau_shift)
                + "," + dropped_r_by_shift[idx][j];
            result.dropped_reasons.push_back(std::move(reason));
        }
    }

    const auto estimation_end = std::chrono::steady_clock::now();
    result.estimation_wall_seconds =
        std::chrono::duration<double>(estimation_end - estimation_start).count();

    report_progress(1.0, true);

    const auto end_time = std::chrono::steady_clock::now();
    result.runtime_seconds = std::chrono::duration<double>(end_time - start_time).count();
    return result;
}

std::uint64_t derive_run_seed(
    std::uint64_t base_seed,
    std::size_t model_index,
    int alpha_index,
    const std::string& model_name) {
    std::uint64_t x = splitmix64(base_seed + 0x9E3779B97F4A7C15ULL);
    x ^= splitmix64(static_cast<std::uint64_t>(model_index));
    x ^= splitmix64(static_cast<std::uint64_t>(alpha_index < 0 ? 0 : alpha_index));
    x ^= splitmix64(fnv1a_64(model_name));
    return splitmix64(x);
}

std::string sanitize_name_for_file(const std::string& name) {
    std::string out;
    out.reserve(name.size());

    bool previous_underscore = false;
    for (char ch : name) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch) != 0) {
            out.push_back(static_cast<char>(std::tolower(uch)));
            previous_underscore = false;
            continue;
        }

        if (!previous_underscore) {
            out.push_back('_');
            previous_underscore = true;
        }
    }

    while (!out.empty() && out.front() == '_') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    if (out.empty()) {
        out = "model";
    }
    return out;
}

}  // namespace wck
