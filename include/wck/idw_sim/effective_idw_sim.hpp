#pragma once

#include "wck/idw_sim/effective_idw_sim_config.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace wck {

struct EffectiveIdwEstimate {
    double tau = 0.0;
    double tau_shift = 0.0;
    int tau_shift_index = 0;
    int level = 0;
    double t = 0.0;
    int r_bins = 0;
    std::size_t n_windows = 0U;
    double ey_hat = std::numeric_limits<double>::quiet_NaN();
    double vary_hat = std::numeric_limits<double>::quiet_NaN();
    double ev_hat = std::numeric_limits<double>::quiet_NaN();
    double idw_hat = std::numeric_limits<double>::quiet_NaN();
};

struct EstimatorConfig {
    double tau_base = 1e-2;
    double tau_shift = 1e-2;
    int tau_shift_index = 0;
    int max_level = 12;
    int min_windows_per_t = 50;
    double ev_hat = std::numeric_limits<double>::quiet_NaN();
};

// Consumes the bins as its working buffer: pass with std::move to avoid
// copying what may be a tens-of-GB path.
std::vector<EffectiveIdwEstimate> estimate_effective_idw_from_bins(
    std::vector<double> bins,
    const EstimatorConfig& config,
    std::vector<double>* dropped_horizons = nullptr,
    std::vector<std::string>* dropped_reasons = nullptr,
    const std::function<void(double)>& progress_callback = {});

struct RunParameters {
    std::string model_name{};
    std::size_t model_index = 0U;
    int alpha_index = 0;
    double alpha = 1.0;
    double h = 0.5;
    double c = 0.0;
    double rho_exponent = 0.5;
    double rho = 1.0;
    double lambda = 1.0;

    DistributionSpec arrival{};
    DistributionSpec service{};
    DistributionSpec patience{};
    ScalingConfig scaling{};
    SimulationConfig simulation{};

    std::uint64_t seed = 0U;
};

struct SimulationStats {
    std::uint64_t arrivals_total = 0U;
    std::uint64_t arrivals_sample = 0U;
    std::uint64_t accepted_total = 0U;
    std::uint64_t accepted_sample = 0U;
    std::uint64_t abandoned_total = 0U;
    std::uint64_t abandoned_sample = 0U;

    double sample_effective_work = 0.0;
    double sample_service_sum = 0.0;
    std::uint64_t sample_service_count = 0U;
    double sample_observed_time = 0.0;
};

struct SimulationRunResult {
    RunParameters params{};
    std::vector<EffectiveIdwEstimate> estimates{};
    SimulationStats stats{};
    std::vector<double> dropped_horizons{};
    std::vector<std::string> dropped_reasons{};
    int estimator_threads_used = 1;
    double estimation_wall_seconds = 0.0;
    double runtime_seconds = 0.0;
};

SimulationRunResult simulate_effective_idw(
    const RunParameters& params,
    const std::filesystem::path* event_trace_path = nullptr,
    const std::function<void(double)>& progress_callback = {});

std::uint64_t derive_run_seed(
    std::uint64_t base_seed,
    std::size_t model_index,
    int alpha_index,
    const std::string& model_name);

std::string sanitize_name_for_file(const std::string& name);

}  // namespace wck
