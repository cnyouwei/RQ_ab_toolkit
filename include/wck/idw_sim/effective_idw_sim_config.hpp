#pragma once

#include "wck/common/distributions.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace wck {

struct SimulationConfig {
    double warmup_time = 1e4;
    double sample_time = 2e5;
    double tau = 1e-2;
    int max_level = 12;
    int min_windows_per_t = 50;
    int n_tau_shifts = 1;
    int threads = 1;  // default 1; 0 means auto-detect from hardware_concurrency
    std::uint64_t seed = 123456789ULL;
    bool save_event_trace = false;
};

struct AlphaConfig {
    std::vector<int> indices{};
    double base = 2.0;
};

struct SystemConfig {
    double c = 2.0;
};

struct ScalingConfig {
    int k = 1;
    double beta_patience = 1.0;
    bool has_rho_exponent = false;
    double rho_exponent = 0.5;
};

struct SimulationModelConfig {
    std::string name{};
    DistributionSpec arrival{};
    DistributionSpec service{};
    DistributionSpec patience{};
    SystemConfig system{};
    ScalingConfig scaling{};
};

struct EffectiveIdwSimConfig {
    AlphaConfig alpha{};
    SimulationConfig simulation{};
    std::vector<SimulationModelConfig> models{};
};

EffectiveIdwSimConfig load_effective_idw_sim_config(const std::filesystem::path& path);
std::vector<double> build_tau_shift_grid(double tau, int n_tau_shifts);
double alpha_from_index(int index, double base);

}  // namespace wck
