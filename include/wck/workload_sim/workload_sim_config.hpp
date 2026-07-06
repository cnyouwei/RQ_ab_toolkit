#pragma once

#include "wck/common/distributions.hpp"
#include "wck/common/mini_json.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace wck {

struct WorkloadSimulationConfig {
    double warmup_time = 1e4;
    double sample_time = 2e5;
    int replications = 128;
    int threads = 0;  // 0 => auto detect
    std::uint64_t seed = 123456789ULL;
    bool normalize_service_mean_to_one = true;
};

struct WorkloadModelConfig {
    std::string name{};
    std::string alias{};
    DistributionSpec arrival{};
    DistributionSpec service{};
    DistributionSpec patience{};
};

struct WorkloadConfig {
    WorkloadSimulationConfig simulation{};
    WorkloadModelConfig model{};
};

// Parses the "simulation" block shared by the single-station and tandem
// workload MC configs.
WorkloadSimulationConfig parse_workload_simulation_config(const json::Value::Object& root);

WorkloadConfig load_workload_config(const std::filesystem::path& path);

}  // namespace wck
