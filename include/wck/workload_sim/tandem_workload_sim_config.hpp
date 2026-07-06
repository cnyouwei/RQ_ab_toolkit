#pragma once

#include "wck/common/distributions.hpp"
#include "wck/workload_sim/workload_sim_config.hpp"

#include <filesystem>
#include <string>

namespace wck {

// The tandem config shares the exact same "simulation" block as the
// single-station config.
using TandemWorkloadSimulationConfig = WorkloadSimulationConfig;

struct TandemQueue1Config {
    double traffic_intensity = 0.9;
    DistributionSpec arrival{};
    DistributionSpec service{};
};

struct TandemQueue2Config {
    DistributionSpec service{};
    DistributionSpec patience{};
};

struct TandemWorkloadModelConfig {
    std::string name{};
    std::string alias{};
    TandemQueue1Config queue1{};
    TandemQueue2Config queue2{};
};

struct TandemWorkloadConfig {
    TandemWorkloadSimulationConfig simulation{};
    TandemWorkloadModelConfig model{};
};

TandemWorkloadConfig load_tandem_workload_config(const std::filesystem::path& path);

}  // namespace wck
