#pragma once

#include "wck/common/distributions.hpp"
#include "wck/workload_sim/workload_sim.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace wck {

struct TandemWorkloadRunParams {
    std::string model_name{};
    double lambda = 1.0;
    double alpha = 1.0;

    double queue1_traffic_intensity = 0.9;
    DistributionSpec queue1_arrival{};
    DistributionSpec queue1_service{};
    DistributionSpec queue2_service{};
    DistributionSpec queue2_patience{};

    double warmup_time = 1e4;
    double sample_time = 2e5;
    int replications = 128;
    int threads = 0;  // 0 => auto detect
    std::uint64_t seed = 123456789ULL;
    bool normalize_service_mean_to_one = true;
};

// The tandem driver reports the exact same summary fields as the
// single-station driver.
using TandemWorkloadSummary = WorkloadSummary;

TandemWorkloadSummary simulate_tandem_workload_mc(
    const TandemWorkloadRunParams& params,
    std::vector<double>* per_rep_estimates = nullptr);

}  // namespace wck
