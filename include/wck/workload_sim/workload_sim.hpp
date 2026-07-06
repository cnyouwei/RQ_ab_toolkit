#pragma once

#include "wck/common/distributions.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace wck {

struct WorkloadRunParams {
    std::string model_name{};
    double lambda = 1.0;
    double alpha = 1.0;

    DistributionSpec arrival{};
    DistributionSpec service{};
    DistributionSpec patience{};

    double warmup_time = 1e4;
    double sample_time = 2e5;
    int replications = 128;
    int threads = 0;  // 0 => auto detect
    std::uint64_t seed = 123456789ULL;
    bool normalize_service_mean_to_one = true;
};

struct WorkloadSummary {
    std::string model_name{};
    double lambda = 0.0;
    double alpha = 0.0;
    int n_reps = 0;
    int threads_used = 1;
    std::uint64_t seed = 0U;

    double mean_workload = std::numeric_limits<double>::quiet_NaN();
    double std_workload = std::numeric_limits<double>::quiet_NaN();
    double runtime_seconds = 0.0;
};

WorkloadSummary simulate_workload_mc(
    const WorkloadRunParams& params,
    std::vector<double>* per_rep_estimates = nullptr);

}  // namespace wck
