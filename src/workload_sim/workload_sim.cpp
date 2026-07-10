#include "wck/workload_sim/workload_sim.hpp"

#include "wck/common/distributions.hpp"
#include "wck/common/hash.hpp"

#include "mc_common.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace wck {

namespace {

// Frozen: the (model_name, lambda, alpha) tuple hash feeds the seed
// derivation; do not modify.
std::uint64_t tuple_hash(const WorkloadRunParams& params) {
    std::uint64_t x = splitmix64(fnv1a_64(params.model_name));
    x ^= splitmix64(std::bit_cast<std::uint64_t>(params.lambda));
    x ^= splitmix64(std::bit_cast<std::uint64_t>(params.alpha));
    return splitmix64(x);
}

// Frozen RNG draw order: interarrival -> service -> patience per arrival,
// with persistent distribution objects for the whole replication.
double simulate_one_replication(
    std::uint64_t seed,
    const DistributionSpec& arrival_runtime,
    const DistributionSpec& service_runtime,
    const DistributionSpec& patience_runtime,
    double warmup_time,
    double sample_time) {
    std::mt19937_64 rng(seed);
    DistributionSampler arrival_sampler(arrival_runtime, &rng);
    DistributionSampler service_sampler(service_runtime, &rng);
    DistributionSampler patience_sampler(patience_runtime, &rng);

    const double sample_start = warmup_time;
    const double sample_end = warmup_time + sample_time;

    double t = 0.0;
    double workload = 0.0;
    double area = 0.0;

    while (t < sample_end) {
        const double interarrival = arrival_sampler.sample();
        const double t_next = t + interarrival;
        area += mc::integrate_segment(t, t_next, workload, sample_start, sample_end);
        if (t_next >= sample_end) {
            break;
        }

        const double workload_before = std::max(workload - interarrival, 0.0);
        const double service = service_sampler.sample();
        const double patience = patience_sampler.sample();
        const bool accepted = (patience > workload_before);
        workload = accepted ? (workload_before + service) : workload_before;
        t = t_next;
    }

    return area / sample_time;
}

}  // namespace

WorkloadSummary simulate_workload_mc(
    const WorkloadRunParams& params,
    std::vector<double>* per_rep_estimates) {
    mc::validate_mc_run_params(
        params.lambda,
        params.alpha,
        params.warmup_time,
        params.sample_time,
        params.replications,
        params.threads);

    validate_distribution_spec(params.arrival, "arrival distribution");
    validate_distribution_spec(params.service, "service distribution");
    validate_distribution_spec(params.patience, "patience distribution");

    const DistributionSpec arrival_runtime =
        mc::rescale_arrival_to_rate(params.arrival, params.lambda, "arrival distribution");
    const DistributionSpec service_runtime = mc::rescale_service_normalized(
        params.service, params.normalize_service_mean_to_one, "service distribution");
    const DistributionSpec patience_runtime = scale_distribution_rates(params.patience, params.alpha);

    return mc::run_replications<WorkloadSummary>(
        params.model_name,
        params.lambda,
        params.alpha,
        params.replications,
        params.threads,
        params.seed,
        tuple_hash(params),
        per_rep_estimates,
        [&](std::uint64_t rep_seed) {
            return simulate_one_replication(
                rep_seed,
                arrival_runtime,
                service_runtime,
                patience_runtime,
                params.warmup_time,
                params.sample_time);
        });
}

}  // namespace wck
