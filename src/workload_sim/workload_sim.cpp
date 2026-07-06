#include "wck/workload_sim/workload_sim.hpp"

#include "wck/common/distributions.hpp"
#include "wck/common/hash.hpp"
#include "wck/common/parallel.hpp"

#include "mc_common.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
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

    const std::size_t n_reps = static_cast<std::size_t>(params.replications);
    const std::size_t workers = mc::resolve_worker_count(params.threads, n_reps);

    const auto start = std::chrono::steady_clock::now();

    std::vector<double> rep_estimates(n_reps, std::numeric_limits<double>::quiet_NaN());
    const std::uint64_t thash = tuple_hash(params);

    parallel_for_index(n_reps, workers, [&](std::size_t rep) {
        const std::uint64_t rep_seed = mc::derive_rep_seed(params.seed, thash, rep);
        rep_estimates[rep] = simulate_one_replication(
            rep_seed,
            arrival_runtime,
            service_runtime,
            patience_runtime,
            params.warmup_time,
            params.sample_time);
    });

    // Sequential ascending-index reductions: part of the bit-reproducibility
    // contract for any thread count.
    const double mean = mc::running_mean(rep_estimates);
    const double std = mc::sample_std(rep_estimates, mean);

    const auto end = std::chrono::steady_clock::now();

    WorkloadSummary summary{};
    summary.model_name = params.model_name;
    summary.lambda = params.lambda;
    summary.alpha = params.alpha;
    summary.n_reps = params.replications;
    summary.threads_used = static_cast<int>(workers);
    summary.seed = params.seed;
    summary.mean_workload = mean;
    summary.std_workload = std;
    summary.runtime_seconds = std::chrono::duration<double>(end - start).count();

    if (per_rep_estimates != nullptr) {
        *per_rep_estimates = std::move(rep_estimates);
    }

    return summary;
}

}  // namespace wck
