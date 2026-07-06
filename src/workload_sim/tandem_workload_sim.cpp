#include "wck/workload_sim/tandem_workload_sim.hpp"

#include "wck/common/distributions.hpp"
#include "wck/common/hash.hpp"
#include "wck/common/parallel.hpp"

#include "mc_common.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace wck {

namespace {

// Frozen: like the single-station tuple hash but additionally mixes the
// queue-1 traffic intensity; do not modify.
std::uint64_t tuple_hash(const TandemWorkloadRunParams& params) {
    std::uint64_t x = splitmix64(fnv1a_64(params.model_name));
    x ^= splitmix64(std::bit_cast<std::uint64_t>(params.lambda));
    x ^= splitmix64(std::bit_cast<std::uint64_t>(params.alpha));
    x ^= splitmix64(std::bit_cast<std::uint64_t>(params.queue1_traffic_intensity));
    return splitmix64(x);
}

// Frozen RNG draw order and event logic for the tandem system.
double simulate_one_replication(
    std::uint64_t seed,
    const DistributionSpec& queue1_arrival,
    const DistributionSpec& queue1_service,
    const DistributionSpec& queue2_service,
    const DistributionSpec& queue2_patience,
    double warmup_time,
    double sample_time) {
    std::mt19937_64 rng(seed);
    DistributionSampler q1_arrival_sampler(queue1_arrival, &rng);
    DistributionSampler q1_service_sampler(queue1_service, &rng);
    DistributionSampler q2_service_sampler(queue2_service, &rng);
    DistributionSampler q2_patience_sampler(queue2_patience, &rng);

    const double sample_start = warmup_time;
    const double sample_end = warmup_time + sample_time;

    double t = 0.0;
    double area = 0.0;
    double w2 = 0.0;

    const double inf = std::numeric_limits<double>::infinity();
    double next_external_arrival = q1_arrival_sampler.sample();
    double next_q1_departure = inf;
    bool q1_busy = false;
    std::deque<double> q1_waiting{};

    while (t < sample_end) {
        const double t_next = std::min(sample_end, std::min(next_external_arrival, next_q1_departure));
        const double elapsed = t_next - t;
        area += mc::integrate_segment(t, t_next, w2, sample_start, sample_end);
        w2 = std::max(w2 - elapsed, 0.0);
        t = t_next;

        if (!(t < sample_end)) {
            break;
        }

        if (next_external_arrival <= next_q1_departure) {
            const double s1 = q1_service_sampler.sample();
            if (!q1_busy) {
                q1_busy = true;
                next_q1_departure = t + s1;
            } else {
                q1_waiting.push_back(s1);
            }
            next_external_arrival = t + q1_arrival_sampler.sample();
            continue;
        }

        // Queue-1 departure immediately becomes Queue-2 arrival.
        const double s2 = q2_service_sampler.sample();
        const double p2 = q2_patience_sampler.sample();
        if (p2 > w2) {
            w2 += s2;
        }

        if (q1_waiting.empty()) {
            q1_busy = false;
            next_q1_departure = inf;
        } else {
            const double next_s1 = q1_waiting.front();
            q1_waiting.pop_front();
            next_q1_departure = t + next_s1;
        }
    }

    return area / sample_time;
}

}  // namespace

TandemWorkloadSummary simulate_tandem_workload_mc(
    const TandemWorkloadRunParams& params,
    std::vector<double>* per_rep_estimates) {
    mc::validate_mc_run_params(
        params.lambda,
        params.alpha,
        params.warmup_time,
        params.sample_time,
        params.replications,
        params.threads);
    if (!(params.queue1_traffic_intensity > 0.0) || !(params.queue1_traffic_intensity < 1.0)
        || !std::isfinite(params.queue1_traffic_intensity)) {
        throw std::invalid_argument("queue1_traffic_intensity must be finite and in (0,1)");
    }

    validate_distribution_spec(params.queue1_arrival, "queue1 arrival distribution");
    validate_distribution_spec(params.queue1_service, "queue1 service distribution");
    validate_distribution_spec(params.queue2_service, "queue2 service distribution");
    validate_distribution_spec(params.queue2_patience, "queue2 patience distribution");

    const DistributionSpec q1_arrival_runtime =
        mc::rescale_arrival_to_rate(params.queue1_arrival, params.lambda, "queue1 arrival");

    // Queue-1 service is rescaled so that rho1 = lambda / mu1 equals the
    // configured traffic intensity.
    const DistributionMoments q1_service_base = distribution_moments(params.queue1_service);
    if (!(q1_service_base.mean > 0.0) || !std::isfinite(q1_service_base.mean)) {
        throw std::invalid_argument("queue1 service mean must be finite and > 0");
    }
    const double mu1_target = params.lambda / params.queue1_traffic_intensity;
    const double q1_service_target_mean = 1.0 / mu1_target;
    const double q1_service_rate_scale = q1_service_base.mean / q1_service_target_mean;
    const DistributionSpec q1_service_runtime =
        scale_distribution_rates(params.queue1_service, q1_service_rate_scale);

    const DistributionSpec q2_service_runtime = mc::rescale_service_normalized(
        params.queue2_service, params.normalize_service_mean_to_one, "queue2 service");
    const DistributionSpec q2_patience_runtime =
        scale_distribution_rates(params.queue2_patience, params.alpha);

    const std::size_t n_reps = static_cast<std::size_t>(params.replications);
    const std::size_t workers = mc::resolve_worker_count(params.threads, n_reps);

    const auto start = std::chrono::steady_clock::now();

    std::vector<double> rep_estimates(n_reps, std::numeric_limits<double>::quiet_NaN());
    const std::uint64_t thash = tuple_hash(params);

    parallel_for_index(n_reps, workers, [&](std::size_t rep) {
        const std::uint64_t rep_seed = mc::derive_rep_seed(params.seed, thash, rep);
        rep_estimates[rep] = simulate_one_replication(
            rep_seed,
            q1_arrival_runtime,
            q1_service_runtime,
            q2_service_runtime,
            q2_patience_runtime,
            params.warmup_time,
            params.sample_time);
    });

    // Sequential ascending-index reductions: part of the bit-reproducibility
    // contract for any thread count.
    const double mean = mc::running_mean(rep_estimates);
    const double std = mc::sample_std(rep_estimates, mean);

    const auto end = std::chrono::steady_clock::now();

    TandemWorkloadSummary summary{};
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
