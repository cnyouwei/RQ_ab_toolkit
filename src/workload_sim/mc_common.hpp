#pragma once

// Internal scaffolding shared by the single-station and tandem workload
// Monte-Carlo drivers. Everything here is deterministic given its inputs;
// derive_rep_seed is part of the frozen seed machinery.

#include "wck/common/distributions.hpp"
#include "wck/common/hash.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace wck::mc {

// Frozen: per-replication seed derivation. The offset constant must not
// change or same-seed runs stop being reproducible.
inline std::uint64_t derive_rep_seed(std::uint64_t base_seed, std::uint64_t tuple_hash_value, std::size_t rep_index) {
    std::uint64_t x = splitmix64(base_seed + 0xD1B54A32D192ED03ULL);
    x ^= splitmix64(tuple_hash_value);
    x ^= splitmix64(static_cast<std::uint64_t>(rep_index));
    return splitmix64(x);
}

inline double primitive_linear_decay(double workload_start, double elapsed) {
    if (elapsed <= 0.0) {
        return 0.0;
    }
    if (elapsed >= workload_start) {
        return 0.5 * workload_start * workload_start;
    }
    return workload_start * elapsed - 0.5 * elapsed * elapsed;
}

// Integral of the linearly decaying workload over [t0,t1] clipped to the
// sampling window [sample_start, sample_end].
inline double integrate_segment(
    double t0,
    double t1,
    double workload_start,
    double sample_start,
    double sample_end) {
    const double left = std::max(t0, sample_start);
    const double right = std::min(t1, sample_end);
    if (!(right > left)) {
        return 0.0;
    }

    const double x_left = left - t0;
    const double x_right = right - t0;
    return primitive_linear_decay(workload_start, x_right)
        - primitive_linear_decay(workload_start, x_left);
}

// Welford running mean over ascending replication index; the iteration order
// is part of the bit-reproducibility contract.
inline double running_mean(const std::vector<double>& values) {
    double mean = 0.0;
    for (std::size_t i = 0U; i < values.size(); ++i) {
        mean += (values[i] - mean) / static_cast<double>(i + 1U);
    }
    return mean;
}

// Two-pass sample standard deviation around a precomputed mean.
inline double sample_std(const std::vector<double>& values, double mean) {
    if (values.size() < 2U) {
        return 0.0;
    }
    double sum_sq = 0.0;
    for (double x : values) {
        const double delta = x - mean;
        sum_sq += delta * delta;
    }
    double var = sum_sq / static_cast<double>(values.size() - 1U);
    if (var < 0.0 && var > -1e-12) {
        var = 0.0;
    }
    return std::sqrt(var);
}

inline void validate_mc_run_params(
    double lambda,
    double alpha,
    double warmup_time,
    double sample_time,
    int replications,
    int threads) {
    if (!(lambda > 0.0) || !std::isfinite(lambda)) {
        throw std::invalid_argument("lambda must be finite and > 0");
    }
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        throw std::invalid_argument("alpha must be finite and > 0");
    }
    if (!(warmup_time >= 0.0) || !std::isfinite(warmup_time)) {
        throw std::invalid_argument("warmup_time must be finite and >= 0");
    }
    if (!(sample_time > 0.0) || !std::isfinite(sample_time)) {
        throw std::invalid_argument("sample_time must be finite and > 0");
    }
    if (replications < 1) {
        throw std::invalid_argument("replications must be >= 1");
    }
    if (threads < 0) {
        throw std::invalid_argument("threads must be >= 0");
    }
}

// Rescales an arrival spec so its effective arrival rate equals lambda.
inline DistributionSpec rescale_arrival_to_rate(
    const DistributionSpec& spec,
    double lambda,
    const std::string& context) {
    const DistributionMoments base = distribution_moments(spec);
    if (!(base.mean > 0.0) || !std::isfinite(base.mean)) {
        throw std::invalid_argument(context + " mean must be finite and > 0");
    }
    const double base_rate = 1.0 / base.mean;
    const double rate_scale = lambda / base_rate;
    return scale_distribution_rates(spec, rate_scale);
}

// Optionally rescales a service spec so its mean is one.
inline DistributionSpec rescale_service_normalized(
    const DistributionSpec& spec,
    bool normalize_mean_to_one,
    const std::string& context) {
    const DistributionMoments base = distribution_moments(spec);
    if (!(base.mean > 0.0) || !std::isfinite(base.mean)) {
        throw std::invalid_argument(context + " mean must be finite and > 0");
    }
    const double scale = normalize_mean_to_one ? base.mean : 1.0;
    return scale_distribution_rates(spec, scale);
}

// Resolves the worker count: threads == 0 means auto-detect, and the result
// is clamped to [1, n_items].
inline std::size_t resolve_worker_count(int threads, std::size_t n_items) {
    std::size_t resolved = 1U;
    if (threads == 0) {
        resolved = static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (resolved == 0U) {
            resolved = 1U;
        }
    } else {
        resolved = static_cast<std::size_t>(threads);
    }
    return std::max<std::size_t>(1U, std::min(resolved, n_items));
}

}  // namespace wck::mc
