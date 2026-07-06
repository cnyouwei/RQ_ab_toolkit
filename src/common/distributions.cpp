#include "wck/common/distributions.hpp"

#include <cmath>
#include <stdexcept>

namespace wck {

namespace {

[[noreturn]] void fail(const std::string& context, const std::string& message) {
    throw std::invalid_argument(context + ": " + message);
}

long double inverse_factorial(int n) {
    long double inv = 1.0L;
    for (int i = 2; i <= n; ++i) {
        inv /= static_cast<long double>(i);
    }
    return inv;
}

}  // namespace

std::string distribution_family_name(DistributionFamily family) {
    switch (family) {
    case DistributionFamily::kExponential:
        return "exponential";
    case DistributionFamily::kErlangK:
        return "erlang_k";
    case DistributionFamily::kLognormal:
        return "lognormal";
    case DistributionFamily::kHyperexponential2:
        return "hyperexponential2";
    }
    return "unknown";
}

void validate_distribution_spec(const DistributionSpec& spec, const std::string& context) {
    switch (spec.family) {
    case DistributionFamily::kExponential:
        if (!(spec.exponential.rate > 0.0)) {
            fail(context, "exponential.rate must be > 0");
        }
        return;
    case DistributionFamily::kErlangK:
        if (spec.erlang_k.k < 1) {
            fail(context, "erlang_k.k must be >= 1");
        }
        if (!(spec.erlang_k.rate > 0.0)) {
            fail(context, "erlang_k.rate must be > 0");
        }
        return;
    case DistributionFamily::kLognormal:
        if (!(spec.lognormal.mean > 0.0) || !std::isfinite(spec.lognormal.mean)) {
            fail(context, "lognormal.mean must be finite and > 0");
        }
        if (!(spec.lognormal.scv > 0.0) || !std::isfinite(spec.lognormal.scv)) {
            fail(context, "lognormal.scv must be finite and > 0");
        }
        return;
    case DistributionFamily::kHyperexponential2:
        if (!(spec.hyperexponential2.p > 0.0 && spec.hyperexponential2.p < 1.0)) {
            fail(context, "hyperexponential2.p must be strictly between 0 and 1");
        }
        if (!(spec.hyperexponential2.rate1 > 0.0)) {
            fail(context, "hyperexponential2.rate1 must be > 0");
        }
        if (!(spec.hyperexponential2.rate2 > 0.0)) {
            fail(context, "hyperexponential2.rate2 must be > 0");
        }
        return;
    }

    fail(context, "unknown distribution family");
}

DistributionMoments distribution_moments(const DistributionSpec& spec) {
    validate_distribution_spec(spec, "distribution");
    DistributionMoments out{};

    switch (spec.family) {
    case DistributionFamily::kExponential:
        out.mean = 1.0 / spec.exponential.rate;
        out.scv = 1.0;
        return out;
    case DistributionFamily::kErlangK:
        out.mean = static_cast<double>(spec.erlang_k.k) / spec.erlang_k.rate;
        out.scv = 1.0 / static_cast<double>(spec.erlang_k.k);
        return out;
    case DistributionFamily::kLognormal:
        out.mean = spec.lognormal.mean;
        out.scv = spec.lognormal.scv;
        return out;
    case DistributionFamily::kHyperexponential2: {
        const double p = spec.hyperexponential2.p;
        const double q = 1.0 - p;
        const double r1 = spec.hyperexponential2.rate1;
        const double r2 = spec.hyperexponential2.rate2;

        const double mean = p / r1 + q / r2;
        const double second = 2.0 * (p / (r1 * r1) + q / (r2 * r2));
        double var = second - mean * mean;
        if (var < 0.0 && var > -1e-12) {
            var = 0.0;
        }

        out.mean = mean;
        out.scv = var / (mean * mean);
        return out;
    }
    }

    throw std::invalid_argument("distribution: unknown family");
}

double distribution_beta_at_zero(const DistributionSpec& spec, int k) {
    validate_distribution_spec(spec, "distribution");
    if (k < 1) {
        throw std::invalid_argument("distribution beta order k must be >= 1");
    }

    const long double inv_fact = inverse_factorial(k);
    long double beta = 0.0L;

    switch (spec.family) {
    case DistributionFamily::kExponential: {
        const long double rate = static_cast<long double>(spec.exponential.rate);
        const long double sign = (k % 2 == 1) ? 1.0L : -1.0L;
        beta = sign * std::pow(rate, static_cast<long double>(k)) * inv_fact;
        break;
    }
    case DistributionFamily::kErlangK: {
        const int m = spec.erlang_k.k;
        const long double rate = static_cast<long double>(spec.erlang_k.rate);

        // F(x) = 1 - e^{-x} * sum_{j=0}^{m-1} x^j/j!, x = rate * t.
        // beta = coefficient of t^k in F(t) = F^(k)(0)/k!.
        long double coeff_xk = 0.0L;
        const int j_max = std::min(k, m - 1);
        for (int j = 0; j <= j_max; ++j) {
            const int rem = k - j;
            long double term = inverse_factorial(j) * inverse_factorial(rem);
            if ((rem % 2) == 1) {
                term = -term;
            }
            coeff_xk += term;
        }
        coeff_xk = -coeff_xk;
        beta = coeff_xk * std::pow(rate, static_cast<long double>(k));
        break;
    }
    case DistributionFamily::kLognormal:
        // Lognormal CDF and all derivatives vanish at the origin.
        beta = 0.0L;
        break;
    case DistributionFamily::kHyperexponential2: {
        const long double p = static_cast<long double>(spec.hyperexponential2.p);
        const long double q = 1.0L - p;
        const long double r1 = static_cast<long double>(spec.hyperexponential2.rate1);
        const long double r2 = static_cast<long double>(spec.hyperexponential2.rate2);
        const long double sign = (k % 2 == 1) ? 1.0L : -1.0L;
        const long double weighted =
            p * std::pow(r1, static_cast<long double>(k))
            + q * std::pow(r2, static_cast<long double>(k));
        beta = sign * weighted * inv_fact;
        break;
    }
    }

    const double beta_d = static_cast<double>(beta);
    if (!std::isfinite(beta_d)) {
        throw std::invalid_argument("distribution beta at zero is not finite");
    }
    return beta_d;
}

DistributionSpec scale_distribution_rates(const DistributionSpec& spec, double factor) {
    validate_distribution_spec(spec, "distribution");
    if (!(factor > 0.0) || !std::isfinite(factor)) {
        throw std::invalid_argument("distribution rate scale factor must be finite and > 0");
    }

    DistributionSpec scaled = spec;
    switch (scaled.family) {
    case DistributionFamily::kExponential:
        scaled.exponential.rate *= factor;
        break;
    case DistributionFamily::kErlangK:
        scaled.erlang_k.rate *= factor;
        break;
    case DistributionFamily::kLognormal:
        scaled.lognormal.mean /= factor;
        break;
    case DistributionFamily::kHyperexponential2:
        scaled.hyperexponential2.rate1 *= factor;
        scaled.hyperexponential2.rate2 *= factor;
        break;
    }
    return scaled;
}

DistributionSampler::DistributionSampler(const DistributionSpec& spec, std::mt19937_64* rng)
    : family_(spec.family),
      rng_(rng),
      exp_(1.0),
      erlang_(1.0, 1.0),
      lognormal_(0.0, 1.0),
      hyper2_branch_(0.5),
      hyper2_exp1_(1.0),
      hyper2_exp2_(1.0) {
    if (rng_ == nullptr) {
        throw std::invalid_argument("DistributionSampler requires non-null rng");
    }
    validate_distribution_spec(spec, "distribution");

    switch (spec.family) {
    case DistributionFamily::kExponential:
        exp_ = std::exponential_distribution<double>(spec.exponential.rate);
        break;
    case DistributionFamily::kErlangK:
        erlang_ = std::gamma_distribution<double>(
            static_cast<double>(spec.erlang_k.k),
            1.0 / spec.erlang_k.rate);
        break;
    case DistributionFamily::kLognormal: {
        const double sigma2 = std::log1p(spec.lognormal.scv);
        const double sigma = std::sqrt(sigma2);
        const double mu = std::log(spec.lognormal.mean) - 0.5 * sigma2;
        lognormal_ = std::lognormal_distribution<double>(mu, sigma);
        break;
    }
    case DistributionFamily::kHyperexponential2:
        hyper2_branch_ = std::bernoulli_distribution(spec.hyperexponential2.p);
        hyper2_exp1_ = std::exponential_distribution<double>(spec.hyperexponential2.rate1);
        hyper2_exp2_ = std::exponential_distribution<double>(spec.hyperexponential2.rate2);
        break;
    }
}

double DistributionSampler::sample() {
    switch (family_) {
    case DistributionFamily::kExponential:
        return exp_(*rng_);
    case DistributionFamily::kErlangK:
        return erlang_(*rng_);
    case DistributionFamily::kLognormal:
        return lognormal_(*rng_);
    case DistributionFamily::kHyperexponential2:
        return hyper2_branch_(*rng_) ? hyper2_exp1_(*rng_) : hyper2_exp2_(*rng_);
    }
    throw std::invalid_argument("DistributionSampler: unknown family");
}

}  // namespace wck
