#pragma once

#include <cstdint>
#include <random>
#include <string>

namespace wck {

enum class DistributionFamily {
    kExponential,
    kErlangK,
    kLognormal,
    kHyperexponential2,
};

struct ExponentialParams {
    double rate = 1.0;
};

struct ErlangKParams {
    int k = 1;
    double rate = 1.0;
};

struct LognormalParams {
    double mean = 1.0;
    double scv = 1.0;
};

struct Hyperexponential2Params {
    double p = 0.5;
    double rate1 = 1.0;
    double rate2 = 1.0;
};

struct DistributionSpec {
    DistributionFamily family = DistributionFamily::kExponential;
    ExponentialParams exponential{};
    ErlangKParams erlang_k{};
    LognormalParams lognormal{};
    Hyperexponential2Params hyperexponential2{};
};

struct DistributionMoments {
    double mean = 0.0;
    double scv = 0.0;
};

std::string distribution_family_name(DistributionFamily family);

void validate_distribution_spec(const DistributionSpec& spec, const std::string& context);

DistributionMoments distribution_moments(const DistributionSpec& spec);

double distribution_beta_at_zero(const DistributionSpec& spec, int k);

DistributionSpec scale_distribution_rates(const DistributionSpec& spec, double factor);

class DistributionSampler {
public:
    DistributionSampler(const DistributionSpec& spec, std::mt19937_64* rng);

    double sample();

private:
    DistributionFamily family_ = DistributionFamily::kExponential;
    std::mt19937_64* rng_ = nullptr;
    std::exponential_distribution<double> exp_{1.0};
    std::gamma_distribution<double> erlang_{1.0, 1.0};
    std::lognormal_distribution<double> lognormal_{0.0, 1.0};
    std::bernoulli_distribution hyper2_branch_{0.5};
    std::exponential_distribution<double> hyper2_exp1_{1.0};
    std::exponential_distribution<double> hyper2_exp2_{1.0};
};

}  // namespace wck
