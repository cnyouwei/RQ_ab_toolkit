#include "wck/idw_sim/effective_idw_sim.hpp"
#include "wck/idw_sim/effective_idw_sim_config.hpp"
#include "wck/common/distributions.hpp"

#include "support.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <array>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string replace_once(std::string text, const std::string& needle, const std::string& replacement) {
    const std::size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        throw std::runtime_error("replace_once failed; needle not found: " + needle);
    }
    text.replace(pos, needle.size(), replacement);
    return text;
}

std::string valid_config_json() {
    return R"JSON({
  "alpha": { "indices": [0], "base": 2.0 },
  "simulation": {
    "warmup_time": 10.0,
    "sample_time": 200.0,
    "tau": 0.02,
    "max_level": 8,
    "min_windows_per_t": 10,
    "n_tau_shifts": 1,
    "threads": 1,
    "seed": 123,
    "save_event_trace": false
  },
  "models": [
    {
      "name": "M/M/1+M",
      "arrival": {
        "distribution": { "family": "exponential", "params": { "rate": 1.0 } }
      },
      "system": { "c": 0.2 },
      "service": {
        "distribution": { "family": "exponential", "params": { "rate": 1.0 } }
      },
      "patience": {
        "distribution": { "family": "erlang_k", "params": { "k": 1, "rate": 1.0 } }
      },
      "scaling": { "k": 1, "beta_patience": 1.0 }
    }
  ]
})JSON";
}

std::string config_with_distribution_snippets(
    const std::string& arrival_dist,
    const std::string& service_dist,
    const std::string& patience_dist) {
    std::string cfg = R"JSON({
  "alpha": { "indices": [0], "base": 2.0 },
  "simulation": {
    "warmup_time": 10.0,
    "sample_time": 200.0,
    "tau": 0.02,
    "max_level": 8,
    "min_windows_per_t": 10,
    "n_tau_shifts": 1,
    "threads": 1,
    "seed": 123,
    "save_event_trace": false
  },
  "models": [
    {
      "name": "dist_smoke",
      "arrival": {
        "distribution": "__ARRIVAL_DIST__"
      },
      "system": { "c": 0.2 },
      "service": {
        "distribution": "__SERVICE_DIST__"
      },
      "patience": {
        "distribution": "__PATIENCE_DIST__"
      },
      "scaling": { "k": 1 }
    }
  ]
})JSON";
    cfg = replace_once(cfg, "\"__ARRIVAL_DIST__\"", arrival_dist);
    cfg = replace_once(cfg, "\"__SERVICE_DIST__\"", service_dist);
    cfg = replace_once(cfg, "\"__PATIENCE_DIST__\"", patience_dist);
    return cfg;
}

double csv_column_mean(const std::filesystem::path& path, int zero_based_column, std::size_t* n_rows = nullptr) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open csv: " + path.string());
    }
    std::string line;
    std::getline(in, line);  // header

    std::size_t n = 0U;
    double mean = 0.0;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string field;
        for (int col = 0; col <= zero_based_column; ++col) {
            if (!std::getline(ss, field, ',')) {
                throw std::runtime_error("malformed csv row in " + path.string());
            }
        }
        const double x = std::stod(field);
        ++n;
        mean += (x - mean) / static_cast<double>(n);
    }
    if (n_rows != nullptr) {
        *n_rows = n;
    }
    if (n == 0U) {
        throw std::runtime_error("csv has no data rows: " + path.string());
    }
    return mean;
}

void test_config_validation() {
    const auto dir = make_temp_dir("wck_cfg_validation");

    const auto valid_path = write_text_file(dir, "valid.json", valid_config_json());
    const auto cfg = wck::load_effective_idw_sim_config(valid_path);
    expect(cfg.simulation.threads == 1, "config validation: default threads parse mismatch");
    expect(std::abs(cfg.simulation.tau - 0.02) < 1e-12, "config validation: tau parse mismatch");
    expect(
        std::abs(cfg.models[0].scaling.beta_patience - 1.0) < 1e-12,
        "config validation: expected derived beta_patience for M patience");

    const auto no_beta_cfg = replace_once(
        valid_config_json(),
        "\"scaling\": { \"k\": 1, \"beta_patience\": 1.0 }",
        "\"scaling\": { \"k\": 1 }");
    const auto no_beta_path = write_text_file(dir, "no_beta.json", no_beta_cfg);
    const auto no_beta = wck::load_effective_idw_sim_config(no_beta_path);
    expect(
        std::abs(no_beta.models[0].scaling.beta_patience - 1.0) < 1e-12,
        "config validation: omitted beta_patience should be derived from patience distribution");

    const auto inconsistent_beta_cfg = replace_once(
        valid_config_json(),
        "\"beta_patience\": 1.0",
        "\"beta_patience\": 0.75");
    const auto inconsistent_beta_path = write_text_file(dir, "inconsistent_beta.json", inconsistent_beta_cfg);
    expect_throw_contains(
        [&]() { (void)wck::load_effective_idw_sim_config(inconsistent_beta_path); },
        "model.scaling.beta_patience",
        "config validation: inconsistent beta_patience should be rejected");

    const auto valid_e2 = replace_once(
        valid_config_json(),
        "\"distribution\": { \"family\": \"exponential\", \"params\": { \"rate\": 1.0 } }",
        "\"distribution\": { \"family\": \"erlang_k\", \"params\": { \"k\": 2, \"rate\": 2.0 } }");
    const auto valid_e2_path = write_text_file(dir, "valid_e2.json", valid_e2);
    const auto cfg_e2 = wck::load_effective_idw_sim_config(valid_e2_path);
    expect(cfg_e2.models.size() == 1U, "config validation: expected one model in e2 config");
    expect(
        cfg_e2.models[0].arrival.family == wck::DistributionFamily::kErlangK,
        "config validation: erlang family parse mismatch");
    expect(cfg_e2.models[0].arrival.erlang_k.k == 2, "config validation: erlang k parse mismatch");

    const auto all_families_a = config_with_distribution_snippets(
        R"JSON({ "family": "hyperexponential2", "params": { "p": 0.5, "rate1": 3.0, "rate2": 0.3333333333333333 } })JSON",
        R"JSON({ "family": "erlang_k", "params": { "k": 3, "rate": 1.8 } })JSON",
        R"JSON({ "family": "exponential", "params": { "rate": 2.2 } })JSON");
    const auto all_families_a_path = write_text_file(dir, "all_families_a.json", all_families_a);
    const auto cfg_families_a = wck::load_effective_idw_sim_config(all_families_a_path);
    expect(
        cfg_families_a.models[0].arrival.family == wck::DistributionFamily::kHyperexponential2,
        "config validation: arrival h2 parse mismatch");
    expect(
        cfg_families_a.models[0].service.family == wck::DistributionFamily::kErlangK,
        "config validation: service erlang parse mismatch");
    expect(
        cfg_families_a.models[0].patience.family == wck::DistributionFamily::kExponential,
        "config validation: patience exponential parse mismatch");

    const auto all_families_b = config_with_distribution_snippets(
        R"JSON({ "family": "erlang_k", "params": { "k": 2, "rate": 2.0 } })JSON",
        R"JSON({ "family": "exponential", "params": { "rate": 1.1 } })JSON",
        R"JSON({ "family": "hyperexponential2", "params": { "p": 0.7, "rate1": 3.5, "rate2": 0.4 } })JSON");
    const auto all_families_b_path = write_text_file(dir, "all_families_b.json", all_families_b);
    const auto cfg_families_b = wck::load_effective_idw_sim_config(all_families_b_path);
    expect(
        cfg_families_b.models[0].arrival.family == wck::DistributionFamily::kErlangK,
        "config validation: arrival erlang parse mismatch");
    expect(
        cfg_families_b.models[0].service.family == wck::DistributionFamily::kExponential,
        "config validation: service exponential parse mismatch");
    expect(
        cfg_families_b.models[0].patience.family == wck::DistributionFamily::kHyperexponential2,
        "config validation: patience h2 parse mismatch");

    const auto invalid_erlang_k = replace_once(valid_e2, "\"k\": 2", "\"k\": 0");
    const auto invalid_e2_scv_path = write_text_file(dir, "invalid_erlang_k.json", invalid_erlang_k);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_e2_scv_path); },
        "config validation: erlang k < 1 should throw");

    const auto missing_sim = write_text_file(
        dir,
        "missing_sim.json",
        R"JSON({
  "alpha": { "indices": [0], "base": 2.0 },
  "models": []
})JSON");
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(missing_sim); },
        "config validation: missing simulation should throw");

    const auto invalid_sample = replace_once(
        valid_config_json(),
        "\"sample_time\": 200.0",
        "\"sample_time\": 0.0");
    const auto invalid_sample_path = write_text_file(dir, "invalid_sample.json", invalid_sample);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_sample_path); },
        "config validation: nonpositive sample_time should throw");

    const auto invalid_tau = replace_once(
        valid_config_json(),
        "\"tau\": 0.02",
        "\"tau\": 0.0");
    const auto invalid_tau_path = write_text_file(dir, "invalid_tau.json", invalid_tau);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_tau_path); },
        "config validation: nonpositive tau should throw");

    const auto invalid_level = replace_once(
        valid_config_json(),
        "\"max_level\": 8",
        "\"max_level\": -1");
    const auto invalid_level_path = write_text_file(dir, "invalid_level.json", invalid_level);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_level_path); },
        "config validation: negative max_level should throw");

    const auto invalid_min_windows = replace_once(
        valid_config_json(),
        "\"min_windows_per_t\": 10",
        "\"min_windows_per_t\": 1");
    const auto invalid_min_windows_path = write_text_file(dir, "invalid_min_windows.json", invalid_min_windows);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_min_windows_path); },
        "config validation: min_windows_per_t < 2 should throw");

    const auto invalid_shift_count = replace_once(
        valid_config_json(),
        "\"n_tau_shifts\": 1",
        "\"n_tau_shifts\": 0");
    const auto invalid_shift_count_path = write_text_file(dir, "invalid_shift_count.json", invalid_shift_count);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_shift_count_path); },
        "config validation: n_tau_shifts < 1 should throw");

    const auto invalid_arrival = replace_once(
        valid_config_json(),
        "\"family\": \"exponential\"",
        "\"family\": \"mmpp\"");
    const auto invalid_arrival_path = write_text_file(dir, "invalid_arrival.json", invalid_arrival);
    expect_throw_contains(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_arrival_path); },
        "model.arrival.distribution.family",
        "config validation: invalid arrival family should include field path");

    const auto missing_arrival_rate = replace_once(
        valid_config_json(),
        "\"params\": { \"rate\": 1.0 }",
        "\"params\": {}");
    const auto missing_arrival_rate_path = write_text_file(dir, "missing_arrival_rate.json", missing_arrival_rate);
    expect_throw_contains(
        [&]() { (void)wck::load_effective_idw_sim_config(missing_arrival_rate_path); },
        "model.arrival.distribution.params.rate",
        "config validation: missing arrival rate should include field path");

    const auto invalid_threads_negative = replace_once(
        valid_config_json(),
        "\"threads\": 1",
        "\"threads\": -2");
    const auto invalid_threads_negative_path =
        write_text_file(dir, "invalid_threads_negative.json", invalid_threads_negative);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_threads_negative_path); },
        "config validation: negative threads should throw");

    const auto invalid_threads_fraction = replace_once(
        valid_config_json(),
        "\"threads\": 1",
        "\"threads\": 1.25");
    const auto invalid_threads_fraction_path =
        write_text_file(dir, "invalid_threads_fraction.json", invalid_threads_fraction);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_threads_fraction_path); },
        "config validation: non-integer threads should throw");

    const auto legacy_key = replace_once(
        valid_config_json(),
        "\"min_windows_per_t\": 10,",
        "\"min_windows_per_t\": 10,\n    \"overlap_stride\": \"half\",");
    const auto legacy_key_path = write_text_file(dir, "legacy_key.json", legacy_key);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(legacy_key_path); },
        "config validation: legacy overlap_stride should be rejected");

    const auto legacy_grid = replace_once(
        valid_config_json(),
        "\"tau\": 0.02,",
        "\"tau\": 0.02,\n    \"estimate_grid\": { \"t_min\": 0.1, \"t_max\": 1.0, \"n_t\": 10 },");
    const auto legacy_grid_path = write_text_file(dir, "legacy_grid.json", legacy_grid);
    expect_throw(
        [&]() { (void)wck::load_effective_idw_sim_config(legacy_grid_path); },
        "config validation: legacy estimate_grid should be rejected");

    const auto legacy_arrival_type = replace_once(
        valid_config_json(),
        "\"arrival\": {\n        \"distribution\": { \"family\": \"exponential\", \"params\": { \"rate\": 1.0 } }\n      },",
        "\"arrival\": {\n        \"type\": \"exp\",\n        \"distribution\": { \"family\": \"exponential\", \"params\": { \"rate\": 1.0 } }\n      },");
    const auto legacy_arrival_type_path = write_text_file(dir, "legacy_arrival_type.json", legacy_arrival_type);
    expect_throw_contains(
        [&]() { (void)wck::load_effective_idw_sim_config(legacy_arrival_type_path); },
        "model.arrival.type",
        "config validation: legacy arrival.type should provide migration error");

    const auto legacy_service_mu = replace_once(
        valid_config_json(),
        "\"service\": {\n        \"distribution\": { \"family\": \"exponential\", \"params\": { \"rate\": 1.0 } }\n      },",
        "\"service\": {\n        \"mu\": 1.0,\n        \"distribution\": { \"family\": \"exponential\", \"params\": { \"rate\": 1.0 } }\n      },");
    const auto legacy_service_mu_path = write_text_file(dir, "legacy_service_mu.json", legacy_service_mu);
    expect_throw_contains(
        [&]() { (void)wck::load_effective_idw_sim_config(legacy_service_mu_path); },
        "model.service.mu",
        "config validation: legacy service.mu should provide migration error");

    const auto legacy_patience_k = replace_once(
        valid_config_json(),
        "\"patience\": {\n        \"distribution\": { \"family\": \"erlang_k\", \"params\": { \"k\": 1, \"rate\": 1.0 } }\n      },",
        "\"patience\": {\n        \"k\": 1,\n        \"distribution\": { \"family\": \"erlang_k\", \"params\": { \"k\": 1, \"rate\": 1.0 } }\n      },");
    const auto legacy_patience_k_path = write_text_file(dir, "legacy_patience_k.json", legacy_patience_k);
    expect_throw_contains(
        [&]() { (void)wck::load_effective_idw_sim_config(legacy_patience_k_path); },
        "model.patience.k",
        "config validation: legacy patience.k should provide migration error");

    const auto legacy_model_rho_exponent = replace_once(
        valid_config_json(),
        "\"scaling\": { \"k\": 1, \"beta_patience\": 1.0 }",
        "\"rho_exponent\": 0.5,\n      \"scaling\": { \"k\": 1, \"beta_patience\": 1.0 }");
    const auto legacy_model_rho_exponent_path =
        write_text_file(dir, "legacy_model_rho_exponent.json", legacy_model_rho_exponent);
    expect_throw_contains(
        [&]() { (void)wck::load_effective_idw_sim_config(legacy_model_rho_exponent_path); },
        "model.rho_exponent",
        "config validation: legacy model.rho_exponent should provide migration error");
}

void test_tau_shift_grid() {
    const auto single = wck::build_tau_shift_grid(0.1, 1);
    expect(single.size() == 1U, "tau shifts: single size mismatch");
    expect(std::abs(single[0] - 0.1) < 1e-12, "tau shifts: single value mismatch");

    const auto shifts = wck::build_tau_shift_grid(0.1, 3);
    expect(shifts.size() == 3U, "tau shifts: size mismatch");
    expect(std::abs(shifts[0] - 0.1) < 1e-12, "tau shifts: first mismatch");
    expect(std::abs(shifts[1] - 0.1 * std::sqrt(2.0)) < 1e-12, "tau shifts: middle mismatch");
    expect(std::abs(shifts[2] - 0.2) < 1e-12, "tau shifts: last mismatch");

    expect_throw(
        [&]() { (void)wck::build_tau_shift_grid(0.0, 2); },
        "tau shifts: nonpositive tau should throw");
    expect_throw(
        [&]() { (void)wck::build_tau_shift_grid(0.1, 0); },
        "tau shifts: nonpositive shift count should throw");
}

void test_h2_roundtrip() {
    constexpr double rate = 1.3;
    constexpr double scv = 4.0;
    constexpr double r = 0.5;

    const wck::H2Params params = wck::recover_h2_params(rate, scv, r);
    expect(params.mu1 + 1e-12 >= params.mu2, "H2 roundtrip: expected mu1 >= mu2");

    const double mean = params.p / params.mu1 + (1.0 - params.p) / params.mu2;
    const double rec_rate = 1.0 / mean;
    const double second = 2.0 * (params.p / (params.mu1 * params.mu1)
        + (1.0 - params.p) / (params.mu2 * params.mu2));
    const double var = second - mean * mean;
    const double rec_scv = var / (mean * mean);
    const double rec_r = (params.p / params.mu1) / mean;

    expect(std::abs(rec_rate - rate) < 1e-10, "H2 roundtrip: rate mismatch");
    expect(std::abs(rec_scv - scv) < 1e-9, "H2 roundtrip: scv mismatch");
    expect(std::abs(rec_r - r) < 1e-10, "H2 roundtrip: r mismatch");
}

void test_e2_arrival_sample_scv() {
    const auto dir = make_temp_dir("wck_e2_arrival");
    const auto event_trace = dir / "e2_events.csv";

    wck::RunParameters p{};
    p.model_name = "e2_arrival_test";
    p.model_index = 0U;
    p.alpha_index = 0;
    p.alpha = 1.0;
    p.h = 0.5;
    p.c = 0.2;
    p.rho_exponent = 0.5;
    p.rho = 2.0;
    p.lambda = 2.0;
    p.arrival = make_erlang(2, 2.0);
    p.service = make_exp(1.0);
    p.patience = make_erlang(1, 1.0);
    p.scaling.k = 1;
    p.scaling.beta_patience = 1.0;
    p.simulation = wck::SimulationConfig{};
    p.simulation.warmup_time = 0.0;
    p.simulation.sample_time = 3000.0;
    p.simulation.tau = 0.05;
    p.simulation.max_level = 8;
    p.simulation.min_windows_per_t = 20;
    p.simulation.n_tau_shifts = 1;
    p.simulation.threads = 1;
    p.seed = 1122334455ULL;

    const auto result = wck::simulate_effective_idw(p, &event_trace);
    expect(!result.estimates.empty(), "e2 arrival: expected nonempty estimates");
    expect(result.stats.arrivals_total > 2000U, "e2 arrival: expected sufficient arrival samples");

    std::ifstream in(event_trace);
    expect(in.is_open(), "e2 arrival: failed to open event trace");

    std::string line;
    std::getline(in, line);  // header

    std::size_t n = 0U;
    double mean = 0.0;
    double m2 = 0.0;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string field;
        for (int col = 0; col < 3; ++col) {
            if (!std::getline(ss, field, ',')) {
                throw std::runtime_error("e2 arrival: malformed event trace row");
            }
        }

        const double x = std::stod(field);
        ++n;
        const double delta = x - mean;
        mean += delta / static_cast<double>(n);
        const double delta2 = x - mean;
        m2 += delta * delta2;
    }

    expect(n > 2000U, "e2 arrival: insufficient interarrival samples");
    const double var = m2 / static_cast<double>(n - 1U);
    const double scv = var / (mean * mean);

    expect(std::abs(mean - 0.5) < 0.03, "e2 arrival: interarrival mean mismatch");
    expect(std::abs(scv - 0.5) < 0.06, "e2 arrival: interarrival scv mismatch");
}

void test_arrival_coupling_normalization() {
    const auto dir = make_temp_dir("wck_arrival_coupling");
    const auto trace_slow = dir / "arrival_slow.csv";
    const auto trace_fast = dir / "arrival_fast.csv";

    auto make_params = [](const wck::DistributionSpec& arrival, std::uint64_t seed) {
        wck::RunParameters p{};
        p.model_name = "arrival_coupling";
        p.model_index = 0U;
        p.alpha_index = 0;
        p.alpha = 1.0;
        p.h = 0.5;
        p.c = 0.2;
        p.rho_exponent = 0.5;
        p.rho = 1.5;
        p.lambda = 1.5;
        p.arrival = arrival;
        p.service = make_exp(1.0);
        p.patience = make_erlang(1, 1.0);
        p.scaling.k = 1;
        p.scaling.beta_patience = 1.0;
        p.simulation = wck::SimulationConfig{};
        p.simulation.warmup_time = 0.0;
        p.simulation.sample_time = 2500.0;
        p.simulation.tau = 0.05;
        p.simulation.max_level = 8;
        p.simulation.min_windows_per_t = 20;
        p.simulation.n_tau_shifts = 1;
        p.simulation.threads = 1;
        p.seed = seed;
        return p;
    };

    const auto r1 = wck::simulate_effective_idw(make_params(make_erlang(2, 2.0), 101ULL), &trace_slow);
    const auto r2 = wck::simulate_effective_idw(make_params(make_erlang(2, 8.0), 102ULL), &trace_fast);
    expect(!r1.estimates.empty() && !r2.estimates.empty(), "arrival coupling: expected nonempty estimates");

    std::size_t n1 = 0U;
    std::size_t n2 = 0U;
    const double mean1 = csv_column_mean(trace_slow, 2, &n1);
    const double mean2 = csv_column_mean(trace_fast, 2, &n2);
    expect(n1 > 3000U && n2 > 3000U, "arrival coupling: expected enough arrivals");

    const double target_mean = 1.0 / 1.5;
    expect(std::abs(mean1 - target_mean) < 0.04, "arrival coupling: mean1 did not normalize to 1/lambda");
    expect(std::abs(mean2 - target_mean) < 0.04, "arrival coupling: mean2 did not normalize to 1/lambda");
    expect(std::abs(mean1 - mean2) < 0.03, "arrival coupling: normalized means diverged");
}

void test_patience_alpha_rate_scaling() {
    const auto dir = make_temp_dir("wck_patience_alpha_scale");
    const auto trace_low_alpha = dir / "patience_alpha_low.csv";
    const auto trace_high_alpha = dir / "patience_alpha_high.csv";

    auto make_params = [](double alpha, std::uint64_t seed) {
        wck::RunParameters p{};
        p.model_name = "patience_alpha_scaling";
        p.model_index = 0U;
        p.alpha_index = 0;
        p.alpha = alpha;
        p.h = 0.5;
        p.c = 0.2;
        p.rho_exponent = 0.5;
        p.rho = 1.2;
        p.lambda = 1.2;
        p.arrival = make_exp(1.2);
        p.service = make_exp(1.0);
        p.patience = make_erlang(2, 2.0);  // mean=1.0 before alpha scaling.
        p.scaling.k = 2;
        p.scaling.beta_patience = 2.0;
        p.simulation = wck::SimulationConfig{};
        p.simulation.warmup_time = 0.0;
        p.simulation.sample_time = 3000.0;
        p.simulation.tau = 0.05;
        p.simulation.max_level = 8;
        p.simulation.min_windows_per_t = 20;
        p.simulation.n_tau_shifts = 1;
        p.simulation.threads = 1;
        p.seed = seed;
        return p;
    };

    const auto low = wck::simulate_effective_idw(make_params(0.5, 201ULL), &trace_low_alpha);
    const auto high = wck::simulate_effective_idw(make_params(2.0, 202ULL), &trace_high_alpha);
    expect(!low.estimates.empty() && !high.estimates.empty(), "patience scaling: expected nonempty estimates");

    std::size_t n_low = 0U;
    std::size_t n_high = 0U;
    const double mean_low = csv_column_mean(trace_low_alpha, 5, &n_low);
    const double mean_high = csv_column_mean(trace_high_alpha, 5, &n_high);
    expect(n_low > 2500U && n_high > 2500U, "patience scaling: expected enough samples");

    const double ratio = mean_low / mean_high;
    expect(std::abs(ratio - 4.0) < 0.35, "patience scaling: mean ratio inconsistent with alpha rate scaling");
}

void test_distribution_family_smoke_for_service_and_patience() {
    const std::array<wck::DistributionSpec, 3> families{
        make_exp(1.0),
        make_erlang(2, 2.0),
        make_h2(0.5, 3.0, 0.3333333333333333),
    };

    int case_index = 0;
    for (const auto& service : families) {
        for (const auto& patience : families) {
            wck::RunParameters p{};
            p.model_name = "family_smoke_" + std::to_string(case_index++);
            p.model_index = 0U;
            p.alpha_index = 0;
            p.alpha = 1.0;
            p.h = 0.5;
            p.c = 0.2;
            p.rho_exponent = 0.5;
            p.rho = 1.1;
            p.lambda = 1.1;
            p.arrival = make_exp(1.1);
            p.service = service;
            p.patience = patience;
            p.scaling.k = 1;
            p.scaling.beta_patience = 1.0;
            p.simulation = wck::SimulationConfig{};
            p.simulation.warmup_time = 0.0;
            p.simulation.sample_time = 500.0;
            p.simulation.tau = 0.05;
            p.simulation.max_level = 7;
            p.simulation.min_windows_per_t = 10;
            p.simulation.n_tau_shifts = 1;
            p.simulation.threads = 1;
            p.seed = static_cast<std::uint64_t>(9000 + case_index);

            const auto r = wck::simulate_effective_idw(p, nullptr);
            expect(!r.estimates.empty(), "family smoke: expected nonempty estimates");
            expect(r.stats.sample_service_count > 0U, "family smoke: expected sampled service observations");
        }
    }
}

std::vector<double> sample_compound_poisson_bins(
    std::size_t n_bins,
    double delta,
    double lambda,
    double mu,
    std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::poisson_distribution<int> pois(lambda * delta);
    std::exponential_distribution<double> exp(mu);

    std::vector<double> bins(n_bins, 0.0);
    for (std::size_t i = 0U; i < n_bins; ++i) {
        const int n = pois(rng);
        double total = 0.0;
        for (int j = 0; j < n; ++j) {
            total += exp(rng);
        }
        bins[i] = total;
    }
    return bins;
}

void test_estimator_deterministic_bins() {
    std::vector<double> bins(4096U, 0.75);

    wck::EstimatorConfig cfg{};
    cfg.tau_base = 0.1;
    cfg.tau_shift = 0.1;
    cfg.tau_shift_index = 0;
    cfg.max_level = 8;
    cfg.min_windows_per_t = 20;
    cfg.ev_hat = 1.0;

    const auto rows = wck::estimate_effective_idw_from_bins(bins, cfg, nullptr, nullptr);
    expect(!rows.empty(), "deterministic bins: expected at least one estimate row");
    for (const auto& row : rows) {
        expect(std::abs(row.idw_hat) < 1e-10, "deterministic bins: IDW should be ~0");
        expect(row.n_windows >= 20, "deterministic bins: min windows constraint violated");
    }
}

void test_compound_poisson_idw_near_two() {
    const auto bins = sample_compound_poisson_bins(50000U, 0.05, 1.2, 1.0, 42ULL);

    wck::EstimatorConfig cfg{};
    cfg.tau_base = 0.05;
    cfg.tau_shift = 0.05;
    cfg.tau_shift_index = 0;
    cfg.max_level = 8;
    cfg.min_windows_per_t = 40;
    cfg.ev_hat = 1.0;

    const auto rows = wck::estimate_effective_idw_from_bins(bins, cfg, nullptr, nullptr);
    expect(!rows.empty(), "compound poisson: expected nonempty rows");
    for (const auto& row : rows) {
        expect(std::abs(row.idw_hat - 2.0) < 0.3, "compound poisson: IDW not near 2");
        expect(std::abs(row.t - std::ldexp(row.tau_shift, row.level)) < 1e-12,
               "compound poisson: t should follow dyadic level");
    }
}

void test_estimator_progress_monotone() {
    const auto bins = sample_compound_poisson_bins(40000U, 0.02, 1.15, 1.0, 123456ULL);

    wck::EstimatorConfig cfg{};
    cfg.tau_base = 0.02;
    cfg.tau_shift = 0.02;
    cfg.tau_shift_index = 0;
    cfg.max_level = 10;
    cfg.min_windows_per_t = 20;
    cfg.ev_hat = 1.0;

    std::vector<double> progress_values;
    const auto rows = wck::estimate_effective_idw_from_bins(
        bins,
        cfg,
        nullptr,
        nullptr,
        [&](double p) { progress_values.push_back(p); });

    expect(!rows.empty(), "progress monotone: expected nonempty rows");
    expect(!progress_values.empty(), "progress monotone: no progress updates");

    double prev = -1.0;
    for (double p : progress_values) {
        expect(p + 1e-12 >= prev, "progress monotone: progress not monotone");
        prev = p;
    }
    expect(std::abs(progress_values.back() - 1.0) < 1e-12, "progress monotone: final progress must be 1");
}

void test_warmup_discard_effect() {
    wck::RunParameters p1{};
    p1.model_name = "warmup_test";
    p1.model_index = 0U;
    p1.alpha_index = 0;
    p1.alpha = 1.0;
    p1.h = 0.5;
    p1.c = 0.2;
    p1.rho_exponent = 0.5;
    p1.rho = 1.2;
    p1.lambda = 1.2;
    p1.arrival = make_exp(1.0);
    p1.service = make_exp(1.0);
    p1.patience = make_erlang(1, 1.0);
    p1.scaling.k = 1;
    p1.scaling.beta_patience = 1.0;
    p1.simulation = wck::SimulationConfig{};
    p1.simulation.warmup_time = 0.0;
    p1.simulation.sample_time = 1200.0;
    p1.simulation.tau = 0.05;
    p1.simulation.max_level = 8;
    p1.simulation.min_windows_per_t = 40;
    p1.simulation.n_tau_shifts = 1;
    p1.simulation.threads = 1;
    p1.seed = 987654321ULL;

    wck::RunParameters p2 = p1;
    p2.simulation.warmup_time = 200.0;

    const auto r1 = wck::simulate_effective_idw(p1, nullptr);
    const auto r2 = wck::simulate_effective_idw(p2, nullptr);

    expect(r1.estimates.size() == r2.estimates.size(), "warmup effect: expected same row count");

    double first_diff = 0.0;
    double last_diff = 0.0;
    bool first_set = false;
    for (std::size_t i = 0U; i < r1.estimates.size(); ++i) {
        const double diff = std::abs(r1.estimates[i].idw_hat - r2.estimates[i].idw_hat);
        if (!first_set) {
            first_diff = diff;
            first_set = true;
        }
        last_diff = diff;
    }

    expect(first_set, "warmup effect: no comparable estimate points");
    expect(first_diff > 0.01, "warmup effect: early-horizon difference too small");
    expect(last_diff < 0.4, "warmup effect: long-horizon estimate did not stabilize enough");
}

void test_tau_shift_parallel_execution() {
    wck::RunParameters p{};
    p.model_name = "tau_shift_parallel";
    p.model_index = 0U;
    p.alpha_index = 0;
    p.alpha = 1.0;
    p.h = 0.5;
    p.c = 0.2;
    p.rho_exponent = 0.5;
    p.rho = 1.1;
    p.lambda = 1.1;
    p.arrival = make_exp(1.0);
    p.service = make_exp(1.0);
    p.patience = make_erlang(1, 1.0);
    p.scaling.k = 1;
    p.scaling.beta_patience = 1.0;
    p.simulation = wck::SimulationConfig{};
    p.simulation.warmup_time = 0.0;
    p.simulation.sample_time = 800.0;
    p.simulation.tau = 0.02;
    p.simulation.max_level = 7;
    p.simulation.min_windows_per_t = 20;
    p.simulation.n_tau_shifts = 4;
    p.simulation.threads = 3;
    p.seed = 99991ULL;

    const auto r = wck::simulate_effective_idw(p, nullptr);
    expect(!r.estimates.empty(), "tau shift parallel: expected nonempty estimates");
    expect(r.estimator_threads_used == 3, "tau shift parallel: expected 3 estimator threads");

    std::vector<bool> seen_shift(4U, false);
    for (const auto& row : r.estimates) {
        expect(row.tau_shift_index >= 0 && row.tau_shift_index < 4,
               "tau shift parallel: tau_shift_index out of range");
        seen_shift[static_cast<std::size_t>(row.tau_shift_index)] = true;
        expect(std::abs(row.t - std::ldexp(row.tau_shift, row.level)) < 1e-12,
               "tau shift parallel: t != 2^level * tau_shift");
    }

    for (std::size_t i = 0U; i < seen_shift.size(); ++i) {
        expect(seen_shift[i], "tau shift parallel: missing rows for a tau shift");
    }
}

void test_cli_smoke() {
    const auto dir = make_temp_dir("wck_effective_idw_cli_smoke");
    const auto out_dir = dir / "out";
    std::filesystem::create_directories(out_dir);

    const auto config_path = write_text_file(
        dir,
        "smoke_config.json",
        R"JSON({
  "alpha": { "indices": [0], "base": 2.0 },
  "simulation": {
    "warmup_time": 5.0,
    "sample_time": 40.0,
    "tau": 0.02,
    "max_level": 5,
    "min_windows_per_t": 5,
    "n_tau_shifts": 2,
    "seed": 321,
    "save_event_trace": false,
    "threads": 1
  },
  "models": [
    {
      "name": "M/M/1+M",
      "arrival": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "system": { "c": 0.2 },
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "patience": { "distribution": { "family": "erlang_k", "params": { "k": 1, "rate": 1.0 } } },
      "scaling": { "k": 1, "beta_patience": 1.0 }
    },
    {
      "name": "E2/M/1+M",
      "arrival": { "distribution": { "family": "erlang_k", "params": { "k": 2, "rate": 2.0 } } },
      "system": { "c": 0.2 },
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "patience": { "distribution": { "family": "erlang_k", "params": { "k": 1, "rate": 1.0 } } },
      "scaling": { "k": 1, "beta_patience": 1.0 }
    },
    {
      "name": "H2(4)/M/1+M",
      "arrival": { "distribution": { "family": "hyperexponential2", "params": { "p": 0.5, "rate1": 3.0, "rate2": 0.3333333333333333 } } },
      "system": { "c": 0.2 },
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "patience": { "distribution": { "family": "erlang_k", "params": { "k": 1, "rate": 1.0 } } },
      "scaling": { "k": 1, "beta_patience": 1.0 }
    },
    {
      "name": "H2(4)/M/1+E2",
      "arrival": { "distribution": { "family": "hyperexponential2", "params": { "p": 0.5, "rate1": 3.0, "rate2": 0.3333333333333333 } } },
      "system": { "c": 0.2 },
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "patience": { "distribution": { "family": "erlang_k", "params": { "k": 2, "rate": 2.0 } } },
      "scaling": { "k": 2, "beta_patience": 2.0 }
    },
    {
      "name": "M/M/1+E2",
      "arrival": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "system": { "c": 0.2 },
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "patience": { "distribution": { "family": "erlang_k", "params": { "k": 2, "rate": 2.0 } } },
      "scaling": { "k": 2, "beta_patience": 2.0 }
    }
  ]
})JSON");

    const std::string cmd =
        std::string("\"") + WCK_IDW_SIM_CLI_PATH + "\""
        + " --config \"" + config_path.string() + "\""
        + " --out-dir \"" + out_dir.string() + "\"";

    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "cli smoke: simulator command failed");

    int curve_count = 0;
    int summary_count = 0;
    std::filesystem::path one_curve_path;
    std::filesystem::path one_summary_path;
    for (const auto& entry : std::filesystem::directory_iterator(out_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.find("_curve.csv") != std::string::npos) {
            ++curve_count;
            one_curve_path = entry.path();
        }
        if (name.find("_summary.json") != std::string::npos) {
            ++summary_count;
            one_summary_path = entry.path();
        }
    }

    expect(curve_count == 5, "cli smoke: expected 5 curve csv outputs");
    expect(summary_count == 5, "cli smoke: expected 5 summary json outputs");

    std::ifstream curve_in(one_curve_path);
    expect(curve_in.is_open(), "cli smoke: failed to open one curve CSV");
    std::string header;
    std::getline(curve_in, header);
    expect(header.find("idw_hat") != std::string::npos, "cli smoke: curve CSV missing idw_hat column");

    std::ifstream summary_in(one_summary_path);
    expect(summary_in.is_open(), "cli smoke: failed to open one summary JSON");
    std::ostringstream summary_contents;
    summary_contents << summary_in.rdbuf();
    const std::string summary_text = summary_contents.str();
    expect(summary_text.find("\"distributions\":") != std::string::npos,
           "cli smoke: summary missing distributions");
    expect(summary_text.find("\"arrival_scv\":") != std::string::npos,
           "cli smoke: summary missing arrival_scv");
    expect(summary_text.find("\"service_scv\":") != std::string::npos,
           "cli smoke: summary missing service_scv");
    expect(summary_text.find("\"service_mu_effective\":") != std::string::npos,
           "cli smoke: summary missing service_mu_effective");
}

void test_cli_threads_override() {
    const auto dir = make_temp_dir("wck_effective_idw_cli_threads_override");
    const auto out_dir = dir / "out";
    std::filesystem::create_directories(out_dir);

    const auto config_path = write_text_file(
        dir,
        "threads_override_config.json",
        R"JSON({
  "alpha": { "indices": [0], "base": 2.0 },
  "simulation": {
    "warmup_time": 5.0,
    "sample_time": 120.0,
    "tau": 0.02,
    "max_level": 8,
    "min_windows_per_t": 5,
    "n_tau_shifts": 4,
    "threads": 1,
    "seed": 1234,
    "save_event_trace": false
  },
  "models": [
    {
      "name": "M/M/1+M",
      "arrival": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "system": { "c": 0.2 },
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "patience": { "distribution": { "family": "erlang_k", "params": { "k": 1, "rate": 1.0 } } },
      "scaling": { "k": 1, "beta_patience": 1.0 }
    }
  ]
})JSON");

    const std::string cmd_ok =
        std::string("\"") + WCK_IDW_SIM_CLI_PATH + "\""
        + " --config \"" + config_path.string() + "\""
        + " --out-dir \"" + out_dir.string() + "\""
        + " --threads 3";
    const int rc_ok = std::system(cmd_ok.c_str());
    expect(rc_ok == 0, "cli threads override: simulator command failed");

    std::filesystem::path summary_path;
    for (const auto& entry : std::filesystem::directory_iterator(out_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.find("_summary.json") != std::string::npos) {
            summary_path = entry.path();
            break;
        }
    }
    expect(!summary_path.empty(), "cli threads override: summary output missing");

    std::ifstream in(summary_path);
    expect(in.is_open(), "cli threads override: failed to open summary");
    std::ostringstream contents;
    contents << in.rdbuf();
    const std::string text = contents.str();
    expect(text.find("\"threads\": 3") != std::string::npos,
           "cli threads override: expected simulation.threads override in summary");
    expect(text.find("\"estimator_threads_used\": ") != std::string::npos,
           "cli threads override: missing estimator_threads_used in summary");
    expect(text.find("\"estimation_wall_seconds\": ") != std::string::npos,
           "cli threads override: missing estimation_wall_seconds in summary");

    const std::string cmd_bad =
        std::string("\"") + WCK_IDW_SIM_CLI_PATH + "\""
        + " --config \"" + config_path.string() + "\""
        + " --out-dir \"" + out_dir.string() + "\""
        + " --threads -1";
    const int rc_bad = std::system(cmd_bad.c_str());
    expect(rc_bad != 0, "cli threads override: negative thread argument should fail");
}

}  // namespace

void run_effective_idw_sim_tests() {
    test_config_validation();
    test_tau_shift_grid();
    test_h2_roundtrip();
    test_e2_arrival_sample_scv();
    test_arrival_coupling_normalization();
    test_patience_alpha_rate_scaling();
    test_distribution_family_smoke_for_service_and_patience();
    test_estimator_deterministic_bins();
    test_compound_poisson_idw_near_two();
    test_estimator_progress_monotone();
    test_warmup_discard_effect();
    test_tau_shift_parallel_execution();
    test_cli_smoke();
    test_cli_threads_override();
}
