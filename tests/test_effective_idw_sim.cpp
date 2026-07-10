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
      "curve_label_template": "{model}",
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
    expect(cfg.simulation.threads == 1, "config validation: threads parse mismatch");
    expect(std::abs(cfg.simulation.tau - 0.02) < 1e-12, "config validation: tau parse mismatch");
    expect(
        std::abs(cfg.models[0].scaling.beta_patience - 1.0) < 1e-12,
        "config validation: expected derived beta_patience for M patience");

    const auto invalid_sample = replace_once(
        valid_config_json(),
        "\"sample_time\": 200.0",
        "\"sample_time\": 0.0");
    const auto invalid_sample_path = write_text_file(dir, "invalid_sample.json", invalid_sample);
    expect_throw_contains(
        [&]() { (void)wck::load_effective_idw_sim_config(invalid_sample_path); },
        "simulation.sample_time",
        "config validation: range error should include the field path");

    const auto unknown_model_key = replace_once(
        valid_config_json(),
        "\"scaling\": { \"k\": 1, \"beta_patience\": 1.0 }",
        "\"rho_exponent\": 0.5,\n      \"scaling\": { \"k\": 1, \"beta_patience\": 1.0 }");
    const auto unknown_model_key_path = write_text_file(dir, "unknown_model_key.json", unknown_model_key);
    expect_throw_contains(
        [&]() { (void)wck::load_effective_idw_sim_config(unknown_model_key_path); },
        "model.rho_exponent",
        "config validation: unknown model key should include the field path");

    const std::filesystem::path config_dir = std::filesystem::path(WCK_SOURCE_DIR) / "configs";
    std::size_t loaded_configs = 0U;
    for (const auto& entry : std::filesystem::directory_iterator(config_dir)) {
        const std::string filename = entry.path().filename().string();
        if (!entry.is_regular_file() || filename.rfind("effective_idw_", 0U) != 0U
            || entry.path().extension() != ".json") {
            continue;
        }
        const auto shipped = wck::load_effective_idw_sim_config(entry.path());
        expect(!shipped.models.empty(), "config validation: shipped config has no models");
        ++loaded_configs;
    }
    expect(loaded_configs > 0U, "config validation: no shipped effective-IDW configs found");
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

}  // namespace

void run_effective_idw_sim_tests() {
    test_config_validation();
    test_tau_shift_grid();
    test_e2_arrival_sample_scv();
    test_arrival_coupling_normalization();
    test_patience_alpha_rate_scaling();
    test_distribution_family_smoke_for_service_and_patience();
    test_estimator_deterministic_bins();
    test_compound_poisson_idw_near_two();
    test_tau_shift_parallel_execution();
    test_cli_smoke();
}
