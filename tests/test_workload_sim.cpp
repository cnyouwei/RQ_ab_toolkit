#include "wck/workload_sim/workload_sim.hpp"
#include "wck/workload_sim/workload_sim_config.hpp"

#include "support.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

wck::WorkloadRunParams base_params() {
    wck::WorkloadRunParams params{};
    params.model_name = "M/M/1+M";
    params.lambda = 1.0;
    params.alpha = 1.0;
    params.arrival = make_exp(1.0);
    params.service = make_exp(1.0);
    params.patience = make_exp(1.0);
    params.warmup_time = 100.0;
    params.sample_time = 1000.0;
    params.replications = 96;
    params.threads = 4;
    params.seed = 777U;
    params.normalize_service_mean_to_one = true;
    return params;
}

void test_determinism_and_thread_invariance() {
    const wck::WorkloadRunParams params = base_params();

    std::vector<double> rep1{};
    std::vector<double> rep2{};
    const wck::WorkloadSummary s1 = wck::simulate_workload_mc(params, &rep1);
    const wck::WorkloadSummary s2 = wck::simulate_workload_mc(params, &rep2);

    expect(std::abs(s1.mean_workload - s2.mean_workload) < 1e-15, "determinism: mean mismatch");
    expect(std::abs(s1.std_workload - s2.std_workload) < 1e-15, "determinism: std mismatch");
    expect(rep1.size() == rep2.size(), "determinism: replication size mismatch");
    for (std::size_t i = 0U; i < rep1.size(); ++i) {
        expect(std::abs(rep1[i] - rep2[i]) < 1e-15, "determinism: per-rep mismatch");
    }

    wck::WorkloadRunParams single = params;
    single.threads = 1;
    std::vector<double> rep_single{};
    const wck::WorkloadSummary s_single = wck::simulate_workload_mc(single, &rep_single);
    expect(std::abs(s1.mean_workload - s_single.mean_workload) < 1e-15, "thread invariance: mean mismatch");
    expect(std::abs(s1.std_workload - s_single.std_workload) < 1e-15, "thread invariance: std mismatch");
    expect(rep1.size() == rep_single.size(), "thread invariance: replication size mismatch");
    for (std::size_t i = 0U; i < rep1.size(); ++i) {
        expect(std::abs(rep1[i] - rep_single[i]) < 1e-15, "thread invariance: per-rep mismatch");
    }
}

void test_monotonic_sanity() {
    wck::WorkloadRunParams params = base_params();
    params.replications = 128;
    params.sample_time = 1500.0;

    params.lambda = 0.6;
    params.alpha = 1.0;
    const wck::WorkloadSummary low_lambda = wck::simulate_workload_mc(params);
    params.lambda = 1.4;
    const wck::WorkloadSummary high_lambda = wck::simulate_workload_mc(params);
    expect(
        high_lambda.mean_workload > low_lambda.mean_workload,
        "monotonic lambda: expected higher lambda to increase mean workload");

    params.lambda = 1.1;
    params.alpha = 0.25;
    const wck::WorkloadSummary low_alpha = wck::simulate_workload_mc(params);
    params.alpha = 4.0;
    const wck::WorkloadSummary high_alpha = wck::simulate_workload_mc(params);
    expect(
        low_alpha.mean_workload > high_alpha.mean_workload,
        "monotonic alpha: expected lower alpha to increase mean workload");
}

void test_distribution_family_coverage() {
    wck::WorkloadRunParams exp_params = base_params();
    exp_params.replications = 16;
    exp_params.sample_time = 200.0;
    exp_params.warmup_time = 30.0;
    exp_params.arrival = make_exp(1.0);
    exp_params.service = make_exp(0.8);
    exp_params.patience = make_exp(1.3);
    const wck::WorkloadSummary exp_summary = wck::simulate_workload_mc(exp_params);
    expect(std::isfinite(exp_summary.mean_workload), "distribution coverage: exp mean non-finite");
    expect(exp_summary.std_workload >= 0.0, "distribution coverage: exp std < 0");

    wck::WorkloadRunParams erlang_params = exp_params;
    erlang_params.model_name = "E2/E2/1+E2";
    erlang_params.arrival = make_erlang(2, 2.0);
    erlang_params.service = make_erlang(3, 2.1);
    erlang_params.patience = make_erlang(2, 1.6);
    const wck::WorkloadSummary erlang_summary = wck::simulate_workload_mc(erlang_params);
    expect(std::isfinite(erlang_summary.mean_workload), "distribution coverage: erlang mean non-finite");
    expect(erlang_summary.std_workload >= 0.0, "distribution coverage: erlang std < 0");

    wck::WorkloadRunParams h2_params = exp_params;
    h2_params.model_name = "H2/H2/1+H2";
    h2_params.arrival = make_h2(0.5, 3.0, 1.0 / 3.0);
    h2_params.service = make_h2(0.7, 2.8, 0.5);
    h2_params.patience = make_h2(0.6, 2.0, 0.6);
    const wck::WorkloadSummary h2_summary = wck::simulate_workload_mc(h2_params);
    expect(std::isfinite(h2_summary.mean_workload), "distribution coverage: h2 mean non-finite");
    expect(h2_summary.std_workload >= 0.0, "distribution coverage: h2 std < 0");

    wck::WorkloadRunParams ln_params = exp_params;
    ln_params.model_name = "M/LN/1+H2";
    ln_params.service = make_lognormal(1.0, 4.0);
    ln_params.patience = make_h2(0.6, 2.0, 0.6);
    const wck::WorkloadSummary ln_summary = wck::simulate_workload_mc(ln_params);
    expect(std::isfinite(ln_summary.mean_workload), "distribution coverage: lognormal mean non-finite");
    expect(ln_summary.std_workload >= 0.0, "distribution coverage: lognormal std < 0");
}

std::string valid_cli_config_json() {
    return R"JSON({
  "simulation": {
    "warmup_time": 30.0,
    "sample_time": 200.0,
    "replications": 24,
    "threads": 2,
    "seed": 123,
    "normalize_service_mean_to_one": true
  },
  "model": {
    "name": "CLI workload model",
    "alias": "mm1m",
    "arrival": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
    "service": { "distribution": { "family": "erlang_k", "params": { "k": 2, "rate": 1.6 } } },
    "patience": { "distribution": { "family": "hyperexponential2", "params": { "p": 0.6, "rate1": 2.0, "rate2": 0.5 } } }
  }
})JSON";
}

std::string valid_cli_lognormal_config_json() {
    return R"JSON({
  "simulation": {
    "warmup_time": 30.0,
    "sample_time": 200.0,
    "replications": 24,
    "threads": 2,
    "seed": 123,
    "normalize_service_mean_to_one": true
  },
  "model": {
    "name": "CLI workload model LN",
    "alias": "mln1_41h2_4",
    "arrival": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
    "service": { "distribution": { "family": "lognormal", "params": { "mean": 1.0, "scv": 4.0 } } },
    "patience": { "distribution": { "family": "hyperexponential2", "params": { "p": 0.6, "rate1": 2.0, "rate2": 0.5 } } }
  }
})JSON";
}

void test_cli_smoke_and_invalid_config() {
    const auto dir = make_temp_dir("wck_workload_cli");
    const auto config_path = write_text_file(dir, "config.json", valid_cli_config_json());
    const auto summary_path = dir / "summary.json";

    const std::string cmd_ok =
        std::string("\"") + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --config \"" + config_path.string() + "\""
        + " --lambda 1.1 --alpha 0.5"
        + " --summary-json \"" + summary_path.string() + "\""
        + " --threads 1 --seed 98765";
    const int rc_ok = std::system(cmd_ok.c_str());
    expect(rc_ok == 0, "workload cli smoke: command failed");
    expect(std::filesystem::exists(summary_path), "workload cli smoke: summary file missing");

    const std::string summary = read_text_file(summary_path);
    expect(summary.find("\"mean_workload\":") != std::string::npos, "workload cli smoke: missing mean_workload");
    expect(summary.find("\"std_workload\":") != std::string::npos, "workload cli smoke: missing std_workload");
    expect(summary.find("\"n_reps\":") != std::string::npos, "workload cli smoke: missing n_reps");
    expect(summary.find("\"model_name\":") != std::string::npos, "workload cli smoke: missing model_name");

    const auto config_ln_path = write_text_file(dir, "config_lognormal.json", valid_cli_lognormal_config_json());
    const auto summary_ln_path = dir / "summary_lognormal.json";
    const std::string cmd_ln =
        std::string("\"") + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --config \"" + config_ln_path.string() + "\""
        + " --lambda 1.1 --alpha 0.5"
        + " --summary-json \"" + summary_ln_path.string() + "\""
        + " --threads 1 --seed 98766";
    const int rc_ln = std::system(cmd_ln.c_str());
    expect(rc_ln == 0, "workload cli lognormal smoke: command failed");
    expect(std::filesystem::exists(summary_ln_path), "workload cli lognormal smoke: summary file missing");

    const auto invalid_config_path = write_text_file(
        dir,
        "invalid_config.json",
        R"JSON({
  "simulation": {
    "warmup_time": 30.0,
    "sample_time": 200.0,
    "replications": 0
  },
  "model": {
    "name": "invalid",
    "arrival": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
    "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
    "patience": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } }
  }
})JSON");

    const std::string cmd_bad =
        std::string("\"") + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --config \"" + invalid_config_path.string() + "\""
        + " --lambda 1.1 --alpha 0.5"
        + " --summary-json \"" + (dir / "bad_summary.json").string() + "\"";
    const int rc_bad = std::system(cmd_bad.c_str());
    expect(rc_bad != 0, "workload cli invalid-config: expected failure");
}

bool has_python3() {
    const int rc = std::system("python3 --version > /dev/null 2>&1");
    return rc == 0;
}

std::size_t count_csv_data_rows(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open csv: " + path.string());
    }
    std::string header;
    std::getline(in, header);
    expect(!header.empty(), "wrapper smoke: csv header missing");

    std::size_t rows = 0U;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            ++rows;
        }
    }
    return rows;
}

void test_wrapper_smoke() {
    if (!has_python3()) {
        return;
    }

    const auto dir = make_temp_dir("wck_workload_wrapper");
    const auto model_path = write_text_file(dir, "model.json", valid_cli_config_json());
    const auto grid_path = write_text_file(
        dir,
        "grid.json",
        R"JSON({
  "tuples": [
    { "tuple_id": 1, "lambda": 0.8, "alpha": 4.0, "lambda_k": 1, "lambda_form": "1-2^-k", "alpha_k": -2 },
    { "tuple_id": 2, "lambda": 1.0, "alpha": 1.0, "lambda_k": 0, "lambda_form": "mid", "alpha_k": 0 },
    { "tuple_id": 3, "lambda": 1.2, "alpha": 0.25, "lambda_k": 3, "lambda_form": "1+2^-k", "alpha_k": 2 }
  ]
})JSON");

    const std::filesystem::path out_csv = dir / "aggregate.csv";
    const std::filesystem::path summary_dir = dir / "summaries";

    const std::string script_path = std::string(WCK_SOURCE_DIR) + "/scripts/run_grid.py";
    const std::string cmd =
        std::string("python3 \"") + script_path + "\""
        + " --method workload"
        + " --grid \"" + grid_path.string() + "\""
        + " --model-config \"" + model_path.string() + "\""
        + " --binary \"" + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --out-csv \"" + out_csv.string() + "\""
        + " --summary-dir \"" + summary_dir.string() + "\""
        + " --threads 1 --seed 1234";

    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "wrapper smoke: command failed");
    expect(std::filesystem::exists(out_csv), "wrapper smoke: aggregate csv missing");
    expect(count_csv_data_rows(out_csv) == 3U, "wrapper smoke: unexpected aggregate row count");

    std::size_t summary_count = 0U;
    for (const auto& entry : std::filesystem::directory_iterator(summary_dir)) {
        if (entry.is_regular_file()) {
            ++summary_count;
        }
    }
    expect(summary_count == 3U, "wrapper smoke: unexpected summary file count");
}

void test_wrapper_no_summary_dir_uses_temp_summaries() {
    if (!has_python3()) {
        return;
    }

    const auto dir = make_temp_dir("wck_workload_wrapper_default");
    const auto model_path = write_text_file(dir, "workload_mm1m.json", valid_cli_config_json());
    const auto grid_path = write_text_file(
        dir,
        "grid.json",
        R"JSON({
  "tuples": [
    { "tuple_id": 1, "lambda": 0.8, "alpha": 4.0, "lambda_k": 1, "lambda_form": "1-2^-k", "alpha_k": -2 },
    { "tuple_id": 2, "lambda": 1.0, "alpha": 1.0, "lambda_k": 0, "lambda_form": "mid", "alpha_k": 0 }
  ]
})JSON");

    const std::filesystem::path out_csv = dir / "aggregate_default.csv";
    const std::string script_path = std::string(WCK_SOURCE_DIR) + "/scripts/run_grid.py";
    const std::string cmd =
        std::string("cd \"") + dir.string() + "\" && python3 \"" + script_path + "\""
        + " --method workload"
        + " --grid \"" + grid_path.string() + "\""
        + " --model-config \"" + model_path.string() + "\""
        + " --binary \"" + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --out-csv \"" + out_csv.string() + "\""
        + " --threads 1 --seed 1234";

    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "wrapper temp summaries: command failed");
    expect(std::filesystem::exists(out_csv), "wrapper temp summaries: aggregate csv missing");
    expect(count_csv_data_rows(out_csv) == 2U, "wrapper temp summaries: unexpected aggregate row count");

    // Without --summary-dir, per-tuple summaries live in a temp dir and are
    // cleaned up; nothing must appear next to the outputs.
    const std::filesystem::path default_summary_dir = dir / "results" / "workload_summaries";
    expect(
        !std::filesystem::exists(default_summary_dir),
        "wrapper temp summaries: no summary directory should be created");
}

}  // namespace

void run_workload_sim_tests() {
    test_determinism_and_thread_invariance();
    test_monotonic_sanity();
    test_distribution_family_coverage();
    test_cli_smoke_and_invalid_config();
    test_wrapper_smoke();
    test_wrapper_no_summary_dir_uses_temp_summaries();
}
