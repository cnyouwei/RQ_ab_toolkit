#include "wck/workload_sim/tandem_workload_sim.hpp"
#include "wck/workload_sim/tandem_workload_sim_config.hpp"

#include "support.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

wck::TandemWorkloadRunParams base_params() {
    wck::TandemWorkloadRunParams params{};
    params.model_name = "H2/E2 -> M/H2";
    params.lambda = 1.0;
    params.alpha = 1.0;
    params.queue1_traffic_intensity = 0.9;
    params.queue1_arrival = make_h2(0.8872983346207416, 1.7745966692414832, 0.22540333075851682);
    params.queue1_service = make_erlang(2, 2.0);
    params.queue2_service = make_exp(1.0);
    params.queue2_patience = make_h2(0.8872983346207416, 1.7745966692414832, 0.22540333075851682);
    params.warmup_time = 100.0;
    params.sample_time = 1000.0;
    params.replications = 64;
    params.threads = 4;
    params.seed = 777U;
    params.normalize_service_mean_to_one = true;
    return params;
}

void test_determinism_and_thread_invariance() {
    const wck::TandemWorkloadRunParams params = base_params();

    std::vector<double> rep1{};
    std::vector<double> rep2{};
    const wck::TandemWorkloadSummary s1 = wck::simulate_tandem_workload_mc(params, &rep1);
    const wck::TandemWorkloadSummary s2 = wck::simulate_tandem_workload_mc(params, &rep2);

    expect(std::abs(s1.mean_workload - s2.mean_workload) < 1e-15, "determinism: mean mismatch");
    expect(std::abs(s1.std_workload - s2.std_workload) < 1e-15, "determinism: std mismatch");
    expect(rep1.size() == rep2.size(), "determinism: replication size mismatch");
    for (std::size_t i = 0U; i < rep1.size(); ++i) {
        expect(std::abs(rep1[i] - rep2[i]) < 1e-15, "determinism: per-rep mismatch");
    }

    wck::TandemWorkloadRunParams single = params;
    single.threads = 1;
    std::vector<double> rep_single{};
    const wck::TandemWorkloadSummary s_single = wck::simulate_tandem_workload_mc(single, &rep_single);
    expect(std::abs(s1.mean_workload - s_single.mean_workload) < 1e-15, "thread invariance: mean mismatch");
    expect(std::abs(s1.std_workload - s_single.std_workload) < 1e-15, "thread invariance: std mismatch");
    expect(rep1.size() == rep_single.size(), "thread invariance: replication size mismatch");
    for (std::size_t i = 0U; i < rep1.size(); ++i) {
        expect(std::abs(rep1[i] - rep_single[i]) < 1e-15, "thread invariance: per-rep mismatch");
    }
}

void test_queue1_service_scaling_enforces_target_rho() {
    // If queue1 service scaling is tied to lambda/rho1, changing base-service mean while
    // keeping shape and lambda should produce identical runtime service process.
    wck::TandemWorkloadRunParams p_a = base_params();
    wck::TandemWorkloadRunParams p_b = p_a;

    p_a.queue1_service = make_exp(0.5);  // base mean 2
    p_b.queue1_service = make_exp(2.0);  // base mean 0.5
    p_a.threads = 1;
    p_b.threads = 1;

    std::vector<double> rep_a{};
    std::vector<double> rep_b{};
    const wck::TandemWorkloadSummary s_a = wck::simulate_tandem_workload_mc(p_a, &rep_a);
    const wck::TandemWorkloadSummary s_b = wck::simulate_tandem_workload_mc(p_b, &rep_b);

    expect(std::abs(s_a.mean_workload - s_b.mean_workload) < 1e-15, "rho1 scaling: mean mismatch");
    expect(std::abs(s_a.std_workload - s_b.std_workload) < 1e-15, "rho1 scaling: std mismatch");
    expect(rep_a.size() == rep_b.size(), "rho1 scaling: replication size mismatch");
    for (std::size_t i = 0U; i < rep_a.size(); ++i) {
        expect(std::abs(rep_a[i] - rep_b[i]) < 1e-15, "rho1 scaling: per-rep mismatch");
    }
}

void test_monotonic_sanity() {
    wck::TandemWorkloadRunParams params = base_params();
    params.replications = 96;
    params.sample_time = 1500.0;

    params.lambda = 0.6;
    params.alpha = 1.0;
    const wck::TandemWorkloadSummary low_lambda = wck::simulate_tandem_workload_mc(params);
    params.lambda = 1.3;
    const wck::TandemWorkloadSummary high_lambda = wck::simulate_tandem_workload_mc(params);
    expect(
        high_lambda.mean_workload > low_lambda.mean_workload,
        "monotonic lambda: expected higher lambda to increase queue2 mean workload");

    params.lambda = 1.1;
    params.alpha = 0.25;
    const wck::TandemWorkloadSummary low_alpha = wck::simulate_tandem_workload_mc(params);
    params.alpha = 4.0;
    const wck::TandemWorkloadSummary high_alpha = wck::simulate_tandem_workload_mc(params);
    expect(
        low_alpha.mean_workload > high_alpha.mean_workload,
        "monotonic alpha: expected lower alpha to increase queue2 mean workload");
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
    "name": "tandem_cli_model",
    "alias": "tandem_cli",
    "queue1": {
      "traffic_intensity": 0.9,
      "arrival": { "distribution": { "family": "erlang_k", "params": { "k": 2, "rate": 2.0 } } },
      "service": { "distribution": { "family": "hyperexponential2", "params": { "p": 0.6, "rate1": 2.0, "rate2": 0.5 } } }
    },
    "queue2": {
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "patience": { "distribution": { "family": "erlang_k", "params": { "k": 2, "rate": 2.0 } } }
    }
  }
})JSON";
}

std::string valid_single_station_config_json() {
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
    "name": "routing_single_model",
    "alias": "routing_single",
    "arrival": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
    "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
    "patience": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } }
  }
})JSON";
}

void test_cli_smoke_and_invalid_config() {
    const auto dir = make_temp_dir("wck_tandem_workload_cli");
    const auto config_path = write_text_file(dir, "config.json", valid_cli_config_json());
    const auto summary_path = dir / "summary.json";

    const std::string cmd_ok =
        std::string("\"") + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --config \"" + config_path.string() + "\""
        + " --lambda 1.1 --alpha 0.5"
        + " --summary-json \"" + summary_path.string() + "\""
        + " --threads 1 --seed 98765";
    const int rc_ok = std::system(cmd_ok.c_str());
    expect(rc_ok == 0, "tandem workload cli smoke: command failed");
    expect(std::filesystem::exists(summary_path), "tandem workload cli smoke: summary file missing");

    const std::string summary = read_text_file(summary_path);
    expect(summary.find("\"mean_workload\":") != std::string::npos, "tandem workload cli smoke: missing mean_workload");
    expect(summary.find("\"std_workload\":") != std::string::npos, "tandem workload cli smoke: missing std_workload");
    expect(summary.find("\"n_reps\":") != std::string::npos, "tandem workload cli smoke: missing n_reps");
    expect(summary.find("\"model_name\":") != std::string::npos, "tandem workload cli smoke: missing model_name");

    const auto invalid_config_path = write_text_file(
        dir,
        "invalid_config.json",
        R"JSON({
  "simulation": {
    "warmup_time": 30.0,
    "sample_time": 200.0,
    "replications": 24
  },
  "model": {
    "name": "invalid_tandem",
    "alias": "bad",
    "queue1": {
      "traffic_intensity": 1.2,
      "arrival": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } }
    },
    "queue2": {
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "patience": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } }
    }
  }
})JSON");

    const std::string cmd_bad =
        std::string("\"") + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --config \"" + invalid_config_path.string() + "\""
        + " --lambda 1.1 --alpha 0.5"
        + " --summary-json \"" + (dir / "bad_summary.json").string() + "\"";
    const int rc_bad = std::system(cmd_bad.c_str());
    expect(rc_bad != 0, "tandem workload cli invalid-config: expected failure");
}

// Routing tests for the merged workload_mc binary: it must dispatch on the
// presence of model.queue1.
void test_merged_binary_routing() {
    const auto dir = make_temp_dir("wck_workload_mc_routing");

    // Single-station config routes to the single-station simulation. A tandem
    // parse of this config would fail (model.arrival is not a tandem field),
    // so a successful run proves single-station routing.
    const auto single_config = write_text_file(dir, "single.json", valid_single_station_config_json());
    const auto single_summary = dir / "single_summary.json";
    const std::string cmd_single =
        std::string("\"") + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --config \"" + single_config.string() + "\""
        + " --lambda 1.1 --alpha 0.5"
        + " --summary-json \"" + single_summary.string() + "\""
        + " --threads 1 --seed 4242";
    expect(std::system(cmd_single.c_str()) == 0, "routing: single-station run failed");
    const std::string single_text = read_text_file(single_summary);
    expect(
        single_text.find("\"model_name\": \"routing_single_model\"") != std::string::npos,
        "routing: single summary should carry the single-station model name");

    // Tandem config routes to the tandem simulation. A single-station parse
    // of this config would fail (model.queue1 is not a single-station field),
    // so a successful run proves tandem routing.
    const auto tandem_config = write_text_file(dir, "tandem.json", valid_cli_config_json());
    const auto tandem_summary = dir / "tandem_summary.json";
    const std::string cmd_tandem =
        std::string("\"") + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --config \"" + tandem_config.string() + "\""
        + " --lambda 1.1 --alpha 0.5"
        + " --summary-json \"" + tandem_summary.string() + "\""
        + " --threads 1 --seed 4242";
    expect(std::system(cmd_tandem.c_str()) == 0, "routing: tandem run failed");
    const std::string tandem_text = read_text_file(tandem_summary);
    expect(
        tandem_text.find("\"model_name\": \"tandem_cli_model\"") != std::string::npos,
        "routing: tandem summary should carry the tandem model name");

    // A config with queue1 but no queue2 is routed to the tandem loader and
    // must be rejected.
    const auto missing_queue2 = write_text_file(
        dir,
        "missing_queue2.json",
        R"JSON({
  "simulation": {
    "warmup_time": 30.0,
    "sample_time": 200.0,
    "replications": 24
  },
  "model": {
    "name": "missing_queue2",
    "queue1": {
      "traffic_intensity": 0.9,
      "arrival": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } },
      "service": { "distribution": { "family": "exponential", "params": { "rate": 1.0 } } }
    }
  }
})JSON");
    const std::string cmd_missing =
        std::string("\"") + WCK_WORKLOAD_MC_CLI_PATH + "\""
        + " --config \"" + missing_queue2.string() + "\""
        + " --lambda 1.1 --alpha 0.5"
        + " --summary-json \"" + (dir / "missing_queue2_summary.json").string() + "\"";
    expect(
        std::system(cmd_missing.c_str()) != 0,
        "routing: config with queue1 but no queue2 should fail");
}

}  // namespace

void run_tandem_workload_sim_tests() {
    test_determinism_and_thread_invariance();
    test_queue1_service_scaling_enforces_target_rho();
    test_monotonic_sanity();
    test_cli_smoke_and_invalid_config();
    test_merged_binary_routing();
}
