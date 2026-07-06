#include "wck/workload_sim/workload_sim_config.hpp"

#include "wck/common/json_config.hpp"
#include "wck/common/mini_json.hpp"

#include <string>

namespace wck {

namespace {

using namespace json_config;

WorkloadModelConfig parse_model(const json::Value::Object& root) {
    const auto& model = require_object(require_key(root, "model", "config"), "model");
    reject_unknown_keys(model, {"name", "alias", "arrival", "service", "patience"}, "model");

    auto parse_field = [&](const std::string& field_name) {
        const std::string path = "model." + field_name;
        return parse_distribution_spec(require_object(require_key(model, field_name, "model"), path), path);
    };

    WorkloadModelConfig out{};
    const json::Value* name = optional_key(model, "name");
    out.name = (name == nullptr) ? "model" : require_string(*name, "model.name");
    const json::Value* alias = optional_key(model, "alias");
    out.alias = (alias == nullptr) ? "" : require_string(*alias, "model.alias");
    out.arrival = parse_field("arrival");
    out.service = parse_field("service");
    out.patience = parse_field("patience");
    return out;
}

}  // namespace

WorkloadSimulationConfig parse_workload_simulation_config(const json::Value::Object& root) {
    const auto& sim = require_object(require_key(root, "simulation", "config"), "simulation");
    reject_unknown_keys(
        sim,
        {"warmup_time", "sample_time", "replications", "threads", "seed", "normalize_service_mean_to_one"},
        "simulation");

    WorkloadSimulationConfig out{};
    out.warmup_time = require_number(require_key(sim, "warmup_time", "simulation"), "simulation.warmup_time");
    out.sample_time = require_number(require_key(sim, "sample_time", "simulation"), "simulation.sample_time");
    out.replications = double_to_int(
        require_number(require_key(sim, "replications", "simulation"), "simulation.replications"),
        "simulation.replications");

    const json::Value* threads = optional_key(sim, "threads");
    if (threads != nullptr) {
        out.threads = double_to_int(require_number(*threads, "simulation.threads"), "simulation.threads");
    }

    const json::Value* seed = optional_key(sim, "seed");
    if (seed != nullptr) {
        out.seed = require_uint64(*seed, "simulation.seed");
    }

    const json::Value* normalize = optional_key(sim, "normalize_service_mean_to_one");
    if (normalize != nullptr) {
        out.normalize_service_mean_to_one =
            require_bool(*normalize, "simulation.normalize_service_mean_to_one");
    }

    if (out.warmup_time < 0.0) {
        fail("simulation.warmup_time", "must be >= 0");
    }
    if (!(out.sample_time > 0.0)) {
        fail("simulation.sample_time", "must be > 0");
    }
    if (out.replications < 1) {
        fail("simulation.replications", "must be >= 1");
    }
    if (out.threads < 0) {
        fail("simulation.threads", "must be >= 0");
    }

    return out;
}

WorkloadConfig load_workload_config(const std::filesystem::path& path) {
    const json::Value root_value = json::parse_file(path);
    const auto& root = require_object(root_value, "config");
    reject_unknown_keys(root, {"simulation", "model"}, "config");

    WorkloadConfig out{};
    out.simulation = parse_workload_simulation_config(root);
    out.model = parse_model(root);
    return out;
}

}  // namespace wck
