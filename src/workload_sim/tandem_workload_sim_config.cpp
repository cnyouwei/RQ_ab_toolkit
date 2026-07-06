#include "wck/workload_sim/tandem_workload_sim_config.hpp"

#include "wck/common/json_config.hpp"
#include "wck/common/mini_json.hpp"

#include <string>

namespace wck {

namespace {

using namespace json_config;

DistributionSpec parse_queue_distribution(
    const json::Value::Object& queue,
    const std::string& queue_path,
    const std::string& field_name) {
    const std::string path = queue_path + "." + field_name;
    return parse_distribution_spec(require_object(require_key(queue, field_name, queue_path), path), path);
}

TandemWorkloadModelConfig parse_model(const json::Value::Object& root) {
    const auto& model = require_object(require_key(root, "model", "config"), "model");
    reject_unknown_keys(model, {"name", "alias", "queue1", "queue2"}, "model");

    TandemWorkloadModelConfig out{};
    const json::Value* name = optional_key(model, "name");
    out.name = (name == nullptr) ? "model" : require_string(*name, "model.name");
    const json::Value* alias = optional_key(model, "alias");
    out.alias = (alias == nullptr) ? "" : require_string(*alias, "model.alias");

    const auto& queue1 = require_object(require_key(model, "queue1", "model"), "model.queue1");
    reject_unknown_keys(queue1, {"traffic_intensity", "arrival", "service"}, "model.queue1");

    out.queue1.traffic_intensity = require_number(
        require_key(queue1, "traffic_intensity", "model.queue1"),
        "model.queue1.traffic_intensity");
    if (!(out.queue1.traffic_intensity > 0.0 && out.queue1.traffic_intensity < 1.0)) {
        fail("model.queue1.traffic_intensity", "must be in (0,1)");
    }
    out.queue1.arrival = parse_queue_distribution(queue1, "model.queue1", "arrival");
    out.queue1.service = parse_queue_distribution(queue1, "model.queue1", "service");

    const auto& queue2 = require_object(require_key(model, "queue2", "model"), "model.queue2");
    reject_unknown_keys(queue2, {"service", "patience"}, "model.queue2");

    out.queue2.service = parse_queue_distribution(queue2, "model.queue2", "service");
    out.queue2.patience = parse_queue_distribution(queue2, "model.queue2", "patience");

    return out;
}

}  // namespace

TandemWorkloadConfig load_tandem_workload_config(const std::filesystem::path& path) {
    const json::Value root_value = json::parse_file(path);
    const auto& root = require_object(root_value, "config");
    reject_unknown_keys(root, {"simulation", "model"}, "config");

    TandemWorkloadConfig out{};
    out.simulation = parse_workload_simulation_config(root);
    out.model = parse_model(root);
    return out;
}

}  // namespace wck
