#include "wck/common/json_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace wck::json_config {

void fail(const std::string& path, const std::string& message) {
    throw std::invalid_argument(path + ": " + message);
}

const json::Value& require_key(
    const json::Value::Object& object,
    const std::string& key,
    const std::string& path) {
    const auto it = object.find(key);
    if (it == object.end()) {
        fail(path + "." + key, "missing required field");
    }
    return it->second;
}

const json::Value* optional_key(const json::Value::Object& object, const std::string& key) {
    const auto it = object.find(key);
    if (it == object.end()) {
        return nullptr;
    }
    return &it->second;
}

const json::Value::Object& require_object(const json::Value& value, const std::string& path) {
    if (!value.is_object()) {
        fail(path, "expected object, got " + json::type_name(value.type()));
    }
    return value.as_object();
}

const json::Value::Array& require_array(const json::Value& value, const std::string& path) {
    if (!value.is_array()) {
        fail(path, "expected array, got " + json::type_name(value.type()));
    }
    return value.as_array();
}

std::string require_string(const json::Value& value, const std::string& path) {
    if (!value.is_string()) {
        fail(path, "expected string, got " + json::type_name(value.type()));
    }
    return value.as_string();
}

bool require_bool(const json::Value& value, const std::string& path) {
    if (!value.is_bool()) {
        fail(path, "expected bool, got " + json::type_name(value.type()));
    }
    return value.as_bool();
}

double require_number(const json::Value& value, const std::string& path) {
    if (!value.is_number()) {
        fail(path, "expected number, got " + json::type_name(value.type()));
    }
    const double out = value.as_number();
    if (!std::isfinite(out)) {
        fail(path, "number must be finite");
    }
    return out;
}

int double_to_int(double value, const std::string& path) {
    if (!std::isfinite(value)) {
        fail(path, "expected finite integer-valued number");
    }
    const double rounded = std::round(value);
    if (std::abs(rounded - value) > 1e-9) {
        fail(path, "expected integer-valued number");
    }
    if (rounded < static_cast<double>(std::numeric_limits<int>::min())
        || rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        fail(path, "integer out of range");
    }
    return static_cast<int>(rounded);
}

std::uint64_t require_uint64(const json::Value& value, const std::string& path) {
    const double raw = require_number(value, path);
    if (raw < 0.0 || raw > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
        fail(path, "must be in [0, 2^64-1]");
    }
    return static_cast<std::uint64_t>(raw);
}

void reject_unknown_keys(
    const json::Value::Object& object,
    const std::vector<std::string>& allowed,
    const std::string& path) {
    for (const auto& [key, _] : object) {
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
            fail(path + "." + key, "unsupported field");
        }
    }
}

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

DistributionFamily parse_distribution_family(const std::string& family, const std::string& path) {
    const std::string v = lowercase(family);
    if (v == "exponential" || v == "exp") {
        return DistributionFamily::kExponential;
    }
    if (v == "erlang_k" || v == "erlang") {
        return DistributionFamily::kErlangK;
    }
    if (v == "lognormal" || v == "ln") {
        return DistributionFamily::kLognormal;
    }
    if (v == "hyperexponential2" || v == "h2") {
        return DistributionFamily::kHyperexponential2;
    }
    fail(path, "expected family in {exponential|exp, erlang_k|erlang, lognormal|ln, hyperexponential2|h2}");
}

DistributionSpec parse_distribution_spec(const json::Value::Object& wrapper, const std::string& path) {
    reject_unknown_keys(wrapper, {"distribution"}, path);

    const std::string dist_path = path + ".distribution";
    const auto& dist_obj = require_object(require_key(wrapper, "distribution", path), dist_path);
    reject_unknown_keys(dist_obj, {"family", "params"}, dist_path);

    const DistributionFamily family = parse_distribution_family(
        require_string(require_key(dist_obj, "family", dist_path), dist_path + ".family"),
        dist_path + ".family");
    const auto& params = require_object(require_key(dist_obj, "params", dist_path), dist_path + ".params");
    const std::string ppath = dist_path + ".params";

    DistributionSpec spec{};
    spec.family = family;
    switch (family) {
    case DistributionFamily::kExponential:
        reject_unknown_keys(params, {"rate"}, ppath);
        spec.exponential.rate = require_number(require_key(params, "rate", ppath), ppath + ".rate");
        break;
    case DistributionFamily::kErlangK:
        reject_unknown_keys(params, {"k", "rate"}, ppath);
        spec.erlang_k.k = double_to_int(
            require_number(require_key(params, "k", ppath), ppath + ".k"), ppath + ".k");
        spec.erlang_k.rate = require_number(require_key(params, "rate", ppath), ppath + ".rate");
        break;
    case DistributionFamily::kLognormal:
        reject_unknown_keys(params, {"mean", "scv"}, ppath);
        spec.lognormal.mean = require_number(require_key(params, "mean", ppath), ppath + ".mean");
        spec.lognormal.scv = require_number(require_key(params, "scv", ppath), ppath + ".scv");
        break;
    case DistributionFamily::kHyperexponential2:
        reject_unknown_keys(params, {"p", "rate1", "rate2"}, ppath);
        spec.hyperexponential2.p = require_number(require_key(params, "p", ppath), ppath + ".p");
        spec.hyperexponential2.rate1 = require_number(require_key(params, "rate1", ppath), ppath + ".rate1");
        spec.hyperexponential2.rate2 = require_number(require_key(params, "rate2", ppath), ppath + ".rate2");
        break;
    }

    validate_distribution_spec(spec, dist_path);
    return spec;
}

}  // namespace wck::json_config
