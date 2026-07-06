#include "wck/idw_sim/effective_idw_sim_config.hpp"

#include "wck/common/distributions.hpp"
#include "wck/common/json_config.hpp"
#include "wck/common/mini_json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wck {

namespace {

using namespace json_config;

std::vector<int> parse_indices(const json::Value& value, const std::string& path) {
    std::vector<int> out;
    if (value.is_array()) {
        const auto& arr = value.as_array();
        out.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            const std::string item_path = path + "[" + std::to_string(i) + "]";
            const int idx = double_to_int(require_number(arr[i], item_path), item_path);
            if (idx < 0) {
                fail(item_path, "index must be nonnegative");
            }
            out.push_back(idx);
        }
    } else if (value.is_string()) {
        std::string token;
        std::stringstream ss(value.as_string());
        while (std::getline(ss, token, ',')) {
            token.erase(
                std::remove_if(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch) != 0; }),
                token.end());
            if (token.empty()) {
                continue;
            }
            std::size_t parsed = 0U;
            int idx = 0;
            try {
                idx = std::stoi(token, &parsed);
            } catch (...) {
                fail(path, "invalid integer token in string list: " + token);
            }
            if (parsed != token.size()) {
                fail(path, "invalid integer token in string list: " + token);
            }
            if (idx < 0) {
                fail(path, "index must be nonnegative");
            }
            out.push_back(idx);
        }
    } else {
        fail(path, "expected array or string");
    }

    if (out.empty()) {
        fail(path, "must contain at least one index");
    }
    return out;
}

void reject_legacy_distribution_fields(
    const json::Value::Object& obj,
    const std::string& path,
    const std::vector<std::string>& legacy) {
    for (const std::string& key : legacy) {
        if (optional_key(obj, key) != nullptr) {
            fail(
                path + "." + key,
                "legacy field removed; use " + path + ".distribution.{family,params} instead");
        }
    }
}

DistributionSpec parse_model_distribution(const json::Value::Object& parent, const std::string& field) {
    const std::string path = "model." + field;
    const auto& wrapper = require_object(require_key(parent, field, "model"), path);

    if (field == "arrival") {
        reject_legacy_distribution_fields(wrapper, path, {"type", "scv", "r", "mean_interarrival"});
    } else if (field == "service") {
        reject_legacy_distribution_fields(wrapper, path, {"type", "mu", "c_s2"});
    } else if (field == "patience") {
        reject_legacy_distribution_fields(wrapper, path, {"type", "k", "beta_patience", "scv", "r", "mean"});
    }

    return parse_distribution_spec(wrapper, path);
}

void reject_legacy_simulation_keys(const json::Value::Object& sim) {
    const std::vector<std::string> legacy_keys{
        "estimate_grid",
        "bin_width",
        "bin_width_large",
        "bin_width_switch_t",
        "overlap_stride",
        "estimators",
        "use_idc_overlap",
        "use_fft",
    };

    for (const std::string& key : legacy_keys) {
        if (optional_key(sim, key) != nullptr) {
            fail(
                "simulation." + key,
                "not supported in dyadic effective-IDW mode; use {tau, max_level, n_tau_shifts, min_windows_per_t}");
        }
    }
}

SimulationConfig parse_simulation(const json::Value::Object& root) {
    const json::Value& sim_value = require_key(root, "simulation", "config");
    const auto& sim = require_object(sim_value, "simulation");
    reject_legacy_simulation_keys(sim);
    reject_unknown_keys(
        sim,
        {"warmup_time", "sample_time", "tau", "max_level", "min_windows_per_t", "n_tau_shifts", "threads", "seed",
         "save_event_trace"},
        "simulation");

    SimulationConfig cfg{};
    cfg.warmup_time = require_number(require_key(sim, "warmup_time", "simulation"), "simulation.warmup_time");
    cfg.sample_time = require_number(require_key(sim, "sample_time", "simulation"), "simulation.sample_time");
    cfg.tau = require_number(require_key(sim, "tau", "simulation"), "simulation.tau");
    cfg.max_level = double_to_int(
        require_number(require_key(sim, "max_level", "simulation"), "simulation.max_level"),
        "simulation.max_level");
    cfg.min_windows_per_t = double_to_int(
        require_number(require_key(sim, "min_windows_per_t", "simulation"), "simulation.min_windows_per_t"),
        "simulation.min_windows_per_t");

    const json::Value* n_tau_shifts = optional_key(sim, "n_tau_shifts");
    if (n_tau_shifts != nullptr) {
        cfg.n_tau_shifts =
            double_to_int(require_number(*n_tau_shifts, "simulation.n_tau_shifts"), "simulation.n_tau_shifts");
    }

    const json::Value* threads = optional_key(sim, "threads");
    if (threads != nullptr) {
        cfg.threads = double_to_int(require_number(*threads, "simulation.threads"), "simulation.threads");
    }

    const json::Value* seed = optional_key(sim, "seed");
    if (seed != nullptr) {
        cfg.seed = require_uint64(*seed, "simulation.seed");
    }

    const json::Value* save_trace = optional_key(sim, "save_event_trace");
    if (save_trace != nullptr) {
        cfg.save_event_trace = require_bool(*save_trace, "simulation.save_event_trace");
    }

    if (cfg.warmup_time < 0.0) {
        fail("simulation.warmup_time", "must be >= 0");
    }
    if (!(cfg.sample_time > 0.0)) {
        fail("simulation.sample_time", "must be > 0");
    }
    if (!(cfg.tau > 0.0)) {
        fail("simulation.tau", "must be > 0");
    }
    if (cfg.max_level < 0) {
        fail("simulation.max_level", "must be >= 0");
    }
    if (cfg.max_level > 30) {
        fail("simulation.max_level", "must be <= 30");
    }
    if (cfg.min_windows_per_t < 2) {
        fail("simulation.min_windows_per_t", "must be >= 2");
    }
    if (cfg.n_tau_shifts < 1) {
        fail("simulation.n_tau_shifts", "must be >= 1");
    }
    if (cfg.threads < 0) {
        fail("simulation.threads", "must be >= 0");
    }

    return cfg;
}

AlphaConfig parse_alpha(const json::Value::Object& root) {
    AlphaConfig cfg{};
    const json::Value* alpha_value = optional_key(root, "alpha");
    if (alpha_value == nullptr) {
        fail("alpha", "missing required field");
    }
    const auto& alpha = require_object(*alpha_value, "alpha");
    reject_unknown_keys(alpha, {"indices", "base"}, "alpha");

    const json::Value* indices = optional_key(alpha, "indices");
    if (indices == nullptr) {
        fail("alpha.indices", "missing required field");
    }
    cfg.indices = parse_indices(*indices, "alpha.indices");

    const json::Value* base = optional_key(alpha, "base");
    if (base != nullptr) {
        cfg.base = require_number(*base, "alpha.base");
    }
    if (!(cfg.base > 1.0)) {
        fail("alpha.base", "must be > 1");
    }
    return cfg;
}

SystemConfig parse_system(const json::Value::Object& model) {
    const auto& system = require_object(require_key(model, "system", "model"), "model.system");
    reject_unknown_keys(system, {"c"}, "model.system");

    SystemConfig cfg{};
    cfg.c = require_number(require_key(system, "c", "model.system"), "model.system.c");
    return cfg;
}

ScalingConfig parse_scaling(const json::Value::Object& model, const DistributionSpec& patience) {
    if (optional_key(model, "rho_exponent") != nullptr) {
        fail(
            "model.rho_exponent",
            "legacy field removed; use model.scaling.rho_exponent instead");
    }

    const auto& scaling = require_object(require_key(model, "scaling", "model"), "model.scaling");
    reject_unknown_keys(scaling, {"k", "beta_patience", "rho_exponent"}, "model.scaling");

    ScalingConfig cfg{};
    cfg.k = double_to_int(
        require_number(require_key(scaling, "k", "model.scaling"), "model.scaling.k"),
        "model.scaling.k");

    const json::Value* rho_exponent = optional_key(scaling, "rho_exponent");
    if (rho_exponent != nullptr) {
        cfg.has_rho_exponent = true;
        cfg.rho_exponent = require_number(*rho_exponent, "model.scaling.rho_exponent");
    }

    if (cfg.k < 1) {
        fail("model.scaling.k", "must be >= 1");
    }
    const double beta_from_distribution = distribution_beta_at_zero(patience, cfg.k);
    if (!(beta_from_distribution > 0.0)) {
        std::ostringstream oss;
        oss << "inconsistent with patience distribution: computed F^(k)(0)/k! = "
            << beta_from_distribution << " (must be > 0)";
        fail("model.scaling.k", oss.str());
    }

    const json::Value* beta_patience = optional_key(scaling, "beta_patience");
    if (beta_patience != nullptr) {
        const double supplied = require_number(*beta_patience, "model.scaling.beta_patience");
        if (!(supplied > 0.0)) {
            fail("model.scaling.beta_patience", "must be > 0");
        }
        const double tol = 1e-9 * std::max(1.0, std::abs(beta_from_distribution));
        if (std::abs(supplied - beta_from_distribution) > tol) {
            std::ostringstream oss;
            oss << "inconsistent with patience distribution and scaling.k; expected "
                << beta_from_distribution << ", got " << supplied;
            fail("model.scaling.beta_patience", oss.str());
        }
    }
    cfg.beta_patience = beta_from_distribution;
    return cfg;
}

SimulationModelConfig parse_model(const json::Value& value, std::size_t idx) {
    const auto& obj = require_object(value, "models[" + std::to_string(idx) + "]");
    SimulationModelConfig cfg{};

    const json::Value* name = optional_key(obj, "name");
    if (name != nullptr) {
        cfg.name = require_string(*name, "models[" + std::to_string(idx) + "].name");
    } else {
        cfg.name = "model_" + std::to_string(idx);
    }

    cfg.arrival = parse_model_distribution(obj, "arrival");
    cfg.service = parse_model_distribution(obj, "service");
    cfg.patience = parse_model_distribution(obj, "patience");
    cfg.system = parse_system(obj);
    cfg.scaling = parse_scaling(obj, cfg.patience);

    return cfg;
}

std::vector<SimulationModelConfig> parse_models(const json::Value::Object& root) {
    const auto& models_array = require_array(require_key(root, "models", "config"), "models");
    if (models_array.empty()) {
        fail("models", "must contain at least one model");
    }

    std::vector<SimulationModelConfig> out;
    out.reserve(models_array.size());
    for (std::size_t i = 0; i < models_array.size(); ++i) {
        out.push_back(parse_model(models_array[i], i));
    }
    return out;
}

}  // namespace

EffectiveIdwSimConfig load_effective_idw_sim_config(const std::filesystem::path& path) {
    const json::Value root_value = json::parse_file(path);
    const auto& root = require_object(root_value, "config");

    EffectiveIdwSimConfig config{};
    config.alpha = parse_alpha(root);
    config.simulation = parse_simulation(root);
    config.models = parse_models(root);
    return config;
}

std::vector<double> build_tau_shift_grid(double tau, int n_tau_shifts) {
    if (!(tau > 0.0)) {
        throw std::invalid_argument("tau must be > 0");
    }
    if (n_tau_shifts < 1) {
        throw std::invalid_argument("n_tau_shifts must be >= 1");
    }

    std::vector<double> out(static_cast<std::size_t>(n_tau_shifts), tau);
    if (n_tau_shifts == 1) {
        return out;
    }

    const double den = static_cast<double>(n_tau_shifts - 1);
    for (int j = 0; j < n_tau_shifts; ++j) {
        const double frac = static_cast<double>(j) / den;
        out[static_cast<std::size_t>(j)] = tau * std::pow(2.0, frac);
    }
    return out;
}

double alpha_from_index(int index, double base) {
    if (index < 0) {
        throw std::invalid_argument("alpha index must be nonnegative");
    }
    if (!(base > 1.0)) {
        throw std::invalid_argument("alpha base must be > 1");
    }
    return std::pow(base, -static_cast<double>(index));
}

}  // namespace wck
