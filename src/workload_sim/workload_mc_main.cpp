#include "wck/common/mini_json.hpp"
#include "wck/workload_sim/tandem_workload_sim.hpp"
#include "wck/workload_sim/tandem_workload_sim_config.hpp"
#include "wck/workload_sim/workload_sim.hpp"
#include "wck/workload_sim/workload_sim_config.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace {

struct CliOptions {
    std::filesystem::path config_path{};
    std::filesystem::path summary_path{};
    bool has_lambda = false;
    double lambda = 0.0;
    bool has_alpha = false;
    double alpha = 0.0;
    bool has_threads_override = false;
    int threads_override = 0;
    bool has_seed_override = false;
    std::uint64_t seed_override = 0U;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument(message);
}

bool starts_with_dash_dash(const std::string& text) {
    return text.rfind("--", 0U) == 0U;
}

double parse_double_arg(const std::string& raw, const std::string& name) {
    std::size_t parsed = 0U;
    double value = 0.0;
    try {
        value = std::stod(raw, &parsed);
    } catch (...) {
        fail("invalid value for " + name + ": " + raw);
    }
    if (parsed != raw.size()) {
        fail("invalid value for " + name + ": " + raw);
    }
    if (!std::isfinite(value)) {
        fail(name + " must be finite");
    }
    return value;
}

std::uint64_t parse_uint64_arg(const std::string& raw, const std::string& name) {
    std::size_t parsed = 0U;
    unsigned long long value = 0ULL;
    try {
        value = std::stoull(raw, &parsed);
    } catch (...) {
        fail("invalid value for " + name + ": " + raw);
    }
    if (parsed != raw.size()) {
        fail("invalid value for " + name + ": " + raw);
    }
    return static_cast<std::uint64_t>(value);
}

int parse_nonnegative_int_arg(const std::string& raw, const std::string& name) {
    std::size_t parsed = 0U;
    long long value = 0LL;
    try {
        value = std::stoll(raw, &parsed);
    } catch (...) {
        fail("invalid value for " + name + ": " + raw);
    }
    if (parsed != raw.size()) {
        fail("invalid value for " + name + ": " + raw);
    }
    if (value < 0LL) {
        fail(name + " must be >= 0");
    }
    if (value > static_cast<long long>(std::numeric_limits<int>::max())) {
        fail(name + " is too large");
    }
    return static_cast<int>(value);
}

void print_usage() {
    std::cout
        << "Usage: workload_mc --config <path> --lambda <double> --alpha <double> "
           "--summary-json <path> [options]\n"
        << "The config is auto-detected: model.queue1 present => tandem simulation,\n"
        << "otherwise single-station simulation.\n"
        << "Options:\n"
        << "  --threads <int>      override config simulation.threads (0=auto)\n"
        << "  --seed <uint64>      override config simulation.seed\n"
        << "  --help\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            print_usage();
            std::exit(0);
        }

        auto require_value = [&](const std::string& opt) -> std::string {
            if (i + 1 >= argc) {
                fail("missing value for option " + opt);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--config") {
            options.config_path = require_value(arg);
        } else if (arg == "--summary-json") {
            options.summary_path = require_value(arg);
        } else if (arg == "--lambda") {
            options.lambda = parse_double_arg(require_value(arg), arg);
            options.has_lambda = true;
        } else if (arg == "--alpha") {
            options.alpha = parse_double_arg(require_value(arg), arg);
            options.has_alpha = true;
        } else if (arg == "--threads") {
            options.threads_override = parse_nonnegative_int_arg(require_value(arg), arg);
            options.has_threads_override = true;
        } else if (arg == "--seed") {
            options.seed_override = parse_uint64_arg(require_value(arg), arg);
            options.has_seed_override = true;
        } else if (starts_with_dash_dash(arg)) {
            fail("unknown option: " + arg);
        } else {
            fail("unexpected positional argument: " + arg);
        }
    }

    if (options.config_path.empty()) {
        fail("required option missing: --config");
    }
    if (options.summary_path.empty()) {
        fail("required option missing: --summary-json");
    }
    if (!options.has_lambda) {
        fail("required option missing: --lambda");
    }
    if (!options.has_alpha) {
        fail("required option missing: --alpha");
    }
    return options;
}

std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char ch : input) {
        switch (ch) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (ch < 0x20U) {
                std::ostringstream oss;
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(ch);
                out += oss.str();
            } else {
                out.push_back(static_cast<char>(ch));
            }
            break;
        }
    }
    return out;
}

void write_summary_json(
    const std::filesystem::path& path,
    const wck::WorkloadSummary& summary,
    double warmup_time,
    double sample_time,
    bool normalize_service_mean_to_one) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open summary output: " + path.string());
    }
    out << std::setprecision(17) << std::scientific;
    out << "{\n";
    out << "  \"model_name\": \"" << json_escape(summary.model_name) << "\",\n";
    out << "  \"lambda\": " << summary.lambda << ",\n";
    out << "  \"alpha\": " << summary.alpha << ",\n";
    out << "  \"mean_workload\": " << summary.mean_workload << ",\n";
    out << "  \"std_workload\": " << summary.std_workload << ",\n";
    out << "  \"n_reps\": " << summary.n_reps << ",\n";
    out << "  \"threads_used\": " << summary.threads_used << ",\n";
    out << "  \"seed\": " << summary.seed << ",\n";
    out << "  \"runtime_seconds\": " << summary.runtime_seconds << ",\n";
    out << "  \"warmup_time\": " << warmup_time << ",\n";
    out << "  \"sample_time\": " << sample_time << ",\n";
    out << "  \"normalize_service_mean_to_one\": "
        << (normalize_service_mean_to_one ? "true" : "false") << "\n";
    out << "}\n";
}

// A config is a tandem config iff model.queue1 is present.
bool config_is_tandem(const std::filesystem::path& config_path) {
    const wck::json::Value root = wck::json::parse_file(config_path);
    if (!root.is_object()) {
        return false;
    }
    const wck::json::Value* model = root.get("model");
    if (model == nullptr || !model->is_object()) {
        return false;
    }
    return model->get("queue1") != nullptr;
}

wck::WorkloadSummary run_single_station(
    const CliOptions& options,
    const std::filesystem::path& config_path,
    wck::WorkloadSimulationConfig* simulation_out) {
    const wck::WorkloadConfig config = wck::load_workload_config(config_path);

    wck::WorkloadRunParams params{};
    params.model_name = config.model.name;
    params.lambda = options.lambda;
    params.alpha = options.alpha;
    params.arrival = config.model.arrival;
    params.service = config.model.service;
    params.patience = config.model.patience;
    params.warmup_time = config.simulation.warmup_time;
    params.sample_time = config.simulation.sample_time;
    params.replications = config.simulation.replications;
    params.threads = config.simulation.threads;
    params.seed = config.simulation.seed;
    params.normalize_service_mean_to_one = config.simulation.normalize_service_mean_to_one;

    if (options.has_threads_override) {
        params.threads = options.threads_override;
    }
    if (options.has_seed_override) {
        params.seed = options.seed_override;
    }

    *simulation_out = config.simulation;
    return wck::simulate_workload_mc(params);
}

wck::WorkloadSummary run_tandem(
    const CliOptions& options,
    const std::filesystem::path& config_path,
    wck::WorkloadSimulationConfig* simulation_out) {
    const wck::TandemWorkloadConfig config = wck::load_tandem_workload_config(config_path);

    wck::TandemWorkloadRunParams params{};
    params.model_name = config.model.name;
    params.lambda = options.lambda;
    params.alpha = options.alpha;
    params.queue1_traffic_intensity = config.model.queue1.traffic_intensity;
    params.queue1_arrival = config.model.queue1.arrival;
    params.queue1_service = config.model.queue1.service;
    params.queue2_service = config.model.queue2.service;
    params.queue2_patience = config.model.queue2.patience;
    params.warmup_time = config.simulation.warmup_time;
    params.sample_time = config.simulation.sample_time;
    params.replications = config.simulation.replications;
    params.threads = config.simulation.threads;
    params.seed = config.simulation.seed;
    params.normalize_service_mean_to_one = config.simulation.normalize_service_mean_to_one;

    if (options.has_threads_override) {
        params.threads = options.threads_override;
    }
    if (options.has_seed_override) {
        params.seed = options.seed_override;
    }

    *simulation_out = config.simulation;
    return wck::simulate_tandem_workload_mc(params);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);
        const std::filesystem::path config_path = options.config_path.is_absolute()
            ? options.config_path
            : (std::filesystem::current_path() / options.config_path);

        wck::WorkloadSimulationConfig simulation{};
        const wck::WorkloadSummary summary = config_is_tandem(config_path)
            ? run_tandem(options, config_path, &simulation)
            : run_single_station(options, config_path, &simulation);

        const std::filesystem::path summary_path = options.summary_path.is_absolute()
            ? options.summary_path
            : (std::filesystem::current_path() / options.summary_path);
        const std::filesystem::path summary_parent = summary_path.parent_path();
        if (!summary_parent.empty()) {
            std::filesystem::create_directories(summary_parent);
        }
        write_summary_json(
            summary_path,
            summary,
            simulation.warmup_time,
            simulation.sample_time,
            simulation.normalize_service_mean_to_one);

        std::cout << std::setprecision(6) << std::fixed
                  << "lambda=" << summary.lambda
                  << " alpha=" << summary.alpha
                  << " mean_workload=" << summary.mean_workload
                  << " std_workload=" << summary.std_workload
                  << " reps=" << summary.n_reps
                  << " runtime=" << std::setprecision(3) << summary.runtime_seconds << "s\n";
        std::cout.unsetf(std::ios::floatfield);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "Use --help for usage.\n";
        return 1;
    }

    return 0;
}
