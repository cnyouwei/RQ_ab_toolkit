#include "wck/idw_sim/effective_idw_sim.hpp"
#include "wck/idw_sim/effective_idw_sim_config.hpp"
#include "wck/common/distributions.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <chrono>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path config_path{};
    std::filesystem::path out_dir{};
    bool has_out_dir = false;
    bool has_seed_override = false;
    std::uint64_t seed_override = 0U;
    bool has_threads_override = false;
    int threads_override = 0;
    bool no_events = false;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument(message);
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

bool starts_with_dash_dash(const std::string& s) {
    return s.rfind("--", 0U) == 0U;
}

void print_usage() {
    std::cout
        << "Usage: idw_sim --config <path> [options]\n"
        << "Options:\n"
        << "  --out-dir <path>     output directory (default: ./results)\n"
        << "  --seed <uint64>      override simulation.seed from config\n"
        << "  --threads <int>      estimator threads (0=auto, overrides simulation.threads)\n"
        << "  --no-events          force disable event-trace output\n"
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
        } else if (arg == "--out-dir") {
            options.out_dir = require_value(arg);
            options.has_out_dir = true;
        } else if (arg == "--seed") {
            options.seed_override = parse_uint64_arg(require_value(arg), arg);
            options.has_seed_override = true;
        } else if (arg == "--threads") {
            options.threads_override = parse_nonnegative_int_arg(require_value(arg), arg);
            options.has_threads_override = true;
        } else if (arg == "--no-events") {
            options.no_events = true;
        } else if (starts_with_dash_dash(arg)) {
            fail("unknown option: " + arg);
        } else {
            fail("unexpected positional argument: " + arg);
        }
    }

    if (options.config_path.empty()) {
        fail("required option missing: --config");
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

void render_progress_bar(
    double fraction,
    std::size_t completed_runs,
    std::size_t total_runs,
    const std::string& model_name,
    int alpha_index,
    double eta_seconds) {
    auto format_eta = [](double seconds) -> std::string {
        if (!std::isfinite(seconds) || seconds < 0.0) {
            return "--:--";
        }

        long long total = static_cast<long long>(std::llround(seconds));
        const long long hours = total / 3600LL;
        total %= 3600LL;
        const long long minutes = total / 60LL;
        const long long secs = total % 60LL;

        std::ostringstream tmp;
        if (hours > 0LL) {
            tmp << hours << 'h'
                << std::setw(2) << std::setfill('0') << minutes << 'm'
                << std::setw(2) << std::setfill('0') << secs << 's';
        } else {
            tmp << std::setw(2) << std::setfill('0') << minutes
                << ':' << std::setw(2) << std::setfill('0') << secs;
        }
        return tmp.str();
    };

    const double clamped = std::clamp(fraction, 0.0, 1.0);
    constexpr int kBarWidth = 36;
    const int filled = static_cast<int>(std::round(clamped * static_cast<double>(kBarWidth)));

    std::ostringstream oss;
    oss << '\r' << '[';
    for (int i = 0; i < kBarWidth; ++i) {
        oss << (i < filled ? '#' : '-');
    }
    oss << "] " << std::fixed << std::setprecision(1) << (100.0 * clamped) << "% "
        << "run " << (completed_runs + 1) << '/' << total_runs
        << " model='" << model_name << "' alpha_index=" << alpha_index
        << " ETA " << format_eta(eta_seconds);

    std::cout << oss.str() << std::flush;
}

void clear_progress_bar_line() {
    std::cout << '\r' << std::string(220, ' ') << '\r' << std::flush;
}

void write_curve_csv(const std::filesystem::path& path, const std::vector<wck::EffectiveIdwEstimate>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open curve CSV: " + path.string());
    }

    out << std::setprecision(17) << std::scientific;
    out << "tau,tau_shift,tau_shift_index,level,t,r_bins,n_windows,ey_hat,vary_hat,ev_hat,idw_hat\n";
    for (const auto& row : rows) {
        out << row.tau << ','
            << row.tau_shift << ','
            << row.tau_shift_index << ','
            << row.level << ','
            << row.t << ','
            << row.r_bins << ','
            << row.n_windows << ','
            << row.ey_hat << ','
            << row.vary_hat << ','
            << row.ev_hat << ','
            << row.idw_hat << '\n';
    }
}

void write_distribution_json(std::ostream& out, const wck::DistributionSpec& spec, const std::string& indent) {
    out << indent << "{\n";
    out << indent << "  \"family\": \"" << wck::distribution_family_name(spec.family) << "\",\n";
    out << indent << "  \"params\": {\n";
    switch (spec.family) {
    case wck::DistributionFamily::kExponential:
        out << indent << "    \"rate\": " << spec.exponential.rate << '\n';
        break;
    case wck::DistributionFamily::kErlangK:
        out << indent << "    \"k\": " << spec.erlang_k.k << ",\n";
        out << indent << "    \"rate\": " << spec.erlang_k.rate << '\n';
        break;
    case wck::DistributionFamily::kLognormal:
        out << indent << "    \"mean\": " << spec.lognormal.mean << ",\n";
        out << indent << "    \"scv\": " << spec.lognormal.scv << '\n';
        break;
    case wck::DistributionFamily::kHyperexponential2:
        out << indent << "    \"p\": " << spec.hyperexponential2.p << ",\n";
        out << indent << "    \"rate1\": " << spec.hyperexponential2.rate1 << ",\n";
        out << indent << "    \"rate2\": " << spec.hyperexponential2.rate2 << '\n';
        break;
    }
    out << indent << "  }\n";
    out << indent << "}";
}

void write_summary_json(
    const std::filesystem::path& path,
    const wck::SimulationRunResult& result,
    const std::filesystem::path& curve_path,
    const std::filesystem::path* event_path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open summary JSON: " + path.string());
    }

    const auto& p = result.params;
    const auto& s = result.stats;
    const double served_fraction =
        (s.arrivals_sample > 0U)
        ? static_cast<double>(s.accepted_sample) / static_cast<double>(s.arrivals_sample)
        : 0.0;
    const double effective_work_rate =
        (s.sample_observed_time > 0.0)
        ? s.sample_effective_work / s.sample_observed_time
        : 0.0;
    const double ev_hat_sample =
        (s.sample_service_count > 0U)
        ? s.sample_service_sum / static_cast<double>(s.sample_service_count)
        : std::numeric_limits<double>::quiet_NaN();
    const wck::DistributionMoments service_moments = wck::distribution_moments(p.service);
    const double service_mu_effective = 1.0 / service_moments.mean;
    const wck::DistributionMoments arrival_base_moments = wck::distribution_moments(p.arrival);
    const double arrival_base_rate = 1.0 / arrival_base_moments.mean;
    const wck::DistributionSpec arrival_runtime =
        wck::scale_distribution_rates(p.arrival, p.lambda / arrival_base_rate);
    const wck::DistributionSpec patience_runtime =
        wck::scale_distribution_rates(p.patience, p.alpha);
    const wck::DistributionMoments arrival_runtime_moments = wck::distribution_moments(arrival_runtime);

    out << std::setprecision(17) << std::scientific;
    out << "{\n";
    out << "  \"model_name\": \"" << json_escape(p.model_name) << "\",\n";
    out << "  \"model_index\": " << p.model_index << ",\n";
    out << "  \"alpha_index\": " << p.alpha_index << ",\n";
    out << "  \"alpha\": " << p.alpha << ",\n";
    out << "  \"h\": " << p.h << ",\n";
    out << "  \"c\": " << p.c << ",\n";
    out << "  \"rho_exponent\": " << p.rho_exponent << ",\n";
    out << "  \"rho\": " << p.rho << ",\n";
    out << "  \"lambda\": " << p.lambda << ",\n";
    out << "  \"mu\": " << service_mu_effective << ",\n";
    out << "  \"arrival_scv\": " << arrival_runtime_moments.scv << ",\n";
    out << "  \"service_scv\": " << service_moments.scv << ",\n";
    out << "  \"service_mu_effective\": " << service_mu_effective << ",\n";
    out << "  \"seed\": " << p.seed << ",\n";
    out << "  \"runtime_seconds\": " << result.runtime_seconds << ",\n";
    out << "  \"estimator_threads_used\": " << result.estimator_threads_used << ",\n";
    out << "  \"estimation_wall_seconds\": " << result.estimation_wall_seconds << ",\n";

    out << "  \"scaling\": {\n";
    out << "    \"k\": " << p.scaling.k << ",\n";
    out << "    \"beta_patience\": " << p.scaling.beta_patience << ",\n";
    out << "    \"rho_exponent\": " << p.rho_exponent << '\n';
    out << "  },\n";

    out << "  \"distributions\": {\n";
    out << "    \"arrival\": ";
    write_distribution_json(out, arrival_runtime, "    ");
    out << ",\n";
    out << "    \"service\": ";
    write_distribution_json(out, p.service, "    ");
    out << ",\n";
    out << "    \"patience\": ";
    write_distribution_json(out, patience_runtime, "    ");
    out << '\n';
    out << "  },\n";

    out << "  \"simulation\": {\n";
    out << "    \"warmup_time\": " << p.simulation.warmup_time << ",\n";
    out << "    \"sample_time\": " << p.simulation.sample_time << ",\n";
    out << "    \"sample_observed_time\": " << s.sample_observed_time << ",\n";
    out << "    \"tau\": " << p.simulation.tau << ",\n";
    out << "    \"max_level\": " << p.simulation.max_level << ",\n";
    out << "    \"min_windows_per_t\": " << p.simulation.min_windows_per_t << ",\n";
    out << "    \"n_tau_shifts\": " << p.simulation.n_tau_shifts << ",\n";
    out << "    \"threads\": " << p.simulation.threads << "\n";
    out << "  },\n";

    out << "  \"counts\": {\n";
    out << "    \"arrivals_total\": " << s.arrivals_total << ",\n";
    out << "    \"accepted_total\": " << s.accepted_total << ",\n";
    out << "    \"abandoned_total\": " << s.abandoned_total << ",\n";
    out << "    \"arrivals_sample\": " << s.arrivals_sample << ",\n";
    out << "    \"accepted_sample\": " << s.accepted_sample << ",\n";
    out << "    \"abandoned_sample\": " << s.abandoned_sample << "\n";
    out << "  },\n";

    out << "  \"served_fraction_sample\": " << served_fraction << ",\n";
    out << "  \"effective_work_rate\": " << effective_work_rate << ",\n";
    out << "  \"ev_hat_sample\": " << ev_hat_sample << ",\n";

    out << "  \"dropped_horizons\": [\n";
    for (std::size_t i = 0; i < result.dropped_horizons.size(); ++i) {
        out << "    { \"t\": " << result.dropped_horizons[i] << ", \"reason\": \""
            << json_escape(result.dropped_reasons[i]) << "\" }";
        if (i + 1U < result.dropped_horizons.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ],\n";

    out << "  \"output\": {\n";
    out << "    \"curve_csv\": \"" << json_escape(curve_path.string()) << "\",\n";
    if (event_path != nullptr) {
        out << "    \"event_trace_csv\": \"" << json_escape(event_path->string()) << "\"\n";
    } else {
        out << "    \"event_trace_csv\": null\n";
    }
    out << "  }\n";

    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);
        const std::filesystem::path config_path =
            options.config_path.is_absolute()
            ? options.config_path
            : (std::filesystem::current_path() / options.config_path);

        wck::EffectiveIdwSimConfig config = wck::load_effective_idw_sim_config(config_path);
        if (options.has_seed_override) {
            config.simulation.seed = options.seed_override;
        }
        if (options.has_threads_override) {
            config.simulation.threads = options.threads_override;
        }

        std::filesystem::path out_dir;
        if (options.has_out_dir) {
            out_dir = options.out_dir.is_absolute()
                ? options.out_dir
                : (std::filesystem::current_path() / options.out_dir);
        } else {
            out_dir = std::filesystem::current_path() / "results";
        }
        std::filesystem::create_directories(out_dir);

        const std::size_t total_runs = config.models.size() * config.alpha.indices.size();
        std::size_t run_count = 0U;
        const auto overall_start = std::chrono::steady_clock::now();
        for (std::size_t model_idx = 0U; model_idx < config.models.size(); ++model_idx) {
        const auto& model = config.models[model_idx];
            for (const int alpha_index : config.alpha.indices) {
                const double alpha = wck::alpha_from_index(alpha_index, config.alpha.base);
                const double h = static_cast<double>(model.scaling.k) / static_cast<double>(model.scaling.k + 1);
                const double rho_exponent = model.scaling.has_rho_exponent ? model.scaling.rho_exponent : h;
                const double rho = 1.0 + model.system.c * std::pow(alpha, rho_exponent);
                const wck::DistributionMoments service_moments = wck::distribution_moments(model.service);
                const double mu = 1.0 / service_moments.mean;
                const double lambda = rho * mu;
                if (!(lambda > 0.0)) {
                    throw std::runtime_error(
                        "invalid lambda <= 0 for model '" + model.name + "' alpha index "
                        + std::to_string(alpha_index));
                }

                wck::RunParameters params{};
                params.model_name = model.name;
                params.model_index = model_idx;
                params.alpha_index = alpha_index;
                params.alpha = alpha;
                params.h = h;
                params.c = model.system.c;
                params.rho_exponent = rho_exponent;
                params.rho = rho;
                params.lambda = lambda;
                params.arrival = model.arrival;
                params.service = model.service;
                params.patience = model.patience;
                params.scaling = model.scaling;
                params.simulation = config.simulation;
                params.seed = wck::derive_run_seed(config.simulation.seed, model_idx, alpha_index, model.name);

                const std::string prefix = "model" + std::to_string(model_idx)
                    + "_" + wck::sanitize_name_for_file(model.name)
                    + "_idx" + std::to_string(alpha_index);

                const std::filesystem::path curve_path = out_dir / (prefix + "_curve.csv");
                const std::filesystem::path summary_path = out_dir / (prefix + "_summary.json");

                std::filesystem::path event_path;
                const std::filesystem::path* event_ptr = nullptr;
                const bool save_events = params.simulation.save_event_trace && !options.no_events;
                if (save_events) {
                    event_path = out_dir / (prefix + "_events.csv");
                    event_ptr = &event_path;
                }

                auto progress_callback = [&](double run_fraction) {
                    const double overall =
                        (static_cast<double>(run_count) + std::clamp(run_fraction, 0.0, 1.0))
                        / static_cast<double>(std::max<std::size_t>(total_runs, 1U));
                    double eta_seconds = std::numeric_limits<double>::quiet_NaN();
                    if (overall > 1e-6 && overall < 1.0) {
                        const auto now = std::chrono::steady_clock::now();
                        const double elapsed_seconds =
                            std::chrono::duration<double>(now - overall_start).count();
                        eta_seconds = elapsed_seconds * (1.0 - overall) / overall;
                    } else if (overall >= 1.0) {
                        eta_seconds = 0.0;
                    }
                    render_progress_bar(
                        overall,
                        run_count,
                        total_runs,
                        model.name,
                        alpha_index,
                        eta_seconds);
                };

                const wck::SimulationRunResult result =
                    wck::simulate_effective_idw(params, event_ptr, progress_callback);
                write_curve_csv(curve_path, result.estimates);
                write_summary_json(summary_path, result, curve_path, event_ptr);
                clear_progress_bar_line();

                std::cout << "[run " << (++run_count) << "] model='" << model.name
                          << "' alpha_index=" << alpha_index
                          << " alpha=" << std::setprecision(6) << std::fixed << alpha
                          << " rho=" << rho
                          << " runtime=" << std::setprecision(3) << result.runtime_seconds << "s"
                          << " dropped=" << result.dropped_horizons.size()
                          << "\n";
                std::cout.unsetf(std::ios::floatfield);
            }
        }

    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "Use --help for usage.\n";
        return 1;
    }

    return 0;
}
