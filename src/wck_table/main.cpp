#include "wck/wck_table/solver.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

bool starts_with_dash_dash(const std::string& s) {
    return s.rfind("--", 0U) == 0U;
}

std::string format_c_for_name(double c) {
    std::ostringstream oss;
    oss << std::setprecision(15) << std::defaultfloat << c;
    std::string out = oss.str();
    out.erase(std::remove(out.begin(), out.end(), '+'), out.end());
    return out;
}

std::string default_csv_name(double c, int k) {
    return "wck_c" + format_c_for_name(c) + "_k" + std::to_string(k) + ".csv";
}

double parse_double_arg(const std::string& value, const std::string& name) {
    std::size_t parsed = 0U;
    double out = 0.0;
    try {
        out = std::stod(value, &parsed);
    } catch (...) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
    if (parsed != value.size()) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
    return out;
}

int parse_int_arg(const std::string& value, const std::string& name) {
    std::size_t parsed = 0U;
    int out = 0;
    try {
        out = std::stoi(value, &parsed);
    } catch (...) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
    if (parsed != value.size()) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
    return out;
}

bool parse_bool_arg(const std::string& value, const std::string& name) {
    if (value == "1" || value == "true" || value == "TRUE" || value == "True") {
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE" || value == "False") {
        return false;
    }
    throw std::invalid_argument("invalid value for " + name + ": " + value);
}

void print_usage() {
    std::cout
        << "Usage: wck --c <double> --k <int> [--out <path>] [options]\n"
        << "Default output naming rule (when --out is omitted):\n"
        << "  wck_c{c}_k{k}.csv\n"
        << "Options:\n"
        << "  --dir <path>             default: ./results/\n"
        << "  --t-min <double>         default: 1e-4\n"
        << "  --t-max <double>         default: 1e8\n"
        << "  --n-t <int>              default: 1200\n"
        << "  --include-t0 [bool]      default: true\n"
        << "  --no-include-t0          force include_t0=false\n"
        << "  --nx <int>               default: 2000\n"
        << "  --x-max <double>         use fixed x_max and disable auto selection\n"
        << "  --tail-tol <double>      default: 1e-14\n"
        << "  --rel-dt-max <double>    default: 0.03\n"
        << "  --steady-tol <double>    default: 1e-9\n"
        << "  --steady-window <int>    default: 25\n"
        << "  --help\n";
}

}  // namespace

int main(int argc, char** argv) {
    wck::WckParams params{};
    wck::SolverConfig config{};
    std::string out_path{};
    std::filesystem::path out_dir = "./results";
    bool have_c = false;
    bool have_k = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                print_usage();
                return 0;
            }

            auto require_value = [&](const std::string& opt) -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("missing value for option " + opt);
                }
                return std::string(argv[++i]);
            };

            if (arg == "--c") {
                params.c = parse_double_arg(require_value(arg), arg);
                have_c = true;
            } else if (arg == "--k") {
                params.k = parse_int_arg(require_value(arg), arg);
                have_k = true;
            } else if (arg == "--dir") {
                out_dir = require_value(arg);
            } else if (arg == "--t-min") {
                config.time.t_min = parse_double_arg(require_value(arg), arg);
            } else if (arg == "--t-max") {
                config.time.t_max = parse_double_arg(require_value(arg), arg);
            } else if (arg == "--n-t") {
                config.time.n_points = parse_int_arg(require_value(arg), arg);
            } else if (arg == "--include-t0") {
                if (i + 1 < argc && !starts_with_dash_dash(argv[i + 1])) {
                    config.time.include_t0 = parse_bool_arg(require_value(arg), arg);
                } else {
                    config.time.include_t0 = true;
                }
            } else if (arg == "--no-include-t0") {
                config.time.include_t0 = false;
            } else if (arg == "--nx") {
                config.space.nx = parse_int_arg(require_value(arg), arg);
            } else if (arg == "--x-max") {
                config.space.x_max = parse_double_arg(require_value(arg), arg);
                config.space.auto_x_max = false;
            } else if (arg == "--tail-tol") {
                config.space.tail_tol = parse_double_arg(require_value(arg), arg);
            } else if (arg == "--rel-dt-max") {
                config.rel_dt_max = parse_double_arg(require_value(arg), arg);
            } else if (arg == "--steady-tol") {
                config.steady_tol = parse_double_arg(require_value(arg), arg);
            } else if (arg == "--steady-window") {
                config.steady_window = parse_int_arg(require_value(arg), arg);
            } else if (arg == "--out") {
                out_path = require_value(arg);
            } else {
                throw std::invalid_argument("unknown option: " + arg);
            }
        }

        if (!have_c || !have_k) {
            throw std::invalid_argument("required options: --c, --k");
        }
        if (out_dir.empty()) {
            throw std::invalid_argument("--dir cannot be empty");
        }

        std::filesystem::path resolved_path;
        if (out_path.empty()) {
            resolved_path = out_dir / default_csv_name(params.c, params.k);
        } else {
            resolved_path = std::filesystem::path(out_path);
            if (resolved_path.is_relative()) {
                resolved_path = out_dir / resolved_path;
            }
        }

        const std::filesystem::path parent_dir = resolved_path.parent_path();
        if (!parent_dir.empty()) {
            std::filesystem::create_directories(parent_dir);
        }

        const wck::SolveResult result = wck::solve_w_grid(params, config);

        std::ofstream out(resolved_path);
        if (!out.is_open()) {
            throw std::runtime_error("failed to open output file: " + resolved_path.string());
        }

        out << std::setprecision(17) << std::scientific;
        out << "t,w,m_term,var_h_term\n";
        for (const auto& point : result.points) {
            out << point.t
                << ',' << point.w
                << ',' << point.m_term
                << ',' << point.var_h_term
                << '\n';
        }
        out.flush();

        if (!out.good()) {
            throw std::runtime_error("failed while writing output file: " + resolved_path.string());
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "Use --help for usage.\n";
        return 1;
    }

    return 0;
}
