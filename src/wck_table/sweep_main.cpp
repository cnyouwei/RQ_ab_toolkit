#include "wck/wck_table/solver.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool starts_with_dash_dash(const std::string& s) {
    return s.rfind("--", 0U) == 0U;
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

std::size_t parse_size_arg(const std::string& value, const std::string& name) {
    std::size_t parsed = 0U;
    unsigned long long out = 0ULL;
    try {
        out = std::stoull(value, &parsed);
    } catch (...) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
    if (parsed != value.size()) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
    return static_cast<std::size_t>(out);
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
        << "Usage: wck_sweep --k <int> [options]\n"
        << "Computes w(c,t) directly in memory over a c-grid and writes one matrix CSV.\n"
        << "Options:\n"
        << "  --k <int>                required, k>=1\n"
        << "  --c-min <double>         default: -20\n"
        << "  --c-max <double>         default: 20\n"
        << "  --dc <double>            default: 0.1\n"
        << "  --dir <path>             default: ./results/\n"
        << "  --out <path>             default: w_table_matrix_k{k}.csv (under --dir)\n"
        << "  --jobs <int>             default: hardware concurrency\n"
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

std::vector<double> build_c_grid(double c_min, double c_max, double dc) {
    if (!(dc > 0.0)) {
        throw std::invalid_argument("dc must be > 0");
    }
    if (c_max < c_min) {
        throw std::invalid_argument("c_max must be >= c_min");
    }

    const double n_float = (c_max - c_min) / dc;
    const long long n = std::llround(n_float);
    if (std::abs(n_float - static_cast<double>(n)) > 1e-9) {
        throw std::invalid_argument("(c_max - c_min) must be an integer multiple of dc");
    }

    std::vector<double> grid{};
    grid.reserve(static_cast<std::size_t>(n + 1));
    for (long long i = 0; i <= n; ++i) {
        double c = c_min + static_cast<double>(i) * dc;
        c = std::round(c * 1e12) / 1e12;
        if (std::abs(c) < 5e-13) {
            c = 0.0;
        }
        grid.push_back(c);
    }
    return grid;
}

std::string default_matrix_name(int k) {
    return "w_table_matrix_k" + std::to_string(k) + ".csv";
}

bool almost_equal(double a, double b) {
    const double abs_tol = 1e-14;
    const double rel_tol = 1e-12;
    return std::abs(a - b) <= std::max(abs_tol, rel_tol * std::max({1.0, std::abs(a), std::abs(b)}));
}

void write_matrix_csv(
    const std::filesystem::path& path,
    const std::vector<double>& c_grid,
    const std::vector<wck::SolveResult>& results) {
    if (results.empty()) {
        throw std::invalid_argument("no sweep results to write");
    }
    const std::vector<wck::GridPoint>& ref_points = results.front().points;
    if (ref_points.empty()) {
        throw std::runtime_error("reference result has no points");
    }
    for (std::size_t i = 1; i < results.size(); ++i) {
        const auto& pts = results[i].points;
        if (pts.size() != ref_points.size()) {
            throw std::runtime_error("inconsistent time-grid size across c results");
        }
        for (std::size_t j = 0; j < pts.size(); ++j) {
            if (!almost_equal(pts[j].t, ref_points[j].t)) {
                throw std::runtime_error("inconsistent time-grid values across c results");
            }
        }
    }

    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }

    out << std::setprecision(17) << std::scientific;
    out << "c";
    for (const auto& point : ref_points) {
        out << "," << point.t;
    }
    out << '\n';

    for (std::size_t i = 0; i < c_grid.size(); ++i) {
        out << c_grid[i];
        for (const auto& point : results[i].points) {
            out << "," << point.w;
        }
        out << '\n';
    }
    out.flush();
    if (!out.good()) {
        throw std::runtime_error("failed while writing output file: " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    int k = -1;
    double c_min = -20.0;
    double c_max = 20.0;
    double dc = 0.1;
    std::filesystem::path out_dir = "./results";
    std::string out_path{};

    wck::SolverConfig config{};

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

            if (arg == "--k") {
                k = parse_int_arg(require_value(arg), arg);
            } else if (arg == "--c-min") {
                c_min = parse_double_arg(require_value(arg), arg);
            } else if (arg == "--c-max") {
                c_max = parse_double_arg(require_value(arg), arg);
            } else if (arg == "--dc") {
                dc = parse_double_arg(require_value(arg), arg);
            } else if (arg == "--dir") {
                out_dir = require_value(arg);
            } else if (arg == "--out") {
                out_path = require_value(arg);
            } else if (arg == "--jobs") {
                config.thread_count = parse_size_arg(require_value(arg), arg);
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
            } else {
                throw std::invalid_argument("unknown option: " + arg);
            }
        }

        if (k < 1) {
            throw std::invalid_argument("required option: --k (k>=1)");
        }
        if (out_dir.empty()) {
            throw std::invalid_argument("--dir cannot be empty");
        }

        const std::vector<double> c_grid = build_c_grid(c_min, c_max, dc);
        std::vector<wck::WckParams> params{};
        params.reserve(c_grid.size());
        for (double c : c_grid) {
            params.push_back(wck::WckParams{c, k});
        }

        const std::vector<wck::SolveResult> results = wck::solve_many(params, config);
        if (results.size() != c_grid.size()) {
            throw std::runtime_error("internal error: result size mismatch");
        }

        std::filesystem::path resolved_out{};
        if (out_path.empty()) {
            resolved_out = out_dir / default_matrix_name(k);
        } else {
            resolved_out = std::filesystem::path(out_path);
            if (resolved_out.is_relative()) {
                resolved_out = out_dir / resolved_out;
            }
        }

        write_matrix_csv(resolved_out, c_grid, results);

        std::cout << "Wrote matrix table: " << resolved_out << '\n';
        std::cout << "Grid sizes: |c|=" << c_grid.size()
                  << ", |t|=" << results.front().points.size() << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "Use --help for usage.\n";
        return 1;
    }

    return 0;
}

