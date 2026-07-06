#include "wck/rq_calibration/b_calibration.hpp"
#include "wck/rq_calibration/w_table_interpolator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
    int k = -1;
    std::filesystem::path w_table{};
    bool has_w_table = false;

    std::filesystem::path out_dir = "./results";
    std::string out_path{};

    bool has_c_min = false;
    bool has_c_max = false;
    bool has_dc = false;
    double c_min = -20.0;
    double c_max = 20.0;
    double dc = 0.1;

    std::size_t jobs = 0U;
    bool has_jobs = false;
};

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

void print_usage() {
    std::cout
        << "Usage: wck_calibrate_b --k <int> [options]\n"
        << "Options:\n"
        << "  --k <int>                required, k>=1\n"
        << "  --w-table <path>         default: ./results/w_table_matrix_k{k}.csv\n"
        << "  --dir <path>             default: ./results/\n"
        << "  --out <path>             default: b_table_k{k}.csv (under --dir)\n"
        << "  --c-min <double>         optional calibration c-grid min\n"
        << "  --c-max <double>         optional calibration c-grid max\n"
        << "  --dc <double>            optional calibration c-grid step\n"
        << "                           (must specify --c-min, --c-max, and --dc together)\n"
        << "  --jobs <int>             worker count (default: hardware concurrency)\n"
        << "  --help\n";
}

std::string default_b_table_name(int k) {
    return "b_table_k" + std::to_string(k) + ".csv";
}

std::string default_w_table_name(int k) {
    return "w_table_matrix_k" + std::to_string(k) + ".csv";
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
                throw std::invalid_argument("missing value for option " + opt);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--k") {
            options.k = parse_int_arg(require_value(arg), arg);
        } else if (arg == "--w-table") {
            options.w_table = require_value(arg);
            options.has_w_table = true;
        } else if (arg == "--dir") {
            options.out_dir = require_value(arg);
        } else if (arg == "--out") {
            options.out_path = require_value(arg);
        } else if (arg == "--c-min") {
            options.c_min = parse_double_arg(require_value(arg), arg);
            options.has_c_min = true;
        } else if (arg == "--c-max") {
            options.c_max = parse_double_arg(require_value(arg), arg);
            options.has_c_max = true;
        } else if (arg == "--dc") {
            options.dc = parse_double_arg(require_value(arg), arg);
            options.has_dc = true;
        } else if (arg == "--jobs") {
            options.jobs = parse_size_arg(require_value(arg), arg);
            options.has_jobs = true;
        } else if (starts_with_dash_dash(arg)) {
            throw std::invalid_argument("unknown option: " + arg);
        } else {
            throw std::invalid_argument("unexpected positional argument: " + arg);
        }
    }

    if (options.k < 1) {
        throw std::invalid_argument("required option: --k (k>=1)");
    }
    if (options.out_dir.empty()) {
        throw std::invalid_argument("--dir cannot be empty");
    }

    const int c_range_count = static_cast<int>(options.has_c_min) + static_cast<int>(options.has_c_max)
        + static_cast<int>(options.has_dc);
    if (c_range_count != 0 && c_range_count != 3) {
        throw std::invalid_argument("must specify --c-min, --c-max, and --dc together");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        std::filesystem::path w_table_path{};
        if (options.has_w_table) {
            w_table_path = options.w_table;
        } else {
            w_table_path = std::filesystem::path("./results") / default_w_table_name(options.k);
        }
        if (w_table_path.is_relative()) {
            w_table_path = std::filesystem::current_path() / w_table_path;
        }

        const wck::WTableInterpolator w_table = wck::WTableInterpolator::from_matrix_csv(w_table_path);

        std::vector<double> c_grid{};
        if (options.has_c_min) {
            c_grid = wck::build_c_grid(options.c_min, options.c_max, options.dc);
        } else {
            c_grid = w_table.c_grid();
        }

        const std::size_t jobs = options.has_jobs ? options.jobs : 0U;
        const wck::BCalibrationResult result = wck::calibrate_b_table(options.k, c_grid, w_table, jobs);

        std::filesystem::path resolved_out{};
        if (options.out_path.empty()) {
            resolved_out = options.out_dir / default_b_table_name(options.k);
        } else {
            resolved_out = std::filesystem::path(options.out_path);
            if (resolved_out.is_relative()) {
                resolved_out = options.out_dir / resolved_out;
            }
        }
        if (resolved_out.is_relative()) {
            resolved_out = std::filesystem::current_path() / resolved_out;
        }

        wck::write_b_calibration_table_csv(resolved_out, result);

        std::size_t exact_optimized_count = 0U;
        std::size_t sqrt2_capped_count = 0U;
        std::size_t zero_fallback_count = 0U;
        for (const auto& row : result.rows) {
            if (std::isfinite(row.u_star)) {
                ++exact_optimized_count;
            } else {
                if (std::abs(row.b) < 1e-14) {
                    ++zero_fallback_count;
                } else {
                    ++sqrt2_capped_count;
                }
            }
        }

        std::cout << "Wrote calibrated b-table: " << resolved_out << '\n';
        std::cout << "Rows: " << result.rows.size()
                  << " (optimized=" << exact_optimized_count
                  << ", sqrt2_capped=" << sqrt2_capped_count
                  << ", zero_fallback=" << zero_fallback_count << ")\n";
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "Use --help for usage.\n";
        return 1;
    }

    return 0;
}
