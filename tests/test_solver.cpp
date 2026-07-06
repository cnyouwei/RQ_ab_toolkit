#include "wck/wck_table/solver.hpp"

#include "support.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

void run_effective_idw_sim_tests();
void run_b_calibration_tests();
void run_workload_sim_tests();
void run_tandem_workload_sim_tests();

namespace {

double relative_error(double a, double b) {
    const double denom = std::max(1e-12, std::abs(b));
    return std::abs(a - b) / denom;
}

wck::SolverConfig base_test_config() {
    wck::SolverConfig cfg{};
    cfg.time.t_min = 1e-4;
    cfg.time.t_max = 1e8;
    cfg.time.n_points = 120;
    cfg.time.include_t0 = true;
    cfg.space.nx = 320;
    cfg.space.auto_x_max = true;
    cfg.space.tail_tol = 1e-12;
    cfg.rel_dt_max = 0.05;
    cfg.steady_tol = 2e-8;
    cfg.steady_window = 20;
    return cfg;
}

void test_log_grid() {
    auto cfg = base_test_config();
    cfg.time.t_min = 1e-3;
    cfg.time.t_max = 1e3;
    cfg.time.n_points = 50;
    cfg.time.include_t0 = false;

    const wck::SolveResult result = wck::solve_w_grid(wck::WckParams{0.0, 1}, cfg);
    expect(result.points.size() == static_cast<std::size_t>(cfg.time.n_points),
           "log-grid test: unexpected point count");

    const auto& pts = result.points;
    for (std::size_t i = 1; i < pts.size(); ++i) {
        expect(pts[i].t > pts[i - 1].t, "log-grid test: time grid is not strictly increasing");
    }

    const double step0 = std::log(pts[1].t) - std::log(pts[0].t);
    for (std::size_t i = 2; i < pts.size(); ++i) {
        const double step = std::log(pts[i].t) - std::log(pts[i - 1].t);
        expect(std::abs(step - step0) < 1e-10, "log-grid test: unequal spacing on log scale");
    }
}

void test_stationary_mass() {
    wck::SpaceGridConfig space{};
    space.nx = 1000;
    space.auto_x_max = true;
    space.tail_tol = 1e-14;

    double x_max_used = 0.0;
    const double mass = wck::estimate_stationary_mass(wck::WckParams{1.0, 2}, space, &x_max_used);
    expect(std::abs(mass - 1.0) < 2e-6, "stationary mass test: density mass not near 1");
    expect(x_max_used > 0.0, "stationary mass test: invalid x_max");
}

void test_pde_invariants_and_scenarios() {
    auto cfg = base_test_config();
    const std::vector<wck::WckParams> scenarios{
        {-2.0, 1},
        {0.0, 1},
        {2.0, 1},
        {0.0, 2},
        {3.0, 2},
        {20.0, 3},
    };

    for (const auto& params : scenarios) {
        const wck::SolveResult result = wck::solve_w_grid(params, cfg);
        expect(!result.points.empty(), "invariants test: no output points");
        expect(result.points.front().t == 0.0, "invariants test: first time should be t=0");
        expect(std::abs(result.points.front().w - 1.0) < 1e-12,
               "invariants test: w(0) must be 1");

        for (std::size_t i = 0; i < result.points.size(); ++i) {
            const auto& point = result.points[i];
            expect(std::isfinite(point.w), "invariants test: w is not finite");
            expect(std::isfinite(point.m_term), "invariants test: m_term is not finite");
            expect(std::isfinite(point.var_h_term), "invariants test: var_h_term is not finite");
            expect(point.w >= -1e-10, "invariants test: w below 0");
            expect(point.m_term >= -1e-10, "invariants test: m_term below 0");
            expect(point.var_h_term >= -1e-10, "invariants test: var_h_term below 0");
            expect(std::abs(point.w - (point.m_term + point.var_h_term)) < 2e-7,
                   "invariants test: w decomposition mismatch");
        }
    }
}

void test_upwind_stability() {
    // Large |b(x)| dx used to flip the upwind stencil anti-diffusive: h blew
    // up in the density tail, Var_pi[h] went NaN and was clamped to 0,
    // putting a step discontinuity into w at small t. k=3 exercises the
    // b < 0 branch (steep tail drift); the coarse-grid k=1, c=20 case
    // exercises the b >= 0 branch near x=0 (b dx up to ~5.6 at nx=100).
    auto coarse = base_test_config();
    coarse.space.nx = 100;
    const std::vector<std::pair<wck::WckParams, wck::SolverConfig>> cases{
        {{5.0, 3}, base_test_config()},
        {{20.0, 3}, base_test_config()},
        {{20.0, 1}, coarse},
    };

    for (const auto& [params, cfg] : cases) {
        const wck::SolveResult result = wck::solve_w_grid(params, cfg);
        double prev_var_h = 0.0;
        for (const auto& point : result.points) {
            if (point.t <= 0.0) {
                continue;
            }
            expect(point.var_h_term > 0.0,
                   "upwind stability test: var_h_term must stay strictly positive");
            expect(point.var_h_term > 1e-3 * prev_var_h,
                   "upwind stability test: var_h_term collapsed between grid points");
            prev_var_h = point.var_h_term;
        }
    }
}

void test_convergence_consistency() {
    auto coarse = base_test_config();
    coarse.time.t_max = 1e5;
    coarse.time.n_points = 100;
    coarse.space.nx = 400;

    auto fine = coarse;
    fine.space.nx = 800;

    const auto params = wck::WckParams{1.0, 2};
    const wck::SolveResult r_coarse = wck::solve_w_grid(params, coarse);
    const wck::SolveResult r_fine = wck::solve_w_grid(params, fine);

    expect(r_coarse.points.size() == r_fine.points.size(),
           "convergence test: mismatched grid size");

    double max_rel = 0.0;
    for (std::size_t i = 1; i < r_coarse.points.size(); ++i) {
        const double t = r_fine.points[i].t;
        if (t < 1e-2) {
            continue;
        }
        const double rel = relative_error(r_coarse.points[i].w, r_fine.points[i].w);
        max_rel = std::max(max_rel, rel);
    }
    expect(max_rel < 0.08, "convergence test: coarse vs fine mismatch too large");
}

struct CsvRow {
    double t = 0.0;
    double w = 0.0;
    double m_term = 0.0;
    double var_h_term = 0.0;
};

CsvRow parse_csv_line(const std::string& line) {
    std::vector<double> fields{};
    std::size_t start = 0U;
    while (start <= line.size()) {
        const std::size_t comma = line.find(',', start);
        const std::size_t end = (comma == std::string::npos) ? line.size() : comma;
        fields.push_back(std::stod(line.substr(start, end - start)));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1U;
    }
    if (fields.size() != 4U) {
        throw std::runtime_error("csv parse error: expected 4 fields");
    }
    return CsvRow{fields[0], fields[1], fields[2], fields[3]};
}

void test_cli_output() {
    const std::filesystem::path out_path =
        std::filesystem::temp_directory_path() / "wck_cli_test_output.csv";
    std::error_code ec;
    std::filesystem::remove(out_path, ec);

    const std::string cmd =
        std::string("\"") + WCK_CLI_PATH + "\""
        + " --c 0 --k 1 --t-min 1e-4 --t-max 1e2 --n-t 30 --nx 220 --out \""
        + out_path.string() + "\"";

    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "cli test: command failed");

    expect(std::filesystem::exists(out_path), "cli test: output file missing");

    std::ifstream in(out_path);
    expect(in.is_open(), "cli test: failed to open output csv");

    std::vector<std::string> lines{};
    std::string line{};
    while (std::getline(in, line)) {
        lines.push_back(line);
    }

    expect(!lines.empty(), "cli test: empty output csv");
    expect(lines[0] == "t,w,m_term,var_h_term", "cli test: header mismatch");
    expect(lines.size() == 32U, "cli test: unexpected line count");

    const CsvRow first_row = parse_csv_line(lines[1]);
    expect(std::abs(first_row.t) < std::numeric_limits<double>::epsilon(), "cli test: first row t is not 0");
    expect(std::abs(first_row.w - 1.0) < 1e-12, "cli test: first row w is not 1");
    expect(std::abs(first_row.m_term - 1.0) < 1e-12, "cli test: first row m_term is not 1");
    expect(std::abs(first_row.var_h_term) < 1e-12, "cli test: first row var_h_term is not 0");
}

void test_cli_default_naming_rule() {
    const std::filesystem::path run_dir =
        std::filesystem::temp_directory_path() / "wck_cli_default_name_test_dir";
    std::error_code ec;
    std::filesystem::remove_all(run_dir, ec);
    std::filesystem::create_directories(run_dir, ec);
    expect(!ec, "cli default-name test: failed to create temp directory");

    const std::filesystem::path expected_csv = run_dir / "results" / "wck_c0.3_k2.csv";
    std::filesystem::remove(expected_csv, ec);

    const std::string cmd =
        std::string("cd \"") + run_dir.string() + "\" && \""
        + WCK_CLI_PATH
        + "\" --c 0.3 --k 2 --t-min 1e-4 --t-max 1e2 --n-t 30 --nx 220";

    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "cli default-name test: command failed");
    expect(std::filesystem::exists(expected_csv), "cli default-name test: expected csv missing");

    std::ifstream in(expected_csv);
    expect(in.is_open(), "cli default-name test: failed to open output csv");
    std::string header{};
    std::getline(in, header);
    expect(header == "t,w,m_term,var_h_term", "cli default-name test: header mismatch");
}

void test_cli_custom_dir_default_name() {
    const std::filesystem::path run_dir =
        std::filesystem::temp_directory_path() / "wck_cli_custom_dir_test_dir";
    std::error_code ec;
    std::filesystem::remove_all(run_dir, ec);
    std::filesystem::create_directories(run_dir, ec);
    expect(!ec, "cli custom-dir test: failed to create temp directory");

    const std::filesystem::path target_dir = run_dir / "my_outputs";
    const std::filesystem::path expected_csv = target_dir / "wck_c-2_k1.csv";
    std::filesystem::remove(expected_csv, ec);

    const std::string cmd =
        std::string("cd \"") + run_dir.string() + "\" && \""
        + WCK_CLI_PATH
        + "\" --c -2 --k 1 --dir \""
        + target_dir.string()
        + "\" --t-min 1e-4 --t-max 1e2 --n-t 30 --nx 220";

    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "cli custom-dir test: command failed");
    expect(std::filesystem::exists(expected_csv), "cli custom-dir test: expected csv missing");
}

bool looks_like_intermediate_wck_file(std::string_view filename) {
    return filename.rfind("wck_c", 0U) == 0U
        && filename.find("_k") != std::string_view::npos
        && filename.size() >= 4U
        && filename.substr(filename.size() - 4U) == ".csv";
}

void test_sweep_cli_matrix_only_no_intermediate_files() {
    const std::filesystem::path run_dir =
        std::filesystem::temp_directory_path() / "wck_sweep_cli_test_dir";
    std::error_code ec;
    std::filesystem::remove_all(run_dir, ec);
    std::filesystem::create_directories(run_dir, ec);
    expect(!ec, "sweep cli test: failed to create temp directory");

    const std::filesystem::path out_dir = run_dir / "tables";
    const std::filesystem::path expected_csv = out_dir / "w_table_matrix_k2.csv";

    const std::string cmd =
        std::string("cd \"") + run_dir.string() + "\" && \""
        + WCK_SWEEP_CLI_PATH
        + "\" --k 2 --c-min -0.2 --c-max 0.2 --dc 0.2 --n-t 8 --t-max 1e2 --nx 140 --dir \""
        + out_dir.string() + "\"";

    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "sweep cli test: command failed");
    expect(std::filesystem::exists(expected_csv), "sweep cli test: matrix output missing");

    std::ifstream in(expected_csv);
    expect(in.is_open(), "sweep cli test: failed to open matrix csv");
    std::string header{};
    std::getline(in, header);
    expect(!header.empty(), "sweep cli test: empty matrix header");
    expect(header.rfind("c,", 0U) == 0U, "sweep cli test: header should start with c,");

    int intermediate_count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(run_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (looks_like_intermediate_wck_file(name)) {
            ++intermediate_count;
        }
    }
    expect(intermediate_count == 0, "sweep cli test: found intermediate wck_c*_k*.csv files");
}

}  // namespace

int main() {
    try {
        test_log_grid();
        test_stationary_mass();
        test_pde_invariants_and_scenarios();
        test_upwind_stability();
        test_convergence_consistency();
        test_cli_output();
        test_cli_default_naming_rule();
        test_cli_custom_dir_default_name();
        test_sweep_cli_matrix_only_no_intermediate_files();
        run_effective_idw_sim_tests();
        run_b_calibration_tests();
        run_workload_sim_tests();
        run_tandem_workload_sim_tests();
    } catch (const std::exception& ex) {
        std::cerr << "Test failure: " << ex.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";
    return 0;
}
