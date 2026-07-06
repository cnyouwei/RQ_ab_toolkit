#include "wck/rq_calibration/b_calibration.hpp"
#include "wck/rq_calibration/w_table_interpolator.hpp"

#include "support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path write_w_table_matrix_csv(
    const std::filesystem::path& dir,
    int k,
    const std::vector<double>& c_values) {
    const std::vector<double> t_values{0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0};
    const std::filesystem::path path = dir / ("w_table_matrix_k" + std::to_string(k) + ".csv");

    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to create w-table CSV: " + path.string());
    }

    out << std::setprecision(17) << std::scientific;
    out << "c";
    for (double t : t_values) {
        out << "," << t;
    }
    out << '\n';

    for (double c : c_values) {
        out << c;
        for (double t : t_values) {
            double w = 1.0;
            if (t > 0.0) {
                const double c_factor = std::exp(-0.04 * std::max(c, 0.0));
                const double t_factor = std::exp(-0.06 * std::log1p(t));
                w = c_factor * t_factor;
            }
            out << "," << w;
        }
        out << '\n';
    }
    out.flush();
    if (!out.good()) {
        throw std::runtime_error("failed while writing w-table CSV: " + path.string());
    }
    return path;
}

double normal_pdf(double x) {
    static constexpr double inv_sqrt_2pi = 0.39894228040143267794;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

double normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double exact_rhs_sup_from_row(const wck::BCalibrationRow& row, const wck::WTableInterpolator& w_table, int k, double beta, double tau) {
    const double c_scale = std::pow(beta, -1.0 / static_cast<double>(k + 1));
    const double tilde_c = row.c * c_scale;
    const double a_psi = row.a_psi;
    const double b = row.b;

    auto objective = [&](double log_u) -> double {
        const double u = std::exp(log_u);
        const double variance_term = 2.0 * w_table.w(tilde_c, tau * u) * u;
        if (!(variance_term > 0.0) || !std::isfinite(variance_term)) {
            return -std::numeric_limits<double>::infinity();
        }
        return a_psi * u + b * std::sqrt(variance_term);
    };

    constexpr int n_points = 5001;
    constexpr double log_u_min = -20.0;
    constexpr double log_u_max = 20.0;
    double best_val = 0.0;
    double best_log_u = log_u_min;
    for (int i = 0; i < n_points; ++i) {
        const double ratio = static_cast<double>(i) / static_cast<double>(n_points - 1);
        const double log_u = log_u_min + (log_u_max - log_u_min) * ratio;
        const double val = objective(log_u);
        if (val > best_val) {
            best_val = val;
            best_log_u = log_u;
        }
    }

    double left = std::max(log_u_min, best_log_u - 1.5);
    double right = std::min(log_u_max, best_log_u + 1.5);
    if (!(right > left)) {
        return best_val;
    }

    const double inv_phi = (std::sqrt(5.0) - 1.0) / 2.0;
    double c1 = right - inv_phi * (right - left);
    double c2 = left + inv_phi * (right - left);
    double f1 = objective(c1);
    double f2 = objective(c2);

    for (int iter = 0; iter < 120; ++iter) {
        if (f1 > f2) {
            right = c2;
            c2 = c1;
            f2 = f1;
            c1 = right - inv_phi * (right - left);
            f1 = objective(c1);
        } else {
            left = c1;
            c1 = c2;
            f1 = f2;
            c2 = left + inv_phi * (right - left);
            f2 = objective(c2);
        }
    }

    return std::max(best_val, objective(0.5 * (left + right)));
}

void test_psi_against_mm1m_closed_form() {
    const std::vector<double> c_values{-2.0, 0.0, 2.0};
    for (double c : c_values) {
        const double psi_numeric = wck::exact_psi_target(1, c);
        const double upper_tail = 1.0 - normal_cdf(-c);
        const double psi_closed = c + normal_pdf(-c) / upper_tail;
        expect_close(
            psi_numeric,
            psi_closed,
            1e-8,
            "psi check failed for k=1 at c=" + std::to_string(c));
    }
}

void test_exact_calibration_residual() {
    const auto dir = make_temp_dir("wck_b_calib_exact");
    const auto w_table_path = write_w_table_matrix_csv(dir, 1, {-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0});
    const wck::WTableInterpolator w_table = wck::WTableInterpolator::from_matrix_csv(w_table_path);

    const std::vector<double> c_grid{-3.0, -1.0, 1.0};
    const wck::BCalibrationResult result = wck::calibrate_b_table(1, c_grid, w_table, 1U);
    expect(result.rows.size() == c_grid.size(), "exact residual test: row count mismatch");

    for (const auto& row : result.rows) {
        expect(
            row.status == wck::BCalibrationStatus::kExact,
            "exact residual test: expected exact status");
        const double rhs_sup = exact_rhs_sup_from_row(row, w_table, 1, result.beta, result.tau);
        const double residual = std::abs(rhs_sup - row.psi);
        expect(residual < 1e-6, "exact residual test: residual too large");
    }
}

void test_infeasible_k_gt_1_uses_zero_fallback() {
    const auto dir = make_temp_dir("wck_b_calib_zero_fallback");
    const auto w_table_path = write_w_table_matrix_csv(dir, 2, {-6.0, -3.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0});
    const wck::WTableInterpolator w_table = wck::WTableInterpolator::from_matrix_csv(w_table_path);

    const wck::BCalibrationResult result = wck::calibrate_b_table(2, std::vector<double>{2.0}, w_table, 1U);
    expect(result.rows.size() == 1U, "zero fallback test: expected one row");

    const auto& row = result.rows.front();
    expect(row.status == wck::BCalibrationStatus::kExact, "zero fallback test: expected exact status");
    expect_close(row.b, 0.0, 1e-12, "zero fallback test: expected b=0");
    expect(std::isfinite(row.psi), "zero fallback test: psi should be finite");
    expect(std::isfinite(row.z_model), "zero fallback test: z_model should be finite");
    expect(std::isfinite(row.abs_error), "zero fallback test: abs_error should be finite");
    expect(!std::isfinite(row.u_star), "zero fallback test: u_star should be nan");
}

void test_value_is_capped_at_sqrt2_for_large_c() {
    const auto dir = make_temp_dir("wck_b_calib_cap_sqrt2");
    const auto w_table_path = write_w_table_matrix_csv(dir, 1, {-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0});
    const wck::WTableInterpolator w_table = wck::WTableInterpolator::from_matrix_csv(w_table_path);

    // c=8: closed-form a_psi = -phi(c)/Phi(c) ~ -5e-15 is above the -1e-12
    // infeasibility threshold, so the k=1 fallback b=sqrt(2) applies. (At
    // c=7, a_psi ~ -9.1e-12 still selects the exact branch, whose b depends
    // on the toy w-table.)
    const wck::BCalibrationResult result = wck::calibrate_b_table(1, std::vector<double>{8.0}, w_table, 1U);
    expect(result.rows.size() == 1U, "cap test: expected one row");
    const auto& row = result.rows.front();
    expect(row.status == wck::BCalibrationStatus::kExact, "cap test: expected exact status");
    expect_close(row.b, std::sqrt(2.0), 1e-12, "cap test: expected capped b=sqrt(2)");
    expect(!std::isfinite(row.u_star), "cap test: expected u_star to be nan for capped row");
}

void test_infeasible_k_eq_1_uses_sqrt2_fallback() {
    const auto dir = make_temp_dir("wck_b_calib_sqrt2_fallback_k1");
    const auto w_table_path = write_w_table_matrix_csv(dir, 1, {-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0});
    const wck::WTableInterpolator w_table = wck::WTableInterpolator::from_matrix_csv(w_table_path);

    const wck::BCalibrationResult result = wck::calibrate_b_table(1, std::vector<double>{20.0}, w_table, 1U);
    expect(result.rows.size() == 1U, "k1 fallback test: expected one row");
    const auto& row = result.rows.front();
    expect(row.status == wck::BCalibrationStatus::kExact, "k1 fallback test: expected exact status");
    expect_close(row.b, std::sqrt(2.0), 1e-12, "k1 fallback test: expected b=sqrt(2)");
    expect(!std::isfinite(row.u_star), "k1 fallback test: expected u_star to be nan");
}

void test_csv_roundtrip_and_lookup() {
    const auto dir = make_temp_dir("wck_b_calib_roundtrip");
    const auto w_table_path = write_w_table_matrix_csv(dir, 1, {-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0});
    const wck::WTableInterpolator w_table = wck::WTableInterpolator::from_matrix_csv(w_table_path);

    const std::vector<double> c_grid{-2.0, -1.0, 0.0};
    const wck::BCalibrationResult result = wck::calibrate_b_table(1, c_grid, w_table, 1U);

    const std::filesystem::path csv_path = dir / "b_table_k1.csv";
    wck::write_b_calibration_table_csv(csv_path, result);
    const wck::BCalibrationTable loaded = wck::load_b_calibration_table_csv(csv_path);

    for (const auto& row : result.rows) {
        const double value = loaded.evaluate(row.c);
        expect_close(value, row.b, 1e-12, "roundtrip test: gridpoint evaluation mismatch");
    }

    const double mid = 0.5 * (result.rows[0].c + result.rows[1].c);
    const double expected_mid = 0.5 * (result.rows[0].b + result.rows[1].b);
    const double mid_val = loaded.evaluate(mid);
    expect_close(mid_val, expected_mid, 1e-12, "roundtrip test: midpoint interpolation mismatch");

    const double left_val = loaded.evaluate(-1e6);
    expect_close(left_val, std::sqrt(2.0), 1e-12, "roundtrip test: left extrapolation mismatch");

    const double right_val = loaded.evaluate(1e6);
    expect_close(right_val, result.rows.back().b, 1e-12, "roundtrip test: right extrapolation mismatch");
}

void test_calibration_cli_smoke() {
    const auto dir = make_temp_dir("wck_b_calib_cli");
    const auto w_table_path = write_w_table_matrix_csv(dir, 1, {-1.0, 0.0, 1.0});
    const std::filesystem::path out_path = dir / "b_cli_output.csv";

    const std::string cmd =
        std::string("\"") + WCK_CALIBRATE_B_CLI_PATH + "\""
        + " --k 1 --w-table \"" + w_table_path.string() + "\""
        + " --out \"" + out_path.string() + "\""
        + " --jobs 1";

    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "cli smoke test: command failed");
    expect(std::filesystem::exists(out_path), "cli smoke test: output file missing");

    std::ifstream in(out_path);
    expect(in.is_open(), "cli smoke test: failed to open output CSV");

    std::vector<std::string> lines{};
    std::string line{};
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    expect(!lines.empty(), "cli smoke test: empty output CSV");
    expect(
        lines[0] == "c,b,status,psi,z_model,abs_error,a_psi,u_star",
        "cli smoke test: header mismatch");
    expect(lines.size() == 4U, "cli smoke test: unexpected row count");
}

}  // namespace

void run_b_calibration_tests() {
    test_psi_against_mm1m_closed_form();
    test_exact_calibration_residual();
    test_infeasible_k_gt_1_uses_zero_fallback();
    test_value_is_capped_at_sqrt2_for_large_c();
    test_infeasible_k_eq_1_uses_sqrt2_fallback();
    test_csv_roundtrip_and_lookup();
    test_calibration_cli_smoke();
}
