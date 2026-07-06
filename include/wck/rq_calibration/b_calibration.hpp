#pragma once

#include "wck/rq_calibration/w_table_interpolator.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace wck {

enum class BCalibrationStatus {
    kExact,
    kBestFit,
};

const char* b_calibration_status_name(BCalibrationStatus status);
BCalibrationStatus parse_b_calibration_status(const std::string& text);

struct BCalibrationRow {
    double c = 0.0;
    double b = 0.0;
    BCalibrationStatus status = BCalibrationStatus::kExact;
    double psi = 0.0;
    double z_model = 0.0;
    double abs_error = 0.0;
    double a_psi = 0.0;
    double u_star = 0.0;
};

struct BCalibrationResult {
    int k = 1;
    double beta = 1.0;
    double tau = 1.0;
    std::vector<BCalibrationRow> rows{};
};

std::vector<double> build_c_grid(double c_min, double c_max, double dc);

// Exact heavy-traffic target in eq. (HT_exact) for gamma=h, mu=1:
// psi(c,k) = I1(c,k) / I0(c,k), where Ij integrate over [0,inf).
double exact_psi_target(int k, double c);

BCalibrationResult calibrate_b_table(
    int k,
    const std::vector<double>& c_grid,
    const WTableInterpolator& w_table,
    std::size_t thread_count = 0U);

void write_b_calibration_table_csv(const std::filesystem::path& path, const BCalibrationResult& result);

struct BCalibrationTable {
    std::vector<BCalibrationRow> rows{};

    double evaluate(double c) const;
};

BCalibrationTable load_b_calibration_table_csv(const std::filesystem::path& path);

}  // namespace wck
