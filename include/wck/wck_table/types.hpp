#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace wck {

struct WckParams {
    double c = 0.0;
    int k = 1;
};

struct TimeGridConfig {
    double t_min = 1e-4;
    double t_max = 1e8;
    int n_points = 1200;
    bool include_t0 = true;
};

struct SpaceGridConfig {
    int nx = 2000;
    double x_max = 0.0;
    bool auto_x_max = true;
    double tail_tol = 1e-14;
};

struct SolverConfig {
    TimeGridConfig time{};
    SpaceGridConfig space{};
    double rel_dt_max = 0.03;
    double steady_tol = 1e-9;
    int steady_window = 25;
    std::size_t thread_count = 0;
};

struct GridPoint {
    double t = 0.0;
    double w = 1.0;
    double m_term = 1.0;
    double var_h_term = 0.0;
};

struct SolveResult {
    std::vector<GridPoint> points{};
    double w_infty_est = std::numeric_limits<double>::quiet_NaN();
    double x_max_used = 0.0;
    int steps = 0;
};

}  // namespace wck
