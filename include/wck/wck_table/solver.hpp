#pragma once

#include "wck/wck_table/types.hpp"

#include <vector>

namespace wck {

SolveResult solve_w_grid(const WckParams& params, const SolverConfig& config);

std::vector<SolveResult> solve_many(
    const std::vector<WckParams>& params_list,
    const SolverConfig& config);

// Returns trapezoidal mass integral of the stationary density on the truncated domain.
double estimate_stationary_mass(
    const WckParams& params,
    const SpaceGridConfig& config,
    double* x_max_used = nullptr);

}  // namespace wck

