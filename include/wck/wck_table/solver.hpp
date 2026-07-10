#pragma once

#include "wck/wck_table/types.hpp"

#include <vector>

namespace wck {

SolveResult solve_w_grid(const WckParams& params, const SolverConfig& config);

std::vector<SolveResult> solve_many(
    const std::vector<WckParams>& params_list,
    const SolverConfig& config);

}  // namespace wck
