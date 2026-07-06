#pragma once

#include <vector>

namespace wck {

// Solves a tridiagonal linear system in-place on rhs using Thomas algorithm.
// lower[0] and upper[n-1] are ignored. All vectors must have identical length.
void solve_tridiagonal(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    std::vector<double>& rhs);

}  // namespace wck

