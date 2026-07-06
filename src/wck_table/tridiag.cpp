#include "wck/wck_table/tridiag.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace wck {

void solve_tridiagonal(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    std::vector<double>& rhs) {
    const std::size_t n = diag.size();
    if (lower.size() != n || upper.size() != n || rhs.size() != n) {
        throw std::invalid_argument("tridiagonal vectors must have equal size");
    }
    if (n == 0U) {
        return;
    }

    constexpr double kPivotTol = 1e-14;
    std::vector<double> c_prime(n, 0.0);
    std::vector<double> d_prime(n, 0.0);

    double pivot = diag[0];
    if (std::abs(pivot) < kPivotTol) {
        throw std::runtime_error("tridiagonal solver encountered near-zero pivot at row 0");
    }
    c_prime[0] = (n > 1U) ? upper[0] / pivot : 0.0;
    d_prime[0] = rhs[0] / pivot;

    for (std::size_t i = 1; i < n; ++i) {
        pivot = diag[i] - lower[i] * c_prime[i - 1U];
        if (std::abs(pivot) < kPivotTol) {
            throw std::runtime_error("tridiagonal solver encountered near-zero pivot");
        }
        c_prime[i] = (i + 1U < n) ? (upper[i] / pivot) : 0.0;
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1U]) / pivot;
    }

    rhs[n - 1U] = d_prime[n - 1U];
    for (std::size_t i = n - 1U; i > 0U; --i) {
        rhs[i - 1U] = d_prime[i - 1U] - c_prime[i - 1U] * rhs[i];
    }
}

}  // namespace wck

