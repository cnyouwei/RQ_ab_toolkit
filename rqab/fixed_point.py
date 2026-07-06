"""Shared fixed-point machinery for the RQ approximations.

Both RQ algorithms solve z = RHS(z) where RHS is nonincreasing in z (through
the patience survival term), by bracketing (doubling z_hi) and bisection.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from .models import DistributionComponent

A_INF_TOL = 1e-14

DEFAULT_BISECT_ABS_TOL = 1e-8
DEFAULT_BISECT_REL_TOL = 1e-8
DEFAULT_BISECT_MAX_ITERS = 200
DEFAULT_BRACKET_MAX_DOUBLINGS = 80


@dataclass(frozen=True)
class BisectOptions:
    abs_tol: float = DEFAULT_BISECT_ABS_TOL
    rel_tol: float = DEFAULT_BISECT_REL_TOL
    max_iters: int = DEFAULT_BISECT_MAX_ITERS
    bracket_max_doublings: int = DEFAULT_BRACKET_MAX_DOUBLINGS

    def validate(self) -> None:
        if not (self.abs_tol > 0.0):
            raise ValueError("bisect abs_tol must be > 0")
        if not (self.rel_tol > 0.0):
            raise ValueError("bisect rel_tol must be > 0")
        if self.max_iters < 1:
            raise ValueError("bisect max_iters must be >= 1")
        if self.bracket_max_doublings < 1:
            raise ValueError("bracket_max_doublings must be >= 1")


@dataclass(frozen=True)
class FixedPointResult:
    z: float
    rhs_at_solution: float
    survival_at_solution: float
    bisect_iters: int
    bracket_lo: float
    bracket_hi: float
    status: str


def survival_alpha(z: float, alpha: float, patience: DistributionComponent) -> float:
    """Survival Fbar(alpha * z) of the base (mean-scaled) patience distribution."""
    if z <= 0.0:
        return 1.0

    x = alpha * z
    if patience.family == "exponential":
        rate = float(patience.params["rate"])
        return math.exp(-rate * x)

    if patience.family == "erlang_k":
        m = int(patience.params["k"])
        rate = float(patience.params["rate"])
        y = rate * x
        term = 1.0
        total = 1.0
        for j in range(1, m):
            term *= y / float(j)
            total += term
        return math.exp(-y) * total

    p = float(patience.params["p"])
    rate1 = float(patience.params["rate1"])
    rate2 = float(patience.params["rate2"])
    return p * math.exp(-rate1 * x) + (1.0 - p) * math.exp(-rate2 * x)


RhsFn = Callable[[float], tuple[float, float]]
"""z -> (rhs, survival); rhs may be +inf when the drift is nonnegative."""


def solve_fixed_point(rhs_fn: RhsFn, options: BisectOptions) -> FixedPointResult:
    """Solve z = rhs_fn(z)[0] by doubling bracket + bisection."""

    def f(z: float) -> tuple[float, float, float]:
        rhs, surv = rhs_fn(z)
        if math.isfinite(rhs):
            return rhs - z, rhs, surv
        return float("inf"), rhs, surv

    f0, rhs0, surv0 = f(0.0)
    if math.isfinite(f0) and f0 <= 0.0:
        return FixedPointResult(
            z=0.0,
            rhs_at_solution=rhs0,
            survival_at_solution=surv0,
            bisect_iters=0,
            bracket_lo=0.0,
            bracket_hi=0.0,
            status="ok",
        )

    lo = 0.0
    hi = 1.0
    bracket_found = False
    for _ in range(options.bracket_max_doublings):
        f_hi, _, _ = f(hi)
        if math.isfinite(f_hi) and f_hi <= 0.0:
            bracket_found = True
            break
        hi *= 2.0
    if not bracket_found:
        raise RuntimeError(
            "failed to bracket fixed-point root within bracket-max-doublings; "
            f"last_hi={hi:.6g}"
        )

    iters = 0
    while iters < options.max_iters:
        iters += 1
        mid = 0.5 * (lo + hi)
        f_mid, _, _ = f(mid)
        if not math.isfinite(f_mid) or f_mid > 0.0:
            lo = mid
        else:
            hi = mid
        width = hi - lo
        threshold = options.abs_tol + options.rel_tol * max(1.0, 0.5 * (lo + hi))
        if width <= threshold:
            break

    z_sol = 0.5 * (lo + hi)
    rhs_sol, surv_sol = rhs_fn(z_sol)
    return FixedPointResult(
        z=z_sol,
        rhs_at_solution=rhs_sol,
        survival_at_solution=surv_sol,
        bisect_iters=iters,
        bracket_lo=lo,
        bracket_hi=hi,
        status="ok",
    )
