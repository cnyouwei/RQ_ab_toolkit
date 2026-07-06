"""First (crude) RQ approximation of RQ_ab.tex (eq:RQ_ab_1).

    Z = sup_{u >= 0} { rho*u - u/barF_alpha(Z) + b*sqrt((rho*u/mu) * I_w(lambda*u)) },

with I_w(lambda*u) = I_a(u) + c_s^2 the crude IDW (eq:IDW_first_RQ); for
tandem models I_a is the queue-1 departure IDC.  The robustness parameter b
is calibrated per tuple by the closed-form rule (eq:b):

    b(c) = sqrt( 2*| -c*psi + beta*psi^(k+1) | ),   c = (rho - 1) / alpha^h,

where psi is the exact critically-loaded heavy-traffic constant (eq:HT_exact,
mu = 1), evaluated by adaptive Simpson quadrature after localizing the
integrand around its mode.  No w/b tables are required.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

from .fixed_point import (
    A_INF_TOL,
    BisectOptions,
    FixedPointResult,
    solve_fixed_point,
    survival_alpha,
)
from .idc import arrival_idc_curve_for
from .models import AnyModel, BaseSystemStats
from .util import require_float, require_int, require_str

CSV_COLUMNS = [
    "tuple_id",
    "lambda",
    "alpha",
    "lambda_k",
    "lambda_form",
    "alpha_k",
    "z_rq_first",
    "c",
    "b",
    "psi",
    "k",
    "h",
    "rho",
    "mu",
    "beta_patience",
    "c_a2",
    "c_s2",
    "survival_at_solution",
    "rhs_at_solution",
    "solver_status",
    "bisect_iters",
    "bracket_lo",
    "bracket_hi",
    "model_name",
    "model_alias",
]

PSI_SIMPSON_EPS = 1e-12
PSI_SIMPSON_MAX_DEPTH = 40
PSI_LOG_TAIL_CUTOFF = 80.0


def adaptive_simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float,
    max_depth: int,
) -> float:
    fa = f(a)
    fb = f(b)
    c = 0.5 * (a + b)
    fc = f(c)
    s = (b - a) * (fa + 4.0 * fc + fb) / 6.0

    def recurse(l: float, r: float, f_l: float, f_r: float, f_m: float, s_lr: float, depth: int) -> float:
        m = 0.5 * (l + r)
        lm = 0.5 * (l + m)
        mr = 0.5 * (m + r)
        f_lm = f(lm)
        f_mr = f(mr)
        s_left = (m - l) * (f_l + 4.0 * f_lm + f_m) / 6.0
        s_right = (r - m) * (f_m + 4.0 * f_mr + f_r) / 6.0
        s2 = s_left + s_right
        if depth <= 0 or abs(s2 - s_lr) <= 15.0 * eps:
            return s2 + (s2 - s_lr) / 15.0
        return recurse(l, m, f_l, f_m, f_lm, s_left, depth - 1) + recurse(
            m, r, f_m, f_r, f_mr, s_right, depth - 1
        )

    return recurse(a, b, fa, fb, fc, s, max_depth)


def psi_critical(c: float, k: int, beta_patience: float) -> float:
    """Exact psi from eq:HT_exact with gamma = h and mu = 1.

    psi = E[X] for the density proportional to exp(g(x)) on [0, inf) with
    g(x) = c*x - beta*x^(k+1)/(k+1).  The integration window is localized
    around the maximizer of g so the computation stays accurate for large |c|.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (beta_patience > 0.0 and math.isfinite(beta_patience)):
        raise ValueError("beta_patience must be finite and > 0")
    if not math.isfinite(c):
        raise ValueError("c must be finite")

    beta = float(beta_patience)
    kp1 = float(k + 1)

    def g(x: float) -> float:
        return c * x - beta * (x**kp1) / kp1

    # g'(x) = c - beta*x^k: interior mode for c > 0, boundary mode at 0 otherwise.
    if c > 0.0:
        x_mode = (c / beta) ** (1.0 / float(k))
    else:
        x_mode = 0.0
    g_max = g(x_mode)

    def g_shift(x: float) -> float:
        return g(x) - g_max

    # Right endpoint: expand then bisect to where the integrand is negligible.
    step = max(x_mode, 1.0, beta ** (-1.0 / kp1))
    hi = x_mode + step
    for _ in range(200):
        if g_shift(hi) < -PSI_LOG_TAIL_CUTOFF:
            break
        hi = x_mode + (hi - x_mode) * 2.0
    else:
        raise RuntimeError("failed to localize right tail of psi integrand")
    lo_r = x_mode
    hi_r = hi
    for _ in range(200):
        mid = 0.5 * (lo_r + hi_r)
        if g_shift(mid) < -PSI_LOG_TAIL_CUTOFF:
            hi_r = mid
        else:
            lo_r = mid
        if hi_r - lo_r <= 1e-12 * max(1.0, hi_r):
            break
    x_hi = hi_r

    # Left endpoint: 0 unless the integrand at 0 is already negligible.
    x_lo = 0.0
    if x_mode > 0.0 and g_shift(0.0) < -PSI_LOG_TAIL_CUTOFF:
        lo_l = 0.0
        hi_l = x_mode
        for _ in range(200):
            mid = 0.5 * (lo_l + hi_l)
            if g_shift(mid) < -PSI_LOG_TAIL_CUTOFF:
                lo_l = mid
            else:
                hi_l = mid
            if hi_l - lo_l <= 1e-12 * max(1.0, hi_l):
                break
        x_lo = lo_l

    if not (x_hi > x_lo):
        raise RuntimeError("degenerate psi integration window")

    def exp_term(x: float) -> float:
        e = g_shift(x)
        if e < -745.0:
            return 0.0
        return math.exp(e)

    width = x_hi - x_lo
    eps0 = PSI_SIMPSON_EPS * width
    eps1 = PSI_SIMPSON_EPS * width * max(1.0, x_hi)
    i0 = adaptive_simpson(exp_term, x_lo, x_hi, eps0, PSI_SIMPSON_MAX_DEPTH)
    i1 = adaptive_simpson(lambda x: x * exp_term(x), x_lo, x_hi, eps1, PSI_SIMPSON_MAX_DEPTH)
    if not (i0 > 0.0 and math.isfinite(i0) and math.isfinite(i1)):
        raise RuntimeError("failed to compute finite psi integrals")
    psi = i1 / i0
    if not (psi >= 0.0 and math.isfinite(psi)):
        raise RuntimeError("computed psi is invalid")
    return psi


def calibrate_b_first_rq(c: float, k: int, beta_patience: float) -> tuple[float, float]:
    """Closed-form calibration (eq:b): b(c) = sqrt(2*|-c*psi + beta*psi^(k+1)|)."""
    psi = psi_critical(c=c, k=k, beta_patience=beta_patience)
    value = -c * psi + beta_patience * (psi ** float(k + 1))
    b = math.sqrt(2.0 * abs(value))
    if not (b >= 0.0 and math.isfinite(b)):
        raise RuntimeError("calibrated b is invalid")
    return b, psi


@dataclass(frozen=True)
class FirstKernel:
    sqrt_q: list[float]
    tail_sqrt_q: float


class FirstSolver:
    """Per-model context for solving first-RQ tuples."""

    def __init__(
        self,
        model: AnyModel,
        base: BaseSystemStats,
        u_grid: list[float],
        b_override: float | None = None,
        bisect: BisectOptions | None = None,
    ) -> None:
        self.model = model
        self.base = base
        self.u_grid = u_grid
        self.b_override = b_override
        self.bisect = bisect if bisect is not None else BisectOptions()
        self.bisect.validate()

    def build_kernel(self, lam: float) -> FirstKernel:
        """Precompute sqrt(u * I_w(lambda*u)) with I_w = I_a + c_s^2."""
        ia = arrival_idc_curve_for(self.model, s_values=self.u_grid, lam=lam)
        if len(ia) != len(self.u_grid):
            raise RuntimeError("internal error: arrival IDC curve length mismatch")

        sqrt_q: list[float] = []
        for u, ia_u in zip(self.u_grid, ia):
            q = u * (float(ia_u) + self.base.c_s2)
            sqrt_q.append(math.sqrt(q) if q > 0.0 else 0.0)
        return FirstKernel(sqrt_q=sqrt_q, tail_sqrt_q=sqrt_q[-1])

    def _rhs(self, z: float, lam: float, alpha: float, b: float, kernel: FirstKernel) -> tuple[float, float]:
        surv = survival_alpha(z=z, alpha=alpha, patience=self.model.patience)
        surv = min(1.0, max(0.0, surv))

        rho = lam / self.base.mu
        kappa = b * math.sqrt(lam) / self.base.mu

        if surv <= 0.0:
            return 0.0, surv

        a = rho - 1.0 / surv
        if a > A_INF_TOL:
            return float("inf"), surv
        if abs(a) <= A_INF_TOL and kappa > 0.0 and kernel.tail_sqrt_q > 0.0:
            return float("inf"), surv

        if kappa <= 0.0:
            return 0.0, surv

        best = 0.0
        for u, sq in zip(self.u_grid, kernel.sqrt_q):
            value = a * u + kappa * sq
            if value > best:
                best = value
        return best, surv

    def solve_one_tuple(self, row: dict[str, Any]) -> dict[str, Any]:
        lam = require_float(row, "lambda")
        alpha = require_float(row, "alpha")
        if lam <= 0.0:
            raise ValueError(f"invalid lambda={lam}; expected > 0")
        if alpha <= 0.0:
            raise ValueError(f"invalid alpha={alpha}; expected > 0")

        base = self.base
        rho = lam / base.mu
        c = (rho - 1.0) / (alpha**base.h)
        if self.b_override is not None:
            b = float(self.b_override)
            psi = float("nan")
        else:
            b, psi = calibrate_b_first_rq(c=c, k=base.k, beta_patience=base.beta_patience)

        kernel = self.build_kernel(lam=lam)
        solution: FixedPointResult = solve_fixed_point(
            lambda z: self._rhs(z, lam=lam, alpha=alpha, b=b, kernel=kernel),
            self.bisect,
        )

        return {
            "tuple_id": require_int(row, "tuple_id"),
            "lambda": lam,
            "alpha": alpha,
            "lambda_k": require_int(row, "lambda_k"),
            "lambda_form": require_str(row, "lambda_form"),
            "alpha_k": require_int(row, "alpha_k"),
            "z_rq_first": solution.z,
            "c": c,
            "b": b,
            "psi": psi,
            "k": base.k,
            "h": base.h,
            "rho": rho,
            "mu": base.mu,
            "beta_patience": base.beta_patience,
            "c_a2": base.c_a2,
            "c_s2": base.c_s2,
            "survival_at_solution": solution.survival_at_solution,
            "rhs_at_solution": solution.rhs_at_solution,
            "solver_status": solution.status,
            "bisect_iters": solution.bisect_iters,
            "bracket_lo": solution.bracket_lo,
            "bracket_hi": solution.bracket_hi,
            "model_name": self.model.model_name,
            "model_alias": self.model.model_alias,
        }


def default_out_csv(results_dir: Path, model_alias: str) -> Path:
    return results_dir / f"first_rq_grid_{model_alias}.csv"
