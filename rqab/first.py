"""First (crude) RQ approximation of RQ_ab.tex (eq:RQ_ab_1).

    Z = sup_{u >= 0} { rho*u - u/barF_alpha(Z) + b*sqrt((rho*u/mu) * I_w(lambda*u)) },

with I_w(lambda*u) = I_a(u) + c_s^2 the crude IDW (eq:IDW_first_RQ); for
tandem models I_a is the queue-1 departure IDC.  The robustness parameter b
is calibrated per tuple at the standardized load coordinate

    q = c * ((c_a^2 + c_s^2)/(2*mu))^(-k/(k+1)) * beta^(-1/(k+1)),
    b_k(q)^2 = 2*m_k(q)*(m_k(q)^k - q),

where m_k(q) is the mean of the normalized critical diffusion.  Exact
matching is used only when the right-hand side is positive; otherwise b=0 is
the fluid-boundary fallback.  No w/b tables are required.
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
from .effective_idw import tau_tilde_c
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
    "tilde_c",
    "b",
    "psi",
    "calibration_status",
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
CALIBRATION_FEASIBILITY_REL_TOL = 1e-12

CALIBRATION_EXACT = "exact"
CALIBRATION_FLUID_FALLBACK = "fluid_fallback"
CALIBRATION_OVERRIDE = "override"

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_INVERSE_MILLS_CF_TERMS = 200


@dataclass(frozen=True)
class FirstCalibration:
    """Normalized first-RQ calibration result at one standardized load."""

    b: float
    standardized_mean: float
    status: str


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


def _truncated_normal_mean_and_log_gap(q: float) -> tuple[float, float]:
    """Return m_1(q) and log(m_1(q) - q) without tail cancellation."""
    if q < -8.0:
        # For x=-q, the inverse Mills ratio has the continued fraction
        #   phi(x)/Phi(-x) = x + 1/(x + 2/(x + 3/(x + ...))).
        # Evaluate the part after x directly so m_1(-x) is never formed by
        # subtracting two nearly equal numbers.
        x = -q
        tail = 0.0
        for numerator in range(_INVERSE_MILLS_CF_TERMS, 1, -1):
            tail = float(numerator) / (x + tail)
        mean = 1.0 / (x + tail)
        log_gap = math.log(x + mean)
        return mean, log_gap

    cdf = 0.5 * math.erfc(-q / _SQRT2)
    if not (cdf > 0.0 and math.isfinite(cdf)):
        raise RuntimeError("failed to evaluate the truncated-normal normalizer")
    log_gap = -0.5 * q * q - 0.5 * _LOG_2PI - math.log(cdf)
    gap = math.exp(log_gap)
    mean = q + gap
    if not (mean > 0.0 and math.isfinite(mean)):
        raise RuntimeError("computed standardized critical mean is invalid")
    return mean, log_gap


def _standardized_critical_mean_quadrature(q: float, k: int) -> float:
    """Compute m_k(q) for k>1 using a moment centered at the density mode."""
    kp1 = k + 1

    def g(x: float) -> float:
        return q * x - (x**kp1) / float(kp1)

    # g'(x) = q - x^k: interior mode for q > 0, boundary mode otherwise.
    if q > 0.0:
        x_mode = q ** (1.0 / float(k))
    else:
        x_mode = 0.0
    g_max = g(x_mode)

    def g_shift(x: float) -> float:
        return g(x) - g_max

    # Expand and then bisect to a right endpoint with negligible mass.
    step = max(x_mode, 1.0)
    hi = x_mode + step
    for _ in range(200):
        if g_shift(hi) < -PSI_LOG_TAIL_CUTOFF:
            break
        hi = x_mode + (hi - x_mode) * 2.0
    else:
        raise RuntimeError("failed to localize right tail of critical-mean integrand")
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

    # Use zero unless the left tail is already negligible before the boundary.
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
        raise RuntimeError("degenerate critical-mean integration window")

    def exp_term(x: float) -> float:
        exponent = g_shift(x)
        if exponent < -745.0:
            return 0.0
        return math.exp(exponent)

    width = x_hi - x_lo
    eps0 = PSI_SIMPSON_EPS * width
    eps1 = PSI_SIMPSON_EPS * width * width
    i0 = adaptive_simpson(exp_term, x_lo, x_hi, eps0, PSI_SIMPSON_MAX_DEPTH)
    i1 = adaptive_simpson(
        lambda x: (x - x_mode) * exp_term(x),
        x_lo,
        x_hi,
        eps1,
        PSI_SIMPSON_MAX_DEPTH,
    )
    if not (i0 > 0.0 and math.isfinite(i0) and math.isfinite(i1)):
        raise RuntimeError("failed to compute finite critical-mean integrals")
    mean = x_mode + i1 / i0
    if not (mean >= 0.0 and math.isfinite(mean)):
        raise RuntimeError("computed standardized critical mean is invalid")
    return mean


def standardized_critical_mean(tilde_c: float, k: int) -> float:
    """Mean m_k(q) under density exp(q*y - y^(k+1)/(k+1)), y >= 0."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not math.isfinite(tilde_c):
        raise ValueError("tilde_c must be finite")
    q = float(tilde_c)
    if k == 1:
        mean, _log_gap = _truncated_normal_mean_and_log_gap(q)
        return mean
    return _standardized_critical_mean_quadrature(q=q, k=k)


def calibrate_b_first_rq_standardized(tilde_c: float, k: int) -> FirstCalibration:
    """Calibrate at q=tilde_c, with an explicit infeasible-match fallback."""
    mean = standardized_critical_mean(tilde_c=tilde_c, k=k)
    q = float(tilde_c)

    if k == 1:
        # Every finite q is feasible for k=1.  Work in log scale because the
        # gap m_1(q)-q is far below machine precision in positive overload.
        _mean_check, log_gap = _truncated_normal_mean_and_log_gap(q)
        log_b = 0.5 * (math.log(2.0) + math.log(mean) + log_gap)
        b = math.exp(log_b)
        status = CALIBRATION_EXACT
    else:
        mean_power = mean**k
        gap = mean_power - q
        tolerance = CALIBRATION_FEASIBILITY_REL_TOL * max(
            1.0, abs(mean_power), abs(q)
        )
        if gap > tolerance:
            b = math.sqrt(2.0 * mean * gap)
            status = CALIBRATION_EXACT
        else:
            b = 0.0
            status = CALIBRATION_FLUID_FALLBACK

    if not (b >= 0.0 and math.isfinite(b)):
        raise RuntimeError("calibrated b is invalid")
    return FirstCalibration(b=b, standardized_mean=mean, status=status)


def psi_critical(c: float, k: int, beta_patience: float) -> float:
    """Canonical (R=1) physical heavy-traffic mean retained for compatibility."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (beta_patience > 0.0 and math.isfinite(beta_patience)):
        raise ValueError("beta_patience must be finite and > 0")
    if not math.isfinite(c):
        raise ValueError("c must be finite")

    scale = float(beta_patience) ** (-1.0 / float(k + 1))
    tilde_c = float(c) * scale
    return scale * standardized_critical_mean(tilde_c=tilde_c, k=k)


def calibrate_b_first_rq(c: float, k: int, beta_patience: float) -> tuple[float, float]:
    """Canonical R=1 wrapper returning the historical ``(b, psi)`` pair."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (beta_patience > 0.0 and math.isfinite(beta_patience)):
        raise ValueError("beta_patience must be finite and > 0")
    if not math.isfinite(c):
        raise ValueError("c must be finite")
    scale = float(beta_patience) ** (-1.0 / float(k + 1))
    calibration = calibrate_b_first_rq_standardized(
        tilde_c=float(c) * scale,
        k=k,
    )
    return calibration.b, scale * calibration.standardized_mean


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
        _tau, tilde_c = tau_tilde_c(
            c=c,
            k=base.k,
            mu=base.mu,
            c_a2=base.c_a2,
            c_s2=base.c_s2,
            beta_patience=base.beta_patience,
        )
        if self.b_override is not None:
            b = float(self.b_override)
            psi = float("nan")
            calibration_status = CALIBRATION_OVERRIDE
        else:
            calibration = calibrate_b_first_rq_standardized(
                tilde_c=tilde_c,
                k=base.k,
            )
            b = calibration.b
            c_x2 = base.c_a2 + base.c_s2
            physical_scale = (
                c_x2 / (2.0 * base.mu * base.beta_patience)
            ) ** (1.0 / float(base.k + 1))
            psi = physical_scale * calibration.standardized_mean
            calibration_status = calibration.status

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
            "tilde_c": tilde_c,
            "b": b,
            "psi": psi,
            "calibration_status": calibration_status,
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
