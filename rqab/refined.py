"""Refined RQ approximation (RQ_ab.tex eq:RQ_ab_2) over a tuple grid.

The combined output row also carries the WG/Hazard/HG benchmark columns
(computed independently, so a refined-solver failure does not blank them).
Works for single-station and tandem models; the only structural differences
are the arrival-IDC source and, for tandem, the queue-2 arrival SCV derived
from the departure-IDC tail.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .effective_idw import hat_idw_refined, tau_tilde_c
from .fixed_point import (
    A_INF_TOL,
    BisectOptions,
    FixedPointResult,
    solve_fixed_point,
    survival_alpha,
)
from .idc import arrival_idc_curve_for
from .models import AnyModel, BaseSystemStats
from .secondary import QuadOptions, SecondaryStats, build_secondary_stats, solve_secondary
from .tables import BCalibrationInterpolator, WTableInterpolator
from .util import require_float, require_int, require_str

CSV_COLUMNS = [
    "tuple_id",
    "lambda",
    "alpha",
    "lambda_k",
    "lambda_form",
    "alpha_k",
    "z_rq_refined",
    "c",
    "b",
    "k",
    "h",
    "rho",
    "mu",
    "tau",
    "tilde_c",
    "beta_patience",
    "c_a2",
    "c_s2",
    "survival_at_solution",
    "rhs_at_solution",
    "solver_status",
    "bisect_iters",
    "bracket_lo",
    "bracket_hi",
    "secondary_method",
    "z_secondary",
    "z_wg",
    "z_hazard",
    "z_hg",
    "c_x2",
    "tilde_cx2",
    "f1_at_zero",
    "status_secondary",
    "status_hg",
    "integration_points",
    "integration_y_max",
    "model_name",
    "model_alias",
]

_SECONDARY_BLANKS = {
    "secondary_method": "",
    "z_secondary": "",
    "z_wg": "",
    "z_hazard": "",
    "z_hg": "",
    "c_x2": "",
    "tilde_cx2": "",
    "f1_at_zero": "",
    "status_secondary": "",
    "status_hg": "",
    "integration_points": "",
    "integration_y_max": "",
}


def canonical_b_table_c(tilde_c: float, k: int) -> float:
    """Return the canonical raw-load coordinate for ``tilde_c``.

    The calibration table stores ``c_ref`` for the mean-one M/M/1+E_k
    reference model, where
    ``tilde_c = c_ref * (k**k / k!)**(-1 / (k + 1))``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    beta_ref = float(k**k) / float(math.factorial(k))
    return float(tilde_c) * beta_ref ** (1.0 / float(k + 1))


@dataclass(frozen=True)
class RefinedKernel:
    c: float
    b: float
    tau: float
    tilde_c: float
    rho: float
    sqrt_q: list[float]
    tail_sqrt_q: float
    c_a2_effective: float


class RefinedSolver:
    """Per-model context for solving refined-RQ tuples."""

    def __init__(
        self,
        model: AnyModel,
        base: BaseSystemStats,
        b_table: BCalibrationInterpolator,
        w_interp: WTableInterpolator,
        s_grid: list[float],
        bisect: BisectOptions | None = None,
        quad: QuadOptions | None = None,
    ) -> None:
        self.model = model
        self.base = base
        self.b_table = b_table
        self.w_interp = w_interp
        self.s_grid = s_grid
        self.bisect = bisect if bisect is not None else BisectOptions()
        self.quad = quad if quad is not None else QuadOptions()
        self.bisect.validate()
        self.quad.validate()

    def build_kernel(self, lam: float, alpha: float) -> RefinedKernel:
        base = self.base
        mu = base.mu
        rho = lam / mu
        c = (rho - 1.0) / (alpha**base.h)
        tau, tilde_c = tau_tilde_c(
            c=c,
            k=base.k,
            mu=mu,
            c_a2=base.c_a2,
            c_s2=base.c_s2,
            beta_patience=base.beta_patience,
        )

        b = self.b_table.evaluate(canonical_b_table_c(tilde_c, base.k))

        ia = arrival_idc_curve_for(self.model, s_values=self.s_grid, lam=lam)
        if len(ia) != len(self.s_grid):
            raise RuntimeError("internal error: arrival IDC curve length mismatch")

        # Effective long-run arrival SCV feeding the abandonment station.
        if self.model.is_tandem:
            tail_count = min(10, len(ia))
            c_a2_effective = float(sum(ia[-tail_count:]) / float(tail_count))
            if not (c_a2_effective > 0.0 and math.isfinite(c_a2_effective)):
                raise RuntimeError(
                    "internal error: invalid queue2 c_a^2 derived from departure IDC"
                )
        else:
            c_a2_effective = base.c_a2

        hat = hat_idw_refined(t=self.s_grid, ia_t=ia, rho=rho, c_s2=base.c_s2)
        hat_values = [float(v) for v in hat] if isinstance(hat, list) else [float(hat)]
        if len(hat_values) != len(self.s_grid):
            raise RuntimeError("internal error: hat_Iw curve length mismatch")

        scale_t = (alpha ** (2.0 * base.h)) * tau
        sqrt_q: list[float] = []
        for s, hat_s in zip(self.s_grid, hat_values):
            w_val = float(self.w_interp.w(tilde_c, scale_t * s))
            q = hat_s * w_val * s
            sqrt_q.append(math.sqrt(q) if q > 0.0 else 0.0)
        return RefinedKernel(
            c=c,
            b=b,
            tau=tau,
            tilde_c=tilde_c,
            rho=rho,
            sqrt_q=sqrt_q,
            tail_sqrt_q=sqrt_q[-1],
            c_a2_effective=c_a2_effective,
        )

    def _rhs(self, z: float, lam: float, alpha: float, kernel: RefinedKernel) -> tuple[float, float]:
        base = self.base
        surv = survival_alpha(z=z, alpha=alpha, patience=self.model.patience)
        surv = min(1.0, max(0.0, surv))

        lam_surv_over_mu = lam * surv / base.mu
        a = lam_surv_over_mu - 1.0
        lam_surv_over_mu2 = lam * surv / (base.mu * base.mu)

        if a > A_INF_TOL:
            return float("inf"), surv
        if abs(a) <= A_INF_TOL and kernel.b > 0.0 and lam_surv_over_mu2 > 0.0 and kernel.tail_sqrt_q > 0.0:
            return float("inf"), surv

        if lam_surv_over_mu2 <= 0.0:
            return 0.0, surv
        kappa = kernel.b * math.sqrt(lam_surv_over_mu2)

        best = 0.0
        for s, sq in zip(self.s_grid, kernel.sqrt_q):
            value = a * s + kappa * sq
            if value > best:
                best = value
        return best, surv

    def solve_fixed_point(self, lam: float, alpha: float, kernel: RefinedKernel) -> FixedPointResult:
        return solve_fixed_point(
            lambda z: self._rhs(z, lam=lam, alpha=alpha, kernel=kernel),
            self.bisect,
        )

    def solve_one_tuple(self, row: dict[str, Any], continue_on_error: bool = False) -> dict[str, Any]:
        """Solve one grid tuple; returns the combined CSV row.

        With continue_on_error=True, refined and secondary failures are
        recorded in their status columns instead of raising.
        """
        lam = require_float(row, "lambda")
        alpha = require_float(row, "alpha")
        if lam <= 0.0:
            raise ValueError(f"invalid lambda={lam}; expected > 0")
        if alpha <= 0.0:
            raise ValueError(f"invalid alpha={alpha}; expected > 0")

        base = self.base
        out: dict[str, Any] = {
            "tuple_id": require_int(row, "tuple_id"),
            "lambda": lam,
            "alpha": alpha,
            "lambda_k": require_int(row, "lambda_k"),
            "lambda_form": require_str(row, "lambda_form"),
            "alpha_k": require_int(row, "alpha_k"),
            "z_rq_refined": "",
            "c": "",
            "b": "",
            "k": base.k,
            "h": base.h,
            "rho": "",
            "mu": base.mu,
            "tau": "",
            "tilde_c": "",
            "beta_patience": base.beta_patience,
            "c_a2": base.c_a2,
            "c_s2": base.c_s2,
            "survival_at_solution": "",
            "rhs_at_solution": "",
            "solver_status": "",
            "bisect_iters": "",
            "bracket_lo": "",
            "bracket_hi": "",
            **_SECONDARY_BLANKS,
            "model_name": self.model.model_name,
            "model_alias": self.model.model_alias,
        }

        kernel: RefinedKernel | None = None
        try:
            kernel = self.build_kernel(lam=lam, alpha=alpha)
            out.update(
                {
                    "c": kernel.c,
                    "b": kernel.b,
                    "rho": kernel.rho,
                    "tau": kernel.tau,
                    "tilde_c": kernel.tilde_c,
                }
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            out["solver_status"] = f"error:{exc}"
            out["status_secondary"] = f"error:{exc}"
            out["status_hg"] = f"error:{exc}"
            return out

        try:
            solution = self.solve_fixed_point(lam=lam, alpha=alpha, kernel=kernel)
            out.update(
                {
                    "z_rq_refined": solution.z,
                    "survival_at_solution": solution.survival_at_solution,
                    "rhs_at_solution": solution.rhs_at_solution,
                    "solver_status": solution.status,
                    "bisect_iters": solution.bisect_iters,
                    "bracket_lo": solution.bracket_lo,
                    "bracket_hi": solution.bracket_hi,
                }
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            out["solver_status"] = f"error:{exc}"

        try:
            stats = build_secondary_stats(
                self.model,
                mu=base.mu,
                c_a2=kernel.c_a2_effective,
                c_s2=base.c_s2,
            )
            out.update(
                solve_secondary(
                    lam=lam,
                    alpha=alpha,
                    stats=stats,
                    patience=self.model.patience,
                    opts=self.quad,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            out["status_secondary"] = f"error:{exc}"
            out["status_hg"] = f"error:{exc}"

        return out


def default_out_csv(results_dir: Path, model_alias: str) -> Path:
    return results_dir / f"refined_rq_grid_{model_alias}.csv"
