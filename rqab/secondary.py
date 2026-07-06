"""Benchmark approximations: Ward-Glynn (WG), hazard-rate scaling, Huang-Gurvich (HG).

These are the comparison methods in RQ_ab.tex Section "Comparison with other
approximations" / Appendix A.  The secondary method is WG when the patience
density is positive at zero (f(0) > 0), otherwise the hazard-rate scaling
approximation.  HG is always computed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from .effective_idw import distribution_beta_at_zero
from .models import AnyModel, DistributionComponent

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
WG_F1_TOL = 1e-12


@dataclass(frozen=True)
class QuadOptions:
    """y-grid quadrature options for the hazard/HG stationary-density integrals."""

    dy: float = 0.02
    y_max_init: float = 16.0
    y_max_limit: float = 2048.0
    tail_log_gap: float = 30.0
    tail_window: int = 40
    min_y_for_tail: float = 8.0

    def validate(self) -> None:
        if not (self.dy > 0.0):
            raise ValueError("dy must be > 0")
        if not (self.y_max_init > 0.0):
            raise ValueError("y_max_init must be > 0")
        if not (self.y_max_limit >= self.y_max_init):
            raise ValueError("y_max_limit must be >= y_max_init")
        if not (self.tail_log_gap > 0.0):
            raise ValueError("tail_log_gap must be > 0")
        if self.tail_window < 2:
            raise ValueError("tail_window must be >= 2")
        if not (self.min_y_for_tail >= 0.0):
            raise ValueError("min_y_for_tail must be >= 0")


@dataclass(frozen=True)
class SecondaryStats:
    """Inputs the WG/Hazard/HG formulas need.

    c_a2 here is the LONG-RUN arrival SCV feeding the abandonment station;
    for tandem models it is derived from the departure-IDC tail.
    """

    mu: float
    c_a2: float
    c_s2: float
    c_x2: float
    f1_at_zero: float
    secondary_method: str


@dataclass(frozen=True)
class IntegrationResult:
    z: float
    n_points: int
    y_max: float
    status: str


def classify_secondary_method(f1_at_zero: float, tol: float = WG_F1_TOL) -> str:
    return "wg" if f1_at_zero > tol else "hazard"


def build_secondary_stats(
    model: AnyModel,
    mu: float,
    c_a2: float,
    c_s2: float,
) -> SecondaryStats:
    """Build SecondaryStats given the effective arrival SCV c_a2."""
    c_x2 = float(c_a2) + float(c_s2)
    if not (c_x2 > 0.0 and math.isfinite(c_x2)):
        raise ValueError("invalid variability: c_a2 + c_s2 must be finite and > 0")

    patience = model.patience
    f1_at_zero = float(distribution_beta_at_zero(patience.family, patience.params, k=1))
    if not math.isfinite(f1_at_zero):
        raise ValueError("invalid F'(0): non-finite")

    return SecondaryStats(
        mu=float(mu),
        c_a2=float(c_a2),
        c_s2=float(c_s2),
        c_x2=c_x2,
        f1_at_zero=f1_at_zero,
        secondary_method=classify_secondary_method(f1_at_zero),
    )


def survival_base(y: float, patience: DistributionComponent) -> float:
    if y <= 0.0:
        return 1.0
    if patience.family == "exponential":
        rate = float(patience.params["rate"])
        return math.exp(-rate * y)
    if patience.family == "erlang_k":
        m = int(patience.params["k"])
        rate = float(patience.params["rate"])
        x = rate * y
        term = 1.0
        total = 1.0
        for j in range(1, m):
            term *= x / float(j)
            total += term
        return math.exp(-x) * total
    p = float(patience.params["p"])
    rate1 = float(patience.params["rate1"])
    rate2 = float(patience.params["rate2"])
    return p * math.exp(-rate1 * y) + (1.0 - p) * math.exp(-rate2 * y)


def log_survival_base(y: float, patience: DistributionComponent) -> float:
    if y <= 0.0:
        return 0.0
    if patience.family == "exponential":
        rate = float(patience.params["rate"])
        return -rate * y
    if patience.family == "erlang_k":
        m = int(patience.params["k"])
        rate = float(patience.params["rate"])
        x = rate * y
        term = 1.0
        total = 1.0
        for j in range(1, m):
            term *= x / float(j)
            total += term
        return -x + math.log(total)
    p = float(patience.params["p"])
    rate1 = float(patience.params["rate1"])
    rate2 = float(patience.params["rate2"])
    a = math.log(p) - rate1 * y
    b = math.log(1.0 - p) - rate2 * y
    m0 = max(a, b)
    return m0 + math.log(math.exp(a - m0) + math.exp(b - m0))


def normal_survival(x: float) -> float:
    return 0.5 * math.erfc(x / SQRT2)


def inverse_mills_ratio_upper_cf(x: float, n_terms: int = 120) -> float:
    if not (x > 0.0):
        raise ValueError("continued-fraction inverse Mills ratio requires x > 0")
    if n_terms < 1:
        raise ValueError("continued-fraction inverse Mills ratio requires n_terms >= 1")

    # Stable backward recurrence for:
    # R(x) = x + 1/(x + 2/(x + 3/(x + ...))).
    r = x + float(n_terms + 1) / x
    for k in range(n_terms, 0, -1):
        r = x + float(k) / r
    return r


def inverse_mills_ratio_upper(x: float) -> float:
    # Ratio phi(x)/(1-Phi(x)) with stable handling for large positive x.
    if x >= 6.0:
        return inverse_mills_ratio_upper_cf(x)
    phi = math.exp(-0.5 * x * x) * INV_SQRT_2PI
    sf = normal_survival(x)
    if sf <= 0.0:
        if x > 0.0:
            return inverse_mills_ratio_upper_cf(x)
        inv = 1.0 / x
        inv2 = inv * inv
        return x + inv + 2.0 * inv * inv2
    return phi / sf


def compute_wg(lam: float, alpha: float, stats: SecondaryStats) -> tuple[float, str]:
    """Ward-Glynn truncated-normal ROU mean (RQ_ab.tex eq:ROU_expectation)."""
    if not (alpha > 0.0):
        raise ValueError(f"invalid alpha={alpha}; expected > 0")
    if not (lam > 0.0):
        raise ValueError(f"invalid lambda={lam}; expected > 0")
    f1 = stats.f1_at_zero
    if not (f1 > WG_F1_TOL):
        raise ValueError(f"WG requires F'(0)>0; got F'(0)={f1}")

    rho = lam / stats.mu
    tilde_cx2 = rho * stats.c_a2 + min(rho, 1.0) * stats.c_s2
    if not (tilde_cx2 > 0.0 and math.isfinite(tilde_cx2)):
        raise ValueError("WG invalid tilde_cx2; expected > 0 and finite")

    c = (rho - 1.0) / math.sqrt(alpha)
    xi = -math.sqrt(2.0 * stats.mu) * c / math.sqrt(f1 * tilde_cx2)
    mills = inverse_mills_ratio_upper(xi)
    term = c / f1 + mills * math.sqrt(tilde_cx2 / (2.0 * stats.mu * f1))
    z = (alpha ** -0.5) * term
    if not math.isfinite(z):
        raise ValueError("WG produced non-finite value")
    return max(0.0, z), "ok"


def integrate_ratio_y(
    alpha: float,
    coeff: float,
    drift_fn: Callable[[float], float],
    opts: QuadOptions,
) -> IntegrationResult:
    """E[Y]/alpha for the density proportional to exp(coeff * int_0^y drift).

    Adaptive head refinement near y=0 plus doubling of the truncation bound
    until the exponent tail has decayed by tail_log_gap and is monotone.
    """
    dy = float(opts.dy)
    y_max = float(opts.y_max_init)
    y_limit = float(opts.y_max_limit)
    tail_window = int(opts.tail_window)
    tail_log_gap = float(opts.tail_log_gap)
    min_y_for_tail = float(opts.min_y_for_tail)

    while True:
        g0 = float(drift_fn(0.0))
        if not math.isfinite(g0):
            raise ValueError("non-finite inner drift at y=0")

        use_refined_head = False
        dy_fine = dy
        y_fine_end = 0.0
        if g0 < 0.0:
            ell0 = 1.0 / (coeff * (-g0))
            if ell0 > 0.0 and math.isfinite(ell0):
                use_refined_head = True
                dy_fine = min(dy, ell0 / 24.0)
                y_fine_end = min(y_max, max(10.0 * ell0, 4.0 * dy))
        elif g0 > 0.0:
            # Near-critical overloaded cases can peak at very small y before drifting down.
            # Detect the first sign change and refine the head to resolve that local mode.
            probe_step = max(dy / 16.0, 1e-12)
            y_prev = 0.0
            g_prev = g0
            y_probe = probe_step
            y_root: float | None = None
            while y_probe <= y_max:
                g_probe = float(drift_fn(y_probe))
                if not math.isfinite(g_probe):
                    raise ValueError(f"non-finite inner drift at y={y_probe}")
                if g_probe <= 0.0:
                    denom = g_prev - g_probe
                    if denom > 0.0:
                        frac = g_prev / denom
                        frac = min(1.0, max(0.0, frac))
                        y_root = y_prev + frac * (y_probe - y_prev)
                    else:
                        y_root = y_probe
                    break
                y_prev = y_probe
                g_prev = g_probe
                if y_probe >= y_max:
                    break
                y_probe = min(y_max, y_probe * 2.0)

            if y_root is not None and y_root > 0.0:
                use_refined_head = True
                dy_fine = min(dy, max(y_root / 24.0, dy / 40.0))
                y_fine_end = min(y_max, max(10.0 * y_root, 4.0 * dy))

        local_exp_gap = abs(coeff * g0 * dy)
        if local_exp_gap < 0.5:
            use_refined_head = True
            dy_fine = min(dy_fine, dy / 40.0)
            y_fine_end = min(y_max, max(y_fine_end, 4.0 * dy))

        y: list[float]
        if use_refined_head and dy_fine > 0.0 and y_fine_end > 0.0:
            # Guardrail against runaway point counts when the boundary layer is tiny.
            max_head_points = 20000
            if y_fine_end / dy_fine > max_head_points:
                dy_fine = y_fine_end / float(max_head_points)

            y = [0.0]
            n_head = int(math.ceil(y_fine_end / dy_fine))
            for i in range(1, n_head + 1):
                yy = min(y_fine_end, float(i) * dy_fine)
                if yy > y[-1]:
                    y.append(yy)

            if y[-1] < y_max:
                n_tail = int(math.ceil((y_max - y[-1]) / dy))
                start = y[-1]
                for i in range(1, n_tail + 1):
                    yy = min(y_max, start + float(i) * dy)
                    if yy > y[-1]:
                        y.append(yy)
        else:
            n_points = int(math.ceil(y_max / dy)) + 1
            if n_points < 2:
                n_points = 2
            y = [float(i) * dy for i in range(n_points)]
            if y[-1] < y_max:
                y.append(y_max)
            else:
                y[-1] = y_max

        n_points = len(y)
        if n_points < 2:
            raise ValueError("y-grid must contain at least two points")

        g: list[float] = []
        for yy in y:
            gv = float(drift_fn(yy))
            if not math.isfinite(gv):
                raise ValueError(f"non-finite inner drift at y={yy}")
            g.append(gv)

        inner: list[float] = [0.0] * n_points
        acc = 0.0
        for i in range(1, n_points):
            step = y[i] - y[i - 1]
            acc += 0.5 * (g[i - 1] + g[i]) * step
            inner[i] = acc

        exponent: list[float] = [coeff * iv for iv in inner]
        max_exp = max(exponent)
        weights: list[float] = []
        for ev in exponent:
            gap = ev - max_exp
            weights.append(math.exp(gap) if gap > -745.0 else 0.0)

        denom = 0.0
        numer = 0.0
        for i in range(1, n_points):
            step = y[i] - y[i - 1]
            w0 = weights[i - 1]
            w1 = weights[i]
            denom += 0.5 * (w0 + w1) * step
            numer += 0.5 * (y[i - 1] * w0 + y[i] * w1) * step

        if not (denom > 0.0 and math.isfinite(denom)):
            raise ValueError("outer denominator integral is non-positive/non-finite")
        if not math.isfinite(numer):
            raise ValueError("outer numerator integral is non-finite")

        ratio_y = numer / denom
        z = max(0.0, ratio_y / alpha)

        end_gap = max_exp - exponent[-1]
        trend_ok = False
        if n_points > tail_window:
            start = n_points - tail_window
            trend_ok = True
            for i in range(start + 1, n_points):
                if exponent[i] > exponent[i - 1]:
                    trend_ok = False
                    break

        tail_mass = 0.0
        if n_points > tail_window:
            start = n_points - tail_window
            for i in range(start + 1, n_points):
                step = y[i] - y[i - 1]
                tail_mass += 0.5 * (weights[i - 1] + weights[i]) * step
            tail_mass /= denom
        tail_mass_limit = math.exp(-0.5 * min(tail_log_gap, 50.0))

        if (
            y_max >= min_y_for_tail
            and end_gap >= tail_log_gap
            and trend_ok
            and tail_mass <= tail_mass_limit
        ):
            return IntegrationResult(z=z, n_points=n_points, y_max=y_max, status="ok")

        if y_max >= y_limit:
            status = (
                "error:tail_not_converged("
                f"y_max={y_max:.6g},end_gap={end_gap:.4g},tail_mass={tail_mass:.4g})"
            )
            return IntegrationResult(z=z, n_points=n_points, y_max=y_max, status=status)

        y_max = min(y_limit, y_max * 2.0)


def compute_hazard(
    lam: float,
    alpha: float,
    stats: SecondaryStats,
    patience: DistributionComponent,
    opts: QuadOptions,
) -> IntegrationResult:
    """Hazard-rate scaling approximation (Reed-Ward), for f(0) = 0 patience."""
    if not (alpha > 0.0):
        raise ValueError(f"invalid alpha={alpha}; expected > 0")
    if not (lam > 0.0):
        raise ValueError(f"invalid lambda={lam}; expected > 0")
    rho = lam / stats.mu
    coeff = (2.0 * stats.mu) / (stats.c_x2 * alpha)

    def drift(y: float) -> float:
        return log_survival_base(y, patience) + (rho - 1.0)

    return integrate_ratio_y(alpha=alpha, coeff=coeff, drift_fn=drift, opts=opts)


def compute_hg(
    lam: float,
    alpha: float,
    stats: SecondaryStats,
    patience: DistributionComponent,
    opts: QuadOptions,
) -> IntegrationResult:
    """Huang-Gurvich diffusion approximation."""
    if not (alpha > 0.0):
        raise ValueError(f"invalid alpha={alpha}; expected > 0")
    if not (lam > 0.0):
        raise ValueError(f"invalid lambda={lam}; expected > 0")
    rho = lam / stats.mu
    rho_wedge = min(rho, 1.0)
    if not (rho_wedge > 0.0 and math.isfinite(rho_wedge)):
        raise ValueError("invalid rho for HG; expected positive finite")

    coeff = (2.0 * stats.mu) / (stats.c_x2 * rho_wedge * alpha)

    def drift(y: float) -> float:
        return rho * survival_base(y, patience) - 1.0

    return integrate_ratio_y(alpha=alpha, coeff=coeff, drift_fn=drift, opts=opts)


def solve_secondary(
    lam: float,
    alpha: float,
    stats: SecondaryStats,
    patience: DistributionComponent,
    opts: QuadOptions,
) -> dict[str, Any]:
    """Compute the WG-or-hazard secondary value plus HG for one tuple.

    Returns the CSV column values (empty string for the branch not taken).
    Raises if either method fails to converge.
    """
    rho = lam / stats.mu
    tilde_cx2 = rho * stats.c_a2 + min(rho, 1.0) * stats.c_s2

    z_wg: float | None = None
    z_hazard: float | None = None
    secondary_points = 0
    secondary_ymax = 0.0

    if stats.secondary_method == "wg":
        z_wg, status_secondary = compute_wg(lam=lam, alpha=alpha, stats=stats)
        z_secondary: float | None = z_wg
    else:
        hazard_res = compute_hazard(
            lam=lam, alpha=alpha, stats=stats, patience=patience, opts=opts
        )
        z_hazard = hazard_res.z
        z_secondary = z_hazard
        status_secondary = hazard_res.status
        secondary_points = hazard_res.n_points
        secondary_ymax = hazard_res.y_max

    hg_res = compute_hg(lam=lam, alpha=alpha, stats=stats, patience=patience, opts=opts)
    status_hg = hg_res.status
    if (not str(status_secondary).startswith("ok")) or (not str(status_hg).startswith("ok")):
        raise RuntimeError(
            "secondary-method solve failed with statuses "
            f"secondary='{status_secondary}', hg='{status_hg}'"
        )

    return {
        "secondary_method": stats.secondary_method,
        "z_secondary": "" if z_secondary is None else z_secondary,
        "z_wg": "" if z_wg is None else z_wg,
        "z_hazard": "" if z_hazard is None else z_hazard,
        "z_hg": hg_res.z,
        "c_x2": stats.c_x2,
        "tilde_cx2": tilde_cx2,
        "f1_at_zero": stats.f1_at_zero,
        "status_secondary": status_secondary,
        "status_hg": status_hg,
        "integration_points": max(secondary_points, hg_res.n_points),
        "integration_y_max": max(secondary_ymax, hg_res.y_max),
    }
