"""IDC curves for equilibrium-renewal arrivals and tandem departures.

The tandem approximation blends arrival and service IDC using an RBM weight.
"""
from __future__ import annotations

import math

from .effective_idw import (
    distribution_moments,
    h2_shape_from_canonical,
    idc_erlang_equilibrium,
    idc_h2_equilibrium,
)
from .models import DistributionComponent, ParsedTandemModel

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def renewal_idc_curve(
    s_values: list[float],
    lam: float,
    component: DistributionComponent,
) -> list[float]:
    """IDC of the equilibrium renewal process with rate lam at horizons s."""
    if component.family == "exponential":
        return [1.0 for _ in s_values]

    if component.family == "erlang_k":
        k_a = int(component.params["k"])
        rate = float(k_a) * lam
        out = idc_erlang_equilibrium(t=s_values, k=k_a, rate=rate)
        if isinstance(out, list):
            return [float(v) for v in out]
        return [float(out)]

    p = float(component.params["p"])
    rate1 = float(component.params["rate1"])
    rate2 = float(component.params["rate2"])
    scv, r = h2_shape_from_canonical(p=p, rate1=rate1, rate2=rate2)
    out_h2 = idc_h2_equilibrium(t=s_values, rate=lam, scv=scv, r=r)
    if isinstance(out_h2, list):
        return [float(v) for v in out_h2]
    return [float(out_h2)]


def phi_standard(x: float) -> float:
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


def phi_c_standard(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def rbm_c_star(t: float) -> float:
    if t <= 0.0:
        return 1.0
    rt = math.sqrt(t)
    tail = phi_c_standard(rt)
    return 2.0 * (1.0 - 2.0 * t - t * t) * tail + 2.0 * rt * phi_standard(rt) * (1.0 + t)


def w_star_rbm(t: float) -> float:
    if t <= 0.0:
        return 0.0
    c_val = rbm_c_star(t)
    return 1.0 - (1.0 - c_val) / (2.0 * t)


def departure_idc_curve(
    s_values: list[float],
    lam: float,
    model: ParsedTandemModel,
) -> list[float]:
    """Approximate IDC of the queue-1 departure process (queue-2 arrivals)."""
    rho1 = model.queue1.traffic_intensity
    ia = renewal_idc_curve(s_values=s_values, lam=lam, component=model.queue1.arrival)
    is_curve = renewal_idc_curve(s_values=s_values, lam=lam, component=model.queue1.service)

    _, c_a1_2 = distribution_moments(model.queue1.arrival.family, model.queue1.arrival.params)
    _, c_s1_2 = distribution_moments(model.queue1.service.family, model.queue1.service.params)
    c_x1_2 = float(c_a1_2 + c_s1_2)
    if not (c_x1_2 > 0.0 and math.isfinite(c_x1_2)):
        raise ValueError("queue1 c_a^2 + c_s^2 must be finite and > 0")

    out: list[float] = []
    for s, ia_s, is_s in zip(s_values, ia, is_curve):
        if s <= 0.0:
            out.append(1.0)
            continue
        u = ((1.0 - rho1) ** 2) * lam * s / (rho1 * c_x1_2)
        w = w_star_rbm(u)
        out.append(w * ia_s + (1.0 - w) * is_s)
    return out


def arrival_idc_curve_for(model, s_values: list[float], lam: float) -> list[float]:
    """The IDC of the arrival process feeding the abandonment station.

    Single-station models: equilibrium renewal IDC of the arrival process.
    Tandem models: RQNA-IDC blend of the queue-1 departure process.
    """
    if model.is_tandem:
        return departure_idc_curve(s_values=s_values, lam=lam, model=model)
    return renewal_idc_curve(s_values=s_values, lam=lam, component=model.arrival)
