#!/usr/bin/env python3
"""
Utilities for hyperexponential (H2) IDC and effective-IDW approximations.

Implements formulas used in RQ_ab.tex:
- IDC of equilibrium H2 renewal process
- Refined IDW surrogate and effective-IDW approximation
- Scaling parameters (tau, tilde c) from Lemma var_expression
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class H2Params:
    """Parameterization of an H2 interarrival model."""

    rate: float
    scv: float
    r: float
    p: float
    mu1: float
    mu2: float
    gamma: float
    beta: float


def _is_sequence(x: object) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes))


def _as_float_list(x: Sequence[float] | float) -> tuple[list[float], bool]:
    if _is_sequence(x):
        return [float(v) for v in x], True
    return [float(x)], False


def _from_list(values: list[float], as_list: bool) -> list[float] | float:
    if as_list:
        return values
    return values[0]


def h2_params_from_rate_scv_r(rate: float, scv: float, r: float = 0.5) -> H2Params:
    """
    Recover H2 parameters from (rate, scv, r).

    rate = 1 / E[X], scv = Var(X) / E[X]^2,
    r = (p/mu1) / (p/mu1 + (1-p)/mu2), with mu1 >= mu2.
    """
    if rate <= 0.0:
        raise ValueError("rate must be > 0")
    if scv < 1.0:
        raise ValueError("scv must be >= 1 for H2")
    if not (0.0 < r < 1.0):
        raise ValueError("r must be strictly between 0 and 1")

    a = 0.5 * (scv + 1.0)
    b = a - 1.0 + 2.0 * r
    disc = b * b - 4.0 * a * r * r
    tol = 1e-13
    if disc < -tol:
        raise ValueError("invalid (scv, r) pair: no real H2 solution")
    disc = max(0.0, disc)
    sqrt_disc = math.sqrt(disc)

    candidates = [
        (b + sqrt_disc) / (2.0 * a),
        (b - sqrt_disc) / (2.0 * a),
    ]

    selected: tuple[float, float, float] | None = None
    for p in candidates:
        if not (tol < p < 1.0 - tol):
            continue
        mu1 = rate * p / r
        mu2 = rate * (1.0 - p) / (1.0 - r)
        if mu1 + tol >= mu2:
            selected = (p, mu1, mu2)
            break

    if selected is None:
        raise ValueError("could not recover a valid H2 branch with mu1 >= mu2")

    p, mu1, mu2 = selected
    gamma = (1.0 - p) * mu1 + p * mu2
    if gamma <= 0.0:
        raise ValueError("invalid H2 parameters: gamma must be > 0")
    beta = p * (1.0 - p) * (mu1 - mu2) ** 2 / (gamma * gamma)

    return H2Params(
        rate=rate,
        scv=scv,
        r=r,
        p=p,
        mu1=mu1,
        mu2=mu2,
        gamma=gamma,
        beta=beta,
    )


def exponential_moments(rate: float) -> tuple[float, float]:
    """Return (mean, scv) for Exp(rate)."""
    if rate <= 0.0:
        raise ValueError("rate must be > 0")
    return (1.0 / rate, 1.0)


def erlang_k_moments(k: int, rate: float) -> tuple[float, float]:
    """Return (mean, scv) for Erlang(k, rate)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if rate <= 0.0:
        raise ValueError("rate must be > 0")
    return (k / rate, 1.0 / k)


def lognormal_moments(mean: float, scv: float) -> tuple[float, float]:
    """Return (mean, scv) for LogNormal parameterized by mean and SCV."""
    if not (math.isfinite(mean) and mean > 0.0):
        raise ValueError("mean must be finite and > 0")
    if not (math.isfinite(scv) and scv > 0.0):
        raise ValueError("scv must be finite and > 0")
    return (mean, scv)


def hyperexponential2_moments(p: float, rate1: float, rate2: float) -> tuple[float, float]:
    """Return (mean, scv) for two-phase hyperexponential."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be strictly between 0 and 1")
    if rate1 <= 0.0 or rate2 <= 0.0:
        raise ValueError("rate1 and rate2 must be > 0")

    q = 1.0 - p
    mean = p / rate1 + q / rate2
    second = 2.0 * (p / (rate1 * rate1) + q / (rate2 * rate2))
    var = second - mean * mean
    if var < 0.0 and var > -1e-12:
        var = 0.0
    return (mean, var / (mean * mean))


def h2_shape_from_canonical(p: float, rate1: float, rate2: float) -> tuple[float, float]:
    """
    Convert canonical H2 params (p, rate1, rate2) to (scv, r),
    where r is the mean-mass split on branch 1.
    """
    mean, scv = hyperexponential2_moments(p, rate1, rate2)
    r = (p / rate1) / mean
    return (scv, r)


def distribution_moments(family: str, params: dict[str, float | int]) -> tuple[float, float]:
    """
    Return (mean, scv) for canonical distribution specs.
    Supported families: exponential, erlang_k, lognormal, hyperexponential2.
    """
    f = family.strip().lower()
    if f in ("exponential", "exp"):
        return exponential_moments(rate=float(params["rate"]))
    if f in ("erlang_k", "erlang"):
        return erlang_k_moments(k=int(params["k"]), rate=float(params["rate"]))
    if f in ("lognormal", "ln"):
        return lognormal_moments(mean=float(params["mean"]), scv=float(params["scv"]))
    if f in ("hyperexponential2", "h2"):
        return hyperexponential2_moments(
            p=float(params["p"]),
            rate1=float(params["rate1"]),
            rate2=float(params["rate2"]),
        )
    raise ValueError(f"unsupported distribution family: {family}")


def distribution_beta_at_zero(
    family: str,
    params: dict[str, float | int],
    k: int,
) -> float:
    """
    Return beta = F^(k)(0) / k! for the configured distribution.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    f = family.strip().lower()
    inv_fact = 1.0 / float(math.factorial(k))

    if f in ("exponential", "exp"):
        rate = float(params["rate"])
        if rate <= 0.0:
            raise ValueError("rate must be > 0")
        sign = 1.0 if (k % 2 == 1) else -1.0
        return sign * (rate**k) * inv_fact

    if f in ("erlang_k", "erlang"):
        m = int(params["k"])
        rate = float(params["rate"])
        if m < 1:
            raise ValueError("k must be >= 1")
        if rate <= 0.0:
            raise ValueError("rate must be > 0")

        # F(x)=1-exp(-x)*sum_{j=0}^{m-1} x^j/j!, x=rate*t.
        # beta is the coefficient of t^k in F(t).
        coeff_xk = 0.0
        for j in range(0, min(k, m - 1) + 1):
            rem = k - j
            term = ((-1.0) ** rem) / (math.factorial(j) * math.factorial(rem))
            coeff_xk += term
        coeff_xk = -coeff_xk
        return coeff_xk * (rate**k)

    if f in ("hyperexponential2", "h2"):
        p = float(params["p"])
        rate1 = float(params["rate1"])
        rate2 = float(params["rate2"])
        if not (0.0 < p < 1.0):
            raise ValueError("p must be strictly between 0 and 1")
        if rate1 <= 0.0 or rate2 <= 0.0:
            raise ValueError("rate1 and rate2 must be > 0")
        sign = 1.0 if (k % 2 == 1) else -1.0
        return sign * (p * (rate1**k) + (1.0 - p) * (rate2**k)) * inv_fact

    raise ValueError(f"unsupported distribution family: {family}")


def _ratio_one_minus_exp_over_x(x: float) -> float:
    """Compute (1-exp(-x))/x stably with the x->0 limit = 1."""
    ax = abs(x)
    if ax < 1e-8:
        # 1 - x/2 + x^2/6 + O(x^3)
        return 1.0 - 0.5 * x + (x * x) / 6.0
    return -math.expm1(-x) / x


def idc_h2_equilibrium(
    t: Sequence[float] | float,
    rate: float,
    scv: float,
    r: float = 0.5,
) -> list[float] | float:
    """
    IDC of equilibrium H2 renewal process:
      I_H2(t) = c^2 - (2*beta/(gamma*t)) * (1 - exp(-gamma*t))
    with exact I_H2(0) = 1.
    """
    params = h2_params_from_rate_scv_r(rate=rate, scv=scv, r=r)
    t_values, as_list = _as_float_list(t)
    out: list[float] = []
    for tv in t_values:
        if tv < 0.0:
            raise ValueError("t must be >= 0")
        if tv == 0.0:
            out.append(1.0)
            continue
        x = params.gamma * tv
        ratio = _ratio_one_minus_exp_over_x(x)
        out.append(scv - 2.0 * params.beta * ratio)
    return _from_list(out, as_list)


def idc_erlang_equilibrium(
    t: Sequence[float] | float,
    k: int,
    rate: float,
) -> list[float] | float:
    """
    Exact IDC of stationary Erlang-k renewal process using modulo-Poisson decomposition.

    For M_t ~ Poisson(rate * t), J ~ Unif{0,...,k-1} independent:
      N(t) = floor((J + M_t)/k),
      IDC(t) = 1/k + E[R(k-R)] / (k * rate * t), R = M_t mod k.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if rate <= 0.0:
        raise ValueError("rate must be > 0")

    t_values, as_list = _as_float_list(t)
    out: list[float] = []
    two_pi_over_k = 2.0 * math.pi / float(k)

    for tv in t_values:
        if tv < 0.0:
            raise ValueError("t must be >= 0")
        if tv == 0.0:
            out.append(1.0)
            continue

        mu = rate * tv
        probs = [0.0] * k
        for r in range(k):
            pr = 1.0 / k
            for j in range(1, k):
                theta = two_pi_over_k * float(j)
                amp = math.exp(mu * (math.cos(theta) - 1.0))
                phase = mu * math.sin(theta) - float(r) * theta
                pr += (amp * math.cos(phase)) / k
            probs[r] = max(0.0, pr)

        total = sum(probs)
        if total <= 0.0:
            raise ValueError("failed to compute modulo-poisson probabilities")
        probs = [p / total for p in probs]

        er_term = 0.0
        for r, pr in enumerate(probs):
            er_term += float(r * (k - r)) * pr

        idc = 1.0 / float(k) + er_term / (float(k) * rate * tv)
        out.append(idc)

    return _from_list(out, as_list)


def hat_idw_refined(
    t: Sequence[float] | float,
    ia_t: Sequence[float] | float,
    rho: float,
    c_s2: float,
) -> list[float] | float:
    """
    Refined surrogate IDW:
      hat_Iw(t) = I_a(t)/(rho vee 1) + (1 - 1/(rho vee 1)) + c_s^2
    """
    _ = t  # Kept for API consistency with paper notation.
    rho_eff = max(float(rho), 1.0)
    base = 1.0 - 1.0 / rho_eff + float(c_s2)
    ia_values, as_list = _as_float_list(ia_t)
    out = [ia / rho_eff + base for ia in ia_values]
    return _from_list(out, as_list)


def tau_tilde_c(
    c: float,
    k: int,
    mu: float,
    c_a2: float,
    c_s2: float,
    beta_patience: float,
) -> tuple[float, float]:
    """
    Compute (tau, tilde_c) from Lemma var_expression in RQ_ab.tex.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if mu <= 0.0:
        raise ValueError("mu must be > 0")
    if beta_patience <= 0.0:
        raise ValueError("beta_patience must be > 0")

    cx2 = float(c_a2) + float(c_s2)
    if cx2 <= 0.0:
        raise ValueError("c_a2 + c_s2 must be > 0")

    ratio = cx2 / (2.0 * float(mu))
    kp1 = k + 1.0
    tau = ratio ** ((k - 1.0) / kp1) * float(beta_patience) ** (2.0 / kp1)
    tilde_c = float(c) * ratio ** (-k / kp1) * float(beta_patience) ** (-1.0 / kp1)
    return tau, tilde_c


def effective_idw_approx(
    t: Sequence[float] | float,
    ia_t: Sequence[float] | float,
    rho: float,
    c_s2: float,
    w_values: Sequence[float] | float,
) -> list[float] | float:
    """
    Effective-IDW approximation:
      hat_Iw_ab(t) ~= hat_Iw(t) * w_{tilde c,k}(...)
    """
    hat = hat_idw_refined(t=t, ia_t=ia_t, rho=rho, c_s2=c_s2)
    hat_list, hat_is_list = _as_float_list(hat)
    w_list, w_is_list = _as_float_list(w_values)

    if hat_is_list and w_is_list:
        if len(hat_list) != len(w_list):
            raise ValueError("ia_t and w_values must have the same length")
        out = [a * b for a, b in zip(hat_list, w_list)]
        return out
    if hat_is_list and not w_is_list:
        return [a * w_list[0] for a in hat_list]
    if not hat_is_list and w_is_list:
        return [hat_list[0] * b for b in w_list]
    return hat_list[0] * w_list[0]
