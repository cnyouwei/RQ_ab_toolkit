"""Model-config parsing for single-station GI/GI/1+GI and tandem models.

A model config JSON has the shape::

    {"simulation": {...}, "model": {"name": ..., "alias": ...,
        "arrival": {"distribution": {...}},          # single-station
        "service": {...}, "patience": {...}}}

or, for tandem GI/GI/1 -> ./GI/1+GI models, ``model.queue1`` / ``model.queue2``
blocks.  `load_model_config` auto-detects which one it is.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Union

from .effective_idw import distribution_beta_at_zero, distribution_moments
from .util import sanitize_alias


@dataclass(frozen=True)
class DistributionComponent:
    family: str
    params: dict[str, float | int]


@dataclass(frozen=True)
class ParsedModel:
    """Single-station GI/GI/1+GI model."""

    model_name: str
    model_alias: str
    normalize_service_mean_to_one: bool
    arrival: DistributionComponent
    service: DistributionComponent
    patience: DistributionComponent

    @property
    def is_tandem(self) -> bool:
        return False


@dataclass(frozen=True)
class Queue1Spec:
    traffic_intensity: float
    arrival: DistributionComponent
    service: DistributionComponent


@dataclass(frozen=True)
class Queue2Spec:
    service: DistributionComponent
    patience: DistributionComponent


@dataclass(frozen=True)
class ParsedTandemModel:
    """Tandem GI/GI/1 -> ./GI/1+GI model; only queue 2 has abandonment."""

    model_name: str
    model_alias: str
    normalize_service_mean_to_one: bool
    queue1: Queue1Spec
    queue2: Queue2Spec

    @property
    def is_tandem(self) -> bool:
        return True

    @property
    def patience(self) -> DistributionComponent:
        return self.queue2.patience

    @property
    def service(self) -> DistributionComponent:
        return self.queue2.service


AnyModel = Union[ParsedModel, ParsedTandemModel]


@dataclass(frozen=True)
class BaseSystemStats:
    """Derived per-model constants used by the RQ approximations.

    mu: service rate of the (abandonment) station.
    c_a2/c_s2: arrival/service SCVs (for tandem: queue-1 arrival, queue-2 service).
    k: first index with F^(k)(0) > 0 for the patience CDF; h = k/(k+1).
    beta_patience: F^(k)(0)/k! of the base (mean-one) patience distribution.
    """

    mu: float
    c_a2: float
    c_s2: float
    k: int
    h: float
    beta_patience: float


def canonical_family(raw: str) -> str:
    value = raw.strip().lower()
    if value in ("exponential", "exp"):
        return "exponential"
    if value in ("erlang_k", "erlang"):
        return "erlang_k"
    if value in ("lognormal", "ln"):
        return "lognormal"
    if value in ("hyperexponential2", "h2"):
        return "hyperexponential2"
    raise ValueError(
        f"unsupported distribution family '{raw}'; "
        "expected exponential|erlang_k|lognormal|hyperexponential2"
    )


def parse_distribution_component(
    parent_cfg: dict[str, Any],
    key: str,
    model_name: str,
    allow_lognormal: bool = False,
) -> DistributionComponent:
    comp = parent_cfg.get(key)
    if not isinstance(comp, dict):
        raise ValueError(f"model '{model_name}': {key} must be an object")
    if set(comp.keys()) != {"distribution"}:
        raise ValueError(f"model '{model_name}': {key} must contain only 'distribution'")

    dist = comp.get("distribution")
    if not isinstance(dist, dict):
        raise ValueError(f"model '{model_name}': {key}.distribution must be an object")
    if set(dist.keys()) != {"family", "params"}:
        raise ValueError(
            f"model '{model_name}': {key}.distribution must contain exactly family and params"
        )

    family = canonical_family(str(dist.get("family")))
    params_raw = dist.get("params")
    if not isinstance(params_raw, dict):
        raise ValueError(f"model '{model_name}': {key}.distribution.params must be an object")

    if family == "exponential":
        if set(params_raw.keys()) != {"rate"}:
            raise ValueError(f"model '{model_name}': {key} exponential params must be {{rate}}")
        params: dict[str, float | int] = {"rate": float(params_raw["rate"])}
    elif family == "erlang_k":
        if set(params_raw.keys()) != {"k", "rate"}:
            raise ValueError(f"model '{model_name}': {key} erlang params must be {{k,rate}}")
        params = {"k": int(params_raw["k"]), "rate": float(params_raw["rate"])}
    elif family == "lognormal":
        if not allow_lognormal:
            raise ValueError(
                f"model '{model_name}': {key} does not support lognormal; "
                "lognormal is supported for service only"
            )
        if set(params_raw.keys()) != {"mean", "scv"}:
            raise ValueError(
                f"model '{model_name}': {key} lognormal params must be {{mean,scv}}"
            )
        params = {"mean": float(params_raw["mean"]), "scv": float(params_raw["scv"])}
    else:
        if set(params_raw.keys()) != {"p", "rate1", "rate2"}:
            raise ValueError(
                f"model '{model_name}': {key} hyperexponential2 params must be {{p,rate1,rate2}}"
            )
        params = {
            "p": float(params_raw["p"]),
            "rate1": float(params_raw["rate1"]),
            "rate2": float(params_raw["rate2"]),
        }

    return DistributionComponent(family=family, params=params)


def _load_config_header(path: Path) -> tuple[dict[str, Any], str, str, bool]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    simulation = raw.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("model config must contain object field 'simulation'")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("model config must contain object field 'model'")

    name = model.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model config must contain non-empty string field 'model.name'")
    alias = model.get("alias")
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("model config must contain non-empty string field 'model.alias'")

    normalize = simulation.get("normalize_service_mean_to_one")
    if normalize is None:
        normalize_flag = True
    elif isinstance(normalize, bool):
        normalize_flag = normalize
    else:
        raise ValueError("simulation.normalize_service_mean_to_one must be a boolean")

    return model, name, sanitize_alias(alias), normalize_flag


def is_tandem_config(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    model = raw.get("model")
    return isinstance(model, dict) and "queue1" in model


def load_model_config(path: Path) -> AnyModel:
    """Load a model config, returning ParsedModel or ParsedTandemModel."""
    model, name, alias, normalize_flag = _load_config_header(path)

    if "queue1" in model:
        queue1 = model.get("queue1")
        queue2 = model.get("queue2")
        if not isinstance(queue1, dict) or not isinstance(queue2, dict):
            raise ValueError("tandem model config must contain objects 'queue1' and 'queue2'")
        rho1 = queue1.get("traffic_intensity")
        if not isinstance(rho1, (int, float)) or not (0.0 < float(rho1) < 1.0):
            raise ValueError("queue1.traffic_intensity must be a number in (0,1)")
        return ParsedTandemModel(
            model_name=name,
            model_alias=alias,
            normalize_service_mean_to_one=normalize_flag,
            queue1=Queue1Spec(
                traffic_intensity=float(rho1),
                arrival=parse_distribution_component(queue1, "arrival", name),
                service=parse_distribution_component(queue1, "service", name),
            ),
            queue2=Queue2Spec(
                service=parse_distribution_component(queue2, "service", name),
                patience=parse_distribution_component(queue2, "patience", name),
            ),
        )

    return ParsedModel(
        model_name=name,
        model_alias=alias,
        normalize_service_mean_to_one=normalize_flag,
        arrival=parse_distribution_component(model, "arrival", name),
        service=parse_distribution_component(model, "service", name, allow_lognormal=True),
        patience=parse_distribution_component(model, "patience", name),
    )


def infer_k_from_patience(component: DistributionComponent) -> int:
    """First index k with F^(k)(0) > 0: k for Erlang-k patience, else 1."""
    if component.family == "erlang_k":
        return int(component.params["k"])
    return 1


def build_base_stats(model: AnyModel, k: int) -> BaseSystemStats:
    if model.is_tandem:
        assert isinstance(model, ParsedTandemModel)
        _, c_a2 = distribution_moments(model.queue1.arrival.family, model.queue1.arrival.params)
        service = model.queue2.service
        patience = model.queue2.patience
    else:
        assert isinstance(model, ParsedModel)
        _, c_a2 = distribution_moments(model.arrival.family, model.arrival.params)
        service = model.service
        patience = model.patience

    service_mean, c_s2 = distribution_moments(service.family, service.params)
    if model.normalize_service_mean_to_one:
        mu = 1.0
    else:
        mu = 1.0 / service_mean
    if not (mu > 0.0 and math.isfinite(mu)):
        raise ValueError(f"invalid derived service rate mu={mu}")

    beta_patience = float(distribution_beta_at_zero(patience.family, patience.params, k))
    if not (beta_patience > 0.0 and math.isfinite(beta_patience)):
        raise ValueError(
            f"invalid patience local coefficient F^(k)(0)/k!={beta_patience}; "
            "check patience distribution and chosen k"
        )

    h = float(k) / float(k + 1)
    return BaseSystemStats(
        mu=mu,
        c_a2=float(c_a2),
        c_s2=float(c_s2),
        k=k,
        h=h,
        beta_patience=beta_patience,
    )


def distribution_mean(component: DistributionComponent) -> float:
    mean, _ = distribution_moments(component.family, component.params)
    return float(mean)


def normalize_model_title(raw_name: str) -> str:
    """Turn a config's model.name into a LaTeX-ish plot title."""

    def latexize_tokens(title: str) -> str:
        title = title.replace("→", r" $\to$ ")
        title = title.replace("->", r" $\to$ ")
        title = re.sub(r"\s*-\s*¿\s*", r" $\to$ ", title)
        title = re.sub(r"(?<![A-Za-z0-9$])E_2(?![A-Za-z0-9])", r"$E_2$", title)
        title = re.sub(r"(?<![A-Za-z0-9$])E2(?![A-Za-z0-9])", r"$E_2$", title)
        title = re.sub(r"(?<![A-Za-z0-9$])H_2\((\d+)\)", r"$H_2(\1)$", title)
        title = re.sub(r"(?<![A-Za-z0-9$])H2\((\d+)\)", r"$H_2(\1)$", title)
        title = re.sub(r"(?<![A-Za-z0-9$])LN\((\d+),\s*(\d+)\)", r"$LN(\1,\2)$", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title

    title = raw_name.strip()
    if not title:
        return "model"
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    title = re.sub(r"\s+example\s*$", "", title, flags=re.IGNORECASE).strip()
    title = latexize_tokens(title)
    return title or raw_name.strip() or "model"


def model_plot_metadata(path: Path) -> tuple[str, float, str]:
    """Return (alias, base mean patience, plot title) for a model config."""
    model = load_model_config(path)
    patience_mean = distribution_mean(model.patience)
    if not (patience_mean > 0.0 and math.isfinite(patience_mean)):
        raise ValueError("base mean patience must be finite and > 0")
    return model.model_alias, patience_mean, normalize_model_title(model.model_name)
