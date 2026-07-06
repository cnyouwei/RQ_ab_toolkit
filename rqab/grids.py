"""The (lambda, alpha) tuple grid and the s/u supremum grids."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def build_lambda_values() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for k in range(1, 11):
        values.append(
            {
                "lambda": 1.0 - 2.0 ** (-k),
                "lambda_k": k,
                "lambda_form": "1-2^-k",
            }
        )
    for k in range(10, -3, -1):
        values.append(
            {
                "lambda": 1.0 + 2.0 ** (-k),
                "lambda_k": k,
                "lambda_form": "1+2^-k",
            }
        )
    return values


def build_alpha_values() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for k in range(-3, 14):
        values.append({"alpha": 2.0 ** (-k), "alpha_k": k})
    return values


def build_tuples() -> list[dict[str, object]]:
    tuples: list[dict[str, object]] = []
    tuple_id = 1
    for lv in build_lambda_values():
        for av in build_alpha_values():
            tuples.append(
                {
                    "tuple_id": tuple_id,
                    "lambda": lv["lambda"],
                    "alpha": av["alpha"],
                    "lambda_k": lv["lambda_k"],
                    "lambda_form": lv["lambda_form"],
                    "alpha_k": av["alpha_k"],
                }
            )
            tuple_id += 1
    return tuples


def write_grid_json(out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tuples = build_tuples()
    payload = {
        "metadata": {
            "lambda_count": 23,
            "alpha_count": 17,
            "tuple_count": len(tuples),
            "lambda_spec": "{1-2^-k, k=1..10} U {1+2^-k, k=10..-2}",
            "alpha_spec": "{2^-k, k=-3..13}",
        },
        "tuples": tuples,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return len(tuples)


def ensure_grid_json(grid_json: Path, auto_generate: bool = True) -> None:
    if grid_json.exists():
        return
    if not auto_generate:
        raise FileNotFoundError(f"grid JSON not found: {grid_json}")
    n = write_grid_json(grid_json)
    print(f"[auto-gen] created grid JSON ({n} tuples): {grid_json}")


def load_grid(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    tuples = raw.get("tuples")
    if not isinstance(tuples, list) or not tuples:
        raise ValueError("grid JSON must contain non-empty array field 'tuples'")
    return tuples


def build_s_grid(s_min: float, s_max: float, n_s: int) -> list[float]:
    """Log-uniform grid on [s_min, s_max] with 0.0 prepended (supremum scan)."""
    if n_s < 2:
        raise ValueError("n_s must be >= 2")
    if not (s_min > 0.0 and s_max > s_min):
        raise ValueError("require 0 < s_min < s_max")

    lo = math.log(s_min)
    hi = math.log(s_max)
    step = (hi - lo) / float(n_s - 1)
    positive = [math.exp(lo + step * i) for i in range(n_s)]
    return [0.0] + positive
