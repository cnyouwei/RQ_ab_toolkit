"""Shared helpers for the rqab Python test suite.

Importing this module bootstraps sys.path so ``import rqab`` works both under
``python -m unittest discover -s tests`` and when a test file is run directly.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterator

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIGS_DIR = REPO_ROOT / "configs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUN_GRID_SCRIPT = SCRIPTS_DIR / "run_grid.py"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Return (rows, fieldnames) of a CSV file."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


@contextmanager
def temp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="rqab_test_") as tmpdir:
        yield Path(tmpdir)


def run_cli(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def fail_message(title: str, cmd: list[str], proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f"{title}\n"
        f"cmd: {' '.join(cmd)}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


def write_test_w_table(path: Path) -> Path:
    """Write a small deterministic w-table suitable for solver tests."""
    write_csv(
        path,
        ["c", "0", "0.1", "10", "10000"],
        [
            {"c": -20.0, "0": 1.0, "0.1": 0.98, "10": 0.90, "10000": 0.85},
            {"c": 0.0, "0": 1.0, "0.1": 0.90, "10": 0.70, "10000": 0.60},
            {"c": 20.0, "0": 1.0, "0.1": 0.70, "10": 0.30, "10000": 0.20},
        ],
    )
    return path


def write_test_b_table(path: Path, *, k: int) -> Path:
    """Write a status-free b-table using the current calibration schema."""
    if k < 1:
        raise ValueError("k must be >= 1")
    write_csv(
        path,
        ["c", "b", "psi", "z_model", "abs_error", "a_psi", "u_star"],
        [
            {
                "c": -20.0,
                "b": 1.40,
                "psi": 0.05,
                "z_model": "nan",
                "abs_error": 0.0,
                "a_psi": -20.01,
                "u_star": 0.0025,
            },
            {
                "c": 0.0,
                "b": 1.0,
                "psi": 0.5,
                "z_model": 0.5,
                "abs_error": 0.0,
                "a_psi": -0.5,
                "u_star": 0.5,
            },
            {
                "c": 20.0,
                "b": math.sqrt(2.0) if k == 1 else 0.0,
                "psi": 1.0,
                "z_model": "nan",
                "abs_error": 0.1,
                "a_psi": 0.01,
                "u_star": "nan",
            },
        ],
    )
    return path


def write_test_calibration_tables(root: Path, *, k: int) -> tuple[Path, Path]:
    """Return explicit temporary (w-table, b-table) paths for a patience index."""
    return (
        write_test_w_table(root / f"w_table_k{k}.csv"),
        write_test_b_table(root / f"b_table_k{k}.csv", k=k),
    )


def tiny_grid_payload() -> dict:
    """Three (lambda, alpha) tuples spanning under-, critically- and over-loaded."""
    return {
        "tuples": [
            {
                "tuple_id": 1,
                "lambda": 0.8,
                "alpha": 2.0,
                "lambda_k": 1,
                "lambda_form": "1-2^-k",
                "alpha_k": -1,
            },
            {
                "tuple_id": 2,
                "lambda": 1.0,
                "alpha": 1.0,
                "lambda_k": 0,
                "lambda_form": "1+2^-k",
                "alpha_k": 0,
            },
            {
                "tuple_id": 3,
                "lambda": 1.2,
                "alpha": 0.5,
                "lambda_k": 3,
                "lambda_form": "1+2^-k",
                "alpha_k": 1,
            },
        ]
    }


def _simulation_block() -> dict[str, Any]:
    return {
        "warmup_time": 1000.0,
        "sample_time": 10000.0,
        "replications": 16,
        "threads": 1,
        "seed": 12345,
        "normalize_service_mean_to_one": True,
    }


H2_MEAN1_SCV4_PARAMS = {
    "p": 0.8872983346207416,
    "rate1": 1.7745966692414832,
    "rate2": 0.22540333075851682,
}


def valid_model_payload(alias: str = "mm1m") -> dict:
    """Single-station M/M/1+M model config payload."""
    return {
        "simulation": _simulation_block(),
        "model": {
            "name": f"M/M/1+M ({alias})",
            "alias": alias,
            "arrival": {
                "distribution": {"family": "exponential", "params": {"rate": 1.0}}
            },
            "service": {
                "distribution": {"family": "exponential", "params": {"rate": 1.0}}
            },
            "patience": {
                "distribution": {"family": "exponential", "params": {"rate": 1.0}}
            },
        },
    }


def valid_e2_patience_payload(alias: str = "mm1e2") -> dict:
    """Single-station M/M/1+E2 payload (f(0) = 0 patience -> hazard secondary)."""
    return {
        "simulation": _simulation_block(),
        "model": {
            "name": f"M/M/1+E_2 ({alias})",
            "alias": alias,
            "arrival": {
                "distribution": {"family": "exponential", "params": {"rate": 1.0}}
            },
            "service": {
                "distribution": {"family": "exponential", "params": {"rate": 1.0}}
            },
            "patience": {
                "distribution": {"family": "erlang_k", "params": {"k": 2, "rate": 2.0}}
            },
        },
    }


def valid_lognormal_model_payload(alias: str = "mln1_41h2_4") -> dict:
    """Single-station M/LN(1,4)/1+H2(4) payload (lognormal service)."""
    return {
        "simulation": _simulation_block(),
        "model": {
            "name": "M/LN(1,4)/1+H2(4)",
            "alias": alias,
            "arrival": {
                "distribution": {"family": "exponential", "params": {"rate": 1.0}}
            },
            "service": {
                "distribution": {"family": "lognormal", "params": {"mean": 1.0, "scv": 4.0}}
            },
            "patience": {
                "distribution": {
                    "family": "hyperexponential2",
                    "params": dict(H2_MEAN1_SCV4_PARAMS),
                }
            },
        },
    }


def tandem_h2e2_to_m1h2_payload(alias: str = "tandem_h2_4e2_to_m1h2_4") -> dict:
    """Tandem H2(4)/E2/1 -> ./M/1+H2(4) payload (queue1/queue2 blocks)."""
    return {
        "simulation": _simulation_block(),
        "model": {
            "name": "H_2(4)/E_2/1 -> ./M/1+H_2(4)",
            "alias": alias,
            "queue1": {
                "traffic_intensity": 0.9,
                "arrival": {
                    "distribution": {
                        "family": "hyperexponential2",
                        "params": dict(H2_MEAN1_SCV4_PARAMS),
                    }
                },
                "service": {
                    "distribution": {
                        "family": "erlang_k",
                        "params": {"k": 2, "rate": 2.0},
                    }
                },
            },
            "queue2": {
                "service": {
                    "distribution": {
                        "family": "exponential",
                        "params": {"rate": 1.0},
                    }
                },
                "patience": {
                    "distribution": {
                        "family": "hyperexponential2",
                        "params": dict(H2_MEAN1_SCV4_PARAMS),
                    }
                },
            },
        },
    }
