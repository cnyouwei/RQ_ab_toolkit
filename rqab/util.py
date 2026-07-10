"""Small shared helpers: paths, aliases, subprocesses, progress rendering."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_GRID_JSON = CONFIGS_DIR / "workload_lambda_alpha_grid_391.json"


def resolve(path: Path, cwd: Path | None = None) -> Path:
    base = cwd if cwd is not None else Path.cwd()
    return path if path.is_absolute() else (base / path)


def sanitize_alias(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "model"


def run_checked(cmd: list[str], title: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        message = [f"error: {title} failed", f"command: {' '.join(cmd)}"]
        if proc.stdout:
            message.append(f"stdout:\n{proc.stdout.strip()}")
        if proc.stderr:
            message.append(f"stderr:\n{proc.stderr.strip()}")
        raise RuntimeError("\n".join(message))


def require_int(row: dict[str, Any], key: str) -> int:
    if key not in row:
        raise ValueError(f"tuple is missing key '{key}'")
    return int(row[key])


def require_float(row: dict[str, Any], key: str) -> float:
    if key not in row:
        raise ValueError(f"tuple is missing key '{key}'")
    return float(row[key])


def require_str(row: dict[str, Any], key: str) -> str:
    if key not in row:
        raise ValueError(f"tuple is missing key '{key}'")
    return str(row[key])


def format_eta(seconds: float) -> str:
    if seconds < 0.0 or not seconds < float("inf"):
        return "--:--"
    total = int(round(seconds))
    hours = total // 3600
    rem = total % 3600
    minutes = rem // 60
    secs = rem % 60
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}:{secs:02d}"


def render_progress(
    completed: int,
    total: int,
    tuple_id: int,
    lam: float,
    alpha: float,
    start_time: float,
) -> None:
    frac = completed / max(total, 1)
    width = 36
    filled = int(round(width * frac))
    elapsed = time.time() - start_time
    if frac <= 1e-12:
        eta = float("inf")
    elif frac >= 1.0:
        eta = 0.0
    else:
        eta = elapsed * (1.0 - frac) / frac
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    msg = (
        f"\r{bar} {100.0 * frac:5.1f}% "
        f"tuple {completed}/{total} "
        f"(id={tuple_id}, lambda={lam:.6g}, alpha={alpha:.6g}) "
        f"ETA {format_eta(eta)}"
    )
    print(msg, end="", flush=True)
