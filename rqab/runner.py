"""Grid drivers: the shared tuple-loop CSV writer and the MC-binary driver."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


def _staging_path(out_csv: Path) -> Path:
    """A unique staging file in the system temp dir for building out_csv."""
    fd, name = tempfile.mkstemp(prefix=out_csv.stem + "_", suffix=".csv.partial")
    os.close(fd)
    return Path(name)

from .util import render_progress, require_float, require_int

WORKLOAD_CSV_COLUMNS = [
    "tuple_id",
    "lambda",
    "alpha",
    "lambda_k",
    "lambda_form",
    "alpha_k",
    "mean_workload",
    "std_workload",
    "n_reps",
    "warmup_time",
    "sample_time",
    "threads_used",
    "seed",
    "runtime_seconds",
    "model_name",
]

SolveFn = Callable[[dict[str, Any]], dict[str, Any]]
ErrorRowFn = Callable[[dict[str, Any], Exception], dict[str, Any]]


def run_analytic_grid(
    tuples: list[dict[str, Any]],
    solve_fn: SolveFn,
    columns: list[str],
    out_csv: Path,
    force: bool = False,
    continue_on_error: bool = False,
    error_row_fn: ErrorRowFn | None = None,
) -> int:
    """Run a deterministic per-tuple solver over the grid, writing one CSV.

    Returns a process exit code (0 ok, 1 failure).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not force:
        print(f"Output already exists, skipping (use --force-rerun to overwrite): {out_csv}")
        return 0

    # Write to a staging file and move into place on success so an
    # interrupted run never leaves a truncated CSV at the final path (which
    # skip-if-exists logic would silently reuse).  The staging file lives in
    # the system temp dir, NOT next to out_csv: results/ may be inside a
    # cloud-synced folder (Dropbox), whose sync engine can revert an in-place
    # rename that races an in-flight upload of the staging file.
    partial_csv = _staging_path(out_csv)
    start = time.time()
    with partial_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        handle.flush()

        total = len(tuples)
        for idx, row in enumerate(tuples, start=1):
            tuple_id = require_int(row, "tuple_id")
            lam = require_float(row, "lambda")
            alpha = require_float(row, "alpha")
            try:
                solved = solve_fn(row)
                writer.writerow(solved)
                handle.flush()
            except Exception as exc:
                if not continue_on_error or error_row_fn is None:
                    print("\nerror: tuple solve failed.", file=sys.stderr)
                    print(f"tuple_id={tuple_id} lambda={lam} alpha={alpha}", file=sys.stderr)
                    print(f"details: {exc}", file=sys.stderr)
                    print(f"partial aggregate CSV retained at: {partial_csv}", file=sys.stderr)
                    return 1
                writer.writerow(error_row_fn(row, exc))
                handle.flush()

            render_progress(
                completed=idx,
                total=total,
                tuple_id=tuple_id,
                lam=lam,
                alpha=alpha,
                start_time=start,
            )

    shutil.move(str(partial_csv), str(out_csv))
    print("\ncompleted.")
    print(f"Aggregate CSV: {out_csv}")
    return 0


def run_workload_grid(
    tuples: list[dict[str, Any]],
    binary: Path,
    model_config: Path,
    out_csv: Path,
    threads: int | None = None,
    seed: int | None = None,
    summary_dir: Path | None = None,
    force: bool = False,
) -> int:
    """Run the workload Monte-Carlo binary for each grid tuple.

    Lambda/alpha are serialized with %.17g so the binary parses the exact
    doubles from the grid JSON (they also enter the per-tuple RNG seeds).
    Returns a process exit code.
    """
    if not binary.exists():
        print(f"error: binary not found: {binary}", file=sys.stderr)
        return 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not force:
        print(f"Output already exists, skipping (use --force-rerun to overwrite): {out_csv}")
        return 0

    keep_summaries = summary_dir is not None
    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if keep_summaries:
        assert summary_dir is not None
        summary_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="rqab_workload_grid_")
        summary_dir = Path(temp_ctx.name)

    # Same staging protocol as run_analytic_grid: an interrupted MC run must
    # not leave a truncated CSV where skip-if-exists would find it, and the
    # staging file must live outside any cloud-synced folder.
    partial_csv = _staging_path(out_csv)
    try:
        start = time.time()
        with partial_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=WORKLOAD_CSV_COLUMNS)
            writer.writeheader()
            csv_file.flush()

            total = len(tuples)
            for idx, row in enumerate(tuples, start=1):
                tuple_id = require_int(row, "tuple_id")
                lam = require_float(row, "lambda")
                alpha = require_float(row, "alpha")
                summary_path = summary_dir / f"tuple_{tuple_id:04d}_summary.json"

                cmd = [
                    str(binary),
                    "--config",
                    str(model_config),
                    "--lambda",
                    f"{lam:.17g}",
                    "--alpha",
                    f"{alpha:.17g}",
                    "--summary-json",
                    str(summary_path),
                ]
                if threads is not None:
                    cmd.extend(["--threads", str(threads)])
                if seed is not None:
                    cmd.extend(["--seed", str(seed)])

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    print("\nerror: tuple run failed.", file=sys.stderr)
                    print(f"tuple_id={tuple_id} lambda={lam} alpha={alpha}", file=sys.stderr)
                    print(f"command stderr:\n{proc.stderr}", file=sys.stderr)
                    print(f"partial aggregate CSV retained at: {partial_csv}", file=sys.stderr)
                    if keep_summaries:
                        print(f"per-tuple summaries retained in: {summary_dir}", file=sys.stderr)
                    return proc.returncode

                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)

                writer.writerow(
                    {
                        "tuple_id": tuple_id,
                        "lambda": lam,
                        "alpha": alpha,
                        "lambda_k": require_int(row, "lambda_k"),
                        "lambda_form": str(row.get("lambda_form", "")),
                        "alpha_k": require_int(row, "alpha_k"),
                        "mean_workload": float(summary["mean_workload"]),
                        "std_workload": float(summary["std_workload"]),
                        "n_reps": int(summary["n_reps"]),
                        "warmup_time": float(summary["warmup_time"]),
                        "sample_time": float(summary["sample_time"]),
                        "threads_used": int(summary["threads_used"]),
                        "seed": int(summary["seed"]),
                        "runtime_seconds": float(summary["runtime_seconds"]),
                        "model_name": str(summary["model_name"]),
                    }
                )
                csv_file.flush()

                if not keep_summaries:
                    summary_path.unlink(missing_ok=True)

                render_progress(
                    completed=idx,
                    total=total,
                    tuple_id=tuple_id,
                    lam=lam,
                    alpha=alpha,
                    start_time=start,
                )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    shutil.move(str(partial_csv), str(out_csv))
    print("\ncompleted.")
    print(f"Aggregate CSV: {out_csv}")
    if keep_summaries:
        print(f"Summary JSON directory: {summary_dir}")
    return 0


def default_workload_csv(results_dir: Path, model_alias: str) -> Path:
    return results_dir / f"workload_grid_aggregate_{model_alias}.csv"
