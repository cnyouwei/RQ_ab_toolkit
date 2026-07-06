#!/usr/bin/env python3
"""Signed-relative-error heatmap panels vs the workload MC ground truth.

Panel presets
-------------
ratio      1 panel : refined RQ / simulation        -> refined_rq_ratio_<alias>.pdf
twopanel   2 panels: first RQ | refined RQ          -> approx_ratio_twopanel_<alias>.pdf
tripanel   3 panels: refined | WG-or-Hazard | HG    -> approx_ratio_tripanel_<alias>.pdf

Missing input CSVs are generated automatically via scripts/run_grid.py
(disable with --no-auto-run).  Tandem model configs work unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rqab.models import model_plot_metadata
from rqab.util import RESULTS_DIR, resolve

SCRIPTS = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--panels", choices=("ratio", "twopanel", "tripanel"), required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/workload_mm1m.json"),
        help="Model config JSON (single-station or tandem).",
    )
    parser.add_argument("--workload-csv", type=Path, default=None)
    parser.add_argument("--refined-csv", type=Path, default=None)
    parser.add_argument("--first-rq-csv", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None, help="Output figure path.")
    parser.add_argument("--vmin", type=float, default=None, help="Color min (percent, >= -30).")
    parser.add_argument("--vmax", type=float, default=None, help="Color max (percent, <= 30).")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    parser.add_argument("--no-auto-run", action="store_true", help="Fail if input CSVs are missing.")
    parser.add_argument("--threads", type=int, default=None, help="Threads for auto-run workload MC.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for auto-run workload MC.")
    parser.add_argument("--force-rerun-grid", action="store_true", help="Regenerate input CSVs.")
    return parser.parse_args()


def ensure_csv(
    args: argparse.Namespace, csv_path: Path, method: str, model_config: Path
) -> None:
    if csv_path.exists() and not args.force_rerun_grid:
        return
    if args.no_auto_run:
        raise FileNotFoundError(f"{method} CSV not found: {csv_path} (auto-run disabled)")
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_grid.py"),
        "--method",
        method,
        "--model-config",
        str(model_config),
        "--out-csv",
        str(csv_path),
    ]
    if args.force_rerun_grid:
        cmd.append("--force-rerun")
    if method == "workload":
        if args.threads is not None:
            cmd += ["--threads", str(args.threads)]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
    print(f"[auto-run] {method} grid -> {csv_path}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"auto-run of {method} grid failed (exit {proc.returncode})")


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()

    try:
        from rqab.plotting import ratio_panels
    except ModuleNotFoundError as exc:
        print(f"warning: plotting dependencies unavailable ({exc})", file=sys.stderr)
        print("skipping plot generation because plotting dependencies are unavailable.", file=sys.stderr)
        return 0

    try:
        model_config = resolve(args.model_config, cwd)
        alias, patience_base_mean, title = model_plot_metadata(model_config)

        workload_csv = (
            resolve(args.workload_csv, cwd)
            if args.workload_csv is not None
            else RESULTS_DIR / f"workload_grid_aggregate_{alias}.csv"
        )
        refined_csv = (
            resolve(args.refined_csv, cwd)
            if args.refined_csv is not None
            else RESULTS_DIR / f"refined_rq_grid_{alias}.csv"
        )
        first_csv = (
            resolve(args.first_rq_csv, cwd)
            if args.first_rq_csv is not None
            else RESULTS_DIR / f"first_rq_grid_{alias}.csv"
        )

        default_name = {
            "ratio": f"refined_rq_ratio_{alias}.pdf",
            "twopanel": f"approx_ratio_twopanel_{alias}.pdf",
            "tripanel": f"approx_ratio_tripanel_{alias}.pdf",
        }[args.panels]
        save_path = resolve(args.save, cwd) if args.save is not None else RESULTS_DIR / default_name

        ensure_csv(args, workload_csv, "workload", model_config)
        ensure_csv(args, refined_csv, "refined", model_config)
        if args.panels == "twopanel":
            ensure_csv(args, first_csv, "first", model_config)

        common = dict(
            workload_csv=workload_csv,
            refined_csv=refined_csv,
            patience_base_mean=patience_base_mean,
            model_title=title,
            save_path=save_path,
            vmin=args.vmin,
            vmax=args.vmax,
            no_show=args.no_show,
        )
        if args.panels == "ratio":
            ratio_panels.figure_ratio(**common)
        elif args.panels == "twopanel":
            ratio_panels.figure_twopanel(first_csv=first_csv, **common)
        else:
            ratio_panels.figure_tripanel(**common)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
