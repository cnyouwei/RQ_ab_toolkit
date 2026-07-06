#!/usr/bin/env python3
"""Heatmap of the refined-RQ drift parameter c on the ratio-plot grid.

Same (mean patience x arrival rate) grid as the ratio heatmaps, colored by
the c value each tuple feeds into the w_{c,k} lookup
->  results/c_heatmap_<alias>.pdf

A missing refined-RQ CSV is generated automatically via scripts/run_grid.py
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/workload_mm1m.json"),
        help="Model config JSON (single-station or tandem).",
    )
    parser.add_argument("--refined-csv", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None, help="Output figure path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    parser.add_argument("--no-auto-run", action="store_true", help="Fail if the input CSV is missing.")
    parser.add_argument("--force-rerun-grid", action="store_true", help="Regenerate the input CSV.")
    args = parser.parse_args()

    try:
        from rqab.plotting import ratio_panels
    except ModuleNotFoundError as exc:
        print(f"warning: plotting dependencies unavailable ({exc})", file=sys.stderr)
        print("skipping plot generation because plotting dependencies are unavailable.", file=sys.stderr)
        return 0

    try:
        cwd = Path.cwd()
        model_config = resolve(args.model_config, cwd)
        alias, patience_base_mean, title = model_plot_metadata(model_config)

        refined_csv = (
            resolve(args.refined_csv, cwd)
            if args.refined_csv is not None
            else RESULTS_DIR / f"refined_rq_grid_{alias}.csv"
        )
        save_path = (
            resolve(args.save, cwd)
            if args.save is not None
            else RESULTS_DIR / f"c_heatmap_{alias}.pdf"
        )

        if not refined_csv.exists() or args.force_rerun_grid:
            if args.no_auto_run:
                raise FileNotFoundError(f"refined CSV not found: {refined_csv} (auto-run disabled)")
            cmd = [
                sys.executable,
                str(SCRIPTS / "run_grid.py"),
                "--method",
                "refined",
                "--model-config",
                str(model_config),
                "--out-csv",
                str(refined_csv),
            ]
            if args.force_rerun_grid:
                cmd.append("--force-rerun")
            print(f"[auto-run] refined grid -> {refined_csv}")
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                raise RuntimeError(f"auto-run of refined grid failed (exit {proc.returncode})")

        ratio_panels.figure_c_heatmap(
            refined_csv=refined_csv,
            patience_base_mean=patience_base_mean,
            model_title=title,
            save_path=save_path,
            no_show=args.no_show,
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
