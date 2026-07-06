#!/usr/bin/env python3
"""Effective-IDW overlay figure (RQ_ab.tex fig:Var_approx).

Plots simulated effective IDW curves (solid; from the idw_sim binary's
model{idx}_*_idx{i}_curve.csv outputs) against the w-table approximation
(dashed) for each alpha = base^-i in the config.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rqab.util import RESULTS_DIR, resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="effective_idw_*.json config.")
    parser.add_argument(
        "--curves-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory holding the simulation curve CSVs (default: results/).",
    )
    parser.add_argument("--save", type=Path, default=None, help="Output PDF override.")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    try:
        from rqab.plotting import idw_curves
    except ModuleNotFoundError as exc:
        print(f"warning: plotting dependencies unavailable ({exc})", file=sys.stderr)
        print("skipping plot generation because plotting dependencies are unavailable.", file=sys.stderr)
        return 0

    try:
        cwd = Path.cwd()
        idw_curves.plot_idw_effective(
            config_path=resolve(args.config, cwd),
            curves_dir=resolve(args.curves_dir, cwd),
            out_pdf=resolve(args.save, cwd) if args.save is not None else None,
            no_show=args.no_show,
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
