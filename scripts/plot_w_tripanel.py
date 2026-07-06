#!/usr/bin/env python3
"""Tripanel w_{c,k}(t) figure for k = 1, 2, 3 with one shared c colorbar.

3:1 landscape, LaTeX-rendered labels, one viridis colorbar for the integer
c values across all panels  ->  results/w_tripanel.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rqab.util import resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="k values, one panel each (default: 1 2 3).",
    )
    parser.add_argument(
        "--table",
        type=Path,
        nargs="+",
        default=None,
        help="w-table CSV overrides, one per k (default: results/w_table_matrix_k<k>.csv).",
    )
    parser.add_argument("--points", type=int, default=500, help="Plot points per curve.")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.table is not None and len(args.table) != len(args.k):
        print("error: --table needs one path per --k value", file=sys.stderr)
        return 2

    try:
        from rqab.plotting import diagnostics
    except ModuleNotFoundError as exc:
        print(f"warning: plotting dependencies unavailable ({exc})", file=sys.stderr)
        print("skipping plot generation because plotting dependencies are unavailable.", file=sys.stderr)
        return 0

    try:
        cwd = Path.cwd()
        diagnostics.plot_w_tripanel(
            ks=args.k,
            table_paths=[resolve(p, cwd) for p in args.table] if args.table is not None else None,
            plot_points=args.points,
            save_path=resolve(args.save, cwd) if args.save is not None else None,
            no_show=args.no_show,
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
