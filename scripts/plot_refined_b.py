#!/usr/bin/env python3
"""Plot refined-RQ calibrated b(c) for k = 1, 2, 3 in matching panels.

Calibrated/capped and infeasible-match fallback branches are distinguished
-> results/refined_rq_b_tripanel.pdf
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
        help="b-table CSV overrides, one per k (default: results/b_table_k<k>.csv).",
    )
    parser.add_argument(
        "--c-min", type=float, default=-8.0, help="Left plot limit (default: -8)."
    )
    parser.add_argument(
        "--c-max", type=float, default=4.0, help="Right plot limit (default: 4)."
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.table is not None and len(args.table) != len(args.k):
        print("error: --table needs one path per --k value", file=sys.stderr)
        return 2

    try:
        from rqab.plotting.diagnostics import plot_refined_b_tripanel
    except ModuleNotFoundError as exc:
        print(f"error: plotting dependencies unavailable ({exc})", file=sys.stderr)
        return 1

    try:
        cwd = Path.cwd()
        plot_refined_b_tripanel(
            ks=args.k,
            table_paths=(
                [resolve(path, cwd) for path in args.table]
                if args.table is not None
                else None
            ),
            c_min=args.c_min,
            c_max=args.c_max,
            save_path=resolve(args.save, cwd) if args.save is not None else None,
            no_show=args.no_show,
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
