#!/usr/bin/env python3
"""Plot the standardized first-RQ calibrated b curves for k = 1, 2, 3.

One panel per k, with exact-match and fluid-fallback regions distinguished
-> results/first_rq_b_tripanel.pdf
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
        "--q-min",
        type=float,
        default=-8.0,
        help="Smallest standardized load coordinate (default: -8).",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=4.0,
        help="Largest standardized load coordinate (default: 4).",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=601,
        help="Number of q samples per curve (default: 601).",
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    try:
        from rqab.plotting.first_calibration import plot_first_b_tripanel
    except ModuleNotFoundError as exc:
        print(f"error: plotting dependencies unavailable ({exc})", file=sys.stderr)
        return 1

    try:
        cwd = Path.cwd()
        plot_first_b_tripanel(
            ks=args.k,
            q_min=args.q_min,
            q_max=args.q_max,
            points=args.points,
            save_path=resolve(args.save, cwd) if args.save is not None else None,
            no_show=args.no_show,
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
