#!/usr/bin/env python3
"""Diagnostics for the w_{c,k}(t) and b(c) tables.

Subcommands:
  overlay   w-table matrix vs single-curve solves  -> results/w_overlay_k<k>.png
  b         b(c) calibration curve                 -> results/b_calibration_k<k>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rqab.util import RESULTS_DIR, resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_overlay = sub.add_parser("overlay", help="w-table overlay figure")
    p_overlay.add_argument("--k", type=int, required=True)
    p_overlay.add_argument("--table", type=Path, default=None, help="w-table CSV override.")
    p_overlay.add_argument("--save", type=Path, default=None)
    p_overlay.add_argument("--no-show", action="store_true")

    p_b = sub.add_parser("b", help="b(c) calibration figure")
    p_b.add_argument("--k", type=int, required=True)
    p_b.add_argument("--table", type=Path, default=None, help="b-table CSV override.")
    p_b.add_argument("--save", type=Path, default=None)
    p_b.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    try:
        from rqab.plotting import diagnostics
    except ModuleNotFoundError as exc:
        print(f"warning: plotting dependencies unavailable ({exc})", file=sys.stderr)
        print("skipping plot generation because plotting dependencies are unavailable.", file=sys.stderr)
        return 0

    try:
        cwd = Path.cwd()
        if args.command == "overlay":
            save = (
                resolve(args.save, cwd)
                if args.save is not None
                else RESULTS_DIR / f"w_overlay_k{args.k}.png"
            )
            diagnostics.plot_w_overlay(
                k=args.k,
                table_path=resolve(args.table, cwd) if args.table is not None else None,
                save_path=save,
                no_show=args.no_show,
            )
        else:
            save = (
                resolve(args.save, cwd)
                if args.save is not None
                else RESULTS_DIR / f"b_calibration_k{args.k}.png"
            )
            diagnostics.plot_b_calibration(
                k=args.k,
                table_path=resolve(args.table, cwd) if args.table is not None else None,
                save_path=save,
                no_show=args.no_show,
            )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
