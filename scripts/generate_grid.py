#!/usr/bin/env python3
"""Generate the explicit (lambda, alpha) tuple grid JSON.

Lambda spec: {1-2^-k, k=1..10} U {1+2^-k, k=10..-2} (23 values).
Alpha spec:  {2^-k, k=-3..13} (17 values). 391 tuples total.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rqab.grids import write_grid_json
from rqab.util import DEFAULT_GRID_JSON, resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_GRID_JSON, help="Output JSON path.")
    args = parser.parse_args()

    out_path = resolve(args.out)
    n = write_grid_json(out_path)
    print(f"Wrote {n} tuples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
