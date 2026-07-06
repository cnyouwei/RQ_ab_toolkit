#!/usr/bin/env python3
"""Run one grid experiment over the (lambda, alpha) tuple grid.

Methods
-------
workload   Monte-Carlo ground truth via the workload_mc binary
           -> results/workload_grid_aggregate_<alias>.csv
refined    Refined RQ fixed point (RQ_ab.tex eq:RQ_ab_2) + WG/Hazard/HG
           benchmark columns -> results/refined_rq_grid_<alias>.csv
first      First (crude) RQ fixed point (eq:RQ_ab_1) with closed-form b(c)
           -> results/first_rq_grid_<alias>.csv

Tandem models (configs with model.queue1/queue2) are detected automatically;
the same commands work unchanged.

Examples
--------
  python3 scripts/run_grid.py --method workload --model-config configs/workload_mm1m.json
  python3 scripts/run_grid.py --method refined  --model-config configs/workload_mm1m.json
  python3 scripts/run_grid.py --method first    --model-config configs/workload_tandem_h2_4e2_to_m1h2_4.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rqab import first as first_mod
from rqab import refined as refined_mod
from rqab import runner
from rqab.fixed_point import BisectOptions
from rqab.grids import build_s_grid, ensure_grid_json, load_grid
from rqab.models import build_base_stats, infer_k_from_patience, load_model_config
from rqab.secondary import QuadOptions
from rqab.tables import (
    BCalibrationInterpolator,
    WTableInterpolator,
    default_b_table_path,
    default_w_table_path,
    ensure_b_table,
    ensure_w_table,
)
from rqab.util import DEFAULT_GRID_JSON, RESULTS_DIR, BUILD_DIR, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=("workload", "refined", "first"),
        required=True,
        help="Which grid experiment to run.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/workload_mm1m.json"),
        help="Model config JSON (single-station or tandem).",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID_JSON,
        help="Tuple-grid JSON (auto-generated if missing).",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV path override.")
    parser.add_argument("--force-rerun", action="store_true", help="Overwrite existing output CSV.")
    parser.add_argument(
        "--no-auto-generate",
        action="store_true",
        help="Fail instead of auto-generating missing grid/w-table/b-table.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Emit status-tagged rows for failed tuples instead of aborting.",
    )

    mc = parser.add_argument_group("workload (Monte-Carlo) options")
    mc.add_argument(
        "--binary",
        type=Path,
        default=BUILD_DIR / "workload_mc",
        help="Path to the workload_mc simulator binary.",
    )
    mc.add_argument("--threads", type=int, default=None, help="Thread override for the binary.")
    mc.add_argument("--seed", type=int, default=None, help="Seed override for the binary.")
    mc.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Keep per-tuple summary JSONs in this directory.",
    )

    an = parser.add_argument_group("refined/first options")
    an.add_argument("--k", type=int, default=None, help="Patience index override (default: inferred).")
    an.add_argument("--w-table", type=Path, default=None, help="w-table CSV (refined only).")
    an.add_argument("--b-table", type=Path, default=None, help="b-table CSV (refined only).")
    an.add_argument("--b-override", type=float, default=None, help="Constant b override (first only).")
    an.add_argument("--n-s", type=int, default=800, help="Positive supremum-grid points.")
    an.add_argument("--s-min", type=float, default=1e-4, help="Smallest positive supremum-grid point.")
    an.add_argument("--s-max", type=float, default=1e8, help="Largest supremum-grid point.")
    an.add_argument("--bisect-abs-tol", type=float, default=1e-8)
    an.add_argument("--bisect-rel-tol", type=float, default=1e-8)
    an.add_argument("--bisect-max-iters", type=int, default=200)
    an.add_argument("--bracket-max-doublings", type=int, default=80)

    quad = parser.add_argument_group("WG/Hazard/HG quadrature options (refined)")
    quad.add_argument("--dy", type=float, default=0.02)
    quad.add_argument("--y-max-init", type=float, default=16.0)
    quad.add_argument("--y-max-limit", type=float, default=2048.0)
    quad.add_argument("--tail-log-gap", type=float, default=30.0)
    quad.add_argument("--tail-window", type=int, default=40)
    quad.add_argument("--min-y-for-tail", type=float, default=8.0)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()

    try:
        model_config = resolve(args.model_config, cwd)
        grid_json = resolve(args.grid, cwd)
        if not model_config.exists():
            raise FileNotFoundError(f"model config not found: {model_config}")
        ensure_grid_json(grid_json, auto_generate=not args.no_auto_generate)

        model = load_model_config(model_config)
        tuples = load_grid(grid_json)

        if args.method == "workload":
            out_csv = (
                resolve(args.out_csv, cwd)
                if args.out_csv is not None
                else runner.default_workload_csv(RESULTS_DIR, model.model_alias)
            )
            if args.threads is not None and args.threads < 0:
                raise ValueError("--threads must be >= 0")
            if args.seed is not None and args.seed < 0:
                raise ValueError("--seed must be >= 0")
            return runner.run_workload_grid(
                tuples=tuples,
                binary=resolve(args.binary, cwd),
                model_config=model_config,
                out_csv=out_csv,
                threads=args.threads,
                seed=args.seed,
                summary_dir=resolve(args.summary_dir, cwd) if args.summary_dir else None,
                force=args.force_rerun,
            )

        # Analytic methods.
        k_value = args.k if args.k is not None else infer_k_from_patience(model.patience)
        if k_value < 1:
            raise ValueError(f"derived k={k_value} is invalid")
        if (
            args.k is not None
            and args.out_csv is None
            and args.k != infer_k_from_patience(model.patience)
        ):
            raise ValueError(
                f"--k {args.k} differs from the model's inferred k="
                f"{infer_k_from_patience(model.patience)} but the default output name "
                "does not encode k; pass --out-csv to avoid overwriting the default CSV"
            )
        base = build_base_stats(model, k=k_value)
        s_grid = build_s_grid(s_min=args.s_min, s_max=args.s_max, n_s=args.n_s)
        bisect = BisectOptions(
            abs_tol=args.bisect_abs_tol,
            rel_tol=args.bisect_rel_tol,
            max_iters=args.bisect_max_iters,
            bracket_max_doublings=args.bracket_max_doublings,
        )

        if args.method == "refined":
            w_table_path = (
                resolve(args.w_table, cwd) if args.w_table is not None else default_w_table_path(k_value)
            )
            b_table_path = (
                resolve(args.b_table, cwd) if args.b_table is not None else default_b_table_path(k_value)
            )
            ensure_w_table(w_table_path, k=k_value, auto_generate=not args.no_auto_generate)
            ensure_b_table(
                b_table_path, w_table_path, k=k_value, auto_generate=not args.no_auto_generate
            )

            solver = refined_mod.RefinedSolver(
                model=model,
                base=base,
                b_table=BCalibrationInterpolator.from_csv(b_table_path),
                w_interp=WTableInterpolator.from_matrix_csv(w_table_path),
                s_grid=s_grid,
                bisect=bisect,
                quad=QuadOptions(
                    dy=args.dy,
                    y_max_init=args.y_max_init,
                    y_max_limit=args.y_max_limit,
                    tail_log_gap=args.tail_log_gap,
                    tail_window=args.tail_window,
                    min_y_for_tail=args.min_y_for_tail,
                ),
            )
            out_csv = (
                resolve(args.out_csv, cwd)
                if args.out_csv is not None
                else refined_mod.default_out_csv(RESULTS_DIR, model.model_alias)
            )
            return runner.run_analytic_grid(
                tuples=tuples,
                solve_fn=lambda row: solver.solve_one_tuple(
                    row, continue_on_error=args.continue_on_error
                ),
                columns=refined_mod.CSV_COLUMNS,
                out_csv=out_csv,
                force=args.force_rerun,
                continue_on_error=args.continue_on_error,
                error_row_fn=lambda row, exc: {
                    **{col: "" for col in refined_mod.CSV_COLUMNS},
                    "tuple_id": row.get("tuple_id", ""),
                    "lambda": row.get("lambda", ""),
                    "alpha": row.get("alpha", ""),
                    "solver_status": f"error:{exc}",
                    "status_secondary": f"error:{exc}",
                    "status_hg": f"error:{exc}",
                    "model_name": model.model_name,
                    "model_alias": model.model_alias,
                },
            )

        # first
        if args.b_override is not None and not (args.b_override >= 0.0):
            raise ValueError("--b-override must be >= 0")
        if abs(base.mu - 1.0) > 1e-12:
            print(
                f"warning: eq:b calibration assumes mu=1 but derived mu={base.mu}; "
                "results may be miscalibrated.",
                file=sys.stderr,
            )
        solver_first = first_mod.FirstSolver(
            model=model,
            base=base,
            u_grid=s_grid,
            b_override=args.b_override,
            bisect=bisect,
        )
        out_csv = (
            resolve(args.out_csv, cwd)
            if args.out_csv is not None
            else first_mod.default_out_csv(RESULTS_DIR, model.model_alias)
        )
        return runner.run_analytic_grid(
            tuples=tuples,
            solve_fn=solver_first.solve_one_tuple,
            columns=first_mod.CSV_COLUMNS,
            out_csv=out_csv,
            force=args.force_rerun,
            continue_on_error=args.continue_on_error,
            error_row_fn=lambda row, exc: {
                **{col: "" for col in first_mod.CSV_COLUMNS},
                "tuple_id": row.get("tuple_id", ""),
                "lambda": row.get("lambda", ""),
                "alpha": row.get("alpha", ""),
                "solver_status": f"error:{exc}",
                "model_name": model.model_name,
                "model_alias": model.model_alias,
            },
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
