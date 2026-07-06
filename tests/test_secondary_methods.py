#!/usr/bin/env python3
"""Tests for the WG/Hazard/HG benchmark approximations (rqab.secondary).

Port of the pre-refactor tests/test_other_methods_grid.py.  The numeric unit
tests keep the old golden anchors; the end-to-end CLI tests are retargeted at
scripts/run_grid.py --method refined, whose combined output CSV now carries
the z_secondary/z_wg/z_hazard/z_hg and status_secondary/status_hg columns.
"""
from __future__ import annotations

import math
import sys
import unittest

import helpers
from helpers import (
    B_TABLE_K1,
    CONFIGS_DIR,
    RUN_GRID_SCRIPT,
    W_TABLE_K1,
    read_csv_rows,
    temp_dir,
    tiny_grid_payload,
    valid_e2_patience_payload,
    valid_lognormal_model_payload,
    valid_model_payload,
    write_json,
)

from rqab import refined as refined_mod
from rqab.models import build_base_stats, infer_k_from_patience, load_model_config
from rqab.secondary import (
    QuadOptions,
    build_secondary_stats,
    classify_secondary_method,
    compute_hazard,
    compute_hg,
    compute_wg,
    inverse_mills_ratio_upper,
    inverse_mills_ratio_upper_cf,
)


def stats_for_config(config_name: str):
    """(model, SecondaryStats) for a config in configs/."""
    model = load_model_config(CONFIGS_DIR / config_name)
    k = infer_k_from_patience(model.patience)
    base = build_base_stats(model, k=k)
    stats = build_secondary_stats(model, mu=base.mu, c_a2=base.c_a2, c_s2=base.c_s2)
    return model, stats


def quad_stub() -> QuadOptions:
    """The old CLI-args stub used by the smoke tests."""
    return QuadOptions(
        dy=0.05,
        y_max_init=8.0,
        y_max_limit=64.0,
        tail_log_gap=20.0,
        tail_window=20,
        min_y_for_tail=4.0,
    )


def quad_with_dy(dy: float) -> QuadOptions:
    return QuadOptions(
        dy=float(dy),
        y_max_init=16.0,
        y_max_limit=16.0,
        tail_log_gap=30.0,
        tail_window=40,
        min_y_for_tail=8.0,
    )


class TestSecondaryMethodInternals(unittest.TestCase):
    def test_method_classification(self) -> None:
        _, mm1m = stats_for_config("workload_mm1m.json")
        self.assertEqual(mm1m.secondary_method, "wg")

        _, mm1h2 = stats_for_config("workload_mm1h2_4.json")
        self.assertEqual(mm1h2.secondary_method, "wg")

        _, mm1e2 = stats_for_config("workload_mm1e2.json")
        self.assertEqual(mm1e2.secondary_method, "hazard")

        # Direct classifier: WG iff F'(0) > 0.
        self.assertEqual(classify_secondary_method(1.0), "wg")
        self.assertEqual(classify_secondary_method(0.0), "hazard")
        self.assertEqual(classify_secondary_method(-0.0), "hazard")

    def test_inverse_mills_consistency(self) -> None:
        # Continued fraction agrees with phi/sf in the crossover region.
        for x in (6.0, 6.5, 8.0):
            phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
            sf = 0.5 * math.erfc(x / math.sqrt(2.0))
            direct = phi / sf
            self.assertTrue(
                math.isclose(inverse_mills_ratio_upper_cf(x), direct, rel_tol=1e-10)
            )
            self.assertTrue(
                math.isclose(inverse_mills_ratio_upper(x), direct, rel_tol=1e-10)
            )
        with self.assertRaises(ValueError):
            inverse_mills_ratio_upper_cf(0.0)

    def test_wg_smoke(self) -> None:
        _, stats = stats_for_config("workload_mm1m.json")
        z, status = compute_wg(lam=1.25, alpha=0.5, stats=stats)
        self.assertTrue(status.startswith("ok"))
        self.assertGreaterEqual(z, 0.0)
        self.assertTrue(z == z)

    def test_hazard_smoke(self) -> None:
        model, stats = stats_for_config("workload_mm1e2.json")
        res = compute_hazard(
            lam=1.25,
            alpha=0.5,
            stats=stats,
            patience=model.patience,
            opts=quad_stub(),
        )
        self.assertTrue(res.status.startswith("ok"))
        self.assertGreaterEqual(res.z, 0.0)
        self.assertTrue(res.z == res.z)

    def test_hg_smoke_mm1m_and_mm1e2(self) -> None:
        model_m, stats_m = stats_for_config("workload_mm1m.json")
        res_m = compute_hg(
            lam=1.25,
            alpha=0.5,
            stats=stats_m,
            patience=model_m.patience,
            opts=quad_stub(),
        )
        self.assertTrue(res_m.status.startswith("ok"))
        self.assertGreaterEqual(res_m.z, 0.0)

        model_e2, stats_e2 = stats_for_config("workload_mm1e2.json")
        res_e2 = compute_hg(
            lam=1.25,
            alpha=0.5,
            stats=stats_e2,
            patience=model_e2.patience,
            opts=quad_stub(),
        )
        self.assertTrue(res_e2.status.startswith("ok"))
        self.assertGreaterEqual(res_e2.z, 0.0)

    def test_wg_inverse_mills_large_x_accuracy(self) -> None:
        _, stats = stats_for_config("workload_mm1m.json")
        lam = 0.5
        alpha = 2.0**-8

        z_impl, status = compute_wg(lam=lam, alpha=alpha, stats=stats)
        self.assertTrue(status.startswith("ok"))

        rho = lam / stats.mu
        tilde_cx2 = rho * stats.c_a2 + min(rho, 1.0) * stats.c_s2
        c = (rho - 1.0) / math.sqrt(alpha)
        xi = -math.sqrt(2.0 * stats.mu) * c / math.sqrt(stats.f1_at_zero * tilde_cx2)
        phi = math.exp(-0.5 * xi * xi) / math.sqrt(2.0 * math.pi)
        sf = 0.5 * math.erfc(xi / math.sqrt(2.0))
        mills = phi / sf
        term = c / stats.f1_at_zero + mills * math.sqrt(
            tilde_cx2 / (2.0 * stats.mu * stats.f1_at_zero)
        )
        z_direct = max(0.0, (alpha ** -0.5) * term)

        self.assertTrue(math.isclose(z_impl, z_direct, rel_tol=1e-6, abs_tol=1e-9))

    def test_hg_long_patience_underload_adaptive_head(self) -> None:
        model, stats = stats_for_config("workload_mm1h2_4.json")
        lam = 0.5
        alpha = 2.0**-13

        coarse = compute_hg(
            lam=lam,
            alpha=alpha,
            stats=stats,
            patience=model.patience,
            opts=quad_with_dy(0.02),
        )
        reference = compute_hg(
            lam=lam,
            alpha=alpha,
            stats=stats,
            patience=model.patience,
            opts=quad_with_dy(0.002),
        )

        self.assertTrue(coarse.status.startswith("ok"))
        self.assertTrue(reference.status.startswith("ok"))
        self.assertGreater(coarse.z, 0.5)
        self.assertTrue(math.isclose(coarse.z, reference.z, rel_tol=1e-6, abs_tol=1e-8))

    def test_hazard_long_patience_underload_adaptive_head(self) -> None:
        model, stats = stats_for_config("workload_mm1e2.json")
        lam = 0.5
        alpha = 2.0**-13

        coarse = compute_hazard(
            lam=lam,
            alpha=alpha,
            stats=stats,
            patience=model.patience,
            opts=quad_with_dy(0.02),
        )
        reference = compute_hazard(
            lam=lam,
            alpha=alpha,
            stats=stats,
            patience=model.patience,
            opts=quad_with_dy(0.002),
        )

        self.assertTrue(coarse.status.startswith("ok"))
        self.assertTrue(reference.status.startswith("ok"))
        self.assertGreater(coarse.z, 1.0)
        self.assertTrue(math.isclose(coarse.z, reference.z, rel_tol=1e-6, abs_tol=1e-8))

    def test_hg_long_patience_near_critical_overload_adaptive_head(self) -> None:
        model, stats = stats_for_config("workload_mm1h2_4.json")
        lam = 1.0009765625
        alpha = 2.0**-13

        coarse = compute_hg(
            lam=lam,
            alpha=alpha,
            stats=stats,
            patience=model.patience,
            opts=quad_with_dy(0.02),
        )
        reference = compute_hg(
            lam=lam,
            alpha=alpha,
            stats=stats,
            patience=model.patience,
            opts=quad_with_dy(0.002),
        )

        self.assertTrue(coarse.status.startswith("ok"))
        self.assertTrue(reference.status.startswith("ok"))
        self.assertGreater(coarse.z, 40.0)
        self.assertTrue(math.isclose(coarse.z, reference.z, rel_tol=1e-2, abs_tol=1e-6))


QUAD_CLI_ARGS = [
    "--dy",
    "0.05",
    "--y-max-init",
    "8",
    "--y-max-limit",
    "64",
    "--tail-log-gap",
    "20",
    "--tail-window",
    "20",
    "--min-y-for-tail",
    "4",
]

S_GRID_CLI_ARGS = ["--n-s", "80", "--s-min", "1e-3", "--s-max", "1e4"]


def run_refined_cli(model_path, grid_path, out_csv, extra=(), quad_args=tuple(QUAD_CLI_ARGS)):
    cmd = [
        sys.executable,
        str(RUN_GRID_SCRIPT),
        "--method",
        "refined",
        "--model-config",
        str(model_path),
        "--grid",
        str(grid_path),
        "--out-csv",
        str(out_csv),
        "--no-auto-generate",
        *S_GRID_CLI_ARGS,
        *quad_args,
        *extra,
    ]
    return cmd, helpers.run_cli(cmd)


class TestSecondaryMethodsCLI(unittest.TestCase):
    """Old run_other_methods_grid.py CLI tests, retargeted at run_grid.py --method refined."""

    def test_tiny_grid_end_to_end(self) -> None:
        with temp_dir() as tmp:
            model_path = tmp / "workload_mm1m.json"
            grid_path = tmp / "grid.json"
            out_csv = tmp / "out.csv"
            payload = valid_model_payload()
            write_json(model_path, payload)
            write_json(grid_path, tiny_grid_payload())

            cmd, proc = run_refined_cli(
                model_path,
                grid_path,
                out_csv,
                extra=("--w-table", str(W_TABLE_K1), "--b-table", str(B_TABLE_K1)),
            )
            if proc.returncode != 0:
                self.fail(helpers.fail_message("tiny-grid refined (secondary) run failed.", cmd, proc))
            self.assertTrue(out_csv.exists())

            rows, fieldnames = read_csv_rows(out_csv)
            self.assertEqual(len(rows), 3)
            for col in refined_mod.CSV_COLUMNS:
                self.assertIn(col, fieldnames)

            # Exponential patience: F'(0) = 1 > 0 -> WG secondary everywhere.
            model = load_model_config(model_path)
            base = build_base_stats(model, k=infer_k_from_patience(model.patience))
            stats = build_secondary_stats(model, mu=base.mu, c_a2=base.c_a2, c_s2=base.c_s2)

            for row in rows:
                self.assertTrue(str(row["status_secondary"]).startswith("ok"))
                self.assertTrue(str(row["status_hg"]).startswith("ok"))
                self.assertEqual(row["secondary_method"], "wg")
                self.assertEqual(row["z_hazard"], "")
                self.assertEqual(float(row["z_secondary"]), float(row["z_wg"]))
                self.assertGreater(float(row["z_hg"]), 0.0)

                # CSV values must match the library WG formula exactly.
                z_expected, status = compute_wg(
                    lam=float(row["lambda"]), alpha=float(row["alpha"]), stats=stats
                )
                self.assertTrue(status.startswith("ok"))
                self.assertTrue(
                    math.isclose(float(row["z_wg"]), z_expected, rel_tol=1e-10, abs_tol=1e-12)
                )

            # Golden anchor: at (lambda, alpha) = (1, 1), rho = 1 so c = 0 and
            # z_wg = sqrt(2/pi) for the M/M/1+M base stats.
            row2 = next(r for r in rows if int(r["tuple_id"]) == 2)
            self.assertAlmostEqual(float(row2["z_wg"]), math.sqrt(2.0 / math.pi), places=10)

    def test_tiny_grid_e2_patience_hazard_end_to_end(self) -> None:
        with temp_dir() as tmp:
            model_path = tmp / "workload_mm1e2.json"
            grid_path = tmp / "grid.json"
            out_csv = tmp / "out_e2.csv"
            write_json(model_path, valid_e2_patience_payload())
            write_json(grid_path, tiny_grid_payload())

            # Erlang-2 patience -> k=2; rely on the default k2 tables in results/.
            cmd, proc = run_refined_cli(model_path, grid_path, out_csv)
            if proc.returncode != 0:
                self.fail(helpers.fail_message("tiny-grid E2-patience refined run failed.", cmd, proc))

            rows, _ = read_csv_rows(out_csv)
            self.assertEqual(len(rows), 3)

            model = load_model_config(model_path)
            base = build_base_stats(model, k=infer_k_from_patience(model.patience))
            stats = build_secondary_stats(model, mu=base.mu, c_a2=base.c_a2, c_s2=base.c_s2)

            for row in rows:
                self.assertTrue(str(row["status_secondary"]).startswith("ok"))
                self.assertTrue(str(row["status_hg"]).startswith("ok"))
                self.assertEqual(row["secondary_method"], "hazard")
                self.assertEqual(row["z_wg"], "")
                self.assertEqual(float(row["z_secondary"]), float(row["z_hazard"]))

            # CSV hazard/HG values reproduce the library integrators under the
            # same quadrature options that were passed on the command line.
            row3 = next(r for r in rows if int(r["tuple_id"]) == 3)
            hazard = compute_hazard(
                lam=1.2, alpha=0.5, stats=stats, patience=model.patience, opts=quad_stub()
            )
            hg = compute_hg(
                lam=1.2, alpha=0.5, stats=stats, patience=model.patience, opts=quad_stub()
            )
            self.assertTrue(
                math.isclose(float(row3["z_hazard"]), hazard.z, rel_tol=1e-9, abs_tol=1e-12)
            )
            self.assertTrue(
                math.isclose(float(row3["z_hg"]), hg.z, rel_tol=1e-9, abs_tol=1e-12)
            )

    def test_tiny_grid_lognormal_service_end_to_end(self) -> None:
        with temp_dir() as tmp:
            model_path = tmp / "workload_mln1_41h2_4.json"
            grid_path = tmp / "grid.json"
            out_csv = tmp / "out_ln.csv"
            write_json(model_path, valid_lognormal_model_payload())
            write_json(grid_path, tiny_grid_payload())

            # High-variability model: keep the default quadrature limits so
            # the HG tail can converge (the old test also used the defaults).
            cmd, proc = run_refined_cli(
                model_path,
                grid_path,
                out_csv,
                extra=("--w-table", str(W_TABLE_K1), "--b-table", str(B_TABLE_K1)),
                quad_args=(),
            )
            if proc.returncode != 0:
                self.fail(
                    helpers.fail_message("tiny-grid lognormal-service refined run failed.", cmd, proc)
                )
            self.assertTrue(out_csv.exists())

            rows, _ = read_csv_rows(out_csv)
            self.assertEqual(len(rows), 3)
            for row in rows:
                # H2(4) patience has F'(0) > 0 -> WG secondary.
                self.assertEqual(row["secondary_method"], "wg")
                self.assertTrue(str(row["status_secondary"]).startswith("ok"))
                self.assertTrue(str(row["status_hg"]).startswith("ok"))

    def test_continue_on_error_status_rows(self) -> None:
        with temp_dir() as tmp:
            model_path = tmp / "workload_mm1m.json"
            grid_path = tmp / "grid_bad.json"
            out_csv = tmp / "out_bad.csv"
            write_json(model_path, valid_model_payload())
            bad_grid = {
                "tuples": [
                    {
                        "tuple_id": 1,
                        "lambda": 1.0,
                        "alpha": 0.0,
                        "lambda_k": 0,
                        "lambda_form": "custom",
                        "alpha_k": 0,
                    },
                    {
                        "tuple_id": 2,
                        "lambda": 1.0,
                        "alpha": 1.0,
                        "lambda_k": 0,
                        "lambda_form": "custom",
                        "alpha_k": 0,
                    },
                ]
            }
            write_json(grid_path, bad_grid)

            cmd, proc = run_refined_cli(
                model_path,
                grid_path,
                out_csv,
                extra=(
                    "--w-table",
                    str(W_TABLE_K1),
                    "--b-table",
                    str(B_TABLE_K1),
                    "--continue-on-error",
                ),
            )
            if proc.returncode != 0:
                self.fail(
                    helpers.fail_message("continue-on-error run failed unexpectedly.", cmd, proc)
                )

            rows, _ = read_csv_rows(out_csv)
            self.assertEqual(len(rows), 2)

            # Adapted to the combined-CSV schema: the failed tuple is tagged
            # in solver_status AND status_secondary/status_hg.
            bad = next(r for r in rows if int(r["tuple_id"]) == 1)
            good = next(r for r in rows if int(r["tuple_id"]) == 2)
            self.assertTrue(str(bad["status_secondary"]).startswith("error:"))
            self.assertTrue(str(bad["status_hg"]).startswith("error:"))
            self.assertTrue(str(bad["solver_status"]).startswith("error:"))
            self.assertEqual(bad["z_rq_refined"], "")
            self.assertTrue(str(good["solver_status"]).startswith("ok"))
            self.assertTrue(str(good["status_secondary"]).startswith("ok"))
            self.assertTrue(str(good["status_hg"]).startswith("ok"))


if __name__ == "__main__":
    unittest.main()
