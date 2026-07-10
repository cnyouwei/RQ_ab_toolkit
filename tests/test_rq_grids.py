#!/usr/bin/env python3
"""Tests for the refined/first RQ grid pipeline (single-station and tandem)."""
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
    tandem_h2e2_to_m1h2_payload,
    temp_dir,
    tiny_grid_payload,
    valid_lognormal_model_payload,
    valid_model_payload,
    write_csv,
    write_json,
)

from rqab import first as first_mod
from rqab import refined as refined_mod
from rqab.effective_idw import tau_tilde_c
from rqab.fixed_point import BisectOptions, survival_alpha
from rqab.grids import build_s_grid
from rqab.idc import departure_idc_curve
from rqab.models import (
    DistributionComponent,
    build_base_stats,
    infer_k_from_patience,
    load_model_config,
    model_plot_metadata,
    parse_distribution_component,
)
from rqab.plotting.ratio_panels import (
    load_combined_secondary_rows,
    load_workload_rows,
    load_z_rows,
)
from rqab.tables import SQRT2, BCalibrationInterpolator, WTableInterpolator


class TestModelParsing(unittest.TestCase):
    def test_lognormal_service_parse_and_component_scope(self) -> None:
        model_cfg = valid_lognormal_model_payload()["model"]
        service = parse_distribution_component(
            model_cfg, "service", model_cfg["name"], allow_lognormal=True
        )
        self.assertEqual(service.family, "lognormal")
        self.assertAlmostEqual(float(service.params["mean"]), 1.0, places=12)
        self.assertAlmostEqual(float(service.params["scv"]), 4.0, places=12)

        model_cfg["arrival"]["distribution"] = {
            "family": "ln",
            "params": {"mean": 1.0, "scv": 4.0},
        }
        with self.assertRaisesRegex(ValueError, "service only"):
            parse_distribution_component(model_cfg, "arrival", model_cfg["name"])

        model_cfg["arrival"]["distribution"] = {
            "family": "exponential",
            "params": {"rate": 1.0},
        }
        model_cfg["patience"]["distribution"] = {
            "family": "lognormal",
            "params": {"mean": 1.0, "scv": 4.0},
        }
        with self.assertRaisesRegex(ValueError, "service only"):
            parse_distribution_component(model_cfg, "patience", model_cfg["name"])

    def test_k_inference(self) -> None:
        exp_comp = DistributionComponent("exponential", {"rate": 1.0})
        self.assertEqual(infer_k_from_patience(exp_comp), 1)

        erlang_comp = DistributionComponent("erlang_k", {"k": 2, "rate": 2.0})
        self.assertEqual(infer_k_from_patience(erlang_comp), 2)

        h2_comp = DistributionComponent(
            "hyperexponential2",
            {"p": 0.5, "rate1": 3.0, "rate2": 1.0 / 3.0},
        )
        self.assertEqual(infer_k_from_patience(h2_comp), 1)

    def test_tandem_model_parse_and_departure_idc_smoke(self) -> None:
        model_path = CONFIGS_DIR / "workload_tandem_h2_4e2_to_m1h2_4.json"
        parsed = load_model_config(model_path)
        self.assertTrue(parsed.is_tandem)
        self.assertAlmostEqual(parsed.queue1.traffic_intensity, 0.9, places=12)
        self.assertEqual(parsed.queue1.arrival.family, "hyperexponential2")
        self.assertEqual(parsed.queue1.service.family, "erlang_k")
        self.assertEqual(parsed.queue2.service.family, "exponential")
        self.assertEqual(parsed.queue2.patience.family, "hyperexponential2")

        s_grid = build_s_grid(1e-3, 1e3, 30)
        dep = departure_idc_curve(s_values=s_grid, lam=1.0, model=parsed)
        self.assertEqual(len(dep), len(s_grid))
        self.assertAlmostEqual(dep[0], 1.0, places=12)
        self.assertTrue(all(v == v and v > 0.0 for v in dep))


class TestBCalibrationInterpolator(unittest.TestCase):
    def test_interpolation_policy(self) -> None:
        interp = BCalibrationInterpolator(
            c_values=[-1.0, 0.0, 2.0],
            b_values=[1.2, 1.0, 0.6],
        )
        # Below the c-grid: the underloaded sqrt(2) limit.
        self.assertAlmostEqual(interp.evaluate(-2.0), SQRT2, places=12)
        # Above the c-grid: hold the last table value.
        self.assertAlmostEqual(interp.evaluate(10.0), 0.6, places=12)
        # Interior: linear interpolation.
        self.assertAlmostEqual(interp.evaluate(1.0), 0.8, places=12)

    def test_endpoint_snapping(self) -> None:
        interp = BCalibrationInterpolator(
            c_values=[-1.0, 0.0, 2.0],
            b_values=[1.2, 1.0, 0.6],
        )
        # Exact knots.
        self.assertAlmostEqual(interp.evaluate(-1.0), 1.2, places=15)
        self.assertAlmostEqual(interp.evaluate(0.0), 1.0, places=15)
        self.assertAlmostEqual(interp.evaluate(2.0), 0.6, places=15)
        # Snap within 1e-14 of a knot.
        self.assertAlmostEqual(interp.evaluate(0.0 + 5e-15), 1.0, places=15)
        self.assertAlmostEqual(interp.evaluate(2.0 - 5e-15), 0.6, places=15)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            BCalibrationInterpolator(c_values=[], b_values=[])
        with self.assertRaises(ValueError):
            BCalibrationInterpolator(c_values=[0.0, 0.0], b_values=[1.0, 1.0])


class TestSGrid(unittest.TestCase):
    def test_build_s_grid_shape(self) -> None:
        grid = build_s_grid(1e-3, 1e5, 200)
        self.assertEqual(len(grid), 201)
        self.assertEqual(grid[0], 0.0)
        self.assertAlmostEqual(grid[1], 1e-3, places=15)
        self.assertAlmostEqual(grid[-1], 1e5, places=6)
        # Log-uniform: constant ratio between consecutive positive points.
        ratios = [grid[i + 1] / grid[i] for i in range(1, len(grid) - 1)]
        for r in ratios:
            self.assertAlmostEqual(r, ratios[0], places=10)

    def test_build_s_grid_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_s_grid(1e-3, 1e5, 1)
        with self.assertRaises(ValueError):
            build_s_grid(0.0, 1e5, 10)
        with self.assertRaises(ValueError):
            build_s_grid(1.0, 1.0, 10)


class TestSurvivalAlpha(unittest.TestCase):
    def test_exponential(self) -> None:
        comp = DistributionComponent("exponential", {"rate": 1.5})
        self.assertAlmostEqual(
            survival_alpha(2.0, 0.5, comp), math.exp(-1.5 * 1.0), places=14
        )

    def test_erlang_k(self) -> None:
        comp = DistributionComponent("erlang_k", {"k": 2, "rate": 2.0})
        x = 0.5 * 2.0  # alpha * z
        expected = math.exp(-2.0 * x) * (1.0 + 2.0 * x)
        self.assertAlmostEqual(survival_alpha(2.0, 0.5, comp), expected, places=14)

    def test_hyperexponential2(self) -> None:
        comp = DistributionComponent(
            "hyperexponential2", {"p": 0.3, "rate1": 2.0, "rate2": 0.5}
        )
        expected = 0.3 * math.exp(-2.0) + 0.7 * math.exp(-0.5)
        self.assertAlmostEqual(survival_alpha(2.0, 0.5, comp), expected, places=14)

    def test_zero_boundary(self) -> None:
        for comp in (
            DistributionComponent("exponential", {"rate": 1.0}),
            DistributionComponent("erlang_k", {"k": 2, "rate": 2.0}),
            DistributionComponent(
                "hyperexponential2", {"p": 0.3, "rate1": 2.0, "rate2": 0.5}
            ),
        ):
            self.assertEqual(survival_alpha(0.0, 1.0, comp), 1.0)
            self.assertEqual(survival_alpha(-1.0, 1.0, comp), 1.0)


def make_refined_solver(n_s: int = 160) -> refined_mod.RefinedSolver:
    parsed = load_model_config(CONFIGS_DIR / "workload_mm1m.json")
    k = infer_k_from_patience(parsed.patience)
    base = build_base_stats(parsed, k=k)
    return refined_mod.RefinedSolver(
        model=parsed,
        base=base,
        b_table=BCalibrationInterpolator.from_csv(B_TABLE_K1),
        w_interp=WTableInterpolator.from_matrix_csv(W_TABLE_K1),
        s_grid=build_s_grid(1e-3, 1e5, n_s),
        bisect=BisectOptions(
            abs_tol=1e-7, rel_tol=1e-7, max_iters=120, bracket_max_doublings=60
        ),
    )


class TestRefinedSolverInternals(unittest.TestCase):
    def test_fixed_point_smoke(self) -> None:
        solver = make_refined_solver(n_s=200)
        kernel = solver.build_kernel(lam=1.25, alpha=0.5)
        self.assertTrue(kernel.c == kernel.c)  # finite check
        self.assertGreaterEqual(kernel.b, 0.0)

        sol = solver.solve_fixed_point(lam=1.25, alpha=0.5, kernel=kernel)
        self.assertGreaterEqual(sol.z, 0.0)
        self.assertTrue(sol.rhs_at_solution == sol.rhs_at_solution)
        self.assertTrue(sol.status.startswith("ok"))

    def test_monotonic_sanity(self) -> None:
        solver = make_refined_solver(n_s=160)

        def z_of(lam: float, alpha: float) -> float:
            row = {
                "tuple_id": 1,
                "lambda": lam,
                "alpha": alpha,
                "lambda_k": 0,
                "lambda_form": "custom",
                "alpha_k": 0,
            }
            solved = solver.solve_one_tuple(row)
            return float(solved["z_rq_refined"])

        z_low_lam = z_of(0.8, 1.0)
        z_high_lam = z_of(1.2, 1.0)
        self.assertGreaterEqual(z_high_lam, z_low_lam)

        z_low_alpha = z_of(1.1, 0.5)
        z_high_alpha = z_of(1.1, 2.0)
        self.assertGreaterEqual(z_low_alpha, z_high_alpha)

    def test_canonical_b_table_c_is_identity_for_reference_models(self) -> None:
        for k in (1, 2, 3):
            beta_ref = float(k**k) / float(math.factorial(k))
            for c in (-5.0, -0.125, 0.0, 0.7, 12.0):
                _tau, tilde_c = tau_tilde_c(
                    c=c, k=k, mu=1.0, c_a2=1.0, c_s2=1.0, beta_patience=beta_ref
                )
                self.assertAlmostEqual(
                    refined_mod.canonical_b_table_c(tilde_c, k), c, places=12
                )

    def test_canonical_b_table_c_matches_primitive_form(self) -> None:
        k, mu, c_a2, c_s2, beta_model = 2, 1.3, 1.7, 0.4, 2.9
        beta_ref = float(k**k) / float(math.factorial(k))
        ratio = (c_a2 + c_s2) / (2.0 * mu)
        for c in (-3.0, -0.2, 1.1, 8.0):
            _tau, tilde_c = tau_tilde_c(
                c=c, k=k, mu=mu, c_a2=c_a2, c_s2=c_s2, beta_patience=beta_model
            )
            expected = (
                c * ratio ** (-k / (k + 1.0)) * (beta_ref / beta_model) ** (1.0 / (k + 1.0))
            )
            self.assertAlmostEqual(
                refined_mod.canonical_b_table_c(tilde_c, k), expected, places=12
            )

    def test_build_kernel_queries_b_table_at_canonical_coordinate(self) -> None:
        class RecordingBTable:
            def __init__(self) -> None:
                self.queries: list[float] = []

            def evaluate(self, c: float) -> float:
                self.queries.append(c)
                return 1.0

        model = load_model_config(CONFIGS_DIR / "workload_h2_4ln1_21e2.json")
        base = build_base_stats(model, k=infer_k_from_patience(model.patience))
        b_table = RecordingBTable()
        w_table = WTableInterpolator(
            c_grid=[-100.0, 100.0],
            t_grid=[0.0, 1.0e8],
            w_matrix=[[1.0, 1.0], [1.0, 1.0]],
        )
        solver = refined_mod.RefinedSolver(
            model=model,
            base=base,
            b_table=b_table,  # type: ignore[arg-type]
            w_interp=w_table,
            s_grid=[0.0, 1.0],
        )

        kernel = solver.build_kernel(lam=1.2, alpha=0.5)

        self.assertEqual(len(b_table.queries), 1)
        self.assertAlmostEqual(
            b_table.queries[0],
            refined_mod.canonical_b_table_c(kernel.tilde_c, base.k),
            places=12,
        )
        self.assertNotAlmostEqual(b_table.queries[0], kernel.c, places=6)

    def test_canonical_b_table_c_rejects_bad_k(self) -> None:
        with self.assertRaises(ValueError):
            refined_mod.canonical_b_table_c(1.0, 0)


class TestPlotLoaders(unittest.TestCase):
    def test_plot_metadata_accepts_tandem_patience(self) -> None:
        alias, patience_mean, _title = model_plot_metadata(
            CONFIGS_DIR / "workload_tandem_e2h2_4_to_m1e2.json"
        )
        self.assertEqual(alias, "tandem_e2h2_4_to_m1e2")
        self.assertAlmostEqual(patience_mean, 1.0, places=12)

    def test_plot_metadata_accepts_tandem_patience_h2(self) -> None:
        alias, patience_mean, title = model_plot_metadata(
            CONFIGS_DIR / "workload_tandem_h2_4e2_to_m1h2_4.json"
        )
        self.assertEqual(alias, "tandem_h2_4e2_to_m1h2_4")
        self.assertAlmostEqual(patience_mean, 1.0, places=12)
        self.assertTrue(title)

    def _combined_csv_rows(self, method_label: str) -> list[dict[str, object]]:
        return [
            {
                "tuple_id": 1,
                "z_rq_refined": 0.18,
                "solver_status": "ok",
                "secondary_method": method_label,
                "z_secondary": 0.16,
                "z_hg": 0.19,
                "status_secondary": "ok",
                "status_hg": "ok",
            },
            {
                "tuple_id": 2,
                "z_rq_refined": 0.88,
                "solver_status": "ok",
                "secondary_method": method_label,
                "z_secondary": 0.85,
                "z_hg": 0.90,
                "status_secondary": "ok",
                "status_hg": "ok",
            },
            {
                "tuple_id": 3,
                "z_rq_refined": 2.2,
                "solver_status": "ok",
                "secondary_method": method_label,
                "z_secondary": 2.6,
                "z_hg": 2.3,
                "status_secondary": "error:tail_not_converged",
                "status_hg": "ok",
            },
        ]

    def _combined_fieldnames(self) -> list[str]:
        return [
            "tuple_id",
            "z_rq_refined",
            "solver_status",
            "secondary_method",
            "z_secondary",
            "z_hg",
            "status_secondary",
            "status_hg",
        ]

    def test_combined_loader_accepts_wg_and_gw_labels(self) -> None:
        # Both accepted Ward-Glynn labels normalize to the same display name.
        for label in ("wg", "gw"):
            with temp_dir() as tmp:
                combined_csv = tmp / "combined.csv"
                write_csv(
                    combined_csv,
                    self._combined_fieldnames(),
                    self._combined_csv_rows(label),
                )
                rows, method_title = load_combined_secondary_rows(combined_csv)
                self.assertEqual(method_title, "WG")
                self.assertEqual(len(rows), 3)
                # Rows are keyed by tuple_id.
                self.assertEqual(rows[2]["secondary_method"], label)
                self.assertAlmostEqual(rows[2]["z_secondary"], 0.85, places=12)
                self.assertAlmostEqual(rows[2]["z_hg"], 0.90, places=12)
                # Non-ok status rows become NaN, not dropped.
                self.assertTrue(math.isnan(rows[3]["z_secondary"]))
                self.assertAlmostEqual(rows[3]["z_hg"], 2.3, places=12)

    def test_z_and_workload_loaders(self) -> None:
        with temp_dir() as tmp:
            combined_csv = tmp / "combined.csv"
            write_csv(
                combined_csv,
                self._combined_fieldnames(),
                self._combined_csv_rows("wg"),
            )
            z_by_id = load_z_rows(combined_csv, z_col="z_rq_refined", label="refined")
            self.assertEqual(set(z_by_id.keys()), {1, 2, 3})
            self.assertAlmostEqual(z_by_id[2], 0.88, places=12)

            workload_csv = tmp / "workload.csv"
            write_csv(
                workload_csv,
                ["tuple_id", "lambda", "alpha", "mean_workload", "model_name"],
                [
                    {
                        "tuple_id": 1,
                        "lambda": 0.8,
                        "alpha": 2.0,
                        "mean_workload": 0.2,
                        "model_name": "M/M/1+M",
                    },
                ],
            )
            workload_rows = load_workload_rows(workload_csv)
            self.assertEqual(set(workload_rows.keys()), {1})
            self.assertAlmostEqual(workload_rows[1]["mean_workload"], 0.2, places=12)


S_GRID_CLI_ARGS = ["--n-s", "80", "--s-min", "1e-3", "--s-max", "1e4"]


def run_grid_cli(method, model_path, grid_path, out_csv, extra=()):
    cmd = [
        sys.executable,
        str(RUN_GRID_SCRIPT),
        "--method",
        method,
        "--model-config",
        str(model_path),
        "--grid",
        str(grid_path),
        "--out-csv",
        str(out_csv),
        "--no-auto-generate",
        *S_GRID_CLI_ARGS,
        *extra,
    ]
    return cmd, helpers.run_cli(cmd)


class TestGridCLI(unittest.TestCase):
    """End-to-end runs of scripts/run_grid.py on the tiny 3-tuple grid.

    The pinned z values were produced by this implementation with the s-grid
    settings below and the checked-in k=1 tables; they act as regression
    anchors for the refined (eq:RQ_ab_2) and first (eq:RQ_ab_1) solvers.
    """

    REFINED_SINGLE_Z = [0.3755364939570427, 0.6497223302721977, 1.1397427096962929]
    REFINED_TANDEM_Z = [0.283348448574543, 0.4733736142516136, 0.8205712214112282]
    FIRST_SINGLE_Z = [0.3879299536347389, 0.6687495037913322, 1.1627303883433342]
    FIRST_TANDEM_Z = [0.29919707030057907, 0.4956096336245537, 0.8317910209298134]

    def _rows_sorted(self, out_csv):
        rows, fieldnames = read_csv_rows(out_csv)
        rows.sort(key=lambda r: int(r["tuple_id"]))
        return rows, fieldnames

    def test_refined_single_station_end_to_end(self) -> None:
        with temp_dir() as tmp:
            model_path = tmp / "workload_mm1m.json"
            grid_path = tmp / "grid.json"
            out_csv = tmp / "out.csv"
            write_json(model_path, valid_model_payload())
            write_json(grid_path, tiny_grid_payload())

            cmd, proc = run_grid_cli(
                "refined",
                model_path,
                grid_path,
                out_csv,
                extra=("--w-table", str(W_TABLE_K1), "--b-table", str(B_TABLE_K1)),
            )
            if proc.returncode != 0:
                self.fail(helpers.fail_message("tiny-grid refined RQ run failed.", cmd, proc))
            self.assertTrue(out_csv.exists())

            rows, fieldnames = self._rows_sorted(out_csv)
            self.assertEqual(len(rows), 3)
            for col in refined_mod.CSV_COLUMNS:
                self.assertIn(col, fieldnames)

            z_values = []
            for row in rows:
                self.assertTrue(str(row["solver_status"]).startswith("ok"))
                self.assertTrue(str(row["status_secondary"]).startswith("ok"))
                self.assertTrue(str(row["status_hg"]).startswith("ok"))
                self.assertEqual(int(row["k"]), 1)
                z_values.append(float(row["z_rq_refined"]))

            # Increasing lambda and decreasing alpha both push z up.
            self.assertLess(z_values[0], z_values[1])
            self.assertLess(z_values[1], z_values[2])
            for got, expected in zip(z_values, self.REFINED_SINGLE_Z):
                self.assertAlmostEqual(got, expected, delta=1e-6)

    def test_refined_tandem_end_to_end(self) -> None:
        with temp_dir() as tmp:
            model_path = tmp / "workload_tandem_h2_4e2_to_m1h2_4.json"
            grid_path = tmp / "grid.json"
            out_csv = tmp / "out.csv"
            write_json(model_path, tandem_h2e2_to_m1h2_payload())
            write_json(grid_path, tiny_grid_payload())

            cmd, proc = run_grid_cli(
                "refined",
                model_path,
                grid_path,
                out_csv,
                extra=("--w-table", str(W_TABLE_K1), "--b-table", str(B_TABLE_K1)),
            )
            if proc.returncode != 0:
                self.fail(helpers.fail_message("tiny-grid tandem refined RQ run failed.", cmd, proc))
            self.assertTrue(out_csv.exists())

            rows, fieldnames = self._rows_sorted(out_csv)
            self.assertEqual(len(rows), 3)
            for col in refined_mod.CSV_COLUMNS:
                self.assertIn(col, fieldnames)

            z_values = []
            for row in rows:
                self.assertTrue(str(row["solver_status"]).startswith("ok"))
                self.assertTrue(str(row["status_secondary"]).startswith("ok"))
                self.assertTrue(str(row["status_hg"]).startswith("ok"))
                # H2 patience has F'(0) > 0, so WG is the secondary method.
                self.assertEqual(str(row["secondary_method"]), "wg")
                self.assertNotEqual(str(row["z_hg"]), "")
                self.assertNotEqual(str(row["c_x2"]), "")
                # Tandem c_x2 includes the departure-IDC-derived arrival SCV.
                self.assertGreater(float(row["c_x2"]), 1.0)
                z_values.append(float(row["z_rq_refined"]))

            self.assertLess(z_values[0], z_values[1])
            self.assertLess(z_values[1], z_values[2])
            for got, expected in zip(z_values, self.REFINED_TANDEM_Z):
                self.assertAlmostEqual(got, expected, delta=1e-6)

    def test_first_single_station_end_to_end(self) -> None:
        with temp_dir() as tmp:
            model_path = tmp / "workload_mm1m.json"
            grid_path = tmp / "grid.json"
            out_csv = tmp / "out.csv"
            write_json(model_path, valid_model_payload())
            write_json(grid_path, tiny_grid_payload())

            cmd, proc = run_grid_cli("first", model_path, grid_path, out_csv)
            if proc.returncode != 0:
                self.fail(helpers.fail_message("tiny-grid first RQ run failed.", cmd, proc))
            self.assertTrue(out_csv.exists())

            rows, fieldnames = self._rows_sorted(out_csv)
            self.assertEqual(len(rows), 3)
            for col in first_mod.CSV_COLUMNS:
                self.assertIn(col, fieldnames)

            z_values = []
            for row in rows:
                self.assertTrue(str(row["solver_status"]).startswith("ok"))
                self.assertGreater(float(row["b"]), 0.0)
                self.assertGreater(float(row["psi"]), 0.0)
                z_values.append(float(row["z_rq_first"]))

            self.assertLess(z_values[0], z_values[1])
            self.assertLess(z_values[1], z_values[2])
            for got, expected in zip(z_values, self.FIRST_SINGLE_Z):
                self.assertAlmostEqual(got, expected, delta=1e-6)

    def test_first_tandem_end_to_end(self) -> None:
        with temp_dir() as tmp:
            model_path = tmp / "workload_tandem_h2_4e2_to_m1h2_4.json"
            grid_path = tmp / "grid.json"
            out_csv = tmp / "out.csv"
            write_json(model_path, tandem_h2e2_to_m1h2_payload())
            write_json(grid_path, tiny_grid_payload())

            cmd, proc = run_grid_cli("first", model_path, grid_path, out_csv)
            if proc.returncode != 0:
                self.fail(helpers.fail_message("tiny-grid tandem first RQ run failed.", cmd, proc))
            self.assertTrue(out_csv.exists())

            rows, fieldnames = self._rows_sorted(out_csv)
            self.assertEqual(len(rows), 3)
            for col in first_mod.CSV_COLUMNS:
                self.assertIn(col, fieldnames)

            z_values = []
            for row in rows:
                self.assertTrue(str(row["solver_status"]).startswith("ok"))
                z_values.append(float(row["z_rq_first"]))

            self.assertLess(z_values[0], z_values[1])
            self.assertLess(z_values[1], z_values[2])
            for got, expected in zip(z_values, self.FIRST_TANDEM_Z):
                self.assertAlmostEqual(got, expected, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
