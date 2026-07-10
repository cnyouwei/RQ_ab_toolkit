#!/usr/bin/env python3
"""Tests for figure-target expansion in reproduce.py."""
from __future__ import annotations

import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import helpers  # noqa: F401  # Adds the repository root to sys.path.

import reproduce
from rqab import runner


class TestReproductionPlan(unittest.TestCase):
    def setUp(self) -> None:
        provenance_patcher = patch.object(reproduce, "check_workload_provenance")
        provenance_patcher.start()
        self.addCleanup(provenance_patcher.stop)
        self.ctx = reproduce.Context(quick=False, threads=None)

    def test_heatmap_target_includes_first_vs_refined_panels(self) -> None:
        plan = reproduce.Plan()

        self.assertTrue(reproduce.expand_target(plan, self.ctx, "mm1-gi"))

        for model in ("mm1m", "mm1e2", "mm1h2_4"):
            self.assertIn(f"tripanel:{model}", plan.steps)
            self.assertIn(f"twopanel:{model}", plan.steps)
            self.assertIn(f"first:{model}", plan.steps)

    def test_all_includes_ten_twopanel_figures(self) -> None:
        plan = reproduce.Plan()

        self.assertTrue(reproduce.expand_target(plan, self.ctx, "all"))

        twopanel_steps = [key for key in plan.steps if key.startswith("twopanel:")]
        self.assertEqual(len(twopanel_steps), 10)

    def test_idw_plot_is_recreated_without_forcing_simulation(self) -> None:
        plan = reproduce.Plan()

        with patch.object(reproduce, "idw_curves_exist", return_value=True):
            reproduce.add_idw_steps(plan, self.ctx, "effective_idw_h2m1m")

        plot_step = plan.steps["idw-plot:effective_idw_h2m1m"]
        self.assertTrue(plot_step.always)
        self.assertNotIn("idw-sim:effective_idw_h2m1m", plan.steps)


class TestWorkloadProvenance(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config_path = self.root / "workload.json"
        self.csv_path = self.root / "aggregate.csv"
        self.config_path.write_text(
            json.dumps(
                {
                    "simulation": {
                        "warmup_time": 100.0,
                        "sample_time": 1000.0,
                        "replications": 20,
                        "seed": 123456789,
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_csv(self, *, include_timing: bool = True, **overrides: object) -> None:
        row: dict[str, object] = {
            "n_reps": 20,
            "seed": 123456789,
            "warmup_time": 100.0,
            "sample_time": 1000.0,
        }
        row.update(overrides)
        if not include_timing:
            row.pop("warmup_time")
            row.pop("sample_time")
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def provenance_warning(self) -> str:
        stderr = io.StringIO()
        with patch.object(reproduce, "_grid_tuple_count", return_value=1):
            with redirect_stderr(stderr):
                reproduce.check_workload_provenance(self.csv_path, self.config_path)
        return stderr.getvalue()

    def test_matching_simulation_provenance_emits_no_warning(self) -> None:
        self.write_csv()

        self.assertEqual(self.provenance_warning(), "")

    def test_timing_mismatch_requests_regeneration(self) -> None:
        self.write_csv(warmup_time=50.0, sample_time=500.0)

        warning = self.provenance_warning()
        self.assertIn("warmup_time 50.0 != config warmup_time 100.0", warning)
        self.assertIn("sample_time 500.0 != config sample_time 1000.0", warning)
        self.assertIn("--force to regenerate", warning)

    def test_legacy_aggregate_reports_missing_timing_provenance(self) -> None:
        self.write_csv(include_timing=False)

        warning = self.provenance_warning()
        self.assertIn(
            "legacy aggregate missing simulation provenance columns warmup_time,sample_time",
            warning,
        )
        self.assertIn("--force to regenerate", warning)

    def test_workload_aggregate_retains_timing_provenance(self) -> None:
        binary = self.root / "workload_mc"
        binary.touch()
        out_csv = self.root / "aggregate_with_timing.csv"
        tuples = [
            {
                "tuple_id": 1,
                "lambda": 1.0,
                "alpha": 0.5,
                "lambda_k": 0,
                "lambda_form": "critical",
                "alpha_k": 1,
            }
        ]

        def fake_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
            summary_path = Path(cmd[cmd.index("--summary-json") + 1])
            summary_path.write_text(
                json.dumps(
                    {
                        "model_name": "test model",
                        "mean_workload": 2.5,
                        "std_workload": 0.1,
                        "n_reps": 20,
                        "warmup_time": 100.0,
                        "sample_time": 1000.0,
                        "threads_used": 2,
                        "seed": 123456789,
                        "runtime_seconds": 0.25,
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stderr="")

        with patch.object(runner.subprocess, "run", side_effect=fake_run):
            with patch.object(runner, "render_progress"):
                with redirect_stdout(io.StringIO()):
                    rc = runner.run_workload_grid(
                        tuples=tuples,
                        binary=binary,
                        model_config=self.config_path,
                        out_csv=out_csv,
                    )

        self.assertEqual(rc, 0)
        with out_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["warmup_time"]), 100.0)
        self.assertEqual(float(rows[0]["sample_time"]), 1000.0)


if __name__ == "__main__":
    unittest.main()
