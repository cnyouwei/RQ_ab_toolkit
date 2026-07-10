#!/usr/bin/env python3
"""Tests for figure-target expansion in reproduce.py."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import helpers  # noqa: F401  # Adds the repository root to sys.path.

import reproduce


class TestReproductionPlan(unittest.TestCase):
    def setUp(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
