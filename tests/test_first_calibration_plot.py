"""Tests for the standardized first-RQ b_k(q) figure."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import helpers  # noqa: F401  # Adds the repository root to sys.path.

from rqab.first import CALIBRATION_EXACT, CALIBRATION_FLUID_FALLBACK
from rqab.plotting.first_calibration import (
    plot_first_b_tripanel,
    sample_first_b_curve,
)


class TestFirstBCurve(unittest.TestCase):
    def test_sampler_uses_inclusive_grid(self) -> None:
        curve = sample_first_b_curve(k=1, q_min=-2.0, q_max=2.0, points=5)

        self.assertEqual(curve.q, (-2.0, -1.0, 0.0, 1.0, 2.0))
        self.assertEqual(len(curve.b), 5)
        self.assertTrue(all(status == CALIBRATION_EXACT for status in curve.status))
        self.assertAlmostEqual(curve.b[2], 2.0 / math.sqrt(math.pi), places=12)

    def test_k2_and_k3_mark_the_fluid_fallback(self) -> None:
        expected_boundaries = {2: 1.048592, 3: 0.503264}
        for k, expected in expected_boundaries.items():
            with self.subTest(k=k):
                curve = sample_first_b_curve(k=k)
                first_fallback = curve.status.index(CALIBRATION_FLUID_FALLBACK)

                self.assertGreater(first_fallback, 0)
                self.assertEqual(curve.status[first_fallback - 1], CALIBRATION_EXACT)
                self.assertLess(curve.q[first_fallback - 1], expected)
                self.assertGreater(curve.q[first_fallback], expected)
                self.assertTrue(
                    all(
                        status == CALIBRATION_FLUID_FALLBACK
                        for status in curve.status[first_fallback:]
                    )
                )
                self.assertTrue(all(b == 0.0 for b in curve.b[first_fallback:]))

    def test_sampler_rejects_invalid_grids(self) -> None:
        invalid_calls = (
            {"k": 0},
            {"k": 1, "q_min": 1.0, "q_max": 1.0},
            {"k": 1, "q_min": float("nan")},
            {"k": 1, "points": 1},
        )
        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    sample_first_b_curve(**kwargs)


class TestFirstBPlot(unittest.TestCase):
    def test_tripanel_adapts_sampled_curves(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "first-rq-b.pdf"

            with patch(
                "rqab.plotting.first_calibration.render_calibration_tripanel",
                return_value=out,
            ) as render:
                returned = plot_first_b_tripanel(save_path=out, no_show=True)

            self.assertEqual(returned, out)
            self.assertEqual(render.call_args.kwargs["ks"], (1, 2, 3))
            self.assertEqual(render.call_args.kwargs["save_path"], out)
            curves = render.call_args.kwargs["curves"]
            self.assertEqual(len(curves), 3)
            self.assertFalse(any(curves[0].fallback))
            self.assertTrue(any(curves[1].fallback))


if __name__ == "__main__":
    unittest.main()
