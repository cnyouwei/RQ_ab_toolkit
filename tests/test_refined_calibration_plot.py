"""Tests for the refined-RQ b_k(c) calibration tripanel."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import helpers

from rqab.plotting.calibration_tripanel import (  # noqa: E402
    CalibrationPlotCurve,
    render_calibration_tripanel,
)
from rqab.plotting.diagnostics import (  # noqa: E402
    _read_b_curve,
    plot_refined_b_tripanel,
)


FIELDS = ["c", "b", "a_psi"]


def _write_table(
    path: Path,
    k: int,
    c_min: float = -2.0,
    c_max: float = 2.0,
    *,
    include_status: bool = False,
) -> None:
    fallback_b = 2.0**0.5 if k == 1 else 0.0
    capped_b = 2.0**0.5 if k == 1 else 1.35
    fieldnames = FIELDS + (["status"] if include_status else [])
    helpers.write_csv(
        path,
        fieldnames,
        [
            {
                "c": c_max,
                "b": fallback_b,
                "a_psi": 0.01,
                **({"status": "exact"} if include_status else {}),
            },
            {
                "c": 1.0,
                "b": capped_b,
                "a_psi": -0.01,
                **({"status": "ignored"} if include_status else {}),
            },
            {
                "c": c_min,
                "b": 1.40,
                "a_psi": -2.1,
                **({"status": "unused"} if include_status else {}),
            },
        ],
    )


class TestRefinedBCurve(unittest.TestCase):
    def test_loader_sorts_rows_and_distinguishes_cap_from_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            table = Path(temp_dir) / "b.csv"
            _write_table(table, k=1)

            curve = _read_b_curve(table)

            self.assertEqual(curve.x, (-2.0, 1.0, 2.0))
            self.assertEqual(curve.b, (1.4, 2.0**0.5, 2.0**0.5))
            self.assertEqual(curve.fallback, (False, False, True))

    def test_optional_status_column_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            table = Path(temp_dir) / "with-status.csv"
            _write_table(table, k=2, include_status=True)

            curve = _read_b_curve(table)

            self.assertEqual(curve.fallback, (False, False, True))

    def test_tripanel_validates_k_and_table_counts(self) -> None:
        with self.assertRaises(ValueError):
            plot_refined_b_tripanel(ks=())
        with self.assertRaises(ValueError):
            plot_refined_b_tripanel(ks=(1, 2), table_paths=[Path("one.csv")])
        with self.assertRaises(ValueError):
            plot_refined_b_tripanel(ks=(0,), table_paths=[Path("unused.csv")])

    def test_default_display_range_is_minus8_to_4(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tables = []
            for k in (1, 2, 3):
                table = root / f"b{k}.csv"
                _write_table(table, k=k, c_min=-20.0, c_max=20.0)
                tables.append(table)
            out = root / "refined-rq-b.pdf"

            with patch(
                "rqab.plotting.diagnostics.render_calibration_tripanel",
                return_value=out,
            ) as render:
                returned = plot_refined_b_tripanel(
                    table_paths=tables,
                    save_path=out,
                )

            self.assertEqual(returned, out)
            self.assertEqual(render.call_args.kwargs["x_min"], -8.0)
            self.assertEqual(render.call_args.kwargs["x_max"], 4.0)


class TestCalibrationRenderer(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "matplotlib not installed",
    )
    def test_shared_renderer_writes_pdf(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out = root / "calibration.pdf"
            curve = CalibrationPlotCurve(
                x=(-2.0, 0.0, 2.0),
                b=(1.4, 1.0, 0.0),
                fallback=(False, False, True),
            )

            returned = render_calibration_tripanel(
                ks=(1, 2, 3),
                curves=(curve, curve, curve),
                x_min=-2.0,
                x_max=2.0,
                x_label="load",
                y_label="b",
                calibrated_label="calibrated",
                fallback_label="fallback",
                save_path=out,
                no_show=True,
            )

            self.assertEqual(returned, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1_000)
            self.assertEqual(out.read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
