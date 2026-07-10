"""Tests for the refined-RQ b_k(c) calibration tripanel."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import helpers

if importlib.util.find_spec("matplotlib") is None:
    raise unittest.SkipTest("matplotlib not installed")

from rqab.plotting.diagnostics import (  # noqa: E402
    _read_refined_b_curve,
    plot_refined_b_tripanel,
)


FIELDS = [
    "c",
    "b",
    "status",
    "psi",
    "z_model",
    "abs_error",
    "a_psi",
    "u_star",
]


def _write_table(path: Path, k: int, c_min: float = -2.0, c_max: float = 2.0) -> None:
    fallback_b = 2.0**0.5 if k == 1 else 0.0
    capped_b = 2.0**0.5 if k == 1 else 1.35
    helpers.write_csv(
        path,
        FIELDS,
        [
            {
                "c": c_max,
                "b": fallback_b,
                "status": "exact",
                "psi": 1.0,
                "z_model": "nan",
                "abs_error": 0.1,
                "a_psi": 0.01,
                "u_star": "nan",
            },
            {
                "c": 1.0,
                "b": capped_b,
                "status": "exact",
                "psi": 1.0,
                "z_model": "nan",
                "abs_error": 0.0,
                "a_psi": -0.01,
                "u_star": "nan",
            },
            {
                "c": c_min,
                "b": 1.40,
                "status": "exact",
                "psi": 0.2,
                "z_model": "nan",
                "abs_error": 0.0,
                "a_psi": -2.1,
                "u_star": 0.2,
            },
        ],
    )


class TestRefinedBCurve(unittest.TestCase):
    def test_loader_sorts_rows_and_distinguishes_cap_from_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            table = Path(temp_dir) / "b.csv"
            _write_table(table, k=1)

            curve = _read_refined_b_curve(table)

            self.assertEqual(curve.x, (-2.0, 1.0, 2.0))
            self.assertEqual(curve.b, (1.4, 2.0**0.5, 2.0**0.5))
            self.assertEqual(curve.fallback, (False, False, True))

    def test_legacy_best_fit_row_is_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            table = Path(temp_dir) / "legacy.csv"
            _write_table(table, k=2)
            rows, _fields = helpers.read_csv_rows(table)
            rows[1]["status"] = "best_fit"
            helpers.write_csv(table, FIELDS, rows)

            curve = _read_refined_b_curve(table)

            self.assertEqual(curve.fallback, (False, True, True))

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


class TestRefinedBPlot(unittest.TestCase):
    def test_tripanel_writes_pdf(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tables = []
            for k in (1, 2, 3):
                table = root / f"b{k}.csv"
                _write_table(table, k=k)
                tables.append(table)
            out = root / "refined-rq-b.pdf"

            returned = plot_refined_b_tripanel(
                table_paths=tables,
                save_path=out,
                no_show=True,
            )

            self.assertEqual(returned, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1_000)
            self.assertEqual(out.read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
