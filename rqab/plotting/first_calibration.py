"""Plots for the standardized first-RQ calibration ``b_k(q)``.

Unlike the refined-RQ calibration diagnostics, these curves are evaluated
directly from :mod:`rqab.first`; they do not use the generated b tables.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from ..first import (
    CALIBRATION_FLUID_FALLBACK,
    calibrate_b_first_rq_standardized,
)
from ..util import RESULTS_DIR
from .calibration_tripanel import CalibrationPlotCurve, render_calibration_tripanel


@dataclass(frozen=True)
class FirstBCurve:
    """Sampled values of one standardized first-RQ calibration curve."""

    q: tuple[float, ...]
    b: tuple[float, ...]
    status: tuple[str, ...]


def sample_first_b_curve(
    k: int,
    q_min: float = -8.0,
    q_max: float = 4.0,
    points: int = 601,
) -> FirstBCurve:
    """Sample ``b_k(q)`` on an inclusive, evenly spaced q-grid."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (math.isfinite(q_min) and math.isfinite(q_max)):
        raise ValueError("q_min and q_max must be finite")
    if q_max <= q_min:
        raise ValueError("q_max must be greater than q_min")
    if points < 2:
        raise ValueError("points must be >= 2")

    step = (float(q_max) - float(q_min)) / float(points - 1)
    q_list = [float(q_min) + i * step for i in range(points)]
    q_list[0] = float(q_min)
    q_list[-1] = float(q_max)
    q_values = tuple(q_list)
    calibrations = tuple(
        calibrate_b_first_rq_standardized(tilde_c=q, k=k) for q in q_values
    )
    return FirstBCurve(
        q=q_values,
        b=tuple(calibration.b for calibration in calibrations),
        status=tuple(calibration.status for calibration in calibrations),
    )


def plot_first_b_tripanel(
    ks: Sequence[int] = (1, 2, 3),
    q_min: float = -8.0,
    q_max: float = 4.0,
    points: int = 601,
    save_path: Path | None = None,
    no_show: bool = True,
) -> Path:
    """Plot one standardized first-RQ calibration curve per k.

    Exact-match portions are solid.  Regions where exact matching is
    infeasible use the first-RQ fluid fallback ``b=0`` and are shown with a
    dashed red line and a light red background.
    """
    if not ks:
        raise ValueError("ks must be nonempty")

    curves = [
        sample_first_b_curve(k=k, q_min=q_min, q_max=q_max, points=points)
        for k in ks
    ]

    if save_path is not None:
        out = Path(save_path)
    elif tuple(ks) == (1, 2, 3):
        out = RESULTS_DIR / "first_rq_b_tripanel.pdf"
    else:
        out = RESULTS_DIR / (
            "first_rq_b_tripanel_k" + "_".join(str(k) for k in ks) + ".pdf"
        )
    plot_curves = [
        CalibrationPlotCurve(
            x=curve.q,
            b=curve.b,
            fallback=tuple(
                status == CALIBRATION_FLUID_FALLBACK for status in curve.status
            ),
        )
        for curve in curves
    ]
    return render_calibration_tripanel(
        ks=ks,
        curves=plot_curves,
        x_min=q_min,
        x_max=q_max,
        x_label=r"standardized load $q$",
        y_label=r"calibrated $b_k(q)$",
        calibrated_label="exact match",
        fallback_label=r"fluid fallback ($b=0$)",
        save_path=out,
        no_show=no_show,
    )
