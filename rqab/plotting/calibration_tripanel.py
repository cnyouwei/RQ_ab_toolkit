"""Shared renderer for first- and refined-RQ calibration tripanels."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CalibrationPlotCurve:
    """One calibrated curve and the samples belonging to its fallback branch."""

    x: tuple[float, ...]
    b: tuple[float, ...]
    fallback: tuple[bool, ...]

    def validate(self) -> None:
        if not self.x:
            raise ValueError("calibration curve is empty")
        if not (len(self.x) == len(self.b) == len(self.fallback)):
            raise ValueError("calibration curve columns must have equal lengths")
        if any(self.x[i + 1] <= self.x[i] for i in range(len(self.x) - 1)):
            raise ValueError("calibration curve x values must be strictly increasing")


def _fallback_spans(curve: CalibrationPlotCurve) -> list[tuple[float, float]]:
    """Return plotting spans for contiguous fallback samples."""
    spans: list[tuple[float, float]] = []
    start: int | None = None
    for i, fallback in enumerate(curve.fallback):
        if fallback and start is None:
            start = i
        if not fallback and start is not None:
            left = (
                curve.x[start]
                if start == 0
                else 0.5 * (curve.x[start - 1] + curve.x[start])
            )
            right = 0.5 * (curve.x[i - 1] + curve.x[i])
            spans.append((left, right))
            start = None
    if start is not None:
        left = (
            curve.x[start]
            if start == 0
            else 0.5 * (curve.x[start - 1] + curve.x[start])
        )
        spans.append((left, curve.x[-1]))
    return spans


def _calibrated_plot_values(curve: CalibrationPlotCurve) -> list[float]:
    """Mask fallback samples and connect calibrated runs to their fallback."""
    values = [
        float("nan") if fallback else b
        for b, fallback in zip(curve.b, curve.fallback)
    ]
    for i in range(1, len(values)):
        if curve.fallback[i] and not curve.fallback[i - 1]:
            values[i] = curve.b[i]
    return values


def render_calibration_tripanel(
    *,
    ks: Sequence[int],
    curves: Sequence[CalibrationPlotCurve],
    x_min: float,
    x_max: float,
    x_label: str,
    y_label: str,
    calibrated_label: str,
    fallback_label: str,
    save_path: Path,
    no_show: bool,
    dpi: int | None = None,
) -> Path:
    """Render calibration curves with the shared three-panel paper style."""
    if not ks:
        raise ValueError("ks must be nonempty")
    if len(curves) != len(ks):
        raise ValueError("curves must match ks in length")
    if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min):
        raise ValueError("expected finite x_min < x_max")
    for curve in curves:
        curve.validate()

    from .style import setup_rcparams

    setup_rcparams()

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_panels = len(ks)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.0 * n_panels, 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_list = list(axes.flat) if n_panels > 1 else [axes]

    for i, (ax, k, curve) in enumerate(zip(axes_list, ks, curves)):
        for left, right in _fallback_spans(curve):
            ax.axvspan(left, right, color="tab:red", alpha=0.07, linewidth=0)

        ax.axhline(
            math.sqrt(2.0),
            color="0.45",
            linestyle=(0, (1, 3)),
            linewidth=1.2,
        )
        ax.plot(
            curve.x,
            _calibrated_plot_values(curve),
            color=f"C{i % 10}",
            linewidth=2.0,
            solid_capstyle="round",
        )
        fallback_values = [
            b if fallback else float("nan")
            for b, fallback in zip(curve.b, curve.fallback)
        ]
        ax.plot(
            curve.x,
            fallback_values,
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.045, 1.50)
        ax.set_title(rf"$k = {k}$", fontsize=13)
        ax.set_xlabel(x_label, fontsize=11)
        ax.grid(True, alpha=0.28, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    axes_list[0].set_ylabel(y_label, fontsize=11)
    legend_handles = [
        Line2D([0], [0], color="0.15", linewidth=2.0, label=calibrated_label),
        Line2D(
            [0],
            [0],
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
            label=fallback_label,
        ),
        Patch(
            facecolor="tab:red",
            alpha=0.07,
            edgecolor="none",
            label="fallback region",
        ),
        Line2D(
            [0],
            [0],
            color="0.45",
            linestyle=(0, (1, 3)),
            linewidth=1.2,
            label=r"$\sqrt{2}$",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        fontsize=10,
        frameon=False,
    )

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    print(f"Saved plot: {out}")
    if not no_show:
        plt.show()
    else:
        plt.close(fig)
    return out
