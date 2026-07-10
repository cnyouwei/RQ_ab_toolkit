"""Diagnostic figures for the w and calibrated-b tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

from ..tables import (
    WTableInterpolator,
    default_b_table_path,
    default_w_table_path,
    make_log_grid,
)
from ..util import RESULTS_DIR
from .calibration_tripanel import CalibrationPlotCurve, render_calibration_tripanel

_FEASIBILITY_THRESHOLD = -1e-12


def plot_w_overlay(
    k: int,
    table_path: Path | None = None,
    overlay_c_min: int | None = None,
    overlay_c_max: int | None = None,
    plot_points: int = 500,
    save_path: Path | None = None,
    dpi: int = 180,
    no_show: bool = True,
) -> Path:
    """Overlay w_{c,k}(t) for integer c values interpolated from the matrix table.

    Output default: results/w_overlay_k{k}.png
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    table = Path(table_path) if table_path is not None else default_w_table_path(k)
    interp = WTableInterpolator.from_matrix_csv(table)

    c_lo = overlay_c_min if overlay_c_min is not None else math.ceil(interp.c_min)
    c_hi = overlay_c_max if overlay_c_max is not None else math.floor(interp.c_max)
    if c_hi < c_lo:
        raise ValueError("overlay integer c range is empty")
    c_values = list(range(c_lo, c_hi + 1))

    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    t_plot = make_log_grid(interp.t_min_pos, interp.t_max, plot_points)
    c_min = min(c_values)
    c_max = max(c_values)
    denom = max(1, c_max - c_min)

    fig, ax = plt.subplots(figsize=(9, 6))
    all_w_values: list[float] = []
    for c in c_values:
        frac = (c - c_min) / denom
        color = cm.viridis(frac)
        w_vals = interp.curve(float(c), t_plot)
        all_w_values.extend(w_vals)
        ax.plot(t_plot, w_vals, color=color, linewidth=1.2, alpha=0.92)

    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("w")
    y_min = min(all_w_values)
    y_max = max(all_w_values)
    span = max(1e-9, y_max - y_min)
    pad = 0.05 * span
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"Overlay of $w_{{c,{k}}}(t)$ for integer c")

    norm = colors.Normalize(vmin=c_min, vmax=c_max)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("c (integer)")

    fig.tight_layout()
    out = Path(save_path) if save_path is not None else RESULTS_DIR / f"w_overlay_k{k}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    print(f"Saved plot: {out}")
    if not no_show:
        plt.show()
    return out


def plot_w_tripanel(
    ks: Sequence[int] = (1, 2, 3),
    table_paths: Sequence[Path | None] | None = None,
    plot_points: int = 500,
    save_path: Path | None = None,
    no_show: bool = True,
) -> Path:
    """One panel of w_{c,k}(t) per k, sharing a single colorbar for c.

    LaTeX-rendered labels (via style.setup_rcparams), 3:1 landscape figure,
    integer-c curves colored by a shared viridis colorbar.
    Output default: results/w_tripanel.pdf
    """
    if len(ks) == 0:
        raise ValueError("ks must be nonempty")
    if table_paths is not None and len(table_paths) != len(ks):
        raise ValueError("table_paths must match ks in length")

    interps = []
    for i, k in enumerate(ks):
        if k < 1:
            raise ValueError("k must be >= 1")
        override = table_paths[i] if table_paths is not None else None
        table = Path(override) if override is not None else default_w_table_path(k)
        interps.append(WTableInterpolator.from_matrix_csv(table))

    # Shared integer-c range: intersection across tables, so one colorbar is
    # honest for every panel.
    c_lo = max(math.ceil(interp.c_min) for interp in interps)
    c_hi = min(math.floor(interp.c_max) for interp in interps)
    if c_hi < c_lo:
        raise ValueError("tables have no common integer c range")
    c_values = list(range(c_lo, c_hi + 1))

    from .style import setup_rcparams

    setup_rcparams()

    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    norm = colors.Normalize(vmin=c_lo, vmax=c_hi)
    # Nearly the full viridis range for hue diversity; the last few percent
    # are dropped because the palest yellow washes out on white.
    cmap = colors.ListedColormap([cm.viridis(0.95 * i / 255.0) for i in range(256)])

    n_panels = len(ks)
    fig_width = 12.0
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width, fig_width / 3.0),
        sharey=True,
        constrained_layout=True,
    )
    fig.get_layout_engine().set(wspace=0.08)
    axes_list = list(axes.flat) if n_panels > 1 else [axes]

    for ax, k, interp in zip(axes_list, ks, interps):
        t_plot = make_log_grid(interp.t_min_pos, interp.t_max, plot_points)
        for c in c_values:
            ax.plot(
                t_plot,
                interp.curve(float(c), t_plot),
                color=cmap(norm(c)),
                linewidth=1.3,
                alpha=0.9,
                solid_capstyle="round",
            )
        ax.set_xscale("log")
        ax.set_xlim(interp.t_min_pos, interp.t_max)
        # Labeled majors every two decades, quiet minors at every decade.
        ax.xaxis.set_major_locator(LogLocator(base=100.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(rf"$k = {k}$", fontsize=13)
        ax.set_xlabel(r"$t$", fontsize=12)
        ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    axes_list[0].set_ylabel(r"$w_{c,k}(t)$", fontsize=12)
    axes_list[0].set_ylim(-0.03, 1.03)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_list, pad=0.015, aspect=28)
    cbar.ax.set_title(r"$c$", fontsize=12, pad=8)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_visible(False)

    if save_path is not None:
        out = Path(save_path)
    elif tuple(ks) == (1, 2, 3):
        out = RESULTS_DIR / "w_tripanel.pdf"
    else:
        out = RESULTS_DIR / ("w_tripanel_k" + "_".join(str(k) for k in ks) + ".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved plot: {out}")
    if not no_show:
        plt.show()
    return out


def plot_b_calibration(
    k: int,
    table_path: Path | None = None,
    save_path: Path | None = None,
    dpi: int = 180,
    no_show: bool = True,
) -> Path:
    """Plot one calibrated b(c) table with its fallback region."""
    if k < 1:
        raise ValueError("k must be >= 1")

    table = Path(table_path) if table_path is not None else default_b_table_path(k)
    curve = _read_b_curve(table)
    out = Path(save_path) if save_path is not None else RESULTS_DIR / f"b_calibration_k{k}.png"
    return render_calibration_tripanel(
        ks=(k,),
        curves=(curve,),
        x_min=curve.x[0],
        x_max=curve.x[-1],
        x_label=r"drift $c$",
        y_label=rf"calibrated $b_{k}(c)$",
        calibrated_label="calibrated branch",
        fallback_label="infeasible-match fallback",
        save_path=out,
        no_show=no_show,
        dpi=dpi,
    )


def _read_b_curve(csv_path: Path) -> CalibrationPlotCurve:
    """Load a b table and mark rows where exact matching is infeasible."""
    if not csv_path.exists():
        raise FileNotFoundError(f"calibration table not found: {csv_path}")

    rows: list[tuple[float, float, bool]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV is empty or missing header")
        required = {"c", "b", "a_psi"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must contain columns: c,b,a_psi")

        for row in reader:
            a_psi = float(row["a_psi"])
            fallback = math.isfinite(a_psi) and a_psi >= _FEASIBILITY_THRESHOLD
            rows.append((float(row["c"]), float(row["b"]), fallback))

    if not rows:
        raise ValueError("CSV has no data rows")
    rows.sort(key=lambda row: row[0])
    return CalibrationPlotCurve(
        x=tuple(row[0] for row in rows),
        b=tuple(row[1] for row in rows),
        fallback=tuple(row[2] for row in rows),
    )


def plot_refined_b_tripanel(
    ks: Sequence[int] = (1, 2, 3),
    table_paths: Sequence[Path | None] | None = None,
    c_min: float | None = -8.0,
    c_max: float | None = 4.0,
    save_path: Path | None = None,
    no_show: bool = True,
) -> Path:
    """Plot the refined-RQ calibrated b(c) curves in matching panels.

    The default displayed range is c in [-8, 4], clamped to the shared table
    domain.  Calibrated/capped portions are solid; explicit infeasible-match
    fallbacks are dashed red with a light red background.
    """
    if not ks:
        raise ValueError("ks must be nonempty")
    if table_paths is not None and len(table_paths) != len(ks):
        raise ValueError("table_paths must match ks in length")

    curves: list[CalibrationPlotCurve] = []
    for i, k in enumerate(ks):
        if k < 1:
            raise ValueError("k must be >= 1")
        override = table_paths[i] if table_paths is not None else None
        table = Path(override) if override is not None else default_b_table_path(k)
        curves.append(_read_b_curve(table))

    table_min = max(curve.x[0] for curve in curves)
    table_max = min(curve.x[-1] for curve in curves)
    plot_min = table_min if c_min is None else max(float(c_min), table_min)
    plot_max = table_max if c_max is None else min(float(c_max), table_max)
    if not (math.isfinite(plot_min) and math.isfinite(plot_max) and plot_max > plot_min):
        raise ValueError("tables and requested c range have no common interval")

    if save_path is not None:
        out = Path(save_path)
    elif tuple(ks) == (1, 2, 3):
        out = RESULTS_DIR / "refined_rq_b_tripanel.pdf"
    else:
        out = RESULTS_DIR / (
            "refined_rq_b_tripanel_k" + "_".join(str(k) for k in ks) + ".pdf"
        )
    return render_calibration_tripanel(
        ks=ks,
        curves=curves,
        x_min=plot_min,
        x_max=plot_max,
        x_label=r"drift $c$",
        y_label=r"calibrated $b_k(c)$",
        calibrated_label="calibrated branch",
        fallback_label="infeasible-match fallback",
        save_path=out,
        no_show=no_show,
    )
