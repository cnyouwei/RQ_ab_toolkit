"""Diagnostic figures for the w_{c,k}(t) and b(c) tables.

Ports of the old scripts plot_w_overlay_k.py (w_overlay_k{k}.png),
plot_b_calibration.py (b_calibration_k{k}.png) and plot_w.py
(single-curve w(t) plots).  Pure plotting: tables/CSVs must already
exist; nothing is generated here.
"""
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


def _read_b_table_with_status(csv_path: Path) -> tuple[list[float], list[float], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"calibration table not found: {csv_path}")

    c_values: list[float] = []
    b_values: list[float] = []
    status_values: list[str] = []

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV is empty or missing header")
        required = {"c", "b", "status"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must contain columns: c,b,status")

        for row in reader:
            c_values.append(float(row["c"]))
            b_values.append(float(row["b"]))
            status_values.append(str(row["status"]).strip())

    if not c_values:
        raise ValueError("CSV has no data rows")
    return c_values, b_values, status_values


def plot_b_calibration(
    k: int,
    table_path: Path | None = None,
    save_path: Path | None = None,
    dpi: int = 180,
    no_show: bool = True,
) -> Path:
    """Plot calibrated b(c) for a fixed k from b_table_k{k}.csv.

    Segments are colored by calibration status (exact / best_fit) with a
    sqrt(2) reference line.  Output default: results/b_calibration_k{k}.png
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    table = Path(table_path) if table_path is not None else default_b_table_path(k)
    c_values, b_values, status_values = _read_b_table_with_status(table)

    import matplotlib.pyplot as plt

    points = sorted(zip(c_values, b_values, status_values), key=lambda x: x[0])
    c_sorted = [p[0] for p in points]
    sqrt2 = math.sqrt(2.0)
    b_sorted = [p[1] for p in points]
    s_sorted = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5))
    style_by_status = {
        "exact": ("tab:blue", "-"),
        "best_fit": ("tab:red", "--"),
    }
    shown_labels: set[str] = set()
    start = 0
    while start < len(c_sorted):
        status = s_sorted[start]
        end = start
        while end + 1 < len(c_sorted) and s_sorted[end + 1] == status:
            end += 1

        color, linestyle = style_by_status.get(status, ("tab:gray", "-."))
        label = status if status not in shown_labels else None
        ax.plot(
            c_sorted[start : end + 1],
            b_sorted[start : end + 1],
            linewidth=2.0,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        if label is not None:
            shown_labels.add(status)
        start = end + 1

    ax.axhline(sqrt2, color="gray", linestyle="--", linewidth=1.2, label=r"$\sqrt{2}$")
    ax.set_xlabel("c")
    ax.set_ylabel("b")
    ax.set_title(f"Calibrated b(c) for k={k}")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()

    out = Path(save_path) if save_path is not None else RESULTS_DIR / f"b_calibration_k{k}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    print(f"Saved plot: {out}")
    if not no_show:
        plt.show()
    return out


def plot_b_overlay(
    ks: Sequence[int] = (1, 2, 3),
    table_paths: Sequence[Path | None] | None = None,
    save_path: Path | None = None,
    no_show: bool = True,
) -> Path:
    """Overlay the calibrated b(c) curves for several k in one figure.

    LaTeX-rendered labels (via style.setup_rcparams); one line style per k
    with a sqrt(2) reference line.  Output default: results/b_overlay.pdf
    """
    if len(ks) == 0:
        raise ValueError("ks must be nonempty")
    if table_paths is not None and len(table_paths) != len(ks):
        raise ValueError("table_paths must match ks in length")

    curves: list[tuple[list[float], list[float]]] = []
    for i, k in enumerate(ks):
        if k < 1:
            raise ValueError("k must be >= 1")
        override = table_paths[i] if table_paths is not None else None
        table = Path(override) if override is not None else default_b_table_path(k)
        c_values, b_values, _ = _read_b_table_with_status(table)
        points = sorted(zip(c_values, b_values), key=lambda p: p[0])
        curves.append(([p[0] for p in points], [p[1] for p in points]))

    from .style import setup_rcparams

    setup_rcparams()

    import matplotlib.pyplot as plt

    line_styles = ["-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for i, (k, (c_sorted, b_sorted)) in enumerate(zip(ks, curves)):
        ax.plot(
            c_sorted,
            b_sorted,
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.8,
            color=f"C{i}",
            label=rf"$k = {k}$",
        )
    ax.axhline(
        math.sqrt(2.0),
        color="gray",
        linestyle=(0, (1, 3)),
        linewidth=1.2,
        label=r"$\sqrt{2}$",
    )
    ax.set_xlabel(r"$c$", fontsize=12)
    ax.set_ylabel(r"$b(c)$", fontsize=12)
    ax.set_title(r"Calibrated $b(c)$", fontsize=13)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=11, frameon=False)

    if save_path is not None:
        out = Path(save_path)
    elif tuple(ks) == (1, 2, 3):
        out = RESULTS_DIR / "b_overlay.pdf"
    else:
        out = RESULTS_DIR / ("b_overlay_k" + "_".join(str(k) for k in ks) + ".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved plot: {out}")
    if not no_show:
        plt.show()
    return out


def plot_w_curves(
    csv_paths: Sequence[Path],
    save: Path | None = None,
    no_show: bool = True,
) -> None:
    """Plot one or more single-curve w_{c,k}(t) CSVs (columns t,w) on log-x.

    Folds in the old scripts/plot_w.py single-curve plot; with several
    paths the curves are overlaid and labeled by file stem.
    """
    if not csv_paths:
        raise ValueError("plot_w_curves needs at least one CSV path")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    y_all: list[float] = []
    for raw_path in csv_paths:
        csv_path = Path(raw_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        t_values: list[float] = []
        w_values: list[float] = []
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV is empty or missing header.")
            if "t" not in reader.fieldnames or "w" not in reader.fieldnames:
                raise ValueError("CSV header must contain columns: t,w")
            for row in reader:
                t_values.append(float(row["t"]))
                w_values.append(float(row["w"]))
        if not t_values:
            raise ValueError("CSV has no data rows.")

        # log-x plot excludes t=0 from the line, if present.
        t_positive = [t for t in t_values if t > 0.0]
        w_positive = [w for t, w in zip(t_values, w_values) if t > 0.0]
        if not t_positive:
            raise ValueError("No positive t values found. Cannot plot on log x-axis.")

        ax.plot(t_positive, w_positive, linewidth=2.0, label=csv_path.stem)
        y_all.extend(w_positive)
        if any(t == 0.0 for t in t_values):
            y_all.append(w_values[t_values.index(0.0)])

    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("w")
    ax.set_title(r"$w_{c,k}(t)$")
    ax.grid(True, which="both", alpha=0.35)
    y_min = min(y_all)
    y_max = max(y_all)
    span = max(1e-9, y_max - y_min)
    pad = 0.05 * span
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend()
    fig.tight_layout()

    if save is not None:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        print(f"Saved plot: {out}")

    if not no_show:
        plt.show()
