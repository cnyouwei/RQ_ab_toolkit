"""Signed-relative-error heatmap panels: approximation vs simulated workload.

Consolidates the old scripts plot_refined_rq_ratio.py (1 panel),
plot_approx_ratio_twopanel.py (First RQ | Refined RQ) and
plot_approx_ratio_tripanel.py (RQ | WG-or-Hazard | HG).

Pure plotting: these functions load existing CSVs and draw figures.  They
never run upstream simulation/grid stages; missing inputs raise
FileNotFoundError with a pointer to the file that is needed.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
from typing import Any

from .style import DIVERGING_CMAP, SINGLE_PANEL_CMAP, make_norm, setup_rcparams

Panel = tuple[str, list[list[float]]]


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_workload_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Load the workload-MC aggregate CSV keyed by tuple_id."""
    if not path.exists():
        raise FileNotFoundError(f"workload aggregate CSV not found: {path}")

    by_id: dict[int, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"tuple_id", "lambda", "alpha", "mean_workload", "model_name"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "workload aggregate CSV must contain: tuple_id,lambda,alpha,mean_workload,model_name"
            )
        for row in reader:
            tuple_id = int(row["tuple_id"])
            by_id[tuple_id] = {
                "lambda": float(row["lambda"]),
                "alpha": float(row["alpha"]),
                "mean_workload": float(row["mean_workload"]),
                "model_name": str(row["model_name"]),
            }
    if not by_id:
        raise ValueError("workload aggregate CSV has no data rows")
    return by_id


def load_z_rows(
    path: Path,
    z_col: str,
    status_col: str = "solver_status",
    label: str = "aggregate",
) -> dict[int, float]:
    """Load one approximation column keyed by tuple_id.

    Keeps only rows whose ``status_col`` starts with "ok" and whose z value
    is non-empty (the old refined/first-RQ loader semantics).  ``label`` is
    used in error messages, e.g. "refined" or "first-RQ".
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} aggregate CSV not found: {path}")

    by_id: dict[int, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"tuple_id", z_col, status_col}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{label} aggregate CSV must contain: tuple_id,{z_col},{status_col}"
            )
        for row in reader:
            if not str(row[status_col]).startswith("ok"):
                continue
            z_raw = row.get(z_col, "")
            if z_raw is None or z_raw == "":
                continue
            by_id[int(row["tuple_id"])] = float(z_raw)
    return by_id


def load_combined_secondary_rows(path: Path) -> tuple[dict[int, dict[str, Any]], str]:
    """Load the secondary-method (WG or Hazard) and HG columns.

    Reads the combined refined/other aggregate CSV and returns
    ``(rows, method_title)`` where each row is
    ``{"secondary_method", "z_secondary", "z_hg"}`` keyed by tuple_id.
    Values whose status column does not start with "ok" become NaN.  Both
    "wg" and the legacy tandem "gw" CSV labels are accepted for the
    Ward--Glynn method; the displayed title is always "WG".
    """
    if not path.exists():
        raise FileNotFoundError(f"combined aggregate CSV not found: {path}")

    other_rows: dict[int, dict[str, Any]] = {}
    secondary_set: set[str] = set()

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "tuple_id",
            "secondary_method",
            "z_secondary",
            "z_hg",
            "status_secondary",
            "status_hg",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError("combined CSV missing required columns: " + ",".join(missing))

        for row in reader:
            tuple_id = int(row["tuple_id"])

            method = str(row.get("secondary_method", "")).strip().lower()
            if method:
                secondary_set.add(method)

            sec_ok = str(row.get("status_secondary", "")).startswith("ok")
            hg_ok = str(row.get("status_hg", "")).startswith("ok")

            z_sec_raw = row.get("z_secondary", "")
            z_hg_raw = row.get("z_hg", "")
            z_sec = float(z_sec_raw) if sec_ok and z_sec_raw not in ("", None) else math.nan
            z_hg = float(z_hg_raw) if hg_ok and z_hg_raw not in ("", None) else math.nan
            other_rows[tuple_id] = {
                "secondary_method": method,
                "z_secondary": z_sec,
                "z_hg": z_hg,
            }

    if not other_rows:
        raise ValueError("combined CSV has no data rows")

    method_title = "WG/Hazard"
    if len(secondary_set) == 1:
        only = next(iter(secondary_set))
        if only in ("wg", "gw"):
            method_title = "WG"
        elif only == "hazard":
            method_title = "Hazard rate"
    elif "hazard" in secondary_set and ("wg" not in secondary_set and "gw" not in secondary_set):
        method_title = "Hazard rate"
    elif ("wg" in secondary_set or "gw" in secondary_set) and "hazard" not in secondary_set:
        method_title = "WG"
    return other_rows, method_title


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_panel_grid(
    workload_rows: dict[int, dict[str, Any]],
    z_by_id: dict[int, float],
    patience_base_mean: float,
    missing_label: str = "results",
    warn_bad_workload: bool = True,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Arrange z/mean_workload ratios on the (mean patience x arrival rate) grid.

    Filters to alpha <= 1, sorts lambda values DESCENDING (y axis) and
    mean-patience values (patience_base_mean / alpha) ASCENDING (x axis),
    and fills NaN where a tuple has no usable z value.  Returns
    ``(patience_values, lambda_values, ratio_grid)``.
    """
    alpha_max = 1.0
    valid_workload_rows = {
        tuple_id: row
        for tuple_id, row in workload_rows.items()
        if float(row["alpha"]) <= alpha_max
    }
    if not valid_workload_rows:
        raise ValueError("no tuples remain after alpha filter (alpha <= 1)")

    lambda_values = sorted({float(v["lambda"]) for v in valid_workload_rows.values()}, reverse=True)
    patience_values = sorted(
        {patience_base_mean / float(v["alpha"]) for v in valid_workload_rows.values()}
    )
    y_index = {v: i for i, v in enumerate(lambda_values)}
    x_index = {v: i for i, v in enumerate(patience_values)}

    grid = [[math.nan for _ in patience_values] for _ in lambda_values]

    missing = 0
    bad_workload = 0
    for tuple_id, w in valid_workload_rows.items():
        mean_w = float(w["mean_workload"])
        if not (mean_w > 0.0 and math.isfinite(mean_w)):
            bad_workload += 1
            continue

        lam = float(w["lambda"])
        alpha = float(w["alpha"])
        iy = y_index[lam]
        ix = x_index[patience_base_mean / alpha]

        if tuple_id not in z_by_id:
            missing += 1
            continue
        z = float(z_by_id[tuple_id])
        grid[iy][ix] = z / mean_w if math.isfinite(z) else math.nan

    if warn_bad_workload and bad_workload > 0:
        print(
            f"warning: skipped {bad_workload} tuples with non-positive/non-finite mean workload",
            file=sys.stderr,
        )
    if missing > 0:
        print(f"warning: {missing} tuples missing {missing_label}", file=sys.stderr)

    return patience_values, lambda_values, grid


def load_refined_c_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Load (lambda, alpha, c) per tuple from a refined-RQ aggregate CSV.

    Keeps rows with a non-empty c value: c is set whenever the scaling
    kernel was built, even when the later fixed point failed.
    """
    if not path.exists():
        raise FileNotFoundError(f"refined aggregate CSV not found: {path}")

    by_id: dict[int, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"tuple_id", "lambda", "alpha", "c"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("refined aggregate CSV must contain: tuple_id,lambda,alpha,c")
        for row in reader:
            c_raw = row.get("c", "")
            if c_raw is None or c_raw == "":
                continue
            by_id[int(row["tuple_id"])] = {
                "lambda": float(row["lambda"]),
                "alpha": float(row["alpha"]),
                "c": float(c_raw),
            }
    return by_id


def build_c_grid(
    c_rows: dict[int, dict[str, Any]],
    patience_base_mean: float,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Arrange the refined-RQ drift parameter c on the heatmap grid.

    Same conventions as build_panel_grid: filters to alpha <= 1, lambda
    DESCENDING (y axis), mean patience (patience_base_mean / alpha)
    ASCENDING (x axis), NaN where a tuple has no usable c.
    """
    alpha_max = 1.0
    valid_rows = {
        tuple_id: row
        for tuple_id, row in c_rows.items()
        if float(row["alpha"]) <= alpha_max
    }
    if not valid_rows:
        raise ValueError("no tuples remain after alpha filter (alpha <= 1)")

    lambda_values = sorted({float(v["lambda"]) for v in valid_rows.values()}, reverse=True)
    patience_values = sorted(
        {patience_base_mean / float(v["alpha"]) for v in valid_rows.values()}
    )
    y_index = {v: i for i, v in enumerate(lambda_values)}
    x_index = {v: i for i, v in enumerate(patience_values)}

    grid = [[math.nan for _ in patience_values] for _ in lambda_values]
    for row in valid_rows.values():
        iy = y_index[float(row["lambda"])]
        ix = x_index[patience_base_mean / float(row["alpha"])]
        c = float(row["c"])
        grid[iy][ix] = c if math.isfinite(c) else math.nan
    return patience_values, lambda_values, grid


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_ratio_panels(
    patience_values: list[float],
    lambda_values: list[float],
    panels: list[Panel],
    model_title: str,
    save_path: Path,
    vmin_override: float | None = None,
    vmax_override: float | None = None,
    no_show: bool = True,
) -> None:
    """Render 1, 2 or 3 signed-percent-error heatmap panels.

    ``panels`` is a list of ``(panel_title, ratio_grid)``; each grid entry
    is the ratio approximation/simulated-workload (NaN = missing).  The
    layouts replicate the old plotters exactly:

    * 1 panel: plot_refined_rq_ratio.py (7.0x4.8, "+x.x%" annotations)
    * 2 panels: plot_approx_ratio_twopanel.py (7.2x4)
    * 3 panels: plot_approx_ratio_tripanel.py (10x4, suptitle centered on
      the middle panel)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not panels:
        raise ValueError("render_ratio_panels needs at least one panel")

    use_tex = setup_rcparams()
    norm = make_norm(vmin_override, vmax_override)

    panel_mats = [100.0 * (np.array(grid, dtype=float) - 1.0) for _title, grid in panels]
    titles = [title for title, _grid in panels]

    save_path = Path(save_path)

    if len(panels) == 1:
        # Legacy single-panel refined_rq_ratio layout.
        mat = panel_mats[0]
        if np.isfinite(mat).sum() == 0:
            raise ValueError("no finite percentage-difference values available for plotting")

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            norm=norm,
            cmap=SINGLE_PANEL_CMAP,
        )

        x_ticks = np.arange(len(patience_values))
        y_ticks = np.arange(len(lambda_values))
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{x:.4g}" for x in patience_values], rotation=45, ha="right")
        ax.set_yticklabels([f"{y:.4g}" for y in lambda_values])

        ax.set_xlabel("mean patience")
        ax.set_ylabel("arrival rate")
        ax.set_title(model_title)

        for iy in range(mat.shape[0]):
            for ix in range(mat.shape[1]):
                val = mat[iy, ix]
                if not math.isfinite(float(val)):
                    continue
                normalized = float(norm(val))
                text_color = "white" if abs(normalized - 0.5) > 0.28 else "black"
                if use_tex:
                    entry_label = f"{float(val):+.1f}\\%"
                else:
                    entry_label = f"{float(val):+.1f}%"
                ax.text(
                    ix,
                    iy,
                    entry_label,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"Saved plot: {save_path}")
        if not no_show:
            plt.show()
        return

    # Multi-panel (2 or 3) shared layout.
    for mat in panel_mats:
        if np.isfinite(mat).sum() == 0:
            raise ValueError("all panels are empty; no finite ratio values available for plotting")

    n_panels = len(panels)
    figsize = (7.2, 4) if n_panels == 2 else (10, 4)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes], dtype=object)

    x_ticks = np.arange(len(patience_values))
    y_ticks = np.arange(len(lambda_values))

    for ax, mat, panel_title in zip(axes, panel_mats, titles):
        ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            norm=norm,
            cmap=DIVERGING_CMAP,
        )
        ax.set_title(panel_title, fontsize=9)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{x:.4g}" for x in patience_values], rotation=45, ha="right", fontsize=6.5)
        ax.set_yticklabels([f"{y:.4g}" for y in lambda_values], fontsize=7)
        ax.set_xlabel("mean patience", fontsize=11)

        for iy in range(mat.shape[0]):
            for ix in range(mat.shape[1]):
                val = float(mat[iy, ix])
                if not math.isfinite(val):
                    continue
                normalized = float(norm(val))
                text_color = "white" if abs(normalized - 0.5) > 0.28 else "black"
                # Display decimal relative error (e.g., 1% -> 0.01) and keep only negative sign.
                label = f"{val / 100.0:.2f}"
                ax.text(
                    ix,
                    iy,
                    label,
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=text_color,
                )

    axes[0].set_ylabel("arrival rate", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.3, w_pad=0.4, h_pad=0.4)
    if n_panels == 2:
        fig.suptitle(model_title, fontsize=11, x=0.5, y=0.97)
    else:
        mid_ax = axes[len(axes) // 2]
        mid_bbox = mid_ax.get_position()
        title_x = 0.5 * (mid_bbox.x0 + mid_bbox.x1)
        fig.suptitle(model_title, fontsize=11, x=title_x, y=0.97)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"Saved plot: {save_path}")

    if not no_show:
        plt.show()


# ---------------------------------------------------------------------------
# Panel presets
#
# The CLI/reproduce layer resolves the model alias, base mean patience and
# plot title (rqab.models.model_plot_metadata) and passes explicit CSV
# paths; these presets only load CSVs and plot.
# ---------------------------------------------------------------------------

def figure_ratio(
    workload_csv: Path,
    refined_csv: Path,
    patience_base_mean: float,
    model_title: str,
    save_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    no_show: bool = True,
) -> Path:
    """Single-panel refined-RQ / workload ratio heatmap.

    Canonical output name: results/refined_rq_ratio_<alias>.pdf
    """
    workload_rows = load_workload_rows(Path(workload_csv))
    refined_rows = load_z_rows(Path(refined_csv), "z_rq_refined", label="refined")
    if not refined_rows:
        raise ValueError("refined aggregate CSV has no usable rows (solver_status=ok)")

    patience_values, lambda_values, grid = build_panel_grid(
        workload_rows,
        refined_rows,
        patience_base_mean,
        missing_label="refined-RQ results",
    )

    out = Path(save_path)
    render_ratio_panels(
        patience_values,
        lambda_values,
        [("Refined RQ", grid)],
        model_title=model_title,
        save_path=out,
        vmin_override=vmin,
        vmax_override=vmax,
        no_show=no_show,
    )
    return out


def figure_c_heatmap(
    refined_csv: Path,
    patience_base_mean: float,
    model_title: str,
    save_path: Path,
    no_show: bool = True,
) -> Path:
    """Heatmap of the refined-RQ drift parameter c on the ratio-plot grid.

    Same (mean patience x arrival rate) arrangement and cell annotations as
    the single-panel ratio heatmap, colored by the c value each tuple feeds
    into the w_{c,k} lookup.  c is signed and spans orders of magnitude in
    both directions across the grid, so the color scale is a symmetric-log
    diverging norm centered at c = 0.
    Canonical output name: results/c_heatmap_<alias>.pdf
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors

    c_rows = load_refined_c_rows(Path(refined_csv))
    if not c_rows:
        raise ValueError("refined aggregate CSV has no rows with a c value")
    patience_values, lambda_values, grid = build_c_grid(c_rows, patience_base_mean)

    setup_rcparams()

    mat = np.array(grid, dtype=float)
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        raise ValueError("no finite c values available for plotting")
    vmax_abs = float(np.abs(finite).max())
    nonzero = np.abs(finite[finite != 0.0])
    linthresh = float(nonzero.min()) if nonzero.size else 1.0
    norm = colors.SymLogNorm(
        linthresh=linthresh, vmin=-vmax_abs, vmax=vmax_abs, base=10
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    im = ax.imshow(
        mat,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        norm=norm,
        cmap=DIVERGING_CMAP,
    )

    x_ticks = np.arange(len(patience_values))
    y_ticks = np.arange(len(lambda_values))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{x:.4g}" for x in patience_values], rotation=45, ha="right")
    ax.set_yticklabels([f"{y:.4g}" for y in lambda_values])

    ax.set_xlabel("mean patience")
    ax.set_ylabel("arrival rate")
    ax.set_title(model_title)

    for iy in range(mat.shape[0]):
        for ix in range(mat.shape[1]):
            val = mat[iy, ix]
            if not math.isfinite(float(val)):
                continue
            text_color = "white" if abs(float(norm(val)) - 0.5) > 0.28 else "black"
            ax.text(
                ix,
                iy,
                f"{float(val):.2g}",
                ha="center",
                va="center",
                fontsize=6,
                color=text_color,
            )

    # Explicit decade ticks: SymLogNorm's default ticks crowd around zero.
    emax = math.floor(math.log10(vmax_abs))
    decades = [10.0 ** e for e in range(emax - 2, emax + 1)]
    ticks = [-d for d in reversed(decades)] + [0.0] + decades
    cbar = fig.colorbar(im, ax=ax, pad=0.02, ticks=ticks)
    cbar.minorticks_off()
    cbar.ax.set_title(r"$c$", fontsize=12, pad=8)
    cbar.outline.set_visible(False)

    out = Path(save_path)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"Saved plot: {out}")
    if not no_show:
        plt.show()
    return out


def figure_twopanel(
    first_csv: Path,
    workload_csv: Path,
    refined_csv: Path,
    patience_base_mean: float,
    model_title: str,
    save_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    no_show: bool = True,
) -> Path:
    """Two-panel (First RQ | Refined RQ) ratio heatmap.

    Canonical output name: results/approx_ratio_twopanel_<alias>.pdf
    """
    first_csv = Path(first_csv)
    refined_csv = Path(refined_csv)
    workload_rows = load_workload_rows(Path(workload_csv))
    first_rows = load_z_rows(first_csv, "z_rq_first", label="first-RQ")
    if not first_rows:
        raise ValueError("first-RQ aggregate CSV has no usable rows")
    refined_rows = load_z_rows(refined_csv, "z_rq_refined", label="refined")
    if not refined_rows:
        raise ValueError("refined aggregate CSV has no usable rows")

    patience_values, lambda_values, first_grid = build_panel_grid(
        workload_rows,
        first_rows,
        patience_base_mean,
        missing_label="first-RQ results",
    )
    _, _, refined_grid = build_panel_grid(
        workload_rows,
        refined_rows,
        patience_base_mean,
        missing_label="refined-RQ results",
        warn_bad_workload=False,
    )

    out = Path(save_path)
    render_ratio_panels(
        patience_values,
        lambda_values,
        [("First RQ", first_grid), ("Refined RQ", refined_grid)],
        model_title=model_title,
        save_path=out,
        vmin_override=vmin,
        vmax_override=vmax,
        no_show=no_show,
    )
    return out


def figure_tripanel(
    workload_csv: Path,
    refined_csv: Path,
    patience_base_mean: float,
    model_title: str,
    save_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    no_show: bool = True,
) -> Path:
    """Three-panel (RQ | WG-or-Hazard | HG) ratio heatmap.

    The secondary/HG columns come from the combined refined CSV.
    Canonical output name: results/approx_ratio_tripanel_<alias>.pdf
    """
    refined_csv = Path(refined_csv)
    workload_rows = load_workload_rows(Path(workload_csv))
    refined_rows = load_z_rows(refined_csv, "z_rq_refined", label="refined")
    try:
        other_rows, secondary_title = load_combined_secondary_rows(refined_csv)
    except ValueError as exc:
        if "missing required columns" in str(exc):
            raise ValueError(
                f"refined CSV lacks the secondary/HG columns needed for the tripanel "
                f"({exc}); it was probably written by an old refined-only run. "
                f"Regenerate it with: scripts/run_grid.py --method refined --force-rerun"
            ) from exc
        raise

    sec_by_id = {tid: row["z_secondary"] for tid, row in other_rows.items()}
    hg_by_id = {tid: row["z_hg"] for tid, row in other_rows.items()}

    patience_values, lambda_values, rq_grid = build_panel_grid(
        workload_rows,
        refined_rows,
        patience_base_mean,
        missing_label="refined-RQ results",
    )
    _, _, sec_grid = build_panel_grid(
        workload_rows,
        sec_by_id,
        patience_base_mean,
        missing_label="other-method results",
        warn_bad_workload=False,
    )
    _, _, hg_grid = build_panel_grid(
        workload_rows,
        hg_by_id,
        patience_base_mean,
        missing_label="HG results",
        warn_bad_workload=False,
    )

    out = Path(save_path)
    render_ratio_panels(
        patience_values,
        lambda_values,
        [("RQ", rq_grid), (secondary_title, sec_grid), ("HG", hg_grid)],
        model_title=model_title,
        save_path=out,
        vmin_override=vmin,
        vmax_override=vmax,
        no_show=no_show,
    )
    return out
