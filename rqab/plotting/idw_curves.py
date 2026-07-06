"""Effective-IDW curve figures driven by configs/effective_idw_*.json.

Port of the live parts of the old scripts/effective_idw_plotter.py and
plot_effective_idw_from_config.py: for each (model, alpha_i) it plots the
simulated effective IDW (solid, from results/model{idx}_{name}_idx{i}_curve.csv
written by the wck_effective_idw_sim binary) against the analytical
approximation hat_Iw(t) * w_{tilde c,k}(alpha^{2h} tau t) (dashed).

Pure plotting: no simulations are run here.  The optional
``run_missing_sim`` callable lets the CLI layer generate missing overlay
CSVs before the figure is drawn.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

from ..effective_idw import (
    distribution_beta_at_zero,
    distribution_moments,
    effective_idw_approx,
    h2_shape_from_canonical,
    idc_erlang_equilibrium,
    idc_h2_equilibrium,
    tau_tilde_c,
)
from ..models import parse_distribution_component
from ..tables import WTableInterpolator, make_log_grid
from . import style

Curve = dict[str, Any]
ModelResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def parse_indices_csv(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        val = int(item)
        if val < 0:
            raise ValueError("indices must be nonnegative integers")
        out.append(val)
    if not out:
        raise ValueError("indices must contain at least one integer")
    return out


def _parse_indices_any(raw: Any) -> list[int]:
    if isinstance(raw, str):
        return parse_indices_csv(raw)
    if isinstance(raw, list):
        out = [int(v) for v in raw]
        if any(v < 0 for v in out):
            raise ValueError("alpha.indices must be nonnegative integers")
        if not out:
            raise ValueError("alpha.indices cannot be empty")
        return out
    raise ValueError("alpha.indices must be a list or comma-separated string")


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    """Resolve a config path relative to the config file's directory."""
    p = path_value if isinstance(path_value, Path) else Path(str(path_value))
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _sanitize_name_for_file(name: str) -> str:
    out: list[str] = []
    prev_underscore = False
    for ch in name:
        if ch.isalnum():
            out.append(ch.lower())
            prev_underscore = False
            continue
        if not prev_underscore:
            out.append("_")
            prev_underscore = True
    sanitized = "".join(out).strip("_")
    return sanitized if sanitized else "model"


def _extract_overlay_config(
    cfg: dict[str, Any],
    base_dir: Path,
    default_results_dir: Path,
) -> dict[str, Any]:
    raw = cfg.get("simulation_overlay", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("simulation_overlay must be an object when provided")

    enabled = bool(raw.get("enabled", True))
    results_raw = raw.get("results_dir", str(default_results_dir))
    results_dir = _resolve_path(results_raw, base_dir)
    warn_missing = bool(raw.get("warn_missing", True))
    return {
        "enabled": enabled,
        "results_dir": results_dir,
        "preferred_columns": ["idw_hat"],
        "warn_missing": warn_missing,
    }


def _simulation_curve_path(
    results_dir: Path, model_index: int, model_name: str, alpha_index: int
) -> Path:
    prefix = f"model{model_index}_{_sanitize_name_for_file(model_name)}_idx{alpha_index}"
    return results_dir / f"{prefix}_curve.csv"


def _load_estimated_curve_csv(
    path: Path,
    preferred_columns: list[str],
) -> tuple[list[float], list[float], str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "t" not in fieldnames:
            raise ValueError("missing column 't'")

        chosen = ""
        for col in preferred_columns:
            if col in fieldnames:
                chosen = col
                break
        if not chosen:
            raise ValueError(
                "missing estimator columns; expected one of " + ", ".join(preferred_columns)
            )

        points: list[tuple[float, float]] = []
        for row in reader:
            t_raw = row.get("t")
            y_raw = row.get(chosen)
            if t_raw is None or y_raw is None:
                continue
            try:
                t_val = float(t_raw)
                y_val = float(y_raw)
            except Exception:
                continue
            if not (math.isfinite(t_val) and math.isfinite(y_val)):
                continue
            if t_val <= 0.0:
                continue
            points.append((t_val, y_val))

    if not points:
        raise ValueError("no finite data rows found")
    points.sort(key=lambda x: x[0])
    t_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    return t_values, y_values, chosen


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"failed to parse JSON config: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("top-level config must be a JSON object")
    return data


def _curve_y_limits(model: ModelResult) -> tuple[float, float]:
    y_min = 0.0
    c_x2_raw = float(model.get("c_x2", 1.0))
    c_x2 = c_x2_raw if math.isfinite(c_x2_raw) else 1.0
    y_max = max(2.1, c_x2 + 0.2)
    if y_max <= y_min:
        y_max = y_min + 1.0
    return (y_min, y_max)


def _extract_time_grid(cfg: dict[str, Any]) -> tuple[float, float, int]:
    grid = cfg.get("time_grid", {})
    if not isinstance(grid, dict):
        raise ValueError("time_grid must be an object")
    t_min = float(grid.get("t_min", 1e-2))
    t_max = float(grid.get("t_max", 1e8))
    n_t = int(grid.get("n_t", 600))
    return t_min, t_max, n_t


def _extract_alpha(cfg: dict[str, Any]) -> tuple[list[int], float]:
    alpha = cfg.get("alpha", {})
    if not isinstance(alpha, dict):
        raise ValueError("alpha must be an object")
    indices = _parse_indices_any(alpha.get("indices", [0, 3, 6, 9, 12]))
    base = float(alpha.get("base", 2.0))
    if base <= 1.0:
        raise ValueError("alpha.base must be > 1")
    return indices, base


# ---------------------------------------------------------------------------
# Curve construction (numeric core; verbatim from the old plotter)
# ---------------------------------------------------------------------------

def _build_model_curves(
    model_cfg: dict[str, Any],
    model_index: int,
    t_values: list[float],
    indices: list[int],
    alpha_base: float,
    table_cache: dict[Path, WTableInterpolator],
    base_dir: Path,
    global_curve_label_template: str,
    overlay_cfg: dict[str, Any],
    missing_overlay_messages: list[str],
) -> ModelResult:
    name = str(model_cfg.get("name", "model"))

    if "rho_exponent" in model_cfg:
        raise ValueError(
            f"model '{name}': legacy field model.rho_exponent is removed; use model.scaling.rho_exponent"
        )

    scaling = model_cfg.get("scaling")
    if not isinstance(scaling, dict):
        raise ValueError(f"model '{name}': scaling must be an object")
    extra_scaling = set(scaling.keys()) - {"k", "beta_patience", "rho_exponent"}
    if extra_scaling:
        extras = ", ".join(sorted(extra_scaling))
        raise ValueError(f"model '{name}': unsupported scaling field(s): {extras}")
    if "k" not in scaling:
        raise ValueError(f"model '{name}': missing required field scaling.k")

    k = int(scaling["k"])
    if k < 1:
        raise ValueError(f"model '{name}': scaling.k must be >= 1")
    h = k / (k + 1.0)
    rho_exponent = float(scaling.get("rho_exponent", h))

    system = model_cfg.get("system", {})
    if not isinstance(system, dict):
        raise ValueError(f"model '{name}': system must be an object")
    c = float(system.get("c", 2.0))

    arrival = parse_distribution_component(model_cfg, "arrival", name)
    service = parse_distribution_component(model_cfg, "service", name, allow_lognormal=True)
    patience = parse_distribution_component(model_cfg, "patience", name)

    _arrival_mean, arrival_scv = distribution_moments(arrival.family, arrival.params)
    service_mean, c_s2 = distribution_moments(service.family, service.params)
    mu = 1.0 / service_mean
    beta_patience = float(distribution_beta_at_zero(patience.family, patience.params, k))
    if not math.isfinite(beta_patience) or beta_patience <= 0.0:
        raise ValueError(
            f"model '{name}': scaling.k is inconsistent with patience distribution; "
            f"computed F^(k)(0)/k!={beta_patience} (must be > 0)"
        )
    if "beta_patience" in scaling:
        supplied_beta = float(scaling["beta_patience"])
        if supplied_beta <= 0.0:
            raise ValueError(f"model '{name}': scaling.beta_patience must be > 0")
        tol = 1e-9 * max(1.0, abs(beta_patience))
        if abs(supplied_beta - beta_patience) > tol:
            raise ValueError(
                f"model '{name}': scaling.beta_patience={supplied_beta} is inconsistent "
                f"with patience distribution and scaling.k; expected {beta_patience}"
            )

    w_table_raw = model_cfg.get("w_table")
    if w_table_raw is None:
        raise ValueError(f"model '{name}': missing 'w_table'")
    w_table_path = _resolve_path(w_table_raw, base_dir)
    if not w_table_path.exists():
        raise ValueError(f"model '{name}': missing w-table: {w_table_path}")

    if w_table_path not in table_cache:
        table_cache[w_table_path] = WTableInterpolator.from_matrix_csv(w_table_path)
    interp = table_cache[w_table_path]

    tau, tilde_c = tau_tilde_c(
        c=c,
        k=k,
        mu=mu,
        c_a2=arrival_scv,
        c_s2=c_s2,
        beta_patience=beta_patience,
    )

    curve_label_template = str(model_cfg.get("curve_label_template", global_curve_label_template))
    c_x2 = arrival_scv + c_s2
    curves: list[Curve] = []
    for i in indices:
        alpha = alpha_base ** (-i)
        rho = 1.0 + c * (alpha**rho_exponent)
        lam = rho * mu

        if arrival.family == "exponential":
            ia_t = [1.0 for _ in t_values]
        elif arrival.family == "hyperexponential2":
            p = float(arrival.params["p"])
            rate1 = float(arrival.params["rate1"])
            rate2 = float(arrival.params["rate2"])
            h2_scv, h2_r = h2_shape_from_canonical(p=p, rate1=rate1, rate2=rate2)
            ia_t = idc_h2_equilibrium(t=t_values, rate=lam, scv=h2_scv, r=h2_r)
        elif arrival.family == "erlang_k":
            erlang_k = int(arrival.params["k"])
            arrival_rate = float(erlang_k) * lam
            ia_t = idc_erlang_equilibrium(t=t_values, k=erlang_k, rate=arrival_rate)
        else:
            raise ValueError(
                f"model '{name}': unsupported arrival distribution family '{arrival.family}'"
            )

        scaled_t = [(alpha ** (2.0 * h)) * tau * t for t in t_values]
        w_values = [interp.w(tilde_c, st) for st in scaled_t]
        approx = effective_idw_approx(
            t=t_values,
            ia_t=ia_t,
            rho=rho,
            c_s2=c_s2,
            w_values=w_values,
        )
        curves.append(
            {
                "i": i,
                "alpha": alpha,
                "rho": rho,
                "y": approx,
                "label": curve_label_template.format(i=i, alpha=alpha, rho=rho, model=name),
            }
        )

        if overlay_cfg["enabled"]:
            curve_path = _simulation_curve_path(
                results_dir=overlay_cfg["results_dir"],
                model_index=model_index,
                model_name=name,
                alpha_index=i,
            )
            if curve_path.exists():
                try:
                    est_t, est_y, est_column = _load_estimated_curve_csv(
                        curve_path,
                        preferred_columns=overlay_cfg["preferred_columns"],
                    )
                    curves[-1]["estimated_t"] = est_t
                    curves[-1]["estimated_y"] = est_y
                    curves[-1]["estimated_column"] = est_column
                except Exception as exc:
                    missing_overlay_messages.append(
                        f"model='{name}', alpha_index={i}, file='{curve_path}': {exc}"
                    )
            else:
                missing_overlay_messages.append(
                    f"model='{name}', alpha_index={i}, expected file='{curve_path}'"
                )

    return {"name": name, "curves": curves, "c_x2": c_x2}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _plot_models(
    save_path: Path,
    t_values: list[float],
    models: list[ModelResult],
    plot_cfg: dict[str, Any],
    dpi: int,
    show: bool,
) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_tex = bool(plot_cfg.get("use_tex", True))
    if use_tex and shutil.which("latex") is None:
        use_tex = False
        print(
            "warning: plot.use_tex=true but no 'latex' executable found; using matplotlib mathtext instead.",
            file=sys.stderr,
        )
    if use_tex and not style.usetex_pdf_ok():
        use_tex = False
        print(
            "warning: this matplotlib drops minus signs from usetex PDF output "
            "(Type-1 subsetting bug, present in 3.11.0); using matplotlib mathtext instead.",
            file=sys.stderr,
        )
    plt.rcParams["text.usetex"] = use_tex
    if use_tex:
        plt.rcParams["font.family"] = str(plot_cfg.get("tex_font_family", "serif"))
    else:
        plt.rcParams["mathtext.fontset"] = "cm"

    n_models = len(models)
    ncols = int(plot_cfg.get("ncols", min(2, n_models)))
    ncols = max(1, min(ncols, n_models))
    nrows = (n_models + ncols - 1) // ncols
    figsize_raw = plot_cfg.get("figsize", [6.4 * ncols, 4.8 * nrows])
    if not isinstance(figsize_raw, list) or len(figsize_raw) != 2:
        raise ValueError("plot.figsize must be [width, height]")
    figsize = (float(figsize_raw[0]), float(figsize_raw[1]))
    sharey = bool(plot_cfg.get("sharey", n_models > 1))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    color_cycle: list[str] = []
    if prop_cycle is not None:
        color_cycle = list(prop_cycle.by_key().get("color", []))
    if not color_cycle:
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#17becf"]

    alpha_order: list[int] = []
    color_by_i: dict[int, str] = {}
    marker_by_i: dict[int, str] = {}
    alpha_label_by_i: dict[int, str] = {}
    for model in models:
        for curve in model["curves"]:
            i_val = int(curve["i"])
            if i_val in marker_by_i:
                continue
            style_idx = len(alpha_order)
            alpha_order.append(i_val)
            color_by_i[i_val] = color_cycle[style_idx % len(color_cycle)]
            marker_by_i[i_val] = marker_cycle[style_idx % len(marker_cycle)]
            alpha_label_by_i[i_val] = rf"$\alpha=2^{{-{i_val}}}$"

    marker_size = float(plot_cfg.get("marker_size", 5.6))
    line_weight = float(plot_cfg.get("line_weight", 1.5))
    approx_marker_step = max(1, len(t_values) // 14)

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        for curve in model["curves"]:
            i_val = int(curve["i"])
            color = color_by_i[i_val]
            marker = marker_by_i[i_val]
            ax.plot(
                t_values,
                curve["y"],
                linewidth=line_weight,
                linestyle="--",
                color=color,
                marker=marker,
                markevery=approx_marker_step,
                markersize=marker_size,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                label="_nolegend_",
            )

            est_t = curve.get("estimated_t")
            est_y = curve.get("estimated_y")
            if isinstance(est_t, list) and isinstance(est_y, list) and len(est_t) >= 2:
                est_marker_step = max(1, len(est_t) // 14)
                ax.plot(
                    est_t,
                    est_y,
                    linewidth=line_weight,
                    color=color,
                    linestyle="-",
                    marker=marker,
                    markevery=est_marker_step,
                    markersize=max(2.5, marker_size - 0.6),
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    label="_nolegend_",
                )
        y_min, y_max = _curve_y_limits(model)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(1e-1, 1e6)
        ax.set_xscale("log")
        ax.set_xlabel(r"$t$")
        if idx % ncols == 0:
            ax.set_ylabel("Effective IDW")
        ax.set_title(str(model["name"]))
        ax.grid(True, which="both", alpha=0.3)
        if bool(plot_cfg.get("legend", True)):
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    linestyle="-",
                    color="black",
                    linewidth=line_weight,
                    label="Estimated",
                ),
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color="black",
                    linewidth=line_weight,
                    label="Approximated",
                ),
            ]
            marker_handles = [
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=marker_by_i[i_val],
                    color=color_by_i[i_val],
                    markerfacecolor=color_by_i[i_val],
                    markeredgecolor=color_by_i[i_val],
                    markeredgewidth=1.0,
                    markersize=marker_size + 0.6,
                    label=alpha_label_by_i[i_val],
                )
                for i_val in alpha_order
            ]
            ax.legend(
                handles=legend_handles + marker_handles,
                fontsize=float(plot_cfg.get("legend_fontsize", 9)),
                handlelength=2.4,
            )

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    print(f"Saved plot: {save_path}")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_plot_from_config_dict(
    config: dict[str, Any],
    base_dir: Path,
    save_override: Path | None = None,
    dpi_override: int | None = None,
    no_show_override: bool | None = None,
    config_path_for_hint: Path | None = None,
    curves_dir: Path | None = None,
    run_missing_sim: Callable[[Path, Path], None] | None = None,
) -> Path:
    models_cfg = config.get("models")
    if not isinstance(models_cfg, list) or not models_cfg:
        raise ValueError("config.models must be a non-empty list")

    t_min, t_max, n_t = _extract_time_grid(config)
    t_values = make_log_grid(t_min, t_max, n_t)
    indices, alpha_base = _extract_alpha(config)

    plot_cfg = config.get("plot", {})
    if not isinstance(plot_cfg, dict):
        raise ValueError("plot must be an object")
    curve_label_template = str(plot_cfg.get("curve_label_template", "i={i}, alpha=2^-{i}"))

    output_cfg = config.get("output", {})
    if not isinstance(output_cfg, dict):
        raise ValueError("output must be an object")
    default_output = Path("results/idw_effective_from_config.pdf")
    output_raw = output_cfg.get("path", default_output)

    save_path = save_override if save_override is not None else _resolve_path(output_raw, base_dir)
    show = bool(plot_cfg.get("show", True))
    if no_show_override is not None:
        show = not no_show_override
    dpi = int(plot_cfg.get("dpi", 180))
    if dpi_override is not None:
        dpi = dpi_override

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Simulation curve CSVs are looked up in curves_dir unless the config's
    # simulation_overlay.results_dir overrides it (old default: alongside
    # the output figure).
    default_curves_dir = Path(curves_dir) if curves_dir is not None else save_path.parent
    overlay_cfg = _extract_overlay_config(
        cfg=config,
        base_dir=base_dir,
        default_results_dir=default_curves_dir,
    )

    # If overlay CSVs are missing, let the caller (CLI/reproduce layer)
    # generate them before we plot.  rqab.plotting never runs simulations.
    if overlay_cfg["enabled"] and run_missing_sim is not None and config_path_for_hint is not None:
        any_missing = any(
            not _simulation_curve_path(
                results_dir=overlay_cfg["results_dir"],
                model_index=model_idx,
                model_name=str(model_cfg.get("name", "model")),
                alpha_index=i,
            ).exists()
            for model_idx, model_cfg in enumerate(models_cfg)
            for i in indices
        )
        if any_missing:
            run_missing_sim(config_path_for_hint, overlay_cfg["results_dir"])

    table_cache: dict[Path, WTableInterpolator] = {}
    missing_overlay_messages: list[str] = []
    model_results = [
        _build_model_curves(
            model_cfg=model_cfg,
            model_index=model_idx,
            t_values=t_values,
            indices=indices,
            alpha_base=alpha_base,
            table_cache=table_cache,
            base_dir=base_dir,
            global_curve_label_template=curve_label_template,
            overlay_cfg=overlay_cfg,
            missing_overlay_messages=missing_overlay_messages,
        )
        for model_idx, model_cfg in enumerate(models_cfg)
    ]

    _plot_models(
        save_path=save_path,
        t_values=t_values,
        models=model_results,
        plot_cfg=plot_cfg,
        dpi=dpi,
        show=show,
    )

    if overlay_cfg["enabled"] and overlay_cfg["warn_missing"] and missing_overlay_messages:
        print(
            "warning: missing estimated effective-IDW data for one or more curves; "
            "only approximations were plotted for those cases.",
            file=sys.stderr,
        )
        for msg in missing_overlay_messages:
            print(f"  - {msg}", file=sys.stderr)

        config_hint = (
            str(config_path_for_hint)
            if config_path_for_hint is not None
            else "<path/to/config.json>"
        )
        out_dir_hint = str(overlay_cfg["results_dir"])
        print("hint: generate simulation data with:", file=sys.stderr)
        print(
            f"  wck_effective_idw_sim --config \"{config_hint}\" --out-dir \"{out_dir_hint}\"",
            file=sys.stderr,
        )

    return save_path


def run_plot_from_config_path(
    config_path: Path,
    save_override: Path | None = None,
    dpi_override: int | None = None,
    no_show_override: bool | None = None,
    curves_dir: Path | None = None,
    run_missing_sim: Callable[[Path, Path], None] | None = None,
) -> Path:
    config = _load_json(config_path)
    return run_plot_from_config_dict(
        config=config,
        base_dir=config_path.resolve().parent,
        save_override=save_override,
        dpi_override=dpi_override,
        no_show_override=no_show_override,
        config_path_for_hint=config_path.resolve(),
        curves_dir=curves_dir,
        run_missing_sim=run_missing_sim,
    )


def plot_idw_effective(
    config_path: Path,
    curves_dir: Path | None = None,
    out_pdf: Path | None = None,
    no_show: bool = True,
    run_missing_sim: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Plot the effective-IDW figure for one configs/effective_idw_*.json.

    ``curves_dir`` is the directory holding the simulated
    model{idx}_*_idx{i}_curve.csv overlays (default: next to the output
    figure, i.e. results/).  ``out_pdf`` defaults to the config's
    ``output.path`` resolved relative to the config file's directory,
    matching the old plotter's naming (results/idw_effective_<alias>.pdf).
    ``run_missing_sim(config_path, curves_dir)`` is an optional hook the
    CLI layer can use to generate missing overlay CSVs; rqab.plotting
    itself never runs simulations.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    return run_plot_from_config_path(
        config_path=config_path,
        save_override=Path(out_pdf) if out_pdf is not None else None,
        no_show_override=no_show,
        curves_dir=Path(curves_dir) if curves_dir is not None else None,
        run_missing_sim=run_missing_sim,
    )
