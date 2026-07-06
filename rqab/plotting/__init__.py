"""rqab.plotting: figure generation for the RQ_ab paper.

Pure plotting layer: existing CSVs/tables in, figures out.  Simulation and
grid stages are run by the CLI/reproduce layer, never from here.

Modules
-------
style        shared rcParams / colormap / norm (ratio heatmaps)
ratio_panels 1/2/3-panel signed-relative-error heatmaps vs workload MC
idw_curves   effective-IDW simulated-vs-approximation curve figures
diagnostics  w-table overlay/tripanel, b(c) calibration and single w-curve figures
"""
from .diagnostics import (
    plot_b_calibration,
    plot_b_overlay,
    plot_w_curves,
    plot_w_overlay,
    plot_w_tripanel,
)
from .idw_curves import (
    plot_idw_effective,
    run_plot_from_config_dict,
    run_plot_from_config_path,
)
from .ratio_panels import (
    build_c_grid,
    build_panel_grid,
    figure_c_heatmap,
    figure_ratio,
    figure_tripanel,
    figure_twopanel,
    load_combined_secondary_rows,
    load_refined_c_rows,
    load_workload_rows,
    load_z_rows,
    render_ratio_panels,
)
from .style import DIVERGING_CMAP, SINGLE_PANEL_CMAP, make_norm, setup_rcparams

__all__ = [
    "DIVERGING_CMAP",
    "SINGLE_PANEL_CMAP",
    "build_c_grid",
    "build_panel_grid",
    "figure_c_heatmap",
    "figure_ratio",
    "figure_tripanel",
    "figure_twopanel",
    "load_combined_secondary_rows",
    "load_refined_c_rows",
    "load_workload_rows",
    "load_z_rows",
    "make_norm",
    "plot_b_calibration",
    "plot_b_overlay",
    "plot_idw_effective",
    "plot_w_curves",
    "plot_w_overlay",
    "plot_w_tripanel",
    "render_ratio_panels",
    "run_plot_from_config_dict",
    "run_plot_from_config_path",
    "setup_rcparams",
]
