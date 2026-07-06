"""Single source of figure style for the ratio-heatmap plots.

Matches the styling of the old scripts/plot_approx_ratio_tripanel.py and
plot_approx_ratio_twopanel.py exactly (usetex detection, Computer Modern
fonts, diverging red-white-blue colormap, +/-30% TwoSlopeNorm shade cap).
"""
from __future__ import annotations

import math
import shutil
import sys

from matplotlib import colors

#: Shade values are hard-capped at +/-30 percent.
SHADE_CAP = 30.0

#: Diverging colormap used by the two/tri-panel ratio heatmaps:
#: under-estimation (red) -> accurate (white) -> over-estimation (blue).
DIVERGING_CMAP = colors.LinearSegmentedColormap.from_list(
    "under_white_over",
    ["#9f1121", "#ffffff", "#0e4c8a"],
    N=256,
)

#: Colormap used by the legacy single-panel refined_rq_ratio heatmap
#: (plot_refined_rq_ratio.py used slightly different endpoint colors).
SINGLE_PANEL_CMAP = colors.LinearSegmentedColormap.from_list(
    "under_white_over",
    ["#b2182b", "#ffffff", "#2166ac"],
    N=256,
)


def usetex_pdf_ok() -> bool:
    """True when usetex PDF output renders math glyphs correctly.

    matplotlib 3.11.0 introduced Type-1 font subsetting for usetex PDF
    output with a bug: glyphs addressed at char code 0 — notably the math
    minus sign, slot 0 of cmsy — lose their encoding entry, so every
    negative number and superscript minus shows as a blank in the PDF
    (PNG output is unaffected).  This probes the installed matplotlib's
    actual subsetting behavior rather than version-matching, so it heals
    automatically once upstream fixes it.
    """
    try:
        from matplotlib import _type1font, dviread

        if not hasattr(_type1font.Type1Font, "subset"):
            return True  # pre-3.11 whole-font embedding: not affected
        path = dviread.find_tex_file("cmsy10.pfb")
        if not path:
            return True
        subset = _type1font.Type1Font(path).subset({0x2212}, "PROBE+")
        return subset.prop["Encoding"].get(0) == "minus"
    except Exception:
        # Private APIs moved: assume the (reworked) pipeline is healthy.
        return True


def setup_rcparams() -> bool:
    """Configure matplotlib rcParams for paper figures.

    Uses LaTeX text rendering when a ``latex`` executable is available
    and the installed matplotlib embeds TeX fonts correctly (see
    usetex_pdf_ok), otherwise falls back to mathtext Computer Modern
    (with the same warning the old scripts printed).  Returns the
    resulting usetex flag.
    """
    import matplotlib.pyplot as plt

    use_tex = True
    if shutil.which("latex") is None:
        use_tex = False
        print(
            "warning: latex executable not found; using mathtext Computer Modern fallback.",
            file=sys.stderr,
        )
    elif not usetex_pdf_ok():
        use_tex = False
        print(
            "warning: this matplotlib drops minus signs from usetex PDF output "
            "(Type-1 subsetting bug, present in 3.11.0); "
            "using mathtext Computer Modern fallback.",
            file=sys.stderr,
        )
    plt.rcParams["text.usetex"] = use_tex
    plt.rcParams["font.family"] = "serif"
    if use_tex:
        plt.rcParams["font.serif"] = ["Computer Modern Roman"]
    else:
        plt.rcParams["mathtext.fontset"] = "cm"
    return use_tex


def make_norm(
    vmin_override: float | None = None,
    vmax_override: float | None = None,
) -> colors.TwoSlopeNorm:
    """TwoSlopeNorm centered at 0 with the +/-30 percent shade cap.

    Overrides are clipped into [-SHADE_CAP, +SHADE_CAP] exactly as in the
    old plotters; the range must straddle zero.
    """
    cmin = -SHADE_CAP if vmin_override is None else max(-SHADE_CAP, float(vmin_override))
    cmax = SHADE_CAP if vmax_override is None else min(SHADE_CAP, float(vmax_override))
    if not (math.isfinite(cmin) and math.isfinite(cmax) and cmax > cmin and cmin < 0.0 < cmax):
        raise ValueError("invalid color range; expected vmin < 0 < vmax")
    return colors.TwoSlopeNorm(vmin=cmin, vcenter=0.0, vmax=cmax)
