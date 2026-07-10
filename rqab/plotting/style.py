"""Shared styling for ratio heatmaps and effective-IDW figures."""
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

#: Colormap used by the single-panel refined-RQ ratio heatmap.
SINGLE_PANEL_CMAP = colors.LinearSegmentedColormap.from_list(
    "under_white_over",
    ["#b2182b", "#ffffff", "#2166ac"],
    N=256,
)


def usetex_pdf_ok() -> bool:
    """Return whether usetex PDF output preserves the math minus glyph."""
    try:
        from matplotlib import _type1font, dviread

        if not hasattr(_type1font.Type1Font, "subset"):
            return True
        path = dviread.find_tex_file("cmsy10.pfb")
        if not path:
            return True
        subset = _type1font.Type1Font(path).subset({0x2212}, "PROBE+")
        return subset.prop["Encoding"].get(0) == "minus"
    except Exception:
        # The probe uses private APIs; unknown implementations are allowed.
        return True


def setup_rcparams() -> bool:
    """Configure matplotlib rcParams for paper figures.

    Uses LaTeX text rendering when a ``latex`` executable is available
    and the installed matplotlib embeds TeX fonts correctly; otherwise it
    falls back to mathtext Computer Modern. Returns the resulting usetex flag.
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
            "warning: matplotlib cannot preserve minus signs in usetex PDF output; "
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

    Overrides are clipped into [-SHADE_CAP, +SHADE_CAP]; the range must
    straddle zero.
    """
    cmin = -SHADE_CAP if vmin_override is None else max(-SHADE_CAP, float(vmin_override))
    cmax = SHADE_CAP if vmax_override is None else min(SHADE_CAP, float(vmax_override))
    if not (math.isfinite(cmin) and math.isfinite(cmax) and cmax > cmin and cmin < 0.0 < cmax):
        raise ValueError("invalid color range; expected vmin < 0 < vmax")
    return colors.TwoSlopeNorm(vmin=cmin, vcenter=0.0, vmax=cmax)
