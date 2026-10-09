"""
style.py — shared Matplotlib style + save helpers for the FOB paper figures.

Encodes the fixed publication conventions so every figure reads as one system:
  * method -> colour (Okabe-Ito, colourblind-safe), fixed order
  * architecture -> fill/marker (hue kept; only fill/marker vary, survives grayscale)
  * climate -> fixed facet order + display labels
  * sans-serif typography sized for print; single-/double-column figure widths
  * save_fig() -> vector PDF (primary) + 600-dpi PNG with embedded fonts, and
    (optionally) the plotted values as CSV + a draft caption, all into
    paper_figures/<section>/.

Import this in every figure script:  `import style as S; S.apply()`
"""

from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
#  Taxonomy (shared source of truth; data.py imports these)
# --------------------------------------------------------------------------- #
# Optimization methods (the cost/reliability trade-off comparison)
METHOD_ORDER = ["LP_Avg", "LP_Worst", "SO_CVaR"]
METHOD_LABEL = {"LP_Avg": "LP-Avg", "LP_Worst": "LP-Worst",
                "SO_CVaR": "SO-CVaR", "Diesel": "Diesel"}

# Palette — Okabe-Ito, fixed across ALL figures. SO-CVaR is the highlighted method.
METHOD_COLOR = {
    "LP_Avg":   "#0072B2",   # blue
    "LP_Worst": "#D55E00",   # vermillion
    "SO_CVaR":  "#009E73",   # green (highlight)
    "Diesel":   "#555555",   # dark grey (reference; drawn dashed)
}
DIESEL_LINE = dict(color=METHOD_COLOR["Diesel"], linestyle="--", linewidth=1.3)

# Architecture -> fill/marker. Hue always carries the method; fill/marker carry
# the architecture, so both channels read and figures survive grayscale.
#   no-PCM (PVB)  = solid fill  / open marker (o)
#   PCM           = hatched fill (///) / filled marker (●)
ARCH_ORDER = ["PVB", "PCM"]
ARCH_LABEL = {"PVB": "No PCM (PV+battery)", "PCM": "With PCM", "Diesel": "Diesel"}
ARCH_HATCH = {"PVB": "", "PCM": "///", "Diesel": ".."}

# Climate: fixed facet order cold -> hot, with display labels.
LOCATION_ORDER = ["Alaska", "Minnesota", "California", "Arizona", "Florida"]
CLIMATE_LABEL = {
    "Alaska":     "Polar (AK)",
    "Minnesota":  "Continental (MN)",
    "California": "Marine (CA)",
    "Arizona":    "Arid (AZ)",
    "Florida":    "Tropical (FL)",
}

VOLL_ORDER = ["Low", "Med", "High"]

# --------------------------------------------------------------------------- #
#  Figure sizes (print widths)
# --------------------------------------------------------------------------- #
SINGLE_COL = 3.5   # ~90 mm  (in)
DOUBLE_COL = 7.1   # ~180 mm (in)


def figsize_single(height=2.7):
    return (SINGLE_COL, height)


def figsize_double(height=3.2):
    return (DOUBLE_COL, height)


# --------------------------------------------------------------------------- #
#  rcParams
# --------------------------------------------------------------------------- #
def apply():
    """Install the paper style. Idempotent; call once per figure script."""
    mpl.rcParams.update({
        # typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,            # base
        "axes.labelsize": 9,       # axis titles
        "axes.titlesize": 10,      # panel titles
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "mathtext.default": "regular",
        # embedded, editable fonts in vector output
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # spare, print-friendly axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "patch.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


# --------------------------------------------------------------------------- #
#  Encoding helpers
# --------------------------------------------------------------------------- #
def method_color(method):
    return METHOD_COLOR.get(method, "#333333")


def climate_label(location):
    return CLIMATE_LABEL.get(location, location)


def bar_style(method, architecture="PCM"):
    """kwargs for ax.bar: method hue + architecture fill (hatch)."""
    return dict(color=method_color(method),
                edgecolor="black",
                linewidth=0.6,
                hatch=ARCH_HATCH.get(architecture, ""))


def marker_style(method, architecture="PCM"):
    """kwargs for ax.plot/scatter markers: open (no-PCM) vs filled (PCM)."""
    c = method_color(method)
    if architecture == "PVB":          # open marker
        return dict(marker="o", markerfacecolor="white",
                    markeredgecolor=c, markeredgewidth=1.2, color=c)
    if architecture == "Diesel":
        return dict(marker="^", markerfacecolor=c, markeredgecolor="black",
                    markeredgewidth=0.5, color=c)
    return dict(marker="o", markerfacecolor=c,   # filled marker (PCM)
                markeredgecolor="black", markeredgewidth=0.5, color=c)


def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def ygrid(ax):
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.5)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
#  Saving:  PDF + 600-dpi PNG, and optional CSV of values + draft caption
# --------------------------------------------------------------------------- #
def _section_dir(section):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), section)
    os.makedirs(d, exist_ok=True)
    return d


def save_fig(fig, name, section="main", data=None, caption=None,
             tight=True, verbose=True):
    """Write <name>.pdf and <name>.png (600 dpi, embedded fonts) into
    paper_figures/<section>/. If `data` (a DataFrame) is given, also write
    <name>.csv beside them; if `caption` is given, write <name>.txt.

    Returns the directory written to.
    """
    d = _section_dir(section)
    kw = dict(bbox_inches="tight") if tight else {}
    pdf = os.path.join(d, f"{name}.pdf")
    png = os.path.join(d, f"{name}.png")
    fig.savefig(pdf, **kw)
    fig.savefig(png, dpi=600, **kw)
    if data is not None:
        data.to_csv(os.path.join(d, f"{name}.csv"), index=False)
    if caption is not None:
        with open(os.path.join(d, f"{name}.txt"), "w", encoding="utf-8") as fh:
            fh.write(caption.strip() + "\n")
    if verbose:
        extras = ("+csv" if data is not None else "") + ("+caption" if caption else "")
        print(f"saved {section}/{name}.pdf,.png {extras}".strip())
    return d


apply()  # install on import
