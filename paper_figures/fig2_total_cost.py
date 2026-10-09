import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import style as S
import data as D

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
VOLL = "Med"                 # representative operating point (HVAC $3, Critical $100 /kWh)
ARCHS = ["PVB", "PCM"]       # renewable architectures only
SCALE = 1000.0               # plot in thousand USD/yr for legible tick labels

# Three independent encodings:
#   method -> colour | architecture -> hatch | cost component -> opacity
CAPITAL_ALPHA = 1.0          # capital segment: full-opacity method colour
PENALTY_ALPHA = 0.45         # VoLL-penalty segment: translucent method colour
EDGE_LW = 0.6
matplotlib.rcParams["hatch.linewidth"] = 0.6   # thin, crisp hatch strokes
matplotlib.rcParams["hatch.color"] = "black"


def rgba(color, alpha):
    """Face colour with alpha baked in, so the black edge/hatch stays opaque."""
    return mcolors.to_rgba(color, alpha)


# --------------------------------------------------------------------------- #
# Data: out-of-sample (test), Med VoLL, mean + across-fold SD
# --------------------------------------------------------------------------- #
folds = D.load_folds(architectures=tuple(ARCHS))
t = folds[(folds.split == "test") & (folds.voll == VOLL)].copy()

agg = D.agg_folds(
    t, ["total_cost", "capital_cost", "hvac_penalty", "critical_penalty"],
    by=["architecture", "location", "method"])
agg["penalty_mean"] = agg["hvac_penalty_mean"] + agg["critical_penalty_mean"]

# verify the stack sums to total (capital + VoLL penalty == total)
resid = (agg["capital_cost_mean"] + agg["penalty_mean"] - agg["total_cost_mean"]).abs().max()
assert resid < 1e-2, f"stack does not sum to total (max resid {resid})"

def get(arch, loc, method, col):
    m = ((agg.architecture == arch) & (agg.location == loc) & (agg.method == method))
    return float(agg.loc[m, col].iloc[0])

# --------------------------------------------------------------------------- #
# Figure: one facet per climate (cold -> hot), grouped stacked bars
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, len(S.LOCATION_ORDER), figsize=S.figsize_double(height=3.5),
                         sharey=False)

gx = np.arange(len(S.METHOD_ORDER))    # method group centres
offset, width = 0.205, 0.38
arch_dx = {"PVB": -offset, "PCM": +offset}

def kfmt(v, _pos):
    if v == 0:
        return "0"
    if abs(v) >= 1:
        return f"{v:g}"
    return f"{v:g}"

for ax, loc in zip(axes, S.LOCATION_ORDER):
    ymax = 0.0
    for mi, method in enumerate(S.METHOD_ORDER):
        base_c = S.METHOD_COLOR[method]
        for arch in ARCHS:
            x = gx[mi] + arch_dx[arch]
            cap = get(arch, loc, method, "capital_cost_mean") / SCALE
            pen = get(arch, loc, method, "penalty_mean") / SCALE
            tot = get(arch, loc, method, "total_cost_mean") / SCALE
            sd = get(arch, loc, method, "total_cost_std") / SCALE
            hatch = S.ARCH_HATCH[arch]
            # capital (base)
            ax.bar(x, cap, width, bottom=0.0, facecolor=rgba(base_c, CAPITAL_ALPHA),
                   edgecolor="black", linewidth=EDGE_LW, hatch=hatch, zorder=2)
            # VoLL penalty (translucent version of the same method colour), stacked
            # on top; an opaque white underlay keeps gridlines from showing through
            ax.bar(x, pen, width, bottom=cap, facecolor="white", edgecolor="none",
                   linewidth=0, zorder=1.9)
            ax.bar(x, pen, width, bottom=cap, facecolor=rgba(base_c, PENALTY_ALPHA),
                   edgecolor="black", linewidth=EDGE_LW, hatch=hatch, zorder=2)
            # across-fold SD on the TOTAL height
            ax.errorbar(x, tot, yerr=sd, fmt="none", ecolor="black",
                        elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=4)
            ymax = max(ymax, tot + sd)

    ax.set_title(S.CLIMATE_LABEL[loc], pad=4)
    ax.set_xticks([])
    ax.set_xlim(-0.72, len(gx) - 1 + 0.72)
    ax.set_ylim(0, ymax * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    S.despine(ax)
    S.ygrid(ax)

axes[0].set_ylabel("Annual total system cost  (thousand USD/yr)")

# --------------------------------------------------------------------------- #
# Legend: three titled groups across the top —
#   Method (colour) | Architecture (hatch) | Cost component (opacity)
# Architecture / component swatches use neutral greys so they carry only their
# own channel (hatch or opacity), never a method colour.
# --------------------------------------------------------------------------- #
ARCH_NEUTRAL = "0.82"        # light grey: solid vs hatched
COMP_NEUTRAL = "0.30"        # dark grey: full vs reduced opacity

def patch(face, label, hatch=""):
    return mpatches.Patch(facecolor=face, edgecolor="black", linewidth=EDGE_LW,
                          hatch=hatch, label=label)

# Positional cues in the labels mirror the layout: architecture entries run
# left->right like the bar pair; cost entries stack top->bottom like the bar.
leg_groups = [
    ("Method", [patch(S.METHOD_COLOR[m], S.METHOD_LABEL[m]) for m in S.METHOD_ORDER],
     3, 0.21),
    ("Architecture", [patch(ARCH_NEUTRAL, "PV+battery (left)", S.ARCH_HATCH["PVB"]),
                      patch(ARCH_NEUTRAL, "With PCM (right)", S.ARCH_HATCH["PCM"])],
     2, 0.56),
    ("Cost component", [patch(rgba(COMP_NEUTRAL, PENALTY_ALPHA), "VoLL penalty (top)"),
                        patch(rgba(COMP_NEUTRAL, CAPITAL_ALPHA), "Capital (bottom)")],
     1, 0.87),
]
for title, hs, ncol, xc in leg_groups:
    leg = fig.legend(handles=hs, title=title, loc="upper center",
                     bbox_to_anchor=(xc, 1.0), ncol=ncol, columnspacing=1.1,
                     handlelength=1.6, handleheight=1.0, handletextpad=0.5,
                     labelspacing=0.3,
                     title_fontproperties={"weight": "bold", "size": 8})
    leg._legend_box.align = "center"
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))

# --------------------------------------------------------------------------- #
# Printed stats: %Δ mean out-of-sample total cost, SO-CVaR vs LP-Avg / LP-Worst
# --------------------------------------------------------------------------- #
rows = []
for arch in ARCHS:
    for loc in S.LOCATION_ORDER:
        so = get(arch, loc, "SO_CVaR", "total_cost_mean")
        la = get(arch, loc, "LP_Avg", "total_cost_mean")
        lw = get(arch, loc, "LP_Worst", "total_cost_mean")
        rows.append({
            "architecture": arch,
            "climate": S.CLIMATE_LABEL[loc],
            "SO_CVaR_total": round(so, 1),
            "LP_Avg_total": round(la, 1),
            "LP_Worst_total": round(lw, 1),
            "pct_vs_LP_Avg": round((so - la) / la * 100, 2),
            "pct_vs_LP_Worst": round((so - lw) / lw * 100, 2),
        })
pct = pd.DataFrame(rows)

print("\n%Δ mean out-of-sample total cost: SO-CVaR vs LP-Avg / LP-Worst (test, Med VoLL)")
hdr = f"{'arch':4s} {'climate':16s} {'SO':>9s} {'LP-Avg':>9s} {'LP-Worst':>9s} {'vsAvg%':>8s} {'vsWorst%':>9s}"
print(hdr)
for _, r in pct.iterrows():
    print(f"{r.architecture:4s} {r.climate:16s} {r.SO_CVaR_total:9.0f} {r.LP_Avg_total:9.0f} "
          f"{r.LP_Worst_total:9.0f} {r.pct_vs_LP_Avg:+8.2f} {r.pct_vs_LP_Worst:+9.2f}")

# --------------------------------------------------------------------------- #
# Save (plotted values as CSV, draft caption as TXT)
# --------------------------------------------------------------------------- #
plotted = agg[["architecture", "location", "method",
               "capital_cost_mean", "penalty_mean", "hvac_penalty_mean",
               "critical_penalty_mean", "total_cost_mean", "total_cost_std",
               "total_cost_count"]].copy()
plotted = plotted.sort_values(["location", "method", "architecture"])

caption = (
    "Out-of-sample annual total system cost by climate for the two renewable "
    "architectures under the three sizing methods (color). Within each method, the "
    "left bar is PV+battery (solid) and the right bar adds phase-change material (PCM) "
    "thermal storage (hatched). Each bar is stacked into annualized capital cost "
    "(bottom, full color) and the value-of-lost-load (VoLL) penalty for unmet load "
    "(top, translucent shade of the same color). Bars show the mean over the 5 test "
    "years and 5 cross-validation folds; whiskers are +/-1 standard deviation of total "
    "cost across folds. VoLL is at the representative Med level (thermal $3/kWh, "
    "critical electrical $100/kWh). Note the independent per-climate y-axes "
    "(thousand USD/yr)."
)

S.save_fig(fig, "fig2_total_cost", section="main", data=plotted, caption=caption)

# also drop the pct-change table beside the figure for quoting
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main")
pct.to_csv(os.path.join(outdir, "fig2_total_cost_pct_change.csv"), index=False)
print("\nsaved fig2_total_cost + pct-change table")
