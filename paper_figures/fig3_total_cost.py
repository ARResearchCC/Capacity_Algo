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


def lighten(color, f):
    """Mix `color` a fraction f toward white (0=color, 1=white)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


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
        pen_c = lighten(base_c, 0.58)
        for arch in ARCHS:
            x = gx[mi] + arch_dx[arch]
            cap = get(arch, loc, method, "capital_cost_mean") / SCALE
            pen = get(arch, loc, method, "penalty_mean") / SCALE
            tot = get(arch, loc, method, "total_cost_mean") / SCALE
            sd = get(arch, loc, method, "total_cost_std") / SCALE
            hatch = S.ARCH_HATCH[arch]
            # capital (base)
            ax.bar(x, cap, width, bottom=0.0, color=base_c, edgecolor="black",
                   linewidth=0.6, hatch=hatch, zorder=2)
            # VoLL penalty (lighter tint of same method colour), stacked on top
            ax.bar(x, pen, width, bottom=cap, color=pen_c, edgecolor="black",
                   linewidth=0.6, hatch=hatch, zorder=2)
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
# Legend: one top row — method (colour) | architecture (hatch) | component (shade)
# matches the house style used by the sibling reliability figure.
# --------------------------------------------------------------------------- #
handles = [mpatches.Patch(facecolor=S.METHOD_COLOR[m], edgecolor="black",
                          linewidth=0.6, label=S.METHOD_LABEL[m])
           for m in S.METHOD_ORDER]
handles += [
    mpatches.Patch(facecolor="0.72", edgecolor="black", linewidth=0.6, hatch="",
                   label="No PCM (PV+battery)"),
    mpatches.Patch(facecolor="0.72", edgecolor="black", linewidth=0.6, hatch="///",
                   label="With PCM"),
    mpatches.Patch(facecolor="0.38", edgecolor="black", linewidth=0.6,
                   label="Capital"),
    mpatches.Patch(facecolor=lighten("0.38", 0.58), edgecolor="black",
                   linewidth=0.6, label="VoLL penalty"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.93),
           ncol=7, columnspacing=1.2, handlelength=1.3, handletextpad=0.5)
fig.text(0.5, 0.90,
         "Within each climate: bars ordered LP-Avg | LP-Worst | SO-CVaR (colour); "
         "No-PCM then PCM (hatch); capital (dark) + VoLL penalty (light) = total; "
         "per-climate y-scales",
         ha="center", va="top", color="0.35", fontsize=7.5)

fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))

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
    "architectures (PV+battery, hatched: with phase-change thermal storage) under "
    "the three sizing methods (LP-Avg, LP-Worst, SO-CVaR; colour). Each bar is the "
    "mean over 5 test years and 5 cross-validation folds, decomposed into annualized "
    "capital (solid shade) and unmet-load VoLL penalty (light shade = HVAC/thermal + "
    "critical/electrical penalties); for these renewables non-penalty operating cost "
    "is ~0 so capital + VoLL penalty = total cost exactly. Thin bars show +/-1 SD across "
    "the 5 folds on the total. VoLL is at the representative Med level (HVAC $3/kWh, "
    "critical $100/kWh). Note independent y-axes per climate (thousand USD/yr); costs "
    "range from ~2k USD/yr in warm climates to ~18k in Alaska, and the penalty share "
    "grows sharply in cold climates."
)

S.save_fig(fig, "fig3_total_cost", section="main", data=plotted, caption=caption)

# also drop the pct-change table beside the figure for quoting
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main")
pct.to_csv(os.path.join(outdir, "fig3_total_cost_pct_change.csv"), index=False)
print("\nsaved fig3_total_cost + pct-change table")
