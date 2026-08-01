"""
fig4_risk.py — supporting risk result: relative to the average-year design (LP-Avg),
SO-CVaR reduces BOTH the reliability tail and the year-to-year cost variability.

Two SO-CVaR-vs-LP-Avg scatters (x = LP-Avg, y = SO-CVaR, dashed y=x line); a point
BELOW the line means SO-CVaR is better on that axis. All 30 cells (5 climates x 2
architectures x 3 VoLL); colour = VoLL level.
  (a) worst-fold out-of-sample unmet energy  = reliability TAIL  (coarse proxy: max of
      the 5 fold means; loss of load is ~all thermal)
  (b) across-fold SD of out-of-sample total cost = year-to-year COST STABILITY

Honest caveat (state in text): the tail reduction (a) is partly bought with ~1% more mean
capacity, so it is movement along the cost-reliability frontier; the UNCONFOUNDED win is the
cost-variance reduction (b), which holds in 30/30 cells and is not paid for by extra spend.

Run:  .\\.venv_verify\\Scripts\\python.exe paper_figures\\fig4_risk.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S
import data as D

# --- data: per (arch, location, method, voll) tail + cost-SD over the 5 test folds -----
folds = D.load_folds(architectures=("PCM", "PVB"))
folds = D.add_unmet_energy(folds)
te = folds[folds.split == "test"].copy()
KEYS = ["architecture", "location", "method", "voll"]
agg = (te.groupby(KEYS, observed=True)
       .agg(worst_unmet=("total_unmet_kwh", "max"),
            cost_sd=("total_cost", lambda x: float(np.std(x, ddof=1))))
       .reset_index())

wu = agg.pivot_table(index=["architecture", "location", "voll"], columns="method",
                     values="worst_unmet", observed=True)
cs = agg.pivot_table(index=["architecture", "location", "voll"], columns="method",
                     values="cost_sd", observed=True)

VOLL_COL = {"Low": "#9ecae1", "Med": "#4292c6", "High": "#08519c"}

# --- figure -------------------------------------------------------------------
fig, (axa, axb) = plt.subplots(1, 2, figsize=S.figsize_double(3.4))

def scatter_panel(ax, piv, lo, hi, axis_label, title):
    xs = np.array([lo, hi])
    ax.fill_between(xs, lo, xs, color=S.METHOD_COLOR["SO_CVaR"], alpha=0.06, zorder=0)  # below diag = SO-CVaR better
    ax.plot(xs, xs, ls="--", color="0.4", lw=1.1, zorder=2)
    nbetter = 0
    ntot = 0
    for idx, row in piv.iterrows():
        arch, loc, voll = idx
        x, y = row.get("LP_Avg"), row.get("SO_CVaR")
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ntot += 1
        nbetter += int(y < x)
        mk = "o" if arch == "PVB" else "D"
        ax.scatter([x], [y], s=34, marker=mk, facecolor=VOLL_COL.get(voll, "0.5"),
                   edgecolor="black", linewidth=0.5, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"LP-Avg {axis_label}")
    ax.set_ylabel(f"SO-CVaR {axis_label}")
    ax.set_title(title, loc="left")
    ax.text(0.96, 0.06, f"SO-CVaR better\n({nbetter}/{ntot} below line)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color=S.METHOD_COLOR["SO_CVaR"], style="italic")
    S.despine(ax)

scatter_panel(axa, wu, 8.0, 4000.0, "worst-fold unmet (kWh/yr, log)",
              "(a) Reliability tail")
scatter_panel(axb, cs, 4.0, 4000.0, "across-fold cost SD ($/yr, log)",
              "(b) Cost stability")

# legend: VoLL colour + architecture marker
voll_h = [Line2D([0], [0], marker="s", linestyle="none", markersize=7,
                 markerfacecolor=VOLL_COL[v], markeredgecolor="black",
                 markeredgewidth=0.4, label=f"{v} VoLL") for v in S.VOLL_ORDER]
arch_h = [Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                 markerfacecolor="0.6", markeredgecolor="black", markeredgewidth=0.4,
                 label="PV+battery"),
          Line2D([0], [0], marker="D", linestyle="none", markersize=7,
                 markerfacecolor="0.6", markeredgecolor="black", markeredgewidth=0.4,
                 label="With PCM")]
axa.legend(handles=voll_h + arch_h, loc="upper left", fontsize=6.0,
           handletextpad=0.3, labelspacing=0.3, borderaxespad=0.4)

fig.tight_layout(w_pad=1.8)

# --- stats + save -------------------------------------------------------------
def pct_reduction(piv):
    r = (piv["SO_CVaR"] / piv["LP_Avg"]).dropna()
    return 100 * (1 - r)   # positive = SO-CVaR reduction vs LP-Avg

wu_red = pct_reduction(wu)
cs_red = pct_reduction(cs)
out = agg.copy()
caption = (
    "SO-CVaR versus the average-year design (LP-Avg) on the out-of-sample test split. "
    "Each point is one climate x architecture x VoLL cell (30 total); x = LP-Avg, "
    "y = SO-CVaR, dashed line = equality, so points below favour SO-CVaR. Colour = "
    "VoLL level, marker = architecture. (a) Worst-fold unmet energy (a coarse tail "
    "proxy -- the largest of the 5 fold means; loss of load is essentially all "
    f"thermal): SO-CVaR is lower in {int((wu_red > 0).sum())}/{int(wu_red.notna().sum())} "
    f"cells (median {wu_red.median():.0f}% reduction). (b) Across-fold SD of total cost "
    f"(year-to-year cost stability): SO-CVaR is lower in {int((cs_red > 0).sum())}/"
    f"{int(cs_red.notna().sum())} cells (median {cs_red.median():.0f}% reduction). The "
    "tail gain in (a) is partly bought with ~1% more mean spend; the cost-stability "
    "gain in (b) is the cleaner, unconfounded result.")
S.save_fig(fig, "fig4_risk", section="main", data=out, caption=caption)

print("=== Risk reduction vs LP-Avg (test split, all 30 cells) ===")
print(f"Worst-fold unmet:  SO-CVaR lower in {int((wu_red>0).sum())}/{int(wu_red.notna().sum())}"
      f" cells; median reduction {wu_red.median():.1f}%")
print("  by VoLL (median reduction):")
for v in S.VOLL_ORDER:
    m = wu.index.get_level_values("voll") == v
    rr = pct_reduction(wu[m])
    print(f"    {v:>4}: {rr.median():.1f}%  ({int((rr>0).sum())}/{int(rr.notna().sum())} lower)")
print(f"Across-fold cost SD: SO-CVaR lower in {int((cs_red>0).sum())}/{int(cs_red.notna().sum())}"
      f" cells; median reduction {cs_red.median():.1f}%")
