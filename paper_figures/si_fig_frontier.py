import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import style as S; import data as D

VOLL = "Med"   # representative operating point (HVAC $3/kWh, Critical $100/kWh)

# --------------------------------------------------------------------------- #
#  Data: per-method x architecture point, mean over folds (test split)
# --------------------------------------------------------------------------- #
folds = D.load_folds(architectures=("PCM", "PVB"))
folds = D.add_unmet_energy(folds)
sub = folds[(folds.voll == VOLL) & (folds.split == "test")].copy()

agg = D.agg_folds(
    sub, ["total_unmet_kwh", "cost_ex_penalty"],
    by=["location", "architecture", "method"],
)


def pareto_mask(x, y):
    """Lower-left efficient set: minimise BOTH x (unmet energy) and y (cost).
    Point i is efficient if no other point is <= in both and < in at least one."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); eff = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i])):
                eff[i] = False
                break
    return eff


# flag Pareto-efficient points per climate
agg["pareto"] = False
for loc in S.LOCATION_ORDER:
    m = agg["location"] == loc
    d = agg[m]
    eff = pareto_mask(d["total_unmet_kwh_mean"].values, d["cost_ex_penalty_mean"].values)
    agg.loc[d.index[eff], "pareto"] = True

# --------------------------------------------------------------------------- #
#  Figure: 2 x 3 facets (5 climates + shared legend in the 6th cell)
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(S.DOUBLE_COL, 4.7), constrained_layout=True)
axes_flat = axes.flatten()


def kfmt(v, _):
    if abs(v) >= 1000:
        return f"{v/1000:g}k"
    return f"{v:g}"


printed = []
for ax, loc in zip(axes_flat[:5], S.LOCATION_ORDER):
    d = agg[agg["location"] == loc]

    # Pareto frontier line (light dashed, drawn under the points)
    dp = d[d["pareto"]].sort_values("total_unmet_kwh_mean")
    if len(dp) >= 2:
        ax.plot(dp["total_unmet_kwh_mean"], dp["cost_ex_penalty_mean"],
                linestyle="--", color="0.55", linewidth=1.0, zorder=1)

    # points: colour = method, marker = architecture (open PVB / filled PCM)
    for _, r in d.iterrows():
        ms = S.marker_style(r["method"], r["architecture"])
        x, y = r["total_unmet_kwh_mean"], r["cost_ex_penalty_mean"]
        xe, ye = r["total_unmet_kwh_std"], r["cost_ex_penalty_std"]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none",
                    ecolor=ms["color"], elinewidth=0.7, capsize=1.5,
                    alpha=0.45, zorder=2)
        ax.plot([x], [y], linestyle="none", markersize=6, zorder=3, **ms)

    ax.set_title(S.CLIMATE_LABEL[loc])
    S.despine(ax); S.ygrid(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.35, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.margins(0.13)

    # record Pareto set (ordered lower-left -> upper-left along frontier)
    order = dp[["method", "architecture", "total_unmet_kwh_mean", "cost_ex_penalty_mean"]]
    tags = [f"{S.METHOD_LABEL[m]}/{a}" for m, a in zip(order["method"], order["architecture"])]
    printed.append((loc, tags, order))

# shared axis titles
fig.supxlabel("Out-of-sample expected unmet energy (kWh/yr)  -  more reliable  <-",
              fontsize=9)
fig.supylabel("Investment + O&M cost, excl. VoLL penalty ($/yr)", fontsize=9)

# --------------------------------------------------------------------------- #
#  Legend in the 6th cell
# --------------------------------------------------------------------------- #
lax = axes_flat[5]
lax.axis("off")

method_handles = [
    Line2D([], [], marker="s", linestyle="none", markersize=8,
           markerfacecolor=S.METHOD_COLOR[m], markeredgecolor="black",
           markeredgewidth=0.5, label=S.METHOD_LABEL[m])
    for m in S.METHOD_ORDER
]
arch_handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=8,
           markerfacecolor="white", markeredgecolor="0.25", markeredgewidth=1.2,
           label=S.ARCH_LABEL["PVB"]),
    Line2D([], [], marker="o", linestyle="none", markersize=8,
           markerfacecolor="0.35", markeredgecolor="black", markeredgewidth=0.5,
           label=S.ARCH_LABEL["PCM"]),
]
frontier_handle = [Line2D([], [], linestyle="--", color="0.55", linewidth=1.0,
                          label="Pareto frontier")]

leg1 = lax.legend(handles=method_handles, title="Method",
                  loc="upper left", bbox_to_anchor=(0.0, 1.02),
                  handletextpad=0.4, borderaxespad=0.0, alignment="left")
leg1._legend_box.align = "left"
lax.add_artist(leg1)
leg2 = lax.legend(handles=arch_handles + frontier_handle, title="Architecture",
                  loc="lower left", bbox_to_anchor=(0.0, 0.0),
                  handletextpad=0.4, borderaxespad=0.0, alignment="left")
leg2._legend_box.align = "left"

# --------------------------------------------------------------------------- #
#  Printed stats
# --------------------------------------------------------------------------- #
print("=" * 72)
print(f"COST-RELIABILITY FRONTIER  |  VoLL={VOLL}  |  test split, mean over 5 folds")
print("x = total expected unmet energy (kWh/yr); y = cost excl. VoLL penalty ($/yr)")
print("=" * 72)
for loc, tags, order in printed:
    print(f"\n{loc} ({S.CLIMATE_LABEL[loc]}) Pareto-efficient (low-unmet -> low-cost):")
    for (_, r) in order.iterrows():
        print(f"   {S.METHOD_LABEL[r['method']]:9s}/{r['architecture']:3s}  "
              f"unmet={r['total_unmet_kwh_mean']:9.1f} kWh/yr  "
              f"cost={r['cost_ex_penalty_mean']:9.0f} $/yr")
    print(f"   frontier set: {', '.join(tags)}")

# SO-CVaR/PCM position note
print("\nSO-CVaR / PCM position:")
for loc in S.LOCATION_ORDER:
    r = agg[(agg.location == loc) & (agg.method == "SO_CVaR") & (agg.architecture == "PCM")].iloc[0]
    on = "ON frontier" if r["pareto"] else "dominated"
    print(f"   {loc:11s}: SO-CVaR/PCM {on}  "
          f"(unmet={r['total_unmet_kwh_mean']:7.1f} kWh/yr, cost={r['cost_ex_penalty_mean']:6.0f} $/yr)")

# --------------------------------------------------------------------------- #
#  Save
# --------------------------------------------------------------------------- #
out = agg.rename(columns={
    "total_unmet_kwh_mean": "unmet_kwh_yr_mean",
    "total_unmet_kwh_std": "unmet_kwh_yr_sd",
    "cost_ex_penalty_mean": "cost_ex_penalty_usd_yr_mean",
    "cost_ex_penalty_std": "cost_ex_penalty_usd_yr_sd",
})[["location", "architecture", "method",
    "unmet_kwh_yr_mean", "unmet_kwh_yr_sd",
    "cost_ex_penalty_usd_yr_mean", "cost_ex_penalty_usd_yr_sd", "pareto"]]

caption = (
    "Cost-reliability frontier by climate at the Med VoLL operating point "
    "(HVAC $3/kWh, Critical $100/kWh). Each point is one optimisation method "
    "(colour) x architecture (open = No PCM, PV+battery; filled = With PCM) on "
    "the out-of-sample test split, averaged over the 5 folds; error bars are "
    "+/-1 SD across folds. The x-axis is physical reliability = total expected "
    "annual unmet energy (thermal + electrical, kWh/yr; leftward = more "
    "reliable) recovered from the VoLL penalty terms; the y-axis is investment "
    "plus non-penalty operating cost only (cost_ex_penalty, $/yr), which "
    "excludes the VoLL penalty so reliability is not double-counted. The light "
    "dashed line links the Pareto-efficient (lower-left) points, i.e. the "
    "minimum cost achievable for a given reliability. Axes are per-facet "
    "because absolute scales differ ~100x from Polar (AK) to Tropical (FL). "
    "PCM configurations sweep the frontier in the temperate climates (MN, CA, "
    "AZ), while some worst-case PV+battery designs remain Pareto-efficient in "
    "Polar (AK) and Tropical (FL). SO-CVaR/PCM lies on the frontier in all five "
    "climates and, in Florida, dominates the worst-case LP-Worst/PCM design on "
    "both cost and reliability -- evidence that fitting a single worst year "
    "overfits. Within-climate differences are comparable to the +/-1 SD fold "
    "spread, so the frontier shows tradeoff geometry, not statistically "
    "separated points."
)

S.save_fig(fig, "si_fig_frontier", section="si", data=out, caption=caption)
