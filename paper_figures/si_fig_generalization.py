"""
si_fig_generalization — out-of-sample (train -> test) generalization motivation.

For each method x architecture, a paired dumbbell connects the mean training-year
value (grey tick) to the mean test-year value (coloured, arch-styled marker); the
line length/direction = the generalization gap. Two metric rows:
  * total annualized system cost (total_cost)
  * expected annual unmet energy (total_unmet_kwh, recovered from VoLL penalties)
Faceted by climate (cold -> hot), each facet auto-scaled (climates differ ~10x),
so real magnitudes and honest across-fold error bars are preserved. Message:
SO-CVaR's out-of-sample RELIABILITY gap is the smallest; LP-Avg overfits (unmet
energy rises out of sample) and LP-Worst swings the most (over-conservative).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import style as S; import data as D

# --------------------------------------------------------------------------- #
#  Data: Med VoLL, renewable architectures (PVB, PCM), both split values.
# --------------------------------------------------------------------------- #
VOLL = "Med"
METRICS = [
    ("total_cost",      "Total annualized system cost ($/yr)"),
    ("total_unmet_kwh", "Expected annual unmet energy (kWh/yr)"),
]

folds = D.load_folds(architectures=("PCM", "PVB"))
folds = D.add_unmet_energy(folds)
folds = folds[folds["voll"] == VOLL].copy()

KEYS = ["architecture", "location", "method"]


def cell_table(df, metric):
    """Per (arch, location, method): fold-mean train & test, across-fold SD of
    each, plus normalized/absolute train->test gaps."""
    tr = (df[df.split == "train"][KEYS + ["fold", metric]]
          .rename(columns={metric: "train"}))
    te = (df[df.split == "test"][KEYS + ["fold", metric]]
          .rename(columns={metric: "test"}))
    m = tr.merge(te, on=KEYS + ["fold"])
    g = m.groupby(KEYS, observed=True)
    out = pd.DataFrame({
        "train_mean": g["train"].mean(),  "train_sd": g["train"].std(ddof=1),
        "test_mean":  g["test"].mean(),    "test_sd":  g["test"].std(ddof=1),
    }).reset_index()
    out["gap_abs"] = out["test_mean"] - out["train_mean"]
    out["deg_pct"] = out["gap_abs"] / out["train_mean"] * 100.0
    out["metric"]  = metric
    return out


tables = {metric: cell_table(folds, metric) for metric, _ in METRICS}

# --------------------------------------------------------------------------- #
#  Printed statistics: pooled (arch x climate) mean degradation per method.
# --------------------------------------------------------------------------- #
def pooled(metric):
    return (tables[metric].groupby("method", observed=True)
            .agg(deg_pct=("deg_pct", "mean"), gap_abs=("gap_abs", "mean"))
            .reindex(S.METHOD_ORDER))


print("=== Generalization degradation, Med VoLL, pooled over 2 arch x 5 climates ===")
for metric, _ in METRICS:
    p = pooled(metric)
    print(f"\n[{metric}]  mean (test-train)/train per method:")
    for meth in S.METHOD_ORDER:
        print(f"  {S.METHOD_LABEL[meth]:9s}  {p.loc[meth,'deg_pct']:+6.2f}%   "
              f"(mean abs gap {p.loc[meth,'gap_abs']:+9.2f})")

cost_p, unmet_p = pooled("total_cost"), pooled("total_unmet_kwh")
print(f"\nSmallest |cost gap| : {S.METHOD_LABEL[cost_p['deg_pct'].abs().idxmin()]}")
print(f"Smallest |unmet gap|: {S.METHOD_LABEL[unmet_p['deg_pct'].abs().idxmin()]}"
      f"  -> SO-CVaR degrades least in reliability")

# --------------------------------------------------------------------------- #
#  Figure: 2 metric rows x 5 climate columns of absolute-value dumbbells.
# --------------------------------------------------------------------------- #
GROUP_DY, ARCH_DY = 1.4, 0.30                       # vertical layout of the 6 rows
ypos, yctr = {}, {}
for i, meth in enumerate(S.METHOD_ORDER):
    c = -i * GROUP_DY
    yctr[meth] = c
    ypos[(meth, "PVB")] = c + ARCH_DY
    ypos[(meth, "PCM")] = c - ARCH_DY


def kfmt(v, _):
    if abs(v) >= 1000:
        return f"{v/1000:g}k"
    return f"{v:g}"


fig, axes = plt.subplots(2, len(S.LOCATION_ORDER),
                         figsize=(S.DOUBLE_COL, 4.8), sharey=True)

for r, (metric, unit_lab) in enumerate(METRICS):
    t = tables[metric]
    for c, loc in enumerate(S.LOCATION_ORDER):
        ax = axes[r, c]
        sub = t[t.location == loc]
        for meth in S.METHOD_ORDER:
            for arch in S.ARCH_ORDER:
                row = sub[(sub.method == meth) & (sub.architecture == arch)]
                if row.empty:
                    continue
                y = ypos[(meth, arch)]
                tr_v = float(row["train_mean"].iloc[0])
                te_v = float(row["test_mean"].iloc[0])
                sd = float(row["test_sd"].iloc[0])
                col = S.method_color(meth)
                # connecting line (train -> test)
                ax.plot([tr_v, te_v], [y, y], color=col, lw=1.4, zorder=2,
                        solid_capstyle="round")
                # across-fold dispersion of the test value (clipped >= 0)
                low = min(sd, te_v)
                ax.errorbar(te_v, y, xerr=[[low], [sd]], fmt="none", ecolor=col,
                            elinewidth=0.9, capsize=1.8, zorder=3, alpha=0.9)
                # train marker (grey tick) + test marker (arch style)
                ax.plot([tr_v], [y], marker="|", color="0.4", ms=8,
                        markeredgewidth=1.4, zorder=4)
                ax.plot([te_v], [y], linestyle="none", markersize=6, zorder=5,
                        **S.marker_style(meth, arch))

        ax.set_ylim(yctr[S.METHOD_ORDER[-1]] - 0.72, yctr[S.METHOD_ORDER[0]] + 0.72)
        ax.margins(x=0.16)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", ls="--", alpha=0.30, lw=0.5)
        ax.set_axisbelow(True)
        S.despine(ax, left=False)
        ax.tick_params(axis="y", length=0)
        if r == 0:
            ax.set_title(S.CLIMATE_LABEL[loc], pad=4)
        if c == 0:                                   # method labels once (shared y)
            ax.set_yticks([yctr[m] for m in S.METHOD_ORDER])
            ax.set_yticklabels([S.METHOD_LABEL[m] for m in S.METHOD_ORDER])
            for lab, m in zip(ax.get_yticklabels(), S.METHOD_ORDER):
                lab.set_color(S.method_color(m))
                lab.set_fontweight("bold" if m == "SO_CVaR" else "normal")
    axes[r, 2].set_xlabel(unit_lab)                  # units, centred per row

# legend: train / test ends + architecture (marker fill)
handles = [
    Line2D([0], [0], marker="|", color="0.4", markeredgewidth=1.4, ms=9,
           linestyle="none", label="Training-year mean"),
    Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black",
           markeredgewidth=1.2, ms=6, linestyle="none",
           label="Test-year - No PCM (PV+battery)"),
    Line2D([0], [0], marker="o", markerfacecolor="black", markeredgecolor="black",
           markeredgewidth=0.5, ms=6, linestyle="none", label="Test-year - With PCM"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.55, 1.03), handletextpad=0.4, columnspacing=1.4)

fig.subplots_adjust(left=0.115, right=0.99, top=0.86, bottom=0.10,
                    wspace=0.30, hspace=0.42)

# --------------------------------------------------------------------------- #
#  Save (tidy value table + caption)
# --------------------------------------------------------------------------- #
out = pd.concat(tables.values(), ignore_index=True)[
    ["metric", "architecture", "location", "method",
     "train_mean", "train_sd", "test_mean", "test_sd", "gap_abs", "deg_pct"]]

caption = (
    "Out-of-sample generalization of the three planning methods (Med VoLL: HVAC "
    "$3/kWh, Critical $100/kWh). Each dumbbell connects the mean training-year "
    "value (grey tick) to the mean test-year value (coloured marker) for one "
    "method x architecture; the gap is the train->test generalization "
    "degradation. Top row: total annualized system cost; bottom row: expected "
    "annual unmet energy (essentially all thermal; critical/electrical unmet is "
    "~0). Facets are climates (cold->hot), each auto-scaled because climate costs "
    "and loads differ ~10x; test-year values are primary. Error bars are +/-1 SD "
    "of the test value across the five cross-validation folds (lower whisker "
    "clipped at zero for unmet energy). Architecture is read from the marker "
    "(open = PV+battery, filled = with PCM); method from colour/label. Pooled "
    "over both architectures and the five climates, SO-CVaR has the smallest "
    f"reliability gap (unmet energy {unmet_p.loc['SO_CVaR','deg_pct']:+.1f}%) "
    f"versus LP-Avg ({unmet_p.loc['LP_Avg','deg_pct']:+.1f}%, overfitting) and "
    f"LP-Worst ({unmet_p.loc['LP_Worst','deg_pct']:+.1f}%, over-conservative); "
    "LP-Avg is the only method whose cost also rises out of sample "
    f"({cost_p.loc['LP_Avg','deg_pct']:+.1f}%). LP-Worst attains the lowest test "
    "unmet in every climate; SO-CVaR's advantage is the smallest train->test gap "
    "(reliability delivered as promised) at near-cheapest cost -- the knee of the "
    "tradeoff, not the lowest reliability level.")

S.save_fig(fig, "si_fig_generalization", section="si", data=out, caption=caption)
print("\nrows in value table:", len(out))
