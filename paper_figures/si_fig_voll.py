import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S; import data as D

# --------------------------------------------------------------------------- #
#  Data: VoLL sensitivity in a representative marine climate (California).
#  Pooling across climates would be dominated by Alaska's large loads, so we
#  fix one climate. Test split is primary; dispersion is +/-1 SD across folds.
# --------------------------------------------------------------------------- #
LOC = "California"
VAL = {"total_cost": "Total cost (USD/yr)",
       "total_unmet_kwh": "Total unmet load (kWh/yr)"}

df = D.load_folds(architectures=("PCM", "PVB"))
df = D.add_unmet_energy(df)
sub = df[(df.location == LOC) & (df.split == "test")].copy()

agg = D.agg_folds(sub, list(VAL.keys()),
                  by=["architecture", "voll", "method"])
agg["voll"] = pd.Categorical(agg["voll"], S.VOLL_ORDER, ordered=True)
agg["method"] = pd.Categorical(agg["method"], S.METHOD_ORDER, ordered=True)
agg["architecture"] = pd.Categorical(agg["architecture"], S.ARCH_ORDER, ordered=True)

x = np.arange(len(S.VOLL_ORDER))          # Low, Med, High -> 0,1,2
ARCH_LS = {"PVB": "--", "PCM": "-"}       # extra grayscale cue for architecture

# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=S.figsize_double(3.3))

for ax, (col, ylab) in zip(axes, VAL.items()):
    for arch in S.ARCH_ORDER:
        for method in S.METHOD_ORDER:
            r = agg[(agg.architecture == arch) & (agg.method == method)]
            r = r.sort_values("voll")
            y = r[f"{col}_mean"].to_numpy()
            sd = r[f"{col}_std"].to_numpy()
            c = S.method_color(method)
            ms = S.marker_style(method, arch)
            ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.10, linewidth=0)
            ax.plot(x, y, linestyle=ARCH_LS[arch], linewidth=1.4,
                    markersize=5, zorder=3, **ms)
    ax.set_xticks(x)
    ax.set_xticklabels(S.VOLL_ORDER)
    ax.set_xlabel("VoLL level")
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.25, 2.25)
    S.despine(ax); S.ygrid(ax)

axes[0].set_title("(a) Out-of-sample total cost", loc="left")
axes[1].set_title("(b) Total loss of load", loc="left")

# --------------------------------------------------------------------------- #
#  Legends: method (colour) + architecture (marker fill + line style)
# --------------------------------------------------------------------------- #
method_handles = [Line2D([0], [0], color=S.method_color(m), lw=1.8,
                         label=S.METHOD_LABEL[m]) for m in S.METHOD_ORDER]
arch_handles = [
    Line2D([0], [0], color="0.35", lw=1.4, linestyle=ARCH_LS["PVB"],
           marker="o", markerfacecolor="white", markeredgecolor="0.35",
           markeredgewidth=1.2, label="PV+battery (PVB)"),
    Line2D([0], [0], color="0.35", lw=1.4, linestyle=ARCH_LS["PCM"],
           marker="o", markerfacecolor="0.35", markeredgecolor="black",
           markeredgewidth=0.5, label="With PCM (PCM)"),
]

leg1 = fig.legend(handles=method_handles, title="Method",
                  loc="lower center", bbox_to_anchor=(0.30, -0.13),
                  ncol=3, handlelength=1.8, columnspacing=1.2)
leg1.get_title().set_fontsize(8)
leg2 = fig.legend(handles=arch_handles, title="Architecture",
                  loc="lower center", bbox_to_anchor=(0.78, -0.13),
                  ncol=2, handlelength=2.2, columnspacing=1.2)
leg2.get_title().set_fontsize(8)

fig.tight_layout(w_pad=2.0)
fig.subplots_adjust(bottom=0.22)

# --------------------------------------------------------------------------- #
#  Printed statistics: SO-CVaR total-cost gap vs LP-Avg / LP-Worst (PCM, CA).
#  gap = SO_CVaR - LP  (negative => SO-CVaR is cheaper).
# --------------------------------------------------------------------------- #
pcm = agg[agg.architecture == "PCM"].pivot_table(
    index="voll", columns="method", values="total_cost_mean", observed=True)
print(f"\nPCM, {LOC}, out-of-sample total cost (USD/yr):")
print(pcm.round(1).to_string())
print("\nSO-CVaR total-cost gap (USD/yr; negative = SO-CVaR cheaper):")
for lv in S.VOLL_ORDER:
    g_avg = pcm.loc[lv, "SO_CVaR"] - pcm.loc[lv, "LP_Avg"]
    g_wst = pcm.loc[lv, "SO_CVaR"] - pcm.loc[lv, "LP_Worst"]
    print(f"  {lv:>4}:  vs LP-Avg {g_avg:+7.1f}   vs LP-Worst {g_wst:+7.1f}")

# --------------------------------------------------------------------------- #
CAP = (
    "VoLL sensitivity of the FOB microgrid in a representative marine climate "
    "(California; pooling across climates is dominated by Alaska). Lines show "
    "fold-mean out-of-sample (test-split) results versus the value-of-lost-load "
    "level (Low: HVAC $1/Critical $30; Med: $3/$100; High: $10/$300 per kWh); "
    "colour = method, marker fill and line style = architecture (open/dashed = "
    "PV+battery, filled/solid = with PCM). (a) Annualized total cost (USD/yr); "
    "(b) total unmet load (kWh/yr). Bands are +/-1 SD across the 5 folds. Method "
    "cost differences are small and within the fold bands at Low and Med VoLL, "
    "separating only at High. With PCM at High VoLL, SO-CVaR is the cheapest "
    f"method ({pcm.loc['High','SO_CVaR']-pcm.loc['High','LP_Avg']:+.0f} $/yr vs "
    f"LP-Avg, {pcm.loc['High','SO_CVaR']-pcm.loc['High','LP_Worst']:+.0f} $/yr vs "
    "LP-Worst), while its unmet load tracks the robust LP-Worst (below LP-Avg, "
    "slightly above LP-Worst). Without PCM this cost crossover does not occur "
    "(SO-CVaR remains a small premium over LP-Avg)."
)
S.save_fig(fig, "si_fig_voll", section="si", data=agg, caption=CAP)
