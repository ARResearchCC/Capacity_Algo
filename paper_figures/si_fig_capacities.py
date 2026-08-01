import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import style as S; import data as D

# --------------------------------------------------------------------------- #
# Data: capacities are split-invariant & VoLL-dependent -> Med VoLL, test split.
# Average across the 5 CV folds; ±1 SD across folds for error bars.
# --------------------------------------------------------------------------- #
folds = D.load_folds(architectures=("PCM", "PVB"))
df = folds[(folds.split == "test") & (folds.voll == "Med")].copy()

COMPS = ["pv_kw", "battery_kwh", "pcm_hot_kwh", "pcm_cold_kwh"]
agg = D.agg_folds(df, COMPS, by=["architecture", "location", "method"])

# lookup: (arch, location, method) -> row
LK = {(r.architecture, r.location, r.method): r for r in agg.itertuples(index=False)}

# component rows: (col, y-axis label, architectures present in that row)
ROWS = [
    ("pv_kw",       "PV (kW)",                              ["PVB", "PCM"]),
    ("battery_kwh", "Battery (kWh)",                        ["PVB", "PCM"]),
    ("pcm_hot_kwh", r"Hot PCM (kWh$_\mathrm{th}$)",         ["PCM"]),
    ("pcm_cold_kwh", r"Cold PCM (kWh$_\mathrm{th}$)",       ["PCM"]),
]
COLS = S.LOCATION_ORDER
XM = {m: i for i, m in enumerate(S.METHOD_ORDER)}   # method -> group centre

nrow, ncol = len(ROWS), len(COLS)
fig, axes = plt.subplots(nrow, ncol, figsize=S.figsize_double(6.9),
                         sharey="row")

for ri, (comp, ylab, arches) in enumerate(ROWS):
    # per-row y-limit incl. error bars, for a bit of headroom
    row_top = 0.0
    for ci, loc in enumerate(COLS):
        ax = axes[ri, ci]
        for m in S.METHOD_ORDER:
            for arch in arches:
                r = LK.get((arch, loc, m))
                if r is None:
                    continue
                mean = getattr(r, f"{comp}_mean")
                std = getattr(r, f"{comp}_std")
                if not np.isfinite(mean):
                    continue
                if len(arches) == 2:                     # PV / battery: two bars
                    off = -0.18 if arch == "PVB" else 0.18
                    w = 0.34
                else:                                    # PCM rows: single bar
                    off, w = 0.0, 0.50
                x = XM[m] + off
                ax.bar(x, mean, width=w,
                       yerr=(std if np.isfinite(std) else 0.0),
                       error_kw=dict(ecolor="black", elinewidth=0.7,
                                     capsize=1.8, capthick=0.7),
                       zorder=3, **S.bar_style(m, arch))
                row_top = max(row_top, mean + (std if np.isfinite(std) else 0.0))

        S.ygrid(ax); S.despine(ax)
        ax.set_xlim(-0.6, 2.6)
        ax.set_xticks(list(XM.values()))
        if ri == 0:
            ax.set_title(S.CLIMATE_LABEL[loc].replace(" (", "\n("))
        if ri == nrow - 1:
            ax.set_xticklabels([S.METHOD_LABEL[m] for m in S.METHOD_ORDER],
                               rotation=35, ha="right")
        else:
            ax.set_xticklabels([])
        if ci == 0:
            ax.set_ylabel(ylab)
    # apply headroom to the shared row (set on first ax; sharey propagates)
    axes[ri, 0].set_ylim(0, row_top * 1.16 if row_top > 0 else 1)

# --------------------------------------------------------------------------- #
# Legends: method (colour) + architecture (hatch), colourblind/grayscale safe.
# --------------------------------------------------------------------------- #
method_h = [Patch(facecolor=S.METHOD_COLOR[m], edgecolor="black", linewidth=0.6,
                  label=S.METHOD_LABEL[m]) for m in S.METHOD_ORDER]
arch_h = [Patch(facecolor="0.85", edgecolor="black", linewidth=0.6,
                hatch=S.ARCH_HATCH[a], label=S.ARCH_LABEL[a])
          for a in S.ARCH_ORDER]

leg1 = fig.legend(handles=method_h, title="Method",
                  loc="lower left", bbox_to_anchor=(0.045, 0.965),
                  ncol=3, handlelength=1.3, columnspacing=1.2,
                  handletextpad=0.5, borderaxespad=0.0)
fig.add_artist(leg1)
fig.legend(handles=arch_h, title="Architecture",
           loc="lower right", bbox_to_anchor=(0.995, 0.965),
           ncol=2, handlelength=1.6, columnspacing=1.2,
           handletextpad=0.5, borderaxespad=0.0)

fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=0.8, w_pad=0.6)

# --------------------------------------------------------------------------- #
# Saved data table (plotted means + SD) and caption.
# --------------------------------------------------------------------------- #
out = agg.rename(columns={"location": "location"}).copy()
caption = (
    "Sized component capacities selected by each optimization method "
    "(LP-Avg, LP-Worst, SO-CVaR) for the two renewable architectures "
    "(No PCM = PV+battery, solid; With PCM, hatched), faceted by climate "
    "(cold to hot, left to right). Rows are components: PV (kW), battery "
    "(kWh), hot PCM and cold PCM (kWh thermal); PCM rows are blank for the "
    "No-PCM architecture. Bars are means over the 5 cross-validation folds; "
    "error bars are +/-1 SD across folds. Capacities are split-invariant and "
    "shown at the representative Med VoLL level (HVAC $3/kWh, critical "
    "$100/kWh). Each component row has its own y-scale (shared across climates "
    "within a row); PCM shifts thermal duty into thermal storage, sharply "
    "reducing the battery sizing relative to the No-PCM design.")

S.save_fig(fig, "si_fig_capacities", section="si", data=out, caption=caption)

# --------------------------------------------------------------------------- #
# PRINTED STATS
# --------------------------------------------------------------------------- #
print("\n================  PRINTED STATS (Med VoLL)  ================")

# (1) per component: which method builds most / least (avg over climates+folds)
print("\n-- Per-component method that builds MOST / LEAST "
      "(mean over climates & folds) --")
comp_units = {"pv_kw": "kW", "battery_kwh": "kWh",
              "pcm_hot_kwh": "kWh_th", "pcm_cold_kwh": "kWh_th"}
comp_arch = {"pv_kw": ["PVB", "PCM"], "battery_kwh": ["PVB", "PCM"],
             "pcm_hot_kwh": ["PCM"], "pcm_cold_kwh": ["PCM"]}
for comp in COMPS:
    sub = df[df.architecture.isin(comp_arch[comp])]
    by_m = sub.groupby("method", observed=True)[comp].mean().reindex(S.METHOD_ORDER)
    mmax, mmin = by_m.idxmax(), by_m.idxmin()
    print(f"  {comp:12s} [{comp_units[comp]:6s}]  "
          f"MOST = {S.METHOD_LABEL[mmax]:8s} ({by_m[mmax]:6.1f})   "
          f"LEAST = {S.METHOD_LABEL[mmin]:8s} ({by_m[mmin]:6.1f})   "
          f"| all: " + ", ".join(f"{S.METHOD_LABEL[m]}={by_m[m]:.1f}"
                                 for m in S.METHOD_ORDER))

# (2) PCM vs PVB effect on mean PV and battery (avg over climates, folds, methods)
print("\n-- PCM vs No-PCM (PVB) effect on mean sizing "
      "(over climates, folds, methods) --")
for comp in ["pv_kw", "battery_kwh"]:
    mv = df.groupby("architecture", observed=True)[comp].mean()
    pvb, pcm = mv["PVB"], mv["PCM"]
    pct = (pvb / pcm - 1.0) * 100.0
    print(f"  {comp:12s}: PVB={pvb:6.2f}  PCM={pcm:6.2f}  "
          f"-> PVB is {pct:+.1f}% vs PCM "
          f"(PCM is {(1-pcm/pvb)*100:.1f}% smaller)")
print("===========================================================")
