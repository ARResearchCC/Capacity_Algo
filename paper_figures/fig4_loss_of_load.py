import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import style as S; import data as D

# --------------------------------------------------------------------------- #
# Data: per-fold test-split rows, Med VoLL, renewable architectures only.
# Unmet energy recovered from penalty/VoLL; worst-event durations are raw h.
# --------------------------------------------------------------------------- #
df = D.load_folds(architectures=("PCM", "PVB"))
df = D.add_unmet_energy(df)
df = df[(df.split == "test") & (df.voll == "Med")].copy()

VAL = ["electric_unmet_kwh", "thermal_unmet_kwh",
       "critical_worst_event_h", "hvac_worst_event_h"]
BY = ["location", "method", "architecture"]
agg = D.agg_folds(df, VAL, BY)

# ordered lookup
agg["location"] = pd.Categorical(agg["location"], S.LOCATION_ORDER, ordered=True)
LUT = {(r.location, r.method, r.architecture): r for r in agg.itertuples(index=False)}

def cell(loc, method, arch):
    return LUT.get((loc, method, arch))

# --------------------------------------------------------------------------- #
# Panel spec: (metric key, electrical col, thermal col, y-axis label)
# --------------------------------------------------------------------------- #
PANELS = [
    ("energy", "electric_unmet_kwh", "thermal_unmet_kwh",
     "Exp. annual unmet\nenergy (kWh/yr)"),
    ("worst",  "critical_worst_event_h", "hvac_worst_event_h",
     "Worst single outage\nevent (h)"),
]

# slot geometry: 6 slots (method-major, arch-minor), grouped by method
SLOTS = [(m, a) for m in S.METHOD_ORDER for a in S.ARCH_ORDER]
GROUP_GAP = 0.65          # extra space between method groups
pos = {}
x = 0.0
for i, (m, a) in enumerate(SLOTS):
    pos[(m, a)] = x
    x += 1.0
    if a == S.ARCH_ORDER[-1] and i != len(SLOTS) - 1:
        x += GROUP_GAP     # gap after each method's last arch
SW = 0.40                  # sub-bar width
DX = SW / 2 + 0.02         # electrical/thermal offset within a slot
XLIM = (min(pos.values()) - 0.7, max(pos.values()) + 0.7)

# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 5, figsize=(S.DOUBLE_COL, 5.4))

for ri, (pkey, ecol, tcol, ylab) in enumerate(PANELS):
    for ci, loc in enumerate(S.LOCATION_ORDER):
        ax = axes[ri, ci]
        ymax = 0.0
        for m in S.METHOD_ORDER:
            for a in S.ARCH_ORDER:
                r = cell(loc, m, a)
                if r is None:
                    continue
                p = pos[(m, a)]
                em, es = getattr(r, f"{ecol}_mean"), getattr(r, f"{ecol}_std")
                tm, ts = getattr(r, f"{tcol}_mean"), getattr(r, f"{tcol}_std")
                bs = S.bar_style(m, a)
                # thermal (right, solid) = the dominant service
                ax.bar(p + DX, tm, width=SW, yerr=ts,
                       error_kw=dict(elinewidth=0.7, capsize=1.6,
                                     capthick=0.7, ecolor="0.25"),
                       zorder=3, **bs)
                # electrical (left, translucent) = critical service (~0 here)
                es_bs = dict(bs); es_bs["alpha"] = 0.42
                ax.bar(p - DX, em, width=SW, yerr=es,
                       error_kw=dict(elinewidth=0.7, capsize=1.6,
                                     capthick=0.7, ecolor="0.25"),
                       zorder=3, **es_bs)
                ymax = max(ymax, tm + (ts if np.isfinite(ts) else 0),
                           em + (es if np.isfinite(es) else 0))

        # cosmetics
        S.despine(ax)
        S.ygrid(ax)
        ax.set_xlim(*XLIM)
        ax.set_ylim(0, ymax * 1.20 if ymax > 0 else 1)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7, pad=1.5)
        ax.margins(x=0)
        if ri == 0:
            ax.set_title(S.CLIMATE_LABEL[loc], fontsize=8.5, pad=4)
        if ci == 0:
            ax.set_ylabel(ylab, fontsize=8.5)

# --------------------------------------------------------------------------- #
# Legend: methods (colour) | architecture (hatch) | service (solid/faded)
# --------------------------------------------------------------------------- #
meth_h = [Patch(facecolor=S.METHOD_COLOR[m], edgecolor="black", linewidth=0.6,
                label=S.METHOD_LABEL[m]) for m in S.METHOD_ORDER]
arch_h = [Patch(facecolor="0.82", edgecolor="black", linewidth=0.6,
                hatch=S.ARCH_HATCH[a], label=S.ARCH_LABEL[a])
          for a in S.ARCH_ORDER]
serv_h = [Patch(facecolor="0.45", edgecolor="black", linewidth=0.6,
                label="Thermal / HVAC (right bar)"),
          Patch(facecolor="0.45", edgecolor="black", linewidth=0.6, alpha=0.42,
                label="Electrical / critical (left bar)")]

leg = fig.legend(handles=meth_h + arch_h + serv_h,
                 loc="upper center", bbox_to_anchor=(0.5, 1.005),
                 ncol=7, fontsize=7.3, handlelength=1.4, handleheight=1.2,
                 columnspacing=1.1, borderaxespad=0.0)

fig.text(0.5, 0.945,
         "Within each climate: bars ordered LP-Avg | LP-Worst | SO-CVaR "
         "(colour); No-PCM then PCM (hatch); per-climate y-scales",
         ha="center", va="top", fontsize=6.6, color="0.35")

fig.tight_layout(rect=(0, 0, 1, 0.915), w_pad=0.7, h_pad=1.4)

# --------------------------------------------------------------------------- #
caption = (
    "Out-of-sample loss of load by climate for the renewable architectures "
    "(No-PCM PV+battery vs PCM) under each capacity-planning method, at the Med "
    "VoLL operating point (thermal $3/kWh, critical $100/kWh). Top: expected "
    "annual unmet energy (kWh/yr), derived as penalty divided by VoLL; bottom: "
    "worst single-outage-event duration (h) -- a tail-SEVERITY proxy, not the "
    "quantity SO-CVaR optimises (the objective minimises the CVaR of the "
    "energy-weighted VoLL penalty, not event duration). Each panel shows the "
    "electrical/critical service (left, translucent) and thermal/HVAC service "
    "(right, solid) side by side; bars are grouped by method (colour) and "
    "architecture (hatch). Values are test-split means over 5 test years then 5 "
    "folds; error bars are +/-1 SD across folds. Note the per-climate y-scales "
    "(cold->hot). Electrical/critical loss of load is negligible everywhere "
    "(<=0.7 kWh/yr and <=1.2 h); essentially all unmet load is thermal, and "
    "SO-CVaR lowers thermal unmet ENERGY relative to LP-Avg in every climate. On "
    "worst-event DURATION the ordering is not monotone -- e.g. in Marine (CA) "
    "with PCM SO-CVaR's worst event is longer than LP-Avg's -- because duration "
    "is not the optimised objective."
)

# tidy CSV of plotted values
out = agg.copy()
out = out.sort_values(["location", "method", "architecture"])
S.save_fig(fig, "fig4_loss_of_load", section="main", data=out, caption=caption)

# --------------------------------------------------------------------------- #
# Printed stats
# --------------------------------------------------------------------------- #
def g(loc, m, a, col):
    r = cell(loc, m, a)
    return float("nan") if r is None else getattr(r, f"{col}_mean")

print("\n=== Expected annual unmet energy (kWh/yr): electrical | thermal, "
      "test mean over folds, Med VoLL ===")
hdr = f"{'climate':<16}{'method':<9}{'arch':<5}{'elec_kWh':>10}{'therm_kWh':>11}"
print(hdr)
for loc in S.LOCATION_ORDER:
    for m in S.METHOD_ORDER:
        for a in S.ARCH_ORDER:
            print(f"{S.CLIMATE_LABEL[loc]:<16}{S.METHOD_LABEL[m]:<9}{a:<5}"
                  f"{g(loc,m,a,'electric_unmet_kwh'):>10.3f}"
                  f"{g(loc,m,a,'thermal_unmet_kwh'):>11.1f}")

print("\n=== SO-CVaR vs LP-Avg THERMAL unmet-energy reduction (kWh/yr and %) ===")
for a in S.ARCH_ORDER:
    print(f"-- architecture {a} --")
    for loc in S.LOCATION_ORDER:
        base = g(loc, "LP_Avg", a, "thermal_unmet_kwh")
        so = g(loc, "SO_CVaR", a, "thermal_unmet_kwh")
        red = base - so
        pct = 100 * red / base if base else float("nan")
        print(f"  {S.CLIMATE_LABEL[loc]:<16} LP-Avg={base:8.1f}  "
              f"SO-CVaR={so:8.1f}  ->  -{red:7.1f} kWh/yr ({pct:5.1f}%)")

print("\n=== Worst single-event duration (h): electrical | thermal ===")
for loc in S.LOCATION_ORDER:
    for m in S.METHOD_ORDER:
        for a in S.ARCH_ORDER:
            print(f"{S.CLIMATE_LABEL[loc]:<16}{S.METHOD_LABEL[m]:<9}{a:<5}"
                  f"{g(loc,m,a,'critical_worst_event_h'):>8.2f}"
                  f"{g(loc,m,a,'hvac_worst_event_h'):>9.1f}")
