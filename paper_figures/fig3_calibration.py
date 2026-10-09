"""
fig3_calibration.py — LEAD reliability result: out-of-sample calibration.

Does the reliability a method promises at the planning stage actually hold on
unseen weather years? For each method x architecture x climate we plot the mean
TRAINING-year expected unmet energy (x) against the mean TEST-year value (y).
The dashed y=x line is perfect calibration:
  * points ON the line  -> the design delivers its planned reliability (trustworthy)
  * points ABOVE (test > train) -> UNDER-provisioned out of sample (optimistic)
  * points BELOW (test < train) -> OVER-provisioned (wasteful robustness)
SO-CVaR sits on the line; LP-Avg sits above (under-provisions); LP-Worst sits far
below (over-provisions). Panel (b): per-method test/train unmet ratio (1.0 = calibrated).
Med VoLL; log-log because climate scales differ ~100x. Loss of load is ~all thermal.

Run:  .\\.venv_verify\\Scripts\\python.exe paper_figures\\fig3_calibration.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S
import data as D

# --- data: per (arch, location, method, VoLL) fold-mean train & test unmet energy
#     ALL three VoLL levels -> 30 climate x architecture x VoLL cells (10 per method).
folds = D.load_folds(architectures=("PCM", "PVB"))
folds = D.add_unmet_energy(folds)

KEYS = ["architecture", "location", "method", "voll"]
tr = (folds[folds.split == "train"][KEYS + ["fold", "total_unmet_kwh"]]
      .rename(columns={"total_unmet_kwh": "train"}))
te = (folds[folds.split == "test"][KEYS + ["fold", "total_unmet_kwh"]]
      .rename(columns={"total_unmet_kwh": "test"}))
m = tr.merge(te, on=KEYS + ["fold"])
g = (m.groupby(KEYS, observed=True)
     .agg(train=("train", "mean"), test=("test", "mean")).reset_index())
g["ratio"] = g["test"] / g["train"].replace(0, np.nan)

# --- figure -------------------------------------------------------------------
fig, (axa, axb) = plt.subplots(1, 2, figsize=S.figsize_double(3.5),
                               gridspec_kw={"width_ratios": [1.35, 1.0]})

# ============================ Panel (a): calibration ======================== #
lo, hi = 5.0, 6000.0
xs = np.array([lo, hi])
axa.fill_between(xs, xs, hi, color=S.METHOD_COLOR["LP_Avg"], alpha=0.06, zorder=0)   # above: under-provision
axa.fill_between(xs, lo, xs, color=S.METHOD_COLOR["LP_Worst"], alpha=0.06, zorder=0)  # below: over-provision
axa.plot(xs, xs, ls="--", color="0.4", lw=1.1, zorder=2, label="perfect calibration (test = plan)")

# annotations sit in the two off-diagonal triangles, clear of the (upper-left) legend
axa.text(700, 4800, "under-provisioned\n(test > plan)", fontsize=6.4,
         color=S.METHOD_COLOR["LP_Avg"], ha="center", va="top", style="italic")
axa.text(4500, 8, "over-provisioned\n(test < plan)", fontsize=6.4,
         color=S.METHOD_COLOR["LP_Worst"], ha="right", va="bottom", style="italic")

for _, r in g.iterrows():
    ms = S.marker_style(r["method"], r["architecture"])
    axa.plot([r["train"]], [r["test"]], linestyle="none", markersize=6.5,
             zorder=4, **ms)

axa.set_xscale("log"); axa.set_yscale("log")
axa.set_xlim(lo, hi); axa.set_ylim(lo, hi)
axa.set_aspect("equal")
axa.set_xlabel("Planned (training-year) unmet energy (kWh/yr, log)")
axa.set_ylabel("Delivered (test-year) unmet energy (kWh/yr, log)")
axa.set_title("(a) Out-of-sample reliability calibration", loc="left")
S.despine(axa)

# panel-(a) legend: methods (colour) + architectures (marker fill)
meth_h = [Line2D([0], [0], marker="s", linestyle="none", markersize=7,
                 markerfacecolor=S.METHOD_COLOR[mm], markeredgecolor="black",
                 markeredgewidth=0.5, label=S.METHOD_LABEL[mm]) for mm in S.METHOD_ORDER]
arch_h = [Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                 markerfacecolor="white", markeredgecolor="0.25", markeredgewidth=1.2,
                 label=S.ARCH_LABEL["PVB"]),
          Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                 markerfacecolor="0.35", markeredgecolor="black", markeredgewidth=0.5,
                 label=S.ARCH_LABEL["PCM"])]
line_h = [Line2D([0], [0], ls="--", color="0.4", lw=1.1, label="test = plan")]
axa.legend(handles=meth_h + arch_h + line_h, loc="upper left", fontsize=6.0,
           handletextpad=0.4, labelspacing=0.3, borderaxespad=0.4)

# ============================ Panel (b): ratio ============================== #
xpos = {mm: i for i, mm in enumerate(S.METHOD_ORDER)}
axb.axhline(1.0, color="0.4", ls="--", lw=1.0, zorder=1)
rng = np.random.default_rng  # not used; jitter is deterministic below
for mm in S.METHOD_ORDER:
    sub = g[g.method == mm]
    xs_j = xpos[mm] + np.linspace(-0.18, 0.18, len(sub))
    c = S.method_color(mm)
    # architecture encoding matches panel (a): PCM filled, No-PCM (PVB) hollow
    is_pcm = (sub["architecture"] == "PCM").to_numpy()
    facecolors = [c if p else "white" for p in is_pcm]
    edgecolors = ["black" if p else c for p in is_pcm]
    lws = [0.5 if p else 1.1 for p in is_pcm]
    axb.scatter(xs_j, sub["ratio"], s=24, facecolors=facecolors, edgecolors=edgecolors,
                linewidths=lws, alpha=0.85, zorder=3)
    med = sub["ratio"].median()
    axb.plot([xpos[mm] - 0.28, xpos[mm] + 0.28], [med, med], color=c, lw=2.4, zorder=4)
    axb.annotate(f"{med:.2f}", (xpos[mm] + 0.30, med), fontsize=6.6, va="center",
                 ha="left", color=c, weight="bold")

axb.set_yscale("log")
axb.set_yticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
axb.set_yticklabels(["0.3", "0.5", "0.7", "1.0", "1.5", "2.0", "3.0"])
axb.set_ylim(0.30, 3.2)
axb.set_xticks(list(xpos.values()))
axb.set_xticklabels([S.METHOD_LABEL[mm] for mm in S.METHOD_ORDER], rotation=12)
axb.set_ylabel("Test / plan unmet-energy ratio")
axb.set_title("(b) Calibration ratio (1.0 = exact)", loc="left")
axb.text(0.02, 0.97, "under-provision", transform=axb.transAxes, fontsize=6.2,
         color=S.METHOD_COLOR["LP_Avg"], va="top", style="italic")
axb.text(0.02, 0.03, "over-provision", transform=axb.transAxes, fontsize=6.2,
         color=S.METHOD_COLOR["LP_Worst"], va="bottom", style="italic")
S.despine(axb)

fig.tight_layout(w_pad=1.8)

# --- save + printed stats -----------------------------------------------------
out = g.copy()
med_ratio = g.groupby("method", observed=True)["ratio"].median().reindex(S.METHOD_ORDER)
caption = (
    "Out-of-sample reliability calibration across all three VoLL levels "
    "(30 climate x architecture x VoLL cells, 10 per method); loss of load is "
    "essentially all thermal. (a) Mean training-year unmet energy (planned) versus "
    "mean test-year unmet energy (delivered), over the 5 folds; the dashed line is "
    "perfect calibration (test = plan), with points above it under-provisioned out of "
    "sample and points below over-provisioned. Colour = method, marker fill = "
    "architecture; log-log axes (climate scales differ ~100x). (b) Per-method "
    "test/plan unmet-energy ratio (bar = median; 1.0 = calibrated): SO-CVaR "
    f"{med_ratio['SO_CVaR']:.2f}, LP-Avg {med_ratio['LP_Avg']:.2f}, LP-Worst "
    f"{med_ratio['LP_Worst']:.2f}.")
S.save_fig(fig, "fig3_calibration", section="main", data=out, caption=caption)

print("=== Out-of-sample unmet-energy calibration (test/plan ratio), Med VoLL ===")
print("median test/plan ratio by method:")
for mm in S.METHOD_ORDER:
    sub = g[g.method == mm]
    print(f"  {S.METHOD_LABEL[mm]:9s}  median {sub['ratio'].median():.3f}  "
          f"mean {sub['ratio'].mean():.3f}  "
          f"[{sub['ratio'].min():.2f}, {sub['ratio'].max():.2f}]  n={len(sub)}")
smallest = (g.assign(absdev=(g["ratio"] - 1).abs())
            .groupby("method", observed=True)["absdev"].mean().reindex(S.METHOD_ORDER))
print("\nmean |ratio-1| by method (smaller = better calibrated):")
print(smallest.round(3).to_string())
