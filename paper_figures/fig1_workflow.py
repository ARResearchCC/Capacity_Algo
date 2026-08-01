"""
fig1_workflow.py — Fig 1b: analysis-workflow schematic (matplotlib only).

Left-to-right flow of the capacity-planning study:
  1) Data pipeline  (NSRDB weather -> PV / thermal-passive / electrical-load models)
  2) Sizing methods (LP-Avg, LP-Worst, SO-CVaR; coloured by S.METHOD_COLOR)
  3) Five-fold cross-validation (train 20 yr / test 5 yr)
  4) Cost & reliability metrics (annualized cost; loss of load; unmet energy)
  5) Sensitivities (VoLL; risk params) & diesel benchmark

Boxes are FancyBboxPatch; arrows are FancyArrowPatch. No axes/spines. This is a
starting point to be refined in a drawing tool (the architecture schematic Fig 1a
is drawn separately). Vector PDF is the key output.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import style as S
import data as D  # noqa: F401  (kept for consistency; schematic uses no data)

# --------------------------------------------------------------------------- #
#  Canvas
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=S.figsize_double(3.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 96)
ax.axis("off")

INK   = "#333333"           # arrows / edges
PANEL = "#F4F4F4"           # stage-panel background
PEDGE = "#C9C9C9"

# --------------------------------------------------------------------------- #
#  Drawing helpers
# --------------------------------------------------------------------------- #
def box(cx, cy, w, h, text, *, fc="white", ec=INK, tc="black",
        fs=7, fw="normal", lw=0.9, rounding=1.4, ha="center", va="center",
        z=3):
    x0, y0 = cx - w / 2.0, cy - h / 2.0
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       mutation_aspect=2.0,
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    tx = x0 + 0.6 if ha == "left" else cx
    ax.text(tx, cy, text, ha=ha, va=va, fontsize=fs, color=tc,
            fontweight=fw, zorder=z + 1, linespacing=1.15)
    return dict(cx=cx, cy=cy, w=w, h=h, l=x0, r=x0 + w, t=y0 + h, b=y0)

def arrow(p0, p1, *, color=INK, lw=1.4, ms=11, ls="-", z=2):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        lw=lw, color=color, linestyle=ls,
                        shrinkA=1.5, shrinkB=1.5,
                        joinstyle="miter", capstyle="round", zorder=z)
    ax.add_patch(a)

# --------------------------------------------------------------------------- #
#  Stage panels (light containers) + numbered titles
# --------------------------------------------------------------------------- #
CX  = [10, 30, 50, 70, 90]     # panel centres
HW  = 8.9                       # panel half-width
YB, YT = 5, 79                  # panel bottom / top
TITLES = ["1. Data\npipeline", "2. Sizing\nmethods", "3. Cross-\nvalidation",
          "4. Cost & reliability\nmetrics", "5. Sensitivities\n& benchmark"]

for cx, title in zip(CX, TITLES):
    ax.add_patch(FancyBboxPatch((cx - HW, YB), 2 * HW, YT - YB,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                mutation_aspect=2.0,
                                facecolor=PANEL, edgecolor=PEDGE,
                                linewidth=0.8, zorder=0))
    ax.text(cx, 90, title, ha="center", va="center", fontsize=8.2,
            fontweight="bold", color="black", linespacing=1.1, zorder=1)

BW = 15.6            # inner box width
MID = 42             # mid-height for stage-to-stage arrows

# --------------------------------------------------------------------------- #
#  Stage 1 — data pipeline
# --------------------------------------------------------------------------- #
c = CX[0]
BW1 = 12.0
nsrdb = box(c, 70, BW1, 8.5, "NSRDB\nweather data", fc="#E8EEF5", fs=7)
mods = [
    box(c, 51, BW1, 8.5, "PV model", fs=7),
    box(c, 38, BW1, 8.5, "Thermal /\npassive model", fs=7),
    box(c, 25, BW1, 8.5, "Electrical\nload model", fs=7),
]
# manifold: NSRDB feeds all three models in parallel via a left-side bus
xbus, ytop = c - BW1 / 2 - 1.7, 61.0
ax.plot([c, c], [nsrdb["b"], ytop], color=INK, lw=1.0, zorder=2)
ax.plot([c, xbus], [ytop, ytop], color=INK, lw=1.0, zorder=2)
ax.plot([xbus, xbus], [ytop, mods[-1]["cy"]], color=INK, lw=1.0, zorder=2)
for m in mods:
    arrow((xbus, m["cy"]), (m["l"], m["cy"]), lw=1.0, ms=8)

# --------------------------------------------------------------------------- #
#  Stage 2 — sizing methods (coloured)
# --------------------------------------------------------------------------- #
c = CX[1]
methods = []
mrows = {"LP_Avg": 60, "LP_Worst": 42, "SO_CVaR": 24}
for mkey in S.METHOD_ORDER:
    # SO-CVaR is highlighted by its green colour (as elsewhere); borders are uniform
    methods.append(box(c, mrows[mkey], BW, 9.5, S.METHOD_LABEL[mkey],
                       fc=S.METHOD_COLOR[mkey], ec="black", tc="white",
                       fs=8, fw="bold", lw=0.9))

# --------------------------------------------------------------------------- #
#  Stage 3 — five-fold cross-validation (with a small fold illustration)
# --------------------------------------------------------------------------- #
c = CX[2]
box(c, 62, BW, 9.5, "Five-fold\ncross-validation", fc="#EDEDED", fs=7)
fx0, fw_bar, fh = c - 7.2, 14.4, 3.0             # fold-bar geometry
seg = fw_bar / 5.0
frows = [50, 45, 40, 35, 30]
for i, ry in enumerate(frows):
    for k in range(5):
        test = (k == i)
        ax.add_patch(plt.Rectangle((fx0 + k * seg, ry - fh / 2), seg, fh,
                     facecolor=(S.METHOD_COLOR["SO_CVaR"] if test else "#DCDCDC"),
                     edgecolor="white", linewidth=0.6, zorder=3))
ax.text(c, 24, "Train: 20 yr  /  Test: 5 yr", ha="center", va="center",
        fontsize=6.6, color="black", zorder=4)
ax.text(c - 7.2 + seg / 2, 54.0, "test fold", ha="center", va="center",
        fontsize=5.8, color=S.METHOD_COLOR["SO_CVaR"], zorder=4)

# --------------------------------------------------------------------------- #
#  Stage 4 — cost & reliability metrics
# --------------------------------------------------------------------------- #
c = CX[3]
box(c, MID, BW, 34,
    "Annualized cost\n($/yr)\n\n"
    "Loss of load\n(h/yr, events/yr)\n\n"
    "Unmet energy\n(kWh/yr)",
    fc="#EDEDED", fs=7, rounding=1.6)

# --------------------------------------------------------------------------- #
#  Stage 5 — sensitivities & diesel benchmark
# --------------------------------------------------------------------------- #
c = CX[4]
box(c, 55, BW, 19,
    "VoLL levels\n(Low / Med / High)\n\nRisk params\n(CVaR α)",
    fc="#EDEDED", fs=7, rounding=1.6)
box(c, 24, BW, 9.5, "Diesel\nbenchmark",
    fc=S.METHOD_COLOR["Diesel"], ec="black", tc="white", fs=7.5, fw="bold")

# --------------------------------------------------------------------------- #
#  Stage-to-stage arrows (along the spine)
# --------------------------------------------------------------------------- #
for a, b in zip(CX[:-1], CX[1:]):
    arrow((a + HW, MID), (b - HW, MID), lw=1.6, ms=13)

# --------------------------------------------------------------------------- #
#  Save
# --------------------------------------------------------------------------- #
stage_df = pd.DataFrame({
    "stage": [1, 2, 3, 4, 5],
    "label": ["Data pipeline", "Sizing methods", "Cross-validation",
              "Cost & reliability metrics", "Sensitivities & benchmark"],
    "boxes": [
        "NSRDB weather; PV model; thermal/passive model; electrical load model",
        "LP-Avg; LP-Worst; SO-CVaR",
        "Five-fold CV; train 20 yr / test 5 yr",
        "Annualized cost; loss of load (h/yr, events/yr); unmet energy (kWh/yr)",
        "VoLL levels; risk params (CVaR alpha); Diesel benchmark",
    ],
})

caption = (
    "Fig 1b. Analysis workflow. NSRDB weather drives PV, thermal/passive, and "
    "electrical-load models; the resulting time series feed three capacity-sizing "
    "methods (LP-Avg, LP-Worst, and the highlighted SO-CVaR), each evaluated under "
    "five-fold cross-validation (20 training years / 5 test years per fold). "
    "Test-split annualized cost and reliability metrics (loss of load in h/yr and "
    "events/yr; unmet energy in kWh/yr) are reported, with sensitivities over VoLL "
    "levels and CVaR risk parameters and a diesel-genset benchmark. Schematic only "
    "(no data plotted); method colour encoding matches all other figures."
)

S.save_fig(fig, "fig1_workflow", section="main", data=stage_df, caption=caption)
print("stage labels:", "; ".join(stage_df["label"]))
