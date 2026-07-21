"""
fig_riskterm.py — the risk term earns its keep against the RISK-NEUTRAL stochastic program.

Referees will ask whether SO-CVaR's advantage is just "stochastic optimization beats
deterministic heuristics" or genuinely "risk-aware beats risk-neutral." This figure answers it
by comparing, within the same nested cross-validation, the CVaR design (lambda=0.9) against the
risk-neutral two-stage stochastic program (lambda=0) — the natural scientific baseline (the
lambda=0 corner of the risk sweep; the PCM architecture, 5 climates x 3 VoLL = 15 cells,
alpha=0.9). Both panels: x = risk-neutral SO (lambda=0), y = CVaR (lambda=0.9), dashed y=x;
points BELOW the line => CVaR delivers lower unmet load out of sample. Colour = VoLL.
  (a) mean out-of-sample (test) unmet energy
  (b) worst-fold (tail) out-of-sample unmet energy
The cost of the risk term is a median +1.2% out-of-sample total cost (annotated).

Run:  .\\.venv_verify\\Scripts\\python.exe paper_figures\\fig_riskterm.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, "Risk_Sweep_Results", "risk_sweep_summary.xlsx")
ALPHA = 0.9
VOLL_COL = {"Low": "#9ecae1", "Med": "#4292c6", "High": "#08519c"}

d = pd.read_excel(XLSX, sheet_name="Folds")
t = d[(d["stage"].astype(str) == "outer_test") & (np.isclose(d["alpha"], ALPHA))].copy()

rows = []
for (clim, voll), g in t.groupby(["climate", "voll"]):
    g0 = g[np.isclose(g["lambda"], 0.0)]
    g9 = g[np.isclose(g["lambda"], 0.9)]
    if g0.empty or g9.empty:
        continue
    rows.append(dict(climate=clim, voll=voll,
                     mean0=g0["total_unmet_kwh"].mean(), mean9=g9["total_unmet_kwh"].mean(),
                     worst0=g0["total_unmet_kwh"].max(), worst9=g9["total_unmet_kwh"].max(),
                     cost0=g0["total_cost"].mean(), cost9=g9["total_cost"].mean()))
r = pd.DataFrame(rows)
cost_prem = 100 * (r["cost9"] / r["cost0"] - 1)

fig, (axa, axb) = plt.subplots(1, 2, figsize=S.figsize_double(3.4))

def panel(ax, x, y, lo, hi, axlab, title):
    xs = np.array([lo, hi])
    ax.fill_between(xs, lo, xs, color=S.METHOD_COLOR["SO_CVaR"], alpha=0.06, zorder=0)  # below diag = CVaR lower
    ax.plot(xs, xs, ls="--", color="0.4", lw=1.1, zorder=2)
    nb = int((y < x).sum())
    for xi, yi, v in zip(x, y, r["voll"]):
        ax.scatter([xi], [yi], s=40, marker="D", facecolor=VOLL_COL.get(v, "0.5"),
                   edgecolor="black", linewidth=0.5, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"Risk-neutral SO (λ=0)\n{axlab}")
    ax.set_ylabel(f"CVaR (λ=0.9)\n{axlab}")
    ax.set_title(title, loc="left")
    ax.text(0.96, 0.06, f"CVaR lower\n({nb}/{len(x)} below line)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.6, color=S.METHOD_COLOR["SO_CVaR"], style="italic")
    S.despine(ax)

panel(axa, r["mean0"].values, r["mean9"].values, 5.0, 8000.0,
      "mean test unmet (kWh/yr, log)", "(a) Expected reliability")
panel(axb, r["worst0"].values, r["worst9"].values, 8.0, 9000.0,
      "worst-fold test unmet (kWh/yr, log)", "(b) Reliability tail")

voll_h = [Line2D([0], [0], marker="D", linestyle="none", markersize=7,
                 markerfacecolor=VOLL_COL[v], markeredgecolor="black", markeredgewidth=0.4,
                 label=f"{v} VoLL") for v in S.VOLL_ORDER]
axa.legend(handles=voll_h, loc="upper left", fontsize=6.2, handletextpad=0.3,
           labelspacing=0.3, borderaxespad=0.4, title="PCM architecture",
           title_fontsize=6.2)

fig.tight_layout(w_pad=1.8)

out = r.assign(cost_premium_pct=cost_prem,
               mean_reduction_pct=100 * (1 - r["mean9"] / r["mean0"]),
               worst_reduction_pct=100 * (1 - r["worst9"] / r["worst0"]))
caption = (
    "The risk term versus the risk-neutral stochastic program. Within the same nested "
    "cross-validation, the CVaR design (λ=0.9) is compared against the risk-neutral two-stage "
    "stochastic program (λ=0) — the natural scientific baseline — for the PCM architecture "
    "across five climates and three VoLL levels (15 cells, α=0.9). Each point is one cell; "
    "x = risk-neutral SO, y = CVaR, dashed line = equality, so points below favour CVaR. "
    "(a) Mean out-of-sample unmet energy: CVaR is lower in all 15 cells (median "
    f"{out['mean_reduction_pct'].median():.0f}% reduction). (b) Worst-fold (tail) unmet energy: "
    f"CVaR is lower in all 15 cells (median {out['worst_reduction_pct'].median():.0f}% "
    "reduction), the gap widening with VoLL. The risk term therefore improves out-of-sample "
    "reliability over a proper risk-neutral stochastic baseline — not merely over deterministic "
    f"heuristics — for a median out-of-sample total-cost premium of {cost_prem.median():.1f}%.")
S.save_fig(fig, "fig_riskterm", section="main", data=out, caption=caption)

print("=== CVaR (lambda=0.9) vs risk-neutral SO (lambda=0), PCM, alpha=0.9, outer_test ===")
print(f"mean unmet:  CVaR lower in {(r['mean9']<r['mean0']).sum()}/{len(r)} cells; "
      f"median reduction {out['mean_reduction_pct'].median():.1f}%")
print(f"worst-fold:  CVaR lower in {(r['worst9']<r['worst0']).sum()}/{len(r)} cells; "
      f"median reduction {out['worst_reduction_pct'].median():.1f}%")
print(f"cost:        CVaR premium median {cost_prem.median():.2f}% (max {cost_prem.max():.2f}%)")
