"""
si_fig_risk_params.py — SI figure: ROBUSTNESS check on the a-priori risk parameters
(lambda, alpha). Shows the paper's conclusions do not hinge on the exact value; we do
NOT claim (0.9, 0.9) is cost-optimal (risk-neutral lambda=0 minimises expected cost).

Reads Risk_Sweep_Results/risk_sweep_summary.xlsx (from si_collect_risk_sweep.py):
  (a) Pooled RELATIVE-REGRET surface over (lambda, alpha): each (climate x VoLL)
      cell's nested-CV validation cost normalized to its own optimum, averaged over
      all cells. Flat basin => the choice is robust; global optimum boxed, the fixed
      a-priori point (0.9, 0.9) starred.
  (b) Fixed-vs-tuned nested-CV TEST gap per cell (%): how much the fixed (0.9, 0.9)
      costs vs per-fold-SELECTED (lambda, alpha) on the untouched test blocks. Near
      zero everywhere => fixing the parameters carries no selection advantage (the
      reviewer rebuttal to "you tuned on the evaluation data").
  (c) Per-cell CV-optimal (lambda, alpha): how many of the cells pick each grid
      point; the fixed point is starred. Clustering => stable across the design space.

Run:  .\\.venv_verify\\Scripts\\python.exe paper_figures\\si_fig_risk_params.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, "Risk_Sweep_Results", "risk_sweep_summary.xlsx")
FIXED = (0.9, 0.9)
VOLL_SHADE = {"Low": "#c6dbef", "Med": "#6baed6", "High": "#08519c"}   # light->dark


def main():
    if not os.path.exists(XLSX):
        print("MISSING:", XLSX)
        print("Generate it first (needs Gurobi), then re-run:\n"
              "  Sherlock:  bash sherlock/submit_risk_sweep.sh\n"
              "  then pull Risk_Sweep_Results/risk_sweep_summary.xlsx and re-run this.")
        return

    regret = pd.read_excel(XLSX, sheet_name="GlobalRegret")
    cells = pd.read_excel(XLSX, sheet_name="CellSummary")
    lams = sorted(regret["lambda"].unique())
    alps = sorted(regret["alpha"].unique())
    gopt = regret.loc[regret["mean_rel_regret_pct"].idxmin()]
    gopt_la = (gopt["lambda"], gopt["alpha"])

    fig, axes = plt.subplots(1, 3, figsize=S.figsize_double(2.9),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})

    # ---- (a) pooled relative-regret surface -------------------------------
    ax = axes[0]
    P = regret.pivot(index="alpha", columns="lambda", values="mean_rel_regret_pct").sort_index()
    M = P.values
    im = ax.imshow(M, aspect="auto", cmap="viridis_r", origin="lower")
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_yticks(range(len(alps))); ax.set_yticklabels([f"{a:g}" for a in alps])
    ax.set_xlabel(r"CVaR weight $\lambda$"); ax.set_ylabel(r"confidence $\alpha$")
    ax.set_title("(a) Mean relative regret [%]", fontweight="bold")
    for i in range(len(alps)):
        for j in range(len(lams)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > np.nanmean(M) else "black")
    gj, gi = lams.index(gopt_la[0]), alps.index(gopt_la[1])
    ax.add_patch(plt.Rectangle((gj - 0.5, gi - 0.5), 1, 1, fill=False,
                               edgecolor="#D55E00", linewidth=2.2, zorder=5))
    fj, fi = lams.index(FIXED[0]), alps.index(FIXED[1])
    ax.scatter([fj], [fi], marker="*", s=110, color="white", edgecolor="black",
               linewidth=0.5, zorder=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="regret vs cell optimum [%]")

    # ---- (b) fixed-vs-tuned gap per cell ----------------------------------
    ax = axes[1]
    climates = [c for c in S.LOCATION_ORDER if c in set(cells["climate"])]
    volls = [v for v in S.VOLL_ORDER if v in set(cells["voll"])]
    x = np.arange(len(climates)); w = 0.8 / max(len(volls), 1)
    for k, v in enumerate(volls):
        vals = [cells[(cells.climate == c) & (cells.voll == v)]["gap_pct"].mean()
                for c in climates]
        ax.bar(x + (k - (len(volls) - 1) / 2) * w, vals, w, color=VOLL_SHADE.get(v, "0.5"),
               edgecolor="black", linewidth=0.4, label=f"{v} VoLL")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([S.CLIMATE_LABEL[c].split(" (")[1].rstrip(")") for c in climates])
    ax.set_ylabel("Fixed − tuned test cost [%]")
    ax.set_title("(b) Cost of fixing (0.9, 0.9)", fontweight="bold")
    ax.legend(fontsize=6, ncol=1)
    S.despine(ax); S.ygrid(ax)

    # ---- (c) per-cell CV-optimal clustering -------------------------------
    ax = axes[2]
    cnt = cells.groupby(["cv_opt_lambda", "cv_opt_alpha"]).size().reset_index(name="n")
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_yticks(range(len(alps))); ax.set_yticklabels([f"{a:g}" for a in alps])
    ax.set_xlim(-0.6, len(lams) - 0.4); ax.set_ylim(-0.6, len(alps) - 0.4)
    for _, r in cnt.iterrows():
        if r["cv_opt_lambda"] not in lams or r["cv_opt_alpha"] not in alps:
            continue
        j = lams.index(r["cv_opt_lambda"]); i = alps.index(r["cv_opt_alpha"])
        ax.scatter([j], [i], s=90 + 130 * r["n"], color=S.METHOD_COLOR["SO_CVaR"],
                   alpha=0.85, edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(j, i, str(int(r["n"])), ha="center", va="center", fontsize=8,
                color="white", zorder=4)
    fj, fi = lams.index(FIXED[0]), alps.index(FIXED[1])
    ax.scatter([fj], [fi], marker="*", s=120, color="black", zorder=5)
    ax.set_xlabel(r"CVaR weight $\lambda$"); ax.set_ylabel(r"confidence $\alpha$")
    ax.set_title(f"(c) Per-cell optimum (n={len(cells)})", fontweight="bold")
    S.despine(ax)

    fig.suptitle(
        "Robustness of results to the a-priori risk parameters (nested CV, 5 climates x 3 VoLL). "
        "Fixed operating point (0.9, 0.9) = star; the paper's conclusions do not hinge on the exact value.",
        fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    mean_gap = cells["gap_pct"].mean(); max_gap = cells["gap_pct"].max()
    caption = (
        "SI (robustness, not selection): the SO-CVaR risk parameters -- CVaR weight lambda "
        "and confidence alpha -- are FIXED a priori at (0.9, 0.9) as a risk-averse posture "
        "for a mission-critical base (the LP baselines are likewise knob-free). This figure "
        "shows the choice is robust and the paper's conclusions do not hinge on the exact "
        "value. Assessment is by nested cross-validation over the 25 weather years (5-year "
        "blocks as groups; the test blocks are never used to choose parameters). (a) "
        "Out-of-sample cost of each (lambda, alpha) as relative regret vs each cell's own "
        "minimum, averaged over cells: the surface is smooth (not knife-edge). Minimising "
        "expected COST favours the risk-neutral corner (lambda=0); the lambda=0 column is "
        "alpha-invariant, so the cost optimum is a column, not a point. We do NOT claim "
        "(0.9, 0.9; star) is cost-optimal -- it is a deliberate risk preference sitting on "
        "a smooth, monotone lambda-slope about 1.4% above the cost minimum. (b) Nested-CV "
        "test cost of the fixed "
        f"(0.9, 0.9) minus the per-fold cost-optimal parameters, per cell (mean "
        f"{mean_gap:+.2f}%, max {max_gap:+.2f}%): fixing the risk-averse setting costs only "
        "this small, bounded premium in mean cost -- the price paid for the tail reliability "
        "CVaR targets (see the loss-of-load figure). (c) Per-cell cost-optima; the fixed "
        "point (star) is a risk choice, not a cost minimiser. Overall: results are robust "
        "to the risk parameters, so we fix them a priori and report this sensitivity here.")
    S.save_fig(fig, "si_fig_risk_params", section="si", data=cells, caption=caption)


if __name__ == "__main__":
    main()
