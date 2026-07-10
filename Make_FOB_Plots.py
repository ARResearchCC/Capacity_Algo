"""
Make_FOB_Plots.py
------------------
IEEE-style figures for the FOB capacity-planning sensitivity study.

Headline: SO_CVaR vs the two LP baselines (LP_Avg, LP_Worst) across
5 climates x 3 VoLL levels, evaluated out-of-sample (5-fold CV, 1998-2022).

Primary source : FOB_Sensitivity_Results.xlsx      (PV + Battery + PCM heat/cool)
Ablation source: FOB_PVB_Sensitivity_Results.xlsx   (PV + Battery only, PCM = 0)

All figures use the "Summary" sheet (mean over the 5 folds).
Output: PNG (300 dpi) into FOB_Plots/.

Run with the venv that has matplotlib:
    .\.venv_verify\Scripts\python.exe Make_FOB_Plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
FOB_FILE = "FOB_Sensitivity_Results.xlsx"
PVB_FILE = "FOB_PVB_Sensitivity_Results.xlsx"
OUTDIR = "FOB_Plots"
os.makedirs(OUTDIR, exist_ok=True)

# Display order
METHODS = ["LP_Avg", "LP_Worst", "SO_CVaR"]
METHOD_LABEL = {"LP_Avg": "LP-Avg", "LP_Worst": "LP-Worst", "SO_CVaR": "SO-CVaR"}
VOLL = ["Low", "Med", "High"]
LOCATIONS = ["Alaska", "Minnesota", "California", "Arizona", "Florida"]

# Okabe-Ito colorblind-safe palette (hero = SO_CVaR)
METHOD_COLOR = {"LP_Avg": "#0072B2", "LP_Worst": "#E69F00", "SO_CVaR": "#009E73"}
METHOD_MARKER = {"LP_Avg": "o", "LP_Worst": "s", "SO_CVaR": "D"}   # grayscale-safe
METHOD_HATCH = {"LP_Avg": "///", "LP_Worst": "\\\\\\", "SO_CVaR": ""}

# Asset palette (semantic: sun=gold, batt=green, heat=red, cool=blue)
ASSET_COL = {
    "PV_Size": "#E69F00",
    "Battery_Size": "#009E73",
    "PCM_Heating_Size": "#D55E00",
    "PCM_Cooling_Size": "#56B4E9",
}
ASSET_LABEL = {
    "PV_Size": "PV (kW)",
    "Battery_Size": "Battery (kWh)",
    "PCM_Heating_Size": "PCM heat (kWh$_{t}$)",
    "PCM_Cooling_Size": "PCM cool (kWh$_{t}$)",
}

# Cost breakdown (stacked)
COST_COMP = {
    "Testing Capital Cost": ("Capital", "#4477AA"),
    "Testing HVAC Cost": ("HVAC outage", "#E69F00"),
    "Testing Critical Load Cost": ("Critical outage", "#CC3311"),
}

TWO_COL = 7.16   # IEEE double-column width (in)
ONE_COL = 3.5    # IEEE single-column width (in)


# --------------------------------------------------------------------------- #
#  Style + data
# --------------------------------------------------------------------------- #
def set_ieee_style():
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    # Arial on Windows; fall back gracefully
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    mpl.rcParams.update({
        "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.titlesize": 9,
        "axes.linewidth": 0.6, "grid.linewidth": 0.5,
        "lines.linewidth": 1.2, "patch.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def load_summary(path):
    df = pd.read_excel(path, sheet_name="Summary")
    df["Total LoL Hours"] = df["Testing HVAC LoL Hours"] + df["Testing Critical LoL Hours"]
    df["Testing Outage Cost"] = df["Testing HVAC Cost"] + df["Testing Critical Load Cost"]
    # ordered categoricals for stable sorting/plotting
    df["Method"] = pd.Categorical(df["Method"], METHODS, ordered=True)
    df["VoLL Level"] = pd.Categorical(df["VoLL Level"], VOLL, ordered=True)
    df["Location"] = pd.Categorical(df["Location"], LOCATIONS, ordered=True)
    return df.sort_values(["Location", "VoLL Level", "Method"]).reset_index(drop=True)


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.5)


def _facet_grid(ntitle="", figsize=(TWO_COL, 4.0)):
    """2x3 grid: 5 location panels + a 6th cell reserved for the legend."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    return fig, axes


def _method_legend(fig, marker=True, hatch=False, loc="lower right",
                   bbox=(0.98, 0.06), title="Method"):
    handles = []
    for m in METHODS:
        if marker:
            handles.append(Line2D([0], [0], marker=METHOD_MARKER[m], color="w",
                                  markerfacecolor=METHOD_COLOR[m],
                                  markeredgecolor="black", markeredgewidth=0.5,
                                  markersize=6, label=METHOD_LABEL[m], linestyle="None"))
        else:
            handles.append(Patch(facecolor=METHOD_COLOR[m], edgecolor="black",
                                 hatch=METHOD_HATCH[m] if hatch else "",
                                 label=METHOD_LABEL[m]))
    lg = fig.legend(handles=handles, loc=loc, bbox_to_anchor=bbox,
                    frameon=False, title=title, ncol=1)
    return lg


def _save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


# --------------------------------------------------------------------------- #
#  Fig 1 - HEADLINE: aggregate cost-reliability positioning
# --------------------------------------------------------------------------- #
def fig_tradeoff(df, tag):
    """One marker per method: mean total cost vs mean outage cost, each cell
    indexed to LP_Avg = 100. Shows SO_CVaR at the knee (ideal = bottom-left)."""
    set_ieee_style()
    piv_tot = df.pivot_table(index=["Location", "VoLL Level"], columns="Method",
                             values="Testing Total Cost", observed=True)
    piv_out = df.pivot_table(index=["Location", "VoLL Level"], columns="Method",
                             values="Testing Outage Cost", observed=True)
    cost_idx = (piv_tot.div(piv_tot["LP_Avg"], axis=0) * 100)[METHODS].mean()
    out_idx = (piv_out.div(piv_out["LP_Avg"], axis=0) * 100)[METHODS].mean()

    fig, ax = plt.subplots(figsize=(3.7, 3.4))
    for m in METHODS:
        ax.scatter(cost_idx[m], out_idx[m], s=110, marker=METHOD_MARKER[m],
                   color=METHOD_COLOR[m], edgecolor="black", linewidth=0.7, zorder=4)
    # reference lines at the LP_Avg baseline
    ax.axhline(100, color="0.75", lw=0.7, ls=":", zorder=1)
    ax.axvline(100, color="0.75", lw=0.7, ls=":", zorder=1)
    # point labels
    ax.annotate("LP-Avg", (cost_idx["LP_Avg"], out_idx["LP_Avg"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5, va="bottom")
    ax.annotate("LP-Worst", (cost_idx["LP_Worst"], out_idx["LP_Worst"]),
                textcoords="offset points", xytext=(7, -1), fontsize=7.5, va="center")
    dc = cost_idx["SO_CVaR"] - 100
    do = out_idx["SO_CVaR"] - 100
    ax.annotate(f"SO-CVaR\n{dc:+.1f}% cost,  {do:+.0f}% outage\nvs LP-Avg",
                (cost_idx["SO_CVaR"], out_idx["SO_CVaR"]),
                textcoords="offset points", xytext=(14, 26), fontsize=7.5,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.set_xlabel("Out-of-sample total cost  (LP-Avg = 100)")
    ax.set_ylabel("Out-of-sample outage cost  (LP-Avg = 100)")
    ax.set_ylim(46, 112)
    ax.margins(x=0.28)
    # 'better' hint in the ideal corner
    ax.annotate("better", xy=(ax.get_xlim()[0], 50), xytext=(24, 22),
                textcoords="offset points", fontsize=7, color="0.45",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
    _clean(ax)
    ax.grid(axis="both", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_title("Near worst-case reliability\nat average-case cost", fontweight="bold")
    fig.tight_layout()
    _save(fig, f"fig1_tradeoff_{tag}.png")


# --------------------------------------------------------------------------- #
#  Fig 2 - Cost-reliability Pareto by climate (detail)
# --------------------------------------------------------------------------- #
def fig_pareto(df, tag):
    set_ieee_style()
    fig, axes = _facet_grid(figsize=(TWO_COL, 4.2))
    for ax, loc in zip(axes, LOCATIONS):
        d = df[df["Location"] == loc]
        # one line per method, connecting the 3 VoLL points (Low->High)
        for m in METHODS:
            dm = d[d["Method"] == m].sort_values("VoLL Level")
            x = dm["Testing Total Cost"].values
            y = dm["Total LoL Hours"].values
            hero = (m == "SO_CVaR")
            ax.plot(x, y, color=METHOD_COLOR[m], lw=1.4 if hero else 0.9,
                    alpha=0.9 if hero else 0.6, zorder=3 if hero else 2)
            ax.scatter(x, y, s=[22, 34, 48], marker=METHOD_MARKER[m],
                       facecolor=METHOD_COLOR[m], edgecolor="black",
                       linewidth=0.5, zorder=4 if hero else 3)
        ax.set_title(loc, fontweight="bold")
        _clean(ax)
        ax.grid(axis="both", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.margins(0.12)
    # shared axis labels
    fig.supxlabel("Out-of-sample (testing) total cost  [$]", fontsize=8, y=0.04)
    fig.supylabel("Loss-of-load duration  [h]", fontsize=8, x=0.04)
    # legend in 6th cell
    ax6 = axes[5]
    ax6.axis("off")
    handles = [Line2D([0], [0], marker=METHOD_MARKER[m], color=METHOD_COLOR[m],
                      markerfacecolor=METHOD_COLOR[m], markeredgecolor="black",
                      markeredgewidth=0.5, markersize=6, lw=1.2, label=METHOD_LABEL[m])
               for m in METHODS]
    handles.append(Line2D([0], [0], color="none", label=""))
    handles.append(Line2D([0], [0], marker="o", color="0.4", markerfacecolor="0.7",
                          markeredgecolor="k", markeredgewidth=0.4, markersize=4,
                          lw=0.9, label="Low → High VoLL"))
    ax6.legend(handles=handles, loc="center", frameon=False,
               title="Down-left is better", title_fontsize=8)
    fig.suptitle("Cost–reliability trade-off by climate (each curve: Low→High VoLL)",
                 fontweight="bold")
    fig.tight_layout(rect=[0.04, 0.05, 1, 0.96])
    _save(fig, f"fig2_pareto_{tag}.png")


# --------------------------------------------------------------------------- #
#  Fig 3 - Out-of-sample calibration (train vs test ratios)
# --------------------------------------------------------------------------- #
def fig_calibration(df, tag):
    """How well each method's plan holds up on held-out weather years.
    Ratio = testing / training. >1 = under-planned (optimistic);
    <1 = over-provisioned (conservative); ~1 = well-calibrated."""
    set_ieee_style()
    d = df.copy()
    d["cost_ratio"] = d["Testing Total Cost"] / d["Training Total Cost"]
    d["train_out"] = d["Training HVAC Cost"] + d["Training Critical Load Cost"]
    d["test_out"] = d["Testing HVAC Cost"] + d["Testing Critical Load Cost"]
    d["out_ratio"] = d["test_out"] / d["train_out"].replace(0, np.nan)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TWO_COL, 3.0))
    panels = [(axL, "cost_ratio", "(a) Total cost", "Testing / Training total cost"),
              (axR, "out_ratio", "(b) Outage cost", "Testing / Training outage cost")]
    for ax, col, ttl, yl in panels:
        data = [d[d["Method"] == m][col].dropna().values for m in METHODS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                        medianprops=dict(color="black", linewidth=1.1),
                        flierprops=dict(marker="o", markersize=2.5,
                                        markerfacecolor="white", markeredgecolor="0.5"))
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLOR[m]); patch.set_alpha(0.85)
        ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
        if col == "cost_ratio":   # annotate once; avoids overlap in panel (b)
            ax.text(0.985, 1.0, "planned = realized",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=6.3, color="0.35")
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS])
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontweight="bold")
        _clean(ax)

    fig.suptitle("Out-of-sample calibration: SO-CVaR realizes what it plans "
                 "(LP-Avg under-plans)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"fig3_calibration_{tag}.png")


# --------------------------------------------------------------------------- #
#  Fig 3 - Reliability: HVAC LoL hours by method x VoLL, faceted by location
# --------------------------------------------------------------------------- #
def fig_reliability(df, tag):
    set_ieee_style()
    fig, axes = _facet_grid(figsize=(TWO_COL, 4.2))
    x = np.arange(len(VOLL))
    w = 0.26
    for ax, loc in zip(axes, LOCATIONS):
        d = df[df["Location"] == loc]
        for j, m in enumerate(METHODS):
            dm = d[d["Method"] == m].sort_values("VoLL Level")
            vals = dm["Testing HVAC LoL Hours"].values
            bars = ax.bar(x + (j - 1) * w, vals, w, color=METHOD_COLOR[m],
                          edgecolor="black", label=METHOD_LABEL[m])
            for b in bars:
                b.set_hatch(METHOD_HATCH[m])
        ax.set_title(loc, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(VOLL)
        _clean(ax)
    axes[5].axis("off")
    _method_legend(fig, marker=False, hatch=True, loc="center",
                   bbox=(0.83, 0.25), title="Method")
    fig.supxlabel("VoLL level", fontsize=8, y=0.04)
    fig.supylabel("HVAC loss-of-load duration  [h]", fontsize=8, x=0.04)
    fig.suptitle("Reliability by method (lower is better)", fontweight="bold")
    fig.tight_layout(rect=[0.04, 0.05, 1, 0.95])
    _save(fig, f"fig4_reliability_{tag}.png")


# --------------------------------------------------------------------------- #
#  Fig 4 - Cost breakdown stacked (Capital / HVAC outage / Critical outage)
# --------------------------------------------------------------------------- #
def fig_cost_breakdown(df, tag, voll="Med"):
    set_ieee_style()
    fig, axes = _facet_grid(figsize=(TWO_COL, 4.2))
    comps = list(COST_COMP.keys())
    for ax, loc in zip(axes, LOCATIONS):
        d = df[(df["Location"] == loc) & (df["VoLL Level"] == voll)]
        x = np.arange(len(METHODS))
        bottom = np.zeros(len(METHODS))
        for c in comps:
            vals = [d[d["Method"] == m][c].values[0] for m in METHODS]
            label, col = COST_COMP[c]
            ax.bar(x, vals, 0.62, bottom=bottom, color=col, edgecolor="black",
                   label=label, linewidth=0.4)
            bottom += np.array(vals)
        # total annotation
        for xi, tot in zip(x, bottom):
            ax.text(xi, tot, f"{tot:,.0f}", ha="center", va="bottom", fontsize=6)
        ax.set_title(loc, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], rotation=0)
        ax.margins(y=0.12)
        _clean(ax)
    axes[5].axis("off")
    handles = [Patch(facecolor=COST_COMP[c][1], edgecolor="black", label=COST_COMP[c][0])
               for c in comps]
    axes[5].legend(handles=handles, loc="center", frameon=False, title="Cost component")
    fig.supylabel("Out-of-sample (testing) cost  [$]", fontsize=8, x=0.04)
    fig.suptitle(f"Cost composition at {voll} VoLL: SO-CVaR trades a little capital "
                 f"for much less outage", fontweight="bold")
    fig.tight_layout(rect=[0.04, 0, 1, 0.95])
    _save(fig, f"fig5_costbreakdown_{voll}_{tag}.png")


# --------------------------------------------------------------------------- #
#  Fig 5 - Tail risk: Expected vs CVaR outage cost (SO_CVaR only)
# --------------------------------------------------------------------------- #
def fig_tailrisk(df, tag):
    set_ieee_style()
    d = df[df["Method"] == "SO_CVaR"].copy()
    # build one row per (location, VoLL)
    d["cell"] = d["Location"].astype(str) + "  " + d["VoLL Level"].astype(str)
    order = []
    for loc in LOCATIONS:
        for v in VOLL:
            order.append(f"{loc}  {v}")
    d = d.set_index("cell").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(TWO_COL, 4.4))
    y = np.arange(len(d))[::-1]  # top-down
    exp = d["Training Expected Outage Cost"].values
    cv = d["Training CVaR Outage Cost"].values
    # connecting stems
    for yi, e, c in zip(y, exp, cv):
        ax.plot([e, c], [yi, yi], color="0.6", lw=1.0, zorder=1)
    ax.scatter(exp, y, s=26, marker="o", color="#0072B2", edgecolor="black",
               linewidth=0.4, zorder=3, label="Expected outage (mean)")
    ax.scatter(cv, y, s=30, marker="D", color="#CC3311", edgecolor="black",
               linewidth=0.4, zorder=3, label="CVaR outage (worst-10% tail)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["cell"].values, fontsize=6.2)
    ax.set_xscale("log")
    ax.set_xlabel("Training outage cost  [$]  (log scale)")
    ax.set_title("Tail risk quantified by SO-CVaR: expected vs worst-case outage",
                 fontweight="bold")
    _clean(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.35, linewidth=0.5)
    # light separators between locations
    for k in range(1, len(LOCATIONS)):
        ax.axhline(len(d) - k * len(VOLL) - 0.5, color="0.85", lw=0.5)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, f"fig6_tailrisk_{tag}.png")


# --------------------------------------------------------------------------- #
#  Fig 6 - Optimal capacity mix by climate x method
# --------------------------------------------------------------------------- #
def fig_capacity_mix(df, tag, voll="Med"):
    set_ieee_style()
    fig, axes = _facet_grid(figsize=(TWO_COL, 4.2))
    assets = list(ASSET_COL.keys())
    for ax, loc in zip(axes, LOCATIONS):
        d = df[(df["Location"] == loc) & (df["VoLL Level"] == voll)]
        x = np.arange(len(METHODS))
        w = 0.2
        for k, a in enumerate(assets):
            vals = [d[d["Method"] == m][a].values[0] for m in METHODS]
            ax.bar(x + (k - 1.5) * w, vals, w, color=ASSET_COL[a],
                   edgecolor="black", linewidth=0.4, label=ASSET_LABEL[a])
        ax.set_title(loc, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], fontsize=6.2)
        _clean(ax)
        ax.margins(y=0.1)
    axes[5].axis("off")
    handles = [Patch(facecolor=ASSET_COL[a], edgecolor="black", label=ASSET_LABEL[a])
               for a in assets]
    axes[5].legend(handles=handles, loc="center", frameon=False, title="Asset")
    fig.supylabel("Installed capacity", fontsize=8, x=0.04)
    fig.suptitle(f"Optimal capacity mix at {voll} VoLL (climate-adaptive)",
                 fontweight="bold")
    fig.tight_layout(rect=[0.04, 0, 1, 0.95])
    _save(fig, f"fig7_capacity_{voll}_{tag}.png")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def build_all(path, tag):
    df = load_summary(path)
    fig_tradeoff(df, tag)       # fig1 - headline
    fig_pareto(df, tag)         # fig2 - per-climate detail
    fig_calibration(df, tag)    # fig3 - out-of-sample calibration
    fig_reliability(df, tag)    # fig4 - LoL hours
    fig_cost_breakdown(df, tag, "Med")   # fig5
    fig_tailrisk(df, tag)       # fig6 - CVaR tail
    fig_capacity_mix(df, tag, "Med")     # fig7


if __name__ == "__main__":
    build_all(FOB_FILE, "FOB")     # headline: full system (PV+Battery+PCM)
    build_all(PVB_FILE, "PVB")     # ablation: PV+Battery only (PCM = 0)
    print("done")
