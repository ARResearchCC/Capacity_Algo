"""
si_collect_risk_sweep.py — aggregate the nested-CV risk-sweep partials across the
whole design space (climate x VoLL) and produce the OPTION-1 evidence: the fixed,
a-priori (lambda, alpha) is near-optimal everywhere and carries no selection
advantage over data-driven tuning.

Reads Risk_Sweep_Results/partials/*.csv (from si_run_risk_sweep.py). For each
(climate, VoLL) cell it runs the nested-CV bookkeeping (selection on inner
validation only; test never used for selection), then reports:

  * per-cell CV-optimal (lambda, alpha) = argmin pooled inner-validation cost;
  * the UNBIASED nested-CV test cost of the per-fold-SELECTED params vs the FIXED
    a-priori (lambda, alpha) -> their GAP is the reviewer-rebuttal number
    ("fixing the risk parameters costs ~this much vs tuning per fold");
  * a pooled RELATIVE-REGRET surface over (lambda, alpha): each cell's validation
    cost normalized to its own optimum, averaged across all cells (so Alaska's
    large absolute costs don't dominate) -> shows a flat basin and where the fixed
    point sits; the argmin is the best single GLOBAL value if you choose to re-fix.

Writes Risk_Sweep_Results/risk_sweep_summary.xlsx: Folds, CellSummary, GlobalRegret.

Pure pandas/numpy — run in either env:
    .\\.venv_verify\\Scripts\\python.exe paper_figures\\si_collect_risk_sweep.py
"""

import os
import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "Risk_Sweep_Results")
PART_DIR = os.path.join(OUT_DIR, "partials")
FIXED = (0.9, 0.9)              # a-priori fixed (lambda, alpha) used in the mains
SELECT_ON = "total_cost"        # validation metric for selection (matches paper)


def load_partials():
    files = sorted(glob.glob(os.path.join(PART_DIR, "*.csv")))
    if not files:
        raise SystemExit(
            f"No partials in {PART_DIR}. Run the sweep first (Sherlock: "
            "bash sherlock/submit_risk_sweep.sh).")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _k(l, a):
    return (round(float(l), 6), round(float(a), 6))


def process_cell(dc, climate, voll):
    inner = dc[dc.stage == "inner_val"]
    test = dc[dc.stage == "outer_test"]

    # pooled inner-validation surface (selection criterion)
    vs = inner.groupby(["lambda", "alpha"])[SELECT_ON].mean()
    (cv_l, cv_a) = vs.idxmin()
    cell_min = float(vs.min())

    # per-outer inner-validation mean + test lookup
    inner_o = inner.groupby(["outer", "lambda", "alpha"])[SELECT_ON].mean()
    test_lu = {}
    for _, r in test.iterrows():
        test_lu[(int(r["outer"]), *_k(r["lambda"], r["alpha"]))] = r

    sel_c, fix_c, sel_u, fix_u, sel_la = [], [], [], [], []
    for o in sorted(dc["outer"].unique()):
        io = inner_o.xs(o, level="outer")
        (sl, sa) = io.idxmin()
        sel_la.append((sl, sa))
        ts = test_lu[(o, *_k(sl, sa))]
        sel_c.append(ts["total_cost"]); sel_u.append(ts["total_unmet_kwh"])
        td = test_lu.get((o, *_k(*FIXED)))
        if td is not None:
            fix_c.append(td["total_cost"]); fix_u.append(td["total_unmet_kwh"])

    sel_c = np.array(sel_c); fix_c = np.array(fix_c)
    # per-cell surface with relative regret (for the pooled global surface)
    surf = vs.reset_index().rename(columns={SELECT_ON: "val_cost_mean"})
    surf["climate"] = climate; surf["voll"] = voll
    surf["cell_min"] = cell_min
    surf["rel_regret"] = surf["val_cost_mean"] / cell_min - 1.0

    # modal per-fold selection (stability within the cell)
    from collections import Counter
    modal = Counter(_k(*x) for x in sel_la).most_common(1)[0]

    row = {
        "climate": climate, "voll": voll,
        "cv_opt_lambda": cv_l, "cv_opt_alpha": cv_a,
        "modal_sel_lambda": modal[0][0], "modal_sel_alpha": modal[0][1],
        "modal_sel_nfolds": modal[1],
        "selected_test_mean": float(sel_c.mean()), "selected_test_sd": float(sel_c.std(ddof=1)),
        "fixed_test_mean": float(fix_c.mean()) if len(fix_c) else np.nan,
        "fixed_test_sd": float(fix_c.std(ddof=1)) if len(fix_c) else np.nan,
        "gap_fixed_minus_selected": float(fix_c.mean() - sel_c.mean()) if len(fix_c) else np.nan,
        "gap_pct": float(100 * (fix_c.mean() - sel_c.mean()) / sel_c.mean()) if len(fix_c) else np.nan,
        "selected_test_unmet": float(np.mean(sel_u)),
        "fixed_test_unmet": float(np.mean(fix_u)) if fix_u else np.nan,
    }
    return row, surf


def main():
    df = load_partials()
    cells, surfs = [], []
    for (climate, voll), dc in df.groupby(["climate", "voll"]):
        row, surf = process_cell(dc, climate, voll)
        cells.append(row); surfs.append(surf)
    cell_summary = pd.DataFrame(cells)
    all_surf = pd.concat(surfs, ignore_index=True)

    # pooled relative-regret surface across all cells
    global_regret = (all_surf.groupby(["lambda", "alpha"])
                     .agg(mean_rel_regret_pct=("rel_regret", lambda s: 100 * s.mean()),
                          max_rel_regret_pct=("rel_regret", lambda s: 100 * s.max()))
                     .reset_index().sort_values("mean_rel_regret_pct"))
    gopt = global_regret.iloc[0]
    frow = global_regret[(np.isclose(global_regret["lambda"], FIXED[0]))
                         & (np.isclose(global_regret["alpha"], FIXED[1]))]
    fixed_regret = float(frow["mean_rel_regret_pct"].iloc[0]) if len(frow) else np.nan
    fixed_rank = (int(global_regret.reset_index(drop=True)
                      [(np.isclose(global_regret["lambda"], FIXED[0]))
                       & (np.isclose(global_regret["alpha"], FIXED[1]))].index[0]) + 1
                  if len(frow) else np.nan)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "risk_sweep_summary.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.sort_values(["climate", "voll", "lambda", "alpha", "outer", "inner"]
                       ).to_excel(w, sheet_name="Folds", index=False)
        cell_summary.to_excel(w, sheet_name="CellSummary", index=False)
        global_regret.to_excel(w, sheet_name="GlobalRegret", index=False)

    # ---- report ----
    print("\n" + "=" * 72)
    print(f"FIXED a-priori (lambda, alpha) = {FIXED[0]:g}, {FIXED[1]:g}   ->  {out}")
    print(f"\nPooled relative-regret across {cell_summary.shape[0]} (climate x VoLL) cells:")
    print(f"  Global CV-optimum: lambda={gopt['lambda']:g}, alpha={gopt['alpha']:g} "
          f"(mean regret {gopt['mean_rel_regret_pct']:.2f}%)")
    print(f"  Fixed {FIXED}: mean regret {fixed_regret:.2f}% "
          f"(rank {fixed_rank} of {len(global_regret)} grid points)")
    print(f"\nFixed-vs-tuned nested-CV TEST gap per cell "
          f"(gap = fixed - per-fold-selected; + means fixed is costlier):")
    cs = cell_summary.sort_values(["climate", "voll"])
    for _, r in cs.iterrows():
        print(f"  {r.climate:11s} {r.voll:4s}: selected {r.selected_test_mean:9.1f} | "
              f"fixed {r.fixed_test_mean:9.1f} | gap {r.gap_fixed_minus_selected:+7.1f} "
              f"({r.gap_pct:+.2f}%) | cell CV-opt (l={r.cv_opt_lambda:g}, a={r.cv_opt_alpha:g})")
    print(f"\nGap summary: mean {cs.gap_pct.mean():+.2f}%, max {cs.gap_pct.max():+.2f}%, "
          f"worst cell {cs.loc[cs.gap_pct.idxmax(), 'climate']} "
          f"{cs.loc[cs.gap_pct.idxmax(), 'voll']}.")
    print("If the gap is small everywhere, the fixed a-priori choice carries no "
          "selection advantage vs per-fold tuning -> the main comparison is not "
          "inflated by tuning on the evaluation data.")
    print("\ndone")


if __name__ == "__main__":
    main()
