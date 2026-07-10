"""
si_run_risk_sweep.py — WORKER for the SI risk-parameter (lambda, alpha) study,
using NESTED cross-validation so the selection is reviewer-proof.

WHY NESTED CV: the shipped result workbooks fix lambda=0.9, alpha=0.9. To justify
that choice we must (i) select the risk parameters WITHOUT ever touching the test
data, and (ii) obtain an UNBIASED estimate of out-of-sample performance that
accounts for the fact that selection is itself data-dependent. Plain "sweep and
pick the best on a held-out block" leaks: the block that selects a parameter in
one split is used to test it in another. Nested cross-validation removes that.

PROTOCOL (nested leave-one-block-out CV; 5-year weather blocks as groups, so no
temporal leakage). The 25 years (1998-2022) form 5 blocks B0..B4.
  OUTER loop (test): for each outer fold o, TEST = B[o]; DEV = the other 4 blocks.
  INNER loop (validation), within DEV only: leave-one-dev-block-out (4 inner folds).
      For inner fold i: VAL = one dev block (5 yr), TRAIN = other 3 dev blocks (15 yr).
      For each (lambda, alpha): fit SO-CVaR on TRAIN(15) -> score on VAL(5).
  OUTER test: for each (lambda, alpha): refit SO-CVaR on DEV(20) -> score on TEST(5).

Selection (done in si_collect_risk_sweep.py, never using test):
  * Per outer fold: pick (lambda,alpha)*_o = argmin mean inner-VALIDATION cost over
    the 4 inner folds; its TEST cost on B[o] is a leakage-free estimate.
  * Global fixed choice for the paper: argmin of the pooled inner-validation cost.
  * The fixed default (0.9, 0.9) is compared against the per-fold-selected model on
    the SAME untouched test blocks -> shows fixing 0.9/0.9 costs ~nothing.

SWEEP SCOPE: all 5 climates x 3 VoLL levels x (lambda, alpha) grid, so the a-priori
fixed (lambda, alpha) can be justified across the whole design space (not one cell).

PARALLELISM: one (climate, voll, lambda, alpha) combo per SLURM array task. Each task
does 5 outer x (4 inner-validation fits on 15 yr + 1 outer-test fit on 20 yr) = 25
SO-CVaR solves, and writes a resumable partial CSV. si_collect_risk_sweep.py aggregates.

REQUIRES THE MODEL ENV (Gurobi + pyomo). On Sherlock: bash sherlock/submit_risk_sweep.sh
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

# FOB / Simulate / SO_CVaR (gurobipy + pyomo) are imported lazily inside the worker
# functions, so task_grid()/config import cheaply (submit_risk_sweep.sh sizes the array).

# ---- sweep configuration ------------------------------------------------- #
CLIMATES = ["California", "Arizona", "Alaska", "Minnesota", "Florida"]   # all sites
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]      # CVaR weight
ALPHAS = [0.80, 0.90, 0.95]                     # CVaR confidence
VOLLS = {                                        # (HVAC $/kWh, critical $/kWh)
    "Low": (1.0, 30.0), "Med": (3.0, 100.0), "High": (10.0, 300.0),
}
N_BLOCKS = 5
BLOCK = 5                                        # years per block (25 = 5 x 5)
ROWS_PER_TASK = N_BLOCKS * N_BLOCKS              # 5 outer x (4 inner + 1 test) = 25
OUT_DIR = os.path.join(ROOT, "Risk_Sweep_Results")
PART_DIR = os.path.join(OUT_DIR, "partials")


def task_grid():
    """Deterministic (climate, voll, lambda, alpha) task list; array index -> task."""
    return [(c, v, l, a) for c in CLIMATES for v in VOLLS
            for l in LAMBDAS for a in ALPHAS]


def blocks(years):
    return [years[i * BLOCK:(i + 1) * BLOCK] for i in range(N_BLOCKS)]


def fit_and_eval(train_years, eval_years, lam, alp, nested, climate, cap_costs,
                 hvac_voll, crit_voll):
    """Fit SO-CVaR capacities on train_years; simulate on eval_years; aggregate."""
    import FOB
    import Simulate
    import SO_CVaR
    with FOB.hvac_voll_context(hvac_voll):
        train_list = [nested[climate][y] for y in train_years]
        res = SO_CVaR.SO_CVaR_training(train_list, crit_voll, cap_costs, "FOB",
                                       cvar_alpha=alp, cvar_lambda=lam)
        caps = [float(x) for x in res[0:4]]
        sims = [Simulate.simulate(nested[climate][y], crit_voll, caps, cap_costs, "FOB")
                for y in eval_years]
    tot = float(np.mean([s[0] for s in sims]))
    cap = float(np.mean([s[1] for s in sims]))
    hvac_cost = float(np.mean([s[3] for s in sims]))
    crit_cost = float(np.mean([s[4] for s in sims]))
    thermal_unmet = hvac_cost / hvac_voll
    electric_unmet = crit_cost / crit_voll
    return {
        "PV_Size": caps[0], "Battery_Size": caps[1],
        "PCM_Heating_Size": caps[2], "PCM_Cooling_Size": caps[3],
        "total_cost": tot, "capital_cost": cap,
        "thermal_unmet_kwh": thermal_unmet, "electric_unmet_kwh": electric_unmet,
        "total_unmet_kwh": thermal_unmet + electric_unmet,
    }


def partial_path(climate, voll, lam, alp):
    return os.path.join(PART_DIR, f"{climate}_{voll}_l{lam:g}_a{alp:g}.csv")


def run_task(climate, voll, lam, alp):
    import FOB
    hvac_voll, crit_voll = VOLLS[voll]
    out = partial_path(climate, voll, lam, alp)
    if os.path.exists(out) and len(pd.read_csv(out)) == ROWS_PER_TASK:
        print(f"skip (done): {os.path.basename(out)}")
        return
    cap_costs = [FOB.Input_Parameters.C_PV, FOB.Input_Parameters.C_PV_OP,
                 FOB.Input_Parameters.C_B, FOB.Input_Parameters.C_B_OP]
    FOB.locations = [climate]
    nested = FOB.build_input_data()
    years = list(FOB.weather_year_list)
    B = blocks(years)

    def tag(o, i, stage, r):
        return {"climate": climate, "voll": voll, "lambda": lam, "alpha": alp,
                "outer": o, "inner": i, "stage": stage, **r}

    rows = []
    for o in range(N_BLOCKS):                       # OUTER fold: test = B[o]
        test = B[o]
        dev_idx = [j for j in range(N_BLOCKS) if j != o]
        dev = [y for j in dev_idx for y in B[j]]    # 20 yr
        # INNER leave-one-dev-block-out (validation, never touches test)
        for i in dev_idx:
            val = B[i]
            train = [y for j in dev_idx if j != i for y in B[j]]   # 15 yr
            r = fit_and_eval(train, val, lam, alp, nested, climate, cap_costs,
                             hvac_voll, crit_voll)
            rows.append(tag(o, i, "inner_val", r))
        # OUTER test: refit on DEV(20), evaluate on TEST(5)
        rt = fit_and_eval(dev, test, lam, alp, nested, climate, cap_costs,
                          hvac_voll, crit_voll)
        rows.append(tag(o, -1, "outer_test", rt))
        print(f"{climate} {voll} lambda={lam:g} alpha={alp:g} outer={o}: "
              f"test_cost={rt['total_cost']:.1f}")
    os.makedirs(PART_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


def main():
    grid = task_grid()
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if tid is not None:
        idx = int(tid)
        if idx < 0 or idx >= len(grid):
            raise SystemExit(f"array index {idx} out of range 0..{len(grid)-1}")
        climate, voll, lam, alp = grid[idx]
        print(f"=== task {idx}/{len(grid)-1}: {climate} {voll} "
              f"lambda={lam} alpha={alp} ===")
        run_task(climate, voll, lam, alp)
    else:
        print(f"=== serial run of all {len(grid)} tasks (no SLURM array) ===")
        for climate, voll, lam, alp in grid:
            run_task(climate, voll, lam, alp)
    print("done")


if __name__ == "__main__":
    main()
