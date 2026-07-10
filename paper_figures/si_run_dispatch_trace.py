"""
si_run_dispatch_trace.py — GENERATOR for the SI dispatch-trace figure.

The result workbooks store only aggregate metrics, not hourly dispatch. This
script re-solves the hourly dispatch for fixed SO-CVaR capacities and writes the
hourly state/flow traces to CSV, for BOTH the PCM and no-PCM (PVB) designs, so
si_fig_dispatch.py can plot a stressed multi-day window.

The dispatch model here MIRRORS Simulate.py (same variables, balances, storage
dynamics, and objective). Keep it in sync with Simulate.py if that model changes.

REQUIRES THE MODEL ENV (Gurobi + pyomo) — run with .venv, NOT .venv_verify:
    .\\.venv\\Scripts\\python.exe paper_figures\\si_run_dispatch_trace.py
Output: Dispatch_Trace_Results/dispatch_<CLIMATE>_<YEAR>_<ARCH>.csv
NOTE: not executed/verified in the figure build; run in the model env.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import Input_Parameters
import FOB

# ---- choose the stressed window here ------------------------------------- #
CLIMATE = "Alaska"      # cold, low-solar winter -> stresses thermal service
YEAR = 2012             # a representative weather year (edit as desired)
HVAC_VOLL, CRIT_VOLL = 3.0, 100.0            # Med VoLL
OUT_DIR = os.path.join(ROOT, "Dispatch_Trace_Results")

# SO-CVaR capacities (Med VoLL, this climate), read from the result workbooks
CAP_FILES = {"PCM": "FOB_Sensitivity_Results.xlsx",
             "PVB": "FOB_PVB_Sensitivity_Results.xlsx"}


def so_cvar_caps(arch):
    df = pd.read_excel(os.path.join(ROOT, CAP_FILES[arch]), sheet_name="Summary")
    r = df[(df.Location == CLIMATE) & (df["VoLL Level"] == "Med")
           & (df.Method == "SO_CVaR")].iloc[0]
    return [r.PV_Size, r.Battery_Size, r.PCM_Heating_Size, r.PCM_Cooling_Size]


def solve_dispatch(input_df, capacities, scenario="FOB"):
    """Mirror of Simulate.simulate's model; returns the solved model + dt."""
    dt = (input_df["DateTime"].iloc[1] - input_df["DateTime"].iloc[0]).total_seconds() / 3600
    N = len(input_df)
    PV = input_df["pv"].to_numpy(float)
    E_Load = input_df["E_Load"].to_numpy(float)
    Cool = input_df["Cooling_Load"].to_numpy(float)
    Heat = input_df["Heating_Load"].to_numpy(float)

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, N - 1)
    m.NumTime = N
    m.HPSize = Input_Parameters.HPSize
    IP = Input_Parameters
    m.dt = dt
    # fixed capacities
    PVSize, BSize, PCMH, PCMC = capacities

    V = pyo.Var
    for name in ["PV2H", "PV2G", "PV2B", "B2H", "H2HP", "HP2H", "H2C", "C2H",
                 "G2H", "HP2PCM_H", "C2PCM_C", "PCM_H2H", "PCM_C2H",
                 "PV2E", "B2E", "G2E", "SOC_B", "SOC_PCM_H", "SOC_PCM_C"]:
        setattr(m, name, V(m.T, within=pyo.NonNegativeReals))

    eta, eta_pv = IP.η, 0.94
    HVAC_cost = dt * HVAC_VOLL * sum(m.G2H[t] for t in m.T)
    crit_cost = dt * CRIT_VOLL * sum(m.G2E[t] for t in m.T)
    m.obj = pyo.Objective(expr=HVAC_cost + crit_cost, sense=pyo.minimize)

    def con(name, rule):
        setattr(m, name, pyo.Constraint(m.T, rule=rule))

    con("hvac_bal", lambda mm, t: Cool[t] - Heat[t] + (mm.HP2H[t] - mm.C2H[t] + mm.PCM_H2H[t] - mm.PCM_C2H[t]) == 0)
    con("pv_bal", lambda mm, t: PV[t] * PVSize == mm.PV2B[t] + mm.PV2H[t] + mm.PV2G[t] + mm.PV2E[t])
    con("house_e", lambda mm, t: mm.H2HP[t] + mm.H2C[t] == mm.PV2H[t] * eta_pv + mm.B2H[t] * eta + mm.G2H[t])
    con("house_crit", lambda mm, t: E_Load[t] == mm.PV2E[t] * eta_pv + mm.B2E[t] * eta + mm.G2E[t])
    con("bat_dyn", lambda mm, t: pyo.Constraint.Skip if t == N - 1 else
        mm.SOC_B[t + 1] == mm.SOC_B[t] * (1 - IP.BatteryLoss * dt) + dt * (mm.PV2B[t] * eta - mm.B2H[t] - mm.B2E[t]))
    con("bat_dis", lambda mm, t: dt * (mm.B2H[t] + mm.B2E[t]) <= mm.SOC_B[t])
    con("bat_chp", lambda mm, t: mm.PV2B[t] <= BSize * 0.25)
    con("bat_disp", lambda mm, t: mm.B2H[t] + mm.B2E[t] <= BSize * 0.25)
    con("bat_size", lambda mm, t: mm.SOC_B[t] <= BSize)
    con("bat_maxdis", lambda mm, t: mm.SOC_B[t] >= BSize * (1 - IP.MaxDischarge))
    con("heat_bal", lambda mm, t: mm.H2HP[t] == (mm.HP2PCM_H[t] + mm.HP2H[t]) / IP.COP_H)
    con("cool_bal", lambda mm, t: mm.H2C[t] == (mm.C2PCM_C[t] + mm.C2H[t]) / IP.COP_C)
    con("hp_cap", lambda mm, t: mm.H2C[t] + mm.H2HP[t] <= mm.HPSize)
    con("pcmh_dyn", lambda mm, t: pyo.Constraint.Skip if t == N - 1 else
        mm.SOC_PCM_H[t + 1] == mm.SOC_PCM_H[t] + dt * (mm.HP2PCM_H[t] - mm.PCM_H2H[t]))
    con("pcmc_dyn", lambda mm, t: pyo.Constraint.Skip if t == N - 1 else
        mm.SOC_PCM_C[t + 1] == mm.SOC_PCM_C[t] + dt * (mm.C2PCM_C[t] - mm.PCM_C2H[t]))
    con("pcmh_dis", lambda mm, t: dt * mm.PCM_H2H[t] <= mm.SOC_PCM_H[t])
    con("pcmc_dis", lambda mm, t: dt * mm.PCM_C2H[t] <= mm.SOC_PCM_C[t])
    con("pcmh_size", lambda mm, t: mm.SOC_PCM_H[t] <= PCMH)
    con("pcmc_size", lambda mm, t: mm.SOC_PCM_C[t] <= PCMC)

    m.b0 = pyo.Constraint(expr=m.SOC_B[0] == IP.Intial_B_SOC * BSize)
    m.bT = pyo.Constraint(expr=m.SOC_B[N - 1] == IP.Intial_B_SOC * BSize)
    m.h0 = pyo.Constraint(expr=m.SOC_PCM_H[0] == IP.Intial_PCM_H_SOC * PCMH)
    m.hT = pyo.Constraint(expr=m.SOC_PCM_H[N - 1] == IP.Intial_PCM_H_SOC * PCMH)
    m.c0 = pyo.Constraint(expr=m.SOC_PCM_C[0] == IP.Intial_PCM_C_SOC * PCMC)
    m.cT = pyo.Constraint(expr=m.SOC_PCM_C[N - 1] == IP.Intial_PCM_C_SOC * PCMC)

    solver = pyo.SolverFactory("gurobi")
    solver.options["Threads"] = 4
    res = solver.solve(m, tee=False)
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError("dispatch not optimal")
    return m, dt, dict(PV=PV, E_Load=E_Load, Cool=Cool, Heat=Heat,
                       DateTime=input_df["DateTime"].to_numpy(), PVSize=PVSize)


def extract(m, meta):
    N = m.NumTime
    g = lambda var: np.array([pyo.value(getattr(m, var)[t]) for t in range(N)])
    return pd.DataFrame({
        "DateTime": meta["DateTime"],
        "pv_gen_kw": meta["PV"] * meta["PVSize"],
        "e_load_kw": meta["E_Load"],
        "heating_load_kw": meta["Heat"],
        "cooling_load_kw": meta["Cool"],
        "hp_elec_kw": g("H2HP") + g("H2C"),
        "battery_soc_kwh": g("SOC_B"),
        "pcm_hot_soc_kwh": g("SOC_PCM_H"),
        "pcm_cold_soc_kwh": g("SOC_PCM_C"),
        "elec_unmet_kw": g("G2E"),
        "thermal_unmet_kw": g("G2H"),
    })


def main():
    FOB.locations = [CLIMATE]
    nested = FOB.build_input_data()
    input_df = nested[CLIMATE][YEAR]
    os.makedirs(OUT_DIR, exist_ok=True)
    for arch in ("PVB", "PCM"):
        caps = so_cvar_caps(arch)
        with FOB.hvac_voll_context(HVAC_VOLL):
            m, dt, meta = solve_dispatch(input_df, caps)
        out = os.path.join(OUT_DIR, f"dispatch_{CLIMATE}_{YEAR}_{arch}.csv")
        extract(m, meta).to_csv(out, index=False)
        print("wrote", out, "caps=", caps)


if __name__ == "__main__":
    main()
