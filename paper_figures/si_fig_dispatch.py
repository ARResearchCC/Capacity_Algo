"""
si_fig_dispatch.py — SI figure: representative dispatch trace, PCM vs no-PCM.

Reads the hourly traces written by si_run_dispatch_trace.py and plots a stressed
multi-day window in two columns (no-PCM | PCM) sharing a time axis: PV generation
and load, battery SOC, hot/cold PCM SOC, and served-vs-unmet electrical/thermal
load — so the reader sees PCM sustaining thermal service through the shortage.

If the trace CSVs are absent, prints how to generate them (needs .venv + Gurobi).
Run:  .\\.venv_verify\\Scripts\\python.exe paper_figures\\si_fig_dispatch.py
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
TRACE_DIR = os.path.join(ROOT, "Dispatch_Trace_Results")
CLIMATE, YEAR = "Alaska", 2012
WINDOW_DAYS = 7


def _path(arch):
    return os.path.join(TRACE_DIR, f"dispatch_{CLIMATE}_{YEAR}_{arch}.csv")


def _pick_window(df):
    """Center a WINDOW_DAYS window on the worst thermal-unmet stretch."""
    h = WINDOW_DAYS * 24
    tu = df["thermal_unmet_kw"].to_numpy()
    if tu.sum() <= 0:
        lo = max(0, len(df) // 2 - h // 2)         # fallback: mid-year
    else:
        roll = pd.Series(tu).rolling(h, min_periods=1).sum().to_numpy()
        lo = max(0, int(np.argmax(roll)) - h)
    return lo, min(len(df), lo + h)


def main():
    if not all(os.path.exists(_path(a)) for a in ("PVB", "PCM")):
        print("MISSING dispatch traces in", TRACE_DIR)
        print("Generate them first with the model env:\n  "
              ".\\.venv\\Scripts\\python.exe paper_figures\\si_run_dispatch_trace.py")
        return

    traces = {a: pd.read_csv(_path(a), parse_dates=["DateTime"]) for a in ("PVB", "PCM")}
    lo, hi = _pick_window(traces["PVB"])
    t = np.arange(hi - lo) / 24.0   # days from window start

    fig, axes = plt.subplots(4, 2, figsize=S.figsize_double(6.2), sharex=True)
    for j, arch in enumerate(("PVB", "PCM")):
        d = traces[arch].iloc[lo:hi].reset_index(drop=True)
        axes[0, j].set_title(S.ARCH_LABEL[arch], fontweight="bold")

        # row 0: PV + loads
        ax = axes[0, j]
        ax.fill_between(t, d.pv_gen_kw, color="#E69F00", alpha=0.5, label="PV gen")
        ax.plot(t, d.e_load_kw, color="#0072B2", lw=1.0, label="Critical load")
        ax.plot(t, d.hp_elec_kw, color="#CC3311", lw=1.0, label="Heat-pump elec")
        ax.set_ylabel("Power\n[kW]")

        # row 1: battery SOC
        axes[1, j].plot(t, d.battery_soc_kwh, color="#009E73", lw=1.2)
        axes[1, j].set_ylabel("Battery\n[kWh]")

        # row 2: PCM SOC
        ax = axes[2, j]
        ax.plot(t, d.pcm_hot_soc_kwh, color="#D55E00", lw=1.2, label="Hot PCM")
        ax.plot(t, d.pcm_cold_soc_kwh, color="#56B4E9", lw=1.2, label="Cold PCM")
        ax.set_ylabel("PCM\n[kWh$_t$]")
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

        # row 3: unmet load (the shortage)
        ax = axes[3, j]
        ax.fill_between(t, d.thermal_unmet_kw, color="#D55E00", alpha=0.6, label="Thermal unmet")
        ax.fill_between(t, d.elec_unmet_kw, color="#0072B2", alpha=0.6, label="Electrical unmet")
        ax.set_ylabel("Unmet\n[kW]")
        ax.set_xlabel("Days into stressed window")
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

        for i in range(4):
            S.despine(axes[i, j])
    axes[0, 0].legend(fontsize=6, loc="upper right")

    fig.suptitle(f"Dispatch through a stressed window — {S.CLIMATE_LABEL[CLIMATE]}, "
                 f"{YEAR} (SO-CVaR design)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    caption = (f"SI: hourly dispatch over a {WINDOW_DAYS}-day stressed window in "
               f"{S.CLIMATE_LABEL[CLIMATE]} ({YEAR}) for the SO-CVaR design, no-PCM "
               "(left) vs PCM (right). PV generation and load (top), battery and PCM "
               "state of charge (middle), and unmet electrical/thermal load (bottom). "
               "PCM storage carries thermal service through the low-solar shortage, "
               "eliminating the thermal loss-of-load visible in the no-PCM column.")
    # export the plotted window values
    out = pd.concat({a: traces[a].iloc[lo:hi] for a in ("PVB", "PCM")}, names=["architecture"])
    S.save_fig(fig, "si_fig_dispatch", section="si", data=out.reset_index(level=0),
               caption=caption)


if __name__ == "__main__":
    main()
