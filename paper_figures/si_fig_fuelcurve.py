"""
si_fig_fuelcurve.py — SI figure: the diesel generator fuel curve as used in the model.

Hourly fuel use follows the standard load-dependent linear (HOMER/NREL) form
    fuel_rate = F0 * GenSize + F1 * P_out   [gal/h]
with coefficients from Input_Parameters.py (F0 = 0.0215 gal/h per kW rated,
F1 = 0.065 gal/h per kW output). Plotted normalised per kW of rated capacity so the
single line holds for any generator size.

Run:  .\\.venv_verify\\Scripts\\python.exe paper_figures\\si_fig_fuelcurve.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import style as S

F0 = 0.0215   # gal/h per kW rated  (no-load / idle intercept)
F1 = 0.0650   # gal/h per kW output (marginal slope)
DIESEL = S.METHOD_COLOR["Diesel"]

x = np.linspace(0.0, 1.0, 300)          # load fraction = P_out / GenSize

fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot(x, F0 + F1 * x, color=DIESEL, lw=2.0, zorder=3)
ax.axhline(F0, ls=":", color="0.6", lw=1)
ax.scatter([0.0, 1.0], [F0, F0 + F1], color=DIESEL, s=24, zorder=5)

ax.text(0.03, 0.096, r"$\mathrm{fuel\ rate} = F_0\,\mathrm{GenSize} + F_1\,P_\mathrm{out}$",
        fontsize=8, color="0.15", va="top")
ax.annotate(f"idle intercept  $F_0$ = {F0}", xy=(0.0, F0), xytext=(0.13, 0.004),
            fontsize=7, color="0.25", ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
ax.text(0.46, F0 + F1 * 0.46 + 0.005, f"slope  $F_1$ = {F1}", fontsize=7,
        color=DIESEL, rotation=15, rotation_mode="anchor", ha="left", va="bottom")
ax.text(1.0, F0 + F1 + 0.003, f"full load\n{F0 + F1:.3f}", fontsize=7,
        ha="right", va="bottom", color=DIESEL)

ax.set_xlim(0, 1.03)
ax.set_ylim(0, 0.10)
ax.set_xlabel("Load fraction  (output / rated capacity)")
ax.set_ylabel(r"Fuel rate  (gal h$^{-1}$ per kW rated)")
S.despine(ax)
fig.tight_layout()

out = pd.DataFrame({"load_fraction": x,
                    "fuel_rate_gal_per_h_per_kW_rated": F0 + F1 * x})
caption = (
    "Diesel generator fuel curve used in the model. Hourly fuel use follows the standard "
    "load-dependent linear (HOMER/NREL) form, fuel rate = F0 x GenSize + F1 x P_out (gal/h), "
    "shown normalised per kW of rated capacity against the load fraction (output / rated "
    "capacity). The no-load (idle) intercept F0 = 0.0215 gal/h per kW rated is consumed "
    "whenever the generator runs; the marginal slope F1 = 0.065 gal/h per kW is the extra fuel "
    "per kW of output, reaching 0.086 gal/h per kW at full load. The nonzero intercept encodes "
    "part-load inefficiency, so a lightly loaded generator burns more fuel per kWh delivered "
    "than a fully loaded one.")
S.save_fig(fig, "si_fig_fuelcurve", section="si", data=out, caption=caption)
