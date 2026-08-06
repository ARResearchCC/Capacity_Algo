# Risk-aware capacity planning for a fully renewable islanded microgrid

Code and data for the study of **risk-aware (CVaR) capacity planning with phase-change thermal
storage (PCM)** for a fully renewable, islanded forward-operating-base (FOB) microgrid
(PV + battery + heat pump, with and without hot/cold PCM), benchmarked against a diesel genset.

Systems are sized by three methods — **LP-Avg** (average training year), **LP-Worst** (worst
training year), and **SO-CVaR** (risk-averse two-stage stochastic program) — and evaluated
**out-of-sample** with 5-fold cross-validation over the weather years 1998–2022, across
**5 climates × 2 architectures × 3 value-of-lost-load (VoLL) levels**.

> 📄 Paper: _<add DOI / link here>_

---

## Repository layout

| Path | Contents |
|------|----------|
| `FOB.py`, `FOB_PVB.py`, `FOB_Diesel.py` | Run drivers — PCM, PV+battery, and diesel-benchmark cases |
| `SO_CVaR.py`, `SO_CVaR_PVB.py` | Risk-averse (CVaR) stochastic capacity models |
| `Baseline_CO.py`, `Baseline_CO_PVB.py` | Deterministic LP-Avg / LP-Worst capacity models |
| `Simulate.py`, `Diesel_Model.py` | Fixed-capacity hourly dispatch; diesel dispatch/cost |
| `Solar_Generation.py`, `Passive_Model.py`, `Electrical_Load.py` | Input models: PV, building thermal, electrical load |
| `Data_Conversion.py`, `Gather_input_Locations.py`, `Utility_functions.py` | NSRDB ingestion and input-timeseries assembly |
| `Input_Parameters.py` | All physical / economic / VoLL parameters |
| `Data/` | NSRDB weather, 5 sites × 25 years (1998–2022) |
| `FOB_Sensitivity_Results.xlsx`, `FOB_PVB_Sensitivity_Results.xlsx` | Pre-computed result workbooks (PCM, PV+battery) |
| `FOB_Diesel_Results/`, `Risk_Sweep_Results/`, `Yearly_Results/` | Diesel, (λ,α) risk-sweep, and input-variability results |
| `paper_figures/` | Figure pipeline (`main/` = Fig 1–6, `si/` = SI figures) + captions + plotted-value CSVs |
| `manuscript/` | Paper text (Methods, Results, Discussion, SI, …) |
| `sherlock/` | SLURM scripts for the (λ,α) risk-parameter sweep |

The pre-computed workbooks are committed, so **the figures can be rebuilt without a solver**.

---

## 1. Rebuild the figures (no solver required)

The figure scripts read the committed result workbooks — no Gurobi needed.

```powershell
# plotting environment (matplotlib / seaborn / statsmodels)
py -3 -m venv .venv_verify
.\.venv_verify\Scripts\Activate.ps1
pip install matplotlib pandas numpy seaborn statsmodels openpyxl

# build any figure (writes PDF + 600-dpi PNG + CSV + draft caption)
.\.venv_verify\Scripts\python.exe -X utf8 paper_figures\fig3_calibration.py
```

Main-text figures and their scripts:

| Figure | Script |
|--------|--------|
| Fig 1 — workflow | `paper_figures/fig1_workflow.py` |
| Fig 2 — out-of-sample cost | `paper_figures/fig2_total_cost.py` |
| Fig 3 — reliability calibration (lead result) | `paper_figures/fig3_calibration.py` |
| Fig 4 — risk (tail + cost stability) | `paper_figures/fig4_risk.py` |
| Fig 5 — risk term vs risk-neutral λ=0 | `paper_figures/fig5_riskterm.py` |
| Fig 6 — diesel break-even | `paper_figures/fig6_diesel_breakeven.py` |

SI figures are the `paper_figures/si_fig_*.py` scripts (capacities, VoLL sensitivity, risk-parameter
robustness, generalization, loss of load, cost–reliability frontier, interannual variability).
See `paper_figures/README.md` for the full manifest and style conventions.

---

## 2. Re-run the model (requires Gurobi)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Gurobi:** `gurobipy` installs from PyPI but needs a valid Gurobi licence; all optimisation uses
the `gurobi` solver. Keep `Data/` and `Calibration_Model_Input.xlsx` in the project root.

| Step | Command | Output |
|------|---------|--------|
| Assemble input time series | `python Gather_input_Locations.py` | `Yearly_Results/locations_result.xlsx` |
| PCM architecture | `python FOB.py` | `FOB_Sensitivity_Results.xlsx` |
| PV+battery architecture | `python FOB_PVB.py` | `FOB_PVB_Sensitivity_Results.xlsx` |
| Diesel benchmark | `python FOB_Diesel.py` | `FOB_Diesel_Results/…xlsx` |
| (λ,α) risk sweep | `sherlock/` SLURM array → collect | `Risk_Sweep_Results/risk_sweep_summary.xlsx` |

Each run covers 5 locations × 25 years, 5-fold CV, 3 VoLL levels, and the 3 methods (expect a long
runtime). To run only SO-CVaR, set `algorithms = ["SO_CVaR"]` at the top of the driver. Then rebuild
the figures with the plotting environment (Section 1).

### Result workbook structure
`Config` · `VoLL_Scenarios` · `Summary` (fold-mean by location/VoLL/method) · `Fold_1…Fold_5`
(per-fold detail). SO-CVaR rows add three training columns (`Expected Outage Cost`,
`CVaR Outage Cost`, `CVaR_eta`); LP rows omit them.

---

## 3. Key settings

Physical, economic, VoLL, and CVaR parameters live in `Input_Parameters.py`; the modelling choices
are documented in `manuscript/` (Methods and SI). Defaults: lifetime 20 yr, discount rate 3%
(CRF ≈ 0.067); CVaR α = 0.9, λ = 0.9 (fixed a priori; λ = 0 recovers risk-neutral stochastic
optimisation). VoLL levels (thermal / critical $/kWh): Low $1/$30, Med $3/$100, High $10/$300.

## Data

Weather is from the U.S. National Solar Radiation Database (NSRDB), 5 sites spanning distinct
Köppen–Geiger zones (Alaska, Minnesota, California, Arizona, Florida), 1998–2022, in `Data/`.
Source: <https://nsrdb.nrel.gov/>.

## Citation & licence

_<add citation (BibTeX) and a licence (e.g. MIT for code) here before publishing>_
