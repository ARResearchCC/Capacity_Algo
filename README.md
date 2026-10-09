# Capacity_Algo — risk-aware capacity planning for fully renewable islanded microgrids

Code and data accompanying the paper

> Y. F. Fan, M. Jiang, D. J. Sambor, A. Mühlbauer, J. G. Ferguson, S. M. Dutton, M. Z. Jacobson (2026).
> **Risk-Aware Capacity Planning and Phase-Change-Material Storage for Maintaining Reliability
> at Low Cost in Fully-Renewable Islanded Microgrids.** _Journal / DOI to be added on publication._

The study sizes a fully renewable, islanded microgrid for a military forward operating base
(PV + battery + heat pump, with and without hot/cold phase-change-material (PCM) thermal
storage) and benchmarks it against a diesel genset. Systems are sized by three methods —
**LP-Avg** (average training year), **LP-Worst** (worst training year), and **SO-CVaR**
(two-stage stochastic program with a conditional-value-at-risk term) — and evaluated
**out of sample** by 5-fold cross-validation over the weather years 1998–2022, across
**5 climates × 2 architectures × 3 value-of-lost-load (VoLL) levels**.

---

## Repository layout

| Path | Contents |
|------|----------|
| `FOB.py`, `FOB_PVB.py`, `FOB_Diesel.py` | Run drivers: PCM architecture, PV+battery architecture, diesel benchmark |
| `SO_CVaR.py`, `SO_CVaR_PVB.py` | Risk-averse (CVaR) stochastic capacity models |
| `Baseline_CO.py`, `Baseline_CO_PVB.py` | Deterministic LP-Avg / LP-Worst capacity models |
| `Simulate.py`, `Diesel_Model.py` | Fixed-capacity hourly dispatch; diesel dispatch and cost |
| `Solar_Generation.py`, `Passive_Model.py`, `Electrical_Load.py` | Input models: PV output, building thermal load, electrical load |
| `Data_Conversion.py`, `Gather_input_Locations.py`, `Utility_functions.py` | NSRDB ingestion and input-time-series assembly |
| `Input_Parameters.py`, `Calibration_Model_Input.xlsx` | Physical, economic, VoLL and CVaR parameters; building-model calibration inputs |
| `Data/` | NSRDB weather, 5 sites × 25 years (1998–2022) |
| `FOB_Sensitivity_Results.xlsx`, `FOB_PVB_Sensitivity_Results.xlsx` | Result workbooks (PCM, PV+battery) |
| `FOB_Diesel_Results/` | Diesel-benchmark result workbook |
| `Risk_Sweep_Results/` | (λ, α) risk-parameter sweep: per-task partials and the collected summary |
| `Yearly_Results/` | Per-site, per-year input summaries (interannual variability) |
| `paper_figures/` | Figure scripts and outputs (`main/` = Figs 1–5, `si/` = SI figures) |
| `sherlock/` | SLURM scripts used to run the models on Stanford's Sherlock cluster |

All result workbooks are committed, so **every figure can be rebuilt without an
optimization solver** (Section 1). Re-running the models (Section 2) needs Gurobi.

---

## 1. Rebuild the figures (no solver required)

```powershell
# plotting environment
py -3 -m venv .venv_verify
.\.venv_verify\Scripts\Activate.ps1
pip install matplotlib==3.9.2 pandas==2.2.3 numpy==1.26.4 seaborn==0.13.2 statsmodels==0.14.4 openpyxl==3.1.5

# build any figure: writes PDF + 600-dpi PNG + CSV of plotted values + draft caption
python -X utf8 paper_figures\fig3_calibration.py
```

On macOS / Linux use `python3 -m venv .venv_verify` and `source .venv_verify/bin/activate`.

| Paper figure | Script | Output |
|--------------|--------|--------|
| Fig. 1 — workflow | `paper_figures/fig1_workflow.py` | `main/fig1_workflow` |
| Fig. 2 — out-of-sample annual total system cost | `paper_figures/fig2_total_cost.py` | `main/fig2_total_cost` |
| Fig. 3 — reliability calibration | `paper_figures/fig3_calibration.py` | `main/fig3_calibration` |
| Fig. 4 — tail reliability and cost stability | `paper_figures/fig4_risk.py` | `main/fig4_risk` |
| Fig. 5 — diesel break-even fuel price | `paper_figures/fig5_diesel_breakeven.py` | `main/fig5_diesel_breakeven` |
| SI — interannual variability of inputs | `paper_figures/si_fig_variability.py` | `si/si_fig_variability` |
| SI — optimal capacities | `paper_figures/si_fig_capacities.py` | `si/si_fig_capacities` |
| SI — VoLL sensitivity | `paper_figures/si_fig_voll.py` | `si/si_fig_voll` |
| SI — CVaR vs risk-neutral stochastic optimization (λ = 0) | `paper_figures/si_fig_riskterms.py` | `si/si_fig_riskterms` |
| SI — loss of load (thermal and electrical) | `paper_figures/si_fig_loss_of_load.py` | `si/si_fig_loss_of_load` |
| SI — cost–reliability frontier | `paper_figures/si_fig_frontier.py` | `si/si_fig_frontier` |

Shared style and data loaders are in `paper_figures/style.py` and `paper_figures/data.py`;
see `paper_figures/README.md`.

---

## 2. Re-run the models (requires Gurobi)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`gurobipy` installs from PyPI but needs a valid Gurobi licence (free academic licences are
available); all optimization uses the `gurobi` solver through Pyomo. Run from the repository
root so `Data/` and `Calibration_Model_Input.xlsx` are found.

| Step | Command | Output |
|------|---------|--------|
| Assemble input time series | `python Gather_input_Locations.py` | `Yearly_Results/locations_result.xlsx` |
| PCM architecture | `python FOB.py` | `FOB_Sensitivity_Results.xlsx` |
| PV+battery architecture | `python FOB_PVB.py` | `FOB_PVB_Sensitivity_Results.xlsx` |
| Diesel benchmark | `python FOB_Diesel.py` | `FOB_Diesel_Results/FOB_Diesel_Sensitivity_Results.xlsx` |
| (λ, α) risk sweep | `paper_figures/si_run_risk_sweep.py` (SLURM array, see `sherlock/README.md`) → `paper_figures/si_collect_risk_sweep.py` | `Risk_Sweep_Results/partials/*.csv` → `Risk_Sweep_Results/risk_sweep_summary.xlsx` |

Each driver covers 5 locations × 25 years, 5-fold cross-validation, 3 VoLL levels and the
3 methods, so expect a long runtime; the paper's runs were made on a SLURM cluster
(`sherlock/`). To run a single method, edit `algorithms = [...]` at the top of the driver.
Then rebuild the figures as in Section 1.

**Result workbook structure:** `Config` · `VoLL_Scenarios` · `Summary` (fold-mean by
location / VoLL / method) · `Fold_1` … `Fold_5` (per-fold detail). SO-CVaR rows add three
training columns (`Expected Outage Cost`, `CVaR Outage Cost`, `CVaR_eta`).

---

## 3. Key settings

All parameters live in `Input_Parameters.py`; the modelling choices are documented in the
paper's Methods and Supplementary Information. Defaults: project lifetime 20 yr, discount
rate 3 % (capital recovery factor ≈ 0.067); CVaR α = 0.9 and λ = 0.9, fixed a priori
(λ = 0 recovers risk-neutral stochastic optimization). VoLL levels (thermal / critical
electrical, $/kWh): Low 1 / 30, Med 3 / 100, High 10 / 300.

## Data

Weather inputs are from the U.S. National Solar Radiation Database (NSRDB; Sengupta et al.
2018, <https://nsrdb.nrel.gov/>) for five sites spanning distinct Köppen–Geiger zones
(Alaska, Minnesota, California, Arizona, Florida), 1998–2022, stored in `Data/`.

## Citation

If you use this code, please cite the paper above and the repository (also available via
GitHub's "Cite this repository", from `CITATION.cff`):

```bibtex
@misc{capacityalgo,
  author       = {Fan, Yuanbei F. and Jiang, Muyan and Sambor, Daniel J. and M\"uhlbauer, Andreas and Ferguson, Jill G. and Dutton, Spencer M. and Jacobson, Mark Z.},
  title        = {{Capacity\_Algo}: capacity-planning and dispatch-optimisation code for fully-renewable islanded microgrids},
  year         = {2026},
  version      = {v1.0.0},
  howpublished = {GitHub repository},
  url          = {https://github.com/ARResearchCC/Capacity_Algo}
}
```

## License

The code is released under the [MIT License](LICENSE). The NSRDB weather data in `Data/`
are public data from the U.S. National Renewable Energy Laboratory and remain subject to
NREL's terms of use.
