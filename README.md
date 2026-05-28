# Capacity_Algo

CVaR-based capacity planning (`SO_CVaR.py`) for two building scenarios. Use **`FOB.py`** for the Forward Operating Base case and **`DC.py`** for the data center case.

`SO_CVaR.py` is a library module (no `main`); both drivers call `SO_CVaR.SO_CVaR_training()` when the method is `SO_CVaR`.

---

## 1. Virtual environment

From your current folder that looks something like this: (base) PS C:\...\Capacity_Algo>

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Gurobi:** `gurobipy` installs from PyPI, but you need a valid Gurobi license on the machine. All optimization scripts use the `gurobi` solver.

**Data before running:** keep `Data/` (NSRDB weather) and `Calibration_Model_Input.xlsx` in the project folder.

---

## 2. Run SO_CVaR analysis

| Scenario | Script | Main output |
|----------|--------|-------------|
| FOB | `python FOB.py` | `FOB_Results/FOB_Sensitivity_Results.xlsx` |
| DC | `python DC.py` | `DC_Results/DC_Sensitivity_Results.xlsx` |

Each run builds weather/load inputs for 5 locations x 25 years (1998-2022), 5-fold cross-validation, 3 VoLL levels, and 3 methods (`LP_Avg`, `LP_Worst`, `SO_CVaR`). Expect a long runtime.

To run **only** SO_CVaR (fewer solves), edit the top of `FOB.py` or `DC.py`:

```python
algorithms = ["SO_CVaR"]
```

---

## 3. Parameters (edit in `FOB.py` / `DC.py`)

| Parameter | Meaning |
|-----------|---------|
| `locations` | Climate sites: California, Arizona, Alaska, Minnesota, Florida |
| `weather_year_list` | Training pool of weather years (default 1998-2022) |
| `fold` | Number of CV folds (default 5; each fold holds out 5 test years) |
| `VOLL_SCENARIOS` | Low / Med / High **value of lost load** ($/kWh) for HVAC and critical electrical load |
| `CVAR_ALPHA` | CVaR confidence in (0, 1); tail is worst `(1 - alpha)` share of training years (default from `Input_Parameters.CVaR_alpha`, 0.9) |
| `CVAR_LAMBDA` | 0 = minimize expected outage cost only; 1 = pure CVaR on outage cost (default 1.0) |
| `capacity_costs` | `[C_PV, C_PV_OP, C_B, C_B_OP]` ($/kW and $/kWh, capital and O&M); defaults from `Input_Parameters.py` |
| `data_dir` | Folder with NSRDB files (`Data`) |

**Scenario-specific defaults** (in `Input_Parameters.py`, used inside `SO_CVaR.py`):

| | FOB (`FOB.py`) | DC (`DC.py`) |
|--|----------------|--------------|
| Building type | Forward operating base | Data center |
| Heat pump size | `HPSize` (10 kW) | `HPSize_DC` (100 kW) |
| Med HVAC VoLL | `HVAC_lol_cost` ($3/kWh) | `HVAC_lol_cost_DC` ($10/kWh) |
| Med critical VoLL | `lossofloadcost` ($300/kWh) | `lossofloadcost_DC` ($30/kWh) |

---

## 4. Output

### Excel (`FOB_Sensitivity_Results.xlsx` / `DC_Sensitivity_Results.xlsx`)

| Sheet | Contents |
|-------|----------|
| `Config` | Run settings (scenario, CVaR, folds, methods) |
| `VoLL_Scenarios` | Low / Med / High VoLL values used |
| `Summary` | Mean over folds by location, VoLL level, and method |
| `Fold_1` ... `Fold_5` | Per-fold detail rows |

**Important columns for `SO_CVaR` rows:**

| Column | Meaning |
|--------|---------|
| `PV_Size`, `Battery_Size`, `PCM_Heating_Size`, `PCM_Cooling_Size` | Optimal capacities (kW or kWh) from training |
| `Training Total Cost` | Annualized objective on training years |
| `Training Expected Outage Cost` | Mean VoLL-weighted outage cost across training years |
| `Training CVaR Outage Cost` | CVaR of outage cost (tail risk) |
| `Training CVaR_eta` | CVaR auxiliary variable (VaR threshold) |
| `Testing Total Cost` | Simulated cost on held-out test years with fixed training capacities |
| `Testing HVAC LoL Hours`, `Testing Critical LoL Hours` | Loss-of-load duration on test years |

`LP_Avg` / `LP_Worst` rows omit the three CVaR training columns.
