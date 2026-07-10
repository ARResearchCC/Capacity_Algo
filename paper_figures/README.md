# FOB paper figures

Publication figures for the forward-operating-base (FOB) capacity-planning study:
PV + battery + phase-change thermal storage (PCM), sized by three methods
(**LP-Avg**, **LP-Worst**, **SO-CVaR**) and benchmarked against a diesel genset,
across 5 climates x 3 value-of-lost-load (VoLL) levels, evaluated **out-of-sample**
with 5-fold cross-validation over weather years 1998-2022.

## How to build

```powershell
# plotting env (matplotlib/seaborn/statsmodels) — NOT .venv (that is the model/Gurobi env)
.\.venv_verify\Scripts\python.exe -X utf8 paper_figures\<script>.py
```

Every figure script imports the two shared modules and writes vector **PDF** +
**600-dpi PNG** (fonts embedded, `pdf.fonttype=42`) plus a **CSV** of plotted
values and a **draft caption** `.txt`, into `paper_figures/main/` or
`paper_figures/si/`.

- **`style.py`** — the paper style: Okabe-Ito method palette (LP-Avg `#0072B2`,
  LP-Worst `#D55E00`, SO-CVaR `#009E73`, Diesel `#555555`), fixed method order,
  architecture encoding (No-PCM = solid / open marker; PCM = `///` hatch / filled
  marker), fixed cold→hot climate order + labels, single/double-column sizes, and
  `save_fig()`.
- **`data.py`** — tidy long-form loaders (`load_folds`, `load_summary`), unit
  normalization, `add_unmet_energy()` (recovers expected unmet energy kWh/yr and
  cost-excluding-penalty from the penalty terms), and `agg_folds()` for
  across-fold mean±SD. Architecture is tagged by source file.

**Representative operating point:** all main figures use the **Med** VoLL level
(HVAC $3/kWh, critical $300/kWh); VoLL sensitivity is `si_fig_voll`. All costs are
**annualized USD/yr**; test-fold values (per-year means) are primary, with ±1 SD
across the 5 folds as dispersion.

## Figure manifest

| Figure | Script | Source data | Status |
|--------|--------|-------------|--------|
| **fig1_workflow** (Fig 1b) | `fig1_workflow.py` | none (schematic) | ok |
| **fig2_capacities** (Fig 2) | `fig2_capacities.py` | `FOB_Sensitivity_Results.xlsx`, `FOB_PVB_Sensitivity_Results.xlsx` | ok |
| **fig3_total_cost** (Fig 3, headline) | `fig3_total_cost.py` | PCM + PVB workbooks | ok |
| **fig4_loss_of_load** (Fig 4) | `fig4_loss_of_load.py` | PCM + PVB workbooks | ok |
| **fig5_generalization** (Fig 5) | `fig5_generalization.py` | PCM + PVB workbooks | ok |
| **fig6_diesel_breakeven** (Fig 6) | `fig6_diesel_breakeven.py` | PCM + PVB + `FOB_Diesel_Results/…xlsx` | ok |
| **si_fig_frontier** (S1) | `si_fig_frontier.py` | PCM + PVB workbooks | ok |
| **si_fig_voll** (S2) | `si_fig_voll.py` | PCM + PVB workbooks | ok |
| **si_fig_risk_params** (S3) | `si_run_risk_sweep.py` (Sherlock array) → `si_collect_risk_sweep.py` → `si_fig_risk_params.py` | **needs generation** (no sweep in workbooks) | pending data |
| **si_fig_dispatch** (S4) | `si_run_dispatch_trace.py` → `si_fig_dispatch.py` | **needs generation** (hourly dispatch) | pending data |
| **si_fig_variability** (S5) | `si_fig_variability.py` | `Yearly_Results/locations_result.xlsx` | ok (1 panel omitted) |

## Unavailable metrics / gaps

- **S3 (λ, α)** — **justifies the fixed a-priori (λ=0.9, α=0.9)** used in the mains
  (not a per-scenario tuning). Nested cross-validation (5-yr weather blocks as groups):
  outer leave-one-block-out (TEST) × inner leave-one-block-out within the 20 dev years
  (TRAIN 15 yr fits capacities, VALIDATION 5 yr selects (λ,α)); test never used for
  selection → unbiased. Full design space: **5 climates × 3 VoLL** × λ∈{0,.25,.5,.75,.9,1}
  × α∈{.8,.9,.95} (270 tasks). Selection metric = validation total cost. See
  `risk_parameter_justification.md` for the a-priori rationale + citations. **Run on
  Sherlock** (`bash sherlock/submit_risk_sweep.sh` → array `si_run_risk_sweep.py`,
  25 solves/task, then `si_collect_risk_sweep.py`). Pull
  `Risk_Sweep_Results/risk_sweep_summary.xlsx` and plot with `si_fig_risk_params.py`:
  (a) pooled relative-regret surface (flat basin), (b) fixed-vs-tuned nested-CV test
  gap per cell (≈0 ⇒ no selection advantage — the reviewer rebuttal), (c) per-cell
  optima clustering near (0.9, 0.9).
- **S4 (dispatch trace)** — hourly dispatch is not stored in the workbooks.
  `si_run_dispatch_trace.py` re-solves the hourly model (mirrors `Simulate.py`) for the
  SO-CVaR design and writes traces; **requires the model env** (Gurobi). Then
  `si_fig_dispatch.py` plots the PCM vs no-PCM window. *These two scripts are written
  but were NOT executed/verified here (no solver in the plotting env).*
- **S5 peak electrical load** — omitted; `Yearly_Results` exports only annual energy
  totals, not peak power. The PV-yield / heating / cooling panels are shown.
- **Diesel workbook fuel price** — the shipped diesel workbook was generated at a stale
  $4/gal and lacks the fuel-price sweep. Fig 6 sidesteps this by recomputing diesel cost
  price-independently (`fixed_cost + gallons·p`), which is valid. If a diesel bar at a
  fixed price is ever needed, re-run `python FOB_Diesel.py` (model env) to regenerate the
  $100/$400/$600/gal sweep.

## QA (grayscale + colourblind)

- **CVD simulation** (Machado 2009, severity 1.0): the method palette clears the ΔE≥12
  target under normal, deuteranopia, protanopia, and tritanopia (worst case ~16-17,
  tritanopia); the 4-asset palette also passes.
- **Grayscale/print:** architecture reads from hatch (bars) or open/filled markers
  (scatter/dumbbell); method reads from fixed order + text labels — never colour alone.
- All figures use `style.py` (identical fonts, sizes, palette, method order, climate
  labels) and are legible at their intended print width.

## Key quotable results (Med VoLL, out-of-sample)

- **Cost (Fig 3):** SO-CVaR is cheaper than LP-Worst in every climate (0.1–3.3%),
  within ~1.5% of the cost-minimizing LP-Avg, and beats LP-Avg for PCM in Marine
  (−3.9%) and Tropical (−2.0%).
- **Reliability (Fig 4):** electrical loss of load is negligible everywhere
  (≤0.7 kWh/yr); SO-CVaR cuts thermal unmet energy vs LP-Avg by 9–68% across cells.
- **Generalization (Fig 5):** pooled train→test reliability gap is smallest for
  SO-CVaR (+7.4%) vs LP-Avg (+17.2%, overfits) and LP-Worst (−51%, over-conservative).
- **Diesel (Fig 6):** breakeven delivered diesel price is 1.6–9.3 $/gal across
  climates — far below the $100–600/gal fully-burdened resupply band, so the renewable
  microgrid wins at any realistic FOB fuel price.
- **Sizing (Fig 2):** LP-Worst builds the largest systems; adding PCM cuts battery
  sizing ~64% (43.4 → 15.8 kWh mean) while leaving PV essentially unchanged.
