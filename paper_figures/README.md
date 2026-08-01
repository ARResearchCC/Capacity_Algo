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

Numbering matches the manuscript (calibration-lead lineup). Main-text files live in
`main/` and are numbered `fig1`…`fig6` in paste order; everything else is in `si/`.

**Main text (Fig 1–6):**

| Figure | Script | Source data | Status |
|--------|--------|-------------|--------|
| **Fig 1** — workflow (`fig1_workflow`) | `fig1_workflow.py` | none (schematic) | ok |
| **Fig 2** — out-of-sample cost parity (`fig2_total_cost`) | `fig2_total_cost.py` | PCM + PVB workbooks | ok |
| **Fig 3** — reliability calibration (HERO) (`fig3_calibration`) | `fig3_calibration.py` | PCM + PVB workbooks | ok |
| **Fig 4** — risk: tail unmet + cost variance (`fig4_risk`) | `fig4_risk.py` | PCM + PVB workbooks | ok |
| **Fig 5** — risk term vs risk-neutral λ=0 (`fig5_riskterm`) | `fig5_riskterm.py` | risk-sweep λ=0 vs λ=0.9 (PCM, 15 cells) | ok |
| **Fig 6** — diesel break-even (`fig6_diesel_breakeven`) | `fig6_diesel_breakeven.py` | PCM + PVB + `FOB_Diesel_Results/…xlsx` | ok |

**Supporting information (SI):**

| Figure | Script | Source data | Status |
|--------|--------|-------------|--------|
| **si_fig_capacities** — sized capacities | `si_fig_capacities.py` | `FOB_Sensitivity_Results.xlsx`, `FOB_PVB_Sensitivity_Results.xlsx` | ok |
| **si_fig_loss_of_load** — loss of load (thermal + worst event) | `si_fig_loss_of_load.py` | PCM + PVB workbooks | ok |
| **si_fig_generalization** — train→test dumbbell (superseded by Fig 3) | `si_fig_generalization.py` | PCM + PVB workbooks | ok |
| **si_fig_frontier** — cost–unmet frontier | `si_fig_frontier.py` | PCM + PVB workbooks | ok |
| **si_fig_voll** — VoLL sensitivity | `si_fig_voll.py` | PCM + PVB workbooks | ok |
| **si_fig_risk_params** — risk-parameter robustness | `si_run_risk_sweep.py` (Sherlock array) → `si_collect_risk_sweep.py` → `si_fig_risk_params.py` | **needs generation** (no sweep in workbooks) | pending data |
| **si_fig_dispatch** — dispatch trace | `si_run_dispatch_trace.py` → `si_fig_dispatch.py` | **needs generation** (hourly dispatch) | pending data |
| **si_fig_variability** — interannual variability | `si_fig_variability.py` | `Yearly_Results/locations_result.xlsx` | ok (1 panel omitted) |

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

- **Cost parity (Fig 2):** SO-CVaR is cheaper than LP-Worst in every climate (0.1–3.3%),
  within ~1.5% of the cost-minimizing LP-Avg — reliability at *comparable* cost, not cheaper.
- **Calibration (Fig 3, HERO):** median test/plan unmet ratio SO-CVaR 1.02, LP-Avg 1.11
  (under-provisions), LP-Worst 0.51 (over-provisions); best-calibrated in 26/30 cells.
- **Risk (Fig 4):** vs LP-Avg, SO-CVaR cuts worst-fold unmet −21% (30/30) and across-fold
  cost variance −31% (30/30); the tail cut grows with VoLL.
- **Risk term (Fig 5):** vs the risk-neutral SO (λ=0), SO-CVaR lowers mean OOS unmet −40%
  (15/15) and tail −19% (15/15) for a +1.2% cost premium (PCM, 15 cells).
- **Diesel (Fig 6):** breakeven delivered diesel price is 0.78–8.90 $/gal across climates —
  below even the routine forward-delivery band, so the renewable microgrid wins at any
  realistic FOB fuel price.
- **Sizing (SI `si_fig_capacities`):** LP-Worst builds the largest systems; adding PCM
  displaces battery 38–86% while leaving PV essentially unchanged.
- **Loss of load / generalization (SI `si_fig_loss_of_load`, `si_fig_generalization`):**
  electrical loss of load is negligible (≤0.7 kWh/yr); pooled train→test unmet gap is
  smallest for SO-CVaR (+7.6%) vs LP-Avg (+17.2%, overfits) and LP-Worst (−51.4%).
