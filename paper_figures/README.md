# Paper figures

Figure scripts for the paper. Each script reads the committed result workbooks (no
optimization solver needed) and writes, into `main/` or `si/`:

- `<name>.pdf` — vector figure (fonts embedded),
- `<name>.png` — 600-dpi raster,
- `<name>.csv` — the plotted values,
- `<name>.txt` — a draft caption.

```powershell
# from the repository root, in the plotting environment (see the top-level README)
python -X utf8 paper_figures\<script>.py
```

## Shared modules

- **`style.py`** — publication style used by every figure: Okabe–Ito method palette
  (LP-Avg `#0072B2`, LP-Worst `#D55E00`, SO-CVaR `#009E73`, Diesel `#555555`), fixed
  method order, architecture encoding (PV+battery = solid fill / open marker; with PCM =
  `///` hatch / filled marker), fixed cold → hot climate order and labels, figure sizes,
  and `save_fig()`.
- **`data.py`** — tidy long-form loaders for the result workbooks (`load_folds`,
  `load_summary`), `add_unmet_energy()` (recovers expected unmet energy, kWh/yr, from the
  VoLL penalty terms) and `agg_folds()` (across-fold mean ± SD). The architecture is
  tagged by source workbook: PCM → `FOB_Sensitivity_Results.xlsx`, PV+battery →
  `FOB_PVB_Sensitivity_Results.xlsx`, diesel → `FOB_Diesel_Results/…`.

Unless stated otherwise, figures use the **Med** VoLL level (thermal $3/kWh, critical
electrical $100/kWh), annualized USD/yr, and out-of-sample (test-split) values with ±1 SD
across the 5 cross-validation folds.

## Manifest

**Main text**

| Figure | Script | Source data |
|--------|--------|-------------|
| Fig. 1 — workflow schematic | `fig1_workflow.py` | none |
| Fig. 2 — out-of-sample annual total system cost | `fig2_total_cost.py` | PCM + PV+battery workbooks |
| Fig. 3 — reliability calibration | `fig3_calibration.py` | PCM + PV+battery workbooks |
| Fig. 4 — tail reliability and cost stability | `fig4_risk.py` | PCM + PV+battery workbooks |
| Fig. 5 — diesel break-even fuel price | `fig5_diesel_breakeven.py` | PCM + PV+battery + diesel workbooks |

**Supplementary Information** (in SI order)

| Figure | Script | Source data |
|--------|--------|-------------|
| Interannual variability of inputs | `si_fig_variability.py` | `Yearly_Results/locations_result.xlsx` |
| Optimal capacities | `si_fig_capacities.py` | PCM + PV+battery workbooks |
| VoLL sensitivity | `si_fig_voll.py` | PCM + PV+battery workbooks |
| CVaR (λ = 0.9) vs risk-neutral SO (λ = 0) | `si_fig_riskterms.py` | `Risk_Sweep_Results/risk_sweep_summary.xlsx` |
| Loss of load (thermal and electrical) | `si_fig_loss_of_load.py` | PCM + PV+battery workbooks |
| Cost–reliability frontier | `si_fig_frontier.py` | PCM + PV+battery workbooks |

## Risk-parameter sweep

`si_run_risk_sweep.py` is the cluster worker for the (λ, α) sweep: nested
leave-one-block-out cross-validation (5-year weather blocks) over 5 climates × 3 VoLL ×
λ ∈ {0, 0.25, 0.5, 0.75, 0.9, 1} × α ∈ {0.8, 0.9, 0.95} (270 tasks), writing one CSV per
task to `Risk_Sweep_Results/partials/`. It needs Gurobi; see `sherlock/README.md` for the
SLURM submission. `si_collect_risk_sweep.py` aggregates the partials into
`Risk_Sweep_Results/risk_sweep_summary.xlsx` (sheets `Folds`, `CellSummary`,
`GlobalRegret`) and needs only pandas, so it can be re-run from the committed partials.

## Accessibility

The method palette is colour-blind safe (checked under deuteranopia, protanopia and
tritanopia simulation). Architecture is always encoded redundantly by hatch or marker
fill, so figures remain readable in grayscale.
