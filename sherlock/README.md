# Sherlock — two steps only

## Step 1 — setup once (~10–20 min)

```bash
cd $GROUP_HOME/Capacity_Algo
mkdir -p logs
sbatch sherlock/job_setup.sbatch
tail -f logs/capa-setup-<JOBID>.out
```

Wait until you see **`Setup complete`**.

## Step 2 — submit runs

```bash
bash sherlock/submit.sh
```

Monitor: `squeue -u $USER` or `bash sherlock/check_status.sh`

## If setup fails

```bash
rm -rf .venv
sbatch sherlock/job_setup.sbatch
```

## Outputs

- `FOB_Results/FOB_Sensitivity_Results.xlsx`
- `FOB_PVB_Results/FOB_PVB_Sensitivity_Results.xlsx`

---

# Risk-parameter sweep (SI Fig S3) — lambda / alpha selection

Justifies the **fixed a-priori** SO-CVaR risk parameters (λ=0.9, α=0.9) used in the
main results, by **nested cross-validation** (5-year weather blocks as groups, no
temporal leakage) across the **whole design space (5 climates × 3 VoLL)**: an
**outer** leave-one-block-out loop holds out a TEST block; within the remaining
20 years an **inner** leave-one-block-out loop fits capacities on TRAIN (15 yr) and
selects `(lambda, alpha)` on VALIDATION (5 yr). Test is never used for selection.
Runs as a SLURM **array** (one `(climate, VoLL, lambda, alpha)` combo per task, 270
tasks, 25 SO-CVaR solves each), then a collect job reports the **fixed-vs-tuned test
gap** per cell (≈0 ⇒ fixing the parameters carries no selection advantage) and a
pooled relative-regret surface.

```bash
# after the one-time setup job has built .venv:
bash sherlock/submit_risk_sweep.sh
```

This submits `job_risk_sweep.sbatch` (array sized from the grid in
`paper_figures/si_run_risk_sweep.py`) and `job_risk_collect.sbatch`
(`afterok` dependency). Edit the grid (`CLIMATES`, `VOLLS`, `LAMBDAS`, `ALPHAS`) at the
top of `si_run_risk_sweep.py`; the submit script sizes `--array` automatically.
Tasks are **resumable** — a completed `Risk_Sweep_Results/partials/*.csv` is skipped,
so a re-submit only fills gaps. (Tip: probe timing first with
`sbatch --array=0-0 sherlock/job_risk_sweep.sbatch`; let it finish, then submit the rest.)

**Outputs:** `Risk_Sweep_Results/risk_sweep_summary.xlsx` (sheets: `Folds`,
`CellSummary`, `GlobalRegret`). The collect job prints, per climate × VoLL cell, the
fixed-vs-tuned test gap and the pooled global CV-optimum. Pull the xlsx locally and plot:

```powershell
.venv_verify\Scripts\python.exe paper_figures\si_fig_risk_params.py
```
