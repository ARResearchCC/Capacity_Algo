#!/bin/bash
#SBATCH --job-name=capa-risk-sweep
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --qos=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-269
#SBATCH --mail-type=END,FAIL

# SO-CVaR (lambda, alpha) NESTED cross-validation as a SLURM array: one
# (climate, VoLL, lambda, alpha) combo per task. Default --array=0-269 =
# 5 climates x 3 VoLL x 6 lambda x 3 alpha. submit_risk_sweep.sh overrides --array
# to match the grid in paper_figures/si_run_risk_sweep.py (edit CLIMATES/VOLLS/
# LAMBDAS/ALPHAS there). Each task does 5 outer x (4 inner-validation fits on 15 yr
# + 1 outer-test fit on 20 yr) = 25 SO-CVaR solves and writes a resumable partial
# Risk_Sweep_Results/partials/<climate>_<voll>_l<lam>_a<alp>.csv. Budget 2 days/task;
# a single 20-yr solve dominates, so run a 1-task probe (--array=0-0) to calibrate.

set -euo pipefail
trap 'echo "FAILED at line ${LINENO} (exit $?)" >&2' ERR

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs Risk_Sweep_Results/partials

# shellcheck disable=SC1091
source sherlock/_common.sh

echo "=== risk-sweep array task ${SLURM_ARRAY_TASK_ID} at $(date) ==="
python -u paper_figures/si_run_risk_sweep.py
echo "=== finished task ${SLURM_ARRAY_TASK_ID} at $(date) ==="
