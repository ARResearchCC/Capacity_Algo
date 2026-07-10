#!/bin/bash
#SBATCH --job-name=capa-risk-collect
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL

# Aggregate the risk-sweep partials, build validation/test surfaces, and select
# the best (lambda, alpha). Runs after the array job (see submit_risk_sweep.sh,
# which adds an afterok dependency). Pure pandas/openpyxl — no solver needed.

set -euo pipefail
trap 'echo "FAILED at line ${LINENO} (exit $?)" >&2' ERR

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# shellcheck disable=SC1091
source sherlock/_common.sh

echo "=== risk-sweep collect at $(date) ==="
python -u paper_figures/si_collect_risk_sweep.py
echo "=== finished collect at $(date) ==="
echo "Selected params + fixed-vs-tuned gap are printed above."
echo "Pull Risk_Sweep_Results/risk_sweep_summary.xlsx locally, then plot with:"
echo "  .venv_verify\\Scripts\\python.exe paper_figures\\si_fig_risk_params.py"
