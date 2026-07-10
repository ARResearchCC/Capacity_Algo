#!/bin/bash
# Submit the SO-CVaR (lambda, alpha) risk-parameter sweep as an array job, then a
# collect job that selects the best parameters (afterok dependency).
# Prereq: .venv already built (sbatch sherlock/job_setup.sbatch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash sherlock/preflight.sh
mkdir -p logs Risk_Sweep_Results/partials

# preflight runs in a subshell, so load the Sherlock modules in THIS shell too —
# otherwise .venv/bin/python below can't find libpython3.12.so.1.0.
# shellcheck disable=SC1091
source sherlock/modules.sh
load_sherlock_modules

# Size the array from the grid defined in si_run_risk_sweep.py (lazy imports keep
# this cheap — no solver needed just to count tasks).
N="$(.venv/bin/python -c 'import sys; sys.path.insert(0,"paper_figures"); import si_run_risk_sweep as m; print(len(m.task_grid()))')"
LAST=$(( N - 1 ))
echo "Grid has ${N} (climate, lambda, alpha) tasks -> --array=0-${LAST}"

MAIL="${USER}@stanford.edu"
SWEEP_ID="$(sbatch --mail-user="${MAIL}" --parsable --array=0-"${LAST}" sherlock/job_risk_sweep.sbatch)"
COLLECT_ID="$(sbatch --mail-user="${MAIL}" --parsable --dependency=afterok:"${SWEEP_ID}" sherlock/job_risk_collect.sbatch)"

echo ""
echo "Submitted:"
echo "  sweep (array 0-${LAST}): ${SWEEP_ID}  ->  logs/capa-risk-sweep-${SWEEP_ID}_*.out"
echo "  collect (afterok):       ${COLLECT_ID}  ->  logs/capa-risk-collect-${COLLECT_ID}.out"
echo "  Monitor: squeue -u \$USER"
echo ""
echo "When done, pull Risk_Sweep_Results/risk_sweep_*.xlsx and plot locally:"
echo "  .venv_verify\\Scripts\\python.exe paper_figures\\si_fig_risk_params.py"
