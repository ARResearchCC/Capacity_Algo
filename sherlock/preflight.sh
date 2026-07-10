#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

for f in FOB.py FOB_PVB.py requirements-cluster.txt Calibration_Model_Input.xlsx Data; do
  [[ -e "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

[[ -f .venv/bin/python ]] || {
  echo "ERROR: .venv not found — run: sbatch sherlock/job_setup.sbatch"
  exit 1
}

if type conda &>/dev/null; then
  while [[ "${CONDA_DEFAULT_ENV:-}" != "" ]]; do conda deactivate 2>/dev/null || break; done
fi

# Gurobi license is only visible after loading Sherlock modules (not on bare login shell).
# shellcheck disable=SC1091
source "${ROOT}/sherlock/modules.sh"
load_sherlock_modules

if ! .venv/bin/python -c "import h5py, pvlib, gurobipy, pyomo.environ; print('venv OK')"; then
  echo "ERROR: .venv broken — run: rm -rf .venv && sbatch sherlock/job_setup.sbatch"
  exit 1
fi

echo "Preflight OK"
