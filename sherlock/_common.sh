#!/bin/bash
# Runtime env for FOB jobs — assumes .venv already exists (see setup_env.sh).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
VENV="${PROJECT_DIR}/.venv"
export PYTHONUNBUFFERED=1

die() { echo "ERROR: $*" >&2; exit 1; }

if type conda &>/dev/null; then
  while [[ "${CONDA_DEFAULT_ENV:-}" != "" ]]; do conda deactivate 2>/dev/null || break; done
fi

cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/sherlock/modules.sh"
load_sherlock_modules

[[ -f "${VENV}/bin/activate" ]] || die ".venv missing — run first: sbatch sherlock/job_setup.sbatch"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

"${VENV}/bin/python" -c "import h5py, pvlib, gurobipy, pyomo.environ" 2>/dev/null \
  || die ".venv broken — run: rm -rf .venv && sbatch sherlock/job_setup.sbatch"
