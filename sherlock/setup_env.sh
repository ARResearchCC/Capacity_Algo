#!/bin/bash
# One-time venv setup. Run via: sbatch sherlock/job_setup.sbatch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
VENV="${ROOT}/.venv"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

if type conda &>/dev/null; then
  while [[ "${CONDA_DEFAULT_ENV:-}" != "" ]]; do conda deactivate 2>/dev/null || break; done
fi

# shellcheck disable=SC1091
source "${ROOT}/sherlock/modules.sh"
load_sherlock_modules
log "Modules OK — $(python3 --version)"

verify_venv() {
  [[ -f "${VENV}/bin/python" ]] || return 1
  "${VENV}/bin/python" -c "
import h5py, pvlib, numpy, pandas, scipy, pyomo.environ, gurobipy, pysolar, pytz
from pvlib.clearsky import ineichen
from pvlib.irradiance import get_total_irradiance
import gurobipy as gp
import pyomo.environ as pyo
assert pyo.SolverFactory('gurobi').available()
with gp.Env(empty=True) as e:
    e.setParam('OutputFlag', 0)
    e.start()
print('All imports OK')
" 2>/dev/null
}

if verify_venv; then
  log "Existing .venv is complete — nothing to do"
  exit 0
fi

log "Creating .venv and installing packages (~10-20 min)..."
rm -rf "${VENV}"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Binary wheels only — avoids compiling h5py/scipy on the cluster
log "Installing numpy, scipy, h5py (binary wheels)..."
python -m pip install --no-cache-dir --only-binary=:all: \
  "numpy==1.26.4" "pandas==2.2.3" "scipy==1.14.1" "h5py==3.11.0" \
  || die "Binary wheel install failed — contact SRCC or try from login node with VPN"

log "Installing remaining packages..."
python -m pip install --no-cache-dir -r requirements-cluster.txt

verify_venv || die "Setup finished but imports failed"
log "Setup complete. Next: bash sherlock/submit.sh"
