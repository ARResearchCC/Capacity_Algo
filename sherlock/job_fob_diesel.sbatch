#!/bin/bash
#SBATCH --job-name=capa-fob-diesel
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL

# Diesel baseline (FOB_Diesel.py): fast, no LP solves (direct dispatch), sweeps the
# fuel-price ladder (Low $13 / Med $45 / High $400 per gal) x 5 climates x 3 VoLL.
# Writes FOB_Diesel_Results/FOB_Diesel_Sensitivity_Results.xlsx (with a
# Fuel_Price_Scenarios sheet and a Fuel Price Level column).

set -euo pipefail
trap 'echo "FAILED at line ${LINENO} (exit $?)" >&2' ERR

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs FOB_Diesel_Results

# shellcheck disable=SC1091
source sherlock/_common.sh

echo "=== Starting FOB_Diesel.py at $(date) ==="
python -u FOB_Diesel.py
echo "=== Finished FOB_Diesel.py at $(date) ==="
