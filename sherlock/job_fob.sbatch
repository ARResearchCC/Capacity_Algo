#!/bin/bash
#SBATCH --job-name=capa-fob
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --qos=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail
trap 'echo "FAILED at line ${LINENO} (exit $?)" >&2' ERR

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs FOB_Results

# shellcheck disable=SC1091
source sherlock/_common.sh

echo "=== Starting FOB.py at $(date) ==="
python -u FOB.py
echo "=== Finished FOB.py at $(date) ==="
