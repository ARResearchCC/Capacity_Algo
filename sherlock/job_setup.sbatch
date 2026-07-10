#!/bin/bash
#SBATCH --job-name=capa-setup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
trap 'echo "FAILED at line ${LINENO} (exit $?)" >&2' ERR

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
bash sherlock/setup_env.sh
