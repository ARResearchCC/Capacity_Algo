#!/bin/bash
# Submit FOB + FOB_PVB (after setup job completed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash sherlock/preflight.sh
mkdir -p logs FOB_Results FOB_PVB_Results FOB_Diesel_Results

MAIL="${USER}@stanford.edu"
FOB_ID="$(sbatch --mail-user="${MAIL}" --parsable sherlock/job_fob.sbatch)"
PVB_ID="$(sbatch --mail-user="${MAIL}" --parsable sherlock/job_fob_pvb.sbatch)"
DIESEL_ID="$(sbatch --mail-user="${MAIL}" --parsable sherlock/job_fob_diesel.sbatch)"

echo ""
echo "Submitted:"
echo "  FOB:        ${FOB_ID}  ->  logs/capa-fob-${FOB_ID}.out"
echo "  FOB_PVB:    ${PVB_ID}  ->  logs/capa-fob-pvb-${PVB_ID}.out"
echo "  FOB_Diesel: ${DIESEL_ID}  ->  logs/capa-fob-diesel-${DIESEL_ID}.out"
echo "  Monitor: squeue -u \$USER"
echo ""
echo "Outputs land in FOB_Results/, FOB_PVB_Results/, FOB_Diesel_Results/."
echo "The plotting/data code reads the top-level copies, so after the runs finish:"
echo "  cp FOB_Results/FOB_Sensitivity_Results.xlsx ."
echo "  cp FOB_PVB_Results/FOB_PVB_Sensitivity_Results.xlsx ."
echo "  (FOB_Diesel_Results/ is already read in place.)"
