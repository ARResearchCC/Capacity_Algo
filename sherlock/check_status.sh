#!/bin/bash
# Show status of recent Capacity_Algo SLURM jobs and result files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "=== Queue ==="
squeue -u "${USER}" 2>/dev/null || true

echo ""
echo "=== Recent jobs (last 10) ==="
sacct -u "${USER}" --name=capa-fob,capa-fob-pvb,capa-fob-both,capa-setup \
  --format=JobID,JobName%16,State,ExitCode,Elapsed,MaxRSS%12,End -n \
  2>/dev/null | tail -10 || echo "(sacct unavailable)"

echo ""
echo "=== Latest log files ==="
ls -lt logs/capa-*.out 2>/dev/null | head -4 || echo "(no logs yet)"

echo ""
echo "=== Result files ==="
for f in FOB_Results/FOB_Sensitivity_Results.xlsx FOB_PVB_Results/FOB_PVB_Sensitivity_Results.xlsx; do
  if [[ -f "$f" ]]; then
    ls -lh "$f"
  else
    echo "  (not yet) $f"
  fi
done

echo ""
echo "=== .venv ==="
if [[ -f .venv/bin/python ]]; then
  .venv/bin/python -c "import numpy; print('numpy OK:', numpy.__version__)" 2>/dev/null || echo "  .venv exists but packages broken — run: rm -rf .venv .venv.lock"
else
  echo "  (not created yet — first job will create it)"
fi
