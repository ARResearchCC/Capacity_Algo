#!/bin/bash
# Find a working Gurobi+Python setup on Sherlock. Run on login node:
#   bash sherlock/diagnose_modules.sh

set -uo pipefail

echo "=== Sherlock Gurobi / Python diagnostics ==="
echo ""

module_reset() { module --force purge 2>/dev/null || module purge; }

test_combo() {
  local gurobi_mod="$1"
  local python_mod="${2:-}"
  module_reset
  if ! module load math "${gurobi_mod}" 2>/dev/null; then
    echo "FAIL  math ${gurobi_mod} ${python_mod:-'(no python)'}"
    return 1
  fi
  if [[ -n "${python_mod}" ]]; then
    if ! module load devel "${python_mod}" 2>/dev/null; then
      echo "FAIL  math ${gurobi_mod} + devel ${python_mod}"
      return 1
    fi
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "FAIL  math ${gurobi_mod} ${python_mod:-'(no python)'} — no python3"
    return 1
  fi
  echo "OK    math ${gurobi_mod} ${python_mod:-'(gurobi only)'} — $(python3 --version 2>&1)"
  echo "      GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-unset}"
  echo "      GUROBI_HOME=${GUROBI_HOME:-unset}"
  return 0
}

echo "--- Module combos ---"
combos=(
  "gurobi/13.0.1 python/3.12.1"
  "gurobi/11.0.2 python/3.9.0"
  "gurobi/11.0.2 python/3.12.1"
  "gurobi/11.0.2"
  "gurobi/13.0.1"
)
winner=""
for combo in "${combos[@]}"; do
  g="${combo%% *}"
  p="${combo#* }"
  [[ "$p" == "$g" ]] && p=""
  if test_combo "$g" "$p"; then
    winner="math $g ${p:+devel $p}"
    break
  fi
done

echo ""
echo "--- Cluster Gurobi paths (no module) ---"
module_reset
module load devel python/3.12.1 2>/dev/null || true
found=0
for root in /share/software/non-free/Gurobi /software/non-free/Gurobi; do
  [[ -d "$root" ]] || continue
  echo "Found: $root"
  ls -1 "$root" 2>/dev/null | head -5
  found=1
done
[[ "$found" -eq 0 ]] && echo "(no /share/software/non-free/Gurobi or /software/non-free/Gurobi)"

echo ""
if [[ -n "$winner" ]]; then
  echo "RECOMMENDATION: jobs will auto-use: $winner"
else
  echo "RECOMMENDATION: no module combo worked — jobs will try cluster path fallback"
  echo "  If jobs still fail, ask PI/SRCC about Gurobi access on Sherlock"
fi
echo ""
echo "Next: rm -rf .venv .venv.lock && bash sherlock/submit.sh"
