#!/bin/bash
# Sherlock modules — confirmed working via diagnose_modules.sh

load_sherlock_modules() {
  module --force purge 2>/dev/null || module purge
  module load math gurobi/13.0.1
  module load devel python/3.12.1
  module load gcc/12.4.0 2>/dev/null || true
  module load hdf5/1.14.4 2>/dev/null || module load hdf5/1.12.2 2>/dev/null || true
}
