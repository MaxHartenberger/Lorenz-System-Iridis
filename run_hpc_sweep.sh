#!/bin/bash
# ============================================================================ #
# HPC SWEEP LAUNCHER — Automates the full workflow
#
# Usage (from project root on Iridis login node):
#   bash run_hpc_sweep.sh
#
# What it does:
#   1. Loads Julia
#   2. Generates the task list
#   3. Submits the job array
#   4. Prints monitoring commands
# ============================================================================ #

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  L-BFGS Periodic Orbit Search — HPC Launcher"
echo "  Project: $PROJECT_ROOT"
echo "  Date:    $(date)"
echo "=============================================="
echo ""

# --- 1. Load Julia -----------------------------------------------------------
echo "[1/3] Loading Julia module..."
echo "  Available Julia modules:"
module avail julia 2>&1 | grep -i julia || true
echo ""
echo "  Attempting to load julia/1.12.4 (default on Iridis) ..."
if module load julia/1.12.4 2>/dev/null; then
    echo "  Loaded: $(which julia)"
    julia --version
else
    echo "  ERROR: Could not load Julia module 'julia/1.12.4'."
    echo "  Check available versions above and update this script or the .slurm files."
    exit 1
fi
echo ""

# --- 2. Choose strategy ------------------------------------------------------
SLURM_MAX_ARRAY=1001   # Iridis limit: scontrol show config | grep MaxArraySize

echo "[2/3] Select submission strategy:"
echo "  A) Per-recurrence:  one task per (T, rec_id) [RECOMMENDED — fits in array limit]"
echo "  B) Fine-grained:    one task per (T, rec_id, N, m) [only for subsets ≤$SLURM_MAX_ARRAY]"
read -r -p "  Enter A or B [A]: " STRATEGY
STRATEGY=${STRATEGY:-A}
echo ""

if [[ "$STRATEGY" =~ ^[Bb]$ ]]; then
    # Fine-grained — check against array limit
    echo "  Generating per-case task list..."
    julia scripts/generate_task_list.jl --output tasks.txt
    NTASKS=$(wc -l < tasks.txt)
    echo "  Total tasks: $NTASKS"
    echo ""

    if [ "$NTASKS" -gt "$SLURM_MAX_ARRAY" ]; then
        echo "  ⚠ $NTASKS tasks exceeds SLURM MaxArraySize ($SLURM_MAX_ARRAY)."
        echo "    Splitting into chunks of $SLURM_MAX_ARRAY..."
        for ((start=1; start<=NTASKS; start+=SLURM_MAX_ARRAY)); do
            end=$((start + SLURM_MAX_ARRAY - 1))
            if [ "$end" -gt "$NTASKS" ]; then end=$NTASKS; fi
            echo "    Submitting chunk $start-$end ..."
            JOBID=$(sbatch --parsable --array="$start-$end" submit_sweep.slurm)
            echo "      Job ID: $JOBID"
        done
    else
        echo "[3/3] Submitting job array..."
        JOBID=$(sbatch --parsable --array=1-"$NTASKS" submit_sweep.slurm)
        echo ""
        echo "=============================================="
        echo "  Submitted!  Job ID: $JOBID"
        echo "  Tasks:       $NTASKS"
        echo ""
        echo "  Monitor:     squeue -j $JOBID"
        echo "  Logs:        logs/slurm_${JOBID}_*.out"
        echo "  Cancel:      scancel $JOBID"
        echo "=============================================="
    fi
else
    # Per-recurrence (default)
    echo "  Generating recurrence-level task list..."
    julia scripts/generate_rec_task_list.jl --output rec_tasks.txt
    NTASKS=$(wc -l < rec_tasks.txt)
    echo "  Total tasks: $NTASKS"
    echo ""
    echo "[3/3] Submitting job array..."
    JOBID=$(sbatch --parsable --array=1-"$NTASKS" submit_per_recurrence.slurm)
    echo ""
    echo "=============================================="
    echo "  Submitted!  Job ID: $JOBID"
    echo "  Tasks:       $NTASKS"
    echo ""
    echo "  Monitor:     squeue -j $JOBID"
    echo "  Logs:        logs/rec_slurm_${JOBID}_*.out"
    echo "  Cancel:      scancel $JOBID"
    echo "=============================================="
fi
