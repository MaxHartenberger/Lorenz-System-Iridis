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
if module load julia/1.10 2>/dev/null; then
    echo "  Loaded: $(which julia)"
    julia --version
else
    echo "  WARNING: Could not load 'julia/1.10'. Trying system Julia..."
    if command -v julia &>/dev/null; then
        echo "  Using: $(which julia)"
        julia --version
    else
        echo "  ERROR: Julia not found. Load a Julia module or check your PATH."
        exit 1
    fi
fi
echo ""

# --- 2. Choose strategy ------------------------------------------------------
echo "[2/3] Select submission strategy:"
echo "  A) Fine-grained: one SLURM task per (T, rec_id, N, m)  [MAX parallelism]"
echo "  B) Coarse-grained: one SLURM task per (T, rec_id)      [lower overhead]"
read -r -p "  Enter A or B [A]: " STRATEGY
STRATEGY=${STRATEGY:-A}
echo ""

if [[ "$STRATEGY" =~ ^[Bb]$ ]]; then
    # Coarse-grained
    echo "  Generating recurrence-level task list..."
    julia scripts/generate_rec_task_list.jl --output rec_tasks.txt
    NTASKS=$(wc -l < rec_tasks.txt)
    echo "  Total tasks: $NTASKS"
    echo ""
    echo "[3/3] Submitting job array..."
    JOBID=$(sbatch --parsable --array=1-"$NTASKS" submit_per_recurrence.slurm)
else
    # Fine-grained
    echo "  Generating per-case task list..."
    julia scripts/generate_task_list.jl --output tasks.txt
    NTASKS=$(wc -l < tasks.txt)
    echo "  Total tasks: $NTASKS"
    echo ""
    echo "[3/3] Submitting job array..."
    JOBID=$(sbatch --parsable --array=1-"$NTASKS" submit_sweep.slurm)
fi

echo ""
echo "=============================================="
echo "  Submitted!  Job ID: $JOBID"
echo "  Tasks:       $NTASKS"
echo ""
echo "  Monitor:     squeue -j $JOBID"
echo "  Logs:        logs/slurm_${JOBID}_*.out"
echo "  Cancel:      scancel $JOBID"
echo "=============================================="
