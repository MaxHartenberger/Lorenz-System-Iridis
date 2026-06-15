#!/bin/bash
# ============================================================================ #
# SLURM Job Array — L-BFGS Periodic Orbit Search (Lorenz System)
#
# Submits one job-array task per (T, rec_id, N, m) combination.
# Each task runs independently; failed tasks can be re-submitted individually.
#
# PREREQUISITES (run BEFORE submitting):
#   1. module load julia/1.10    (or whichever Julia module is available)
#   2. julia generate_task_list.jl --output tasks.txt
#
# USAGE:
#   sbatch --array=1-<N> submit_sweep.slurm
#
#   where <N> = total number of lines in tasks.txt
#
#   You can also submit subsets:
#     sbatch --array=1-100      submit_sweep.slurm   # first 100 tasks
#     sbatch --array=101-200    submit_sweep.slurm   # next 100 tasks
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# SLURM directives
# ---------------------------------------------------------------------------- #
#SBATCH --job-name=lorenz-lbfgs
#SBATCH --output=logs/slurm_%A_%a.out          # %A = job ID, %a = array task ID
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1                               # single CPU core per task
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000                               # 2 GB RAM per task
#SBATCH --time=04:00:00                          # 4 hours per task (adjust as needed)
#SBATCH --partition=batch                        # default partition

# ---------------------------------------------------------------------------- #
# Environment setup
# ---------------------------------------------------------------------------- #
# Load Julia — check available versions with:  module avail julia
module load julia/1.10  2>/dev/null || echo "WARNING: Could not load julia module — using system Julia"

# Project root (directory containing this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Create log directory
mkdir -p logs

# ---------------------------------------------------------------------------- #
# Read the parameters for this array task
# ---------------------------------------------------------------------------- #
TASK_LIST="tasks.txt"

if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: Task list '$TASK_LIST' not found."
    echo "Run:  julia scripts/generate_task_list.jl --output tasks.txt"
    exit 1
fi

TOTAL_TASKS=$(wc -l < "$TASK_LIST")
if [ "$SLURM_ARRAY_TASK_ID" -gt "$TOTAL_TASKS" ]; then
    echo "WARNING: Array task ID $SLURM_ARRAY_TASK_ID exceeds task list length ($TOTAL_TASKS). Nothing to do."
    exit 0
fi

# Extract the parameter line for this array task (1-indexed)
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")
read -r T REC_ID N M <<< "$PARAMS"

# ---------------------------------------------------------------------------- #
# Run the worker
# ---------------------------------------------------------------------------- #
echo "=============================================="
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Array Task ID:   ${SLURM_ARRAY_TASK_ID} / ${TOTAL_TASKS}"
echo "Parameters:      T=${T}, rec_id=${REC_ID}, N=${N}, m=${M}"
echo "Started at:      $(date)"
echo "Node:            $(hostname)"
echo "=============================================="
echo ""

# Run Julia worker
julia scripts/hpc_worker.jl "$T" "$REC_ID" "$N" "$M"

EXIT_CODE=$?

echo ""
echo "Finished at:     $(date)"
echo "Exit code:       ${EXIT_CODE}"

exit $EXIT_CODE
