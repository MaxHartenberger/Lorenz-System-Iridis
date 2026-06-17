#!/bin/bash
# ============================================================================ #
# SLURM Job Array — L-BFGS Periodic Orbit Search (Lorenz System)
#
# Submits one job-array task per (T, rec_id, N, m) combination.
# Each task runs independently; failed tasks can be re-submitted individually.
#
# PREREQUISITES (run BEFORE submitting on the login node):
#   1. module avail julia              # check which Julia versions exist
#   2. module load julia/1.12.4         # (or whichever version is available)
#   3. julia scripts/generate_task_list.jl --output tasks.txt
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
# SLURM directives  (must have "#SBATCH" starting in column 1)
# ---------------------------------------------------------------------------- #
#SBATCH --job-name=lorenz-lbfgs
#SBATCH --output=logs/slurm_%A_%a.out          # %A = job ID, %a = array task ID
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1                               # single process per task
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000                               # 2 GB RAM per task
#SBATCH --time=04:00:00                          # 4 h per task (adjust as needed)
#SBATCH --partition=amd_student

# ---------------------------------------------------------------------------- #
# Environment setup
# ---------------------------------------------------------------------------- #
# Load Julia — run "module avail julia" on the login node to see versions.
module load julia/1.12.4
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load Julia module 'julia/1.12.4'."
    echo "       Run 'module avail julia' on the login node to check versions."
    exit 2
fi

# Project root — use SLURM_SUBMIT_DIR (the directory you ran sbatch from).
# ${BASH_SOURCE[0]} doesn't work on compute nodes because sbatch copies the
# script to a temp location.
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT" || exit 2

# Create log directory (SLURM needs this to exist before writing output files)
mkdir -p logs

# ---------------------------------------------------------------------------- #
# Read the parameters for this array task
# ---------------------------------------------------------------------------- #
TASK_LIST="tasks.txt"

if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: Task list '$TASK_LIST' not found."
    echo "       Run:  julia scripts/generate_task_list.jl --output tasks.txt"
    exit 2
fi

TOTAL_TASKS=$(wc -l < "$TASK_LIST")
if [ "$SLURM_ARRAY_TASK_ID" -gt "$TOTAL_TASKS" ]; then
    echo "WARNING: Array task ID $SLURM_ARRAY_TASK_ID > task list length ($TOTAL_TASKS)."
    echo "         Nothing to do — exiting."
    exit 0
fi

# Extract the parameter line for this array task (1-indexed)
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")

# Parse space-separated fields:  T  rec_id  N  m
set -- $PARAMS
T="$1"
REC_ID="$2"
N="$3"
M="$4"

if [ -z "$T" ] || [ -z "$REC_ID" ] || [ -z "$N" ] || [ -z "$M" ]; then
    echo "ERROR: Failed to parse parameters from line $SLURM_ARRAY_TASK_ID of $TASK_LIST"
    echo "       Raw line: $PARAMS"
    exit 2
fi

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

# Launch Julia worker with the four positional arguments
julia scripts/hpc_worker.jl "$T" "$REC_ID" "$N" "$M"

EXIT_CODE=$?

echo ""
echo "Finished at:     $(date)"
echo "Exit code:       ${EXIT_CODE}"

exit $EXIT_CODE
