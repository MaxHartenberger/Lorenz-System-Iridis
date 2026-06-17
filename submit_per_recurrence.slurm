#!/bin/bash
# ============================================================================ #
# SLURM Job Array (per-recurrence) — L-BFGS Periodic Orbit Search
#
# Each array task handles ONE recurrence (all N × m combinations) using
# Julia multi-threading.  This has lower overhead than one-task-per-(N,m)
# when you have many N,m combinations and Julia compilation is significant.
#
# Array size = total number of recurrences across all T values.
#
# PREREQUISITES:
#   1. module avail julia               # check available versions
#   2. module load julia/1.12.4
#   3. julia scripts/generate_rec_task_list.jl --output rec_tasks.txt
#
# USAGE:
#   sbatch --array=1-<N> submit_per_recurrence.slurm
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# SLURM directives  (must have "#SBATCH" starting in column 1)
# ---------------------------------------------------------------------------- #
#SBATCH --job-name=lorenz-rec
#SBATCH --output=logs/rec_slurm_%A_%a.out
#SBATCH --error=logs/rec_slurm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                        # max for amd_student partition (4 cores)
#SBATCH --mem=3500                               # 3.5 GB (partition limit: 3.75 GB)
#SBATCH --time=12:00:00                          # 12 h per recurrence
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
mkdir -p logs

# Tell Julia how many threads to use (matches SLURM allocation)
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Julia threads:   ${JULIA_NUM_THREADS} (from SLURM_CPUS_PER_TASK)"

# ---------------------------------------------------------------------------- #
# Read parameters for this array task
# ---------------------------------------------------------------------------- #
TASK_LIST="rec_tasks.txt"

if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: Task list '$TASK_LIST' not found."
    echo "       Run:  julia scripts/generate_rec_task_list.jl --output rec_tasks.txt"
    exit 2
fi

TOTAL_TASKS=$(wc -l < "$TASK_LIST")
if [ "$SLURM_ARRAY_TASK_ID" -gt "$TOTAL_TASKS" ]; then
    echo "WARNING: Array task ID $SLURM_ARRAY_TASK_ID > task list ($TOTAL_TASKS)."
    echo "         Nothing to do — exiting."
    exit 0
fi

PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")

# Parse space-separated fields:  T  rec_id
set -- $PARAMS
T="$1"
REC_ID="$2"

if [ -z "$T" ] || [ -z "$REC_ID" ]; then
    echo "ERROR: Failed to parse parameters from line $SLURM_ARRAY_TASK_ID of $TASK_LIST"
    echo "       Raw line: $PARAMS"
    exit 2
fi

echo "=============================================="
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Array Task ID:   ${SLURM_ARRAY_TASK_ID} / ${TOTAL_TASKS}"
echo "Parameters:      T=${T}, rec_id=${REC_ID}"
echo "Julia threads:   ${JULIA_NUM_THREADS}"
echo "Started at:      $(date)"
echo "Node:            $(hostname)"
echo "=============================================="
echo ""

# Run the multi-threaded worker for this recurrence.
# We pass -t explicitly so Julia knows the thread count even if the env
# variable isn't picked up by the launcher.
julia -t "$SLURM_CPUS_PER_TASK" scripts/hpc_worker_per_rec.jl "$T" "$REC_ID"

EXIT_CODE=$?

echo ""
echo "Finished at:     $(date)"
echo "Exit code:       ${EXIT_CODE}"
exit $EXIT_CODE
