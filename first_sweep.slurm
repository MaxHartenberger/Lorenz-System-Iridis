#!/bin/bash
# ============================================================================ #
# Fine-grained: one (T, rec_id, N, m) per task, 1 CPU.
#
# Generate tasks.txt FIRST:
#   julia scripts/generate_task_list.jl --T 5,10,20
#
# If >1000 tasks, split and submit by chunk:
#   split -d -l 1000 tasks.txt chunk_
#   sbatch --array=1-1000 first_sweep.slurm chunk_00
#   sbatch --array=1-1000 first_sweep.slurm chunk_01
#   ...
# ============================================================================ #

#SBATCH --job-name=lorenz
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3700
#SBATCH --time=04:00:00
#SBATCH --partition=amd_student

module load julia/1.12.4 || { echo "ERROR: Julia module not found"; exit 2; }
cd "$SLURM_SUBMIT_DIR" || exit 2
mkdir -p logs

# Task file — first argument, or default to tasks.txt
TASK_FILE="${1:-tasks.txt}"

PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

# Last chunk may have fewer lines — exit gracefully for empty params
if [ -z "$PARAMS" ]; then
    echo "task ${SLURM_ARRAY_TASK_ID}  (empty — skipping)"
    exit 0
fi

set -- $PARAMS

echo "task ${SLURM_ARRAY_TASK_ID}  T=$1 rec=$2 N=$3 m=$4  $(date)"

julia scripts/hpc_worker.jl "$1" "$2" "$3" "$4"
exit $?