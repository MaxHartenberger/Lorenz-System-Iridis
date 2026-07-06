#!/bin/bash
# ============================================================================ #
# Hookstep (Newton–GMRES trust-region) sweep — first 5 recurrences per T.
#
# One (T, rec_id, N, gmres_maxiter) per task, 1 CPU.
#
# Sweep dimensions:
#   T             ∈ {5, 10, 20, 40, 80, 160}
#   rec_id        ∈ {1, 2, 3, 4, 5}
#   N             ∈ {5, 10, 20, 40, 80, 160, 320}
#   gmres_maxiter ∈ {5, 10, 20}
#
# Total tasks: 630  (fits in a single array; MaxArraySize=1001 on Iridis).
#
# Generate hookstep_tasks.txt FIRST:
#
#   for T in 5 10 20 40 80 160; do \
#     for rec in $(seq 1 5); do \
#       for N in 5 10 20 40 80 160 320; do \
#         for gmres in 5 10 20; do \
#           echo "$T $rec $N $gmres"; \
#         done; \
#       done; \
#     done; \
#   done > hookstep_tasks.txt
#
# Then submit:
#   sbatch --array=1-630 hookstep_first_5_rec_sweep.slurm
# ============================================================================ #

#SBATCH --job-name=lorenz_hk
#SBATCH --output=logs/hookstep_%A_%a.out
#SBATCH --error=logs/hookstep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3700
#SBATCH --time=04:00:00
#SBATCH --partition=amd_student

module load julia/1.12.4 || { echo "ERROR: Julia module not found"; exit 2; }
cd "$SLURM_SUBMIT_DIR" || exit 2
mkdir -p logs

# Task file — first argument, or default to hookstep_tasks.txt
TASK_FILE="${1:-hookstep_tasks.txt}"

PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

# Last chunk may have fewer lines — exit gracefully for empty params
if [ -z "$PARAMS" ]; then
    echo "task ${SLURM_ARRAY_TASK_ID}  (empty — skipping)"
    exit 0
fi

set -- $PARAMS

echo "task ${SLURM_ARRAY_TASK_ID}  T=$1 rec=$2 N=$3 gmres_maxiter=$4  $(date)"

julia scripts/hpc_worker_hookstep.jl "$1" "$2" "$3" "$4"
exit $?
