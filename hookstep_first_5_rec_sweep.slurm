#!/bin/bash
# ============================================================================ #
# Hookstep (Newton–GMRES trust-region) sweep — first 5 recurrences per T.
#
# Sweep dimensions:
#   T     ∈ {5, 10, 20, 40, 80, 160}
#   rec   ∈ {1, 2, 3, 4, 5}                           (first 5 per T)
#   N     ∈ all NS, filtered by T (see below)
#   gmres_maxiter ∈ {5, 10, 20}
#
# N-filtering (same as L-BFGS sweep):
#   T=5,10,20  →  N ∈ {5, 10, 20, 40, 80, 160, 320}   (all 7)
#   T=40       →  N ∈ {10, 20, 40, 80, 160, 320}       (N ≥ 10)
#   T=80       →  N ∈ {20, 40, 80, 160, 320}           (N ≥ 20)
#   T=160      →  N ∈ {40, 80, 160, 320}               (N ≥ 40)
#
# Total tasks: 540  (fits in a single array, MaxArraySize=1001 on Iridis).
#
# --- Generate task list BEFORE submitting ---
#
#   ./hookstep_first_5_rec_sweep.slurm --gen-tasks
#
# Or run this one-liner:
#
#   for T in 5 10 20; do for rec in $(seq 1 5); do for N in 5 10 20 40 80 160 320; do for gmres in 5 10 20; do echo "$T $rec $N $gmres"; done; done; done; done > hookstep_tasks.txt
#   for T in 40; do for rec in $(seq 1 5); do for N in 10 20 40 80 160 320; do for gmres in 5 10 20; do echo "$T $rec $N $gmres"; done; done; done; done >> hookstep_tasks.txt
#   for T in 80; do for rec in $(seq 1 5); do for N in 20 40 80 160 320; do for gmres in 5 10 20; do echo "$T $rec $N $gmres"; done; done; done; done >> hookstep_tasks.txt
#   for T in 160; do for rec in $(seq 1 5); do for N in 40 80 160 320; do for gmres in 5 10 20; do echo "$T $rec $N $gmres"; done; done; done; done >> hookstep_tasks.txt
#
# --- Submit ---
#
#   sbatch --array=1-540 hookstep_first_5_rec_sweep.slurm
#
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

# --- Handle --gen-tasks flag (run directly, not via sbatch) ---
if [ "$1" = "--gen-tasks" ]; then
    TASK_FILE="${2:-hookstep_tasks.txt}"
    echo "Generating $TASK_FILE ..."
    > "$TASK_FILE"
    for T in 5 10 20; do
        for rec in $(seq 1 5); do
            for N in 5 10 20 40 80 160 320; do
                for gmres in 5 10 20; do
                    echo "$T $rec $N $gmres" >> "$TASK_FILE"
                done
            done
        done
    done
    for T in 40; do
        for rec in $(seq 1 5); do
            for N in 10 20 40 80 160 320; do
                for gmres in 5 10 20; do
                    echo "$T $rec $N $gmres" >> "$TASK_FILE"
                done
            done
        done
    done
    for T in 80; do
        for rec in $(seq 1 5); do
            for N in 20 40 80 160 320; do
                for gmres in 5 10 20; do
                    echo "$T $rec $N $gmres" >> "$TASK_FILE"
                done
            done
        done
    done
    for T in 160; do
        for rec in $(seq 1 5); do
            for N in 40 80 160 320; do
                for gmres in 5 10 20; do
                    echo "$T $rec $N $gmres" >> "$TASK_FILE"
                done
            done
        done
    done
    echo "Done — $(wc -l < "$TASK_FILE") tasks written to $TASK_FILE"
    echo ""
    echo "Now submit with:  sbatch --array=1-$(wc -l < "$TASK_FILE") $0"
    exit 0
fi

# --- Environment ---
module load julia/1.12.4 || { echo "ERROR: Julia module not found"; exit 2; }
cd "$SLURM_SUBMIT_DIR" || exit 2
mkdir -p logs

# --- Task file — first argument, or default ---
TASK_FILE="${1:-hookstep_tasks.txt}"

if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: Task file '$TASK_FILE' not found."
    echo "Generate it first:  $0 --gen-tasks"
    exit 2
fi

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
