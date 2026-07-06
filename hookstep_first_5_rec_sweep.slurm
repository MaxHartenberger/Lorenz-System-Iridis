#!/bin/bash
# ============================================================================ #
# Hookstep (Newton–GMRES trust-region) sweep — first 5 recurrences per T.
#
# One (T, rec_id, N, gmres_maxiter) per task, 1 CPU.
# Each task computes its parameters directly from SLURM_ARRAY_TASK_ID —
# no task file, no races.
#
# Sweep:  T ∈ {5,10,20,40,80,160}  ×  rec ∈ {1..5}  ×
#         N ∈ {5,10,20,40,80,160,320}  ×  gmres_maxiter ∈ {5,10,20}
#         6 × 5 × 7 × 3 = 630 tasks
#
# Usage:
#   sbatch --array=1-630 hookstep_first_5_rec_sweep.slurm
# ============================================================================ #

# --- SLURM directives ------------------------------------------------------ #
#SBATCH --job-name=lorenz_hk
#SBATCH --output=logs/hookstep_%A_%a.out
#SBATCH --error=logs/hookstep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3700
#SBATCH --time=04:00:00
#SBATCH --partition=amd_student

# --- Environment ----------------------------------------------------------- #
set -euo pipefail

module load julia/1.12.4 || { echo "ERROR: Julia module not found"; exit 2; }
cd "$SLURM_SUBMIT_DIR" || exit 2
mkdir -p logs

# --- Guard: must be submitted with --array --------------------------------- #
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is unset."
    echo "You submitted without --array.  Use:"
    echo "  sbatch --array=1-630 hookstep_first_5_rec_sweep.slurm"
    exit 2
fi

# --- Parameter arrays (outermost → innermost loop order) ------------------- #
TS=(    5   10   20   40   80  160)   # 6 values
RECS=(  1    2    3    4    5)        # 5 values
NS=(    5   10   20   40   80  160  320)  # 7 values
GMRES=( 5   10   20)                  # 3 values

# --- Compute parameters from SLURM_ARRAY_TASK_ID (1-based) ----------------- #
# Task ordering: for T in TS; for rec in RECS; for N in NS; for gmres in GMRES
#   Stride:  gmres=3  N=21  rec=105  T=630
TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))

gmres_idx=$(( TASK_IDX % 3 ))
N_idx=$((     (TASK_IDX / 3)  % 7 ))
rec_idx=$((   (TASK_IDX / 21) % 5 ))
T_idx=$((     (TASK_IDX / 105)      ))

# Bounds check
if [ "$T_idx" -ge "${#TS[@]}" ]; then
    echo "task ${SLURM_ARRAY_TASK_ID}  (beyond sweep range — skipping)"
    exit 0
fi

T_VAL="${TS[$T_idx]}"
REC_VAL="${RECS[$rec_idx]}"
N_VAL="${NS[$N_idx]}"
GMRES_VAL="${GMRES[$gmres_idx]}"

echo "============================================================"
echo "Hookstep sweep — Task $SLURM_ARRAY_TASK_ID/630"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID  (array $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID)"
echo "Host:         $(hostname)"
echo "Start:        $(date)"
echo "T:            $T_VAL"
echo "rec_id:       $REC_VAL"
echo "N:            $N_VAL  (shooting segments)"
echo "gmres_maxiter: $GMRES_VAL  (Krylov subspace dim)"
echo "============================================================"

# --- Run the hookstep worker ----------------------------------------------- #
julia scripts/hpc_worker_hookstep.jl "$T_VAL" "$REC_VAL" "$N_VAL" "$GMRES_VAL"
EXIT_CODE=$?

echo ""
echo "End:   $(date)"
echo "Exit:  $EXIT_CODE"
exit $EXIT_CODE
