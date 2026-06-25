#!/bin/bash
# ============================================================================ #
# Fine-grained: one (T, rec_id, N, m) per task, 1 CPU.
#
# Generate tasks.txt FIRST:
#   julia scripts/generate_task_list.jl --T 5,10,20
#
# Then submit in chunks (printed by the command above):
#   sbatch --array=1-1000 first_sweep.slurm
# ============================================================================ #

#SBATCH --job-name=lorenz
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=01:00:00
#SBATCH --partition=amd_student

module load julia/1.12.4 || { echo "ERROR: Julia module not found"; exit 2; }
cd "$SLURM_SUBMIT_DIR" || exit 2
mkdir -p logs

PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks.txt)
set -- $PARAMS

echo "task ${SLURM_ARRAY_TASK_ID}  T=$1 rec=$2 N=$3 m=$4  $(date)"

julia scripts/hpc_worker.jl "$1" "$2" "$3" "$4"
exit $?