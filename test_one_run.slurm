#!/bin/bash
# ============================================================================ #
# Single test: T=5.0, rec_id=1, N=5, m=5
#
# Usage:  sbatch test_one_run.slurm
# ============================================================================ #

#SBATCH --job-name=test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=01:00:00
#SBATCH --partition=amd_student

# --- Environment ---
module load julia/1.12.4 || { echo "ERROR: Julia module not found"; exit 2; }
cd "$SLURM_SUBMIT_DIR" || exit 2
mkdir -p logs

echo "Job started at: $(date)"
echo "Node:           $(hostname)"
echo ""

# --- Run single case ---
julia scripts/hpc_worker.jl 5.0 1 5 5

EXIT_CODE=$?
echo ""
echo "Finished at: $(date)"
echo "Exit code:   ${EXIT_CODE}"
exit $EXIT_CODE
