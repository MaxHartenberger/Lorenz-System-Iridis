#!/bin/bash
# ============================================================================ #
# SLURM Job Array — L-BFGS memory sweep on ALL perturbed RPOs
#
# Runs L-BFGS optimisation at 5 different history sizes (lbfgs_memory) in
# PARALLEL using a job array.  Each array task processes all 89 ICs with
# one memory value.  Tasks are independent — failure in one does NOT stop
# the others.
#
# Array mapping:
#   Task 1 → lbfgs_memory = 20
#   Task 2 → lbfgs_memory = 40
#   Task 3 → lbfgs_memory = 80
#   Task 4 → lbfgs_memory = 160
#   Task 5 → lbfgs_memory = 320
#
# All runs use the SAME spatial resolution (m=49, n=33, Δt=0.01).
#
# Usage:
#   sbatch slurm/lbfgs_m_sweep.slurm
#
#   # Monitor:
#   squeue -u $USER
#   tail -f $HOME/Open-Kolmogorov-Flow-Iridis/output/lbfgs_m_sweep/logs/lbfgs_msweep_*.out
#
#   # Cancel all tasks:
#   scancel -u $USER --jobname=lbfgs-msweep
# ============================================================================ #

# --- SLURM directives ------------------------------------------------------ #
#SBATCH --partition=amd_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=48:00:00                # 48h per task (89 ICs each)
#SBATCH --job-name=lbfgs-msweep
#SBATCH --array=1-5                    # 5 tasks: m=20,40,80,160,320
#SBATCH --output=output/lbfgs_m_sweep/logs/lbfgs_msweep_%A_%a.out
#SBATCH --error=output/lbfgs_m_sweep/logs/lbfgs_msweep_%A_%a.err

# --- Environment ----------------------------------------------------------- #
set -euo pipefail

module purge
module load julia/1.12.4

PROJ_DIR="$HOME/Open-Kolmogorov-Flow-Iridis"
cd "$PROJ_DIR"

# Map SLURM_ARRAY_TASK_ID (1..5) to lbfgs_memory
case "$SLURM_ARRAY_TASK_ID" in
    1) LBFGS_M=20  ;;
    2) LBFGS_M=40  ;;
    3) LBFGS_M=80  ;;
    4) LBFGS_M=160 ;;
    5) LBFGS_M=320 ;;
    *) echo "ERROR: Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

OUTPUT_DIR="$PROJ_DIR/output/lbfgs_m_sweep/m${LBFGS_M}"
LOGS_DIR="$PROJ_DIR/output/lbfgs_m_sweep/logs"
mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

# Fixed L-BFGS parameters (only memory varies)
MAXITER=1000
LS_MAXITER=20
E_TOL=1e-8
DZ_TOL=1e-8

echo "============================================================"
echo "L-BFGS memory sweep — Task $SLURM_ARRAY_TASK_ID/5"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID  (array $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID)"
echo "Host:         $(hostname)"
echo "Start:        $(date)"
echo "lbfgs_memory: $LBFGS_M  (history pairs)"
echo "Max iter:     $MAXITER"
echo "Julia:        $(which julia)"
echo "Julia ver:    $(julia --version)"
echo "Threads:      $SLURM_CPUS_PER_TASK"
echo "Project dir:  $PROJ_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "============================================================"

# Quick package check
echo ""
echo "--- Checking package loading ---"
julia -e 'using OpenKolmogorovFlow, Flows, NKSearch; println("  All packages OK")'

# Count ICs
MANIFEST="$PROJ_DIR/initial_conditions/perturbed/perturbation_manifest.csv"
N_ICS=$(($(wc -l < "$MANIFEST") - 1))
echo "Manifest: $N_ICS perturbed ICs"

# ========================================================================== #
# Run L-BFGS on ALL 89 ICs with this memory value
# ========================================================================== #
echo ""
echo "--- Running L-BFGS (lbfgs_memory=$LBFGS_M, $N_ICS ICs) ---"
julia -t "$SLURM_CPUS_PER_TASK" \
    scripts/test_lbfgs.jl \
    --max-test "$N_ICS" \
    --maxiter "$MAXITER" \
    --lbfgs-memory "$LBFGS_M" \
    --ls-maxiter "$LS_MAXITER" \
    --e-tol "$E_TOL" \
    --dz-tol "$DZ_TOL" \
    --perturbed-dir "$PROJ_DIR/initial_conditions/perturbed" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "End: $(date)"
echo ""
echo "Results:  $OUTPUT_DIR/lbfgs_results.csv"
echo "Solutions: $OUTPUT_DIR/"
echo "============================================================"
