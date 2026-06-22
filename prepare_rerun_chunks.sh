#!/bin/bash
# ============================================================================ #
# PREPARE RERUN CHUNKS
#
# Splits rerun_tasks.txt into chunk files and generates one .slurm file per
# chunk.  Each .slurm is self-contained — no symlinks, no sed, no tricks.
#
# Run ONCE to prepare, then submit each chunk MANUALLY.
#
# Usage:
#   bash prepare_rerun_chunks.sh
#   # Then submit the generated .slurm files one at a time:
#   sbatch --array=1-1000 submit_rerun_001.slurm
#   sbatch --array=1-1000 submit_rerun_002.slurm
#   ...
# ============================================================================ #

set -euo pipefail

TASK_FILE="rerun_tasks.txt"
CHUNK_SIZE=1000

if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE not found.  Run: julia scripts/generate_rerun_tasks.jl"
    exit 1
fi

TOTAL=$(wc -l < "$TASK_FILE")
CHUNKS=$(( (TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "Total tasks:  $TOTAL"
echo "Chunk size:   $CHUNK_SIZE"
echo "Num chunks:   $CHUNKS"
echo ""
echo "Preparing chunk files and .slurm scripts..."
echo ""

for ((i=0; i<CHUNKS; i++)); do
    START=$(( i * CHUNK_SIZE + 1 ))
    END=$(( START + CHUNK_SIZE - 1 ))
    if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi
    SIZE=$(( END - START + 1 ))

    CHUNK_NUM=$(printf "%03d" $((i+1)))
    CHUNK_FILE="rerun_chunk_${CHUNK_NUM}.txt"
    SLURM_FILE="submit_rerun_${CHUNK_NUM}.slurm"

    # Extract lines for this chunk
    sed -n "${START},${END}p" "$TASK_FILE" > "$CHUNK_FILE"

    # Create a standalone .slurm file for this chunk
    cat > "$SLURM_FILE" << 'SLURM_EOF'
#!/bin/bash
# ============================================================================ #
# SLURM Job Array — Rerun CHUNK_FILE_PLACEHOLDER
# Reads CHUNK_FILE_PLACEHOLDER
# ============================================================================ #

#SBATCH --job-name=lorenz-rerun
#SBATCH --output=logs/rerun_slurm_%A_%a.out
#SBATCH --error=logs/rerun_slurm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500
#SBATCH --time=12:00:00
#SBATCH --partition=amd_student

module load julia/1.12.4
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load Julia module 'julia/1.12.4'."
    exit 2
fi

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT" || exit 2
mkdir -p logs

TASK_LIST="CHUNK_FILE_PLACEHOLDER"

if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: $TASK_LIST not found."
    exit 2
fi

TOTAL_TASKS=$(wc -l < "$TASK_LIST")
if [ "$SLURM_ARRAY_TASK_ID" -gt "$TOTAL_TASKS" ]; then
    echo "WARNING: Array task $SLURM_ARRAY_TASK_ID > $TOTAL_TASKS. Exiting."
    exit 0
fi

PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")
set -- $PARAMS
T="$1"; REC_ID="$2"; N="$3"; M="$4"

if [ -z "$T" ] || [ -z "$REC_ID" ] || [ -z "$N" ] || [ -z "$M" ]; then
    echo "ERROR: Failed to parse line $SLURM_ARRAY_TASK_ID"
    exit 2
fi

echo "=============================================="
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Array Task ID:   ${SLURM_ARRAY_TASK_ID} / ${TOTAL_TASKS}"
echo "T=${T}, rec_id=${REC_ID}, N=${N}, m=${M}"
echo "Started:  $(date)"
echo "Node:     $(hostname)"
echo "=============================================="
echo ""

julia scripts/hpc_worker.jl "$T" "$REC_ID" "$N" "$M"
EXIT_CODE=$?

echo ""
echo "Finished: $(date)  exit=$EXIT_CODE"
exit $EXIT_CODE
SLURM_EOF

    # Replace placeholder with actual chunk filename
    sed -i "s/CHUNK_FILE_PLACEHOLDER/$CHUNK_FILE/g" "$SLURM_FILE"

    echo "  $SLURM_FILE  ($SIZE tasks, lines $START-$END)"
done

echo ""
echo "============================================================"
echo "  All files prepared.  Submit manually:"
echo ""
for ((i=1; i<=CHUNKS; i++)); do
    CHUNK_NUM=$(printf "%03d" $i)
    echo "  sbatch --array=1-1000 submit_rerun_${CHUNK_NUM}.slurm"
done
echo ""
echo "  (The last chunk may have fewer than 1000 tasks — adjust --array=1-<N>)"
echo "============================================================"
