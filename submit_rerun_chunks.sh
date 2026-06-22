#!/bin/bash
# ============================================================================ #
# RERUN CHUNK SUBMITTER
#
# Splits rerun_tasks.txt into chunks of 1000 lines and submits each as a
# separate --array=1-<N> job.  Works around Iridis rejecting array indices > 1000.
#
# Usage:
#   bash submit_rerun_chunks.sh
# ============================================================================ #

set -euo pipefail

TASK_FILE="rerun_tasks.txt"
CHUNK_SIZE=1000
SLURM_SCRIPT="submit_rerun.slurm"

if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE not found."
    echo "       Run:  julia scripts/generate_rerun_tasks.jl"
    exit 1
fi

TOTAL=$(wc -l < "$TASK_FILE")
CHUNKS=$(( (TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "Total tasks:   $TOTAL"
echo "Chunk size:    $CHUNK_SIZE"
echo "Num chunks:    $CHUNKS"
echo ""

# Process chunk by chunk
for ((i=0; i<CHUNKS; i++)); do
    START=$(( i * CHUNK_SIZE + 1 ))
    END=$(( START + CHUNK_SIZE - 1 ))
    if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi
    SIZE=$(( END - START + 1 ))

    CHUNK_FILE="rerun_chunk_$(printf "%03d" $i).txt"
    sed -n "${START},${END}p" "$TASK_FILE" > "$CHUNK_FILE"

    # Create a temporary symlink so submit_rerun.slurm reads this chunk
    ln -sf "$CHUNK_FILE" rerun_current_chunk.txt

    # Modify a copy of the slurm script to use the chunk file
    sed "s/rerun_tasks.txt/rerun_current_chunk.txt/" "$SLURM_SCRIPT" > submit_rerun_chunk.slurm

    echo "Submitting chunk $((i+1))/$CHUNKS  ($SIZE tasks, lines $START-$END) ..."
    JOBID=$(sbatch --parsable --array=1-"$SIZE" submit_rerun_chunk.slurm)
    echo "  Job ID: $JOBID"

    # Small delay to avoid hammering the scheduler
    sleep 1
done

# Clean up temp script
rm -f submit_rerun_chunk.slurm

echo ""
echo "All $CHUNKS chunks submitted."
echo "Monitor:  squeue -u \$USER | grep lorenz-rerun"
