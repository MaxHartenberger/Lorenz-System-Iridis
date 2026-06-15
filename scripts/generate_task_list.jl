# ============================================================================ #
# HPC TASK-LIST GENERATOR
#
# Generates a plain-text file `tasks.txt` where each line contains the
# parameters for one L-BFGS case:
#
#     <T> <rec_id> <N> <m>
#
# This file is then consumed by the SLURM job array: array task ID K reads
# line K (1-indexed) and launches `hpc_worker.jl` with those arguments.
#
# Usage:
#   julia generate_task_list.jl [--output tasks.txt] [--max-records 100]
#
# The script also prints a summary: total number of tasks, breakdown by T, etc.
# ============================================================================ #

using Printf, DelimitedFiles

# ---------------------------------------------------------------------------- #
# 0.  Parameters  (keep in sync with hpc_worker.jl and your sweep design)
# ---------------------------------------------------------------------------- #
const T_targets = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
const Ms        = [5, 10, 20, 40, 80, 160, 320]
const Ns        = [5, 10, 20, 40, 80, 160, 320]

# ---------------------------------------------------------------------------- #
# 1.  Count recurrences in a CSV file
# ---------------------------------------------------------------------------- #
function count_recurrences(csv_path::String)
    if !isfile(csv_path)
        return 0
    end
    data = readdlm(csv_path, ',', skipstart=1)
    return size(data, 1)
end

# ---------------------------------------------------------------------------- #
# 2.  Get rec_ids present in a recurrences CSV
# ---------------------------------------------------------------------------- #
function get_rec_ids(csv_path::String, max_recs::Int)
    data = readdlm(csv_path, ',', skipstart=1)
    n = min(size(data, 1), max_recs)
    return [Int(data[row, 1]) for row in 1:n]
end

# ---------------------------------------------------------------------------- #
# 3.  Main
# ---------------------------------------------------------------------------- #
function main()
    # --- Parse args -----------------------------------------------------------
    output_file = "tasks.txt"
    max_recs    = 100
    recurrences_dir = joinpath(@__DIR__, "..", "recurrences")

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--output" && i < length(ARGS)
            output_file = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--max-records" && i < length(ARGS)
            max_recs = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--recurrences-dir" && i < length(ARGS)
            recurrences_dir = ARGS[i+1]
            i += 2
        else
            println(stderr, "Unknown argument: $(ARGS[i])")
            exit(1)
        end
    end

    # --- Generate task list ---------------------------------------------------
    total_tasks = 0
    counts_by_T = Dict{Float64, Int}()

    open(output_file, "w") do io
        for T in T_targets
            T_label   = @sprintf("T%02d", round(Int, T))
            csv_path  = joinpath(recurrences_dir, T_label, "recurrences.csv")

            if !isfile(csv_path)
                println("WARNING: $csv_path not found — skipping T=$T")
                continue
            end

            rec_ids = get_rec_ids(csv_path, max_recs)
            n_recs  = length(rec_ids)

            println("T=$T  →  $(n_recs) recurrences × $(length(Ns)) N × $(length(Ms)) m  =  $(n_recs * length(Ns) * length(Ms)) tasks")

            for rec_id in rec_ids
                for N in Ns
                    for m in Ms
                        println(io, "$T $rec_id $N $m")
                        total_tasks += 1
                    end
                end
            end

            counts_by_T[T] = n_recs * length(Ns) * length(Ms)
        end
    end

    # --- Summary --------------------------------------------------------------
    println()
    println("="^60)
    println("  Task list written to:  $output_file")
    println("  Total tasks:           $total_tasks")

    # Warn if exceeds typical SLURM MaxArraySize (1001 on Iridis)
    const SLURM_MAX_ARRAY = 1001
    if total_tasks > SLURM_MAX_ARRAY
        println()
        println("  ⚠ WARNING: $total_tasks tasks exceeds SLURM MaxArraySize ($SLURM_MAX_ARRAY).")
        println("    You CANNOT submit this as a single job array.")
        println()
        println("    Options:")
        println("      A) Use the per-recurrence strategy instead:")
        println("           julia scripts/generate_rec_task_list.jl --output rec_tasks.txt")
        println("           sbatch --array=1-\$(wc -l < rec_tasks.txt) submit_per_recurrence.slurm")
        println("      B) Split into chunks of $SLURM_MAX_ARRAY:")
        for chunk_start in 1:SLURM_MAX_ARRAY:total_tasks
            chunk_end = min(chunk_start + SLURM_MAX_ARRAY - 1, total_tasks)
            println("           sbatch --array=$chunk_start-$chunk_end submit_sweep.slurm")
        end
        println()
    end

    println()
    println("  Breakdown by T:")
    for T in T_targets
        n = get(counts_by_T, T, 0)
        if n > 0
            println("    T=$T  →  $n tasks")
        end
    end
    println()
    if total_tasks <= SLURM_MAX_ARRAY
        println("  Submit with:")
        println("    sbatch --array=1-$total_tasks submit_sweep.slurm")
    end
    println("="^60)
end

main()
