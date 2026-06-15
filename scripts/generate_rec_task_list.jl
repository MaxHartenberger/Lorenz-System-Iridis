# ============================================================================ #
# HPC RECURRENCE TASK-LIST GENERATOR
#
# Generates `rec_tasks.txt` where each line is:
#
#     <T> <rec_id>
#
# One line per recurrence.  Used by submit_per_recurrence.slurm.
#
# Usage:
#   julia generate_rec_task_list.jl [--output rec_tasks.txt] [--max-records 100]
# ============================================================================ #

using Printf, DelimitedFiles

const T_targets = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]

function get_rec_ids(csv_path::String, max_recs::Int)
    data = readdlm(csv_path, ',', skipstart=1)
    n = min(size(data, 1), max_recs)
    return [Int(data[row, 1]) for row in 1:n]
end

function main()
    output_file = "rec_tasks.txt"
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

    total_tasks = 0

    open(output_file, "w") do io
        for T in T_targets
            T_label  = @sprintf("T%02d", round(Int, T))
            csv_path = joinpath(recurrences_dir, T_label, "recurrences.csv")

            if !isfile(csv_path)
                println("WARNING: $csv_path not found — skipping T=$T")
                continue
            end

            rec_ids = get_rec_ids(csv_path, max_recs)
            println("T=$T  →  $(length(rec_ids)) recurrence(s)")

            for rec_id in rec_ids
                println(io, "$T $rec_id")
                total_tasks += 1
            end
        end
    end

    println()
    println("="^60)
    println("  Recurrence task list → $output_file")
    println("  Total tasks:         $total_tasks")
    println()
    println("  Submit with:")
    println("    sbatch --array=1-$total_tasks submit_per_recurrence.slurm")
    println("="^60)
end

main()
