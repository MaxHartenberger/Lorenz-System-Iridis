# ============================================================================ #
# RERUN TASK-LIST GENERATOR
#
# Reads  cases_to_rerun.csv  and generates  rerun_tasks.txt  with format:
#     <T> <rec_id> <N> <m>
#
# One line per case.  Also prints submission commands (chunked for MaxArraySize).
#
# Usage:
#   julia scripts/generate_rerun_tasks.jl
# ============================================================================ #

using Printf, DelimitedFiles

const SLURM_MAX_ARRAY = 1000   # Iridis limit is 1001; stay under

function main()
    csv_path = joinpath(@__DIR__, "..", "cases_to_rerun.csv")
    output_file = joinpath(@__DIR__, "..", "rerun_tasks.txt")

    if !isfile(csv_path)
        println(stderr, "ERROR: $csv_path not found")
        exit(1)
    end

    data = readdlm(csv_path, ',', skipstart=1)
    n_rows = size(data, 1)

    total = 0
    skipped = 0
    open(output_file, "w") do io
        for row in 1:n_rows
            status = strip(String(data[row, 6]))

            # Only rerun cases that NEVER produced output.
            # "hit_maxiter" and "diverged" ran to completion — the recurrence
            # simply doesn't converge; rerunning won't help.
            # "incomplete" and "no_data" are from crashed/timed-out jobs that
            # need a proper rerun.
            if status != "incomplete" && status != "no_data"
                skipped += 1
                continue
            end

            T_str  = strip(String(data[row, 2]))   # e.g. "T05"
            rec_id = Int(data[row, 3])
            N      = Int(data[row, 4])
            m      = Int(data[row, 5])

            T_val = parse(Float64, T_str[2:end])

            # For T=80 and T=160: skip small (N,m) combos that don't converge.
            # N < 40 and m < 40 are known to diverge for these long periods.
            if (T_val == 80.0 || T_val == 160.0) && N < 40 && m < 40
                skipped += 1
                continue
            end

            println(io, "$T_val $rec_id $N $m")
            total += 1
        end
    end

    println("="^60)
    println("  Rerun task list → $output_file")
    println("  Cases written:   $total")
    println("  Skipped:         $skipped  (hit_maxiter / diverged)")
    println()

    # Print chunked submission commands
    n_chunks = cld(total, SLURM_MAX_ARRAY)
    println("  Submit in $n_chunks chunk(s):")
    for chunk in 1:n_chunks
        start_idx = (chunk - 1) * SLURM_MAX_ARRAY + 1
        end_idx   = min(chunk * SLURM_MAX_ARRAY, total)
        println("    sbatch --array=$start_idx-$end_idx submit_rerun.slurm")
    end
    println("="^60)
end

main()
