# ============================================================================ #
# RERUN TASK-LIST GENERATOR
#
# Reads  cases_to_rerun.csv  and generates  rerun_tasks.txt  with format:
#     <T> <rec_id> <N> <m>
#
# Usage:
#   julia scripts/generate_rerun_tasks.jl --T 40 --rec-min 4 --rec-max 84
#   julia scripts/generate_rerun_tasks.jl   # no filters → all cases
# ============================================================================ #

using Printf, DelimitedFiles

const SLURM_MAX_ARRAY = 1000   # Iridis limit is 1001; stay under

function main()
    csv_path = joinpath(@__DIR__, "..", "cases_to_rerun.csv")
    output_file = joinpath(@__DIR__, "..", "rerun_tasks.txt")

    # --- Filters (empty = no filter) ---
    T_filter = Float64[]      # T values to include
    rec_min  = 0              # 0 = no lower bound
    rec_max  = typemax(Int)   # no upper bound

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--T" && i < length(ARGS)
            T_filter = parse.(Float64, split(ARGS[i+1], ','))
            i += 2
        elseif ARGS[i] == "--rec-min" && i < length(ARGS)
            rec_min = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--rec-max" && i < length(ARGS)
            rec_max = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--output" && i < length(ARGS)
            output_file = ARGS[i+1]; i += 2
        else
            println(stderr, "Unknown argument: $(ARGS[i])")
            exit(1)
        end
    end

    if !isfile(csv_path)
        println(stderr, "ERROR: $csv_path not found")
        exit(1)
    end

    data = readdlm(csv_path, ',', skipstart=1)
    n_rows = size(data, 1)

    total = 0
    skipped = 0
    filtered = 0
    open(output_file, "w") do io
        for row in 1:n_rows
            status = strip(String(data[row, 6]))

            # Only rerun cases that NEVER produced output.
            if status != "incomplete" && status != "no_data"
                skipped += 1
                continue
            end

            T_str  = strip(String(data[row, 2]))
            rec_id = Int(data[row, 3])
            N      = Int(data[row, 4])
            m      = Int(data[row, 5])
            T_val  = parse(Float64, T_str[2:end])

            # --- Apply filters ---
            if !isempty(T_filter) && !(T_val in T_filter)
                filtered += 1; continue
            end
            if rec_id < rec_min || rec_id > rec_max
                filtered += 1; continue
            end

            # For T=80 and T=160: skip small (N,m) combos that don't converge.
            if (T_val == 80.0 || T_val == 160.0) && N < 40 && m < 40
                filtered += 1
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
    if filtered > 0
        println("  Filtered out:    $filtered  (--T / --rec-min / --rec-max / small N,m)")
    end
    println()

    # Print chunked submission commands
    n_chunks = cld(total, SLURM_MAX_ARRAY)
    if n_chunks > 1
        println("  Split into $n_chunks chunk(s) and submit:")
        println("    split -d -l $SLURM_MAX_ARRAY $output_file rerun_chunk_")
        for chunk in 0:(n_chunks-1)
            padded = lpad(chunk, 2, '0')
            println("    sbatch --array=1-$SLURM_MAX_ARRAY first_sweep.slurm rerun_chunk_$padded")
        end
    else
        println("  Submit with:")
        println("    sbatch --array=1-$total first_sweep.slurm $output_file")
    end
    println("="^60)
end

main()
