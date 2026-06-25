# ============================================================================ #
# HPC TASK-LIST GENERATOR  (with --T filter)
#
# Usage:
#   julia scripts/generate_task_list.jl --T 5,10,20
#   julia scripts/generate_task_list.jl --T 40 --N-min 10 --m-min 10
#   julia scripts/generate_task_list.jl --T 5,10,20 --max-records 50 --output my_tasks.txt
# ============================================================================ #

using Printf, DelimitedFiles

const Ms = [5, 10, 20, 40, 80, 160, 320]
const Ns = [5, 10, 20, 40, 80, 160, 320]
const SLURM_MAX_ARRAY = 1000   # stay under Iridis limit of 1001

function get_rec_ids(csv_path::String, max_recs::Int)
    data = readdlm(csv_path, ',', skipstart=1)
    n = min(size(data, 1), max_recs)
    return [Int(data[row, 1]) for row in 1:n]
end

function main()
    output_file = "tasks.txt"
    max_recs    = 100
    T_filter    = Float64[]   # empty = all T (fallback)
    N_min       = 0           # 0 = no filter
    m_min       = 0           # 0 = no filter
    recurrences_dir = joinpath(@__DIR__, "..", "recurrences")

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--output" && i < length(ARGS)
            output_file = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--max-records" && i < length(ARGS)
            max_recs = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--T" && i < length(ARGS)
            T_filter = parse.(Float64, split(ARGS[i+1], ','))
            i += 2
        elseif ARGS[i] == "--N-min" && i < length(ARGS)
            N_min = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--m-min" && i < length(ARGS)
            m_min = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--recurrences-dir" && i < length(ARGS)
            recurrences_dir = ARGS[i+1]; i += 2
        else
            println(stderr, "Unknown argument: $(ARGS[i])")
            exit(1)
        end
    end

    if isempty(T_filter)
        println(stderr, "ERROR: --T is required.  Example: --T 5,10,20")
        exit(1)
    end

    # Apply N/m filters
    Ns_use = filter(n -> n >= N_min, Ns)
    Ms_use = filter(m -> m >= m_min, Ms)

    total_tasks = 0
    counts_by_T = Dict{Float64, Int}()

    open(output_file, "w") do io
        for T in T_filter
            T_label  = @sprintf("T%02d", round(Int, T))
            csv_path = joinpath(recurrences_dir, T_label, "recurrences.csv")

            if !isfile(csv_path)
                println("WARNING: $csv_path not found — skipping T=$T")
                continue
            end

            rec_ids = get_rec_ids(csv_path, max_recs)
            n = length(rec_ids) * length(Ns_use) * length(Ms_use)
            println("T=$T  →  $(length(rec_ids)) recs × $(length(Ns_use)) N × $(length(Ms_use)) m  =  $n tasks")

            for rec_id in rec_ids
                for N in Ns_use
                    for m in Ms_use
                        println(io, "$T $rec_id $N $m")
                        total_tasks += 1
                    end
                end
            end
            counts_by_T[T] = n
        end
    end

    println()
    println("  Task list → $output_file")
    println("  Total     → $total_tasks tasks")
    println()

    n_chunks = cld(total_tasks, SLURM_MAX_ARRAY)
    if n_chunks > 1
        println("  Split into $n_chunks chunk(s) and submit:")
        println("    split -d -l $SLURM_MAX_ARRAY $output_file chunk_")
        for chunk in 0:(n_chunks-1)
            padded = lpad(chunk, 2, '0')
            println("    sbatch --array=1-$SLURM_MAX_ARRAY first_sweep.slurm chunk_$padded")
        end
    else
        println("  Submit with:")
        println("    sbatch --array=1-$total_tasks first_sweep.slurm")
    end
    println()
end

main()
