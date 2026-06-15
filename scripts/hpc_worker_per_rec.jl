# ============================================================================ #
# HPC WORKER (per-recurrence, multi-threaded)
#
# Runs ALL (N, m) combinations for ONE recurrence on ONE node using Julia
# multi-threading for the inner (N, m) loops.
#
# Usage:
#   julia -t auto hpc_worker_per_rec.jl <T> <rec_id> [--recurrences-dir <path>] [--output-dir <path>]
#
# Example:
#   julia -t 8 hpc_worker_per_rec.jl 20.0 7
# ============================================================================ #

using Flows,
      StreamingRecurrenceAnalysis,
      ToySystems,
      NKSearch,
      LinearAlgebra,
      Printf,
      Dates,
      DelimitedFiles

# ---------------------------------------------------------------------------- #
# 0.  Parameters (keep in sync with your sweep design)
# ---------------------------------------------------------------------------- #
const Ms     = [5, 10, 20, 40, 80, 160, 320]
const Ns     = [5, 10, 20, 40, 80, 160, 320]
const MAXITER = 1_000_000

# ---------------------------------------------------------------------------- #
# 1.  Lorenz system — global constants
# ---------------------------------------------------------------------------- #
const LorenzEq = ToySystems.LorenzEq
const F_rhs   = LorenzEq.Lorenz(28.0)
const Δt      = 0.01
const MM      = RK4
const u0      = Float64[1, 1, 1]

const ϕ = flow(F_rhs, MM(u0), TimeStepConstant(Δt))

# ---------------------------------------------------------------------------- #
# 2.  Parse command-line arguments
# ---------------------------------------------------------------------------- #
function parse_args()
    if length(ARGS) < 2
        println(stderr, "Usage: julia -t N hpc_worker_per_rec.jl <T> <rec_id> [--recurrences-dir <path>] [--output-dir <path>]")
        exit(2)
    end

    T      = parse(Float64, ARGS[1])
    rec_id = parse(Int, ARGS[2])

    script_dir = @__DIR__
    recurrences_dir = joinpath(script_dir, "..", "recurrences")
    output_dir      = joinpath(script_dir, "..", "output")

    i = 3
    while i <= length(ARGS)
        if ARGS[i] == "--recurrences-dir" && i < length(ARGS)
            recurrences_dir = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--output-dir" && i < length(ARGS)
            output_dir = ARGS[i+1]
            i += 2
        else
            println(stderr, "Unknown argument: $(ARGS[i])")
            exit(2)
        end
    end

    return T, rec_id, recurrences_dir, output_dir
end

# ---------------------------------------------------------------------------- #
# 3.  Load recurrence from CSV
# ---------------------------------------------------------------------------- #
function load_one_recurrence(csv_path::String, rec_id::Int)
    data = readdlm(csv_path, ',', skipstart=1)
    for row in 1:size(data, 1)
        if Int(data[row, 1]) == rec_id
            u_rec   = Float64[data[row, 2], data[row, 3], data[row, 4]]
            T_guess = Float64(data[row, 5])
            return u_rec, T_guess
        end
    end
    error("rec_id=$rec_id not found in $csv_path")
end

# ---------------------------------------------------------------------------- #
# 4.  Build linearised flow operators
# ---------------------------------------------------------------------------- #
function build_linearised_flows()
    D_lin = LorenzEq.LorenzLin(false, 28.0)
    L_flow = flow(D_lin,
                  RK4(zeros(3), Flows.DiscreteMode(false)),
                  TimeStepFromCache())

    D_adj_raw = LorenzEq.LorenzLin(true, 28.0)
    L_adj_flow = flow(D_adj_raw,
                      RK4(zeros(3), Flows.DiscreteMode(true)),
                      TimeStepFromCache())

    phase_lock = (dxdt, x) -> F_rhs(0, x, dxdt)
    return L_flow, L_adj_flow, phase_lock
end

# ---------------------------------------------------------------------------- #
# 5.  Save per-iteration history as CSV
# ---------------------------------------------------------------------------- #
function save_history_csv(outdir::String, rec_id::Int, N::Int, m::Int,
                          iter, e_norm, grad_norm, lambda)
    fname = @sprintf("N%02d_m%02d.csv", N, m)
    rec_dir = joinpath(outdir, @sprintf("rec%03d", rec_id))
    mkpath(rec_dir)
    fpath = joinpath(rec_dir, fname)

    nrows = length(iter)
    open(fpath, "w") do io
        println(io, "iter,e_norm,grad_norm,lambda")
        for row in 1:nrows
            @printf(io, "%d,%.16e,%.16e,%.16e\n",
                    iter[row], e_norm[row], grad_norm[row], lambda[row])
        end
    end
    return fpath
end

# ---------------------------------------------------------------------------- #
# 6.  Run L-BFGS for one (N, m) combination
# ---------------------------------------------------------------------------- #
function run_one_case(T::Float64, rec_id::Int, N::Int, m::Int,
                      u_rec, T_guess,
                      L_flow, L_adj_flow, phase_lock,
                      data_dir::String)
    # Build initial guess z0
    z0_seeds = [copy(u_rec) for _ in 1:N]
    for i in 2:N
        ϕ(z0_seeds[i], (0, (i - 1) * T_guess / N))
    end
    z0 = MVector((copy.(z0_seeds)...,), T_guess)

    # Deep-copy flows for isolation
    Gs     = ntuple(_ -> deepcopy(ϕ), N)
    Ls     = ntuple(_ -> deepcopy(L_flow), N)
    Ls_adj = ntuple(_ -> deepcopy(L_adj_flow), N)

    # History buffers
    history_iter      = Int[]
    history_e_norm    = Float64[]
    history_grad_norm = Float64[]
    history_lambda    = Float64[]

    cb = (iter, z, Fz, f_norm, ∇ϕ_norm, λ) -> begin
        push!(history_iter,      iter)
        push!(history_e_norm,    f_norm)
        push!(history_grad_norm, ∇ϕ_norm)
        push!(history_lambda,    λ)
        return false
    end

    opts = Options(
        method           = :lbfgs_opt,
        maxiter          = MAXITER,
        dz_norm_tol      = 0.0,
        e_norm_tol       = 1e-8,
        verbose          = false,
        ls_maxiter       = 30,
        lbfgs_memory     = m,
        lbfgs_adj_system = Ls_adj,
        callback         = cb,
    )

    t_start = time()
    local final_normF, converged, n_iter, elapsed, error_msg

    try
        NKSearch._search!(Gs, Ls, nothing, (phase_lock,), z0, opts)
    catch err
        elapsed = time() - t_start
        save_history_csv(data_dir, rec_id, N, m, history_iter, history_e_norm,
                         history_grad_norm, history_lambda)
        @printf("  FAIL N=%d m=%d  %s\n", N, m, err)
        return (converged=false, final_T=NaN, final_normF=NaN,
                n_iter=length(history_iter), elapsed=elapsed,
                error_msg=sprint(showerror, err))
    end

    elapsed = time() - t_start

    # Final residual
    fwd_tmp = NKSearch.IterSolCache(Gs, Ls, nothing, nothing, (phase_lock,), z0)
    b_tmp   = similar(z0)
    NKSearch.update!(fwd_tmp, b_tmp, z0)
    final_normF = norm(b_tmp)

    push!(history_iter,      length(history_iter))
    push!(history_e_norm,    final_normF)
    push!(history_grad_norm, NaN)
    push!(history_lambda,    NaN)

    converged = final_normF < 1e-8
    n_iter    = length(history_iter) - 1

    fpath = save_history_csv(data_dir, rec_id, N, m, history_iter, history_e_norm,
                             history_grad_norm, history_lambda)

    status = converged ? "✓" : "✗"
    @printf("  %s N=%3d m=%3d  ‖F‖=%.2e  T=%.6f  %4d it  %6.1f s\n",
            status, N, m, final_normF, z0.d[1], n_iter, elapsed)

    return (converged=converged, final_T=z0.d[1], final_normF=final_normF,
            n_iter=n_iter, elapsed=elapsed, error_msg="")
end

# ---------------------------------------------------------------------------- #
# 7.  Main — run all (N, m) combos for the given recurrence
# ---------------------------------------------------------------------------- #
function main()
    T, rec_id, recurrences_dir, output_dir = parse_args()

    T_label  = @sprintf("T%02d", round(Int, T))
    csv_path = joinpath(recurrences_dir, T_label, "recurrences.csv")

    if !isfile(csv_path)
        println(stderr, "ERROR: Recurrence file not found: $csv_path")
        exit(2)
    end

    u_rec, T_guess = load_one_recurrence(csv_path, rec_id)
    data_dir = joinpath(output_dir, T_label, "data")

    # Warm up the flow cache
    warmup = copy(u_rec)
    ϕ(warmup, (0, 10 * Δt))

    # Build linearised flows (shared across all N, m)
    L_flow, L_adj_flow, phase_lock = build_linearised_flows()

    # Print header
    println("="^72)
    println("  HPC Worker — Per-Recurrence (multi-threaded)")
    println("  T        = $T")
    println("  rec_id   = $rec_id")
    println("  T_guess  = $T_guess")
    println("  u_rec    = [$(round(u_rec[1], digits=6)), $(round(u_rec[2], digits=6)), $(round(u_rec[3], digits=6))]")
    println("  N values = $Ns")
    println("  m values = $Ms")
    println("  Threads  = $(Threads.nthreads())")
    println("  maxiter  = $MAXITER")
    println("  output   = $data_dir")
    println("  started  = $(now())")
    println("="^72)
    println()

    # Generate all (N, m) combos
    combos = [(N, m) for N in Ns for m in Ms]
    n_combos = length(combos)
    println("Total combinations to run: $n_combos")
    println()

    # Thread-safe results collection
    results_lock = ReentrantLock()
    results = Dict{Tuple{Int,Int}, NamedTuple}()

    overall_start = time()

    # --- Run all combos in parallel using @threads ---
    Threads.@threads for idx in 1:n_combos
        N, m = combos[idx]
        res = run_one_case(T, rec_id, N, m, u_rec, T_guess,
                           L_flow, L_adj_flow, phase_lock, data_dir)
        lock(results_lock) do
            results[(N, m)] = res
        end
    end

    overall_elapsed = time() - overall_start

    # --- Print summary table --------------------------------------------------
    println()
    println("="^72)
    println("  Summary for T=$T  rec_id=$rec_id")
    println("  Total wall time: $(round(overall_elapsed, digits=1)) s")
    println()
    println(rpad("", 8), join([rpad("m=$m", 20) for m in Ms], ""))
    for N in Ns
        print(rpad("N=$N", 8))
        for m in Ms
            r = results[(N, m)]
            c = r.converged ? "✓" : "✗"
            s = @sprintf "%s ‖F‖=%.2e" c r.final_normF
            print(rpad(s, 20))
        end
        println()
    end
    println("="^72)
    println()

    # Exit code based on whether ALL converged
    all_converged = all(r -> r.converged, values(results))
    println("Finished at $(now())")
    exit(all_converged ? 0 : 1)
end

main()
