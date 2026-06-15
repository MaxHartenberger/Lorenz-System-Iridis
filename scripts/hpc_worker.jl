# ============================================================================ #
# HPC WORKER — Run a SINGLE (T, rec_id, N, m) L-BFGS case
#
# Designed to be launched by a SLURM job array.  Each array task calls this
# script with the parameters for exactly one case, runs L-BFGS, and saves the
# per-iteration CSV.
#
# Usage:
#   julia hpc_worker.jl <T> <rec_id> <N> <m> [--recurrences-dir <path>] [--output-dir <path>]
#
# Example:
#   julia hpc_worker.jl 5.0 3 10 5
#   julia hpc_worker.jl 20.0 7 20 10 --output-dir /scratch/user/output
#
# Exit codes:
#   0 – converged
#   1 – did not converge
#   2 – error / exception
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# 0.  Dependencies
# ---------------------------------------------------------------------------- #
using Flows,
      StreamingRecurrenceAnalysis,
      ToySystems,
      NKSearch,
      LinearAlgebra,
      Printf,
      Dates,
      DelimitedFiles

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
    if length(ARGS) < 4
        println(stderr, "Usage: julia hpc_worker.jl <T> <rec_id> <N> <m> [--recurrences-dir <path>] [--output-dir <path>] [--maxiter <n>]")
        exit(2)
    end

    T         = parse(Float64, ARGS[1])
    rec_id    = parse(Int, ARGS[2])
    N         = parse(Int, ARGS[3])
    m         = parse(Int, ARGS[4])

    # Default paths (relative to this script's location)
    script_dir = @__DIR__
    recurrences_dir = joinpath(script_dir, "..", "recurrences")
    output_dir      = joinpath(script_dir, "..", "output")
    maxiter         = 1_000_000

    # Parse optional flags
    i = 5
    while i <= length(ARGS)
        if ARGS[i] == "--recurrences-dir" && i < length(ARGS)
            recurrences_dir = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--output-dir" && i < length(ARGS)
            output_dir = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--maxiter" && i < length(ARGS)
            maxiter = parse(Int, ARGS[i+1])
            i += 2
        else
            println(stderr, "Unknown argument: $(ARGS[i])")
            exit(2)
        end
    end

    return T, rec_id, N, m, recurrences_dir, output_dir, maxiter
end

# ---------------------------------------------------------------------------- #
# 3.  Load a specific recurrence from CSV
# ---------------------------------------------------------------------------- #
"""
    load_one_recurrence(csv_path::String, rec_id::Int)

Reads the recurrences CSV and returns `(u_rec, T_guess)` for the row
whose `rec_id` matches.  Throws an error if not found.
"""
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
# 4.  Build linearised & adjoint flow operators
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
                          iter::Vector{Int},
                          e_norm::Vector{Float64},
                          grad_norm::Vector{Float64},
                          lambda::Vector{Float64})
    fname = @sprintf("N%02d_m%02d.csv", N, m)
    # If multiple rec_ids share the same (N,m), include rec_id in the filename
    # to avoid collisions — this worker is designed for one rec_id per call.
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
# 6.  Run L-BFGS for a SINGLE (rec_id, N, m) combination
# ---------------------------------------------------------------------------- #
function run_single_case(T::Float64, rec_id::Int, N::Int, m::Int,
                         u_rec, T_guess,
                         L_flow, L_adj_flow, phase_lock,
                         data_dir::String, maxiter::Int)

    # --- 6a.  Build initial guess z0 ------------------------------------------
    z0_seeds = [copy(u_rec) for _ in 1:N]
    for i in 2:N
        ϕ(z0_seeds[i], (0, (i - 1) * T_guess / N))
    end
    z0 = MVector((copy.(z0_seeds)...,), T_guess)

    # --- 6b.  Build per-segment flow tuples -----------------------------------
    Gs     = ntuple(_ -> deepcopy(ϕ), N)
    Ls     = ntuple(_ -> deepcopy(L_flow), N)
    Ls_adj = ntuple(_ -> deepcopy(L_adj_flow), N)

    # --- 6c.  Callback for per-iteration history ------------------------------
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

    # --- 6d.  Options ---------------------------------------------------------
    opts = Options(
        method           = :lbfgs_opt,
        maxiter          = maxiter,
        dz_norm_tol      = 0.0,
        e_norm_tol       = 1e-8,
        verbose          = false,
        ls_maxiter       = 30,
        lbfgs_memory     = m,
        lbfgs_adj_system = Ls_adj,
        callback         = cb,
    )

    # --- 6e.  Run the L-BFGS optimisation -------------------------------------
    t_start = time()
    local final_normF, converged, n_iter, elapsed, error_msg

    try
        NKSearch._search!(Gs, Ls, nothing, (phase_lock,), z0, opts)
    catch err
        elapsed = time() - t_start
        save_history_csv(data_dir, rec_id, N, m, history_iter, history_e_norm,
                         history_grad_norm, history_lambda)
        println("FAILED: $err")
        return (converged=false, final_T=NaN, final_normF=NaN,
                n_iter=length(history_iter), elapsed=elapsed, error_msg=sprint(showerror, err))
    end

    elapsed = time() - t_start

    # --- 6f.  Compute final residual ------------------------------------------
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

    # --- 6g.  Save CSV --------------------------------------------------------
    fpath = save_history_csv(data_dir, rec_id, N, m, history_iter, history_e_norm,
                             history_grad_norm, history_lambda)

    # --- 6h.  Status line -----------------------------------------------------
    status = converged ? "CONVERGED" : "DID NOT CONVERGE"
    @printf("T=%.1f  rec_id=%d  N=%d  m=%d  |  %s  ‖F‖=%.3e  T_final=%.6f  %d it  %.1f s  → %s\n",
            T, rec_id, N, m, status, final_normF, z0.d[1], n_iter, elapsed, fpath)

    return (converged=converged, final_T=z0.d[1], final_normF=final_normF,
            n_iter=n_iter, elapsed=elapsed, error_msg=error_msg)
end

# ---------------------------------------------------------------------------- #
# 7.  Main
# ---------------------------------------------------------------------------- #
function main()
    # Parse command line
    T, rec_id, N, m, recurrences_dir, output_dir, maxiter = parse_args()

    T_label  = @sprintf("T%02d", round(Int, T))
    csv_path = joinpath(recurrences_dir, T_label, "recurrences.csv")

    if !isfile(csv_path)
        println(stderr, "ERROR: Recurrence file not found: $csv_path")
        exit(2)
    end

    # Load the specific recurrence
    u_rec, T_guess = load_one_recurrence(csv_path, rec_id)

    # Set up output directory
    data_dir = joinpath(output_dir, T_label, "data")
    mkpath(data_dir)

    # Build linearised flows (must happen after ϕ has been used to build its cache)
    # We warm up ϕ first so TimeStepFromCache has something to refer to.
    warmup = copy(u_rec)
    ϕ(warmup, (0, 10 * Δt))   # ensure the flow cache is populated
    L_flow, L_adj_flow, phase_lock = build_linearised_flows()

    # Print job header
    println("="^72)
    println("  HPC Worker — Single L-BFGS Case")
    println("  T       = $T")
    println("  rec_id  = $rec_id")
    println("  N       = $N  (shooting segments)")
    println("  m       = $m  (L-BFGS memory)")
    println("  T_guess = $T_guess")
    println("  u_rec   = [$(round(u_rec[1], digits=6)), $(round(u_rec[2], digits=6)), $(round(u_rec[3], digits=6))]")
    println("  maxiter = $maxiter")
    println("  output  = $data_dir")
    println("  started = $(now())")
    println("="^72)
    println()

    # Run the case
    result = run_single_case(T, rec_id, N, m, u_rec, T_guess,
                             L_flow, L_adj_flow, phase_lock,
                             data_dir, maxiter)

    println()
    println("Finished at $(now())")

    # Exit with appropriate code for SLURM
    if result.error_msg != ""
        exit(2)      # error
    elseif result.converged
        exit(0)      # success — converged
    else
        exit(1)      # ran to completion but didn't converge
    end
end

main()
