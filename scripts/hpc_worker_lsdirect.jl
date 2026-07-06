# ============================================================================ #
# HPC WORKER (LS_DIRECT) — Run a SINGLE (T, rec_id, N) case
#
# Designed to be launched by a SLURM job array.  Each array task calls this
# script with the parameters for exactly one case, runs the direct Newton
# line-search method, and saves the per-iteration CSV.
#
# Usage:
#   julia hpc_worker_lsdirect.jl <T> <rec_id> <N> [--output-dir <path>] [--maxiter <n>]
#
# Example:
#   julia hpc_worker_lsdirect.jl 5.0 1 20
#   julia hpc_worker_lsdirect.jl 20.0 3 40 --output-dir /scratch/user/output
#
# Exit codes:
#   0 – converged
#   1 – did not converge
#   2 – error / exception
#
# NOTE: ls_direct only works with single thread (--cpus-per-task=1).
#
# Output saved to:  <output-dir>/T{XX}/data_lsdirect/rec{XXX}/iteration/N{XX}.csv
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

# Nonlinear flow G — same as hookstep / L-BFGS worker (NormalMode + TimeStepConstant)
const ϕ = flow(F_rhs, MM(u0), TimeStepConstant(Δt))

# ---------------------------------------------------------------------------- #
# 2.  Parse command-line arguments
# ---------------------------------------------------------------------------- #
function parse_args()
    if length(ARGS) < 3
        println(stderr, "Usage: julia hpc_worker_lsdirect.jl <T> <rec_id> <N> [--recurrences-dir <path>] [--output-dir <path>] [--maxiter <n>] [--ls-maxiter <n>]")
        exit(2)
    end

    T      = parse(Float64, ARGS[1])
    rec_id = parse(Int, ARGS[2])
    N      = parse(Int, ARGS[3])

    # Default paths (relative to this script's location)
    script_dir = @__DIR__
    recurrences_dir = joinpath(script_dir, "..", "recurrences")
    output_dir      = joinpath(script_dir, "..", "outputs")
    maxiter         = 10000       # Newton iterations
    ls_maxiter      = 30         # line-search iterations per Newton step

    # Parse optional flags
    i = 4
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
        elseif ARGS[i] == "--ls-maxiter" && i < length(ARGS)
            ls_maxiter = parse(Int, ARGS[i+1])
            i += 2
        else
            println(stderr, "Unknown argument: $(ARGS[i])")
            exit(2)
        end
    end

    return T, rec_id, N, recurrences_dir, output_dir, maxiter, ls_maxiter
end

# ---------------------------------------------------------------------------- #
# 3.  Load a specific recurrence from CSV  (same as L-BFGS / hookstep)
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
# 4.  Build forward linearised flow L  (same couple pattern as hookstep)
# ---------------------------------------------------------------------------- #
"""
    build_linearised_flow()

Constructs the forward tangent-linear flow L using the couple pattern.
Same as the hookstep worker: NormalMode + TimeStepConstant, no adjoint.
"""
function build_linearised_flow()
    D_lin = LorenzEq.LorenzLin(false, 28.0)   # 5-arg: (t, u, dudt, v, dvdt)

    L = flow(Flows.couple(F_rhs, D_lin),
             RK4(Flows.couple(zeros(3), zeros(3)), Flows.NormalMode()),
             TimeStepConstant(Δt))

    return L
end

# ---------------------------------------------------------------------------- #
# 5.  Save per-iteration history as CSV  (same format as L-BFGS / hookstep)
# ---------------------------------------------------------------------------- #
function save_history_csv(outdir::String, rec_id::Int, N::Int,
                          iter::Vector{Int},
                          e_norm::Vector{Float64},
                          grad_norm::Vector{Float64},
                          lambda::Vector{Float64},
                          T_curr::Vector{Float64})
    fname = @sprintf("N%02d.csv", N)
    rec_dir = joinpath(outdir, @sprintf("rec%03d", rec_id), "iteration")
    mkpath(rec_dir)
    fpath = joinpath(rec_dir, fname)

    nrows = length(iter)
    open(fpath, "w") do io
        println(io, "iter,e_norm,grad_norm,lambda,T_curr")
        for row in 1:nrows
            @printf(io, "%d,%.16e,%.16e,%.16e,%.16e\n",
                    iter[row], e_norm[row], grad_norm[row], lambda[row], T_curr[row])
        end
    end
    return fpath
end

# ---------------------------------------------------------------------------- #
# 5b.  Save converged trajectory as CSV
# ---------------------------------------------------------------------------- #
"""
    save_trajectory_csv(outdir, rec_id, N, z0, ϕ)

Integrates the converged multi-shooting orbit and saves all phase-space
points to a CSV with columns: `t, x, y, z, segment`.
"""
function save_trajectory_csv(outdir::String, rec_id::Int, N::Int,
                             z0, ϕ_flow)
    T_final = z0.d[1]          # converged period
    T_seg   = T_final / N       # time per shooting segment
    n_steps = round(Int, T_seg / Δt)

    rec_dir = joinpath(outdir, @sprintf("rec%03d", rec_id), "trajectory")
    mkpath(rec_dir)
    fname   = @sprintf("N%02d_trajectory.csv", N)
    fpath   = joinpath(rec_dir, fname)

    open(fpath, "w") do io
        println(io, "t,x,y,z,segment")
        for seg in 1:N
            u = copy(z0.x[seg])
            t_seg = (seg - 1) * T_seg   # global time at start of this segment
            for step in 0:n_steps
                @printf(io, "%.10e,%.16e,%.16e,%.16e,%d\n",
                        t_seg + step * Δt, u[1], u[2], u[3], seg)
                if step < n_steps
                    ϕ_flow(u, (0.0, Δt))   # integrate one Δt forward
                end
            end
        end
    end
    return fpath
end

# ---------------------------------------------------------------------------- #
# 6.  Run ls_direct for a SINGLE (rec_id, N) combination
# ---------------------------------------------------------------------------- #
function run_single_case(T::Float64, rec_id::Int, N::Int,
                         u_rec, T_guess,
                         L_flow,
                         data_dir::String, maxiter::Int, ls_maxiter::Int)

    # --- 6a.  Build initial guess z0 ------------------------------------------
    z0_seeds = [copy(u_rec) for _ in 1:N]
    for i in 2:N
        ϕ(z0_seeds[i], (0, (i - 1) * T_guess / N))
    end
    z0 = MVector((copy.(z0_seeds)...,), T_guess)

    # --- 6b.  Callback for per-iteration history ------------------------------
    # Same 7-argument signature as L-BFGS / hookstep callback.
    history_iter      = Int[]
    history_e_norm    = Float64[]
    history_grad_norm = Float64[]
    history_lambda    = Float64[]
    history_T         = Float64[]

    cb = (iter, z, Fz, e_norm, ∇ϕ_norm, λ, T_cur) -> begin
        push!(history_iter,      iter)
        push!(history_e_norm,    e_norm)
        push!(history_grad_norm, ∇ϕ_norm)
        push!(history_lambda,    λ)
        push!(history_T,         T_cur)
        return false   # never stop early
    end

    # --- 6c.  Options ---------------------------------------------------------
    opts = Options(
        method          = :ls_direct,
        maxiter         = maxiter,
        e_norm_tol      = 1e-8,
        dz_norm_tol     = 0.0,
        verbose         = false,
        skipiter        = 1,
        ls_maxiter      = ls_maxiter,

        # --- Callback ---
        callback        = cb,
    )

    # Phase-locking condition
    phase_lock = (dxdt, x) -> F_rhs(0, x, dxdt)

    # --- 6d.  Run the line-search Newton search --------------------------------
    t_start = time()
    local converged, n_iter, elapsed, error_msg, final_normF
    error_msg = ""

    try
        search!(ϕ, L_flow, phase_lock, z0, opts)
    catch err
        elapsed = time() - t_start
        fpath_crash = save_history_csv(data_dir, rec_id, N,
                                       history_iter, history_e_norm,
                                       history_grad_norm, history_lambda, history_T)
        open(fpath_crash, "a") do io
            println(io, "# crashed")
        end
        println("FAILED: $err")
        return (converged=false, final_T=NaN, final_normF=NaN,
                n_iter=length(history_iter), elapsed=elapsed, error_msg=sprint(showerror, err))
    end

    elapsed = time() - t_start

    # --- 6e.  Determine convergence from final callback e_norm ----------------
    n_iter      = length(history_iter)
    final_normF = n_iter > 0 ? history_e_norm[end] : NaN
    converged   = n_iter > 0 && final_normF < 1e-8

    # --- 6f.  Save CSV --------------------------------------------------------
    fpath = save_history_csv(data_dir, rec_id, N,
                             history_iter, history_e_norm,
                             history_grad_norm, history_lambda, history_T)

    # Append convergence status as a comment line (safe for CSV readers)
    open(fpath, "a") do io
        println(io, converged ? "# converged" : "# did_not_converge")
    end

    # --- 6g.  Save trajectory if converged ------------------------------------
    if converged
        save_trajectory_csv(data_dir, rec_id, N, z0, ϕ)
    end

    # --- 6h.  Status line -----------------------------------------------------
    status = converged ? "CONVERGED" : "DID NOT CONVERGE"
    @printf("T=%.1f  rec_id=%d  N=%d  |  %s  ‖F‖=%.3e  T_final=%.6f  %d it  %.1f s  → %s\n",
            T, rec_id, N, status, final_normF, z0.d[1], n_iter, elapsed, fpath)

    return (converged=converged, final_T=z0.d[1], final_normF=final_normF,
            n_iter=n_iter, elapsed=elapsed, error_msg=error_msg)
end

# ---------------------------------------------------------------------------- #
# 7.  Main
# ---------------------------------------------------------------------------- #
function main()
    # Parse command line
    T, rec_id, N, recurrences_dir, output_dir, maxiter, ls_maxiter = parse_args()

    T_label  = @sprintf("T%02d", round(Int, T))
    csv_path = joinpath(recurrences_dir, T_label, "recurrences.csv")

    if !isfile(csv_path)
        println(stderr, "ERROR: Recurrence file not found: $csv_path")
        exit(2)
    end

    # Load the specific recurrence
    u_rec, T_guess = load_one_recurrence(csv_path, rec_id)

    # Set up output directory
    data_dir = joinpath(output_dir, T_label, "data_lsdirect")
    mkpath(data_dir)

    # Build linearised flow (same couple pattern as hookstep)
    # Warm up ϕ first so TimeStepConstant cache is populated
    warmup = copy(u_rec)
    ϕ(warmup, (0, 10 * Δt))
    L_flow = build_linearised_flow()

    # Print job header
    println("="^72)
    println("  HPC Worker — Single ls_direct Case")
    println("  T          = $T")
    println("  rec_id     = $rec_id")
    println("  N          = $N  (shooting segments)")
    println("  T_guess    = $T_guess")
    println("  u_rec      = [$(round(u_rec[1], digits=6)), $(round(u_rec[2], digits=6)), $(round(u_rec[3], digits=6))]")
    println("  maxiter    = $maxiter  (Newton iterations)")
    println("  ls_maxiter = $ls_maxiter  (line-search iterations)")
    println("  output     = $data_dir")
    println("  started    = $(now())")
    println("="^72)
    println()

    # Run the case
    result = run_single_case(T, rec_id, N, u_rec, T_guess,
                             L_flow, data_dir, maxiter, ls_maxiter)

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
