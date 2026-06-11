# ============================================================================ #
# L-BFGS Periodic Orbit Search — Lorenz System
# DATA GENERATION SCRIPT  (multi-recurrence version)
#
# Loads DISTINCT near-recurrence candidates pre-computed by find_recurrences.jl
# and runs the same (N, m) parameter sweep as generate_data_single_recurrence.jl,
# but starting from EVERY recurrence instead of a single internally-chosen one.
#
# Parameter grid:
#   T  (target orbit period)  ∈ {5, 10, 20, 40, 80}
#   m  (L-BFGS memory)        ∈ {1, 2, 5, 10, 20, 40}
#   N  (shooting segments)    ∈ {1, 2, 5, 10, 20}
#
# For every (T, recurrence, N, m) combination this script saves a CSV file
# containing  iter, e_norm (‖F(z)‖), grad_norm (‖∇φ‖), lambda (step length)
# at each iteration of the L-BFGS optimisation.
#
# Input:
#   lorenz/recurrences/TXX/recurrences.csv     (from find_recurrences.jl)
#
# Output directory layout:
#   lorenz/output/
#     T05/
#       run_log_T05.txt                        # full console log for this period
#       data/
#         rec001_N01_m01.csv                    # rec=1, N=1, m=1
#         rec001_N01_m02.csv                    # rec=1, N=1, m=2
#         ...
#         rec002_N01_m01.csv                    # rec=2, N=1, m=1
#         ...
#         rec010_N20_m40.csv                    # rec=10, N=20, m=40
#     T10/
#       ...
#     T80/
#       ...
#
# The convergence tolerance ‖F(z)‖ < 1e-8 is used throughout.
# ============================================================================ #

using Pkg

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
const LorenzEq = ToySystems.LorenzEq                 # module holding RHS & linearisation
const F_rhs   = LorenzEq.Lorenz(28.0)                # nonlinear vector field  f(x)
const Δt      = 0.01                                  # time step for RK4 integrator
const MM      = RK4                                   # integrator factory
const u0      = Float64[1, 1, 1]                      # initial condition for recurrence search

# Nonlinear flow  ϕ(u, (0, T))  — integrates u forward by time T *in place*.
const ϕ = flow(F_rhs, MM(u0), TimeStepConstant(Δt))

# ---------------------------------------------------------------------------- #
# 2.  Helper: advance state by M discrete steps
# ---------------------------------------------------------------------------- #
"""Advance `u` by `M` time steps of length Δt, in place.  Returns `u`."""
function g(u, M=1)
    ϕ(u, (0, M * Δt))
    return u
end

# ---------------------------------------------------------------------------- #
# 3.  Load recurrences from CSV
# ---------------------------------------------------------------------------- #
"""
    load_recurrences(csv_path::String; max_recs::Int=typemax(Int))

Reads a recurrences CSV file (output of find_recurrences.jl) and returns a
vector of named tuples  `(u_rec, T_guess, shift, distance, rec_id)`.

The CSV must have columns:  rec_id, u1, u2, u3, T_guess, shift, distance.
Only the first `max_recs` rows are returned (default: all rows).
"""
function load_recurrences(csv_path::String; max_recs::Int=typemax(Int))
    # readdlm with ',' separator; skip the header line.
    data = readdlm(csv_path, ',', skipstart=1)

    n_rows = min(size(data, 1), max_recs)
    recs = NTuple{5, Any}[]

    for row in 1:n_rows
        rec_id   = Int(data[row, 1])
        u_rec    = Float64[data[row, 2], data[row, 3], data[row, 4]]
        T_guess  = Float64(data[row, 5])
        shift    = Int(data[row, 6])
        distance = Float64(data[row, 7])
        push!(recs, (u_rec, T_guess, shift, distance, rec_id))
    end

    return recs
end

# ---------------------------------------------------------------------------- #
# 4.  Build linearised & adjoint flow operators  (shared across all N, m)
# ---------------------------------------------------------------------------- #
"""
    build_linearised_flows()

Returns `(L_flow, L_adj_flow, phase_lock)`:
- `L_flow`     : forward linearised flow  (DiscreteMode{false})
- `L_adj_flow` : adjoint flow             (DiscreteMode{true})
- `phase_lock` : phase-locking derivative  D[1](out, x) = f(x)

Both flows use `TimeStepFromCache()` so they reuse the time-step sequence
already stored by the nonlinear flow ϕ.  The adjoint flow is needed by the
L-BFGS gradient computation  ∇φ = Jᵀ·F(z).
"""
function build_linearised_flows()
    # Forward linearised dynamics:  du/dt = Df(x(t)) · u
    D_lin = LorenzEq.LorenzLin(false, 28.0)
    L_flow = flow(D_lin,
                  RK4(zeros(3), Flows.DiscreteMode(false)),
                  TimeStepFromCache())

    # Adjoint dynamics:  dv/dt = -Df(x(t))ᵀ · v   (integrated backwards)
    D_adj_raw = LorenzEq.LorenzLin(true, 28.0)
    L_adj_flow = flow(D_adj_raw,
                      RK4(zeros(3), Flows.DiscreteMode(true)),
                      TimeStepFromCache())

    # Phase-locking operator: derivative of the flow w.r.t. the period T.
    # For an autonomous system, the derivative of φ_T(x) w.r.t. T is f(φ_T(x)).
    phase_lock = (dxdt, x) -> F_rhs(0, x, dxdt)

    return L_flow, L_adj_flow, phase_lock
end

# ---------------------------------------------------------------------------- #
# 5.  Save per-iteration history as CSV
# ---------------------------------------------------------------------------- #
"""
    save_history_csv(outdir::String, rec_id::Int, N::Int, m::Int,
                     iter, e_norm, grad_norm, lambda)

Writes a CSV file `recXXX_NXX_mXX.csv` in `outdir` with a header row
    iter,e_norm,grad_norm,lambda
followed by one data row per iteration.  Floating-point values use full
precision (%.16e) to avoid loss of information.
"""
function save_history_csv(outdir::String, rec_id::Int, N::Int, m::Int,
                          iter::Vector{Int},
                          e_norm::Vector{Float64},
                          grad_norm::Vector{Float64},
                          lambda::Vector{Float64})
    fname = @sprintf("rec%03d_N%02d_m%02d.csv", rec_id, N, m)
    fpath = joinpath(outdir, fname)

    nrows = length(iter)
    open(fpath, "w") do io
        println(io, "iter,e_norm,grad_norm,lambda")
        for row in 1:nrows
            @printf(io, "%d,%.16e,%.16e,%.16e\n",
                    iter[row], e_norm[row], grad_norm[row], lambda[row])
        end
    end
end

# ---------------------------------------------------------------------------- #
# 6.  Run L-BFGS for one (rec_id, N, m) combination and save per-iteration data
# ---------------------------------------------------------------------------- #
"""
    run_one_case(rec_id, N, m, u_rec, T_guess, L_flow, L_adj_flow, phase_lock,
                 data_dir, maxiter)

Run L-BFGS with `N` shooting segments and memory `m`, starting from the
near-recurrence state `u_rec` with initial period guess `T_guess`.
Saves a CSV file `recXXX_NXX_mXX.csv` in `data_dir`.

Returns a named tuple `(converged, final_T, final_normF, n_iter, elapsed, error_msg)`.
"""
function run_one_case(rec_id::Int, N::Int, m::Int,
                      u_rec, T_guess,
                      L_flow, L_adj_flow, phase_lock,
                      data_dir::String, maxiter::Int)
    label = "rec=$(rec_id), N=$N, m=$m"
    print("  $label ... ")
    flush(stdout)

    # --- 6a.  Build initial guess z0 ------------------------------------------
    # Create N copies of the recurrence state, then advance each copy so that
    # the points are evenly distributed along the approximate orbit.
    z0_seeds = [copy(u_rec) for _ in 1:N]
    for i in 2:N
        ϕ(z0_seeds[i], (0, (i - 1) * T_guess / N))
    end
    z0 = MVector((copy.(z0_seeds)...,), T_guess)

    # --- 6b.  Build per-segment flow tuples -----------------------------------
    # Each segment gets its own deep copy so that internal caches are isolated.
    Gs     = ntuple(_ -> deepcopy(ϕ), N)          # nonlinear flows
    Ls     = ntuple(_ -> deepcopy(L_flow), N)     # forward linearised flows
    Ls_adj = ntuple(_ -> deepcopy(L_adj_flow), N) # adjoint flows

    # --- 6c.  Callback: record (iter, ‖F‖, ‖∇φ‖, λ) at every iteration -------
    history_iter      = Int[]
    history_e_norm    = Float64[]
    history_grad_norm = Float64[]
    history_lambda    = Float64[]

    cb = (iter, z, Fz, f_norm, ∇ϕ_norm, λ) -> begin
        push!(history_iter,      iter)
        push!(history_e_norm,    f_norm)       # ‖F(z)‖  (not squared)
        push!(history_grad_norm, ∇ϕ_norm)      # ‖Jᵀ·F(z)‖
        push!(history_lambda,    λ)            # step length used at this iteration
        return false                            # false → continue iterating
    end

    # --- 6d.  Options ---------------------------------------------------------
    opts = Options(
        method           = :lbfgs_opt,
        maxiter          = maxiter,
        dz_norm_tol      = 0.0,              # disabled — rely on e_norm_tol only
        e_norm_tol       = 1e-8,             # ‖F(z)‖ < 1e-8  ⇒ convergence
        verbose          = false,
        ls_maxiter       = 100,              # max backtracking steps per line search
        lbfgs_memory     = m,                # number of (s, y) pairs stored
        lbfgs_adj_system = Ls_adj,           # adjoint system for gradient computation
        callback         = cb,
    )

    # --- 6e.  Run the L-BFGS optimisation -------------------------------------
    t_start = time()

    try
        NKSearch._search!(Gs, Ls, nothing, (phase_lock,), z0, opts)
    catch err
        # If the search throws (e.g. integration failure), record what we have.
        println("FAILED: $err")
        elapsed = time() - t_start
        save_history_csv(data_dir, rec_id, N, m, history_iter, history_e_norm,
                         history_grad_norm, history_lambda)
        return (converged=false, final_T=NaN, final_normF=NaN,
                n_iter=length(history_iter), elapsed=elapsed,
                error_msg=sprint(showerror, err))
    end

    elapsed = time() - t_start

    # --- 6f.  Compute final residual with a fresh cache -----------------------
    fwd_tmp = NKSearch.IterSolCache(Gs, Ls, nothing, nothing, (phase_lock,), z0)
    b_tmp   = similar(z0)
    NKSearch.update!(fwd_tmp, b_tmp, z0)
    final_normF = norm(b_tmp)

    # Append the *exact* final residual as an extra data point
    push!(history_iter,      length(history_iter))
    push!(history_e_norm,    final_normF)
    push!(history_grad_norm, NaN)              # gradient not recomputed
    push!(history_lambda,    NaN)              # no step taken after final eval

    converged = final_normF < 1e-8
    n_iter    = length(history_iter) - 1       # last row is the final recomputation

    # --- 6g.  Save per-iteration CSV ------------------------------------------
    save_history_csv(data_dir, rec_id, N, m, history_iter, history_e_norm,
                     history_grad_norm, history_lambda)

    # --- 6h.  Status line -----------------------------------------------------
    status = converged ? "CONV" : "DID NOT CONVERGE"
    @printf("%s  ‖F‖=%.3e  T=%.6f  %d it  %.1f s\n",
            status, final_normF, z0.d[1], n_iter, elapsed)

    return (converged=converged, final_T=z0.d[1], final_normF=final_normF,
            n_iter=n_iter, elapsed=elapsed, error_msg="")
end

# ---------------------------------------------------------------------------- #
# 7.  Main — loop over T, then over recurrences, then over (N, m)
# ---------------------------------------------------------------------------- #
function main()
    # --- Parameters -----------------------------------------------------------
    T_targets      = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]   # target orbit periods
    Ms             = [5, 10, 20, 40, 80, 160, 320]            # L-BFGS memory sizes
    Ns             = [5, 10, 20, 40, 80, 160, 320]                # number of shooting segments
    maxiter        = 100000                              # max L-BFGS iterations per case
    max_recs       = 10                                 # max recurrences processed per T
                                                        # (set to typemax(Int) for all)

    # --- Paths ----------------------------------------------------------------
    recurrences_dir = joinpath(@__DIR__, "..", "recurrences")
    base_outdir     = joinpath(@__DIR__, "..", "output")
    mkpath(base_outdir)

    # --- Build linearised flows once (they are T-independent) -----------------
    println("Building linearised & adjoint flow operators ...")
    L_flow, L_adj_flow, phase_lock = build_linearised_flows()
    println("Done.\n")

    # --- Loop over target periods T -------------------------------------------
    for T_target in T_targets
        T_label   = @sprintf("T%02d", round(Int, T_target))
        T_outdir  = joinpath(base_outdir, T_label)
        data_dir  = joinpath(T_outdir, "data")
        mkpath(data_dir)

        log_path  = joinpath(T_outdir, "run_log_$(T_label).txt")
        logfile   = open(log_path, "w")
        original_stdout = stdout
        redirect_stdout(logfile)

        try
            println("="^72)
            println("  L-BFGS Periodic Orbit Search — Lorenz System")
            println("  (multi-recurrence version)")
            println("  Target period  T ≈ $T_target")
            println("  Memory values  m ∈ $Ms")
            println("  Segments       N ∈ $Ns")
            println("  Max iterations = $maxiter")
            println("  Max recurrences = $max_recs")
            println("  Convergence tol  ‖F‖ < 1e-8")
            println("  Date/time        $(now())")
            println("="^72)
            println()

            # --- 7a.  Load recurrences for this T -----------------------------
            csv_path = joinpath(recurrences_dir, T_label, "recurrences.csv")
            if !isfile(csv_path)
                println("WARNING: No recurrence file found at  $csv_path")
                println("Skipping T ≈ $T_target.")
                continue
            end

            println("--- Step 1: Load recurrences ---")
            recs = load_recurrences(csv_path; max_recs=max_recs)
            println("  Loaded $(length(recs)) recurrence(s) from  $csv_path")
            println()

            if isempty(recs)
                println("WARNING: No recurrences to process for T ≈ $T_target.")
                continue
            end

            # --- 7b.  Sweep over recurrences, N, and m ------------------------
            println("--- Step 2: L-BFGS sweeps ---")
            println()

            # Store results:  (rec_id, N, m) → result
            results = Dict{Tuple{Int,Int,Int}, NamedTuple}()

            for (ri, rec) in enumerate(recs)
                u_rec     = rec[1]
                T_guess   = rec[2]
                idx_shift = rec[3]
                distance  = rec[4]
                rec_id    = rec[5]

                println("─"^50)
                @printf("Recurrence #%d  (CSV rec_id=%d)\n", ri, rec_id)
                @printf("  T_guess = %.6f   shift = %d   distance = %.6f\n",
                        T_guess, idx_shift, distance)
                @printf("  u_rec   = [%.6f, %.6f, %.6f]\n",
                        u_rec[1], u_rec[2], u_rec[3])
                println()

                for N in Ns
                    for m in Ms
                        res = run_one_case(rec_id, N, m, u_rec, T_guess,
                                           L_flow, L_adj_flow, phase_lock,
                                           data_dir, maxiter)
                        results[(rec_id, N, m)] = res
                        GC.gc()
                    end
                end
            end

            # --- 7c.  Print summary tables (one per recurrence) ---------------
            println()
            println("="^72)
            println("--- Summary Tables (T ≈ $T_target) ---")
            for (ri, rec) in enumerate(recs)
                rec_id = rec[5]
                println()
                println("Recurrence #$ri  (rec_id=$rec_id)")
                println("  u_rec = [$(round(rec[1][1], digits=6)), ",
                        "$(round(rec[1][2], digits=6)), ",
                        "$(round(rec[1][3], digits=6))]")
                println("  T_guess = $(round(rec[2], digits=6))")
                println()
                println(rpad("", 10), join([rpad("m=$m", 22) for m in Ms], ""))
                for N in Ns
                    print(rpad("N=$N", 10))
                    for m in Ms
                        r = results[(rec_id, N, m)]
                        c = r.converged ? "✓" : "✗"
                        s = @sprintf "%s ‖F‖=%.2e T=%.4f" c r.final_normF r.final_T
                        print(rpad(s, 22))
                    end
                    println()
                end
            end
            println()
            println("CSV files saved to  $data_dir")
            println("Done with T ≈ $T_target.")
            println()

        finally
            redirect_stdout(original_stdout)
            close(logfile)
            println("Log saved to  $log_path")
            println("CSV files in  $data_dir")
            println()
        end
    end

    println("\nAll sweeps complete.")
    println("Output root:  $base_outdir")
end

# ---------------------------------------------------------------------------- #
# 8.  Entry point
# ---------------------------------------------------------------------------- #
main()
