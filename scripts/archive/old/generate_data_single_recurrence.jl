# ============================================================================ #
# L-BFGS Periodic Orbit Search — Lorenz System
# DATA GENERATION SCRIPT
#
# Generates per-iteration convergence data for a parameter grid:
#   T  (target orbit period)  ∈ {5, 10, 20, 40, 80}
#   m  (L-BFGS memory)        ∈ {1, 2, 5, 10, 20, 40}
#   N  (shooting segments)    ∈ {1, 2, 5, 10, 20}
#
# For every (T, m, N) combination this script saves a CSV file containing
#   iter,  e_norm (‖F(z)‖),  grad_norm (‖∇φ‖),  lambda (step length)
# at each iteration of the L-BFGS optimisation.
#
# Output directory layout:
#   lorenz/output/
#     T05/
#       run_log_T05.txt           # full console log for this period
#       data/
#         N01_m01.csv             # per-iteration data: N=1, m=1
#         N01_m02.csv             # N=1, m=2
#         ...
#         N20_m40.csv             # N=20, m=40
#     T10/
#       ...
#     T80/
#       ...
#
# The convergence tolerance ‖F(z)‖ < 1e-8 is used throughout.
# A plotting script (to be written separately) will consume these CSV files.
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
      Dates

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
# 3.  Find a near-recurrence close to a target period T_target
# ---------------------------------------------------------------------------- #
"""
    find_near_recurrence(T_target::Real; window::Int=200, dist_thresh::Real=0.5)

Search the Lorenz time series for a state whose recurrence time is close to
`T_target`.  Returns a named tuple `(u_rec, T_guess, dist, idx_shift)` where
- `u_rec`     : state vector at the recurrence point
- `T_guess`   : approximate period  (= idx_shift * Δt)
- `dist`      : Euclidean distance between the two states
- `idx_shift` : discrete-time shift (number of Δt steps)
"""
function find_near_recurrence(T_target::Real; window::Int=200, dist_thresh::Real=0.5)
    # Centre the search window around the expected discrete shift.
    idx_centre = round(Int, T_target / Δt)
    idx_range  = max(1, idx_centre - window) : (idx_centre + window)

    println("  Searching for near-recurrence in index range $idx_range")
    println("    (T ∈ [$(first(idx_range)*Δt), $(last(idx_range)*Δt)])")

    # streamdistmat produces a lazy stream of distance matrices;
    # recurrences filters those whose distance < dist_thresh.
    recs_iter = recurrences(
        streamdistmat(g, copy(u0), (u, v) -> norm(u - v), idx_range, 300_000),
        d -> d < dist_thresh)

    # Pick the recurrence whose period is closest to T_target.
    best_rec   = nothing
    best_diff  = Inf
    n_found    = 0
    for rec in recs_iter
        n_found += 1
        diff = abs(rec[3] * Δt - T_target)       # rec[3] = Δj = discrete shift
        if diff < best_diff
            best_diff = diff
            best_rec  = rec
        end
    end

    if best_rec === nothing
        error("No near-recurrence found in index range $idx_range. ",
              "Try a wider window or lower dist_thresh.")
    end

    u_rec     = best_rec[1]          # state vector at recurrence
    idx_shift = best_rec[3]          # discrete time shift Δj
    T_guess   = idx_shift * Δt       # approximate continuous period
    dist      = best_rec[4]          # ‖u_i - u_j‖

    println("  Found $n_found near-recurrence(s)")
    println("  Chosen:  d_min = $(round(dist, digits=6))")
    println("           T_guess = $(round(T_guess, digits=6))  (shift = $idx_shift steps)")
    println("           u_rec   = $(round.(u_rec, digits=6))")

    return (u_rec=u_rec, T_guess=T_guess, dist=dist, idx_shift=idx_shift)
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
    save_history_csv(outdir::String, N::Int, m::Int,
                     iter, e_norm, grad_norm, lambda)

Writes a CSV file `NXX_mXX.csv` in `outdir` with a header row
    iter,e_norm,grad_norm,lambda
followed by one data row per iteration.  Floating-point values use full
precision (%.16e) to avoid loss of information.
"""
function save_history_csv(outdir::String, N::Int, m::Int,
                          iter::Vector{Int},
                          e_norm::Vector{Float64},
                          grad_norm::Vector{Float64},
                          lambda::Vector{Float64})
    fname = @sprintf("N%02d_m%02d.csv", N, m)
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
# 6.  Run L-BFGS for one (N, m) combination and save per-iteration data
# ---------------------------------------------------------------------------- #
"""
    run_one_case(N, m, u_rec, T_guess, L_flow, L_adj_flow, phase_lock,
                 data_dir, maxiter)

Run L-BFGS with `N` shooting segments and memory `m`, starting from the
near-recurrence state `u_rec` with initial period guess `T_guess`.
Saves a CSV file `NXX_mXX.csv` in `data_dir`.

Returns a named tuple `(converged, final_T, final_normF, n_iter, elapsed, error_msg)`.
"""
function run_one_case(N::Int, m::Int,
                      u_rec, T_guess,
                      L_flow, L_adj_flow, phase_lock,
                      data_dir::String, maxiter::Int)
    label = "N=$N, m=$m"
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
    # We store data in vectors that are mutated inside the closure.
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
        save_history_csv(data_dir, N, m, history_iter, history_e_norm,
                         history_grad_norm, history_lambda)
        return (converged=false, final_T=NaN, final_normF=NaN,
                n_iter=length(history_iter), elapsed=elapsed,
                error_msg=sprint(showerror, err))
    end

    elapsed = time() - t_start

    # --- 6f.  Compute final residual with a fresh cache -----------------------
    # The L-BFGS internal cache may hold stale state; recompute F(z) cleanly
    # with a forward-only IterSolCache (no adjoint system needed).
    fwd_tmp = NKSearch.IterSolCache(Gs, Ls, nothing, nothing, (phase_lock,), z0)
    b_tmp   = similar(z0)
    NKSearch.update!(fwd_tmp, b_tmp, z0)
    final_normF = norm(b_tmp)

    # Append the *exact* final residual as an extra data point
    # (gradient and step length are not recomputed here, so set to NaN).
    push!(history_iter,      length(history_iter))
    push!(history_e_norm,    final_normF)
    push!(history_grad_norm, NaN)              # gradient not recomputed
    push!(history_lambda,    NaN)              # no step taken after final eval

    converged = final_normF < 1e-8
    n_iter    = length(history_iter) - 1       # last row is the final recomputation

    # --- 6g.  Save per-iteration CSV ------------------------------------------
    save_history_csv(data_dir, N, m, history_iter, history_e_norm,
                     history_grad_norm, history_lambda)

    # --- 6h.  Status line -----------------------------------------------------
    status = converged ? "CONV" : "DID NOT CONVERGE"
    @printf("%s  ‖F‖=%.3e  T=%.6f  %d it  %.1f s\n",
            status, final_normF, z0.d[1], n_iter, elapsed)

    return (converged=converged, final_T=z0.d[1], final_normF=final_normF,
            n_iter=n_iter, elapsed=elapsed, error_msg="")
end

# ---------------------------------------------------------------------------- #
# 7.  Main — loop over T, then over (N, m)
# ---------------------------------------------------------------------------- #
function main()
    # --- Parameters -----------------------------------------------------------
    T_targets = [80]#[5.0, 10.0, 20.0, 40.0, 80.0]   # target orbit periods
    Ms        = [20, 40, 80, 160, 320]#[1, 2, 5, 10, 20, 40, 80]            # L-BFGS memory sizes
    Ns        = [40, 80, 160, 320]#[1, 2, 5, 10, 20]                # number of shooting segments
    maxiter   = 10000                              # max L-BFGS iterations per case
    # NOTE: 150 combinations × up to ~300 s each ≈ 12 h worst-case.
    #       Reduce `maxiter` or `T_targets`/`Ms`/`Ns` for quicker runs.

    # --- Top-level output directory -------------------------------------------
    base_outdir = joinpath(@__DIR__, "..", "output")
    mkpath(base_outdir)

    # --- Build linearised flows once (they are T-independent) -----------------
    println("Building linearised & adjoint flow operators ...")
    L_flow, L_adj_flow, phase_lock = build_linearised_flows()
    println("Done.\n")

    # --- Loop over target periods T -------------------------------------------
    for T_target in T_targets
        # Each T gets its own sub-directory and log file.
        T_label   = @sprintf("T%02d", round(Int, T_target))
        T_outdir  = joinpath(base_outdir, T_label)
        data_dir  = joinpath(T_outdir, "data")      # CSV files go here
        mkpath(data_dir)

        log_path  = joinpath(T_outdir, "run_log_$(T_label).txt")
        logfile   = open(log_path, "w")
        original_stdout = stdout
        redirect_stdout(logfile)

        try
            println("="^72)
            println("  L-BFGS Periodic Orbit Search — Lorenz System")
            println("  Target period  T ≈ $T_target")
            println("  Memory values  m ∈ $Ms")
            println("  Segments       N ∈ $Ns")
            println("  Max iterations = $maxiter")
            println("  Convergence tol  ‖F‖ < 1e-8")
            println("  Date/time        $(now())")
            println("="^72)
            println()

            # --- 7a.  Find near-recurrence for this T -------------------------
            println("--- Step 1: Near-recurrence search ---")
            rec_info  = find_near_recurrence(T_target)
            u_rec     = rec_info.u_rec
            T_guess   = rec_info.T_guess
            println()

            # --- 7b.  Sweep over N and m --------------------------------------
            println("--- Step 2: L-BFGS sweeps ---")
            println()

            # Store summary for the per-T results table.
            results = Dict{Tuple{Int,Int}, NamedTuple}()

            for N in Ns
                for m in Ms
                    res = run_one_case(N, m, u_rec, T_guess,
                                       L_flow, L_adj_flow, phase_lock,
                                       data_dir, maxiter)
                    results[(N, m)] = res
                    # Force garbage collection after each case to prevent
                    # memory accumulation from deep-copied flow caches.
                    GC.gc()
                end
            end

            # --- 7c.  Print summary table for this T --------------------------
            println()
            println("--- Summary Table (T ≈ $T_target) ---")
            println(rpad("", 10), join([rpad("m=$m", 22) for m in Ms], ""))
            for N in Ns
                print(rpad("N=$N", 10))
                for m in Ms
                    r = results[(N, m)]
                    c = r.converged ? "✓" : "✗"
                    s = @sprintf "%s ‖F‖=%.2e T=%.4f" c r.final_normF r.final_T
                    print(rpad(s, 22))
                end
                println()
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
