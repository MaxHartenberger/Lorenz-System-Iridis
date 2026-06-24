# ============================================================================ #
# L-BFGS Periodic Orbit Search — Lorenz System
# Compares L-BFGS memory (m) and shooting segments (N) for T ≈ 20 orbits.
# ============================================================================ #
using Pkg
#Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "jupyter_notebooks"))
#Pkg.develop(path=joinpath(@__DIR__, "..", "..", "..", "StreamingRecurrenceAnalysis.jl"))

using Flows,
      StreamingRecurrenceAnalysis,
      ToySystems,
      NKSearch,
      LinearAlgebra,
      Printf,
      Dates,
      Serialization

using PyPlot

# ---------------------------------------------------------------------------- #
# 1.  Lorenz system setup
# ---------------------------------------------------------------------------- #
const LorenzEq = ToySystems.LorenzEq
const F_rhs   = LorenzEq.Lorenz(28.0)          # nonlinear RHS
const Δt      = 0.01
const MM      = RK4
const u0      = Float64[1, 1, 1]

# Nonlinear flow (NormalMode, TimeStepConstant)
const ϕ = flow(F_rhs, MM(u0), TimeStepConstant(Δt))

function main()

# --- Open log file ---
outdir = joinpath(@__DIR__, "..", "output")
mkpath(outdir)
logfile = open(joinpath(outdir, "run_log_T20.txt"), "w")
original_stdout = stdout
redirect_stdout(logfile)

try

# ---------------------------------------------------------------------------- #
# 2.  Find near-recurrence with T ≈ 20
# ---------------------------------------------------------------------------- #
println("=== Searching for near-recurrence with T ≈ 20 ===")

# Helper: advance state by M timesteps
g(u, M=1) = (ϕ(u, (0, M * Δt)); return u)

# Search for recurrences in index range 1800:2200 → T ∈ [18, 22]
# Recurrences returns (state, Δi, Δj, distance) for each near-recurrence.
# Δj is the discrete time shift → T = Δj * Δt.
# N_views = 300_000 gives a good trade-off between speed and finding recurrences.
recs_iter = recurrences(
    streamdistmat(g, copy(u0), (u, v) -> norm(u - v), 1800:2200, 300_000),
    d -> d < 0.5)

best_rec = nothing
best_diff = Inf
n_found = 0
for rec in recs_iter
    n_found += 1
    diff = abs(rec[3] * Δt - 20.0)    # rec[3] = Δj = discrete shift
    if diff < best_diff
        best_diff = diff
        best_rec = rec
    end
end

if best_rec === nothing
    error("No near-recurrence found in index range 1800:2200. Try a wider range.")
end

println("  Found $n_found near-recurrence(s)")

u_rec    = best_rec[1]          # state vector
T_guess  = best_rec[3] * Δt     # Δj * Δt = approximate period
dist_rec = best_rec[4]          # distance value
idx_rec  = best_rec[3]          # discrete time shift

println("  Chosen recurrence:  d_min = $(round(dist_rec, digits=6))")
println("  T_guess = $(round(T_guess, digits=6))   (shift = $idx_rec steps)")
println("  u_rec   = $(round.(u_rec, digits=6))")

# ---------------------------------------------------------------------------- #
# 3.  Build linearised & adjoint flow operators
# ---------------------------------------------------------------------------- #
# Forward linearised flow — DiscreteMode{false}, TimeStepFromCache
# (needed internally by IterSolCache, not used by L-BFGS mat-vecs themselves)
D_lin = LorenzEq.LorenzLin(false, 28.0)
L_flow = flow(D_lin,
              RK4(zeros(3), Flows.DiscreteMode(false)),
              TimeStepFromCache())

# Adjoint flow — DiscreteMode{true}, TimeStepFromCache
# (used for gradient ∇φ = J^T·F via the adjoint cache)
D_adj_raw = LorenzEq.LorenzLin(true, 28.0)
L_adj_flow = flow(D_adj_raw,
                  RK4(zeros(3), Flows.DiscreteMode(true)),
                  TimeStepFromCache())

# Phase-locking derivative operator:  D[1](out, x) = f(x)
phase_lock = (dxdt, x) -> F_rhs(0, x, dxdt)

# ---------------------------------------------------------------------------- #
# 4.  Parameter sweeps
# ---------------------------------------------------------------------------- #
Ms = [1, 2, 5, 10, 20]
Ns = [1, 2, 5, 10, 20]
maxiter  = 5000

# Storage:  results[(N, m)] = (hist, converged, final_T, final_normF)
results = Dict{Tuple{Int,Int}, NamedTuple}()

println("\n=== Running L-BFGS sweeps (maxiter = $maxiter) ===")

for N in Ns
    # Distribute initial points evenly along the approximate orbit
    z0_seeds = [copy(u_rec) for _ in 1:N]
    for i in 2:N
        ϕ(z0_seeds[i], (0, (i - 1) * T_guess / N))
    end

    for m in Ms
        label = "N=$N, m=$m"
        print("  $label ... ")
        flush(stdout)

        # Build fresh z0
        z0 = MVector((copy.(z0_seeds)...,), T_guess)

        # Build per-segment tuples
        Gs     = ntuple(_ -> deepcopy(ϕ), N)
        Ls     = ntuple(_ -> deepcopy(L_flow), N)
        Ls_adj = ntuple(_ -> deepcopy(L_adj_flow), N)

        # History collector
        hist_normF = Float64[]

        cb = (iter, z, Fz, f_norm, ∇ϕ_norm, λ) -> begin
            push!(hist_normF, f_norm)   # ‖F(z)‖ (not squared)
            return false
        end

        opts = Options(
            method          = :lbfgs_opt,
            maxiter         = maxiter,
            dz_norm_tol     = 0.0,    # disabled: dz_norm ≥ 0 always, so this never triggers.
            e_norm_tol      = 1e-12,  # only ‖F‖ < 1e-16 or maxiter stops the optimizer
            verbose         = false,
            ls_maxiter      = 100,
            lbfgs_memory    = m,
            lbfgs_adj_system = Ls_adj,
            callback        = cb,
        )

        t_start = time()

        try
            # Use the low-level _search! so we control Gs/Ls/Ls_adj directly
            NKSearch._search!(Gs, Ls, nothing, (phase_lock,), z0, opts)
        catch err
            println("FAILED: $err")
            results[(N, m)] = (
                hist      = hist_normF,
                converged = false,
                final_T   = NaN,
                final_normF = length(hist_normF) > 0 ? hist_normF[end] : NaN,
                elapsed   = time() - t_start,
                error_msg = sprint(showerror, err),
            )
            continue
        end

        elapsed = time() - t_start

        # Final residual
        fwd_tmp = NKSearch.IterSolCache(Gs, Ls, nothing, nothing, (phase_lock,), z0)
        b_tmp   = similar(z0)
        NKSearch.update!(fwd_tmp, b_tmp, z0)
        final_normF = norm(b_tmp)
        push!(hist_normF, final_normF)

        converged = final_normF < 1e-12

        results[(N, m)] = (
            hist        = hist_normF,
            converged   = converged,
            final_T     = z0.d[1],
            final_normF = final_normF,
            elapsed     = elapsed,
            error_msg   = "",
        )

        status = converged ? "CONV" : "DID NOT CONVERGE"
        println("$(status)  ‖F‖=$(round(final_normF, sigdigits=3))  T=$(round(z0.d[1], digits=6))  $(round(elapsed, digits=1))s")
    end
end

# ---------------------------------------------------------------------------- #
# 5.  Save results to file
# ---------------------------------------------------------------------------- #
outdir = joinpath(@__DIR__, "..", "output")
mkpath(outdir)
timestamp = replace(string(now()), " " => "_", ":" => "-")
serialize(joinpath(outdir, "results_$(timestamp).jls"), results)
serialize(joinpath(outdir, "results_latest.jls"), results)

println("\n=== Results saved to $(outdir) ===")

# ---------------------------------------------------------------------------- #
# 6.  Plotting
# ---------------------------------------------------------------------------- #
println("\n=== Generating plots ===")

plotdir = joinpath(@__DIR__, "..", "plots")
mkpath(plotdir)

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 12
rcParams["axes.titlesize"] = 14
rcParams["axes.labelsize"] = 12
rcParams["legend.fontsize"] = 9

# --- Plot 1: Per-N subplots, each with all m curves ---
fig, axes = plt.subplots(length(Ns), 1, figsize=(12, 4 * length(Ns)), sharex=true)
fig.suptitle("L-BFGS Convergence — Lorenz System (T ≈ 20)", fontsize=16, fontweight="bold")

for (idx, N) in enumerate(Ns)
    ax = axes[idx]
    for m in Ms
        hist = results[(N, m)].hist
        niter = length(hist) - 1             # last entry is final residual
        conv = results[(N, m)].converged
        lbl  = "m = $m  [$niter it]"
        if !conv
            lbl *= "  ✗"
        else
            lbl *= "  ✓"
        end
        ax.semilogy(0:length(hist)-1, max.(hist, 1e-30), "-", label=lbl, linewidth=1.5)
    end
    ax.set_ylabel("||F(z)||")
    ax.set_title("N = $N segments")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(true, alpha=0.3)
end
axes[end].set_xlabel("Iteration")
fig.tight_layout()
fig.savefig(joinpath(plotdir, "convergence_per_N_T20.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(joinpath(plotdir, "convergence_per_N_T20.png"), dpi=150, bbox_inches="tight")
println("  Saved: convergence_per_N_T20.pdf / .png")

# --- Plot 2: Per-m subplots, each with all N curves ---
fig2, axes2 = plt.subplots(length(Ms), 1, figsize=(12, 4 * length(Ms)), sharex=true)
fig2.suptitle("L-BFGS Convergence — Lorenz System (T ≈ 20)", fontsize=16, fontweight="bold")

for (idx, m) in enumerate(Ms)
    ax = axes2[idx]
    for N in Ns
        hist = results[(N, m)].hist
        niter = length(hist) - 1             # last entry is final residual
        conv = results[(N, m)].converged
        lbl  = "N = $N  [$niter it]"
        if !conv
            lbl *= "  ✗"
        else
            lbl *= "  ✓"
        end
        ax.semilogy(0:length(hist)-1, max.(hist, 1e-30), "-", label=lbl, linewidth=1.5)
    end
    ax.set_ylabel("||F(z)||")
    ax.set_title("m = $m memory")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(true, alpha=0.3)
end
axes2[end].set_xlabel("Iteration")
fig2.tight_layout()
fig2.savefig(joinpath(plotdir, "convergence_per_m_T20.pdf"), dpi=150, bbox_inches="tight")
fig2.savefig(joinpath(plotdir, "convergence_per_m_T20.png"), dpi=150, bbox_inches="tight")
println("  Saved: convergence_per_m_T20.pdf / .png")

# --- Plot 3: Grid of all (N, m) combinations ---
fig3, axes3 = plt.subplots(length(Ns), length(Ms),
                            figsize=(3 * length(Ms), 2.5 * length(Ns)),
                            sharex=true, sharey=true)
fig3.suptitle("L-BFGS Convergence Grid — Lorenz System (T ≈ 20)",
              fontsize=16, fontweight="bold")

for (i, N) in enumerate(Ns)
    for (j, m) in enumerate(Ms)
        ax = axes3[i, j]
        hist = results[(N, m)].hist
        conv = results[(N, m)].converged
        ax.semilogy(0:length(hist)-1, max.(hist, 1e-30), "-", color="C0", linewidth=1.0)
        status_str = conv ? "CONV" : "✗"
        ax.set_title("N=$N, m=$m  [$status_str]", fontsize=9)
        ax.grid(true, alpha=0.3)
        if i == length(Ns)
            ax.set_xlabel("Iter")
        end
        if j == 1
            ax.set_ylabel("||F(z)||")
        end
    end
end
fig3.tight_layout()
fig3.savefig(joinpath(plotdir, "convergence_grid_T20.pdf"), dpi=150, bbox_inches="tight")
fig3.savefig(joinpath(plotdir, "convergence_grid_T20.png"), dpi=150, bbox_inches="tight")
println("  Saved: convergence_grid_T20.pdf / .png")

# --- Plot 4: Summary — final ‖F‖ vs m, coloured by N ---
fig4, ax4 = plt.subplots(figsize=(10, 6))
markers = [:o, :s, :D, :^, :v]
for (idx, N) in enumerate(Ns)
    final_norms = [results[(N, m)].final_normF for m in Ms]
    converged   = [results[(N, m)].converged for m in Ms]
    ax4.semilogy(Ms, max.(final_norms, 1e-30), "-" * string(markers[idx]),
                 label="N = $N", markersize=8, linewidth=1.5)
end
ax4.set_xlabel("L-BFGS memory (m)")
ax4.set_ylabel("Final ||F(z)||")
ax4.set_title("Final Residual vs Memory — Lorenz System (T ≈ 20)")
ax4.legend()
ax4.grid(true, alpha=0.3)
ax4.set_xticks(Ms)
fig4.tight_layout()
fig4.savefig(joinpath(plotdir, "final_residual_vs_m_T20.pdf"), dpi=150, bbox_inches="tight")
fig4.savefig(joinpath(plotdir, "final_residual_vs_m_T20.png"), dpi=150, bbox_inches="tight")
println("  Saved: final_residual_vs_m_T20.pdf / .png")

# --- Plot 5: Final T vs (N, m)  (converged runs only) ---
fig5, ax5 = plt.subplots(figsize=(10, 6))
for (idx, N) in enumerate(Ns)
    final_Ts = [results[(N, m)].converged ? results[(N, m)].final_T : NaN for m in Ms]
    ax5.plot(Ms, final_Ts, "-" * string(markers[idx]), label="N = $N",
             markersize=8, linewidth=1.5)
end
ax5.axhline(T_guess, color="gray", linestyle="--", alpha=0.5, label="T_guess = $(round(T_guess, digits=4))")
ax5.set_xlabel("L-BFGS memory (m)")
ax5.set_ylabel("Final period T")
ax5.set_title("Converged Period vs Memory — Lorenz System (T ≈ 20, converged only)")
ax5.legend()
ax5.grid(true, alpha=0.3)
ax5.set_xticks(Ms)
fig5.tight_layout()
fig5.savefig(joinpath(plotdir, "final_period_vs_m_T20.pdf"), dpi=150, bbox_inches="tight")
fig5.savefig(joinpath(plotdir, "final_period_vs_m_T20.png"), dpi=150, bbox_inches="tight")
println("  Saved: final_period_vs_m_T20.pdf / .png")

println("\n=== All plots saved to $(plotdir) ===")

# ---------------------------------------------------------------------------- #
# 7.  Print summary table
# ---------------------------------------------------------------------------- #
println("\n=== Summary Table ===")
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

println("\nDone.")
finally
    redirect_stdout(original_stdout)
    close(logfile)
    println("Log saved to $(joinpath(outdir, "run_log_T20.txt"))")
end
end # main()

main()

