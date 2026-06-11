# ============================================================================ #
# L-BFGS Periodic Orbit Search — Lorenz System
# RECURRENCE FINDER
#
# Searches for DISTINCT near-recurrences for each target period
# T ∈ {5, 10, 20, 40, 80} without running the optimization.
#
# Deduplication strategy (greedy):
#   1. Collect all recurrences within ±20 % of T_target.
#   2. Sort by distance ‖u_i − u_j‖ (closest first — these are the best
#      candidates for optimization).
#   3. Walk the list and keep a recurrence only if its state is at least
#      `min_state_dist` away from every previously-kept recurrence.
#
# Output (per T, saved to  lorenz/output/TXX/recurrences.csv):
#   rec_id, u1, u2, u3, T_guess, shift, distance
#
# Columns:
#   rec_id   – 1-based identifier (1 = closest recurrence for this T)
#   u1,2,3   – components of the state vector  u_rec ∈ ℝ³
#   T_guess  – approximate period  (= shift × Δt)
#   shift    – discrete-time shift (number of Δt steps)
#   distance – ‖u_i − u_j‖ at the recurrence point
#
# Usage:  julia find_recurrences.jl
# ============================================================================ #

using Printf,
      LinearAlgebra,
      DelimitedFiles

using Flows,
      StreamingRecurrenceAnalysis,
      ToySystems

# ---------------------------------------------------------------------------- #
# 0.  Parameters
# ---------------------------------------------------------------------------- #
const T_targets      = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]   # target orbit periods
const Δt             = 0.01                               # RK4 time step
const search_window  = 200                                # ± steps around T/Δt
const period_tol     = 0.20                               # ±20 % of T_target
const dist_thresh    = 0.5                                # max ‖u_i − u_j‖ for raw hits
const min_state_dist = 3.0                                # min ‖u_a − u_b‖ to be distinct
const n_views        = 900_000                            # #views in streamdistmat

const base_dir = joinpath(@__DIR__, "..", "recurrences")

# ---------------------------------------------------------------------------- #
# 1.  Lorenz system setup (same as generate_data.jl)
# ---------------------------------------------------------------------------- #
const LorenzEq = ToySystems.LorenzEq
const F_rhs   = LorenzEq.Lorenz(28.0)
const MM      = RK4
const u0      = Float64[1, 1, 1]

const ϕ = flow(F_rhs, MM(u0), TimeStepConstant(Δt))

"Advance state u by M time steps in place."
function g(u, M=1)
    ϕ(u, (0, M * Δt))
    return u
end

# ---------------------------------------------------------------------------- #
# 2.  Find all raw near-recurrences for a target T
# ---------------------------------------------------------------------------- #
"""
    find_all_recurrences(T_target::Real)

Returns a vector of named tuples `(u_rec, T_guess, dist, idx_shift)` for every
near-recurrence in the search window whose distance < `dist_thresh`.
"""
function find_all_recurrences(T_target::Real)
    idx_centre = round(Int, T_target / Δt)
    idx_range  = max(1, idx_centre - search_window) : (idx_centre + search_window)

    println("  Searching index range $idx_range")
    println("    (T ∈ [$(round(first(idx_range)*Δt, digits=2)), ",
            "$(round(last(idx_range)*Δt, digits=2))])")

    recs_iter = recurrences(
        streamdistmat(g, copy(u0), (u, v) -> norm(u - v), idx_range, n_views),
        d -> d < dist_thresh)

    results = NTuple{4, Any}[]
    for rec in recs_iter
        push!(results, rec)
    end

    println("  Raw hits: $(length(results))")
    return results
end

# ---------------------------------------------------------------------------- #
# 3.  Filter + deduplicate
# ---------------------------------------------------------------------------- #
"""
    deduplicate_recurrences(raw_recs, T_target; period_tol, min_state_dist)

1. Keep only recurrences whose T_guess is within `period_tol` of T_target.
2. Sort by distance (ascending — closest first).
3. Greedy deduplicate: keep a recurrence only if its state is at least
   `min_state_dist` away from every already-kept recurrence.

Returns a vector of the same named-tuple format.
"""
function deduplicate_recurrences(raw_recs, T_target::Real;
                                  period_tol::Real=0.20,
                                  min_state_dist::Real=3.0)
    # --- 3a.  Filter by period ------------------------------------------------
    filtered = NTuple{4, Any}[]
    for rec in raw_recs
        T_guess = rec[3] * Δt
        if abs(T_guess - T_target) / T_target <= period_tol
            push!(filtered, rec)
        end
    end
    println("  After period filter (±$(round(period_tol*100))%): $(length(filtered))")

    if isempty(filtered)
        return NTuple{4, Any}[]
    end

    # --- 3b.  Sort by distance (closest first) --------------------------------
    sort!(filtered, by=rec -> rec[4])   # rec[4] = distance

    # --- 3c.  Greedy deduplication -------------------------------------------
    kept = NTuple{4, Any}[]
    for rec in filtered
        u_candidate = rec[1]   # state vector
        is_distinct = true
        for k in kept
            if norm(u_candidate - k[1]) < min_state_dist
                is_distinct = false
                break
            end
        end
        if is_distinct
            push!(kept, rec)
        end
    end

    println("  After deduplication (min dist = $min_state_dist): $(length(kept))")
    return kept
end

# ---------------------------------------------------------------------------- #
# 4.  Save to CSV
# ---------------------------------------------------------------------------- #
"""
    save_recurrences_csv(kept_recs, csv_path::String)

Writes a CSV file with header:
    rec_id,u1,u2,u3,T_guess,shift,distance
"""
function save_recurrences_csv(kept_recs, csv_path::String)
    open(csv_path, "w") do io
        println(io, "rec_id,u1,u2,u3,T_guess,shift,distance")
        for (i, rec) in enumerate(kept_recs)
            u_rec     = rec[1]
            idx_shift = rec[3]
            T_guess   = idx_shift * Δt
            dist      = rec[4]
            @printf(io, "%d,%.12f,%.12f,%.12f,%.12f,%d,%.12f\n",
                    i, u_rec[1], u_rec[2], u_rec[3], T_guess, idx_shift, dist)
        end
    end
end

# ---------------------------------------------------------------------------- #
# 5.  Main
# ---------------------------------------------------------------------------- #
function main()
    println("="^60)
    println("  Recurrence Finder — Lorenz System")
    println("  Target periods  T ∈ $T_targets")
    println("  Period tolerance ±$(round(period_tol*100))%")
    println("  Min distinct distance = $min_state_dist")
    println("="^60)
    println()

    for T_target in T_targets
        T_label  = @sprintf("T%02d", round(Int, T_target))
        T_outdir = joinpath(base_dir, T_label)
        mkpath(T_outdir)
        csv_path = joinpath(T_outdir, "recurrences.csv")

        println("─"^50)
        println("T ≈ $T_target")
        println("─"^50)

        # --- 5a.  Find all raw recurrences -----------------------------------
        raw_recs = find_all_recurrences(T_target)

        # --- 5b.  Filter by period & deduplicate -----------------------------
        kept_recs = deduplicate_recurrences(raw_recs, T_target;
                                            period_tol=period_tol,
                                            min_state_dist=min_state_dist)

        # --- 5c.  Save to CSV -------------------------------------------------
        if isempty(kept_recs)
            println("  WARNING: No distinct recurrences found for T ≈ $T_target")
            continue
        end
        save_recurrences_csv(kept_recs, csv_path)
        println("  Saved $(length(kept_recs)) distinct recurrences to  $csv_path")

        # --- 5d.  Quick summary -----------------------------------------------
        println()
        println("  Summary (top 5):")
        for i in 1:min(5, length(kept_recs))
            rec = kept_recs[i]
            u   = rec[1]
            T   = rec[3] * Δt
            d   = rec[4]
            @printf("    #%d  T=%.4f  d=%.6f  u=[%.4f, %.4f, %.4f]\n",
                    i, T, d, u[1], u[2], u[3])
        end
        println()
    end

    println("Done.  All recurrence files saved to  $base_dir")
end

main()
