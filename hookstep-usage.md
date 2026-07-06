# How to Correctly Use the Hookstep (Trust-Region GMRES) Method in NKSearch.jl

## Quick Summary

The hookstep method solves $F(z) = 0$ directly via Newton–GMRES with a
**trust-region** globalization.  At each Newton iteration, the linear
system $J\,\delta z = -F(z)$ is solved matrix-free with GMRES, and the
step is constrained to lie within a trust region of radius $\Delta$.  If
GMRES converges inside the trust region, a full Newton step is taken;
otherwise the constrained (hookstep) solution is used.

It requires **two flows** (nonlinear `G`, forward linearised `L`).

**TL;DR call signature:**
```julia
search!(G, L, (dxdt, x) -> F(0, x, dxdt), z,
        Options(method=:tr_iterative, maxiter=25, gmres_maxiter=5))
```

---

## 1. Two Required Flows

| Flow | Mode | Purpose | Called when |
|---|---|---|---|
| `G` | `NormalMode` | Nonlinear propagate | `update!` — fills end-of-segment states, computes residual |
| `L` | `NormalMode` (standard coupled) | Forward tangent-linear | `mul!` — matrix-vector product $J\cdot v$ for GMRES |

**Both use `TimeStepConstant`.**  Unlike L-BFGS, the hookstep method
does **not** need stage caching (`TimeStepFromCache`) because GMRES
only requires mat-vec products, not adjoint backward integration.
There is **no adjoint flow** in the hookstep method.

---

## 2. Flow Construction Pattern

```julia
using NKSearch, Flows, LinearAlgebra

# --- Nonlinear flow ---
G = flow(F_sys,
         RK4(zeros(dim), Flows.NormalMode()),
         TimeStepConstant(dt))

# --- Forward linearised flow (standard Flows.couple convention) ---
L = flow(couple(F_sys, D_lin),
         RK4(couple(zeros(dim), zeros(dim)), Flows.NormalMode()),
         TimeStepConstant(dt))
```

Here `couple(F, D)` from `Flows.jl` pairs the base and perturbation RHS so
that `L(Flows.couple(x, v), span)` advances both `x` and `v` simultaneously.
See §3b for the `D_lin` signature.

---

## 3. The Two Operator Types You Must Define

### 3a. Nonlinear right-hand side $F(t, x, dxdt)$
Standard 3-argument in-place ODE RHS:
```julia
struct System
    μ::Float64
end
function (s::System)(t, u, dudt)
    # ... fill dudt ...
    return dudt
end
```

### 3b. Forward linearised RHS $D(t, x, dxdt, v, dvdt)$ → 5 arguments

`Flows.couple` passes the linearised RHS **5 arguments**: $(t, x, dxdt, v, dvdt)$.
It must fill `dvdt = J(x)\cdot v` while also filling `dxdt = F(x)`:
```julia
struct SystemLinear
    μ::Float64
    J::Matrix{Float64}
end
function (s::SystemLinear)(t, u, dudt, v, dvdt)
    # fill dudt = F(u)  AND  dvdt = J(u) * v
    s.J[1,1] = ...  # populate Jacobian
    # ...
    dudt[1] = ...   # fill nonlinear RHS
    # ...
    return mul!(dvdt, s.J, v)
end
```

`Flows.couple(F, D)` automatically applies `F` first, then `D`, so both
`dudt` and `dvdt` are updated.

**Note:** The `couple`-based linearised flow uses the standard
`Flows.NormalMode()`, **not** `Flows.DiscreteMode`.  This is a key
difference from the L-BFGS method, which uses `DiscreteMode` for its
stage-cache-based adjoint.

---

## 4. Phase-Locking Closure

The 4th argument to `search!` is the phase-locking condition.  Wrap your RHS:
```julia
phase_lock = (dxdt, x) -> F_sys(0, x, dxdt)
```
This is passed as the `D` operator in the internal dispatch.
- For ordinary periodic orbits (`NS == 1`): only `D[1]` (the RHS) is used.
- For relative periodic orbits (`NS == 2`): internally `D = (F, dS)` where `F`
  is the RHS and `dS` is the derivative of the spatial shift operator w.r.t.
  its parameter.  At the `search!` call site, `F` and `dS` are passed as
  separate arguments (see §5).

---

## 5. Calling `search!`

### Ordinary periodic orbit (NS == 1, no spatial shift)
```julia
search!(G, L, (dxdt, x) -> F_sys(0, x, dxdt), z,
        Options(method=:tr_iterative, ...))
```
4 arguments: `G, L, phase_lock, z`.

### Relative periodic orbit (NS == 2, with spatial shift)
```julia
search!(G, L, S, F, dS, z,
        Options(method=:tr_iterative, ...))
```
6 arguments: `G, L, S, F, dS, z`.  `F` and `dS` are passed separately
(internally combined as `(F, dS)`).  Requires a spatial shift operator
`S` (e.g. `S(x, s)` shifts state `x` by `s` in place) and its derivative
`dS` (`dS(out, x)` computes $\partial S/\partial s$ evaluated at `x`).

The spatial shift operators must be **named callable structs** for thread
safety (same pattern as described in `lbfgs-usage.md` §5).

---

## 6. Callback

A user-supplied callback can monitor progress and optionally halt the
search early.  Set it via `Options(callback = ...)`.  The callback is
called once per Newton iteration with a fixed 7-argument signature:

```julia
callback(iter, z, Fz, e_norm, ∇ϕ_norm, λ, T) -> Bool
```

| Arg | Type | Meaning in the hookstep |
|---|---|---|
| `iter` | `Int` | Newton iteration number (1-based) |
| `z` | `MVector` | Current orbit — seeds and scalar unknowns (period, shift). **Do not mutate.** |
| `Fz` | `Vector{Float64}` | Copy of the right-hand side $b = -F(z)$ (the flat residual). Useful for saving residuals. |
| `e_norm` | `Float64` | Residual norm $\|F(z)\|$ |
| `∇ϕ_norm` | `Float64` | Always `0.0` in the hookstep (placeholder shared with the L-BFGS callback signature). |
| `λ` | `Float64` | Always `1.0` in the hookstep (placeholder; in line-search methods this is the step length). |
| `T` | `Float64` | Period $T =$ `z.d[1]` |

Return `true` to terminate the search immediately.  The returned status
symbol will be `:callback_satisfied`.

### Example: saving the residual history

```julia
residuals = Float64[]
cb = (iter, z, Fz, e_norm, ∇ϕ_norm, λ, T) -> begin
    push!(residuals, e_norm)
    return false   # never stop early
end

search!(G, L, phase_lock, z,
        Options(method=:tr_iterative, callback=cb, ...))
```

---

## 7. Recommended Options

```julia
Options(
    method          = :tr_iterative,  # required — selects the hookstep method
    maxiter         = 25,             # Newton iterations (typically fewer than L-BFGS)
    e_norm_tol      = 1e-10,          # tolerance on ‖F(z)‖
    dz_norm_tol     = 1e-10,          # tolerance on step norm
    verbose         = true,
    skipiter        = 1,

    # --- GMRES options ---
    gmres_maxiter   = 10,             # max GMRES iterations per Newton step
    gmres_rtol      = 1e-3,           # GMRES relative tolerance (≤ 1)
    gmres_verbose   = false,          # print per-GMRES-iteration output
    gmres_start     = dz -> (dz .*= 0; dz),  # warm-start initialiser

    # --- Trust-region options ---
    tr_radius_init  = 1.0,            # initial trust-region radius Δ₀
    tr_radius_max   = 1e8,            # maximum trust-region radius
    min_step        = 1e-4,           # minimum step before stopping
    NR_lim          = 1e-8,           # residual below which full Newton step is taken
    eta             = 0.0,            # minimum reduction ratio for accepting step
    α               = 1.0,            # over-relaxation factor (applied in NR region)

    # --- Not used (ignored) ---
    # ls_maxiter, ls_rho   → ignored (no line search)
    # lbfgs_memory         → ignored (not L-BFGS)
)
```

**Key settings for the hookstep:**
- `gmres_rtol` controls the accuracy of each GMRES solve.  Lower values
  (e.g. `1e-4`) give more accurate Newton steps but cost more Krylov
  iterations.  A value of `1e-3` is a good default.
- `gmres_maxiter` caps the Krylov subspace dimension.  If GMRES hits this
  limit, the solution may be less accurate.
- `tr_radius_init` should be set to a value commensurate with the expected
  step size.  If the Newton step is very large at the start, a small
  initial radius can prevent divergence; if the guess is already good,
  a larger radius allows faster convergence.
- `gmres_start` provides a warm-start for GMRES.  The default
  (`dz -> (dz .*= 0; dz)`) zeroes the initial guess, which is safe.
  Passing a function that returns the previous Newton step can speed up
  convergence in some cases.

---

## 8. How the Hookstep Algorithm Works

The hookstep method is a **trust-region Newton–GMRES** solver.  Each
outer iteration performs:

1. **Update** (`update!`): propagate each seed to its segment endpoint
   with the nonlinear flow `G`, compute the time-derivative via finite
   differences, and assemble the right-hand side $b = -F(z)$.

2. **Solve trust-region subproblem** (`solve_tr_subproblem!`):
   - Run GMRES to solve $J \cdot \delta z = -F(z)$ approximately
     (matrix-free, using `mul!` products with the linearised flow `L`).
   - The GMRES variant used is the **trust-region constrained** version:
     if the GMRES solution at iteration $k$ exceeds the trust-region
     radius $\Delta$, the solution is reflected to lie exactly on the
     boundary $\|\delta z\| = \Delta$ (the "hookstep").
   - If the final GMRES solution lies **inside** the trust region
     and the residual falls below `gmres_rtol`, it is accepted as a
     **Newton step** (`which = :newton`).
   - Otherwise, the boundary-constrained solution is returned as a
     **hookstep** (`which = :hkstep`).

3. **Trust-region update**: compute the actual reduction
   $\|F(z)\|^2 - \|F(z + \delta z)\|^2$ and the predicted reduction
   $\|J\cdot\delta z\|^2$, then compute the ratio $\rho$.
   - If $\rho < \frac{1}{4}$: the model is poor — shrink $\Delta$ by $\frac{1}{4}$.
   - If $\rho > \frac{3}{4}$ *and* the step hit the boundary: the model
     is good — expand $\Delta$ by up to $2\times$ (capped at `tr_radius_max`).
   - If $\|F(z)\| \leq$ `NR_lim`: the residual is already small enough
     that a full Newton step (with over-relaxation $\alpha$) is taken
     unconditionally.

4. **Accept/reject**: the step is accepted if $\rho >$ `eta`.
   Otherwise the state is unchanged but the trust region is still
   shrunk.

The loop terminates when $\|F(z)\| <$ `e_norm_tol`, $\|\delta z\| <$
`dz_norm_tol`, the trust-region radius $\Delta <$ `min_step`, a
callback returns `true`, or `maxiter` is reached.

### Verbose output columns

```
iter  | which  | ||dz|| |  ||e||  |  rho  | tr_radius | GMRES res | GMRES it
------+--------+--------+---------+-------+-----------+-----------+----------
```

- **which**: `:newton` (unconstrained GMRES solution), `:hkstep` (hit the
  trust-region boundary).
- **||dz||**: norm of the accepted step $\|\delta z\|$.
- **||e||**: residual norm $\|F(z)\|$.
- **rho**: ratio of actual to predicted reduction.
- **tr_radius**: current trust-region radius $\Delta$.
- **GMRES res**: final GMRES relative residual $\|J\cdot\delta z + F(z)\| / \|F(z)\|$.
- **GMRES it**: number of GMRES iterations taken.

---

## 9. Threading

The hookstep method parallelises across shooting segments using `@spawn`
tasks.  **Run Julia with `N` threads** where `N` is the number of shooting
segments:

```bash
julia --project=. -t 4 script.jl
```

Each segment's `update!` (nonlinear propagate, finite-difference
time-derivative) and `mul!` (linearised mat-vec) are spawned as
independent tasks and synchronised with `@sync`.  Thread-safety relies
on each task operating on its own slice of the shared arrays.

---

## 10. Comparison with L-BFGS

| Property | Hookstep (`:tr_iterative`) | L-BFGS (`:lbfgs_opt`) |
|---|---|---|
| Solves | $F(z) = 0$ directly | $\min_z \frac{1}{2}\|F(z)\|^2$ |
| Globalization | Trust region | Backtracking line search |
| Linear solve | GMRES (Krylov) | None (quasi-Newton direction) |
| Flows needed | 2 (`G`, `L`) | 3 (`G`, `L`, `L_adj`) |
| Adjoint required? | No | Yes |
| Cost per iteration | Higher (GMRES inner loop) | Lower (two-loop recursion) |
| Iterations to converge | Fewer (Newton) | More (quasi-Newton) |
| Robustness far from solution | Good (trust region) | Moderate (line search) |
| Thread-safe? | Yes (`@spawn`/`@sync`) | Yes (`@spawn`/`@sync`) |

Rule of thumb: use the hookstep when you can afford a few expensive GMRES
solves per Newton iteration and want robust convergence; use L-BFGS when
each linearised flow evaluation is costly and you prefer many cheap
quasi-Newton iterations.

---

## 11. Files Involved

| File | Role |
|---|---|
| `src/newton.jl` | `search!` entry points, dispatch to `_search_hookstep!` |
| `src/search_hookstep.jl` | `_search_hookstep!`, `solve_tr_subproblem!`, `solve_hookstep_subproblem!`, `_solve_tr_boundary!` |
| `src/iter_sol_cache.jl` | `IterSolCache` — matrix-free Jacobian mat-vec `mul!`, `update!`, GMRES wrapper |
| `src/multivector.jl` | `MVector` type (seeds + scalar unknowns) |
| `src/options.jl` | `Options` struct (fields: `gmres_*`, `tr_radius_*`, `eta`, `NR_lim`, etc.) |
| `src/output.jl` | Hookstep verbose status table (`display_header_hks`, `display_status_hks`) |
| `test/runtests.jl` | System definitions (`System`, `SystemLinear`, `TangentSystem`, etc.) |
| `test/test_search.jl` | Hookstep convergence test (Hopf normal form) |

---

## 12. Complete Minimal Working Example (NS == 1)

```julia
using NKSearch, Flows, LinearAlgebra

# --- System definition ---
struct Hopf
    μ::Float64
end
(f::Hopf)(t, u, dudt) = begin
    x, y = u[1], u[2]
    r = sqrt(x^2 + y^2)
    dudt[1] = -y + f.μ*x*(1 - r)
    dudt[2] =  x + f.μ*y*(1 - r)
    return dudt
end

# --- Forward linearised RHS (5-argument form for Flows.couple) ---
struct HopfLin
    μ::Float64; J::Matrix{Float64}
    HopfLin(μ) = new(μ, zeros(2,2))
end
function (h::HopfLin)(t, u, dudt, v, dvdt)
    x, y = u[1], u[2]; r = sqrt(x^2 + y^2)
    # Fill dudt = F(u)  (nonlinear RHS — required by Flows.couple)
    dudt[1] = -y + h.μ*x*(1 - r)
    dudt[2] =  x + h.μ*y*(1 - r)
    # Fill Jacobian
    h.J[1,1] = h.μ*(1 - r - x^2/r); h.J[1,2] = -1 - h.μ*x*y/r
    h.J[2,1] =  1 - h.μ*x*y/r;      h.J[2,2] = h.μ*(1 - r - y^2/r)
    return mul!(dvdt, h.J, v)
end

# --- Build flows ---
μ = 1.0; dim = 2; dt = 1e-3
F_sys = Hopf(μ)

G = flow(F_sys,
         RK4(zeros(dim), Flows.NormalMode()),
         TimeStepConstant(dt))

L = flow(couple(F_sys, HopfLin(μ)),
         RK4(couple(zeros(dim), zeros(dim)), Flows.NormalMode()),
         TimeStepConstant(dt))

# --- Initial guess (2 segments, slightly perturbed unit circle) ---
z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π)

# --- Search ---
status = search!(G, L, (dxdt, x) -> F_sys(0, x, dxdt), z,
                 Options(method=:tr_iterative, maxiter=25,
                         e_norm_tol=1e-12, dz_norm_tol=1e-8,
                         gmres_maxiter=5, gmres_rtol=1e-3,
                         tr_radius_init=0.001, verbose=true))

# --- Verify ---
@assert maximum(map(el -> norm(el) - 1, z.x)) < 1e-9
@assert abs(z.d[1] - 2π) < 1e-9
@assert status == :converged
```

---

## 13. Complete Minimal Working Example (NS == 2, with spatial shift)

```julia
using NKSearch, Flows, LinearAlgebra

# --- System, linearised definitions same as NS == 1 ---
# (Hopf, HopfLin as above)

# --- Spatial shift: rotation by angle s (in place) ---
struct SpatialShift end
function (::SpatialShift)(x, s)
    c, sn = cos(s), sin(s)
    x1, x2 = x[1], x[2]
    x[1] = c*x1 - sn*x2
    x[2] = sn*x1 + c*x2
    return x
end

# --- Derivative of spatial shift: infinitesimal generator ---
struct SpatialShiftDerivative end
function (::SpatialShiftDerivative)(out, x)
    out[1] = -x[2]
    out[2] =  x[1]
    return out
end

# --- Build flows (same as NS == 1) ---
μ = 1.0; dim = 2; dt = 1e-3
F_sys = Hopf(μ)

G = flow(F_sys,
         RK4(zeros(dim), Flows.NormalMode()),
         TimeStepConstant(dt))

L = flow(couple(F_sys, HopfLin(μ)),
         RK4(couple(zeros(dim), zeros(dim)), Flows.NormalMode()),
         TimeStepConstant(dt))

S_op  = SpatialShift()
dS_op = SpatialShiftDerivative()

# --- Initial guess with zero initial shift ---
z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π, 0.0)

# --- Search (6-argument form for NS == 2) ---
status = search!(G, L, S_op, F_sys, dS_op, z,
                 Options(method=:tr_iterative, maxiter=25,
                         e_norm_tol=1e-12, dz_norm_tol=1e-8,
                         gmres_maxiter=5, gmres_rtol=1e-3,
                         tr_radius_init=0.001, verbose=true))

# --- Verify: orbit lies on unit circle, period is 2π, shift is zero ---
@assert maximum(map(el -> norm(el) - 1, z.x)) < 1e-9
@assert abs(z.d[1] - 2π) < 1e-9
@assert abs(z.d[2]) < 1e-9
@assert status == :converged
```
