# Lorenz System — L-BFGS Periodic Orbit Search (HPC Results)

## Project Overview
L-BFGS optimisation over shooting segments to find periodic orbits of the Lorenz system.
Parameter sweep is embarrassingly parallel — every `(T, rec_id, N, m)` combination is independent.

**Lorenz parameters:** σ=10, ρ=28, β=8/3, Δt=0.01, RK4 integrator

## Two HPC Runs

| Run | Folder | Strategy | Task count |
|-----|--------|----------|-------------|
| **old** | `lorenz-results-old/output/` | Fine-grained: 1 task per (T,rec_id,N,m), each with own 12h SLURM slot | 28,273 |
| **new** | `lorenz-results/output/` | Per-recurrence: 1 task per (T,rec_id) with 8 threads, runs all 49 (N,m) combos sequentially in 12h | 577 |

## Parameter Grid

- **T (target periods):** 5, 10, 20, 40, 80, 160 → labelled `T05`, `T10`, `T20`, `T40`, `T80`, `T160`
- **N (segments):** 5, 10, 20, 40, 80, 160, 320
- **m (L-BFGS memory):** 5, 10, 20, 40, 80, 160, 320
- **49 combos** per recurrence (7×7 grid)
- **MAXITER:** 1,000,000
- **Convergence threshold:** ‖F‖ ≤ 1e-8

Recurrence counts per T:
- T05: 84 recs → 4,116 combos
- T10: 99 recs → 4,851 combos
- T20: 99 recs → 4,851 combos
- T40: 99–100 recs → ~4,900 combos
- T80: 93–95 recs → ~4,600 combos
- T160: 100 recs → 4,900 combos

## Directory Structure

```
Lorenz-System-Iridis/
├── scripts/
│   ├── hpc_worker.jl              # Single-case worker (1 CPU per case)
│   ├── hpc_worker_per_rec.jl      # Multi-threaded per-recurrence worker
│   ├── find_recurrences.jl        # Finds near-recurrence candidates
│   ├── generate_data_from_recurrences.jl
│   ├── generate_task_list.jl      # Fine-grained task list generator
│   └── generate_rec_task_list.jl  # Per-recurrence task list generator
├── recurrences/                   # Input: near-recurrence candidates
│   └── T{XX}/recurrences.csv      # Columns: rec_id, u1, u2, u3, T_guess, shift, distance
├── lorenz-results/output/         # NEW run output
│   └── T{XX}/data/rec{XXX}/N{XX}_m{XX}.csv
├── lorenz-results-old/output/     # OLD run output
│   └── T{XX}/data/rec{XXX}/N{XX}_m{XX}.csv
├── plots/
├── analyze_hpc_results.py         # Classification & status matrix script
├── status_matrix_new.csv          # T × (rec_id,N,m) matrix for new run
├── status_matrix_old.csv          # T × (rec_id,N,m) matrix for old run
├── status_summary.csv             # Counts per T, folder, status
└── cases_to_rerun.csv             # All non-converged cases (flat list)
```

## CSV Output Format (per (T,rec_id,N,m) case)

Columns: `iter, e_norm, grad_norm, lambda`

- **Normal completion:** Last row has `grad_norm=NaN, lambda=NaN` (final residual appended after `_search!` returns). Second-to-last row is the last optimizer callback.
- **Crash/error:** Last row has valid `grad_norm` and `lambda` (no final NaN row appended).
- **Diverged:** `e_norm` becomes `NaN` (usually at iter=1).

## Classification Logic (in `analyze_hpc_results.py`)

| Status | Criterion |
|--------|-----------|
| `converged` | Last row has `grad_norm=NaN` AND `e_norm ≤ 1e-8` |
| `hit_maxiter` | Last row has `grad_norm=NaN` AND `e_norm > 1e-8` AND `iter ≥ 1,000,000` |
| `diverged` | Any row in first 5 data lines has `e_norm=NaN` |
| `incomplete` | Everything else (time-limit kill, crash, stopped early) |
| `no_data` | CSV file does not exist |

## Status Matrices

Saved as MultiIndex CSV — load with:
```python
import pandas as pd
new = pd.read_csv("status_matrix_new.csv", header=[0,1,2], index_col=0)
# Rows: T05..T160 (6 rows)
# Columns: (rec_id, N, m) — 4900 columns
# Values: categorical string (converged/no_data/incomplete/hit_maxiter/diverged)
```

Or use the flat `cases_to_rerun.csv`:
```python
rerun = pd.read_csv("cases_to_rerun.csv")
# Columns: folder, T, rec_id, N, m, status
```

## Key Results Summary

| T | New converged | Old converged | Verdict |
|---|---------------|---------------|---------|
| T05 | 4,109 / 4,900 (83.9%) | 3,455 / 4,900 (70.5%) | New better ✅ |
| T10 | 4,839 / 4,900 (98.8%) | 3,914 / 4,900 (79.9%) | New much better ✅ |
| T20 | 4,785 / 4,900 (97.7%) | 3,792 / 4,900 (77.4%) | New much better ✅ |
| T40 | 2,887 / 4,900 (58.9%) | 2,805 / 4,900 (57.2%) | Roughly equal ≈ |
| T80 | 996 / 4,900 (20.3%) | 1,952 / 4,900 (39.8%) | New WORSE ❌ (time limit) |
| T160 | 36 / 4,900 (0.7%) | 331 / 4,900 (6.8%) | New WORSE ❌ (time limit) |

The per-recurrence strategy (new) works great for short periods but runs out of time (12h SLURM limit) for T80/T160 where iterations per combo are much larger. Many T80/T160 cases in the new run are `no_data` — the worker didn't reach those combos before the time limit killed the job.

## New vs Old — Detailed Comparison

| Change type | Count | Meaning |
|-------------|-------|---------|
| Unchanged | 20,535 | Same status in both runs |
| Improved | 3,632 | Old not converged → New converged |
| Regressed | 2,229 | Old converged → New not converged |
| Other changed | 3,004 | Different non-converged statuses |

T05/T10/T20: 655–993 MORE cases converged in new (rescued from incomplete/diverged/no_data).
T80/T160: 1,314 cases at T80 and 322 at T160 regressed from converged → no_data (time limit).
