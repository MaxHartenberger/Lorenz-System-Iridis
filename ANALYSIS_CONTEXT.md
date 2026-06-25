# Project Analysis Pipeline — Lorenz L-BFGS Results Analysis

## What This Analysis Does

After the HPC sweep completes, `analysis/analyze_hpc_results.py` reads every
per-iteration CSV produced by `hpc_worker.jl` and classifies each
`(T, rec_id, N, m)` case into one of five statuses.  It then produces summary
tables, per-recurrence overviews, and a flat re-run list for any non-converged
cases.  All analysis outputs are saved to the `analysis/` folder.

---

## Input: Output Directory Structure

The script reads from a single `outputs/` directory (at the repo root):

```
outputs/
├── logs/                          # SLURM .out / .err per array task
│   ├── sweep_<jobid>_<taskid>.out
│   └── sweep_<jobid>_<taskid>.err
│
├── T05/data/
│   ├── rec001/
│   │   ├── iteration/             # ← per-iteration L-BFGS history
│   │   │   ├── N05_m05.csv
│   │   │   ├── N05_m10.csv
│   │   │   └── ...  (49 combos: 7 N × 7 m)
│   │   └── trajectory/            # ← converged-orbit phase-space data
│   │       ├── N05_m05_trajectory.csv   (only if converged)
│   │       └── ...
│   ├── rec002/ ...
│   └── recXXX/ ...                (up to 100 recurrences per T)
│
├── T10/data/ ...                  # same structure for T=10
├── T20/data/ ...
├── T40/data/ ...
├── T80/data/ ...
└── T160/data/ ...
```

### CSV Formats

**Iteration CSV** (`iteration/NXX_mXX.csv`):
```
iter,e_norm,grad_norm,lambda,T_curr
0,3.7941945225620959e-03,1.9010980005389613e-02,0.0,5.37
...
758,9.8605768907545449e-09,2.7714927072072095e-08,1.0,5.369891
# converged
```

- 5 columns: `iter, e_norm, grad_norm, lambda, T_curr`
- Last line is a **comment marker**: `# converged`, `# did_not_converge`, or `# crashed`
- The `# crashed` marker is written in the `catch` block of `hpc_worker.jl` when an exception occurs

**Trajectory CSV** (`trajectory/NXX_mXX_trajectory.csv`) — only present for converged cases:
```
t,x,y,z,segment
0.0000000000e+00,4.7397090703043219e+00,1.1538803179091690e+00,2.7773011046743786e+01,1
1.0000000000e-02,4.3989270078144740e+00,1.1680179723008781e+00,2.7094377837750933e+01,1
...
```

- 5 columns: `t, x, y, z, segment`
- One row per Δt = 0.01 step along the converged periodic orbit
- `segment` ∈ {1, …, N} indicates which shooting segment the point belongs to
- The total number of rows ≈ N × (T/N) / Δt = T/Δt (e.g., T=5 → ~500 rows)

---

## Sweep Dimensions

| Dimension | Values | Count |
|-----------|--------|-------|
| **T** (target period) | 5, 10, 20, 40, 80, 160 | 6 |
| **rec_id** (recurrence candidate) | 1–100 per T | up to 100 |
| **N** (shooting segments) | 5, 10, 20, 40, 80, 160, 320 | 7 |
| **m** (L-BFGS memory) | 5, 10, 20, 40, 80, 160, 320 | 7 |

**Total combos per T**: 100 × 7 × 7 = 4,900
**Total combos across all T**: 6 × 4,900 = 29,400

---

## Classification Logic

For each iteration CSV, the script classifies the outcome by reading only the
**first ~5 data rows** (scanning for NaN) and the **last line** (final status):

| Status | Condition |
|--------|-----------|
| **converged** | `‖F‖ ≤ 1e-8` (i.e., `e_norm` on the final row ≤ 1e-8) |
| **no_data** | CSV file is missing entirely |
| **incomplete** | Optimisation stopped early (>1 iteration) without converging and without hitting maxiter (e.g., time-limit kill, crash before writing `# crashed`) |
| **hit_maxiter** | Reached 1,000,000 iterations without converging |
| **diverged** | `NaN` appeared in the `e_norm` column during the first few iterations (typically iteration 1) |

### Detailed Decision Tree

```
Read CSV
  ├─ File missing?              → no_data
  ├─ NaN in e_norm (first rows)? → diverged
  └─ Read last line
       ├─ No "# converged" / "# did_not_converge" / "# crashed" comment?
       │    └─ incomplete (killed mid-run)
       ├─ "# crashed"?
       │    └─ incomplete (exception during optimisation)
       ├─ "# converged"?
       │    ├─ e_norm ≤ 1e-8  → converged ✓
       │    └─ e_norm > 1e-8  → incomplete (shouldn't happen)
       └─ "# did_not_converge"?
            ├─ iteration ≥ 1,000,000 → hit_maxiter
            └─ iteration < 1,000,000 → incomplete (stopped early)
```

**Performance note**: The script only reads the first ~5 lines (for NaN scan) and
the last line (via seek-based tail read), so it is fast even with 20k+ iteration
CSV files.

---

## Output Files (produced by `analysis/analyze_hpc_results.py`)

All outputs are saved to the `analysis/` folder (or a custom `--out-dir`):

| File | Description |
|------|-------------|
| `status_matrix.csv` | Full T × (rec_id, N, m) MultiIndex CSV. Each cell is one of: `converged`, `no_data`, `incomplete`, `hit_maxiter`, `diverged` |
| `status_summary.csv` | Counts of each status per T value |
| `cases_to_rerun.csv` | Flat list of all non-converged cases (folder, T, rec_id, N, m, status) |
| `rec_overview.csv` | Per-recurrence overview: one row per (T, rec_id) with aggregated stats |

### `rec_overview.csv` Columns

| Column | Description |
|--------|-------------|
| `T` | Target period |
| `rec_id` | Recurrence candidate ID |
| `converged` | "yes" if ANY (N,m) combo converged, else "no" |
| `num_converged` | How many combos converged (out of 49) |
| `num_diverged` | How many combos diverged |
| `num_hit_maxiter` | How many combos hit maxiter |
| `num_incomplete` | How many combos stopped early |
| `num_no_data` | How many combos have no CSV |
| `best_N` / `best_m` | (N,m) of the converged combo with fewest iterations |
| `best_iter` | Iteration count of that fastest combo |
| `min_N_converged` | Smallest N that achieved convergence |
| `min_m_converged` | Smallest m that achieved convergence |
| `median_iter` | Median iterations among all converged combos for this recurrence |
| `converged_T` | Placeholder column (NaN); user fills this in later with a manual check of whether the recurrence itself is a true periodic orbit |

---

## How to Run the Analysis

```bash
# From the repo root, with default paths:
python analysis/analyze_hpc_results.py

# From the analysis folder:
cd analysis && python analyze_hpc_results.py

# With custom output directory:
python analysis/analyze_hpc_results.py --out-dir ./my-analysis

# Specifying a different results directory:
python analysis/analyze_hpc_results.py --results ./outputs
```

---

## Future: Trajectory Plotting (placeholder)

When trajectory CSVs are available for converged cases, the analysis can be
extended to plot the orbits.  See the `plot_trajectory()` function in
`analysis/analyze_hpc_results.py` (currently a dummy/stub):

- Plot 3D phase-space trajectory (x, y, z) colored by segment
- Overlay the shooting points (segment start positions)
- Export plots to `analysis/plots/trajectories/`

---

## Dependencies

- Python 3.9+
- numpy
- pandas

```bash
pip install numpy pandas
```

---

## Relationship to Computation Pipeline

```
COMPUTATION_CONTEXT.md             ANALYSIS_CONTEXT.md (this file)
        │                                   │
        ▼                                   ▼
scripts/find_recurrences.jl         analysis/analyze_hpc_results.py
        │                                   │
        ▼                                   ▼
scripts/generate_task_list.jl       Reads outputs/*/data/rec*/iteration/*.csv
        │                                   │
        ▼                                   ▼
scripts/hpc_worker.jl (HPC)         Produces analysis/status_matrix.csv,
        │                            analysis/status_summary.csv,
        ▼                            analysis/cases_to_rerun.csv,
outputs/ (CSV files)                analysis/rec_overview.csv
```

---

## ⚠️ Notes

- The analysis script only reads the **first ~5 lines** (NaN scan) and the
  **last 2 lines** (comment marker + final data row) of each CSV — it never
  loads the full file into memory.
- Missing CSV files are classified as `no_data` (not an error).
- The `converged_T` column in `rec_overview.csv` is intentionally left NaN
  for the user to fill in after manually inspecting which recurrences
  correspond to true periodic orbits of the Lorenz system.
- The script lives in `analysis/analyze_hpc_results.py` and by default reads
  from `../outputs` and writes outputs to `analysis/`.
- For the old "new vs old" comparison workflow, see the git history of
  `analyze_hpc_results.py` — this is no longer needed.
