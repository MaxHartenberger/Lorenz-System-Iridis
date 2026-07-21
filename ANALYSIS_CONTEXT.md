# Project Analysis Pipeline — Lorenz L-BFGS Results Analysis

## What This Analysis Does

After the HPC sweep completes, `analysis/analyze_hpc_results.py` reads every
per-iteration CSV produced by `hpc_worker.jl` and classifies each
`(T, rec_id, N, m)` case into one of eight statuses.  It then produces summary
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
├── T05/data_lbfgs/
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
├── T10/data_lbfgs/ ...                  # same structure for T=10
├── T20/data_lbfgs/ ...
├── T40/data_lbfgs/ ...
├── T80/data_lbfgs/ ...
└── T160/data_lbfgs/ ...
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
| **converged** | `‖F‖ ≤ 1e-8` (last line = `# converged` and `e_norm` on final data row ≤ 1e-8) |
| **no_data** | CSV file is missing entirely (set by `collect_results`, not by `classify_csv`) |
| **incomplete** | File exists but can't be classified: too short, unparseable data line, or unknown/missing comment marker |
| **not_converged** | Last line = `# did_not_converge` and iteration < 1,000,000 (optimisation stopped early without converging) |
| **hit_maxiter** | Last line = `# did_not_converge` and iteration ≥ 1,000,000 |
| **diverged** | `NaN` appeared in the `e_norm` column during the first few iterations |
| **crashed** | Last line = `# crashed` (exception caught by `hpc_worker.jl`) |
| **error_should_be_converged** | Last line = `# converged` but `e_norm` > 1e-8 (logic error — should never happen) |

### Detailed Decision Tree

```
Read CSV (classify_csv / get_csv_stats)
  ├─ NaN in e_norm (first few data rows)? → diverged
  ├─ File has < 2 non-empty lines?        → incomplete
  ├─ Data line unparseable?                → incomplete
  └─ Parse last 2 lines [data, comment]
       ├─ "# converged"?
       │    ├─ e_norm ≤ 1e-8  → converged ✓
       │    └─ e_norm > 1e-8  → error_should_be_converged
       ├─ "# did_not_converge"?
       │    ├─ iteration ≥ 1,000,000 → hit_maxiter
       │    └─ iteration < 1,000,000 → not_converged
       ├─ "# crashed"?              → crashed
       └─ Unknown / no comment       → incomplete

Missing files are marked "no_data" by collect_results (never reach classify_csv).
```

**Performance note**: The script only reads the first ~5 lines (for NaN scan) and
the last 2 lines (comment marker + final data row, via seek-based tail read),
so it is fast even with 20k+ iteration CSV files.

---

## Output Files (produced by `analysis/analyze_hpc_results.py`)

All outputs are saved to the `analysis/` folder (or a custom `--out-dir`):

| File | Description |
|------|-------------|
| `status_matrix.csv` | Full T × (rec_id, N, m) MultiIndex CSV. Each cell is one of: `converged`, `no_data`, `incomplete`, `not_converged`, `hit_maxiter`, `diverged`, `crashed`, `error_should_be_converged` |
| `status_summary.csv` | Counts of each status per T value |
| `cases_to_rerun.csv` | Flat list of all non-converged cases (T, rec_id, N, m, status) |
| `rec_overview.csv` | Per-recurrence overview: one row per (T, rec_id) with aggregated stats |
| `best_pair_summary.csv` | Per-T summary of the (N,m) pair that is most often the fastest across recurrences |

### `rec_overview.csv` Columns

| Column | Description |
|--------|-------------|
| `T` | Target period |
| `rec_id` | Recurrence candidate ID |
| `converged` | "yes" if ANY (N,m) combo converged, else "no" |
| `num_converged` | How many combos converged (out of 49) |
| `num_incomplete` | How many combos have incomplete/unparseable CSVs |
| `num_not_converged` | How many combos stopped early without converging |
| `num_hit_maxiter` | How many combos hit the iteration limit |
| `num_diverged` | How many combos diverged (NaN in e_norm) |
| `num_crashed` | How many combos crashed (exception in hpc_worker) |
| `num_error_should_be_converged` | How many combos flagged # converged but e_norm > 1e-8 |
| `num_no_data` | How many combos have no CSV file |
| `best_N` / `best_m` | (N,m) of the converged combo with fewest iterations |
| `best_iter` | Iteration count of that fastest combo |
| `min_N_converged` | Smallest N that achieved convergence |
| `min_m_converged` | Smallest m that achieved convergence |
| `median_iter` | Median iterations among all converged combos for this recurrence |
| `converged_T` | Placeholder column (NaN); user fills this in later with a manual check of whether the recurrence itself is a true periodic orbit |

### `best_pair_summary.csv` Columns

| Column | Description |
|--------|-------------|
| `T` | Target period group (T05, T10, T20, T40, T80, T160) |
| `best_N` | Number of shooting segments of the modal-best pair |
| `best_m` | L-BFGS memory of the modal-best pair |
| `win_count` | How many recurrences this (N,m) was the fastest (fewest iterations) |
| `total_recs_with_converged` | Total recurrences in this T group with ≥1 converged combo |
| `win_fraction` | `win_count / total_recs_with_converged` — how dominant this pair is |
| `runner_up_N` / `runner_up_m` | The second-most-frequent best pair |
| `runner_up_wins` | How many recurrences the runner-up won |

### Tie-Breaking Convention

When two or more (N,m) combinations have the **same number of iterations** for
a given recurrence, the one with the **lower m** (L-BFGS memory) is preferred
as the "best" combo.  This rule is applied consistently across:

- `rec_overview.csv`  — `best_N` / `best_m` columns
- `best_pair_summary.csv`  — `best_N` / `best_m` columns
- All plot functions in `plot_results.py` that identify the fastest combo

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
scripts/generate_task_list.jl       Reads outputs/*/data_lbfgs/rec*/iteration/*.csv
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
