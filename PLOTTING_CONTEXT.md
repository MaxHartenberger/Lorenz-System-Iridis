# Plotting Context — `plot_results.py`

## Overview
`plot_results.py` generates publication-quality plots (PDF + PNG) from L-BFGS periodic-orbit search results on the Lorenz system. It parses `.out` HPC log files for convergence metadata and reads CSV convergence-curve data on demand.

## Quick Start
```bash
# Generate ALL plots
python3 plot_results.py

# Generate only specific plots
python3 plot_results.py --plots convcount maxiter best3 minN minM

# Single per-recurrence plot on a subset (modify list in script)
python3 plot_results.py --plots fig1

# List available plot names
python3 plot_results.py --list

# Dry run (check what would be done)
python3 plot_results.py --dry-run
```

## Output Structure
```
plots/python_plots/
├── fig01_convergence_curves/    # Per-recurrence ‖F‖ vs iter (subplots by N, varying m)
│   └── fig01_T{target}_rec{id}.{pdf,png}
├── fig06_iterations_heatmap/    # Per-recurrence 7×7 (N×m) heatmaps
│   └── fig06_T{target}_rec{id}.{pdf,png}
├── fig12a_min_N_vs_T.{pdf,png}  # Min N to converge vs actual T (one line per m)
├── fig12b_min_m_vs_T.{pdf,png}  # Min m to converge vs actual T (one line per N)
├── maxiter_vs_T.{pdf,png}       # Max iterations (slowest converged) per recurrence
├── best3_combos_vs_T.{pdf,png}  # Max iterations + 3 best (N,m) overlaid
└── convcount_vs_T.{pdf,png}     # Number of converged combos per recurrence (out of 49)
```

## Plot Descriptions

| CLI name | Plot | Description |
|----------|------|-------------|
| `fig1` | Convergence curves per recurrence | One figure per recurrence. Subplots = one per N where ≥1 m converged. Each subplot shows ‖F‖ vs iteration for all converged m values (semilogy). |
| `fig6` | Iterations heatmap per recurrence | One figure per recurrence. 7×7 heatmap (N rows × m columns), annotated with iteration count (converged) or status symbol (✗/·). Log color scale. |
| `minN` | Min N to converge vs T | Scatter/line plot: x = actual converged period T, y = minimum N that converged. One line per m value. |
| `minM` | Min m to converge vs T | Scatter/line plot: x = actual converged period T, y = minimum m that converged. One line per N value. |
| `maxiter` | Max iterations per recurrence | Column chart: x = recurrence (ordered by actual converged T), y = iterations of slowest converged (N,m). Color-coded by T. Log y-scale. |
| `best3` | Best combos overlaid on maxiter | Same as maxiter but with the 3 fastest (N,m) combos shown as scatter markers. Annotated with (N,m) values. |
| `convcount` | Convergence count per recurrence | Column chart: x = recurrence, y = number of converged combos (out of 49 max). Color-coded by T. |

## Data Flow
1. **Parse `.out` logs** (`parse_log_file()`) — extract `T_target`, `rec_id`, and for each (N,m): status (converged/hit_maxiter/failed/no_data), actual converged T, iterations, wall time.
2. **Filter** to recurrences with ≥1 converged combo.
3. **Sort** recurrences by their *representative T* (T of the fastest-converged combo for that recurrence).
4. **Lazy-load CSVs** (`load_csv_data()`) — only when a plot needs convergence curve data.

## Adding a New Plot
1. Write a function `def plot_mynewplot(recs: List[RecurrenceData]):` that creates figures and calls `savefigs()`.
2. Register it in the `PLOT_FUNCTIONS` dict at the bottom of the script:
   ```python
   PLOT_FUNCTIONS = {
       ...
       "mynewplot": (plot_mynewplot, "Description of what this plot shows"),
   }
   ```
3. It will automatically appear in `--list` and be runnable via `--plots mynewplot`.

## Key Conventions
- **Only converged combos** contribute data to curves/heatmaps. Non-converged cases are shown as symbols only.
- **Representative T**: For each recurrence, we use the actual T from the fastest-converged (N,m) combo. If multiple combos converged to slightly different T, the fastest one wins.
- **N-subplot filtering** (fig1): Only N values with ≥1 converged m are shown. But for a given N, all m values are plotted (so non-converged m's for that N are simply absent from the curves — only converged combos have curves).
- **CSV loading**: Only converged CSVs are loaded; this keeps things manageable.

## Dependencies
- Python 3.9+
- numpy, pandas, matplotlib

## Edge Cases Handled
- Recurrences with 0 converged combos → excluded entirely
- FAIL combos (Julia crash) → shown as ✗, no CSV curve
- Timeout cases (no log lines for some combos) → shown as · (not reached)
- Missing CSV files → handled gracefully (None return from `load_csv_data`)
- Large CSVs (up to 1M rows) → loaded on demand only for converged combos
