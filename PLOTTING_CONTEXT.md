# Plotting Context — `plot_results.py`

## Overview
`plot_results.py` generates publication-quality plots (PDF + PNG) from L-BFGS periodic-orbit search results on the Lorenz system. It reads per-iteration CSV files for convergence metadata (status, iterations, T_actual) and loads full CSV data on demand for convergence-curve plots.

## Quick Start
```bash
# Generate ALL plots
python3 plot_results.py

# Generate only specific plots
python3 plot_results.py --plots convcount maxiter best3 fastest

# Single per-recurrence plot on a subset (modify list in script)
python3 plot_results.py --plots fig1

# List available plot names
python3 plot_results.py --list

# Dry run (check what would be done)
python3 plot_results.py --dry-run
```

## Output Structure
```
analysis/plots/
├── fig01_convergence_curves/    # Per-recurrence ‖F‖ vs iter (subplots by N, varying m)
│   └── fig01_T{target}_rec{id}.{pdf,png}
├── fig06_iterations_heatmap/    # Per-recurrence 7×7 (N×m) heatmaps
│   └── fig06_T{target}_rec{id}.{pdf,png}
├── agg_heatmaps/                # Per-T aggregate heatmaps (median iterations + success rate)
│   └── agg_heatmap_{T_label}.{pdf,png}
├── maxiter_vs_T.{pdf,png}       # Max iterations (slowest converged) per recurrence
├── best3_combos_vs_T.{pdf,png}  # Max iterations + 3 best (N,m) overlaid
├── fastest_bar_annotated.{pdf,png} # Fastest converged per recurrence — bars annotated with ID, T, (N,m)
└── convcount_vs_T.{pdf,png}     # Number of converged combos per recurrence (out of 49)
```

## Plot Descriptions

| CLI name | Plot | Description |
|----------|------|-------------|
| `fig1` | Convergence curves per recurrence | One figure per recurrence. Subplots = one per N where ≥1 m converged. Each subplot shows ‖F‖ vs iteration for all converged m values (semilogy). |
| `fig6` | Iterations heatmap per recurrence | One figure per recurrence. 7×7 heatmap (N rows × m columns), annotated with iteration count (converged) or status symbol (✗/·). Log color scale. |
| `maxiter` | Max iterations per recurrence | Column chart: x = recurrence (ordered by actual converged T), y = iterations of slowest converged (N,m). Color-coded by T. Log y-scale. |
| `best3` | Best combos overlaid on maxiter | Same as maxiter but with the 3 fastest (N,m) combos shown as scatter markers. Annotated with (N,m) values. |
| `fastest` | Fastest converged per recurrence | Column chart: x = recurrence, y = iterations of the fastest converged (N,m) combo. Light-grey bars annotated inside with orbit ID, final period T, and (N,m). No colorbar. |
| `fastest_by_T` | Fastest bars split by T group | Same as `fastest` but produces one chart per T-target group (T05, T10, T20, T40, T80, T160). |
| `convcount` | Convergence count per recurrence | Column chart: x = recurrence, y = number of converged combos (out of 49 max). Color-coded by T. |
| `agg_heatmaps` | Aggregate per-T heatmaps | Two-panel figure per T group: (left) median iterations across recurrences (log scale), (right) success rate 0–1 across recurrences. 7×7 (N×m) grid. |

## Input Data Structure

The script reads from the `outputs/` directory:

```
outputs/
├── T05/data_lbfgs/
│   ├── rec001/
│   │   └── iteration/             # per-iteration L-BFGS history
│   │       ├── N05_m05.csv
│   │       ├── N05_m10.csv
│   │       └── ...  (49 combos: 7 N × 7 m)
│   ├── rec002/ ...
│   └── recXXX/ ...
├── T10/data_lbfgs/ ...
└── ...
```

**Iteration CSV** (`iteration/N{XX}_m{XX}.csv`):
```
iter,e_norm,grad_norm,lambda,T_curr
0,3.7941945225620959e-03,...
...
758,9.8605768907545449e-09,...,5.369891
# converged
```
- 5 columns: `iter, e_norm, grad_norm, lambda, T_curr`
- Last line is a **comment marker**: `# converged`, `# did_not_converge`, or `# crashed`

## Data Flow
1. **Scan CSVs** (`_discover_recurrences()`) — find all (T_target, rec_id) pairs with CSV files.
2. **Classify each (N,m) combo** (`_parse_csv_status()`) — read first ~5 lines (NaN scan) and last line (comment marker + final data row) to determine: status (converged/hit_maxiter/incomplete/diverged/no_data), iterations, and actual converged T.
3. **Build RecurrenceData objects** — group combos by recurrence, compute representative_T and best_combo.
4. **Filter** to recurrences with ≥1 converged combo.
5. **Sort** recurrences by their *representative T* (T of the fastest-converged combo for that recurrence).
6. **Lazy-load CSVs** (`load_csv_data()`) — only when a plot needs convergence curve data (fig1).

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
- `diverged` combos (NaN in e_norm) → shown as ✗d, no CSV curve
- `incomplete` combos (optimisation stopped early / crashed) → shown as ✗, no CSV curve
- `hit_maxiter` combos → shown as ✗ₘ, no CSV curve
- Missing CSV files → shown as · (no_data, never reached)
- Missing CSV files (file absent) → `load_csv_data()` returns None gracefully
