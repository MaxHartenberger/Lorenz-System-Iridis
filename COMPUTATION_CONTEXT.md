# Project Computation Pipeline — Lorenz L-BFGS Periodic Orbit Search

## What This Project Does

Searches for periodic orbits of the Lorenz system (σ=10, ρ=28, β=8/3) using
L-BFGS optimisation over multiple shooting segments.  The parameter sweep runs
over these dimensions:

| Dimension | Values |
|-----------|--------|
| **T** (target period) | 5, 10, 20, 40, 80, 160 |
| **N** (shooting segments) | 5, 10, 20, 40, 80, 160, 320 |
| **m** (L-BFGS memory) | 5, 10, 20, 40, 80, 160, 320 |
| **rec_id** (recurrence candidate) | 1–100 per T |

Every `(T, rec_id, N, m)` combination is an independent L-BFGS optimisation.

---

## Pipeline Overview

```
find_recurrences.jl
    ↓
recurrences/TXX/recurrences.csv   (input data)
    ↓
generate_task_list.jl
    ↓
tasks.txt  (one line per case:  T rec_id N m)
    ↓
first_sweep.slurm  +  hpc_worker.jl
    ↓
output/TXX/data/recXXX/NXX_mXX.csv  (per-iteration history)
    ↓   final line of each CSV:
    # converged  |  # did_not_converge  |  # crashed
```

---

## Key Files and Their Roles

| File | Role |
|------|------|
| `scripts/find_recurrences.jl` | Finds near-recurrence candidates; saves to `recurrences/` |
| `scripts/generate_task_list.jl` | Creates `tasks.txt` — one line per `(T, rec_id, N, m)` |
| `scripts/hpc_worker.jl` | Runs ONE case: `julia hpc_worker.jl <T> <rec_id> <N> <m>` |
| `first_sweep.slurm` | SLURM job array: reads a task chunk file, runs `hpc_worker.jl` |
| `test_one_run.slurm` | Minimal SLURM script for a single test case (no array) |

---

## Running a Sweep (Step by Step)

### 1. Generate the task list

```bash
module load julia/1.12.4

# T05, T10, T20 — full grid (all N,m combos converge well):
julia scripts/generate_task_list.jl --T 5,10,20

# T40 — skip N=5, m=5 (small combos don't converge for longer periods):
julia scripts/generate_task_list.jl --T 40 --N-min 10 --m-min 10

# T80 — skip N=5,10 and m=5,10:
julia scripts/generate_task_list.jl --T 80 --N-min 20 --m-min 20

# T160 — skip N=5,10, 20 and m=5,10,20:
julia scripts/generate_task_list.jl --T 160 --N-min 40 --m-min 40

# Limit recurrences per T:
julia scripts/generate_task_list.jl --T 5,10,20 --max-records 50
```

**Why filter by N and m?** for longer orbit periods (T ≥ 40), small shooting
segments (N < 10) and small L-BFGS memory (m < 10) are known to diverge or
fail to converge. Filtering them out saves compute time and avoids unnecessary
`# did_not_converge` results.

### 2. Split and submit

The script prints exact commands. Example for ~14,700 tasks (T05, T10, T20):

```bash
split -d -l 1000 tasks.txt chunk_
sbatch --array=1-1000 first_sweep.slurm chunk_00
sbatch --array=1-1000 first_sweep.slurm chunk_01
# ... up to chunk_14
```

**Why split?** Iridis `MaxArraySize=1001` caps the task ID index, not just the
count. `--array=1001-2000` is rejected because index 2000 > 1001. Splitting
into files of ≤1000 lines and using `--array=1-1000` for each avoids this.

### 3. Monitor

```bash
squeue -u $USER                          # all your jobs
grep -r '# converged' output/ | wc -l    # how many converged
grep -r '# crashed' output/ | wc -l      # how many crashed
```

---

## Output Structure

```
output/
├── T05/data/rec001/N05_m05.csv
│                  N05_m10.csv
│                  ...
│           rec002/...
├── T10/data/...
└── T20/data/...
```

Each CSV has columns: `iter, e_norm, grad_norm, lambda, T_curr`

The last line is a comment: `# converged`, `# did_not_converge`, or `# crashed`.

SLURM logs: `logs/sweep_<jobid>_<taskid>.out` and `.err`

---

## ⚠️ Inode Warning

A full sweep (6 T values × 100 recs × 49 (N,m) combos) produces ~29,000 CSV
files — nearly 20% of the `/home` inode quota (160,000).  Use `--output-dir`
to route CSVs to `/scratch`:

```bash
julia scripts/hpc_worker.jl ... --output-dir /scratch/$USER/output
```

Scratch has 500,000 inodes — plenty of headroom.

---

## Single Test Run

To test one case before a full sweep:

```bash
sbatch test_one_run.slurm
# → runs hpc_worker.jl 5.0 1 5 5
# → output in output/T05/data/rec001/N05_m05.csv
```
