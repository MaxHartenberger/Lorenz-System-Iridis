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
hpc_worker.jl  (submitted via SLURM job array, one task per line in tasks.txt)
    ↓
outputs/TXX/data_lbfgs/recXXX/iteration/NXX_mXX.csv  (per-iteration history)
    ↓   final line of each CSV:
    # converged  |  # did_not_converge  |  # crashed
    ↓   (if converged)
outputs/TXX/data_lbfgs/recXXX/trajectory/NXX_mXX_trajectory.csv  (phase-space trajectory)
```

---

## Key Files and Their Roles

| File | Role |
|------|------|
| `scripts/find_recurrences.jl` | Finds near-recurrence candidates; saves to `recurrences/` |
| `scripts/generate_task_list.jl` | Creates `tasks.txt` — one line per `(T, rec_id, N, m)` |
| `scripts/hpc_worker.jl` | Runs ONE case: `julia hpc_worker.jl <T> <rec_id> <N> <m>` |
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

### 2. Submit

Submit the job array via a SLURM script that reads `tasks.txt` by line number
(create your own `submit_sweep.slurm` based on `test_one_run.slurm` as a template,
adding `#SBATCH --array=1-$NTASKS` and using `$SLURM_ARRAY_TASK_ID` to index into
the task file).

```bash
# Count total tasks and submit:
NTASKS=$(wc -l < tasks.txt)
sbatch --array=1-$NTASKS submit_sweep.slurm
```

**Why this works:** Iridis `MaxArraySize=1001` caps the task ID index, not just
the count. If you have more than 1000 tasks, split `tasks.txt` into separate
chunk files and submit separate arrays for each:

```bash
split -d -l 1000 tasks.txt chunk_
# Submit one array per chunk file (edit submit_sweep.slurm's TASKS_FILE to point to each chunk)
sbatch --array=1-1000 submit_sweep.slurm   # with TASKS_FILE=chunk_00
sbatch --array=1-500 submit_sweep.slurm    # with TASKS_FILE=chunk_01
```

### 3. Monitor

```bash
squeue -u $USER                          # all your jobs
grep -r '# converged' outputs/ | wc -l   # how many converged
grep -r '# crashed' outputs/ | wc -l     # how many crashed
```

---

## Output Structure

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
│   └── rec002/...
├── T10/data_lbfgs/...
└── T20/data_lbfgs/...
```

### CSV Formats

**Iteration CSV** (`iteration/NXX_mXX.csv`):
```
iter,e_norm,grad_norm,lambda,T_curr
0,3.7941945225620959e-03,...
...
758,9.8605768907545449e-09,...,5.369891
# converged
```

**Trajectory CSV** (`trajectory/NXX_mXX_trajectory.csv`) — only present for converged cases:
```
t,x,y,z,segment
0.0000000000e+00,4.7397090703043219e+00,1.1538803179091690e+00,2.7773011046743786e+01,1
...
```

Each iteration CSV has columns: `iter, e_norm, grad_norm, lambda, T_curr`

The last line is a comment: `# converged`, `# did_not_converge`, or `# crashed`.

Each trajectory CSV has columns: `t, x, y, z, segment` — one row per Δt = 0.01
step along the converged periodic orbit, with `segment` ∈ {1, …, N}.

---


## Single Test Run

To test one case before a full sweep:

```bash
sbatch test_one_run.slurm
# → runs hpc_worker.jl 5.0 1 5 5
# → output in outputs/T05/data_lbfgs/rec001/iteration/N05_m05.csv
```
