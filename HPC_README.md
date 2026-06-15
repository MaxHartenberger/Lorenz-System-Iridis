# L-BFGS Periodic Orbit Search — Lorenz System

## HPC (Iridis 5) Usage Guide

This project searches for periodic orbits of the Lorenz system using L-BFGS
optimisation over shooting segments.  The parameter sweep is **embarrassingly
parallel** — every `(T, rec_id, N, m)` combination is independent.

---

## Quick Start (recommended workflow)

```bash
# 1. Load Julia on the login node
module load julia/1.10

# 2. Generate the task list
julia scripts/generate_task_list.jl --output tasks.txt

# 3. Submit the job array  (N = number of lines in tasks.txt)
sbatch --array=1-<N> submit_sweep.slurm
```

---

## Directory Structure (after HPC run)

```
Lorenz-System-Iridis/
├── scripts/
│   ├── hpc_worker.jl              # ← single-case worker (1 CPU per case)
│   ├── hpc_worker_per_rec.jl      # ← multi-threaded per-recurrence worker
│   ├── generate_task_list.jl      # ← generates tasks.txt (one line per case)
│   ├── generate_rec_task_list.jl  # ← generates rec_tasks.txt (one line per rec)
│   ├── generate_data_from_recurrences.jl  # original sequential script
│   └── find_recurrences.jl
├── submit_sweep.slurm             # ← SLURM job-array: one task per (T,rec,N,m)
├── submit_per_recurrence.slurm    # ← SLURM job-array: one task per (T,rec)
├── tasks.txt                      # ← generated task list
├── logs/                          # ← SLURM stdout/stderr per task
├── recurrences/                   # ← input: near-recurrence candidates
│   ├── T05/recurrences.csv
│   ├── T10/recurrences.csv
│   └── ...
└── output/                        # ← output CSVs
    ├── T05/data/rec001/N05_m05.csv
    ├── T10/data/...
    └── ...
```

---

## Two Submission Strategies

### Strategy A: Fine-grained (one task per `(T, rec_id, N, m)`)

| Pro | Con |
|-----|-----|
| Max parallelism — every case runs independently | Julia startup overhead per task (~30 s compilation) |
| Task failure isolates one case | Many small tasks (thousands) |
| Easy to re-submit failed cases | |

**Use when:** You have many nodes available and want minimum wall-clock time.

```bash
julia scripts/generate_task_list.jl --output tasks.txt
# Suppose tasks.txt has 4116 lines:
sbatch --array=1-4116 submit_sweep.slurm
```

### Strategy B: Coarse-grained (one task per `(T, rec_id)`, Julia threads for `(N, m)`)

| Pro | Con |
|-----|-----|
| Lower overhead — Julia compiles once per task | Needs multi-threaded Julia |
| Fewer SLURM tasks to manage | If a task fails, you lose all (N,m) for that recurrence |
| Better CPU utilisation | |

**Use when:** You have fewer nodes, want to use multi-core within each node.

```bash
julia scripts/generate_rec_task_list.jl --output rec_tasks.txt
# Suppose rec_tasks.txt has 300 lines (one per recurrence):
sbatch --array=1-300 submit_per_recurrence.slurm
```

---

## Customising the Parameter Sweep

Edit the constants at the top of the scripts you use:

| File | Constants |
|------|-----------|
| `scripts/generate_task_list.jl` | `T_targets`, `Ms`, `Ns` |
| `scripts/hpc_worker.jl` | (reads from command line) |
| `scripts/hpc_worker_per_rec.jl` | `Ms`, `Ns`, `MAXITER` |

Then regenerate the task list before submitting.

---

## Monitoring Jobs

```bash
# View your queued/running jobs
squeue -u $USER

# View output of a specific job
cat logs/slurm_<JOBID>_*.out

# Cancel a job array
scancel <JOBID>

# Cancel specific array tasks
scancel <JOBID>_[1-100]
```

---

## Handling Failures

If some array tasks fail (exit code ≠ 0):

```bash
# Check which tasks failed
grep -L "CONVERGED\|Finished" logs/slurm_<JOBID>_*.out

# Re-submit only failed tasks (example: tasks 5, 12, 33)
sbatch --array=5,12,33 submit_sweep.slurm
```

---

## Custom Output / Scratch Directory

To write output to `/scratch` (faster, recommended for large runs):

```bash
# Edit submit_sweep.slurm and change the julia invocation to:
julia scripts/hpc_worker.jl "$T" "$REC_ID" "$N" "$M" \
    --output-dir /scratch/$USER/lorenz-output
```

---

## Interactive Testing (before submitting batch)

```bash
# Request an interactive node
sinteractive --partition=serial --time=01:00:00 --mem=4000

# Load Julia & test a single case
module load julia/1.10
julia scripts/hpc_worker.jl 5.0 1 10 5
# Should print: T=5.0  rec_id=1  N=10  m=5  |  CONVERGED/DID NOT CONVERGE  ...
```

---

## SLURM Directive Reference

| Directive | Meaning |
|-----------|---------|
| `--nodes=1` | One physical node |
| `--ntasks=1` | One process |
| `--cpus-per-task=N` | N CPU cores for this task |
| `--mem=2000` | 2000 MB RAM |
| `--time=HH:MM:SS` | Wall-clock time limit |
| `--partition=batch` | Which queue to use |
| `--array=1-N` | Job array of N tasks |

See the [Iridis wiki](https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Submitting-Jobs-Slurm.aspx) for full documentation.

---

## Cautions

1. **Julia module availability**: Run `module avail julia` on the login node to
   see which Julia versions are installed.  Update `submit_sweep.slurm` accordingly.

2. **Package dependencies**: This project depends on `Flows`, `NKSearch`,
   `StreamingRecurrenceAnalysis`, and `ToySystems`.  These must be available
   in your Julia `LOAD_PATH` or installed as a local project.  Consider creating a
   `Project.toml` at the project root and running `julia --project -e 'using Pkg; Pkg.instantiate()'`
   on the login node before submitting.

3. **Disk I/O**: All tasks write to the same output directory.  The per-task CSV
   files are small, but with thousands of tasks the filesystem may become a
   bottleneck.  Consider using `/scratch` for output during runs.

4. **Memory**: Each L-BFGS case with large N and m uses more memory.  The default
   2 GB per task is generous for moderate (N, m).  For large N (≥ 160), you may
   need 4–8 GB.
