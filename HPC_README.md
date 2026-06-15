# L-BFGS Periodic Orbit Search — Lorenz System

## HPC (Iridis) Usage Guide

This project searches for periodic orbits of the Lorenz system using L-BFGS
optimisation over shooting segments.  The parameter sweep is **embarrassingly
parallel** — every `(T, rec_id, N, m)` combination is independent.

---

## ⚠️ Filesystem Quota Warning

| Limit | `/home` |
|-------|---------|
| Soft data | 110 GB |
| Hard data | 130 GB |
| Soft inodes | **160,000** |
| Hard inodes | **200,000** |

The fine-grained sweep produces **~20,000–30,000 CSV files** per full run
(6 T values × 60–100 recurrences × 49 (N,m) combos).  That is ~15 % of your
entire inode quota in `/home`.  **Always send output to `/scratch`** (see below).

---

## Quick Start (recommended workflow)

**Iridis `MaxArraySize = 1001`.**  The per-recurrence strategy (577 tasks) fits;
the fine-grained strategy (28,273 tasks) does not.

```bash
# 1. Load Julia on the login node
module load julia/1.10

# 2. Generate the recurrence-level task list (~577 tasks)
julia scripts/generate_rec_task_list.jl --output rec_tasks.txt

# 3. Submit the job array  (N = number of lines in rec_tasks.txt)
sbatch --array=1-<N> submit_per_recurrence.slurm
```

Each array task gets 8 CPUs and runs all 49 (N,m) combinations for one
recurrence in parallel via Julia threads.

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

### Strategy A: Per-recurrence (recommended ✓) — one task per `(T, rec_id)`

| Pro | Con |
|-----|-----|
| **Fits in 1001 array limit** (577 tasks) | Needs 8 CPUs per task |
| Julia compiles once per recurrence, runs 49 combos in threads | |
| Few SLURM tasks to manage | |

**This is the only strategy that works as a single job array on Iridis.**

```bash
julia scripts/generate_rec_task_list.jl --output rec_tasks.txt
# Prints: Total tasks: 577
sbatch --array=1-577 submit_per_recurrence.slurm
```

### Strategy B: Fine-grained (use only for small subsets) — one task per `(T, rec_id, N, m)`

| Pro | Con |
|-----|-----|
| Max parallelism | 28,273 tasks — **exceeds MaxArraySize=1001** |
| Task failure isolates one case | Must split into 29 separate submissions |
| Easy to re-submit failed cases | Julia startup overhead per task |

```bash
julia scripts/generate_task_list.jl --output tasks.txt
# Must submit in chunks of ≤1000:
sbatch --array=1-1000    submit_sweep.slurm
sbatch --array=1001-2000  submit_sweep.slurm
# ... etc
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

**Always use `/scratch` to avoid exhausting your `/home` inode quota.**

A full sweep creates ~20,000–30,000 small CSV files, consuming ~15 % of your
`/home` inode limit (160K).  `/scratch` has no inode quota and is the correct
place for bulk job output.

Edit `submit_sweep.slurm` (or `submit_per_recurrence.slurm`) and add
`--output-dir` to the Julia invocation:

```bash
julia scripts/hpc_worker.jl "$T" "$REC_ID" "$N" "$M" \
    --output-dir /scratch/$USER/lorenz-output
```

After the run, copy only the files you need back to `/home` for analysis:

```bash
cp -r /scratch/$USER/lorenz-output ~/lorenz-results/
# Or selectively:
rsync -av --include='*/' --include='*CONVERGED*' --exclude='*' \
    /scratch/$USER/lorenz-output/ ~/lorenz-converged-only/
```

### Inode budget per run

| Strategy | CSV files | Log files | Total inodes | % of 160K limit |
|----------|-----------|-----------|-------------|-----------------|
| Fine-grained (per `(T,rec,N,m)`) | ~25,000 | ~25,000 | ~50,000 | **31 %** |
| Coarse-grained (per `(T,rec)`) | ~25,000 | ~400 | ~25,400 | **16 %** |

These are upper-bound estimates (6 T × 85 recs × 49 combos).  Reduce `max_recs`
in `generate_task_list.jl` to lower the count.

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

1. **Filesystem quotas**: A full sweep generates tens of thousands of small CSV
   files.  Do **not** write output to `/home` — you will exhaust your inode quota
   (160K soft / 200K hard).  Always use `/scratch/$USER/...` via `--output-dir`.

2. **Julia module availability**: Run `module avail julia` on the login node to
   see which Julia versions are installed.  Update `submit_sweep.slurm` and
   `submit_per_recurrence.slurm` with the correct version.

3. **Package dependencies**: This project depends on `Flows`, `NKSearch`,
   `StreamingRecurrenceAnalysis`, and `ToySystems`.  These must be available
   in your Julia `LOAD_PATH` or installed as a local project.  Consider creating a
   `Project.toml` at the project root and running `julia --project -e 'using Pkg; Pkg.instantiate()'`
   on the login node before submitting.

4. **Scratch cleanup**: `/scratch` is not backed up and old files may be purged.
   Copy important results back to `/home` after each run.

5. **Memory**: Each L-BFGS case with large N and m uses more memory.  The default
   2 GB per task is generous for moderate (N, m).  For large N (≥ 160), you may
   need 4–8 GB.
