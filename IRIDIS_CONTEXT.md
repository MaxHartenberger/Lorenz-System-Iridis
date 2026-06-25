# Iridis X HPC Cluster — General Reference

## User's Partition
**amd_student** — for UG/PGT students only.

| CPUs | Mem/CPU | Nodes | GPUs | Walltime |
|------|---------|-------|------|----------|
| 256 | 3.75 GB | 4 | None | 60 hours |

No hyperthreading. No GPUs. Max 60h per job.

## Storage Quotas

| Filesystem | Soft Data | Hard Data | Soft Inodes | Hard Inodes |
|-----------|----------|----------|-------------|-------------|
| `/home` | 110 GB | 130 GB | 160,000 | 200,000 |
| `/scratch` | 1500 GB | 2000 GB | 500,000 | 600,000 |

- **Soft limit** can be exceeded for up to 14 days, then becomes hard.
- **Hard limit** cannot be exceeded.
- **Data limit** = total storage used. **Inode limit** = total number of files.
- Put large/output data in `/scratch`, not `/home`.

Check quotas:
```bash
mmlsquota --block-size=G iridisfs:home iridisfs:scratch
myquota   # short summary
```

## Key Constraints
- **No GPUs** — CPU-only code.
- **Max 60h walltime** per job.
- **3.75 GB RAM per CPU** — keep per-core memory usage under this.
- **4 nodes, 256 CPUs total** — use `#SBATCH --ntasks` / `--cpus-per-task` accordingly.
- **Scratch space** (`/scratch`) for job output; `/home` for code and small configs.

## SLURM Submission Basics

Submit with: `sbatch script.slurm` (returns a job ID).

### Minimal Job Script
```bash
#!/bin/bash
#SBATCH --partition=amd_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3700             # MB (~3.75 GB per CPU max)
#SBATCH --time=01:00:00        # HH:MM:SS or dd-HH:MM:SS, max 60h
#SBATCH --job-name=myjob
#SBATCH --output=logs/%x_%j.out   # %x=job name, %j=job ID
#SBATCH --error=logs/%x_%j.err

cd $HOME/path/to/project
./my_program
```

### Common SLURM Directives
| Directive | Meaning |
|-----------|---------|
| `--partition=amd_student` | Which queue/partition |
| `--ntasks=N` | Total MPI processes (or independent tasks) |
| `--cpus-per-task=N` | Threads per task (for OpenMP / Julia `--threads`) |
| `--nodes=N` | Number of nodes (max 4 on amd_student) |
| `--mem=N` | Memory per node in MB |
| `--mem-per-cpu=N` | Alternative: memory per CPU core |
| `--time=HH:MM:SS` | Walltime (max 60h = `60:00:00`) |
| `--output=path` | Stdout file (%j = job ID, %x = job name) |
| `--error=path` | Stderr file |
| `--job-name=name` | Job name visible in `squeue` |

### Job Arrays
```bash
#SBATCH --array=1-100          # 100 tasks, $SLURM_ARRAY_TASK_ID = 1..100
#SBATCH --array=1-100%10       # max 10 running at once
```
Use `$SLURM_ARRAY_TASK_ID` in the script to select input parameters.

### Monitoring & Control
```bash
squeue -u $USER                # See your jobs
squeue -u $USER --start        # With expected start times
scancel <jobid>                # Cancel one job
scancel -u $USER               # Cancel ALL your jobs
scontrol hold <jobid>          # Pause a job
```

### Job Limits
- **MaxArraySize:** 1001. Caps the **task ID index** — `--array=1001-2000` is rejected because index 2000 > 1001. For >1000 tasks, split into multiple arrays (see `COMPUTATION_CONTEXT.md` for the project's chunking pattern).
- **MaxJobs:** unlimited (verify with `sacctmgr show associations user=$USER format=MaxJobs`).

### Key Rules
- **Never run computation on login nodes** — always use `sbatch`/`srun`.
- SLURM inherits your environment; put `module load ...` and `export ...` in the script for reproducibility.
- Working directory is the script's location; `cd` explicitly if needed.
