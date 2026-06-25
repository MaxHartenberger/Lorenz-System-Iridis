#!/usr/bin/env python3
"""
analyze_hpc_results.py
======================
Reads the CSV outputs from L-BFGS periodic-orbit searches on the Lorenz system
and classifies every (T, rec_id, N, m) case as one of:

    converged    –  ‖F‖ ≤ 1e-8  (optimisation succeeded)
    no_data      –  CSV file missing
    incomplete   –  stopped early (>1 iteration) without converging
    hit_maxiter  –  reached 1,000,000 iterations without converging
    diverged     –  NaN appeared in the error norm during iteration 1

Reads from a single ``outputs/`` directory with the structure:
    outputs/TXX/data/recXXX/iteration/NXX_mXX.csv
    outputs/TXX/data/recXXX/trajectory/NXX_mXX_trajectory.csv  (converged only)

Outputs (saved to --out-dir, default: analysis/):
    status_matrix.csv     –  full T × (rec_id, N, m) matrix (MultiIndex CSV)
    status_summary.csv    –  counts of each status per T
    cases_to_rerun.csv    –  all non-converged cases (flat list)
    rec_overview.csv      –  per-recurrence overview (rec_id rows, aggregated stats)
"""

from __future__ import annotations

import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (keep in sync with the Julia scripts)
# ---------------------------------------------------------------------------
NS = [5, 10, 20, 40, 80, 160, 320]
MS = [5, 10, 20, 40, 80, 160, 320]
MAXITER = 1_000_000
CONVERGENCE_TOL = 1e-8

STATUS_DTYPE = pd.CategoricalDtype(
    categories=["converged", "no_data", "incomplete", "hit_maxiter", "diverged"],
    ordered=False,
)

# Regex to parse N##_m##.csv filenames
FNAME_RE = re.compile(r"^N(\d+)_m(\d+)\.csv$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trajectory plotting (stub — not yet implemented)
# ---------------------------------------------------------------------------
def plot_trajectory(trajectory_csv: Path, out_dir: Path | None = None) -> None:
    """Plot a 3D phase-space trajectory from a trajectory CSV.

    This is a **placeholder** function.  When implemented, it will:

    1. Read the trajectory CSV (columns: t, x, y, z, segment).
    2. Plot the 3D orbit (x, y, z) coloured by shooting segment.
    3. Overlay the shooting points (first point of each segment).
    4. Save the figure to ``out_dir`` (or display inline if None).

    Parameters
    ----------
    trajectory_csv : Path
        Path to a ``*_trajectory.csv`` file produced by ``hpc_worker.jl``.
    out_dir : Path or None
        Directory to save the plot.  If None, the plot is displayed
        interactively (if a GUI backend is available).

    Notes
    -----
    - Only converged cases have trajectory CSVs.
    - This function currently does **nothing** (no-op).
    """
    # TODO: Implement trajectory plotting.
    #
    # Example outline:
    #
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    #
    # df = pd.read_csv(trajectory_csv)
    #
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection="3d")
    #
    # for seg in sorted(df["segment"].unique()):
    #     seg_df = df[df["segment"] == seg]
    #     ax.plot(seg_df["x"], seg_df["y"], seg_df["z"],
    #             lw=0.5, label=f"seg {seg}")
    #
    # # Mark shooting points (first point of each segment)
    # start_pts = df.groupby("segment").first()
    # ax.scatter(start_pts["x"], start_pts["y"], start_pts["z"],
    #            c="red", s=20, label="shooting points")
    #
    # ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    # ax.set_title(f"Periodic orbit — {trajectory_csv.stem}")
    #
    # if out_dir:
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(out_dir / f"{trajectory_csv.stem}.png", dpi=150)
    #     plt.close(fig)
    # else:
    #     plt.show()
    pass


def _read_tail_lines(filepath: Path, n: int = 2, chunk_size: int = 8192) -> list[str]:
    """Read the last `n` non-empty lines of a file efficiently.

    Returns a list of at most `n` lines, ordered from first to last
    (i.e., result[-1] is the last non-empty line in the file).
    """
    with open(filepath, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        file_size = fh.tell()
        if file_size == 0:
            return []
        seek_to = max(0, file_size - chunk_size)
        fh.seek(seek_to)
        chunk = fh.read(min(chunk_size, file_size))
        lines = chunk.decode("utf-8", errors="replace").splitlines()
        # Keep only non-empty lines, take up to the last n
        non_empty = [ln for ln in lines if ln.strip()]
        return non_empty[-n:] if len(non_empty) >= n else non_empty


def _read_first_lines(filepath: Path, n: int = 6) -> list[str]:
    """Read the first `n` lines (including header)."""
    with open(filepath, "r") as fh:
        lines = []
        for i, line in enumerate(fh):
            if i >= n:
                break
            lines.append(line)
        return lines


def classify_csv(filepath: Path) -> str:
    """Return the status string for a single CSV file.

    Only reads the first ~5 lines (NaN scan) and the last 2 lines
    (comment marker + final data row), so it is fast even with 20k+ rows.
    """
    try:
        # --- Scan first few data rows for NaN in e_norm (column index 1) ---
        head_lines = _read_first_lines(filepath, 6)
        if len(head_lines) < 2:          # header only → something went wrong
            return "incomplete"

        for line in head_lines[1:]:       # skip header
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[1].strip() == "NaN":
                return "diverged"

        # --- Read last 2 non-empty lines: [second-to-last data, comment] ---
        tail_lines = _read_tail_lines(filepath, n=2)
        if len(tail_lines) < 2:
            return "incomplete"

        comment_line = tail_lines[-1].strip()
        data_line    = tail_lines[-2].strip()

        # --- Parse the comment marker ---
        if comment_line == "# converged":
            parts = data_line.split(",")
            if len(parts) < 2:
                return "incomplete"
            try:
                e_norm = float(parts[1])
            except (ValueError, IndexError):
                return "incomplete"
            if e_norm <= CONVERGENCE_TOL:
                return "converged"
            else:
                return "incomplete"  # shouldn't happen; marker says converged but F > tol

        elif comment_line == "# did_not_converge":
            parts = data_line.split(",")
            if len(parts) < 1:
                return "incomplete"
            try:
                iteration = int(parts[0])
            except (ValueError, IndexError):
                return "incomplete"
            if iteration >= MAXITER:
                return "hit_maxiter"
            else:
                return "incomplete"  # stopped early without converging

        elif comment_line == "# crashed":
            return "incomplete"

        else:
            # Unknown or missing comment marker — treat as incomplete
            return "incomplete"

    except Exception:
        return "incomplete"


def get_csv_stats(filepath: Path) -> tuple[str, float | None, float | None]:
    """Return (status, final_iter, final_e_norm) for a single CSV.

    Like classify_csv() but additionally extracts the iteration count and
    final ‖F‖ from the second-to-last data row.  Returns (None, None) for
    iter/e_norm when they are not meaningful (no_data / diverged / read error).
    """
    try:
        head_lines = _read_first_lines(filepath, 6)
        if len(head_lines) < 2:
            return ("incomplete", None, None)

        for line in head_lines[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[1].strip() == "NaN":
                return ("diverged", None, None)

        tail_lines = _read_tail_lines(filepath, n=2)
        if len(tail_lines) < 2:
            return ("incomplete", None, None)

        comment_line = tail_lines[-1].strip()
        data_line    = tail_lines[-2].strip()

        # Parse data values from the second-to-last line
        parts = data_line.split(",")
        if len(parts) < 2:
            return ("incomplete", None, None)
        try:
            e_norm    = float(parts[1])
            iteration = int(parts[0])
        except (ValueError, IndexError):
            return ("incomplete", None, None)

        if comment_line == "# converged":
            if e_norm <= CONVERGENCE_TOL:
                return ("converged", iteration, e_norm)
            else:
                return ("incomplete", iteration, e_norm)

        elif comment_line == "# did_not_converge":
            if iteration >= MAXITER:
                return ("hit_maxiter", iteration, e_norm)
            else:
                return ("incomplete", iteration, e_norm)

        elif comment_line == "# crashed":
            return ("incomplete", iteration, e_norm)

        else:
            # Unknown marker or legacy format without comment
            return ("incomplete", iteration, e_norm)

    except Exception:
        return ("incomplete", None, None)


def parse_filename(filename: str) -> tuple[int, int] | None:
    """Extract (N, m) from 'N05_m10.csv', or None if it doesn't match."""
    m = FNAME_RE.match(filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def collect_results(output_root: Path) -> pd.DataFrame:
    """Walk `output_root` and build a long-form DataFrame of all cases.

    Returns DataFrame with columns: T, rec_id, N, m, status
    """
    rows: list[dict] = []

    t_dirs = sorted(
        d for d in output_root.iterdir()
        if d.is_dir() and d.name.startswith("T")
    )

    for t_dir in t_dirs:
        T_label = t_dir.name  # e.g. "T05"
        data_dir = t_dir / "data"
        if not data_dir.is_dir():
            continue

        rec_dirs = sorted(
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("rec")
        )

        n_csvs = 0
        for rec_dir in rec_dirs:
            try:
                rec_id = int(rec_dir.name[3:])  # "rec001" → 1
            except ValueError:
                continue

            existing_csvs: set[tuple[int, int]] = set()
            iter_dir = rec_dir / "iteration"
            if iter_dir.is_dir():
                for csv_file in sorted(iter_dir.iterdir()):
                    if not csv_file.name.endswith(".csv"):
                        continue
                    parsed = parse_filename(csv_file.name)
                    if parsed is None:
                        continue
                    N, m = parsed
                    existing_csvs.add((N, m))
                    status, final_iter, final_e_norm = get_csv_stats(csv_file)
                    rows.append({
                        "T": T_label, "rec_id": rec_id, "N": N, "m": m,
                        "status": status, "iter": final_iter, "e_norm_final": final_e_norm,
                    })
                    n_csvs += 1

            # Mark missing (N, m) combos as "no_data"
            for N in NS:
                for m in MS:
                    if (N, m) not in existing_csvs:
                        rows.append({
                            "T": T_label, "rec_id": rec_id, "N": N, "m": m,
                            "status": "no_data", "iter": None, "e_norm_final": None,
                        })

        n_expected = len(rec_dirs) * len(NS) * len(MS)
        print(f"  {T_label}: {n_csvs} CSVs processed, {len(rec_dirs)} recs, {n_expected} total combos",
              flush=True)

    return pd.DataFrame(rows)


def build_matrix(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form DataFrame into T × (rec_id, N, m) MultiIndex columns."""
    # Ensure categorical dtype
    df_long["status"] = df_long["status"].astype(STATUS_DTYPE)

    # Pivot: rows = T, columns = (rec_id, N, m)
    matrix = df_long.pivot_table(
        index="T",
        columns=["rec_id", "N", "m"],
        values="status",
        aggfunc="first",  # there should be exactly one row per combo
        observed=False,
    )

    # Sort the column MultiIndex
    matrix = matrix.sort_index(axis=1)
    return matrix


def build_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts of each status per T value."""
    summary = (
        df_long.groupby(["T", "status"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all categories present
    for cat in STATUS_DTYPE.categories:
        if cat not in summary.columns:
            summary[cat] = 0
    summary = summary[STATUS_DTYPE.categories.tolist()]
    return summary


def build_rerun_list(df_long: pd.DataFrame) -> pd.DataFrame:
    """Return all non-converged cases as a flat DataFrame."""
    rerun = df_long[df_long["status"] != "converged"].copy()
    rerun = rerun[["T", "rec_id", "N", "m", "status"]]
    rerun = rerun.sort_values(["T", "rec_id", "N", "m"]).reset_index(drop=True)
    return rerun


def build_rec_overview(df_long: pd.DataFrame) -> pd.DataFrame:
    """Build a per-recurrence overview CSV.

    One row per (T, rec_id).  Aggregates across all 49 (N,m) combos for
    that recurrence and extracts:

      converged          – "yes" if ANY combo converged, else "no"
      num_converged      – how many combos converged (out of 49)
      num_diverged       – how many combos diverged
      num_hit_maxiter    – how many combos hit maxiter
      num_incomplete     – how many combos stopped early
      num_no_data        – how many combos have no CSV
      best_N / best_m    – (N,m) of the converged combo with FEWEST iterations
      best_iter          – iteration count of that fastest combo
      min_N_converged    – smallest N (in {5,10,…,320}) that achieved convergence
      min_m_converged    – smallest m that achieved convergence
      median_iter        – median iterations among ALL converged combos
      converged_T        – placeholder column (NaN); user will add this later
    """
    # Only look at rows that actually exist (skip no_data for stats)
    existing = df_long[df_long["status"] != "no_data"]
    converged = existing[existing["status"] == "converged"]

    # --- Per-recurrence aggregation ---
    group_keys = ["T", "rec_id"]

    # Status counts
    counts = (
        existing.groupby(group_keys)["status"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for col in STATUS_DTYPE.categories:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["converged", "diverged", "hit_maxiter", "incomplete"]]
    counts["num_no_data"] = 49 - counts.sum(axis=1)  # missing CSVs
    counts = counts.rename(columns=lambda c: f"num_{c}" if c != "num_no_data" else c)

    # Best by fewest iterations (among converged)
    best_iter = (
        converged.dropna(subset=["iter"])
        .sort_values("iter")
        .groupby(group_keys)
        .first()[["N", "m", "iter"]]
        .rename(columns={"N": "best_N", "m": "best_m", "iter": "best_iter"})
    )

    # Median iterations (robustness indicator)
    median_iter = (
        converged.dropna(subset=["iter"])
        .groupby(group_keys)["iter"]
        .median()
        .rename("median_iter")
    )

    # Minimum N and m that converged
    min_n = (
        converged.groupby(group_keys)["N"]
        .min()
        .rename("min_N_converged")
    )
    min_m = (
        converged.groupby(group_keys)["m"]
        .min()
        .rename("min_m_converged")
    )

    # --- Assemble ---
    overview = counts.copy()
    overview["converged"] = (overview["num_converged"] > 0).map({True: "yes", False: "no"})

    for src in [best_iter, median_iter, min_n, min_m]:
        overview = overview.join(src, how="left")

    # Placeholder for user to fill later
    overview["converged_T"] = np.nan

    # Sort columns for readability
    col_order = [
        "converged",
        "num_converged", "num_diverged", "num_hit_maxiter", "num_incomplete", "num_no_data",
        "best_N", "best_m", "best_iter",
        "min_N_converged", "min_m_converged",
        "median_iter",
        "converged_T",
    ]
    overview = overview[col_order]

    # Clean index to have rec_id as a proper column
    overview = overview.reset_index()
    overview = overview.sort_values(["T", "rec_id"]).reset_index(drop=True)

    # Reorder: T, rec_id first
    overview = overview[["T", "rec_id"] + col_order]

    return overview


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyse L-BFGS HPC results and classify every (T, rec_id, N, m) case."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs",
        help="Path to the outputs/ directory (default: ../outputs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for output CSV files (default: analysis/)",
    )
    args = parser.parse_args()

    results_root = args.results
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.is_dir():
        print(f"ERROR: Results directory not found: {results_root}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Processing: {results_root}")
    print(f"{'='*60}")

    df_long = collect_results(results_root)
    matrix = build_matrix(df_long)

    # Save status matrix (MultiIndex CSV — round-trips with
    # pd.read_csv(..., header=[0,1,2], index_col=0))
    matrix_path = out_dir / "status_matrix.csv"
    matrix.to_csv(matrix_path)
    print(f"  → saved {matrix_path}  (shape: {matrix.shape})")

    # Summary
    summary = build_summary(df_long)
    summary_path = out_dir / "status_summary.csv"
    summary.to_csv(summary_path)
    print(f"  → saved {summary_path}")
    print(summary.to_string())

    # Rerun list
    rerun = build_rerun_list(df_long)
    rerun_path = out_dir / "cases_to_rerun.csv"
    rerun.to_csv(rerun_path, index=False)
    print(f"\n  → saved {rerun_path}  ({len(rerun)} cases to re-run)")

    if len(rerun) > 0:
        print("\n  Breakdown of non-converged cases:")
        print(rerun.groupby(["T", "status"]).size().to_string())

    # Per-recurrence overview
    overview = build_rec_overview(df_long)
    overview_path = out_dir / "rec_overview.csv"
    overview.to_csv(overview_path, index=False)
    num_recs = len(overview)
    num_conv = (overview["converged"] == "yes").sum()
    print(f"\n  → saved {overview_path}  ({num_recs} recurrences, {num_conv} with ≥1 converged combo)")

    print("\nDone.")


if __name__ == "__main__":
    main()
