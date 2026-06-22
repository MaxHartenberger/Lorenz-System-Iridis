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

Outputs (saved next to this script):
    status_matrix_{new,old}.csv   –  full T × (rec_id, N, m) matrix (MultiIndex CSV)
    status_summary.csv                –  counts of each status per T & folder
    cases_to_rerun.csv                –  all non-converged cases (flat list)
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
def _read_tail_line(filepath: Path, chunk_size: int = 4096) -> str:
    """Read the last line of a file efficiently without loading the whole file."""
    with open(filepath, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        file_size = fh.tell()
        if file_size == 0:
            return ""
        # Read the last chunk_size bytes (or whole file if smaller)
        seek_to = max(0, file_size - chunk_size)
        fh.seek(seek_to)
        chunk = fh.read(min(chunk_size, file_size))
        # Decode and split lines; return the last non-empty line
        lines = chunk.decode("utf-8", errors="replace").splitlines()
        # The last line might be empty if the file ends with \n
        for line in reversed(lines):
            if line.strip():
                return line
        return ""


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

    Only reads the first ~5 lines (NaN scan) and seeks to the last line
    (final state), so it is fast even with 20k+ large files.
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

        # --- Examine the last data row (seek-based, fast) ---
        last_line = _read_tail_line(filepath)
        if not last_line:
            return "incomplete"

        parts = last_line.strip().split(",")
        if len(parts) < 4:
            return "incomplete"

        try:
            e_norm = float(parts[1])
            grad_norm_str = parts[2].strip()
            iteration = int(parts[0])
        except (ValueError, IndexError):
            return "incomplete"

        # Normal completion: Julia appends one extra row with grad_norm = NaN
        # after _search! returns.
        if grad_norm_str == "NaN":
            if e_norm <= CONVERGENCE_TOL:
                return "converged"
            elif iteration >= MAXITER:
                return "hit_maxiter"
            else:
                # Should be rare — optimisation stopped without converging
                # and without hitting maxiter (e.g. time-limit kill).
                return "incomplete"
        else:
            # No NaN-labelled final row → optimisation crashed / was killed.
            return "incomplete"

    except Exception:
        return "incomplete"


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
            for csv_file in sorted(rec_dir.iterdir()):
                if not csv_file.name.endswith(".csv"):
                    continue
                parsed = parse_filename(csv_file.name)
                if parsed is None:
                    continue
                N, m = parsed
                existing_csvs.add((N, m))
                status = classify_csv(csv_file)
                rows.append({"T": T_label, "rec_id": rec_id, "N": N, "m": m, "status": status})
                n_csvs += 1

            # Mark missing (N, m) combos as "no_data"
            for N in NS:
                for m in MS:
                    if (N, m) not in existing_csvs:
                        rows.append({"T": T_label, "rec_id": rec_id, "N": N, "m": m, "status": "no_data"})

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


def build_summary(df_long: pd.DataFrame, folder_name: str) -> pd.DataFrame:
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
    summary["folder"] = folder_name
    return summary


def build_rerun_list(df_long: pd.DataFrame, folder_name: str) -> pd.DataFrame:
    """Return all non-converged cases as a flat DataFrame."""
    rerun = df_long[df_long["status"] != "converged"].copy()
    rerun["folder"] = folder_name
    rerun = rerun[["folder", "T", "rec_id", "N", "m", "status"]]
    rerun = rerun.sort_values(["T", "rec_id", "N", "m"]).reset_index(drop=True)
    return rerun


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyse L-BFGS HPC results and classify every (T, rec_id, N, m) case."
    )
    parser.add_argument(
        "--new-results",
        type=Path,
        default=Path(__file__).resolve().parent / "lorenz-results" / "output",
        help="Path to lorenz-results/output (default: ./lorenz-results/output)",
    )
    parser.add_argument(
        "--old-results",
        type=Path,
        default=Path(__file__).resolve().parent / "lorenz-results-old" / "output",
        help="Path to lorenz-results-old/output (default: ./lorenz-results-old/output)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for output files (default: script directory)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_rerun = []

    for label, root in [("new", args.new_results), ("old", args.old_results)]:
        if not root.is_dir():
            print(f"⚠  Skipping '{label}' — directory not found: {root}")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing: {label}  ({root})")
        print(f"{'='*60}")

        df_long = collect_results(root)
        matrix = build_matrix(df_long)

        # Save matrix (MultiIndex CSV so it round-trips with pd.read_csv(..., header=[0,1,2], index_col=0))
        matrix_path = out_dir / f"status_matrix_{label}.csv"
        matrix.to_csv(matrix_path)
        print(f"  → saved {matrix_path}  (shape: {matrix.shape})")

        # Summary
        summary = build_summary(df_long, label)
        all_summaries.append(summary)

        # Rerun list
        rerun = build_rerun_list(df_long, label)
        all_rerun.append(rerun)
        print(f"  Non-converged cases: {len(rerun)}")

    # ------------------------------------------------------------------
    # Combined outputs
    # ------------------------------------------------------------------
    if all_summaries:
        combined_summary = pd.concat(all_summaries)
        summary_path = out_dir / "status_summary.csv"
        combined_summary.to_csv(summary_path)
        print(f"\n  → saved {summary_path}")
        print(combined_summary.to_string())

    if all_rerun:
        combined_rerun = pd.concat(all_rerun, ignore_index=True)
        rerun_path = out_dir / "cases_to_rerun.csv"
        combined_rerun.to_csv(rerun_path, index=False)
        print(f"\n  → saved {rerun_path}  ({len(combined_rerun)} cases to re-run)")

        # Quick breakdown
        print("\n  Breakdown of non-converged cases:")
        print(combined_rerun.groupby(["folder", "T", "status"]).size().to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
