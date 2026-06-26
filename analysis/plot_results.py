#!/usr/bin/env python3
"""
plot_results.py
===============
Generates plots from L-BFGS periodic-orbit search results on the Lorenz system.

Reads per-iteration CSV files (outputs/T{XX}/data/rec{XXX}/iteration/N{XX}_m{XX}.csv)
to determine convergence status, actual converged period T, and iteration counts.
Loads full CSV data on demand for convergence-curve plots (fig1).

Plots are registered in the PLOT_FUNCTIONS dict — add new plot functions there
and they will be automatically available from the command line.

Usage:
  python plot_results.py                          # generate ALL plots
  python plot_results.py --plots fig1 fig6 maxiter  # generate only selected
  python plot_results.py --list                    # list available plots
  python plot_results.py --dry-run                 # print what would be done

Output directory:  analysis/plots/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, Normalize

# ---------------------------------------------------------------------------
# Constants (keep in sync with Julia scripts and PROJECT_CONTEXT.md)
# ---------------------------------------------------------------------------
NS: List[int] = [5, 10, 20, 40, 80, 160, 320]
MS: List[int] = [5, 10, 20, 40, 80, 160, 320]
CONVERGENCE_TOL: float = 1e-8
MAXITER: int = 1_000_000

# Paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR: Path = ROOT_DIR / "outputs"           # per-iteration CSV files
_PLOTS_DIR: List[Path] = [ROOT_DIR / "analysis" / "plots"]  # mutable so main() can override


def plots_dir() -> Path:
    return _PLOTS_DIR[0]

# Matplotlib style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

# ---------------------------------------------------------------------------
# CSV comment-marker patterns for status classification
# ---------------------------------------------------------------------------
MARKER_CONVERGED = "# converged"
MARKER_DID_NOT_CONVERGE = "# did_not_converge"
MARKER_CRASHED = "# crashed"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class ComboResult:
    """Stores result for one (N, m) combination within a recurrence."""
    __slots__ = ("status", "T_actual", "iterations")
    def __init__(self, status: str, T_actual: Optional[float] = None,
                 iterations: int = 0):
        self.status = status        # 'converged' | 'hit_maxiter' | 'incomplete' | 'diverged' | 'no_data'
        self.T_actual = T_actual    # actual converged period (None if not converged)
        self.iterations = iterations


class RecurrenceData:
    """Stores all (N,m) results for one recurrence."""
    __slots__ = ("T_target", "rec_id", "combos", "representative_T", "best_combo")
    def __init__(self, T_target: float, rec_id: int):
        self.T_target = T_target
        self.rec_id = rec_id
        self.combos: Dict[Tuple[int, int], ComboResult] = {}
        self.representative_T: Optional[float] = None  # T of fastest converged combo
        self.best_combo: Optional[Tuple[int, int]] = None  # (N,m) with fewest iterations among converged

    @property
    def key(self) -> str:
        return f"T{self.T_target:g}_rec{self.rec_id:03d}"

    @property
    def num_converged(self) -> int:
        return sum(1 for c in self.combos.values() if c.status == "converged")

    @property
    def has_any_converged(self) -> bool:
        return self.num_converged > 0

    def get_converged_combos(self) -> List[Tuple[int, int]]:
        """Return list of (N,m) that converged, sorted by iterations."""
        conv = [(nm, c) for nm, c in self.combos.items() if c.status == "converged"]
        conv.sort(key=lambda x: x[1].iterations)
        return [nm for nm, _ in conv]

    def compute_representative_T(self) -> None:
        """Set representative_T to the T_actual of the fastest converged combo."""
        conv = [(c.T_actual, c.iterations, nm)
                for nm, c in self.combos.items()
                if c.status == "converged" and c.T_actual is not None]
        if conv:
            conv.sort(key=lambda x: x[1])  # sort by iterations (fewest first)
            self.representative_T = conv[0][0]
            self.best_combo = conv[0][2]
        else:
            self.representative_T = None
            self.best_combo = None

    def max_iterations_converged(self) -> int:
        """Return iterations of the slowest converged combo, or 0 if none."""
        iters = [c.iterations for c in self.combos.values() if c.status == "converged"]
        return max(iters) if iters else 0

    def best_combos(self, k: int = 3) -> List[Tuple[Tuple[int, int], int]]:
        """Return the k (N,m) combos with fewest iterations among converged."""
        conv = [(nm, c.iterations)
                for nm, c in self.combos.items()
                if c.status == "converged"]
        conv.sort(key=lambda x: x[1])
        return conv[:k]


# ---------------------------------------------------------------------------
# Data loading — parse per-iteration CSVs
# ---------------------------------------------------------------------------
def _read_csv_last_data_row(csv_path: Path) -> Optional[pd.DataFrame]:
    """Read the last data row (before any # comment line) of a CSV.
    Returns a 1-row DataFrame, or None if file missing/corrupt."""
    if not csv_path.is_file():
        return None
    try:
        # Read the file in reverse to find the last data line efficiently
        with open(csv_path, "rb") as fh:
            # Seek to end, read backwards in chunks to find last non-comment line
            fh.seek(0, 2)
            file_size = fh.tell()
            chunk_size = 4096
            tail_lines: List[str] = []
            pos = file_size
            while pos > 0 and len(tail_lines) < 5:
                read_size = min(chunk_size, pos)
                pos -= read_size
                fh.seek(pos)
                chunk = fh.read(read_size).decode("utf-8", errors="replace")
                tail_lines = chunk.splitlines() + tail_lines
                if pos == 0:
                    break
        # Find the last non-comment line
        last_data_line = None
        for line in reversed(tail_lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                last_data_line = stripped
                break
        if last_data_line is None:
            return None
        # Parse as CSV row
        import io
        # Prepend the header line from the file
        with open(csv_path, "r", errors="replace") as fh:
            header = fh.readline().strip()
        df = pd.read_csv(io.StringIO(f"{header}\n{last_data_line}"))
        return df
    except Exception:
        return None


def _parse_csv_status(csv_path: Path) -> Tuple[str, int, Optional[float]]:
    """Classify a single (N,m) combo by reading its iteration CSV.

    Returns (status, iterations, T_actual) where status is one of:
      'converged', 'hit_maxiter', 'incomplete', 'diverged', 'no_data'

    Classification logic (from ANALYSIS_CONTEXT.md):
      1. File missing → 'no_data'
      2. NaN in first few e_norm rows → 'diverged'
      3. Read last line comment marker:
           '# converged' + e_norm ≤ 1e-8 → 'converged'
           '# did_not_converge' + iter ≥ MAXITER → 'hit_maxiter'
           '# did_not_converge' + iter < MAXITER → 'incomplete'
           '# crashed' → 'incomplete'
           No marker → 'incomplete' (killed mid-run)
    """
    if not csv_path.is_file():
        return ("no_data", 0, None)

    try:
        with open(csv_path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ("no_data", 0, None)

    if len(lines) < 2:
        return ("no_data", 0, None)

    # --- NaN scan: check first few data rows (skip header) ---
    header = lines[0].strip()
    cols = header.split(",")
    try:
        e_norm_idx = cols.index("e_norm")
    except ValueError:
        return ("incomplete", 0, None)

    for line in lines[1:min(6, len(lines))]:
        stripped = line.strip()
        if stripped.startswith("#"):
            break
        if not stripped:
            continue
        parts = stripped.split(",")
        if len(parts) <= e_norm_idx:
            continue
        try:
            val = float(parts[e_norm_idx])
        except ValueError:
            continue
        if np.isnan(val):
            return ("diverged", 0, None)

    # --- Find last comment marker and last data row ---
    last_comment = None
    last_data_line = None
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            last_comment = stripped
            continue
        if stripped and last_data_line is None:
            last_data_line = stripped
            break

    if last_data_line is None:
        return ("no_data", 0, None)

    # Parse last data row
    parts = last_data_line.split(",")
    if len(parts) < len(cols):
        return ("incomplete", 0, None)

    try:
        iter_idx = cols.index("iter")
        t_idx = cols.index("T_curr")
        iterations = int(float(parts[iter_idx]))
        e_norm = float(parts[e_norm_idx])
        T_actual = float(parts[t_idx])
    except (ValueError, IndexError):
        return ("incomplete", 0, None)

    # --- Classify based on comment marker ---
    if last_comment is None:
        # Killed mid-run before writing a comment
        return ("incomplete", iterations, T_actual)

    if last_comment == MARKER_CONVERGED:
        if e_norm <= CONVERGENCE_TOL:
            return ("converged", iterations, T_actual)
        else:
            return ("incomplete", iterations, T_actual)

    if last_comment == MARKER_DID_NOT_CONVERGE:
        if iterations >= MAXITER:
            return ("hit_maxiter", iterations, T_actual)
        else:
            return ("incomplete", iterations, T_actual)

    if last_comment == MARKER_CRASHED:
        return ("incomplete", iterations, T_actual)

    # Unknown comment marker
    return ("incomplete", iterations, T_actual)


def _discover_recurrences(data_dir: Path) -> List[Tuple[float, int]]:
    """Scan the outputs/ directory and return a sorted list of (T_target, rec_id)
    pairs for all recurrences that have at least one CSV file."""
    recs_set: set = set()
    # Pattern: outputs/T{XX}/data/rec{XXX}/iteration/N{XX}_m{XX}.csv
    # Use resolve() for consistent absolute path parts indexing.
    data_dir_resolved = data_dir.resolve()
    for csv_path in sorted(data_dir.glob("T*/data/rec*/iteration/N*.csv")):
        parts = csv_path.resolve().parts
        # parts: .../outputs/T05/data/rec001/iteration/N05_m05.csv
        # Index from end: [-1]=filename, [-2]=iteration, [-3]=recNNN, [-4]=data, [-5]=TNN
        try:
            t_dir = parts[-5]   # e.g. "T05"
            rec_dir = parts[-3] # e.g. "rec001"
            T_target = float(t_dir[1:])  # strip leading 'T'
            rec_id = int(rec_dir[3:])    # strip leading 'rec'
            recs_set.add((T_target, rec_id))
        except (ValueError, IndexError):
            continue
    return sorted(recs_set)


def load_all_data(data_dir: Optional[Path] = None) -> List[RecurrenceData]:
    """Scan per-iteration CSVs to classify every (N,m) combo, build
    RecurrenceData objects, and return only recurrences with ≥1 converged combo,
    sorted by representative T."""
    if data_dir is None:
        data_dir = DATA_DIR

    print(f"Scanning CSVs in {data_dir} ...")
    all_recs = _discover_recurrences(data_dir)
    print(f"  Found {len(all_recs)} recurrences with CSV files.")

    recs: List[RecurrenceData] = []
    for T_target, rec_id in all_recs:
        rec = RecurrenceData(T_target, rec_id)
        T_label = f"T{int(T_target):02d}"
        rec_dir = data_dir / T_label / "data" / f"rec{rec_id:03d}" / "iteration"

        for N in NS:
            for m in MS:
                csv_path = rec_dir / f"N{N:02d}_m{m:02d}.csv"
                status, iterations, T_actual = _parse_csv_status(csv_path)
                rec.combos[(N, m)] = ComboResult(status, T_actual, iterations)

        rec.compute_representative_T()
        if rec.has_any_converged:
            recs.append(rec)

    # Sort by representative_T, then rec_id
    recs.sort(key=lambda r: (r.representative_T if r.representative_T else float("inf"),
                              r.rec_id))

    print(f"  Classified all combos → {len(recs)} recurrences with ≥1 converged.")
    return recs


def load_csv_data(rec: RecurrenceData, N: int, m: int) -> Optional[pd.DataFrame]:
    """Load a single iteration CSV and return as DataFrame with columns
    iter, e_norm, grad_norm, lambda, T_curr.  Returns None if file missing or unreadable."""
    T_label = f"T{int(rec.T_target):02d}"
    csv_path = DATA_DIR / T_label / "data" / f"rec{rec.rec_id:03d}" / "iteration" / f"N{N:02d}_m{m:02d}.csv"
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Colour / marker setup
# ---------------------------------------------------------------------------
COLORS_M = {m: plt.cm.viridis(i / (len(MS) - 1)) for i, m in enumerate(MS)}
COLORS_N = {n: plt.cm.plasma(i / (len(NS) - 1)) for i, n in enumerate(NS)}
MARKERS_M = {m: s for m, s in zip(MS, ["o", "s", "D", "^", "v", "p", "*"])}
MARKERS_N = {n: s for n, s in zip(NS, ["o", "s", "D", "^", "v", "p", "*"])}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_log(v: np.ndarray) -> np.ndarray:
    """Floor values at 1e-30 for safe log-scale plotting."""
    return np.maximum(v, 1e-30)


def savefigs(fig: plt.Figure, name: str, subdir: str = ""):
    """Save figure as both PDF and PNG."""
    out_dir = plots_dir() / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fpath = out_dir / f"{name}.{ext}"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / name}.{{pdf,png}}")


def status_to_display(rec: RecurrenceData, N: int, m: int) -> str:
    """Return a display character for a combo's status."""
    combo = rec.combos.get((N, m))
    if combo is None:
        return "·"       # no_data (never reached)
    if combo.status == "converged":
        return "✓"
    elif combo.status == "hit_maxiter":
        return "✗ₘ"     # hit maxiter
    elif combo.status == "diverged":
        return "✗d"     # diverged
    elif combo.status == "incomplete":
        return "✗"      # incomplete / crashed
    else:
        return "·"      # no_data


def Ns_with_any_converged(rec: RecurrenceData) -> List[int]:
    """Return list of N values that have at least one converged m for this rec."""
    return sorted({N for (N, m), c in rec.combos.items()
                   if c.status == "converged"})


# ===========================================================================
#  PLOT FUNCTIONS  (registered in PLOT_FUNCTIONS dict at bottom)
# ===========================================================================

# ---------------------------------------------------------------------------
# fig1  —  Convergence curves: per recurrence, subplots per N, varying m
# ---------------------------------------------------------------------------
def plot_fig1_convergence_curves(recs: List[RecurrenceData]):
    """For each recurrence, a multi-panel plot: one subplot per N (where ≥1 m
    converged), each showing ‖F‖ vs iteration for all m values.  Only converged
    (N,m) combos have curves plotted."""
    print("\n" + "=" * 60)
    print("  Fig 1: Convergence curves per recurrence (fixed N, varying m)")
    print("=" * 60)

    for i_rec, rec in enumerate(recs):
        ns = Ns_with_any_converged(rec)
        if not ns:
            continue

        ncols = min(4, len(ns))
        nrows = int(np.ceil(len(ns) / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.5 * ncols, 2.8 * nrows),
                                 sharex=True, sharey=True,
                                 squeeze=False)
        axes_flat = axes.flatten()

        label = (f"Recurrence  T≈{rec.representative_T:.2f}  "
                 f"(target T={rec.T_target:g}, rec_id={rec.rec_id})  "
                 f"[{rec.num_converged}/49 converged]")
        fig.suptitle(label, fontsize=11, fontweight="bold")

        for j, N in enumerate(ns):
            ax = axes_flat[j]
            for m in MS:
                combo = rec.combos.get((N, m))
                if combo is None or combo.status != "converged":
                    continue
                df = load_csv_data(rec, N, m)
                if df is None or len(df) < 2:
                    continue
                iters = df["iter"].values
                e_norm = df["e_norm"].values
                cv_mark = "✓" if combo.status == "converged" else "✗"
                lbl = f"m={m} [{len(iters)-1} it]"
                ax.loglog(iters, safe_log(e_norm), "-",
                          color=COLORS_M[m], linewidth=0.8, alpha=0.85,
                          label=lbl)
            ax.set_title(f"N={N}", fontsize=9)
            ax.legend(loc="upper right", ncol=2, fontsize=5)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("‖F(z)‖")

        # Hide unused subplots
        for j in range(len(ns), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        savefigs(fig, f"fig01_T{rec.T_target:g}_rec{rec.rec_id:03d}", subdir="fig01_convergence_curves")
        plt.close(fig)

        if (i_rec + 1) % 20 == 0:
            print(f"  ... {i_rec + 1}/{len(recs)} recurrences done")

    print(f"  Finished: {len(recs)} recurrence plots saved to {plots_dir() / 'fig01_convergence_curves'}")


# ---------------------------------------------------------------------------
# fig6  —  Iterations heatmap per recurrence
# ---------------------------------------------------------------------------
def plot_fig6_iterations_heatmap(recs: List[RecurrenceData]):
    """For each recurrence, a 7×7 heatmap of (N, m) showing iterations to
    convergence.  Non-converged cells are marked with a symbol."""
    print("\n" + "=" * 60)
    print("  Fig 6: Iterations heatmap per recurrence")
    print("=" * 60)

    for i_rec, rec in enumerate(recs):
        # Build data matrix
        mat = np.full((len(NS), len(MS)), np.nan)
        annot = np.empty((len(NS), len(MS)), dtype=object)
        for i, N in enumerate(NS):
            for j, m in enumerate(MS):
                combo = rec.combos.get((N, m))
                if combo is not None and combo.status == "converged":
                    mat[i, j] = combo.iterations
                    annot[i, j] = str(combo.iterations)
                else:
                    mat[i, j] = np.nan
                    annot[i, j] = status_to_display(rec, N, m)

        fig, ax = plt.subplots(figsize=(8, 6))
        label = (f"Recurrence  T≈{rec.representative_T:.2f}  "
                 f"(target T={rec.T_target:g}, rec_id={rec.rec_id})  "
                 f"[{rec.num_converged}/49 converged]")
        ax.set_title(label, fontsize=11, fontweight="bold")

        # Plot heatmap (only non-NaN cells get color)
        masked = np.ma.masked_invalid(mat)
        im = ax.pcolormesh(
            np.arange(len(MS) + 1) - 0.5,
            np.arange(len(NS) + 1) - 0.5,
            masked,
            cmap="plasma", shading="flat",
            norm=LogNorm(vmin=max(1, np.nanmin(mat)) if np.any(np.isfinite(mat)) else 1,
                         vmax=max(1, np.nanmax(mat)) if np.any(np.isfinite(mat)) else 1)
        )

        # Annotate cells
        for i, N in enumerate(NS):
            for j, m in enumerate(MS):
                combo = rec.combos.get((N, m))
                if combo is not None and combo.status == "converged":
                    ax.text(j, i, str(combo.iterations),
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
                else:
                    ax.text(j, i, annot[i, j],
                            ha="center", va="center", fontsize=13,
                            color="black", fontweight="bold")

        ax.set_xticks(range(len(MS)))
        ax.set_xticklabels([str(m) for m in MS])
        ax.set_yticks(range(len(NS)))
        ax.set_yticklabels([str(n) for n in NS])
        ax.set_xlabel("L-BFGS memory m")
        ax.set_ylabel("Segments N")
        plt.colorbar(im, ax=ax, label="Iterations (log scale)")

        fig.tight_layout()
        savefigs(fig, f"fig06_T{rec.T_target:g}_rec{rec.rec_id:03d}", subdir="fig06_iterations_heatmap")
        plt.close(fig)

        if (i_rec + 1) % 50 == 0:
            print(f"  ... {i_rec + 1}/{len(recs)} recurrences done")

    print(f"  Finished: {len(recs)} recurrence plots saved to {plots_dir() / 'fig06_iterations_heatmap'}")


# ---------------------------------------------------------------------------
# fig12a  —  Minimum N to converge vs actual T, one line per m
# ---------------------------------------------------------------------------
def plot_fig12a_min_N_vs_T(recs: List[RecurrenceData]):
    """For each m value, find the minimum N that converged for each recurrence.
    Plot as line+marker vs the actual converged T.  Points where no N converged
    for that m are omitted."""
    print("\n" + "=" * 60)
    print("  Fig 12a: Minimum N to converge vs actual T")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Minimum Shooting Segments N for Convergence  (‖F‖ < 1e-8)",
                 fontsize=13, fontweight="bold")

    for m in MS:
        pts: List[Tuple[float, int]] = []
        for rec in recs:
            converged_Ns = [N for N in NS
                            if rec.combos.get((N, m)) is not None
                            and rec.combos[(N, m)].status == "converged"]
            if converged_Ns and rec.representative_T is not None:
                pts.append((rec.representative_T, min(converged_Ns)))
        if pts:
            pts.sort()
            T_vals = [p[0] for p in pts]
            N_vals = [p[1] for p in pts]
            marker = MARKERS_M[m]
            ax.plot(T_vals, N_vals, f"-{marker}", color=COLORS_M[m],
                    markersize=6, linewidth=1.2, label=f"m = {m}")

    ax.set_xlabel("Actual converged orbit period T")
    ax.set_ylabel("Minimum N to converge")
    ax.set_yticks(NS)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, max(NS) + 1)

    fig.tight_layout()
    savefigs(fig, "fig12a_min_N_vs_T")
    plt.close(fig)
    print("  Saved.")


# ---------------------------------------------------------------------------
# fig12b  —  Minimum m to converge vs actual T, one line per N
# ---------------------------------------------------------------------------
def plot_fig12b_min_m_vs_T(recs: List[RecurrenceData]):
    """For each N value, find the minimum m that converged for each recurrence.
    Plot as line+marker vs the actual converged T."""
    print("\n" + "=" * 60)
    print("  Fig 12b: Minimum m to converge vs actual T")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Minimum L-BFGS Memory m for Convergence  (‖F‖ < 1e-8)",
                 fontsize=13, fontweight="bold")

    for N in NS:
        pts: List[Tuple[float, int]] = []
        for rec in recs:
            converged_ms = [m for m in MS
                            if rec.combos.get((N, m)) is not None
                            and rec.combos[(N, m)].status == "converged"]
            if converged_ms and rec.representative_T is not None:
                pts.append((rec.representative_T, min(converged_ms)))
        if pts:
            pts.sort()
            T_vals = [p[0] for p in pts]
            m_vals = [p[1] for p in pts]
            marker = MARKERS_N[N]
            ax.plot(T_vals, m_vals, f"-{marker}", color=COLORS_N[N],
                    markersize=6, linewidth=1.2, label=f"N = {N}")

    ax.set_xlabel("Actual converged orbit period T")
    ax.set_ylabel("Minimum m to converge")
    ax.set_yticks(MS)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, max(MS) + 1)

    fig.tight_layout()
    savefigs(fig, "fig12b_min_m_vs_T")
    plt.close(fig)
    print("  Saved.")


# ---------------------------------------------------------------------------
# maxiter  —  Max iterations per recurrence vs T (column/bar chart)
# ---------------------------------------------------------------------------
def plot_maxiter_vs_T(recs: List[RecurrenceData]):
    """Column chart: x-axis = recurrence (ordered by actual T), y-axis =
    iterations of the slowest converged (N,m) combo.  Zero if none converged
    (should not happen since we filter)."""
    print("\n" + "=" * 60)
    print("  Max iterations per recurrence vs T")
    print("=" * 60)

    T_vals = [rec.representative_T for rec in recs]
    max_iters = [rec.max_iterations_converged() for rec in recs]
    labels = [f"T={rec.T_target:g}\n#{rec.rec_id}" for rec in recs]

    fig, ax = plt.subplots(figsize=(max(18, len(recs) * 0.15), 7))
    ax.set_title("Slowest Converged (N,m) per Recurrence", fontsize=14, fontweight="bold")

    colors = [plt.cm.viridis(t / max(T_vals)) if max(T_vals) > 0 else "steelblue"
              for t in T_vals]
    bars = ax.bar(range(len(recs)), max_iters, color=colors, edgecolor="none", width=0.8)

    # Color bar for T
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=min(T_vals), vmax=max(T_vals)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=30, pad=0.02)
    cbar.set_label("Actual converged period T")

    ax.set_xticks(range(len(recs)))
    # Show only ~30 tick labels to avoid clutter
    step = max(1, len(recs) // 30)
    ax.set_xticklabels([labels[i] if i % step == 0 else "" for i in range(len(recs))],
                       rotation=90, fontsize=5)
    ax.set_ylabel("Max iterations (slowest converged combo)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    savefigs(fig, "maxiter_vs_T")
    plt.close(fig)
    print("  Saved.")


# ---------------------------------------------------------------------------
# best3  —  Max iterations with 3 best combos annotated
# ---------------------------------------------------------------------------
def plot_best_combos_vs_T(recs: List[RecurrenceData]):
    """Column chart like maxiter, but also plots the iterations of the 3
    fastest (N,m) combos for each recurrence as overlaid scatter markers,
    and adds a legend with the (N,m) pairs used across all recurrences."""
    print("\n" + "=" * 60)
    print("  Max iterations + 3 best combos per recurrence vs T")
    print("=" * 60)

    T_vals = [rec.representative_T for rec in recs]
    max_iters = [rec.max_iterations_converged() for rec in recs]
    labels = [f"T={rec.T_target:g}\n#{rec.rec_id}" for rec in recs]

    fig, ax = plt.subplots(figsize=(max(18, len(recs) * 0.15), 8))
    ax.set_title("Convergence Iterations per Recurrence — Best (N,m) Combos Highlighted",
                 fontsize=13, fontweight="bold")

    # Bars: max iterations (ghost outline)
    ax.bar(range(len(recs)), max_iters, color="lightgray", edgecolor="gray",
           linewidth=0.3, width=0.8, zorder=1)

    # Overlay the 3 best combos as scatter points (rank 1=best, 2, 3)
    best_markers = ["o", "s", "D"]
    best_labels_used: Dict[Tuple[int, int], str] = {}
    rank_colors = ["#d62728", "#2ca02c", "#1f77b4"]  # red, green, blue

    for rank in range(3):
        xs, ys = [], []
        for i, rec in enumerate(recs):
            best = rec.best_combos(3)
            if rank < len(best):
                xs.append(i)
                ys.append(best[rank][1])
                nm = best[rank][0]
                if nm not in best_labels_used:
                    best_labels_used[nm] = f"N={nm[0]}, m={nm[1]}"
        if xs:
            ax.scatter(xs, ys, marker=best_markers[rank], color=rank_colors[rank],
                       s=25, zorder=5, edgecolors="black", linewidths=0.3,
                       label=f"Rank #{rank+1}")

    # Add text annotations showing the (N,m) values for the best combo
    for i, rec in enumerate(recs):
        best = rec.best_combos(3)
        if best:
            nm_str = f"({best[0][0][0]},{best[0][0][1]})"
            ax.annotate(nm_str, (i, best[0][1]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=4, ha="center", rotation=90, alpha=0.7)

    ax.set_yscale("log")
    ax.set_xticks(range(len(recs)))
    step = max(1, len(recs) // 30)
    ax.set_xticklabels([labels[i] if i % step == 0 else "" for i in range(len(recs))],
                       rotation=90, fontsize=5)
    ax.set_ylabel("Iterations (log scale)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)

    # Add a color bar for T
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=min(T_vals), vmax=max(T_vals)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=30, pad=0.02)
    cbar.set_label("Actual converged period T")

    fig.tight_layout()
    savefigs(fig, "best3_combos_vs_T")
    plt.close(fig)
    print("  Saved.")


# ---------------------------------------------------------------------------
# convcount  —  Number of converged combos per recurrence
# ---------------------------------------------------------------------------
def plot_convergence_count_vs_T(recs: List[RecurrenceData]):
    """Column chart: x-axis = recurrence (ordered by actual T), y-axis =
    number of converged (N,m) combos (out of 49)."""
    print("\n" + "=" * 60)
    print("  Convergence count per recurrence vs T")
    print("=" * 60)

    T_vals = [rec.representative_T for rec in recs]
    conv_counts = [rec.num_converged for rec in recs]
    labels = [f"T={rec.T_target:g}\n#{rec.rec_id}" for rec in recs]

    fig, ax = plt.subplots(figsize=(max(18, len(recs) * 0.15), 7))
    ax.set_title("Number of Converged (N,m) Combos per Recurrence (out of 49)",
                 fontsize=13, fontweight="bold")

    colors = [plt.cm.viridis(t / max(T_vals)) if max(T_vals) > 0 else "steelblue"
              for t in T_vals]
    ax.bar(range(len(recs)), conv_counts, color=colors, edgecolor="none", width=0.8)

    ax.axhline(y=49, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Max possible (49)")
    ax.legend(fontsize=9)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=min(T_vals), vmax=max(T_vals)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=30, pad=0.02)
    cbar.set_label("Actual converged period T")

    ax.set_xticks(range(len(recs)))
    step = max(1, len(recs) // 30)
    ax.set_xticklabels([labels[i] if i % step == 0 else "" for i in range(len(recs))],
                       rotation=90, fontsize=5)
    ax.set_ylabel("Number of converged combos")
    ax.set_ylim(0, 53)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    savefigs(fig, "convcount_vs_T")
    plt.close(fig)
    print("  Saved.")


# ===========================================================================
#  PLOT REGISTRY
#  Add new plot functions here.  Key = CLI name, value = (function, description)
# ===========================================================================
PLOT_FUNCTIONS: Dict[str, Tuple[Callable[[List[RecurrenceData]], None], str]] = {
    "fig1":    (plot_fig1_convergence_curves,
                "Convergence curves per recurrence (‖F‖ vs iter, per N, varying m)"),
    "fig6":    (plot_fig6_iterations_heatmap,
                "Iterations-to-convergence heatmap per recurrence (N×m grid)"),
    "minN":    (plot_fig12a_min_N_vs_T,
                "Minimum N to converge vs actual T, one line per m"),
    "minM":    (plot_fig12b_min_m_vs_T,
                "Minimum m to converge vs actual T, one line per N"),
    "maxiter": (plot_maxiter_vs_T,
                "Max iterations (slowest converged combo) per recurrence vs T"),
    "best3":   (plot_best_combos_vs_T,
                "Max iterations per recurrence with 3 best (N,m) combos overlaid"),
    "convcount": (plot_convergence_count_vs_T,
                  "Number of converged (N,m) combos per recurrence"),
}


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from L-BFGS Lorenz-system results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available plots:\n" + "\n".join(
            f"  {k:12s} — {v[1]}" for k, v in PLOT_FUNCTIONS.items()
        ),
    )
    parser.add_argument(
        "--plots", nargs="*", metavar="NAME",
        help="Which plots to generate (default: all).  Use --list to see options.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available plot names and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without actually plotting.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_PLOTS_DIR[0],
        help=f"Output directory (default: {_PLOTS_DIR[0]})",
    )
    args = parser.parse_args()

    # Override output directory
    _PLOTS_DIR[0] = Path(args.output_dir)

    if args.list:
        print("Available plots:")
        for name, (func, desc) in PLOT_FUNCTIONS.items():
            print(f"  {name:12s} — {desc}")
        return

    # Determine which plots to run
    if args.plots:
        selected = {}
        for p in args.plots:
            if p in PLOT_FUNCTIONS:
                selected[p] = PLOT_FUNCTIONS[p]
            else:
                print(f"Warning: unknown plot '{p}'.  Use --list to see options.")
        if not selected:
            print("No valid plots selected.  Use --list to see options.")
            return
    else:
        selected = PLOT_FUNCTIONS

    print(f"Will generate {len(selected)} plot(s): {', '.join(selected.keys())}")
    if args.dry_run:
        print("Dry run — exiting without plotting.")
        return

    # Load data
    recs = load_all_data()
    if not recs:
        print("ERROR: No recurrences with converged combos found.  Exiting.")
        return

    print(f"\nTotal recurrences with ≥1 converged combo: {len(recs)}")
    print(f"T range: {min(r.representative_T for r in recs):.2f} – "
          f"{max(r.representative_T for r in recs):.2f}")
    print(f"Output directory: {plots_dir()}\n")

    # Run selected plot functions
    for name, (func, desc) in selected.items():
        print(f"\n{'—' * 50}")
        print(f"Plot: {name} — {desc}")
        try:
            func(recs)
        except Exception as e:
            print(f"  ERROR in '{name}': {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing with next plot ...")

    print("\n" + "=" * 60)
    print(f"  All plots saved to {plots_dir()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
