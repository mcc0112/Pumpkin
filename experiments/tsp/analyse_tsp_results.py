"""
analyse_tsp_results.py  –  Two-propagator comparison for TSP benchmarks
========================================================================

Loads the two most-recent stats_*.csv files from experiments/tsp/results/,
treats the older one as "baseline" and the newer one as "new", then:

  • Prints a per-instance comparison table to the terminal
  • Writes  experiments/tsp/results/comparison_<timestamp>.csv
  • Saves four PNG plots:
      plot_tsp_runtime.png          – wall-clock time per instance
      plot_tsp_search_reduction.png – failures (backtracks) per instance
      plot_tsp_explanation.png      – noGoods per instance
      plot_tsp_speedup.png          – runtime speedup ratio per instance

Usage (from the Pumpkin project root)
--------------------------------------
    python experiments\\tsp\\analyse_tsp_results.py [options]

Options
-------
  --baseline PATH   Explicit path to the baseline CSV
  --new      PATH   Explicit path to the new-propagator CSV
  --out      PATH   Output directory  (default: experiments/tsp/results/)

Column names
------------
Edit METRIC_COLS if your solver emits different %%%mzn-stat keys.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  uv add pandas")

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    print("matplotlib not found – skipping plots (uv add matplotlib)")
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METRIC_COLS: dict[str, str] = {
    "Runtime (s)"           : "wall_time_s",
    "Failures (backtracks)" : "failures",
    "Average LBD"               : "AverageLbd",
}

RUNTIME_COL     = "wall_time_s"
SEARCH_COL      = "failures"
EXPLANATION_COL = "AverageLbd"

LABEL_BASELINE  = "baseline"
LABEL_NEW       = "new"
COLORS          = {LABEL_BASELINE: "#4C72B0", LABEL_NEW: "#DD8452"}

# Instances are identified by this column (written by run_tsp_experiments.py)
INSTANCE_COL    = "instance_name"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def two_latest_csvs(results_dir: Path) -> tuple[Path, Path]:
    csvs = sorted(results_dir.glob("stats_*.csv"))
    if len(csvs) < 2:
        sys.exit(
            f"Need at least 2 stats_*.csv files in {results_dir}, "
            f"found {len(csvs)}.\n"
            "Run both propagators first, or pass --baseline and --new."
        )
    return csvs[-2], csvs[-1]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return df


def load(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _coerce_numerics(df)
    df["propagator"] = label
    print(f"  [{label:>8}]  {len(df)} rows  ←  {csv_path.name}")

    if INSTANCE_COL not in df.columns:
        sys.exit(
            f"Column '{INSTANCE_COL}' not found in {csv_path}.\n"
            "Make sure this CSV was produced by run_tsp_experiments.py."
        )
    return df


# ---------------------------------------------------------------------------
# Aggregation  (TSP: one row per instance, so mean = the single value)
# ---------------------------------------------------------------------------

def metric_per_instance(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Mean of *col* over successful rows, indexed by instance_name.
    For TSP there is exactly one row per instance, so this is just the
    single value — but using mean() makes it robust if you re-run the
    same instance multiple times.
    """
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if col not in ok.columns:
        return pd.Series(dtype=float)
    return ok.groupby(INSTANCE_COL)[col].mean()


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    df_base: pd.DataFrame,
    df_new:  pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for display_name, col in METRIC_COLS.items():
        base_vals = metric_per_instance(df_base, col)
        new_vals  = metric_per_instance(df_new,  col)
        all_inst  = sorted(base_vals.index.union(new_vals.index))

        for inst in all_inst:
            b = base_vals.get(inst, float("nan"))
            n = new_vals.get(inst,  float("nan"))
            abs_diff = n - b
            rel_diff = (abs_diff / b * 100) if (b and b != 0) else float("nan")
            rows.append({
                "metric"                  : display_name,
                INSTANCE_COL              : inst,
                f"{LABEL_BASELINE}_value" : round(float(b), 4),
                f"{LABEL_NEW}_value"      : round(float(n), 4),
                "abs_diff"                : round(float(abs_diff), 4),
                "rel_diff_%"              : round(float(rel_diff), 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_comparison(
    ax: "plt.Axes",
    instances: list[str],
    base_vals: list[float],
    new_vals:  list[float],
    ylabel: str,
    title:  str,
    rotate_labels: bool = True,
) -> None:
    x     = np.arange(len(instances))
    width = 0.35
    ax.bar(x - width / 2, base_vals, width,
           label=LABEL_BASELINE, color=COLORS[LABEL_BASELINE], alpha=0.85, zorder=2)
    ax.bar(x + width / 2, new_vals,  width,
           label=LABEL_NEW,      color=COLORS[LABEL_NEW],      alpha=0.85, zorder=2)
    ax.set_xticks(x)
    rot = 35 if rotate_labels else 0
    ax.set_xticklabels(instances, rotation=rot, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())


def _save(fig: "plt.Figure", path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_metric(
    df_base: pd.DataFrame,
    df_new:  pd.DataFrame,
    col:      str,
    ylabel:   str,
    title:    str,
    filename: str,
    out_dir:  Path,
) -> None:
    base_vals = metric_per_instance(df_base, col)
    new_vals  = metric_per_instance(df_new,  col)

    if base_vals.empty and new_vals.empty:
        print(f"  Column '{col}' not found in either file – skipping {filename}")
        return

    all_inst = sorted(base_vals.index.union(new_vals.index))
    b_list   = [float(base_vals.get(i, 0)) for i in all_inst]
    n_list   = [float(new_vals.get(i,  0)) for i in all_inst]

    # Wider figure when there are many instances
    w = max(8, len(all_inst) * 0.8)
    fig, ax = plt.subplots(figsize=(w, 5))
    _bar_comparison(ax, all_inst, b_list, n_list, ylabel, title)
    _save(fig, out_dir / filename)


def plot_speedup(
    df_base: pd.DataFrame,
    df_new:  pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Horizontal bar: speedup = baseline_time / new_time per instance.
    Orange  →  new is faster (speedup > 1).
    Blue    →  new is slower (speedup < 1).
    """
    base_vals = metric_per_instance(df_base, RUNTIME_COL)
    new_vals  = metric_per_instance(df_new,  RUNTIME_COL)
    if base_vals.empty:
        print("  No runtime data – skipping speedup plot.")
        return

    all_inst = sorted(base_vals.index.union(new_vals.index))
    speedups = []
    for inst in all_inst:
        b = float(base_vals.get(inst, float("nan")))
        n = float(new_vals.get(inst,  float("nan")))
        speedups.append(b / n if (n and n != 0) else float("nan"))

    bar_colors = [
        COLORS[LABEL_NEW] if (not np.isnan(s) and s >= 1) else COLORS[LABEL_BASELINE]
        for s in speedups
    ]

    h = max(5, len(all_inst) * 0.45)
    fig, ax = plt.subplots(figsize=(6, h))
    y = np.arange(len(all_inst))
    ax.barh(y, speedups, color=bar_colors, alpha=0.85)
    ax.axvline(1.0, color="black", linewidth=1.4, linestyle="--",
               label="no change")
    ax.set_yticks(y)
    ax.set_yticklabels(all_inst, fontsize=7)
    ax.set_xlabel(f"Speedup  ({LABEL_BASELINE} time / {LABEL_NEW} time)")
    ax.set_title(
        f"Runtime speedup per TSP instance\n"
        f"Orange = new faster,  Blue = new slower"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    _save(fig, out_dir / "plot_tsp_speedup.png")


# ---------------------------------------------------------------------------
# Summary statistics (aggregate over all instances)
# ---------------------------------------------------------------------------

def print_summary(table: pd.DataFrame) -> None:
    col_b = f"{LABEL_BASELINE}_value"
    col_n = f"{LABEL_NEW}_value"
    print("\n--- Aggregate summary (mean across all instances) ---")
    for metric, grp in table.groupby("metric", sort=False):
        b_mean = grp[col_b].mean()
        n_mean = grp[col_n].mean()
        wins   = (grp["abs_diff"] < 0).sum()   # new is strictly better
        total  = grp["abs_diff"].notna().sum()
        print(
            f"  {metric:<28}"
            f"  baseline avg={b_mean:>10.3f}"
            f"  new avg={n_mean:>10.3f}"
            f"  new wins={wins}/{total}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two TSP propagator benchmark runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", type=Path, default=None,
                   help="CSV from the baseline run (default: second-most-recent)")
    p.add_argument("--new",      type=Path, default=None,
                   help="CSV from the new propagator run (default: most-recent)")
    p.add_argument("--out",      type=Path, default=RESULTS_DIR,
                   help="Output directory for plots and comparison CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.baseline and args.new:
        path_base, path_new = args.baseline, args.new
    elif args.baseline or args.new:
        sys.exit("Provide both --baseline and --new, or neither.")
    else:
        path_base, path_new = two_latest_csvs(RESULTS_DIR)

    print(f"\nTSP propagator comparison")
    print(f"  {LABEL_BASELINE:>8} ←  {path_base}")
    print(f"  {LABEL_NEW:>8} ←  {path_new}\n")
    print("Loading CSVs:")
    df_base = load(path_base, LABEL_BASELINE)
    df_new  = load(path_new,  LABEL_NEW)

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- Per-instance comparison table ------------------------------------
    print("\n--- Per-instance comparison ---")
    table = build_comparison_table(df_base, df_new)

    col_b = f"{LABEL_BASELINE}_value"
    col_n = f"{LABEL_NEW}_value"
    for metric, grp in table.groupby("metric", sort=False):
        print(f"\n  ── {metric} ──")
        print(
            grp[[INSTANCE_COL, col_b, col_n, "abs_diff", "rel_diff_%"]]
            .to_string(index=False)
        )

    print_summary(table)

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = args.out / f"comparison_{ts}.csv"
    table.to_csv(table_path, index=False)
    print(f"\n  Comparison CSV: {table_path}")

    # ---- Plots ------------------------------------------------------------
    if not HAS_MPL:
        return

    print("\n--- Generating plots ---")

    plot_metric(
        df_base, df_new,
        col      = RUNTIME_COL,
        ylabel   = "Wall-clock time (s)",
        title    = f"Runtime per TSP instance  [{LABEL_BASELINE} vs {LABEL_NEW}]",
        filename = "plot_tsp_runtime.png",
        out_dir  = args.out,
    )
    plot_metric(
        df_base, df_new,
        col      = SEARCH_COL,
        ylabel   = "Failures (backtracks)",
        title    = f"Search reduction per instance  [{LABEL_BASELINE} vs {LABEL_NEW}]",
        filename = "plot_tsp_search_reduction.png",
        out_dir  = args.out,
    )
    plot_metric(
        df_base, df_new,
        col      = EXPLANATION_COL,
        ylabel   = "noGoods generated",
        title    = f"Explanation overhead per instance  [{LABEL_BASELINE} vs {LABEL_NEW}]",
        filename = "plot_tsp_explanation.png",
        out_dir  = args.out,
    )
    plot_speedup(df_base, df_new, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()