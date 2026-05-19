"""
analyse_results.py  –  Two-propagator comparison for circuit benchmark
=======================================================================

Loads the two most-recent stats_*.csv files from experiments/results/,
treats the older one as "baseline" and the newer one as "new", then
produces a comparison across all (n, k) configurations.

Output structure
----------------
  results/
    comparison_<timestamp>.csv        full numeric table
    sat/
      plot_runtime.png
      plot_search_failures.png
      plot_flat_*.png                 ← flattening statistics
      ...

Usage (from the Pumpkin project root)
--------------------------------------
    python experiments/analyse_results.py [options]

Options
-------
  --baseline  PATH   Explicit path to baseline stats CSV
  --new       PATH   Explicit path to new-propagator stats CSV
  --out       PATH   Output directory  (default: experiments/results/)
  --no-plots         Skip plot generation (table only)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas")

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    print("matplotlib/numpy not found – skipping plots")
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Core solve metrics
METRIC_COLS: dict[str, str] = {
    "Wall-clock time (s)"           : "wall_time_s",
    "Solve time (s)"                : "solveTime",
    "Failures (backtracks)"         : "failures",
    "Nodes explored"                : "nodes",
    "Propagations"                  : "propagations",
    "Peak depth"                    : "peakDepth",
    "Restarts"                      : "restarts",
    "Average LBD"                   : "AverageLbd",
    "Avg learned nogood length"     : "AverageLearnedNogoodLength",
    "Nogoods learned (total)"       : "nogoods",
    "Unit nogoods learned"          : "NumUnitNogoodsLearned",
    "Average conflict size"         : "AverageConflictSize",
    "Average backtrack amount"      : "AverageBacktrackAmount",
    # Flattening stats (emitted by MiniZinc before solving)
    "Flat integer variables"        : "flatIntVars",
    "Flat integer constraints"      : "flatIntConstraints",
    "Flattening time (s)"           : "flatTime",
    "Paths"                         : "paths",
}

RUNTIME_COL = "solveTime"
WALL_COL    = "wall_time_s"

SEARCH_COLS: dict[str, tuple[str, str, str]] = {
    "failures"    : ("failures",    "Mean failures (backtracks)", "lower is better"),
    "nodes"       : ("nodes",       "Mean nodes explored",        "lower is better"),
    "propagations": ("propagations","Mean propagations",          "lower is better"),
    "peakdepth"   : ("peakDepth",   "Mean peak search depth",     "informational"),
    "restarts"    : ("restarts",    "Mean restarts",              "informational"),
}

EXPL_COLS: dict[str, tuple[str, str, str]] = {
    "lbd"           : ("AverageLbd",                 "Mean average LBD",             "lower = stronger"),
    "nogood_length" : ("AverageLearnedNogoodLength",  "Mean avg nogood length",       "lower = stronger"),
    "nogoods"       : ("nogoods",                    "Mean nogoods learned (total)",  "informational"),
    "unit_nogoods"  : ("NumUnitNogoodsLearned",       "Mean unit nogoods learned",    "higher = stronger"),
    "conflict_size" : ("AverageConflictSize",         "Mean average conflict size",   "lower = cheaper"),
}

# Flattening metrics plotted as their own group
FLAT_COLS: dict[str, tuple[str, str, str]] = {
    "flat_intvars"    : ("flatIntVars",        "Flat integer variables",   "informational"),
    "flat_intcons"    : ("flatIntConstraints", "Flat integer constraints", "informational"),
    "flat_time"       : ("flatTime",           "Flattening time (s)",      "lower is better"),
}

LABEL_BASELINE = "baseline"
LABEL_NEW      = "new"
COLORS         = {LABEL_BASELINE: "#4C72B0", LABEL_NEW: "#DD8452"}

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
            f"Need at least 2 stats_*.csv files in {results_dir}, found {len(csvs)}.\n"
            "Pass --baseline and --new explicitly if files are elsewhere."
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
    return df

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def config_label(n, k) -> str:
    return f"n={int(n)}, k={int(k)}"


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"] if "status" in df.columns else df


def mean_per_config(df: pd.DataFrame, col: str) -> pd.Series:
    ok = _ok(df)
    if col not in ok.columns:
        return pd.Series(dtype=float)
    return ok.groupby(["config_n", "config_k"])[col].mean()


def raw_per_config(df: pd.DataFrame, col: str) -> dict:
    ok = _ok(df)
    if col not in ok.columns:
        return {}
    result = {}
    for cfg, grp in ok.groupby(["config_n", "config_k"]):
        vals = grp[col].dropna().tolist()
        if vals:
            result[cfg] = vals
    return result

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    df_base: pd.DataFrame,
    df_new:  pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for display_name, col in METRIC_COLS.items():
        base_means = mean_per_config(df_base, col)
        new_means  = mean_per_config(df_new,  col)
        all_configs = sorted(base_means.index.union(new_means.index))
        for cfg in all_configs:
            b = base_means.get(cfg, float("nan"))
            n = new_means.get(cfg,  float("nan"))
            diff = n - b
            rel  = (diff / b * 100) if (b and b != 0) else float("nan")
            rows.append({
                "metric"        : display_name,
                "config_n"      : cfg[0],
                "config_k"      : cfg[1],
                "config"        : config_label(*cfg),
                "baseline_mean" : round(b, 4),
                "new_mean"      : round(n, 4),
                "abs_diff"      : round(diff, 4),
                "rel_diff_%"    : round(rel, 2),
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _grouped_bar_with_points(
    ax, configs, base_vals, new_vals, base_raw, new_raw, all_configs, ylabel, title
):
    x     = np.arange(len(configs))
    width = 0.35
    rng   = np.random.default_rng(seed=0)

    ax.bar(x - width/2, base_vals, width, label=LABEL_BASELINE,
           color=COLORS[LABEL_BASELINE], alpha=0.82, zorder=2)
    ax.bar(x + width/2, new_vals,  width, label=LABEL_NEW,
           color=COLORS[LABEL_NEW],      alpha=0.82, zorder=2)

    for i, cfg in enumerate(all_configs):
        for offset, raw_dict in [(-width/2, base_raw), (+width/2, new_raw)]:
            pts = raw_dict.get(cfg, [])
            if pts:
                jitter = rng.uniform(-0.07, 0.07, len(pts))
                ax.scatter(np.full(len(pts), x[i] + offset) + jitter, pts,
                           color="black", s=20, alpha=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"    Saved: {path.name}")
    plt.close(fig)


def plot_metric(df_base, df_new, col, ylabel, title, filename, out_dir):
    base_means = mean_per_config(df_base, col)
    new_means  = mean_per_config(df_new,  col)
    if base_means.empty and new_means.empty:
        print(f"    Column '{col}' not found – skipping {filename}")
        return

    all_configs = sorted(base_means.index.union(new_means.index))
    labels   = [config_label(*c) for c in all_configs]
    b_vals   = [float(base_means.get(c, 0)) for c in all_configs]
    n_vals   = [float(new_means.get(c,  0)) for c in all_configs]
    base_raw = raw_per_config(df_base, col)
    new_raw  = raw_per_config(df_new,  col)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 5))
    _grouped_bar_with_points(ax, labels, b_vals, n_vals,
                             base_raw, new_raw, all_configs, ylabel, title)
    _save(fig, out_dir / filename)


def plot_speedup(df_base, df_new, out_dir):
    base_means = mean_per_config(df_base, RUNTIME_COL)
    new_means  = mean_per_config(df_new,  RUNTIME_COL)
    if base_means.empty:
        return

    all_configs = sorted(base_means.index.union(new_means.index))
    labels   = [config_label(*c) for c in all_configs]
    speedups = []
    for c in all_configs:
        b = float(base_means.get(c, float("nan")))
        n = float(new_means.get(c,  float("nan")))
        speedups.append(b / n if (n and n != 0) else float("nan"))

    bar_colors = [
        COLORS[LABEL_NEW] if (not np.isnan(s) and s >= 1) else COLORS[LABEL_BASELINE]
        for s in speedups
    ]

    fig, ax = plt.subplots(figsize=(6, max(4, len(labels) * 0.55)))
    y = np.arange(len(labels))
    ax.barh(y, speedups, color=bar_colors, alpha=0.85)
    ax.axvline(1.0, color="black", linewidth=1.4, linestyle="--", label="no change")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Speedup  (baseline solveTime / new solveTime)")
    ax.set_title(
        f"Runtime speedup  '{LABEL_NEW}' over '{LABEL_BASELINE}'\n"
        "Blue = new is slower,  Orange = new is faster"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    _save(fig, out_dir / "plot_speedup.png")

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    df_base: pd.DataFrame,
    df_new:  pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Print comparison table to stdout ---
    table = build_comparison_table(df_base, df_new)
    for metric, grp in table.groupby("metric", sort=False):
        print(f"\n  ── {metric} ──")
        print(
            grp[["config", "baseline_mean", "new_mean", "abs_diff", "rel_diff_%"]]
            .to_string(index=False)
        )

    if not HAS_MPL:
        return table

    # --- Runtime ---
    plot_metric(df_base, df_new,
                col=RUNTIME_COL, ylabel="Mean solve time (s)",
                title=f"Solve time  [{LABEL_BASELINE} vs {LABEL_NEW}]",
                filename="plot_runtime_solve.png", out_dir=out_dir)

    plot_metric(df_base, df_new,
                col=WALL_COL, ylabel="Mean wall-clock time (s)",
                title=f"Wall-clock time  [{LABEL_BASELINE} vs {LABEL_NEW}]",
                filename="plot_runtime_wall.png", out_dir=out_dir)

    plot_speedup(df_base, df_new, out_dir)

    # --- Search reduction ---
    for suffix, (col, ylabel, note) in SEARCH_COLS.items():
        plot_metric(df_base, df_new,
                    col=col, ylabel=ylabel,
                    title=f"{ylabel}  ({note})",
                    filename=f"plot_search_{suffix}.png", out_dir=out_dir)

    # --- Explanation quality ---
    for suffix, (col, ylabel, note) in EXPL_COLS.items():
        plot_metric(df_base, df_new,
                    col=col, ylabel=ylabel,
                    title=f"{ylabel}  ({note})",
                    filename=f"plot_expl_{suffix}.png", out_dir=out_dir)

    # --- Flattening stats ---
    for suffix, (col, ylabel, note) in FLAT_COLS.items():
        plot_metric(df_base, df_new,
                    col=col, ylabel=ylabel,
                    title=f"{ylabel}  ({note})",
                    filename=f"plot_flat_{suffix}.png", out_dir=out_dir)

    return table

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare two circuit-propagator benchmark runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline",  type=Path, default=None,
                   help="Baseline stats CSV")
    p.add_argument("--new",       type=Path, default=None,
                   help="New-propagator stats CSV")
    p.add_argument("--out",       type=Path, default=RESULTS_DIR,
                   help="Output directory for plots and comparison CSV")
    p.add_argument("--no-plots",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_plots:
        global HAS_MPL
        HAS_MPL = False

    print("\nLoading CSVs:")
    if args.baseline and args.new:
        df_base = load(args.baseline, LABEL_BASELINE)
        df_new  = load(args.new,      LABEL_NEW)
    elif args.baseline or args.new:
        sys.exit("Provide both --baseline and --new, or neither.")
    else:
        path_base, path_new = two_latest_csvs(RESULTS_DIR)
        df_base = load(path_base, LABEL_BASELINE)
        df_new  = load(path_new,  LABEL_NEW)

    print(f"\n{'='*60}")
    print(f"  Comparing {len(df_base)} baseline rows vs {len(df_new)} new rows")
    print(f"  Output: {args.out}")
    print(f"{'='*60}")

    out_dir = args.out / "sat"
    table   = run_analysis(df_base, df_new, out_dir)

    # Save combined comparison CSV
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = args.out / f"comparison_{ts}.csv"
    table.to_csv(table_path, index=False)
    print(f"\n  Full comparison CSV: {table_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()