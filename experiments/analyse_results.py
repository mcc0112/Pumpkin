"""
analyse_results.py  –  Two-propagator comparison for circuit benchmark
=======================================================================

Loads the two most-recent stats_*.csv files from experiments/results/,
treats the older one as "baseline" and the newer one as "new", then
produces separate analyses for three instance types:

  SAT geographic instances   (instance_type == "sat")
  UNSAT instances            (instance_type == "unsat_random" or "unsat_forced")
  TSP instances              (detected by presence of "instance_name" column
                              and absence of "instance_type", or loaded via
                              --tsp-baseline / --tsp-new)

Output structure
----------------
  results/
    comparison_<timestamp>.csv        full numeric table (all types)
    sat/
      plot_runtime.png
      plot_search_failures.png
      ...
    unsat_random/
      plot_runtime.png
      ...
    unsat_forced/
      plot_runtime.png
      ...
    tsp/
      plot_runtime.png
      ...

Usage (from the Pumpkin project root)
--------------------------------------
    python experiments/analyse_results.py [options]

Options
-------
  --baseline      PATH   Explicit path to baseline stats CSV
  --new           PATH   Explicit path to new-propagator stats CSV
  --tsp-baseline  PATH   Baseline CSV from run_tsp_experiments.py
  --tsp-new       PATH   New-propagator CSV from run_tsp_experiments.py
  --out           PATH   Output directory  (default: experiments/results/)
  --no-plots             Skip plot generation (table only)
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
}

RUNTIME_COL  = "solveTime"   # primary runtime metric (no Cargo noise)
WALL_COL     = "wall_time_s"

SEARCH_COLS: dict[str, tuple[str, str, str]] = {
    "failures"    : ("failures",    "Mean failures (backtracks)",  "lower is better"),
    "nodes"       : ("nodes",       "Mean nodes explored",         "lower is better"),
    "propagations": ("propagations","Mean propagations",           "lower is better"),
    "peakdepth"   : ("peakDepth",   "Mean peak search depth",      "informational"),
    "restarts"    : ("restarts",    "Mean restarts",               "informational"),
}

EXPL_COLS: dict[str, tuple[str, str, str]] = {
    "lbd"           : ("AverageLbd",                "Mean average LBD",              "lower = stronger"),
    "nogood_length" : ("AverageLearnedNogoodLength", "Mean avg nogood length",        "lower = stronger"),
    "nogoods"       : ("nogoods",                   "Mean nogoods learned (total)",  "informational"),
    "unit_nogoods"  : ("NumUnitNogoodsLearned",      "Mean unit nogoods learned",     "higher = stronger"),
    "conflict_size" : ("AverageConflictSize",        "Mean average conflict size",    "lower = cheaper"),
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
# Loading & splitting
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
    # If no instance_type column exists, infer from columns present
    if "instance_type" not in df.columns:
        if "instance_name" in df.columns:
            df["instance_type"] = "tsp"
        else:
            df["instance_type"] = "sat"
    print(f"  [{label:>8}]  {len(df)} rows  ←  {csv_path.name}")
    if "instance_type" in df.columns:
        for t, grp in df.groupby("instance_type"):
            print(f"              {len(grp):>4} rows  instance_type={t}")
    return df


def split_by_type(
    df_base: pd.DataFrame,
    df_new: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Return a dict mapping instance_type -> (df_base_subset, df_new_subset).
    Only returns types that have data in both dataframes.
    """
    types_base = set(df_base["instance_type"].unique())
    types_new  = set(df_new["instance_type"].unique())
    common     = types_base & types_new

    result = {}
    for t in sorted(common):
        b = df_base[df_base["instance_type"] == t].copy()
        n = df_new[df_new["instance_type"] == t].copy()
        if len(b) > 0 and len(n) > 0:
            result[t] = (b, n)
    return result

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def config_label_geo(n, k) -> str:
    return f"n={int(n)}, k={int(k)}"


def mean_per_geo_config(df: pd.DataFrame, col: str) -> pd.Series:
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if col not in ok.columns:
        return pd.Series(dtype=float)
    return ok.groupby(["config_n", "config_k"])[col].mean()


def raw_per_geo_config(df: pd.DataFrame, col: str) -> dict:
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if col not in ok.columns:
        return {}
    result = {}
    for cfg, grp in ok.groupby(["config_n", "config_k"]):
        vals = grp[col].dropna().tolist()
        if vals:
            result[cfg] = vals
    return result


def mean_per_tsp_instance(df: pd.DataFrame, col: str) -> pd.Series:
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if col not in ok.columns or "instance_name" not in ok.columns:
        return pd.Series(dtype=float)
    return ok.groupby("instance_name")[col].mean()

# ---------------------------------------------------------------------------
# Comparison table builder (works for both geo and TSP)
# ---------------------------------------------------------------------------

def build_comparison_table(
    df_base: pd.DataFrame,
    df_new:  pd.DataFrame,
    instance_type: str,
) -> pd.DataFrame:
    rows = []
    is_tsp = (instance_type == "tsp")

    for display_name, col in METRIC_COLS.items():
        if is_tsp:
            base_means = mean_per_tsp_instance(df_base, col)
            new_means  = mean_per_tsp_instance(df_new,  col)
            all_configs = sorted(base_means.index.union(new_means.index))
            for cfg in all_configs:
                b = base_means.get(cfg, float("nan"))
                n = new_means.get(cfg,  float("nan"))
                diff = n - b
                rel  = (diff / b * 100) if (b and b != 0) else float("nan")
                rows.append({
                    "metric"         : display_name,
                    "instance_type"  : instance_type,
                    "instance"       : cfg,
                    "baseline_mean"  : round(b, 4),
                    "new_mean"       : round(n, 4),
                    "abs_diff"       : round(diff, 4),
                    "rel_diff_%"     : round(rel, 2),
                })
        else:
            base_means = mean_per_geo_config(df_base, col)
            new_means  = mean_per_geo_config(df_new,  col)
            all_configs = sorted(base_means.index.union(new_means.index))
            for cfg in all_configs:
                b = base_means.get(cfg, float("nan"))
                n = new_means.get(cfg,  float("nan"))
                diff = n - b
                rel  = (diff / b * 100) if (b and b != 0) else float("nan")
                rows.append({
                    "metric"         : display_name,
                    "instance_type"  : instance_type,
                    "config_n"       : cfg[0],
                    "config_k"       : cfg[1],
                    "config"         : config_label_geo(*cfg),
                    "baseline_mean"  : round(b, 4),
                    "new_mean"       : round(n, 4),
                    "abs_diff"       : round(diff, 4),
                    "rel_diff_%"     : round(rel, 2),
                })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _grouped_bar_with_points(ax, configs, base_vals, new_vals,
                              base_raw, new_raw, all_configs, ylabel, title):
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

# ---------------------------------------------------------------------------
# Geographic (SAT / UNSAT) plots
# ---------------------------------------------------------------------------

def plot_geo_metric(df_base, df_new, col, ylabel, title, filename, out_dir):
    base_means = mean_per_geo_config(df_base, col)
    new_means  = mean_per_geo_config(df_new,  col)
    if base_means.empty and new_means.empty:
        print(f"    Column '{col}' not found – skipping {filename}")
        return

    all_configs = sorted(base_means.index.union(new_means.index))
    labels   = [config_label_geo(*c) for c in all_configs]
    b_vals   = [float(base_means.get(c, 0)) for c in all_configs]
    n_vals   = [float(new_means.get(c,  0)) for c in all_configs]
    base_raw = raw_per_geo_config(df_base, col)
    new_raw  = raw_per_geo_config(df_new,  col)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 5))
    _grouped_bar_with_points(ax, labels, b_vals, n_vals,
                             base_raw, new_raw, all_configs, ylabel, title)
    _save(fig, out_dir / filename)


def plot_geo_speedup(df_base, df_new, out_dir, title_suffix=""):
    base_means = mean_per_geo_config(df_base, RUNTIME_COL)
    new_means  = mean_per_geo_config(df_new,  RUNTIME_COL)
    if base_means.empty:
        return

    all_configs = sorted(base_means.index.union(new_means.index))
    labels   = [config_label_geo(*c) for c in all_configs]
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
    ax.set_xlabel(f"Speedup  (baseline solveTime / new solveTime)")
    ax.set_title(
        f"Runtime speedup  '{LABEL_NEW}' over '{LABEL_BASELINE}'{title_suffix}\n"
        "Blue = new is slower,  Orange = new is faster"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    _save(fig, out_dir / "plot_speedup.png")

# ---------------------------------------------------------------------------
# TSP plots  (x-axis = instance name, not (n,k))
# ---------------------------------------------------------------------------

def plot_tsp_metric(df_base, df_new, col, ylabel, title, filename, out_dir):
    base_means = mean_per_tsp_instance(df_base, col)
    new_means  = mean_per_tsp_instance(df_new,  col)
    if base_means.empty and new_means.empty:
        print(f"    Column '{col}' not found – skipping {filename}")
        return

    all_instances = sorted(base_means.index.union(new_means.index))
    b_vals = [float(base_means.get(i, 0)) for i in all_instances]
    n_vals = [float(new_means.get(i,  0)) for i in all_instances]

    x     = np.arange(len(all_instances))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(all_instances) * 1.1), 5))
    ax.bar(x - width/2, b_vals, width, label=LABEL_BASELINE,
           color=COLORS[LABEL_BASELINE], alpha=0.82)
    ax.bar(x + width/2, n_vals, width, label=LABEL_NEW,
           color=COLORS[LABEL_NEW], alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(all_instances, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    _save(fig, out_dir / filename)


def plot_tsp_speedup(df_base, df_new, out_dir):
    base_means = mean_per_tsp_instance(df_base, RUNTIME_COL)
    new_means  = mean_per_tsp_instance(df_new,  RUNTIME_COL)
    if base_means.empty:
        return

    all_instances = sorted(base_means.index.union(new_means.index))
    speedups = []
    for i in all_instances:
        b = float(base_means.get(i, float("nan")))
        n = float(new_means.get(i,  float("nan")))
        speedups.append(b / n if (n and n != 0) else float("nan"))

    bar_colors = [
        COLORS[LABEL_NEW] if (not np.isnan(s) and s >= 1) else COLORS[LABEL_BASELINE]
        for s in speedups
    ]

    fig, ax = plt.subplots(figsize=(5, max(3, len(all_instances) * 0.55)))
    y = np.arange(len(all_instances))
    ax.barh(y, speedups, color=bar_colors, alpha=0.85)
    ax.axvline(1.0, color="black", linewidth=1.4, linestyle="--", label="no change")
    ax.set_yticks(y)
    ax.set_yticklabels(all_instances, fontsize=9)
    ax.set_xlabel("Speedup  (baseline / new)")
    ax.set_title("TSP runtime speedup\nBlue = new slower,  Orange = new faster")
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    _save(fig, out_dir / "plot_speedup.png")

# ---------------------------------------------------------------------------
# Full section runner
# ---------------------------------------------------------------------------

def run_section(
    df_base: pd.DataFrame,
    df_new:  pd.DataFrame,
    instance_type: str,
    out_dir: Path,
    all_tables: list,
) -> None:
    """Generate all plots and table rows for one instance_type section."""
    out_dir.mkdir(parents=True, exist_ok=True)
    is_tsp = (instance_type == "tsp")

    print(f"\n{'='*60}")
    print(f"  Section: {instance_type.upper()}  ({len(df_base)} baseline rows)")
    print(f"  Output:  {out_dir}")
    print(f"{'='*60}")

    # --- Comparison table ---
    table = build_comparison_table(df_base, df_new, instance_type)
    all_tables.append(table)

    for metric, grp in table.groupby("metric", sort=False):
        val_col = "instance" if is_tsp else "config"
        if val_col in grp.columns:
            print(f"\n  ── {metric} ──")
            print(
                grp[[val_col, "baseline_mean", "new_mean", "abs_diff", "rel_diff_%"]]
                .to_string(index=False)
            )

    if not HAS_MPL:
        return

    plot_fn      = plot_tsp_metric    if is_tsp else plot_geo_metric
    speedup_fn   = plot_tsp_speedup   if is_tsp else plot_geo_speedup

    type_label = f"  [{instance_type}]"

    # Runtime
    plot_fn(df_base, df_new,
            col=RUNTIME_COL, ylabel="Mean solve time (s)",
            title=f"Solve time{type_label}  [{LABEL_BASELINE} vs {LABEL_NEW}]",
            filename="plot_runtime_solve.png", out_dir=out_dir)

    plot_fn(df_base, df_new,
            col=WALL_COL, ylabel="Mean wall-clock time (s)",
            title=f"Wall-clock time{type_label}  [{LABEL_BASELINE} vs {LABEL_NEW}]",
            filename="plot_runtime_wall.png", out_dir=out_dir)

    # Search reduction
    for suffix, (col, ylabel, note) in SEARCH_COLS.items():
        plot_fn(df_base, df_new,
                col=col, ylabel=ylabel,
                title=f"{ylabel}{type_label}  ({note})",
                filename=f"plot_search_{suffix}.png", out_dir=out_dir)

    # Explanation quality
    for suffix, (col, ylabel, note) in EXPL_COLS.items():
        plot_fn(df_base, df_new,
                col=col, ylabel=ylabel,
                title=f"{ylabel}{type_label}  ({note})",
                filename=f"plot_expl_{suffix}.png", out_dir=out_dir)

    # Speedup
    title_suffix = f"  [{instance_type}]" if not is_tsp else ""
    if is_tsp:
        speedup_fn(df_base, df_new, out_dir)
    else:
        speedup_fn(df_base, df_new, out_dir, title_suffix=title_suffix)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare two circuit-propagator benchmark runs, split by instance type.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline",     type=Path, default=None,
                   help="Baseline stats CSV (geographic + UNSAT instances)")
    p.add_argument("--new",          type=Path, default=None,
                   help="New-propagator stats CSV")
    p.add_argument("--tsp-baseline", type=Path, default=None,
                   help="Baseline CSV from run_tsp_experiments.py")
    p.add_argument("--tsp-new",      type=Path, default=None,
                   help="New-propagator CSV from run_tsp_experiments.py")
    p.add_argument("--out",          type=Path, default=RESULTS_DIR)
    p.add_argument("--no-plots",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_plots:
        global HAS_MPL
        HAS_MPL = False

    # ---- Load geographic/UNSAT CSVs (optional if TSP-only run) -------------
    print("\nLoading CSVs:")
    df_base = pd.DataFrame()
    df_new  = pd.DataFrame()

    if args.baseline and args.new:
        df_base = load(args.baseline, LABEL_BASELINE)
        df_new  = load(args.new,      LABEL_NEW)
    elif args.baseline or args.new:
        sys.exit("Provide both --baseline and --new, or neither.")
    elif not (args.tsp_baseline and args.tsp_new):
        # No explicit files at all — try auto-detect from results dir
        path_base, path_new = two_latest_csvs(RESULTS_DIR)
        df_base = load(path_base, LABEL_BASELINE)
        df_new  = load(path_new,  LABEL_NEW)
    else:
        print("  No geographic CSVs provided — running TSP-only analysis.")

    # ---- Optionally load TSP CSVs ------------------------------------------
    if args.tsp_baseline and args.tsp_new:
        df_tsp_base = load(args.tsp_baseline, LABEL_BASELINE)
        df_tsp_new  = load(args.tsp_new,      LABEL_NEW)
        df_tsp_base["instance_type"] = "tsp"
        df_tsp_new["instance_type"]  = "tsp"
        df_base = pd.concat([df_base, df_tsp_base], ignore_index=True)
        df_new  = pd.concat([df_new,  df_tsp_new],  ignore_index=True)

    # ---- Split and run per-section -----------------------------------------
    sections = split_by_type(df_base, df_new)

    if not sections:
        sys.exit("No matching instance types found between baseline and new CSVs.")

    print(f"\nFound sections: {list(sections.keys())}")

    all_tables = []
    for instance_type, (b, n) in sections.items():
        section_dir = args.out / instance_type
        run_section(b, n, instance_type, section_dir, all_tables)

    # ---- Save combined comparison table ------------------------------------
    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_path = args.out / f"comparison_{ts}.csv"
        combined.to_csv(table_path, index=False)
        print(f"\n  Full comparison CSV: {table_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()