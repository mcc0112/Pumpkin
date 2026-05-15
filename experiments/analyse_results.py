"""
analyse_results.py  –  Two-propagator comparison for circuit benchmark
=======================================================================

Loads the two most-recent stats_*.csv files from experiments/results/,
treats the older one as "baseline" and the newer one as "new", then:

  • Prints a per-configuration comparison table to the terminal
  • Writes  experiments/results/comparison_<timestamp>.csv
  • Saves four PNG plots:
      plot_runtime.png           – wall-clock time  (grouped bars + raw dots)
      plot_search_reduction.png  – failures / backtracks
      plot_explanation.png       – noGoods generated
      plot_speedup.png           – runtime speedup ratio  baseline / new

Usage (from the Pumpkin project root)
--------------------------------------
    python experiments/analyse_results.py [options]

Options
-------
  --baseline PATH   Explicit path to the baseline CSV
  --new      PATH   Explicit path to the new-propagator CSV
  --out      PATH   Output directory  (default: experiments/results/)

If neither --baseline nor --new is given the script auto-detects the two
most-recent stats_*.csv files in experiments/results/ (older = baseline,
newer = new).

Column mapping
--------------
Edit METRIC_COLS near the top of the file if your solver uses different
stat names.  The defaults are:
  runtime             → wall_time_s      (always present)
  search reduction    → failures         (%%%mzn-stat: failures=…)
  explanation overhead→ noGoods          (%%%mzn-stat: noGoods=…)
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
    print("matplotlib/numpy not found – skipping plots (uv add matplotlib)")
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Configuration – edit these if your solver emits different stat names
# ---------------------------------------------------------------------------

METRIC_COLS: dict[str, str] = {
    # Human-readable label        : CSV column name
    "Runtime (s)"                 : "wall_time_s",
    "Failures (backtracks)"       : "failures",
    "noGoods"                     : "noGoods",
}

RUNTIME_COL      = "wall_time_s"
SEARCH_COL       = "failures"
EXPLANATION_COL  = "noGoods"

LABEL_BASELINE   = "baseline"
LABEL_NEW        = "new"
COLORS           = {LABEL_BASELINE: "#4C72B0", LABEL_NEW: "#DD8452"}

GROUP_KEYS       = ["config_n", "config_k"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def two_latest_csvs(results_dir: Path) -> tuple[Path, Path]:
    """Return (older, newer) of the two most-recent stats_*.csv files."""
    csvs = sorted(results_dir.glob("stats_*.csv"))
    if len(csvs) < 2:
        sys.exit(
            f"Need at least 2 stats_*.csv files in {results_dir}, "
            f"found {len(csvs)}.\n"
            "Run the experiments with both propagators first, or pass "
            "--baseline and --new explicitly."
        )
    return csvs[-2], csvs[-1]   # (older = baseline, newer = new)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric where possible (pandas-2.x safe)."""
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

def config_label(n: int | float, k: int | float) -> str:
    return f"n={int(n)}, k={int(k)}"


def mean_per_config(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean of *col* over successful runs, indexed by (config_n, config_k)."""
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if col not in ok.columns:
        return pd.Series(dtype=float)
    return ok.groupby(GROUP_KEYS)[col].mean()


def raw_per_config(
    df: pd.DataFrame,
    col: str,
) -> dict[tuple, list[float]]:
    """All individual (successful) values, keyed by (config_n, config_k)."""
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if col not in ok.columns:
        return {}
    result: dict[tuple, list[float]] = {}
    for cfg, grp in ok.groupby(GROUP_KEYS):
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
    """
    For every metric and every (n, k) configuration return a tidy DataFrame
    with columns:
      metric, config_n, config_k, config,
      baseline_mean, new_mean, abs_diff, rel_diff_%
    """
    rows = []
    for display_name, col in METRIC_COLS.items():
        base_means = mean_per_config(df_base, col)
        new_means  = mean_per_config(df_new,  col)
        all_configs = sorted(base_means.index.union(new_means.index))

        for cfg in all_configs:
            b = base_means.get(cfg, float("nan"))
            n = new_means.get(cfg,  float("nan"))
            abs_diff = n - b
            rel_diff = (abs_diff / b * 100) if (b and b != 0) else float("nan")
            rows.append({
                "metric"                         : display_name,
                "config_n"                       : cfg[0],
                "config_k"                       : cfg[1],
                "config"                         : config_label(*cfg),
                f"{LABEL_BASELINE}_mean"         : round(b, 4),
                f"{LABEL_NEW}_mean"              : round(n, 4),
                "abs_diff"                       : round(abs_diff, 4),
                "rel_diff_%"                     : round(rel_diff, 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _grouped_bar_with_points(
    ax: "plt.Axes",
    configs: list[str],
    base_vals: list[float],
    new_vals:  list[float],
    base_raw:  dict[tuple, list[float]],
    new_raw:   dict[tuple, list[float]],
    all_configs: list[tuple],
    ylabel: str,
    title:  str,
) -> None:
    """
    Grouped bar chart (baseline vs new) with individual run dots overlaid.
    """
    x     = np.arange(len(configs))
    width = 0.35
    rng   = np.random.default_rng(seed=0)

    ax.bar(x - width / 2, base_vals, width,
           label=LABEL_BASELINE, color=COLORS[LABEL_BASELINE], alpha=0.82, zorder=2)
    ax.bar(x + width / 2, new_vals,  width,
           label=LABEL_NEW,      color=COLORS[LABEL_NEW],      alpha=0.82, zorder=2)

    # Overlay raw data points
    for i, cfg in enumerate(all_configs):
        for offset, raw_dict in [(-width / 2, base_raw), (+width / 2, new_raw)]:
            pts = raw_dict.get(cfg, [])
            if pts:
                jitter = rng.uniform(-0.07, 0.07, len(pts))
                ax.scatter(
                    np.full(len(pts), x[i] + offset) + jitter, pts,
                    color="black", s=20, alpha=0.5, zorder=3,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8)
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


# ---------------------------------------------------------------------------
# Individual comparison plots
# ---------------------------------------------------------------------------

def plot_metric(
    df_base: pd.DataFrame,
    df_new:  pd.DataFrame,
    col:      str,
    ylabel:   str,
    title:    str,
    filename: str,
    out_dir:  Path,
) -> None:
    base_means = mean_per_config(df_base, col)
    new_means  = mean_per_config(df_new,  col)

    if base_means.empty and new_means.empty:
        print(f"  Column '{col}' not found in either file – skipping {filename}")
        return

    all_configs = sorted(base_means.index.union(new_means.index))
    labels  = [config_label(*c) for c in all_configs]
    b_vals  = [float(base_means.get(c, 0)) for c in all_configs]
    n_vals  = [float(new_means.get(c,  0)) for c in all_configs]

    base_raw = raw_per_config(df_base, col)
    new_raw  = raw_per_config(df_new,  col)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.05), 5))
    _grouped_bar_with_points(
        ax, labels, b_vals, n_vals,
        base_raw, new_raw, all_configs,
        ylabel, title,
    )
    _save(fig, out_dir / filename)


def plot_speedup(
    df_base: pd.DataFrame,
    df_new:  pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Horizontal bar chart of runtime speedup = baseline_mean / new_mean.
    Values > 1  →  new propagator is faster.
    Values < 1  →  new propagator is slower.
    """
    base_means = mean_per_config(df_base, RUNTIME_COL)
    new_means  = mean_per_config(df_new,  RUNTIME_COL)
    if base_means.empty:
        print("  No runtime data for speedup plot – skipping.")
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

    fig, ax = plt.subplots(figsize=(6, max(4, len(labels) * 0.6)))
    y = np.arange(len(labels))
    ax.barh(y, speedups, color=bar_colors, alpha=0.85)
    ax.axvline(1.0, color="black", linewidth=1.4, linestyle="--",
               label="no change  (speedup = 1)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Speedup  ({LABEL_BASELINE} time / {LABEL_NEW} time)")
    ax.set_title(
        f"Runtime speedup of  '{LABEL_NEW}'  over  '{LABEL_BASELINE}'\n"
        f"Blue = new is slower,  Orange = new is faster"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    _save(fig, out_dir / "plot_speedup.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two circuit-propagator benchmark runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", type=Path, default=None,
                   help="CSV from the baseline propagator run "
                        "(default: second-most-recent stats_*.csv)")
    p.add_argument("--new",      type=Path, default=None,
                   help="CSV from the new propagator run "
                        "(default: most-recent stats_*.csv)")
    p.add_argument("--out",      type=Path, default=RESULTS_DIR,
                   help="Output directory for plots and comparison CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Resolve input files -----------------------------------------------
    if args.baseline and args.new:
        path_base, path_new = args.baseline, args.new
    elif args.baseline or args.new:
        sys.exit("Provide both --baseline and --new, or neither (auto-detect).")
    else:
        path_base, path_new = two_latest_csvs(RESULTS_DIR)

    print(f"\nPropagator comparison")
    print(f"  {LABEL_BASELINE:>8} ←  {path_base}")
    print(f"  {LABEL_NEW:>8} ←  {path_new}\n")
    print("Loading CSVs:")
    df_base = load(path_base, LABEL_BASELINE)
    df_new  = load(path_new,  LABEL_NEW)

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- Comparison table --------------------------------------------------
    print("\n--- Per-configuration comparison ---")
    table = build_comparison_table(df_base, df_new)

    col_b = f"{LABEL_BASELINE}_mean"
    col_n = f"{LABEL_NEW}_mean"
    for metric, grp in table.groupby("metric", sort=False):
        print(f"\n  ── {metric} ──")
        print(
            grp[["config", col_b, col_n, "abs_diff", "rel_diff_%"]]
            .to_string(index=False)
        )

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = args.out / f"comparison_{timestamp}.csv"
    table.to_csv(table_path, index=False)
    print(f"\n  Comparison CSV written to: {table_path}")

    # ---- Plots -------------------------------------------------------------
    if not HAS_MPL:
        return

    print("\n--- Generating plots ---")

    plot_metric(
        df_base, df_new,
        col      = RUNTIME_COL,
        ylabel   = "Mean wall-clock time (s)",
        title    = f"Runtime  [{LABEL_BASELINE} vs {LABEL_NEW}]",
        filename = "plot_runtime.png",
        out_dir  = args.out,
    )

    plot_metric(
        df_base, df_new,
        col      = SEARCH_COL,
        ylabel   = "Mean failures (backtracks)",
        title    = f"Search reduction  [{LABEL_BASELINE} vs {LABEL_NEW}]  – fewer is better",
        filename = "plot_search_reduction.png",
        out_dir  = args.out,
    )

    plot_metric(
        df_base, df_new,
        col      = EXPLANATION_COL,
        ylabel   = "Mean noGoods generated",
        title    = f"Explanation overhead  [{LABEL_BASELINE} vs {LABEL_NEW}]  – fewer is better",
        filename = "plot_explanation.png",
        out_dir  = args.out,
    )

    plot_speedup(df_base, df_new, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()