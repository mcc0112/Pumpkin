"""
analyse_results.py  –  Four-variant circuit propagator analysis
===============================================================

Loads one stats CSV per variant (V0–V3), then produces:

  1. Timeout rate table  – per (variant, n, k)
  2. Per-experiment comparison tables (adjacent pairs + V0 reference):
       Experiment 1: V0 vs V1
       Experiment 2: V0 vs V1 vs V2
       Experiment 3: V0 vs V1 vs V2 vs V3
  3. PAR-2 scores  (timeout instances penalised at 2 × timeout limit)
  4. Plots for runtime, search effort, and explanation quality,
     grouped by experiment

All aggregate statistics use the MEDIAN and IQR (not mean/std), because
CP runtime distributions are heavy-tailed.

FIX (1): Log scale applied to propagation and failure plots, consistent
         with paper figures.
FIX (2): Low-sample warnings emitted when a cell has fewer than
         MIN_SAMPLE_WARNING instances after timeout filtering.
FIX (3): solveTime (solver-reported) is reported alongside wall_time_s
         so solver effort is distinguishable from subprocess overhead.
FIX (4): PAR-2 uses wall_time_s as primary source but falls back to
         solveTime if wall_time_s is unavailable, with a clear warning.

Output structure
----------------
  results/
    analysis_<timestamp>/
      timeout_rates.csv
      par2_scores.csv
      experiment1_V0_vs_V1.csv
      experiment2_V0_V1_V2.csv
      experiment3_V0_V1_V2_V3.csv
      plots/
        exp1_runtime.png
        exp1_failures.png
        exp1_propagations.png
        exp1_lbd.png
        ...
        exp2_*.png
        exp3_*.png

Usage
-----
    python experiments/analyse_results.py [options]

Options
-------
  --v0  PATH   CSV for V0 (decomposed baseline)
  --v1  PATH   CSV for V1 (matching-based conflict detection)
  --v2  PATH   CSV for V2 (full GAC with pruning)
  --v3  PATH   CSV for V3 (matching-guided value ordering)
  --out PATH   Output directory  (default: experiments/results/)
  --timeout INT  Timeout limit used during experiments (default: 300)
  --no-plots     Skip plot generation
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("pandas and numpy are required:  pip install pandas numpy")

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    print("matplotlib not found – skipping plots")
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

VARIANTS = ["V0", "V1", "V2", "V3"]

VARIANT_LABELS = {
    "V0": "V0 (Decomposed)",
    "V1": "V1 (Conflict matching)",
    "V2": "V2 (Full GAC)",
    "V3": "V3 (Value ordering)",
}

VARIANT_COLORS = {
    "V0": "#4C72B0",   # blue
    "V1": "#DD8452",   # orange
    "V2": "#C44E52",   # red
    "V3": "#55A868",   # green
}

TIMEOUT_DEFAULT = 300   # seconds; used for PAR-2 penalty

# FIX (2): Cells with fewer solved instances than this threshold after
# timeout filtering will emit a warning during aggregation.
MIN_SAMPLE_WARNING = 10

# Metrics collected from solver output
RUNTIME_COL  = "solveTime"    # solver-reported solve time (excludes overhead)
WALL_COL     = "wall_time_s"  # wall-clock time measured around subprocess call
FAILURES_COL = "failures"

# Search effort metrics: (column, ylabel, direction note, use_log_scale)
# FIX (1): propagations and failures use log scale to match paper figures.
SEARCH_METRICS = {
    "failures":     ("failures",     "Failures (backtracks)", "lower is better", True),
    "nodes":        ("nodes",        "Nodes explored",        "lower is better", True),
    "propagations": ("propagations", "Propagations",          "lower is better", True),
    "peakDepth":    ("peakDepth",    "Peak search depth",     "informational",   False),
    "restarts":     ("restarts",     "Restarts",              "informational",   False),
}

# Explanation quality metrics: (column, ylabel, direction note, use_log_scale)
EXPL_METRICS = {
    "lbd":           ("AverageLbd",                 "Avg LBD",                 "lower = stronger", False),
    "nogood_length": ("AverageLearnedNogoodLength",  "Avg nogood length",       "lower = stronger", False),
    "nogoods":       ("nogoods",                    "Nogoods learned (total)", "informational",    True),
    "unit_nogoods":  ("NumUnitNogoodsLearned",       "Unit nogoods learned",    "higher = stronger",False),
    "conflict_size": ("AverageConflictSize",         "Avg conflict size",       "lower = cheaper",  False),
}

# FIX (3): Both solver-reported and wall-clock runtime are tracked.
# solveTime excludes subprocess/compilation overhead; wall_time_s is used
# for PAR-2. Reporting both lets the reader distinguish solver effort from
# external overhead.
RUNTIME_METRICS = {
    "solve_time": (RUNTIME_COL, "Solve time (s) [solver-reported]", "lower is better", False),
    "wall_time":  (WALL_COL,    "Wall-clock time (s)",               "lower is better", False),
}

# Experiments: each is a list of variant names to compare, with V0 always included
EXPERIMENTS = {
    1: {
        "variants":     ["V0", "V1"],
        "description":  "Conflict detection: Decomposed vs Matching-based feasibility check",
        "filename":     "experiment1_V0_vs_V1",
    },
    2: {
        "variants":     ["V0", "V1", "V2"],
        "description":  "Domain pruning: adding full Régin GAC",
        "filename":     "experiment2_V0_V1_V2",
    },
    3: {
        "variants":     ["V0", "V1", "V2", "V3"],
        "description":  "Value ordering: matching-guided heuristic",
        "filename":     "experiment3_V0_V1_V2_V3",
    },
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return df


def load_csv(path: Path, variant: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = coerce_numerics(df)
    df["propagator_variant"] = variant
    print(f"  [{variant}]  {len(df):4d} rows  ←  {path.name}")
    return df


def load_all(paths: dict) -> pd.DataFrame:
    """Load and concatenate all variant CSVs into one DataFrame."""
    frames = []
    for variant, path in paths.items():
        if path is not None:
            frames.append(load_csv(path, variant))
    if not frames:
        sys.exit("No CSV files loaded.")
    return pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with status == 'ok'."""
    if "status" in df.columns:
        return df[df["status"] == "ok"]
    return df


def config_label(n, k) -> str:
    return f"n={int(n)}, k={int(k)}"

# ---------------------------------------------------------------------------
# Timeout rates
# ---------------------------------------------------------------------------

def compute_timeout_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (variant, n, k), compute:
      - total instances
      - timeouts
      - timeout_rate  (fraction)
    """
    if "status" not in df.columns:
        return pd.DataFrame()

    rows = []
    for (variant, n, k), grp in df.groupby(["propagator_variant", "config_n", "config_k"]):
        total    = len(grp)
        timeouts = (grp["status"] == "timeout").sum()
        rows.append({
            "variant":      variant,
            "config_n":     n,
            "config_k":     k,
            "config":       config_label(n, k),
            "total":        total,
            "timeouts":     timeouts,
            "timeout_rate": round(timeouts / total, 4) if total > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["variant", "config_n", "config_k"])

# ---------------------------------------------------------------------------
# PAR-2 scores
# ---------------------------------------------------------------------------

def compute_par2(df: pd.DataFrame, timeout_limit: int) -> pd.DataFrame:
    """
    PAR-2 score per (variant, n, k):
      solved instances    → actual wall_time_s  (preferred)
      timed-out instances → 2 * timeout_limit

    FIX (4): Falls back to solveTime if wall_time_s is entirely absent,
    with a printed warning so the user knows which column is being used.
    """
    penalty = 2 * timeout_limit

    # Determine which time column to use for PAR-2
    if WALL_COL in df.columns and df[WALL_COL].notna().any():
        time_col = WALL_COL
    elif RUNTIME_COL in df.columns and df[RUNTIME_COL].notna().any():
        print(f"  WARNING: '{WALL_COL}' not found – using '{RUNTIME_COL}' for PAR-2."
              f" Note: solver-reported time excludes subprocess overhead.")
        time_col = RUNTIME_COL
    else:
        print(f"  WARNING: Neither '{WALL_COL}' nor '{RUNTIME_COL}' found."
              f" PAR-2 cannot be computed.")
        return pd.DataFrame()

    def par2_value(row):
        if "status" in row and row["status"] == "timeout":
            return penalty
        val = row.get(time_col, float("nan"))
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    df = df.copy()
    df["par2"] = df.apply(par2_value, axis=1)

    rows = []
    for (variant, n, k), grp in df.groupby(["propagator_variant", "config_n", "config_k"]):
        vals = grp["par2"].dropna()
        rows.append({
            "variant":       variant,
            "config_n":      n,
            "config_k":      k,
            "config":        config_label(n, k),
            "median_par2":   round(float(vals.median()), 4) if len(vals) > 0 else float("nan"),
            "n_instances":   len(grp),
            "par2_time_col": time_col,
        })
    return pd.DataFrame(rows).sort_values(["variant", "config_n", "config_k"])

# ---------------------------------------------------------------------------
# Median / IQR aggregation
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    For a single metric column, return median and IQR per (variant, n, k),
    computed over solved (status==ok) instances only.

    FIX (2): Emits a warning for any cell where the number of solved
    instances falls below MIN_SAMPLE_WARNING, since medians computed over
    very few instances are unreliable.
    """
    ok = ok_rows(df)
    if col not in ok.columns:
        return pd.DataFrame(columns=["propagator_variant", "config_n", "config_k",
                                     "median", "iqr", "n"])
    rows = []
    for (variant, n, k), grp in ok.groupby(["propagator_variant", "config_n", "config_k"]):
        vals = grp[col].dropna()
        n_solved = len(vals)
        if n_solved == 0:
            continue

        # FIX (2): warn on low effective sample size
        total_in_cell = len(df[
            (df["propagator_variant"] == variant) &
            (df["config_n"] == n) &
            (df["config_k"] == k)
        ])
        if n_solved < MIN_SAMPLE_WARNING:
            print(
                f"  WARNING: [{variant}] {config_label(n, k)} – only {n_solved}/{total_in_cell} "
                f"instances solved. Median '{col}' may be unreliable."
            )

        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        rows.append({
            "propagator_variant": variant,
            "config_n":  n,
            "config_k":  k,
            "config":    config_label(n, k),
            "median":    round(float(vals.median()), 4),
            "iqr":       round(q75 - q25, 4),
            "n":         n_solved,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def build_experiment_table(
    df: pd.DataFrame,
    col: str,
    variants: list,
    metric_label: str,
) -> pd.DataFrame:
    """
    Build a wide comparison table for one metric and a list of variants.
    Columns: config, <variant>_median, <variant>_iqr, ... for each variant.
    V0 is always included as a reference column.
    """
    agg = aggregate(df[df["propagator_variant"].isin(variants)], col)
    if agg.empty:
        return pd.DataFrame()

    all_configs = sorted(
        agg[["config_n", "config_k"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_k"]), axis=1)
    )

    rows = []
    for (n, k) in all_configs:
        row = {"config_n": n, "config_k": k, "config": config_label(n, k)}
        for v in variants:
            sub = agg[(agg["propagator_variant"] == v) &
                      (agg["config_n"] == n) &
                      (agg["config_k"] == k)]
            if len(sub) > 0:
                row[f"{v}_median"] = sub.iloc[0]["median"]
                row[f"{v}_iqr"]    = sub.iloc[0]["iqr"]
                row[f"{v}_n"]      = sub.iloc[0]["n"]
            else:
                row[f"{v}_median"] = float("nan")
                row[f"{v}_iqr"]    = float("nan")
                row[f"{v}_n"]      = 0
        rows.append(row)

    result = pd.DataFrame(rows)
    result.attrs["metric"] = metric_label
    return result


def print_experiment_table(table: pd.DataFrame, variants: list, metric: str) -> None:
    if table.empty:
        print(f"    (no data for {metric})")
        return
    print(f"\n  ── {metric} ──")
    display_cols = ["config"] + [f"{v}_median" for v in variants] + [f"{v}_iqr" for v in variants]
    display_cols = [c for c in display_cols if c in table.columns]
    print(table[display_cols].to_string(index=False))

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# FIX (1): Metrics that should use a log scale on the y-axis.
LOG_SCALE_METRICS = {"propagations", "failures", "nodes", "nogoods"}


def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"      Saved: {path.name}")
    plt.close(fig)


def plot_experiment_metric(
    df: pd.DataFrame,
    col: str,
    variants: list,
    ylabel: str,
    title: str,
    out_path: Path,
    use_log: bool = False,
) -> None:
    """
    Grouped bar chart: one group per (n,k) config, one bar per variant.
    Error bars show IQR (whiskers from Q25 to Q75).

    FIX (1): use_log=True applies a log scale to the y-axis.
    """
    agg = aggregate(df[df["propagator_variant"].isin(variants)], col)
    if agg.empty:
        print(f"      Skipping {out_path.name}: column '{col}' not found")
        return

    all_configs = sorted(
        agg[["config_n", "config_k"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_k"]), axis=1)
    )
    config_labels = [config_label(*c) for c in all_configs]
    x = np.arange(len(all_configs))
    n_variants = len(variants)
    width = 0.8 / n_variants
    offsets = np.linspace(-(n_variants - 1) / 2, (n_variants - 1) / 2, n_variants) * width

    fig, ax = plt.subplots(figsize=(max(8, len(all_configs) * 1.2), 5))

    for i, v in enumerate(variants):
        sub = agg[agg["propagator_variant"] == v].set_index(["config_n", "config_k"])
        medians = []
        iqrs    = []
        for c in all_configs:
            if c in sub.index:
                medians.append(sub.loc[c, "median"])
                iqrs.append(sub.loc[c, "iqr"] / 2)
            else:
                medians.append(0)
                iqrs.append(0)

        label = VARIANT_LABELS.get(v, v)
        color = VARIANT_COLORS.get(v, "#999999")
        ax.bar(x + offsets[i], medians, width,
               label=label, color=color, alpha=0.82)
        ax.errorbar(x + offsets[i], medians, yerr=iqrs,
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=8, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)

    # FIX (1): apply log scale where appropriate
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    else:
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    _save_fig(fig, out_path)


def plot_par2(par2_df: pd.DataFrame, variants: list, timeout_limit: int, out_path: Path) -> None:
    """Grouped bar chart of median PAR-2 scores with timeout penalty line."""
    if par2_df.empty:
        return
    sub = par2_df[par2_df["variant"].isin(variants)]
    all_configs = sorted(
        sub[["config_n", "config_k"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_k"]), axis=1)
    )
    config_labels = [config_label(*c) for c in all_configs]
    x = np.arange(len(all_configs))
    n_variants = len(variants)
    width = 0.8 / n_variants
    offsets = np.linspace(-(n_variants - 1) / 2, (n_variants - 1) / 2, n_variants) * width

    penalty = 2 * timeout_limit

    fig, ax = plt.subplots(figsize=(max(8, len(all_configs) * 1.2), 5))
    for i, v in enumerate(variants):
        vsub = sub[sub["variant"] == v].set_index(["config_n", "config_k"])
        vals = [float(vsub.loc[c, "median_par2"]) if c in vsub.index else 0
                for c in all_configs]
        ax.bar(x + offsets[i], vals, width,
               label=VARIANT_LABELS.get(v, v),
               color=VARIANT_COLORS.get(v, "#999999"), alpha=0.82)

    # Mark the timeout penalty level clearly
    ax.axhline(y=penalty, color="black", linestyle="--", linewidth=1,
               label=f"Timeout penalty ({penalty} s)")

    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=8, rotation=35, ha="right")
    ax.set_ylabel("Median PAR-2 (s)")
    ax.set_title(f"PAR-2 scores (timeout penalty = 2 × {timeout_limit} s)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    _save_fig(fig, out_path)

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    df: pd.DataFrame,
    out_dir: Path,
    timeout_limit: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Timeout rates ──────────────────────────────────────────────────
    print("\n=== Timeout rates ===")
    timeout_df = compute_timeout_rates(df)
    if not timeout_df.empty:
        print(timeout_df.to_string(index=False))
        timeout_df.to_csv(out_dir / "timeout_rates.csv", index=False)

    # ── 2. PAR-2 scores ───────────────────────────────────────────────────
    print("\n=== PAR-2 scores ===")
    par2_df = compute_par2(df, timeout_limit)
    if not par2_df.empty:
        print(par2_df.to_string(index=False))
        par2_df.to_csv(out_dir / "par2_scores.csv", index=False)

    # ── 3. Per-experiment tables and plots ────────────────────────────────
    # FIX (3): combine runtime metrics so both solveTime and wall_time_s
    # appear in tables and plots, letting the reader compare them.
    all_metrics = {}
    all_metrics.update({
        k: (col, label, note, log)
        for k, (col, label, note, log) in RUNTIME_METRICS.items()
    })
    all_metrics.update(SEARCH_METRICS)
    all_metrics.update(EXPL_METRICS)

    for exp_num, exp_info in EXPERIMENTS.items():
        exp_variants = exp_info["variants"]
        exp_desc     = exp_info["description"]
        exp_fname    = exp_info["filename"]

        present = df["propagator_variant"].unique()
        if not all(v in present for v in exp_variants):
            missing = [v for v in exp_variants if v not in present]
            print(f"\n=== Experiment {exp_num} skipped (missing variants: {missing}) ===")
            continue

        print(f"\n{'='*65}")
        print(f"  Experiment {exp_num}: {exp_desc}")
        print(f"  Variants: {exp_variants}")
        print(f"{'='*65}")

        combined_table_rows = []

        for metric_key, metric_spec in all_metrics.items():
            col, ylabel, note, use_log = metric_spec

            tbl = build_experiment_table(df, col, exp_variants,
                                         metric_label=f"{ylabel} ({note})")
            print_experiment_table(tbl, exp_variants, f"{ylabel} ({note})")

            if not tbl.empty:
                tbl.insert(0, "metric", ylabel)
                combined_table_rows.append(tbl)

            if HAS_MPL:
                out_path = plot_dir / f"exp{exp_num}_{metric_key}.png"
                plot_experiment_metric(
                    df, col, exp_variants,
                    ylabel=f"Median {ylabel}",
                    title=f"Exp {exp_num} – {ylabel}  ({note})\n{exp_desc}",
                    out_path=out_path,
                    use_log=use_log,
                )

        # PAR-2 plot per experiment
        if HAS_MPL and not par2_df.empty:
            plot_par2(par2_df, exp_variants, timeout_limit,
                      out_path=plot_dir / f"exp{exp_num}_par2.png")

        if combined_table_rows:
            combined = pd.concat(combined_table_rows, ignore_index=True)
            combined.to_csv(out_dir / f"{exp_fname}.csv", index=False)
            print(f"\n  Saved: {exp_fname}.csv")

    print("\nAnalysis complete.")
    print(f"Output directory: {out_dir}")


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------

def discover_latest_csvs(results_dir: Path) -> dict:
    """
    For each variant, find the most recent stats_<VARIANT>_*.csv in results_dir.
    Returns a dict {variant: path_or_None}.
    """
    found = {}
    for v in VARIANTS:
        matches = sorted(results_dir.glob(f"stats_{v}_*.csv"))
        found[v] = matches[-1] if matches else None
    return found

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse circuit propagator benchmark results (four variants).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--v0",      type=Path, default=None, help="CSV for V0")
    p.add_argument("--v1",      type=Path, default=None, help="CSV for V1")
    p.add_argument("--v2",      type=Path, default=None, help="CSV for V2")
    p.add_argument("--v3",      type=Path, default=None, help="CSV for V3")
    p.add_argument("--out",     type=Path, default=RESULTS_DIR,
                   help="Root output directory")
    p.add_argument("--timeout", type=int,  default=TIMEOUT_DEFAULT,
                   help="Timeout limit used during experiments (for PAR-2)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_plots:
        global HAS_MPL
        HAS_MPL = False

    explicit = {"V0": args.v0, "V1": args.v1, "V2": args.v2, "V3": args.v3}
    if any(v is not None for v in explicit.values()):
        paths = explicit
    else:
        print("No explicit CSVs provided – auto-discovering latest per variant...")
        paths = discover_latest_csvs(RESULTS_DIR)

    print("\nLoading CSVs:")
    available = {v: p for v, p in paths.items() if p is not None}
    if not available:
        sys.exit(f"No stats_V*_*.csv files found in {RESULTS_DIR}. "
                 "Run experiments first or provide --v0 / --v1 / --v2 / --v3.")

    for v, p in paths.items():
        if p is None:
            print(f"  [{v}]  not provided / not found – will be skipped")

    df = load_all(available)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out / f"analysis_{ts}"

    print(f"\n{'='*65}")
    print(f"  Total rows loaded: {len(df)}")
    print(f"  Variants present:  {sorted(df['propagator_variant'].unique())}")
    print(f"  Output:            {out_dir}")
    print(f"{'='*65}")

    run_analysis(df, out_dir, args.timeout)


if __name__ == "__main__":
    main()