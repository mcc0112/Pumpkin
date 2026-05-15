"""
run_experiments.py  –  Circuit propagator benchmark runner
===========================================================

Generates MiniZinc instances, converts each to FlatZinc via the Pumpkin
MiniZinc back-end, solves each with pumpkin-solver, and saves per-run
statistics to a CSV file.

Directory layout inside the Pumpkin project root
-------------------------------------------------
experiments/
├── generate_instances.py       (your existing generator – copy it here)
├── run_experiments.py          (this file)
├── analyse_results.py          (data-analysis helper)
├── instances/
│   └── n<N>_k<K>/
│       ├── instance_n<N>_k<K>_0000.mzn
│       ├── instance_n<N>_k<K>_0000.fzn
│       └── ...
└── results/
    └── stats_<timestamp>.csv

Usage (from the Pumpkin project root)
--------------------------------------
    python experiments/run_experiments.py [options]

Options
-------
  --nodes    INT [INT ...]   Node counts to test       (default: 20 50 100)
  --neighbours INT [INT ...] Neighbour counts to test  (default: 5 7 10)
  --instances-per-config INT Number of instances per (n,k) pair (default: 3)
  --seed     INT             Base seed                 (default: 42)
  --timeout  INT             Per-instance timeout in seconds (default: 300)
  --outdir   PATH            Where to write results CSV (default: experiments/results)
  --no-generate              Skip generation; only run solver on existing .fzn files
  --release                  Use cargo --release for faster solving
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the Pumpkin project root)
# ---------------------------------------------------------------------------

# This script lives at experiments/run_experiments.py
SCRIPT_DIR   = Path(__file__).parent          # .../experiments/
PROJECT_ROOT = SCRIPT_DIR.parent              # Pumpkin repo root
GENERATOR    = SCRIPT_DIR / "generate_instances.py"
INSTANCE_DIR = SCRIPT_DIR / "instances"
RESULTS_DIR  = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path, timeout: int | None = None) -> tuple[str, str, int]:
    """Run *cmd* in *cwd*, return (stdout, stderr, returncode)."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def parse_statistics(raw: str) -> dict:
    """
    Extract key=value statistics from pumpkin-solver -s output.

    The solver prints lines like:
        %%%mzn-stat: failures=123
        %%%mzn-stat: solveTime=4.56
        %%%mzn-stat-end
    We collect every such line into a flat dict.
    """
    stats = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("%%%mzn-stat:"):
            body = line[len("%%%mzn-stat:"):].strip()
            if "=" in body:
                key, _, value = body.partition("=")
                stats[key.strip()] = value.strip()
    return stats


# ---------------------------------------------------------------------------
# Step 1 – Generate .mzn instances
# ---------------------------------------------------------------------------

def generate_instances(n: int, k: int, count: int, seed: int) -> Path:
    """Generate *count* instances for the (n, k) configuration."""
    out_dir = INSTANCE_DIR / f"n{n}_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(GENERATOR),
        "-n", str(n),
        "-k", str(k),
        "-c", str(count),
        "-s", str(seed),
        "-o", str(out_dir),
        "--prefix", "instance",
    ]
    print(f"  [generate] n={n} k={k}  →  {out_dir}")
    stdout, stderr, rc = run(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"    ERROR generating instances:\n{stderr}", file=sys.stderr)
    return out_dir


# ---------------------------------------------------------------------------
# Step 2 – Convert .mzn → .fzn
# ---------------------------------------------------------------------------

def convert_to_fzn(mzn_path: Path) -> Path | None:
    """
    Run:  minizinc --solver pumpkin -c <mzn_path>
    from the project root. Produces a .fzn next to the .mzn.
    Returns the path to the .fzn, or None on failure.
    """
    fzn_path = mzn_path.with_suffix(".fzn")

    # Path relative to project root (required by Pumpkin's MiniZinc set-up)
    rel_mzn = mzn_path.relative_to(PROJECT_ROOT)

    cmd = [
        "minizinc",
        "--solver", "pumpkin",
        "-c",                     # compile only (produce FlatZinc)
        str(rel_mzn),
    ]
    print(f"    [mzn→fzn] {rel_mzn.name}", end="  ")
    stdout, stderr, rc = run(cmd, cwd=PROJECT_ROOT)

    if rc != 0 or not fzn_path.exists():
        print(f"FAILED (rc={rc})")
        print(f"      stderr: {stderr[:300]}", file=sys.stderr)
        return None

    print(f"OK  →  {fzn_path.name}")
    return fzn_path


# ---------------------------------------------------------------------------
# Step 3 – Solve .fzn with pumpkin-solver
# ---------------------------------------------------------------------------

def solve_fzn(
    fzn_path: Path,
    timeout: int,
    use_release: bool,
) -> dict:
    """
    Run:  cargo run -p pumpkin-solver [--release] -- <fzn_path> -s
    from the project root.

    Returns a dict with timing info plus whatever statistics the solver emits.
    """
    rel_fzn = fzn_path.relative_to(PROJECT_ROOT)

    cargo_cmd = ["cargo", "run", "-p", "pumpkin-solver"]
    if use_release:
        cargo_cmd.append("--release")
    cargo_cmd += ["--", str(rel_fzn), "-s"]

    print(f"    [solve]   {rel_fzn.name}", end="  ")
    wall_start = time.perf_counter()

    try:
        stdout, stderr, rc = run(cargo_cmd, cwd=PROJECT_ROOT, timeout=timeout)
        wall_time = time.perf_counter() - wall_start
        timed_out = False
    except subprocess.TimeoutExpired:
        wall_time = timeout
        timed_out = True
        stdout, stderr, rc = "", "TIMEOUT", -1

    status = "timeout" if timed_out else ("ok" if rc == 0 else "error")
    print(f"{status}  ({wall_time:.1f}s)")

    stats = parse_statistics(stdout + stderr)
    stats["wall_time_s"] = f"{wall_time:.3f}"
    stats["status"]      = status
    stats["return_code"] = str(rc)

    return stats


# ---------------------------------------------------------------------------
# Step 4 – Persist results to CSV
# ---------------------------------------------------------------------------

FIXED_COLUMNS = [
    "config_n",
    "config_k",
    "instance_index",
    "instance_seed",
    "mzn_file",
    "fzn_file",
    "status",
    "return_code",
    "wall_time_s",
]

def save_results(rows: list[dict], results_dir: Path) -> Path:
    """Write all rows to a timestamped CSV file. Returns the file path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = results_dir / f"stats_{timestamp}.csv"

    # Collect all column names seen across all rows
    extra_cols = sorted({
        k for row in rows for k in row
        if k not in FIXED_COLUMNS
    })
    all_cols = FIXED_COLUMNS + extra_cols

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n  Results written to: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate instances and run pumpkin-solver benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nodes",      type=int, nargs="+", default=[20, 50, 100],
                   help="Node counts to test")
    p.add_argument("--neighbours", type=int, nargs="+", default=[5, 7, 10],
                   help="Nearest-neighbour counts to test")
    p.add_argument("--instances-per-config", type=int, default=3,
                   help="Instances to generate per (n, k) combination")
    p.add_argument("--seed",    type=int, default=42,
                   help="Base seed; each (n, k, i) gets seed + i")
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-instance solver timeout in seconds")
    p.add_argument("--outdir",  type=Path, default=RESULTS_DIR,
                   help="Directory for the output CSV")
    p.add_argument("--no-generate", action="store_true",
                   help="Skip instance generation; use existing .fzn files")
    p.add_argument("--release", action="store_true",
                   help="Build pumpkin-solver with --release")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict] = []

    for n in args.nodes:
        for k in args.neighbours:
            print(f"\n{'='*60}")
            print(f"Configuration  n={n}, k={k}")
            print(f"{'='*60}")

            config_dir = INSTANCE_DIR / f"n{n}_k{k}"

            # ---- Generate -----------------------------------------------
            if not args.no_generate:
                generate_instances(n, k, args.instances_per_config, args.seed)

            # ---- Find .mzn files ----------------------------------------
            mzn_files = sorted(config_dir.glob("instance_*.mzn"))
            if not mzn_files:
                print(f"  No .mzn files found in {config_dir}, skipping.")
                continue

            # Use only the first `instances_per_config` files
            mzn_files = mzn_files[: args.instances_per_config]

            for idx, mzn_path in enumerate(mzn_files):
                print(f"\n  Instance {idx} : {mzn_path.name}")

                # ---- MZN → FZN ------------------------------------------
                fzn_path = convert_to_fzn(mzn_path)

                row: dict = {
                    "config_n":       n,
                    "config_k":       k,
                    "instance_index": idx,
                    "instance_seed":  args.seed + idx,
                    "mzn_file":       str(mzn_path.relative_to(PROJECT_ROOT)),
                    "fzn_file":       str(fzn_path.relative_to(PROJECT_ROOT)) if fzn_path else "",
                }

                if fzn_path is None:
                    row["status"] = "fzn_conversion_failed"
                    all_rows.append(row)
                    continue

                # ---- Solve -----------------------------------------------
                stats = solve_fzn(fzn_path, args.timeout, args.release)
                row.update(stats)
                all_rows.append(row)

    # ---- Save ---------------------------------------------------------------
    if all_rows:
        save_results(all_rows, args.outdir)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()