"""
run_experiments.py  –  Circuit propagator benchmark runner
===========================================================

Builds the solver once with `cargo build --release`, then generates
MiniZinc instances and solves each with:

    minizinc --solver pumpkin-circuit <instance.mzn> -s

and saves per-run statistics to a CSV file.

Directory layout inside the Pumpkin project root
-------------------------------------------------
experiments/
├── generate_instances.py
├── run_experiments.py          (this file)
├── instances/
│   └── sat/
│       └── n<N>_k<K>/
└── results/
    └── stats_<timestamp>.csv

Usage (from the Pumpkin project root)
--------------------------------------
    python experiments/run_experiments.py [options]

Options
-------
  --nodes    INT [INT ...]        Node counts              (default: 10 20 30 40 50)
  --neighbours INT [INT ...]      k values                 (default: 5 7 10)
  --instances-per-config INT      Instances per (n,k) pair (default: 5)
  --seed     INT                  Base seed                (default: 42)
  --timeout  INT                  Per-instance timeout (s) (default: 60)
  --outdir   PATH                 Results directory
  --no-generate                   Skip generation; use existing .mzn files
  --solver   STR                  MiniZinc solver name     (default: pumpkin-circuit)
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).parent
PROJECT_ROOT  = SCRIPT_DIR.parent
SAT_GENERATOR = SCRIPT_DIR / "generate_instances.py"
INSTANCE_DIR  = SCRIPT_DIR / "instances"
RESULTS_DIR   = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, cwd, timeout=None):
    result = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def parse_statistics(raw: str) -> dict:
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
# Step 1 – Build solver binary
# ---------------------------------------------------------------------------

def build_solver() -> None:
    """
    Run `cargo build --release -p pumpkin-solver` once before the experiment
    loop so the binary is up-to-date and Cargo startup overhead does not
    contaminate per-instance wall-clock timings.
    """
    cmd = ["cargo", "build", "--release", "-p", "pumpkin-solver"]
    print("\nBuilding pumpkin-solver (release)...")
    stdout, stderr, rc = run(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"ERROR: cargo build failed:\n{stderr}", file=sys.stderr)
        sys.exit(1)
    print("  Build OK.\n")


# ---------------------------------------------------------------------------
# Step 2 – Generate instances
# ---------------------------------------------------------------------------

def generate_sat_instances(n, k, count, seed, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SAT_GENERATOR),
        "-n", str(n), "-k", str(k), "-c", str(count),
        "-s", str(seed), "-o", str(out_dir), "--prefix", "instance",
    ]
    print(f"  [generate] n={n} k={k} -> {out_dir}")
    _, stderr, rc = run(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"    ERROR: {stderr}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 2 – Solve with a single minizinc call
# ---------------------------------------------------------------------------

def solve_mzn(mzn_path: Path, solver: str, timeout: int) -> dict:
    rel_mzn = mzn_path.relative_to(PROJECT_ROOT)
    cmd = ["minizinc", "--solver", solver, str(rel_mzn), "-s"]

    print(f"    [solve]  {rel_mzn.name}", end="  ")
    wall_start = time.perf_counter()

    try:
        stdout, stderr, rc = run(cmd, cwd=PROJECT_ROOT, timeout=timeout)
        wall_time = time.perf_counter() - wall_start
        timed_out = False
    except subprocess.TimeoutExpired:
        wall_time = timeout
        timed_out = True
        stdout, stderr, rc = "", "TIMEOUT", -1

    status = "timeout" if timed_out else ("ok" if rc == 0 else "error")
    print(f"{status}  ({wall_time:.3f}s)")

    # MiniZinc prints stats to both stdout and stderr depending on version;
    # parse both to be safe.
    stats = parse_statistics(stdout + stderr)
    stats["wall_time_s"]  = f"{wall_time:.6f}"
    stats["status"]       = status
    stats["return_code"]  = str(rc)
    return stats


# ---------------------------------------------------------------------------
# Step 3 – Persist results
# ---------------------------------------------------------------------------

FIXED_COLUMNS = [
    "instance_type",
    "config_n", "config_k",
    "instance_index", "instance_seed",
    "mzn_file",
    "status", "return_code", "wall_time_s",
]


def save_results(rows, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = results_dir / f"stats_{timestamp}.csv"

    extra_cols = sorted({k for row in rows for k in row if k not in FIXED_COLUMNS})
    all_cols   = FIXED_COLUMNS + extra_cols

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n  Results written to: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Per-configuration runner
# ---------------------------------------------------------------------------

def run_config(
    n: int, k: int,
    config_dir: Path,
    count: int,
    solver: str,
    timeout: int,
    seed: int,
) -> list[dict]:
    rows     = []
    mzn_files = sorted(config_dir.glob("instance_*.mzn"))[:count]

    if not mzn_files:
        print(f"  No .mzn files found in {config_dir}, skipping.")
        return rows

    for idx, mzn_path in enumerate(mzn_files):
        print(f"\n  Instance {idx} : {mzn_path.name}")

        row = {
            "instance_type":  "sat",
            "config_n":       n,
            "config_k":       k,
            "instance_index": idx,
            "instance_seed":  seed + idx,
            "mzn_file":       str(mzn_path.relative_to(PROJECT_ROOT)),
        }

        stats = solve_mzn(mzn_path, solver, timeout)
        row.update(stats)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nodes",                type=int, nargs="+", default=[10, 20, 30, 40, 50])
    p.add_argument("--neighbours",           type=int, nargs="+", default=[5, 7, 10])
    p.add_argument("--instances-per-config", type=int, default=5)
    p.add_argument("--seed",                 type=int, default=42)
    p.add_argument("--timeout",              type=int, default=60)
    p.add_argument("--outdir",               type=Path, default=RESULTS_DIR)
    p.add_argument("--no-generate",          action="store_true")
    p.add_argument("--solver",               type=str, default="pumpkin-circuit",
                   help="MiniZinc solver name passed to --solver")
    return p.parse_args()


def main():
    args = parse_args()

    build_solver()

    all_rows = []

    for n in args.nodes:
        for k in args.neighbours:
            print(f"\n{'='*60}")
            print(f"n={n}, k={k}")
            print(f"{'='*60}")

            config_dir = INSTANCE_DIR / "sat" / f"n{n}_k{k}"

            if not args.no_generate:
                generate_sat_instances(
                    n, k, args.instances_per_config, args.seed, config_dir
                )

            rows = run_config(
                n=n, k=k,
                config_dir=config_dir,
                count=args.instances_per_config,
                solver=args.solver,
                timeout=args.timeout,
                seed=args.seed,
            )
            all_rows.extend(rows)

    if all_rows:
        save_results(all_rows, args.outdir)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()