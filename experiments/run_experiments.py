"""
run_experiments.py  –  Circuit propagator benchmark runner
===========================================================

Generates MiniZinc instances (SAT and/or UNSAT), converts each to FlatZinc,
solves with pumpkin-solver, and saves per-run statistics to a CSV file.

Key improvements over the original:
  - Builds the solver binary ONCE with `cargo build` before the experiment
    loop, so wall-clock timing is not contaminated by Cargo startup overhead.
  - Supports UNSAT instance generation via generate_unsat_instances.py.
  - Adds an `instance_type` column (sat / unsat_random / unsat_forced) to
    the output CSV so SAT and UNSAT results can be analysed separately.

Directory layout inside the Pumpkin project root
-------------------------------------------------
experiments/
├── generate_instances.py
├── generate_unsat_instances.py
├── run_experiments.py          (this file)
├── instances/
│   ├── sat/
│   │   └── n<N>_k<K>/
│   └── unsat/
│       ├── random/
│       │   └── n<N>_k<K>/
│       └── forced/
│           └── n<N>_k<K>/
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
  --no-generate                   Skip generation; use existing .fzn files
  --release                       Build solver with --release (recommended)
  --run-sat                       Run SAT instances         (default: true)
  --run-unsat-random              Run UNSAT random instances
  --run-unsat-forced              Run UNSAT forced-partition instances
  --unsat-nodes INT [INT ...]     Node counts for UNSAT    (default: 20 30)
  --unsat-neighbours INT [INT ...]  k values for UNSAT     (default: 5 7)
  --unsat-per-config INT          UNSAT instances per (n,k)(default: 5)
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

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SAT_GENERATOR   = SCRIPT_DIR / "generate_instances.py"
UNSAT_GENERATOR = SCRIPT_DIR / "generate_unsat_instances.py"
INSTANCE_DIR = SCRIPT_DIR / "instances"
RESULTS_DIR  = SCRIPT_DIR / "results"

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
# Build solver binary once
# ---------------------------------------------------------------------------

def build_solver(use_release: bool) -> Path:
    """
    Run `cargo build -p pumpkin-solver [--release]` once and return the
    path to the compiled binary.  This avoids per-instance Cargo startup
    overhead contaminating wall-clock timing measurements.
    """
    cmd = ["cargo", "build", "-p", "pumpkin-solver"]
    if use_release:
        cmd.append("--release")

    profile = "release" if use_release else "debug"
    binary  = PROJECT_ROOT / "target" / profile / "pumpkin-solver"

    print(f"\nBuilding pumpkin-solver ({profile})...")
    stdout, stderr, rc = run(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"ERROR: cargo build failed:\n{stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"  Binary: {binary}")
    return binary


# ---------------------------------------------------------------------------
# Step 1 – Generate instances
# ---------------------------------------------------------------------------

def generate_sat_instances(n, k, count, seed, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SAT_GENERATOR),
        "-n", str(n), "-k", str(k), "-c", str(count),
        "-s", str(seed), "-o", str(out_dir), "--prefix", "instance",
    ]
    print(f"  [generate SAT] n={n} k={k} -> {out_dir}")
    _, stderr, rc = run(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"    ERROR: {stderr}", file=sys.stderr)


def generate_unsat_instances(n, k, count, seed, mode, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(UNSAT_GENERATOR),
        "-n", str(n), "-k", str(k), "-c", str(count),
        "-s", str(seed), "-o", str(out_dir),
        "--unsat-mode", mode, "--prefix", "unsat",
    ]
    print(f"  [generate UNSAT/{mode}] n={n} k={k} -> {out_dir}")
    _, stderr, rc = run(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"    ERROR: {stderr}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 2 – MZN -> FZN
# ---------------------------------------------------------------------------

def convert_to_fzn(mzn_path: Path) -> Path | None:
    fzn_path = mzn_path.with_suffix(".fzn")
    rel_mzn  = mzn_path.relative_to(PROJECT_ROOT)
    cmd = [
        "minizinc", "--solver", "pumpkin",
        "-c", "--no-output-ozn", str(rel_mzn),
    ]
    print(f"    [mzn->fzn] {rel_mzn.name}", end="  ")
    _, stderr, rc = run(cmd, cwd=PROJECT_ROOT)
    if rc != 0 or not fzn_path.exists():
        print(f"FAILED (rc={rc})")
        print(f"      stderr: {stderr[:300]}", file=sys.stderr)
        return None
    print(f"OK -> {fzn_path.name}")
    return fzn_path


# ---------------------------------------------------------------------------
# Step 3 – Solve using pre-built binary (no Cargo overhead)
# ---------------------------------------------------------------------------

def solve_fzn(fzn_path: Path, binary: Path, timeout: int) -> dict:
    rel_fzn = fzn_path.relative_to(PROJECT_ROOT)
    cmd = [str(binary), str(rel_fzn), "-s"]

    print(f"    [solve]   {rel_fzn.name}", end="  ")
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

    stats = parse_statistics(stdout + stderr)
    stats["wall_time_s"] = f"{wall_time:.6f}"
    stats["status"]      = status
    stats["return_code"] = str(rc)
    return stats


# ---------------------------------------------------------------------------
# Step 4 – Persist results
# ---------------------------------------------------------------------------

FIXED_COLUMNS = [
    "instance_type",
    "config_n", "config_k",
    "instance_index", "instance_seed",
    "mzn_file", "fzn_file",
    "status", "return_code", "wall_time_s",
]


def save_results(rows, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = results_dir / f"stats_{timestamp}.csv"

    extra_cols = sorted({k for row in rows for k in row if k not in FIXED_COLUMNS})
    all_cols = FIXED_COLUMNS + extra_cols

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
    instance_type: str,
    n: int, k: int,
    config_dir: Path,
    prefix: str,
    count: int,
    binary: Path,
    timeout: int,
    seed: int,
) -> list[dict]:
    rows = []
    mzn_files = sorted(config_dir.glob(f"{prefix}_*.mzn"))[:count]

    if not mzn_files:
        print(f"  No .mzn files found in {config_dir}, skipping.")
        return rows

    for idx, mzn_path in enumerate(mzn_files):
        print(f"\n  Instance {idx} : {mzn_path.name}")

        fzn_path = convert_to_fzn(mzn_path)
        row = {
            "instance_type":  instance_type,
            "config_n":       n,
            "config_k":       k,
            "instance_index": idx,
            "instance_seed":  seed + idx,
            "mzn_file":       str(mzn_path.relative_to(PROJECT_ROOT)),
            "fzn_file":       str(fzn_path.relative_to(PROJECT_ROOT)) if fzn_path else "",
        }

        if fzn_path is None:
            row["status"] = "fzn_conversion_failed"
        else:
            stats = solve_fzn(fzn_path, binary, timeout)
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
    p.add_argument("--nodes",      type=int, nargs="+", default=[10, 20, 30, 40, 50])
    p.add_argument("--neighbours", type=int, nargs="+", default=[5, 7, 10])
    p.add_argument("--instances-per-config", type=int, default=5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--timeout",    type=int, default=60)
    p.add_argument("--outdir",     type=Path, default=RESULTS_DIR)
    p.add_argument("--no-generate", action="store_true")
    p.add_argument("--release",    action="store_true")

    # UNSAT controls
    p.add_argument("--run-sat",           action="store_true", default=True)
    p.add_argument("--run-unsat-random",  action="store_true", default=False)
    p.add_argument("--run-unsat-forced",  action="store_true", default=False)
    p.add_argument("--unsat-nodes",       type=int, nargs="+", default=[20, 30])
    p.add_argument("--unsat-neighbours",  type=int, nargs="+", default=[5, 7])
    p.add_argument("--unsat-per-config",  type=int, default=5)

    return p.parse_args()


def main():
    args = parse_args()

    # Build solver binary once — critical for clean timing
    binary = build_solver(args.release)

    all_rows = []

    # ------------------------------------------------------------------ SAT
    if args.run_sat:
        for n in args.nodes:
            for k in args.neighbours:
                print(f"\n{'='*60}")
                print(f"SAT  n={n}, k={k}")
                print(f"{'='*60}")
                config_dir = INSTANCE_DIR / "sat" / f"n{n}_k{k}"
                if not args.no_generate:
                    generate_sat_instances(n, k, args.instances_per_config,
                                           args.seed, config_dir)
                rows = run_config(
                    instance_type="sat",
                    n=n, k=k,
                    config_dir=config_dir,
                    prefix="instance",
                    count=args.instances_per_config,
                    binary=binary,
                    timeout=args.timeout,
                    seed=args.seed,
                )
                all_rows.extend(rows)

    # ----------------------------------------------------------- UNSAT random
    if args.run_unsat_random:
        for n in args.unsat_nodes:
            for k in args.unsat_neighbours:
                print(f"\n{'='*60}")
                print(f"UNSAT/random  n={n}, k={k}")
                print(f"{'='*60}")
                config_dir = INSTANCE_DIR / "unsat" / "random" / f"n{n}_k{k}"
                if not args.no_generate:
                    generate_unsat_instances(n, k, args.unsat_per_config,
                                             args.seed, "random", config_dir)
                rows = run_config(
                    instance_type="unsat_random",
                    n=n, k=k,
                    config_dir=config_dir,
                    prefix="unsat",
                    count=args.unsat_per_config,
                    binary=binary,
                    timeout=args.timeout,
                    seed=args.seed,
                )
                all_rows.extend(rows)

    # ----------------------------------------------------------- UNSAT forced
    if args.run_unsat_forced:
        for n in args.unsat_nodes:
            for k in args.unsat_neighbours:
                print(f"\n{'='*60}")
                print(f"UNSAT/forced  n={n}, k={k}")
                print(f"{'='*60}")
                config_dir = INSTANCE_DIR / "unsat" / "forced" / f"n{n}_k{k}"
                if not args.no_generate:
                    generate_unsat_instances(n, k, args.unsat_per_config,
                                             args.seed, "forced", config_dir)
                rows = run_config(
                    instance_type="unsat_forced",
                    n=n, k=k,
                    config_dir=config_dir,
                    prefix="unsat",
                    count=args.unsat_per_config,
                    binary=binary,
                    timeout=args.timeout,
                    seed=args.seed,
                )
                all_rows.extend(rows)

    # ------------------------------------------------------------------ Save
    if all_rows:
        save_results(all_rows, args.outdir)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()