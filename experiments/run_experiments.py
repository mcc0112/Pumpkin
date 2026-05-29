"""
run_experiments.py  –  Circuit propagator benchmark runner
===========================================================

Builds the solver once with `cargo build --release`, then generates
geographic k-nearest MiniZinc instances and solves each instance with
all four propagator variants (V0–V3) in turn.

Each variant run produces its own stats CSV tagged with the variant name,
so that analyse_results.py can load them independently.

Parameter grid (Table 1 of the paper)
--------------------------------------
  n \\ k    3    5    7   10   15   20
  ------  ---  ---  ---  ---  ---  ---
    20    30   30   30   30    –    –
    50    25   25   25   25   25    –
   100     –   20   20   20   20   20
   150     –    –   20   20   20   20

Cells marked – are excluded (near-degenerate at large n/small k, or
near-complete graph at small n/large k).

Per-cell instance counts are chosen proportional to expected solve-time
variance: more instances where variance is high (hard, sparse cells),
fewer where variance is low (easy, dense cells).

Directory layout
-----------------
experiments/
├── generate_instances.py
├── run_experiments.py          (this file)
├── analyse_results.py
├── instances/
│   └── sat/
│       └── n<N>_k<K>/
│           └── instance_n<N>_k<K>_<idx>.mzn
└── results/
    └── stats_<variant>_<timestamp>.csv

Usage (from the Pumpkin project root)
--------------------------------------
    python experiments/run_experiments.py [options]

Options
-------
  --variant  STR         Variant label for this run: V0, V1, V2, or V3.
                         Rebuild the solver on the correct branch first.
                         If omitted you will be prompted.
  --timeout  INT         Per-instance timeout in seconds  (default: 300)
  --seed     INT         Base random seed for instance generation (default: 42)
                         Each (n,k) cell derives a unique seed from this value
                         so instances are structurally independent across cells.
  --outdir   PATH        Results directory
  --no-generate          Skip instance generation; use existing .mzn files
  --no-build             Skip cargo build (use if binary is already up-to-date)
  --workers  INT         Parallel worker processes (default: all logical cores)
  --solver   STR         MiniZinc solver name as registered  (default: pumpkin-circuit)
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
GENERATOR    = SCRIPT_DIR / "generate_instances.py"
INSTANCE_DIR = SCRIPT_DIR / "instances"
RESULTS_DIR  = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

# Full grid: maps (n, k) -> instance count.
# Excluded cells are simply absent from this dict.
PARAMETER_GRID: dict[tuple[int, int], int] = {
    # n=20
    (20,  3): 30,
    (20,  5): 30,
    (20,  7): 30,
    (20, 10): 30,
    # (20, 15) excluded – near-complete graph for small n
    # (20, 20) excluded – near-complete graph for small n

    # n=50
    (50,  3): 25,
    (50,  5): 25,
    (50,  7): 25,
    (50, 10): 25,
    (50, 15): 25,
    # (50, 20) excluded – near-complete graph for small n

    # n=100
    # (100, 3) excluded – near-degenerate (walk dominates at large n/small k)
    (100,  5): 20,
    (100,  7): 20,
    (100, 10): 20,
    (100, 15): 20,
    (100, 20): 20,

    # n=150
    # (150, 3) excluded – near-degenerate
    # (150, 5) excluded – near-degenerate
    (150,  7): 20,
    (150, 10): 20,
    (150, 15): 20,
    (150, 20): 20,
}

# Variants are purely labels — the solver binary is always the same.
# You switch variant by rebuilding on a different git branch, then pass
# --variant V1 (or whichever) so the CSV is labelled correctly.
DEFAULT_VARIANTS = ["V0", "V1", "V2", "V3"]
DEFAULT_SOLVER   = "pumpkin-circuit"
DEFAULT_TIMEOUT  = 300

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cell_base_seed(global_seed: int, n: int, k: int) -> int:
    """
    Derive a unique, reproducible base seed for each (n, k) cell.

    Using global_seed + idx would cause every cell to share the same
    seed-0 instance layout.  Instead we hash (global_seed, n, k) into a
    4-byte integer so seeds never overlap across cells while still being
    fully reproducible from a single --seed value.
    """
    import hashlib
    tag = f"{global_seed}_{n}_{k}".encode()
    digest = hashlib.md5(tag).digest()
    return int.from_bytes(digest[:4], "big")


def run_subprocess(cmd, cwd, timeout=None):
    """
    Run a subprocess with an optional timeout.

    Windows-safe implementation.  The standard subprocess.run(timeout=)
    pattern hangs on Windows because its exception handler calls
    communicate() a second time while the child is still alive, and the
    background reader threads never finish.

    Fix: use Popen + communicate(timeout=) ourselves.  On TimeoutExpired
    we kill the process and then close the pipes directly instead of
    calling communicate() again — that avoids waiting for threads that
    will never finish because their underlying handles are gone.
    """
    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout, stderr, proc.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
        if proc.stdout: proc.stdout.close()
        if proc.stderr: proc.stderr.close()
        try:
            proc.wait(timeout=5)   # don't wait forever
        except subprocess.TimeoutExpired:
            pass
        raise      # re-raise TimeoutExpired so solve_mzn can handle it


def parse_statistics(raw: str) -> dict:
    """Parse %%%mzn-stat: key=value lines from solver output."""
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
    Run cargo build --release once so the binary is up-to-date and Cargo
    startup overhead does not contaminate per-instance wall-clock timings.
    """
    cmd = ["cargo", "build", "--release", "-p", "pumpkin-solver"]
    print("\nBuilding pumpkin-solver (release)...")
    stdout, stderr, rc = run_subprocess(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"ERROR: cargo build failed:\n{stderr}", file=sys.stderr)
        sys.exit(1)
    print("  Build OK.\n")


# ---------------------------------------------------------------------------
# Step 2 – Generate instances
# ---------------------------------------------------------------------------

def generate_instances(n: int, k: int, count: int, seed: int, out_dir: Path) -> None:
    """
    Generate `count` instances for the (n, k) cell using `seed` as the
    base seed.  The generator script increments the seed per instance, so
    instance i gets seed `seed + i`, giving distinct layouts across seeds.

    A 60-second timeout guards against the generator hanging (e.g. on a
    degenerate graph where the random walk never terminates).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(GENERATOR),
        "-n", str(n), "-k", str(k), "-c", str(count),
        "-s", str(seed), "-o", str(out_dir), "--prefix", "instance",
    ]
    print(f"  [generate] n={n} k={k} seed={seed} count={count} -> {out_dir}")
    try:
        _, stderr, rc = run_subprocess(cmd, cwd=PROJECT_ROOT, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"    ERROR: generator timed out for n={n} k={k}", file=sys.stderr)
        return
    if rc != 0:
        print(f"    WARNING: generator returned non-zero exit code:\n{stderr}",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 3 – Solve one instance with one variant
# ---------------------------------------------------------------------------

def solve_mzn(
    mzn_path: Path,
    solver_name: str,
    timeout: int,
) -> dict:
    """
    Invoke MiniZinc on a single instance and return a statistics dict.
    Wall-clock time is measured around the subprocess call; compilation
    time is excluded because we pass --no-optimize and let the solver
    report solveTime separately via %%%mzn-stat lines.
    """
    rel_mzn = mzn_path.relative_to(PROJECT_ROOT)
    # -s requests statistics output; --no-optimize skips MiniZinc-level
    # presolve so that reported solveTime reflects solver work only.
    cmd = ["minizinc", "--solver", solver_name, str(rel_mzn), "-s", "--no-optimize"]

    wall_start = time.perf_counter()
    try:
        stdout, stderr, rc = run_subprocess(cmd, cwd=PROJECT_ROOT, timeout=timeout)
        wall_time = time.perf_counter() - wall_start
        timed_out = False
    except subprocess.TimeoutExpired:
        wall_time = timeout
        timed_out = True
        stdout, stderr, rc = "", "TIMEOUT", -1

    status = "timeout" if timed_out else ("ok" if rc == 0 else "error")

    # MiniZinc may print stats to stdout or stderr depending on version
    stats = parse_statistics(stdout + stderr)
    stats["wall_time_s"]  = f"{wall_time:.6f}"
    stats["status"]       = status
    stats["return_code"]  = str(rc)
    return stats


# ---------------------------------------------------------------------------
# Step 4 – Solve one instance (top-level so it is picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _solve_one(args_tuple):
    """
    Worker function called by the process pool.
    Must be a top-level function (not a lambda or nested def) so that
    Python's multiprocessing can pickle it on Windows.

    instance_seed = cell_seed + idx, matching exactly what generate_instances
    passed to the generator script, so the CSV records the true seed used.
    """
    n, k, variant, cell_seed, idx, mzn_path_str, solver_name, timeout = args_tuple
    mzn_path = Path(mzn_path_str)
    row = {
        "propagator_variant": variant,
        "instance_type":      "sat",
        "config_n":           n,
        "config_k":           k,
        "instance_index":     idx,
        "instance_seed":      cell_seed + idx,
        "mzn_file":           str(mzn_path.relative_to(PROJECT_ROOT)),
    }
    stats = solve_mzn(mzn_path, solver_name, timeout)
    row.update(stats)
    return row


# ---------------------------------------------------------------------------
# Step 5 – Run one (n, k, variant) configuration in parallel
# ---------------------------------------------------------------------------

def run_config(
    n: int,
    k: int,
    variant: str,
    solver_name: str,
    config_dir: Path,
    count: int,
    timeout: int,
    cell_seed: int,
    workers: int,
) -> list[dict]:
    mzn_files = sorted(config_dir.glob("instance_*.mzn"))[:count]

    if not mzn_files:
        print(f"  No .mzn files found in {config_dir} – skipping.")
        return []

    # Build argument tuples for the pool (Path must be str for pickling)
    tasks = [
        (n, k, variant, cell_seed, idx, str(mzn_path), solver_name, timeout)
        for idx, mzn_path in enumerate(mzn_files)
    ]

    rows = []
    # Use min(workers, len(tasks)) so we don't spin up idle processes
    actual_workers = min(workers, len(tasks))
    with ProcessPoolExecutor(max_workers=actual_workers) as pool:
        futures = {pool.submit(_solve_one, t): t for t in tasks}
        try:
            for future in as_completed(futures):
                row = future.result()
                status    = row.get("status", "?")
                wall_time = float(row.get("wall_time_s", 0))
                idx       = row["instance_index"]
                mzn_name  = Path(row["mzn_file"]).name
                print(f"    [{variant}] instance {idx:3d}  {mzn_name}  {status}  ({wall_time:.3f}s)")
                rows.append(row)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    # Sort by instance_index so the CSV is in a predictable order
    rows.sort(key=lambda r: r["instance_index"])
    return rows


# ---------------------------------------------------------------------------
# Step 6 – Persist results (one CSV per variant)
# ---------------------------------------------------------------------------

FIXED_COLUMNS = [
    "propagator_variant",
    "instance_type",
    "config_n", "config_k",
    "instance_index", "instance_seed",
    "mzn_file",
    "status", "return_code", "wall_time_s",
]


def save_results(rows: list[dict], variant: str, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = results_dir / f"stats_{variant}_{timestamp}.csv"

    extra_cols = sorted({col for row in rows for col in row if col not in FIXED_COLUMNS})
    all_cols   = FIXED_COLUMNS + extra_cols

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n  Results written: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",     type=str,  default=None,
                   choices=DEFAULT_VARIANTS,
                   help=(
                       "Label for this run, e.g. V0, V1, V2, V3. "
                       "Rebuild the solver on the correct branch before running. "
                       "If omitted you will be prompted."
                   ))
    p.add_argument("--timeout",     type=int,  default=DEFAULT_TIMEOUT,
                   help="Per-instance timeout (seconds)")
    p.add_argument("--seed",        type=int,  default=42,
                   help="Base random seed for instance generation")
    p.add_argument("--outdir",      type=Path, default=RESULTS_DIR,
                   help="Directory for results CSVs")
    p.add_argument("--no-generate", action="store_true",
                   help="Skip instance generation; use existing .mzn files")
    p.add_argument("--no-build",    action="store_true",
                   help="Skip cargo build (use if binary is already up-to-date)")
    p.add_argument("--workers",     type=int,  default=os.cpu_count() or 1,
                   help="Parallel worker processes (default: all logical cores)")
    p.add_argument("--solver",      type=str,  default=DEFAULT_SOLVER,
                   help="MiniZinc solver name exactly as registered (no suffix appended)")
    return p.parse_args()


def main():
    # Required on Windows for ProcessPoolExecutor to work when the script
    # is run directly (not imported).  No-op on Linux/macOS.
    import multiprocessing
    multiprocessing.freeze_support()

    args = parse_args()

    if not args.no_build:
        build_solver()
    else:
        print("  Skipping cargo build (--no-build).")

    # Generate instances for all cells first (once, shared across variants).
    # Each cell gets a unique seed derived from the global --seed value so
    # instance layouts are independent across cells.
    if not args.no_generate:
        print("\n=== Generating instances ===")
        for (n, k), count in sorted(PARAMETER_GRID.items()):
            config_dir = INSTANCE_DIR / "sat" / f"n{n}_k{k}"
            seed = cell_base_seed(args.seed, n, k)
            generate_instances(n, k, count, seed, config_dir)

    # Determine which variant label to use for this run.
    # One run = one branch build = one variant.
    variant = args.variant
    if variant is None:
        print("Which variant is currently built? Choose one:")
        for v in DEFAULT_VARIANTS:
            print(f"  {v}")
        variant = input("Variant: ").strip().upper()
        if variant not in DEFAULT_VARIANTS:
            sys.exit(f"Unknown variant '{variant}'. Must be one of {DEFAULT_VARIANTS}.")

    solver_name = args.solver
    all_rows    = []

    print(f"\n{'='*65}")
    print(f"  Variant {variant}  (solver: {solver_name})")
    print(f"{'='*65}")

    for (n, k), count in sorted(PARAMETER_GRID.items()):
        print(f"\n  n={n}, k={k}  ({count} instances)")
        config_dir = INSTANCE_DIR / "sat" / f"n{n}_k{k}"

        rows = run_config(
            n=n, k=k,
            variant=variant,
            solver_name=solver_name,
            config_dir=config_dir,
            count=count,
            timeout=args.timeout,
            cell_seed=cell_base_seed(args.seed, n, k),
            workers=args.workers,
        )
        all_rows.extend(rows)

    if all_rows:
        save_results(all_rows, variant, args.outdir)
    else:
        print(f"\n  No results to save for {variant}.")

    print("\nDone.")


if __name__ == "__main__":
    # freeze_support() must also be called here on Windows before any
    # ProcessPoolExecutor is created in a spawned child process.
    import multiprocessing
    multiprocessing.freeze_support()
    main()