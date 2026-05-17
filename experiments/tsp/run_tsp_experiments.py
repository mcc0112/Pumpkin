"""
run_tsp_experiments.py  –  TSP benchmark pipeline for Pumpkin
==============================================================

For every .tsp / .atsp file found in experiments/tsp/tsplib/:

  1. Convert  .tsp  →  .mzn   (via tsp_to_mzn.py)
  2. Flatten  .mzn  →  .fzn   (via minizinc --solver pumpkin -c)
  3. Solve    .fzn             (via cargo run -p pumpkin-solver -- file.fzn -s)
  4. Parse %%%mzn-stat lines from solver output
  5. Append one row to  experiments/tsp/results/stats_<timestamp>.csv

Directory layout (relative to Pumpkin project root)
----------------------------------------------------
experiments/tsp/
  tsplib/          ← put your .tsp / .atsp files here
  instances/
    berlin52/
      berlin52.mzn
      berlin52.fzn
    ...
  results/
    stats_<timestamp>.csv

Usage (from the Pumpkin project root)
--------------------------------------
  python experiments\\tsp\\run_tsp_experiments.py [options]

Options
-------
  --timeout   INT   Per-instance solver timeout in seconds  (default: 300)
  --release         Use cargo --release
  --no-convert      Skip .tsp→.mzn and .mzn→.fzn; reuse existing .fzn files
                    (use this on the second branch so instances are identical)
  --satisfy         Generate a satisfaction model instead of minimisation.
                    Solver stops at the first circuit within the NN bound.
                    Much faster; isolates pure propagator search reduction.
  --instances PATH [PATH ...]
                    Run only these specific .tsp/.atsp files instead of the
                    whole tsplib/ folder
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
# Paths  (everything relative to the Pumpkin project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent                  # experiments/tsp/
PROJECT_ROOT = SCRIPT_DIR.parent.parent               # Pumpkin/
TSPLIB_DIR   = SCRIPT_DIR / "tsplib"
INSTANCE_DIR = SCRIPT_DIR / "instances"
RESULTS_DIR  = SCRIPT_DIR / "results"
TSP_TO_MZN   = SCRIPT_DIR / "tsp_to_mzn.py"


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout: int | None = None,
) -> tuple[str, str, int]:
    result = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


# ---------------------------------------------------------------------------
# Stat parsing
# ---------------------------------------------------------------------------

def parse_statistics(output: str) -> dict[str, str]:
    """Extract %%%mzn-stat: key=value lines from solver output."""
    stats: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("%%%mzn-stat:"):
            body = line[len("%%%mzn-stat:"):].strip()
            if "=" in body:
                k, _, v = body.partition("=")
                stats[k.strip()] = v.strip()
    return stats


# ---------------------------------------------------------------------------
# Step 1 – .tsp → .mzn
# ---------------------------------------------------------------------------

def convert_to_mzn(tsp_path: Path, out_dir: Path, satisfy: bool = False) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mzn_path = out_dir / (tsp_path.stem + ".mzn")

    cmd = [sys.executable, str(TSP_TO_MZN), str(tsp_path), str(mzn_path)]
    if satisfy:
        cmd.append("--satisfy")
    print(f"    [tsp→mzn]  {tsp_path.name}", end="  ")
    stdout, stderr, rc = run_cmd(cmd, cwd=PROJECT_ROOT)

    if rc != 0:
        print(f"FAILED (rc={rc})")
        print(f"      {stderr[:300]}", file=sys.stderr)
        return None
    print("OK")
    return mzn_path


# ---------------------------------------------------------------------------
# Step 2 – .mzn → .fzn
# ---------------------------------------------------------------------------

def flatten_to_fzn(mzn_path: Path) -> Path | None:
    fzn_path = mzn_path.with_suffix(".fzn")
    rel_mzn  = mzn_path.relative_to(PROJECT_ROOT)

    cmd = ["minizinc", "--solver", "pumpkin", "-c", str(rel_mzn)]
    print(f"    [mzn→fzn]  {mzn_path.name}", end="  ")
    stdout, stderr, rc = run_cmd(cmd, cwd=PROJECT_ROOT)

    if rc != 0 or not fzn_path.exists():
        print(f"FAILED (rc={rc})")
        print(f"      {stderr[:300]}", file=sys.stderr)
        return None
    print(f"OK  →  {fzn_path.name}")
    return fzn_path


# ---------------------------------------------------------------------------
# Step 3 – solve .fzn
# ---------------------------------------------------------------------------

def solve_fzn(
    fzn_path: Path,
    timeout: int,
    use_release: bool,
) -> dict[str, str]:
    rel_fzn = fzn_path.relative_to(PROJECT_ROOT)

    cargo_cmd = ["cargo", "run", "-p", "pumpkin-solver"]
    if use_release:
        cargo_cmd.append("--release")
    cargo_cmd += ["--", str(rel_fzn), "-s"]

    print(f"    [solve]    {fzn_path.name}", end="  ")
    t0 = time.perf_counter()
    try:
        stdout, stderr, rc = run_cmd(cargo_cmd, cwd=PROJECT_ROOT, timeout=timeout)
        wall = time.perf_counter() - t0
        status = "ok" if rc == 0 else "error"
    except subprocess.TimeoutExpired:
        wall   = float(timeout)
        stdout = stderr = ""
        rc     = -1
        status = "timeout"

    print(f"{status}  ({wall:.1f}s)")
    stats = parse_statistics(stdout + stderr)
    stats["wall_time_s"] = f"{wall:.3f}"
    stats["status"]      = status
    stats["return_code"] = str(rc)
    return stats


# ---------------------------------------------------------------------------
# Step 4 – CSV persistence
# ---------------------------------------------------------------------------

FIXED_COLS = [
    "mode",
    "instance_name",
    "tsp_file",
    "mzn_file",
    "fzn_file",
    "status",
    "return_code",
    "wall_time_s",
]


def save_results(rows: list[dict], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"stats_{ts}.csv"

    extra = sorted({k for row in rows for k in row if k not in FIXED_COLS})
    cols  = FIXED_COLS + extra

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\n  Results written to: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TSP benchmarks through Pumpkin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-instance solver timeout (seconds)")
    p.add_argument("--release", action="store_true",
                   help="Use cargo --release")
    p.add_argument("--no-convert", action="store_true",
                   help="Skip .tsp→.mzn and .mzn→.fzn; reuse existing .fzn files")
    p.add_argument("--instances", type=Path, nargs="+", default=None,
                   help="Specific .tsp/.atsp files to run (default: all in tsplib/)")
    p.add_argument("--satisfy", action="store_true",
                   help=(
                       "Use satisfy mode: find first circuit within NN bound "
                       "instead of proving optimality. Faster; better for "
                       "isolating propagator search reduction."
                   ))
    return p.parse_args()


def discover_tsp_files(args: argparse.Namespace) -> list[Path]:
    if args.instances:
        return sorted(args.instances)
    if not TSPLIB_DIR.exists():
        sys.exit(
            f"No TSPLIB directory found at {TSPLIB_DIR}.\n"
            "Create it and place your .tsp / .atsp files inside."
        )
    files = sorted(TSPLIB_DIR.glob("*.tsp")) + sorted(TSPLIB_DIR.glob("*.atsp"))
    if not files:
        sys.exit(f"No .tsp or .atsp files found in {TSPLIB_DIR}")
    return files


def main() -> None:
    args = parse_args()
    tsp_files = discover_tsp_files(args)

    print(f"Found {len(tsp_files)} TSP instance(s) to process.\n")
    all_rows: list[dict] = []

    for tsp_path in tsp_files:
        name    = tsp_path.stem
        out_dir = INSTANCE_DIR / name

        print(f"\n{'─'*60}")
        print(f"  Instance: {name}  ({tsp_path.name})")
        print(f"{'─'*60}")

        row: dict = {
            "instance_name": name,
            "tsp_file"     : str(tsp_path.relative_to(PROJECT_ROOT)),
            "mzn_file"     : "",
            "fzn_file"     : "",
            "mode"         : "satisfy" if args.satisfy else "minimize",
        }

        if args.no_convert:
            # Expect the .fzn to already exist from a previous run
            fzn_path = out_dir / (name + ".fzn")
            mzn_path = out_dir / (name + ".mzn")
            if not fzn_path.exists():
                print(f"    [skip] .fzn not found at {fzn_path}  –  skipping")
                row["status"] = "fzn_missing"
                all_rows.append(row)
                continue
            print(f"    [reuse] {fzn_path.name}")
            row["mzn_file"] = str(mzn_path.relative_to(PROJECT_ROOT)) if mzn_path.exists() else ""
            row["fzn_file"] = str(fzn_path.relative_to(PROJECT_ROOT))
        else:
            # ---- tsp → mzn ------------------------------------------------
            mzn_path = convert_to_mzn(tsp_path, out_dir, satisfy=args.satisfy)
            if mzn_path is None:
                row["status"] = "mzn_conversion_failed"
                all_rows.append(row)
                continue
            row["mzn_file"] = str(mzn_path.relative_to(PROJECT_ROOT))

            # ---- mzn → fzn ------------------------------------------------
            fzn_path = flatten_to_fzn(mzn_path)
            if fzn_path is None:
                row["status"] = "fzn_conversion_failed"
                all_rows.append(row)
                continue
            row["fzn_file"] = str(fzn_path.relative_to(PROJECT_ROOT))

        # ---- solve --------------------------------------------------------
        stats = solve_fzn(fzn_path, args.timeout, args.release)
        row.update(stats)
        all_rows.append(row)

    if all_rows:
        save_results(all_rows, RESULTS_DIR)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()