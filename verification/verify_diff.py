"""
Compare pumpkin-circuit against cp-sat using minizinc-diff.

Runs:

    minizinc-diff diff <model.mzn> <solver1> <solver2>

for each .mzn instance listed below.

Example:
    minizinc-diff diff instances/foo.mzn pumpkin-circuit cp-sat

Results are printed and logged to:
    results/diff_verification.log
"""

import re
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "results"
LOG_FILE = RESULTS_DIR / "diff_verification.log"

TIMEOUT_SEC = 7200  # 30 minutes

LEFT_SOLVER = "pumpkin-circuit"
RIGHT_SOLVER = "cp-sat"


# ---------------------------------------------------------------------------
# Instance List
# ---------------------------------------------------------------------------

# Direct .mzn instances
INSTANCES = [

    (
        ROOT / "instances/tsp.mzn",
        ROOT / "instances/burma14.dzn",
    ),
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_handle = None


def log(msg=""):
    print(msg, flush=True)

    if _log_handle is not None:
        _log_handle.write(msg + "\n")
        _log_handle.flush()


# ---------------------------------------------------------------------------
# Output Normalisation
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_TIME_RE = re.compile(
    r"^(\s*(?:Left|Right) time:\s*)(\d+)\s*$",
    re.MULTILINE,
)


def normalize_output(text):
    if not text:
        return text

    text = _ANSI_RE.sub("", text)

    text = _TIME_RE.sub(
        lambda m: f"{m.group(1)}{int(m.group(2)) / 1000:.3f}s",
        text,
    )

    return text


# ---------------------------------------------------------------------------
# Status Classification
# ---------------------------------------------------------------------------

def classify(stdout, exit_code):
    text = stdout.lower()

    if "timeout" in text:
        return "TIMEOUT"

    if (
        "mismatch" in text
        or "differ" in text
        or "do not match" in text
    ):
        return "MISMATCH"

    if "match" in text:
        return "MATCH"

    if exit_code != 0:
        return "ERROR"

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Single Run
# ---------------------------------------------------------------------------

def run_instance(model_path, instance_path):    
    argv = [
        "minizinc-diff",
        "diff",
        "-t",
        str(TIMEOUT_SEC),
        str(model_path),
        str(instance_path),
        LEFT_SOLVER,
        RIGHT_SOLVER,
    ]

    started = time.monotonic()

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
        )

        stdout = normalize_output(proc.stdout)
        stderr = normalize_output(proc.stderr)

        exit_code = proc.returncode

    except FileNotFoundError:
        log("ERROR: minizinc-diff not found on PATH.")
        sys.exit(1)

    except Exception as e:
        stdout = ""
        stderr = f"{type(e).__name__}: {e}"
        exit_code = -1

    wall = time.monotonic() - started

    status = classify(stdout, exit_code)

    return {
        "cmd": " ".join(argv),
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "status": status,
        "wall": wall,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _log_handle

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _log_handle = LOG_FILE.open("a", encoding="utf-8")

    missing = []

    for model, instance in INSTANCES:
        if not model.exists():
            missing.append(model)

        if not instance.exists():
            missing.append(instance)

    if missing:
        print("ERROR: missing instance files:", file=sys.stderr)

        for p in missing:
            print(f"  {p}", file=sys.stderr)

        sys.exit(1)

    log("=" * 80)
    log(f"Comparing {LEFT_SOLVER} vs {RIGHT_SOLVER}")
    log(f"Instances: {len(INSTANCES)}")
    log(f"Timeout: {TIMEOUT_SEC}s")
    log("")

    summary = []

    for i, (model, instance) in enumerate(INSTANCES, start=1):        
        log("=" * 80)
        log(f"[{i}/{len(INSTANCES)}] {instance.name}")

        result = run_instance(model, instance)

        log(f"$ {result['cmd']}")

        log("--- stdout ---")
        log(result["stdout"].rstrip() or "(empty)")

        log("--- stderr ---")
        log(result["stderr"].rstrip() or "(empty)")

        log(
            f"exit_code: {result['exit_code']}  "
            f"status: {result['status']}  "
            f"wall_time: {result['wall']:.1f}s"
        )

        log("")

        summary.append(
            (
                instance.name,
                result["status"],
                result["wall"],
            )
        )

    log("=" * 80)
    log("SUMMARY")

    for name, status, wall in summary:
        log(f"{name:<30} {status:<10} {wall:>6.1f}s")

    _log_handle.close()


if __name__ == "__main__":
    main()