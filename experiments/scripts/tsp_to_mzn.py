#!/usr/bin/env python3
"""
tsp_to_mzn.py — Convert a TSPLIB (.tsp) file into a MiniZinc (.mzn) model
              for the Pumpkin solver's circuit propagator.

Supported EDGE_WEIGHT_TYPE values:
  EUC_2D   — rounded Euclidean distance (nint)
  CEIL_2D  — ceiling Euclidean distance
  ATT      — pseudo-Euclidean (att48/att532)
  GEO      — geographical (lat/lon) distance
  EXPLICIT — full distance matrix supplied in EDGE_WEIGHT_SECTION

Solve modes (--mode):
  satisfy   — find the first feasible Hamiltonian circuit; best for testing
              the circuit propagator in isolation (default)
  minimize  — minimize total tour cost; tests propagation + branch-and-bound
  bounded   — satisfy, but with an upper bound on the tour cost (--bound K);
              useful to force the solver into a non-trivial search region

Search strategy (--search):
  input_order+indomain_min  — deterministic, identical across solvers (default)
  first_fail+indomain_min   — smaller domain first; still deterministic
  none                      — omit annotation, let each solver choose freely

Usage:
  python tsp_to_mzn.py <input.tsp> [output.mzn] [options]

Examples:
  python tsp_to_mzn.py berlin52.tsp
  python tsp_to_mzn.py berlin52.tsp --mode minimize
  python tsp_to_mzn.py berlin52.tsp --mode bounded --bound 8000
  python tsp_to_mzn.py berlin52.tsp --search first_fail+indomain_min

Running with Pumpkin (MiniZinc backend):
  minizinc --solver pumpkin output.mzn

Running with another solver for comparison:
  minizinc --solver gecode  output.mzn
  minizinc --solver chuffed output.mzn

NOTE: Do NOT use -f (free search) when comparing solvers — that lets the
solver ignore the search annotation and break determinism.
"""

import sys
import math
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# TSPLIB parser
# ---------------------------------------------------------------------------

def parse_tsp(path: Path) -> dict:
    """Parse a TSPLIB file and return a dict with the relevant fields."""
    data = {
        "name": path.stem,
        "dimension": None,
        "edge_weight_type": None,
        "edge_weight_format": None,
        "node_coords": {},   # node_id (1-based) -> (x, y)
        "edge_weights": [],  # flat list (for EXPLICIT)
    }

    with open(path) as f:
        lines = f.readlines()

    section = None
    weight_rows = []

    for raw in lines:
        line = raw.strip()
        if not line or line.upper() == "EOF":
            break

        # Key: value pairs (only outside a data section)
        if ":" in line and section is None:
            key, _, val = line.partition(":")
            key = key.strip().upper()
            val = val.strip()
            if key == "NAME":
                data["name"] = val
            elif key == "DIMENSION":
                data["dimension"] = int(val)
            elif key == "EDGE_WEIGHT_TYPE":
                data["edge_weight_type"] = val.upper()
            elif key == "EDGE_WEIGHT_FORMAT":
                data["edge_weight_format"] = val.upper()
            continue

        # Section headers
        upper = line.upper()
        if upper in ("NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION",
                     "DISPLAY_DATA_SECTION", "TOUR_SECTION"):
            section = upper
            continue

        if section == "NODE_COORD_SECTION":
            parts = line.split()
            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            data["node_coords"][node_id] = (x, y)

        elif section == "EDGE_WEIGHT_SECTION":
            weight_rows.extend(int(v) for v in line.split())

    if weight_rows:
        data["edge_weights"] = weight_rows

    return data


# ---------------------------------------------------------------------------
# Distance functions (TSPLIB spec)
# ---------------------------------------------------------------------------

def dist_euc_2d(a, b):
    xd, yd = a[0] - b[0], a[1] - b[1]
    return int(round(math.sqrt(xd * xd + yd * yd)))


def dist_ceil_2d(a, b):
    xd, yd = a[0] - b[0], a[1] - b[1]
    return math.ceil(math.sqrt(xd * xd + yd * yd))


def dist_att(a, b):
    xd, yd = a[0] - b[0], a[1] - b[1]
    rij = math.sqrt((xd * xd + yd * yd) / 10.0)
    tij = int(round(rij))
    return tij + 1 if tij < rij else tij


def _geo_latlon(coord):
    deg = int(coord)
    minute = coord - deg
    return math.pi * (deg + 5.0 * minute / 3.0) / 180.0


def dist_geo(a, b):
    lat_i, lon_i = _geo_latlon(a[0]), _geo_latlon(a[1])
    lat_j, lon_j = _geo_latlon(b[0]), _geo_latlon(b[1])
    RRR = 6378.388
    q1 = math.cos(lon_i - lon_j)
    q2 = math.cos(lat_i - lat_j)
    q3 = math.cos(lat_i + lat_j)
    return int(RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1)


DIST_FUNCS = {
    "EUC_2D":  dist_euc_2d,
    "CEIL_2D": dist_ceil_2d,
    "ATT":     dist_att,
    "GEO":     dist_geo,
}


# ---------------------------------------------------------------------------
# Distance matrix builder
# ---------------------------------------------------------------------------

def build_distance_matrix(data: dict) -> list:
    n = data["dimension"]
    ewt = data["edge_weight_type"]

    if ewt == "EXPLICIT":
        ewf = data["edge_weight_format"] or "FULL_MATRIX"
        flat = data["edge_weights"]
        matrix = [[0] * n for _ in range(n)]

        if ewf == "FULL_MATRIX":
            for i in range(n):
                for j in range(n):
                    matrix[i][j] = flat[i * n + j]

        elif ewf in ("UPPER_ROW", "UPPER_DIAG_ROW"):
            idx = 0
            start = 1 if ewf == "UPPER_ROW" else 0
            for i in range(n):
                for j in range(i + start, n):
                    matrix[i][j] = flat[idx]
                    matrix[j][i] = flat[idx]
                    idx += 1

        elif ewf in ("LOWER_ROW", "LOWER_DIAG_ROW"):
            idx = 0
            end_offset = 0 if ewf == "LOWER_DIAG_ROW" else -1
            for i in range(n):
                for j in range(0, i + 1 + end_offset):
                    matrix[i][j] = flat[idx]
                    matrix[j][i] = flat[idx]
                    idx += 1

        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {ewf}")

        return matrix

    # Coordinate-based distance
    if ewt not in DIST_FUNCS:
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE: {ewt}. "
            f"Supported: {', '.join(DIST_FUNCS)} and EXPLICIT."
        )

    coords = data["node_coords"]
    dist = DIST_FUNCS[ewt]
    matrix = [[0] * n for _ in range(n)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                matrix[i - 1][j - 1] = dist(coords[i], coords[j])

    return matrix


# ---------------------------------------------------------------------------
# Search annotation builder
# ---------------------------------------------------------------------------

# Maps CLI name -> (var_selection, val_choice) MiniZinc annotation tokens
SEARCH_STRATEGIES = {
    "input_order+indomain_min": ("input_order", "indomain_min"),
    "first_fail+indomain_min":  ("first_fail",  "indomain_min"),
}


def build_search_annotation(strategy: str) -> str:
    """Return the MiniZinc solve annotation string, or '' for 'none'."""
    if strategy == "none":
        return ""
    if strategy not in SEARCH_STRATEGIES:
        raise ValueError(
            f"Unknown search strategy '{strategy}'. "
            f"Valid options: {', '.join(SEARCH_STRATEGIES)} or 'none'."
        )
    var_sel, val_choice = SEARCH_STRATEGIES[strategy]
    return f":: int_search(successor, {var_sel}, {val_choice}, complete)"


# ---------------------------------------------------------------------------
# MZN emitter
# ---------------------------------------------------------------------------

MZN_TEMPLATE = """\
% TSP model generated by tsp_to_mzn.py
% Source  : {name}
% Nodes   : {n}
% Mode    : {mode_label}
% Search  : {search_label}
%
% Solve with Pumpkin:
%   minizinc --solver pumpkin {mzn_filename}
%
% Compare with another solver (do NOT use -f / --free-search):
%   minizinc --solver gecode  {mzn_filename}
%   minizinc --solver chuffed {mzn_filename}

include "circuit.mzn";

int: n = {n};

% distances[i, j] = integer cost of arc from city i to city j (1-indexed)
array[1..n, 1..n] of int: distances = array2d(1..n, 1..n, {flat_distances});

% successor[i] = j  =>  the tour visits city j directly after city i.
% Variables are ordered 1..n so that input_order search is fully deterministic.
array[1..n] of var 1..n: successor;

% The successor array must form a single Hamiltonian circuit.
% This constraint is handled by Pumpkin's circuit propagator.
constraint circuit(successor);
{extra_constraint}
% Total tour length — computed in all modes so output is always comparable.
var int: total_distance = sum(i in 1..n)(distances[i, successor[i]]);

{solve_statement}

output [
  "successor  = " ++ show(successor)      ++ "\\n",
  "total_dist = " ++ show(total_distance) ++ "\\n"
];
"""


def build_mzn(
    data: dict,
    matrix: list,
    mzn_path: Path,
    mode: str,
    search: str,
    bound,
) -> str:
    n = data["dimension"]
    flat = [v for row in matrix for v in row]
    flat_str = "[" + ", ".join(str(v) for v in flat) + "]"

    search_ann = build_search_annotation(search)

    # --- extra constraint (bounded mode) ---
    if mode == "bounded":
        if bound is None:
            raise ValueError("--mode bounded requires --bound K")
        extra = f"\nconstraint total_distance <= {bound};\n"
        mode_label = f"bounded (total_distance <= {bound})"
    else:
        extra = ""
        mode_label = mode

    # --- solve statement ---
    if mode == "minimize":
        if search_ann:
            solve_stmt = f"solve {search_ann}\n  minimize total_distance;"
        else:
            solve_stmt = "solve minimize total_distance;"
    else:
        # satisfy or bounded
        if search_ann:
            solve_stmt = f"solve {search_ann}\n  satisfy;"
        else:
            solve_stmt = "solve satisfy;"

    search_label = search if search != "none" else "solver default (free)"

    return MZN_TEMPLATE.format(
        name=data["name"],
        n=n,
        mode_label=mode_label,
        search_label=search_label,
        mzn_filename=mzn_path.name,
        flat_distances=flat_str,
        extra_constraint=extra,
        solve_statement=solve_stmt,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a TSPLIB .tsp file to a MiniZinc .mzn model "
            "for the Pumpkin solver's circuit propagator."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Path to the .tsp file")
    parser.add_argument(
        "output", nargs="?", default=None,
        help="Output .mzn path (default: same directory, .mzn extension)",
    )
    parser.add_argument(
        "--mode",
        choices=["satisfy", "minimize", "bounded"],
        default="satisfy",
        help=(
            "satisfy  — find any feasible circuit (best for propagator testing). "
            "minimize — minimise total tour cost. "
            "bounded  — satisfy with an upper cost bound (use with --bound)."
        ),
    )
    parser.add_argument(
        "--bound", type=int, default=None,
        help="Upper bound on total_distance for --mode bounded.",
    )
    parser.add_argument(
        "--search",
        choices=list(SEARCH_STRATEGIES.keys()) + ["none"],
        default="input_order+indomain_min",
        help=(
            "Search annotation added to the solve statement. "
            "Use the same value on both solvers for a reproducible comparison. "
            "'none' omits the annotation (solvers may then behave differently)."
        ),
    )
    args = parser.parse_args()

    tsp_path = Path(args.input)
    if not tsp_path.exists():
        sys.exit(f"Error: file not found: {tsp_path}")

    mzn_path = Path(args.output) if args.output else tsp_path.with_suffix(".mzn")

    print(f"Parsing   : {tsp_path}")
    data = parse_tsp(tsp_path)

    n = data["dimension"]
    ewt = data["edge_weight_type"]
    print(f"  Name    : {data['name']}")
    print(f"  Nodes   : {n}")
    print(f"  Weights : {ewt}")

    print("Building distance matrix ...")
    matrix = build_distance_matrix(data)

    print(f"Writing   : {mzn_path}")
    print(f"  Mode    : {args.mode}" + (f" (bound={args.bound})" if args.bound else ""))
    print(f"  Search  : {args.search}")
    mzn_content = build_mzn(data, matrix, mzn_path, args.mode, args.search, args.bound)
    mzn_path.write_text(mzn_content)

    print("\nDone! Run with:")
    print(f"  minizinc --solver pumpkin {mzn_path}")
    print(f"  minizinc --solver gecode  {mzn_path}   # reference solver")
    print()
    print("Tip: avoid -f / --free-search so the search annotation is respected.")



if __name__ == "__main__":
    main()