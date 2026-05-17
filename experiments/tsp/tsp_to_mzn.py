"""
tsp_to_mzn.py  –  Convert a TSPLIB .tsp or .atsp file to a MiniZinc model
===========================================================================

Supported EDGE_WEIGHT_TYPE values
----------------------------------
  EUC_2D   – Euclidean distance in the plane, rounded to nearest integer
  CEIL_2D  – Euclidean distance, ceiling
  ATT      – Pseudo-Euclidean (AT&T) distance
  GEO      – Geographical distance (latitude/longitude)
  EXPLICIT – Matrix given directly (LOWER_DIAG_ROW, UPPER_DIAG_ROW,
             FULL_MATRIX, UPPER_ROW, LOWER_ROW)

MiniZinc model produced
------------------------
  Two modes, selected via --satisfy flag:

  minimize (default)
    Minimise total tour length.  Standard TSP objective; results are
    directly comparable to TSPLIB published optima.  Slow on hard
    instances because the solver must also prove no better tour exists.

  satisfy
    Find the first Hamiltonian circuit whose total length <= ub, where
    ub is the greedy nearest-neighbour tour cost.  The solver stops
    immediately on the first feasible solution.  Pure test of circuit
    propagator search reduction — failures and noGoods measure exactly
    how hard it is to find one good circuit, with no time spent on
    optimality proof.

Usage
-----
  python tsp_to_mzn.py input.tsp [output.mzn] [--satisfy]
  python tsp_to_mzn.py input.atsp [output.mzn] [--satisfy]

  If output path is omitted the .mzn is written next to the input file.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# TSPLIB parser
# ---------------------------------------------------------------------------

class TSPInstance:
    def __init__(self) -> None:
        self.name:       str = "unknown"
        self.comment:    str = ""
        self.dimension:  int = 0
        self.is_atsp:    bool = False
        self.edge_type:  str = ""
        self.dist: list[list[int]] = []


def _geo_dist(coords: list[tuple[float, float]], i: int, j: int) -> int:
    def to_rad(deg_dec: float) -> float:
        deg  = int(deg_dec)
        mins = deg_dec - deg
        return math.pi * (deg + 5.0 * mins / 3.0) / 180.0
    lat1, lon1 = to_rad(coords[i][0]), to_rad(coords[i][1])
    lat2, lon2 = to_rad(coords[j][0]), to_rad(coords[j][1])
    RRR = 6378.388
    q1  = math.cos(lon1 - lon2)
    q2  = math.cos(lat1 - lat2)
    q3  = math.cos(lat1 + lat2)
    return int(RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0)


def _att_dist(coords: list[tuple[float, float]], i: int, j: int) -> int:
    dx  = coords[i][0] - coords[j][0]
    dy  = coords[i][1] - coords[j][1]
    rij = math.sqrt((dx * dx + dy * dy) / 10.0)
    tij = int(rij)
    return tij + 1 if tij < rij else tij


def _euc2d_dist(coords: list[tuple[float, float]], i: int, j: int) -> int:
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def _ceil2d_dist(coords: list[tuple[float, float]], i: int, j: int) -> int:
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return math.ceil(math.sqrt(dx * dx + dy * dy))


def _build_coord_matrix(
    coords: list[tuple[float, float]],
    edge_type: str,
    n: int,
    symmetric: bool,
) -> list[list[int]]:
    dist_fn = {
        "EUC_2D" : _euc2d_dist,
        "CEIL_2D": _ceil2d_dist,
        "ATT"    : _att_dist,
        "GEO"    : _geo_dist,
    }[edge_type]
    d = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i][j] = dist_fn(coords, i, j)
                if symmetric:
                    d[j][i] = d[i][j]
    return d


def _read_explicit(lines: list[str], n: int, fmt: str) -> list[list[int]]:
    nums = [int(x) for x in " ".join(lines).split()]
    d = [[0] * n for _ in range(n)]
    if fmt == "FULL_MATRIX":
        for i in range(n):
            for j in range(n):
                d[i][j] = nums[i * n + j]
    elif fmt in ("LOWER_DIAG_ROW", "LOWER_ROW"):
        idx = 0
        for i in range(n):
            cols = i + 1 if fmt == "LOWER_DIAG_ROW" else i
            for j in range(cols):
                v = nums[idx]; idx += 1
                d[i][j] = v
                d[j][i] = v
    elif fmt in ("UPPER_DIAG_ROW", "UPPER_ROW"):
        idx = 0
        for i in range(n):
            start = i if fmt == "UPPER_DIAG_ROW" else i + 1
            for j in range(start, n):
                v = nums[idx]; idx += 1
                d[i][j] = v
                d[j][i] = v
    else:
        sys.exit(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")
    return d


def parse_tsplib(path: Path) -> TSPInstance:
    inst = TSPInstance()
    inst.is_atsp = path.suffix.lower() == ".atsp"
    text  = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines()]

    kv: dict[str, str] = {}
    section = None
    coord_section:  list[str] = []
    weight_section: list[str] = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.upper().startswith("NODE_COORD_SECTION"):
            section = "COORD"; i += 1; continue
        if ln.upper().startswith("EDGE_WEIGHT_SECTION"):
            section = "WEIGHT"; i += 1; continue
        if ln.upper() in ("EOF", "DEMAND_SECTION", "DEPOT_SECTION", "TOUR_SECTION"):
            section = None; i += 1; continue
        if section == "COORD":
            coord_section.append(ln)
        elif section == "WEIGHT":
            weight_section.append(ln)
        else:
            m = re.match(r"([A-Z_]+)\s*[:=]\s*(.*)", ln, re.IGNORECASE)
            if m:
                kv[m.group(1).upper()] = m.group(2).strip()
        i += 1

    inst.name      = kv.get("NAME", path.stem)
    inst.comment   = kv.get("COMMENT", "")
    inst.dimension = int(kv.get("DIMENSION", 0))
    inst.edge_type = kv.get("EDGE_WEIGHT_TYPE", "").upper()
    weight_fmt     = kv.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper()
    n = inst.dimension

    if n == 0:
        sys.exit(f"Could not determine DIMENSION from {path}")

    coord_types = {"EUC_2D", "CEIL_2D", "ATT", "GEO"}
    if inst.edge_type in coord_types:
        if not coord_section:
            sys.exit(f"NODE_COORD_SECTION missing for {inst.edge_type} in {path}")
        coords: list[tuple[float, float]] = []
        for ln in coord_section:
            parts = ln.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
        if len(coords) != n:
            sys.exit(f"Expected {n} coordinates, got {len(coords)}")
        inst.dist = _build_coord_matrix(coords, inst.edge_type, n, not inst.is_atsp)
    elif inst.edge_type == "EXPLICIT":
        if not weight_section:
            sys.exit(f"EDGE_WEIGHT_SECTION missing for EXPLICIT in {path}")
        inst.dist = _read_explicit(weight_section, n, weight_fmt)
    else:
        sys.exit(
            f"Unsupported EDGE_WEIGHT_TYPE '{inst.edge_type}' in {path}.\n"
            f"Supported: EUC_2D, CEIL_2D, ATT, GEO, EXPLICIT"
        )
    return inst


# ---------------------------------------------------------------------------
# Pre-solve heuristics
# ---------------------------------------------------------------------------

def _greedy_nn_tour(dist: list[list[int]], n: int) -> list[int]:
    """Nearest-neighbour greedy tour from city 0. Returns 0-indexed succ list."""
    visited   = [False] * n
    tour_succ = [0] * n
    current   = 0
    visited[0] = True
    for _ in range(n - 1):
        best_cost, best_next = -1, -1
        for j in range(n):
            if not visited[j] and dist[current][j] > 0:
                if best_cost < 0 or dist[current][j] < best_cost:
                    best_cost, best_next = dist[current][j], j
        if best_next == -1:
            best_next = next(j for j in range(n) if not visited[j])
        tour_succ[current] = best_next
        visited[best_next] = True
        current = best_next
    tour_succ[current] = 0
    return tour_succ


def _tour_length(dist: list[list[int]], succ: list[int]) -> int:
    return sum(dist[i][succ[i]] for i in range(len(succ)))


def _min_outgoing(dist: list[list[int]], n: int) -> int:
    return sum(min(dist[i][j] for j in range(n) if j != i) for i in range(n))


# ---------------------------------------------------------------------------
# MiniZinc templates
# ---------------------------------------------------------------------------

# Shared header — data declarations, circuit constraint, tourLength variable.
# Both modes use this identical block so the FlatZinc structure is the same
# and the only difference is the solve goal.
_MZN_HEADER = """\
%% TSP benchmark instance
%% Source   : {source}
%% Comment  : {comment}
%% Type     : {tsp_type}  ({edge_type})
%% Nodes    : {n}
%% Mode     : {mode_comment}
%% Bounds   : lb={lb}  ub={ub} (greedy nearest-neighbour tour)
%% Converted by tsp_to_mzn.py

include "circuit.mzn";

int: n = {n};
set of int: Cities = 1..n;

%% lb = sum of each city's cheapest outgoing edge  (valid lower bound)
%% ub = greedy nearest-neighbour tour cost         (valid upper bound)
int: lb = {lb};
int: ub = {ub};

%% dist[i,j] = travel cost from city i to city j  (1-indexed)
array[Cities, Cities] of int: dist = array2d(Cities, Cities, [
{dist_rows}
]);

%% succ[i] = next city after i in the Hamiltonian circuit
array[Cities] of var Cities: succ;

%% Hamiltonian circuit
constraint circuit(succ);

%% Total tour length, bounded tightly on both sides
var lb..ub: tourLength;
constraint tourLength = sum(i in Cities)(dist[i, succ[i]]);

{symmetry_break_comment}
{symmetry_break_constraint}
"""

# satisfy mode: stop at the first circuit with tourLength <= ub
_MZN_SATISFY = """\
%% SATISFY MODE
%% The bound  tourLength <= ub  is already enforced by the variable domain.
%% The solver stops as soon as it finds the first feasible assignment —
%% no optimality proof required.  This makes failures/noGoods a clean
%% measure of pure circuit propagator search reduction.
solve satisfy;

output [
  "succ = ",       show(succ),       "\\n",
  "tourLength = ", show(tourLength), "\\n"
];
"""

# minimize mode: prove optimality (slower, comparable to TSPLIB optima)
_MZN_MINIMIZE = """\
%% MINIMIZE MODE
%% Finds and proves the optimal tour.  Comparable to published TSPLIB optima.
solve minimize tourLength;

output [
  "succ = ",       show(succ),       "\\n",
  "tourLength = ", show(tourLength), "\\n"
];
"""


def _format_dist(dist: list[list[int]], n: int) -> str:
    rows = []
    for i in range(n):
        row   = ", ".join(str(dist[i][j]) for j in range(n))
        comma = "," if i < n - 1 else ""
        rows.append(f"  {row}{comma}  %% from city {i + 1}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# MiniZinc writer
# ---------------------------------------------------------------------------

def write_mzn(inst: TSPInstance, out_path: Path, satisfy: bool = False) -> None:
    n = inst.dimension
    d = inst.dist

    nn_succ = _greedy_nn_tour(d, n)
    ub = _tour_length(d, nn_succ)
    lb = _min_outgoing(d, n)

    # Symmetry breaking (symmetric TSP only)
    if not inst.is_atsp:
        sym_comment = (
            "%% Symmetry breaking: pred[i] is the predecessor of city i.\n"
            "%% Requiring succ[1] < pred[1] eliminates all n cyclic\n"
            "%% rotations of every tour, keeping one canonical direction."
        )
        sym_constraint = (
            "array[Cities] of var Cities: pred;\n"
            "constraint forall(i in Cities)(pred[succ[i]] = i);\n"
            "constraint succ[1] < pred[1];"
        )
    else:
        sym_comment    = "%% No symmetry breaking for ATSP (reversals change cost)."
        sym_constraint = ""

    tsp_type     = "ATSP (asymmetric)" if inst.is_atsp else "TSP (symmetric)"
    mode_comment = "satisfy (first good circuit)" if satisfy else "minimize (prove optimality)"

    header = _MZN_HEADER.format(
        source                    = out_path.stem,
        comment                   = inst.comment,
        tsp_type                  = tsp_type,
        edge_type                 = inst.edge_type,
        n                         = n,
        lb                        = lb,
        ub                        = ub,
        mode_comment              = mode_comment,
        dist_rows                 = _format_dist(d, n),
        symmetry_break_comment    = sym_comment,
        symmetry_break_constraint = sym_constraint,
    )

    solve_block = _MZN_SATISFY if satisfy else _MZN_MINIMIZE
    out_path.write_text(header + solve_block, encoding="utf-8")

    mode_tag = "satisfy" if satisfy else "minimize"
    print(
        f"  Written: {out_path.name}  "
        f"(n={n}, mode={mode_tag}, lb={lb}, ub={ub}, gap={ub - lb})"
    )


# ---------------------------------------------------------------------------
# CLI (standalone use)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a TSPLIB .tsp/.atsp file to a MiniZinc model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  type=Path, help=".tsp or .atsp file")
    p.add_argument("output", type=Path, nargs="?", default=None,
                   help=".mzn output path (default: same stem as input)")
    p.add_argument("--satisfy", action="store_true",
                   help=(
                       "Generate a satisfaction model (find first circuit "
                       "within NN bound) instead of the default minimisation "
                       "model.  Much faster; better for propagator comparison."
                   ))
    return p.parse_args()


def convert(
    tsp_path: Path,
    mzn_path: Optional[Path] = None,
    satisfy:  bool = False,
) -> Path:
    """Convert *tsp_path* → *mzn_path*. Returns the mzn path."""
    if mzn_path is None:
        mzn_path = tsp_path.with_suffix(".mzn")
    inst = parse_tsplib(tsp_path)
    write_mzn(inst, mzn_path, satisfy=satisfy)
    return mzn_path


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output, satisfy=args.satisfy)