"""
Instance generator for circuit-constraint benchmark problems,
following the method described in:

  Francis & Stuckey (2014) "Explaining circuit propagation",
  Constraints 19:1-29, Section 3.

Generation procedure:
  1. Place n locations uniformly at random in the unit square.
  2. Compute pairwise Euclidean distances (scaled to integers).
  3. Connect each node to its k nearest neighbours (directed graph;
     edges are added in both directions so the graph is symmetric).
  4. Perform a random walk to guarantee at least one Hamiltonian
     circuit exists: whenever every edge leaving the current node
     leads to an already-visited node, add a fresh random edge to
     an unvisited node; finally close the walk with an edge back to
     the start.
  5. Write a MiniZinc (.mzn) file asking only for FEASIBILITY:
     find any Hamiltonian circuit using only edges present in the
     transport network.  No objective is used, so solver effort
     reflects propagation behaviour directly.

     Rationale for feasibility over min-cost TSP
     --------------------------------------------
     This benchmark suite is designed to study AllDifferent propagator
     behaviour inside the Circuit constraint in an LCG solver.  Adding
     a minimisation objective (e.g. minimise total tour length) causes
     the solver to continue searching after the first solution to prove
     optimality.  The extra search is driven by cost-bound propagation,
     not by AllDifferent or Circuit, making it impossible to cleanly
     attribute differences in node counts / failures / explanation size
     to the propagator under study.  Pure feasibility stops at the first
     solution, so every measured quantity directly reflects how well
     AllDifferent prunes the Circuit search space.

Usage
-----
python generate_instances.py [options]

Options
-------
  -n, --nodes        INT    Number of locations (default: 50)
  -k, --neighbours   INT    Number of nearest neighbours per node (default: 7)
  -c, --count        INT    Number of instances to generate (default: 1)
  -s, --seed         INT    Random seed for reproducibility (default: random)
  -o, --outdir       PATH   Output directory (default: current directory)
  --scale            INT    Multiplier to convert float distances to integers
                            (default: 1000, giving millimetre precision for a
                            unit-square layout)
  --prefix           STR    Filename prefix (default: "instance")

Output
------
One .mzn file per instance, named  <prefix>_n<N>_k<K>_<index>.mzn
"""

import argparse
import math
import os
import random
import sys
from typing import List, Set, Tuple


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def build_distance_matrix(
    coords: List[Tuple[float, float]],
    scale: int,
) -> List[List[int]]:
    """Return a symmetric integer distance matrix (scaled Euclidean)."""
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = round(euclidean(coords[i], coords[j]) * scale)
            dist[i][j] = d
            dist[j][i] = d
    return dist


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def k_nearest_edges(
    dist: List[List[int]],
    k: int,
) -> Set[Tuple[int, int]]:
    """
    Return the set of undirected edges {(i,j) | j is among the k nearest
    neighbours of i}.  Edges are stored as (min, max) pairs to avoid
    duplicates.
    """
    n = len(dist)
    edges: Set[Tuple[int, int]] = set()
    for i in range(n):
        neighbours = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: dist[i][j],
        )
        for j in neighbours[:k]:
            edges.add((min(i, j), max(i, j)))
    return edges


def random_walk_hamiltonian(
    n: int,
    adj: List[Set[int]],
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    Perform a random walk on the current adjacency structure to produce
    a Hamiltonian circuit, adding new edges whenever the walk gets stuck.

    Returns the list of extra (undirected) edges that were added so the
    caller can insert them into the edge set.
    """
    start = rng.randrange(n)
    visited = [False] * n
    path = [start]
    visited[start] = True
    added_edges: List[Tuple[int, int]] = []

    current = start
    while len(path) < n:
        unvisited_neighbours = [v for v in adj[current] if not visited[v]]

        if unvisited_neighbours:
            nxt = rng.choice(unvisited_neighbours)
        else:
            unvisited_all = [v for v in range(n) if not visited[v]]
            nxt = rng.choice(unvisited_all)
            adj[current].add(nxt)
            adj[nxt].add(current)
            added_edges.append((min(current, nxt), max(current, nxt)))

        visited[nxt] = True
        path.append(nxt)
        current = nxt

    # Close the circuit
    if start not in adj[current]:
        adj[current].add(start)
        adj[start].add(current)
        added_edges.append((min(current, start), max(current, start)))

    return added_edges


def build_graph(
    n: int,
    k: int,
    dist: List[List[int]],
    rng: random.Random,
) -> List[List[int]]:
    """
    Build the transport network following Francis & Stuckey Section 3.

    Returns allowed[i][j]:
      False  if no direct connection exists between i and j
      True   if an edge exists (i.e. j is a valid successor of i)

    Note: distances are no longer written to the MZN file because the
    feasibility model has no cost objective.  We retain the distance
    matrix only for k-nearest construction; the output is a plain
    boolean adjacency structure.
    """
    edge_set = k_nearest_edges(dist, k)

    adj: List[Set[int]] = [set() for _ in range(n)]
    for (i, j) in edge_set:
        adj[i].add(j)
        adj[j].add(i)

    extra = random_walk_hamiltonian(n, adj, rng)
    edge_set.update(extra)

    # Boolean adjacency matrix
    allowed = [[False] * n for _ in range(n)]
    for (i, j) in edge_set:
        allowed[i][j] = True
        allowed[j][i] = True

    return allowed, edge_set


# ---------------------------------------------------------------------------
# MiniZinc file writer  –  FEASIBILITY model
# ---------------------------------------------------------------------------

MZN_TEMPLATE = """\
%% Circuit-constraint feasibility benchmark instance
%% Generated by generate_instances.py
%% Method: Francis & Stuckey (2014) "Explaining circuit propagation"
%%
%% Model: pure feasibility -- find any Hamiltonian circuit that uses
%%        only edges present in the transport network.
%%        No objective is included so that solver effort (node count,
%%        failures, explanation size) reflects propagation behaviour
%%        directly, without interference from cost-bound reasoning.
%%
%% Parameters
%%   n    = {n}  (number of locations)
%%   k    = {k}  (nearest-neighbour degree used during generation)
%%   seed = {seed}
%%   edges = {num_edges}  (undirected edges in the transport network)

include "circuit.mzn";

int: n = {n};
set of int: Locations = 1..n;

%% allowed[i,j] = true iff a direct link exists from i to j
array[Locations, Locations] of bool: allowed = array2d(Locations, Locations, [
{allowed_rows}
]);

%% Successor variables: succ[i] = next location after i in the tour
array[Locations] of var Locations: succ;

%% Restrict successors to existing edges only
constraint forall(i in Locations)(
  succ[i] in {{j | j in Locations where allowed[i, j]}}
);

%% Successors must form a Hamiltonian circuit
constraint circuit(succ);

solve satisfy;

output [
  "succ = ", show(succ), "\\n"
];
"""


def format_allowed(allowed: List[List[bool]], n: int) -> str:
    """Format the 2-D boolean adjacency matrix as a flat MiniZinc literal."""
    lines = []
    for i in range(n):
        row = ", ".join("true" if allowed[i][j] else "false" for j in range(n))
        comma = "," if i < n - 1 else ""
        lines.append(f"  {row}{comma}  %% from location {i + 1}")
    return "\n".join(lines)


def write_mzn(
    filepath: str,
    n: int,
    k: int,
    seed: int,
    allowed: List[List[bool]],
    num_edges: int,
) -> None:
    allowed_rows = format_allowed(allowed, n)

    content = MZN_TEMPLATE.format(
        n=n,
        k=k,
        seed=seed,
        num_edges=num_edges,
        allowed_rows=allowed_rows,
    )

    with open(filepath, "w") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate circuit-constraint feasibility benchmark instances "
            "(Francis & Stuckey 2014 method) and write MiniZinc files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--nodes", type=int, default=50,
                        help="Number of locations")
    parser.add_argument("-k", "--neighbours", type=int, default=7,
                        help="Number of nearest neighbours per node")
    parser.add_argument("-c", "--count", type=int, default=1,
                        help="Number of instances to generate")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Base random seed (uses system entropy if omitted)")
    parser.add_argument("-o", "--outdir", type=str, default=".",
                        help="Output directory for .mzn files")
    parser.add_argument("--scale", type=int, default=1000,
                        help="Scale factor: distances = round(Euclidean * scale)")
    parser.add_argument("--prefix", type=str, default="instance",
                        help="Filename prefix")
    return parser.parse_args()


def generate_instance(
    n: int,
    k: int,
    seed: int,
    scale: int,
) -> tuple:
    """Generate one instance and return (allowed matrix, edge count)."""
    rng = random.Random(seed)

    coords = [(rng.random(), rng.random()) for _ in range(n)]
    dist = build_distance_matrix(coords, scale)
    allowed, edge_set = build_graph(n, k, dist, rng)

    return allowed, len(edge_set)


def main() -> None:
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base_seed = args.seed if args.seed is not None else random.randrange(2 ** 32)

    print(f"Generating {args.count} feasibility instance(s):")
    print(f"  n={args.nodes}, k={args.neighbours}, scale={args.scale}")
    print(f"  base seed={base_seed}, output dir='{args.outdir}'")
    print()

    for idx in range(args.count):
        seed = base_seed + idx

        allowed, num_edges = generate_instance(
            n=args.nodes,
            k=args.neighbours,
            seed=seed,
            scale=args.scale,
        )

        filename = (
            f"{args.prefix}_n{args.nodes}_k{args.neighbours}_{idx:04d}.mzn"
        )
        filepath = os.path.join(args.outdir, filename)

        write_mzn(
            filepath=filepath,
            n=args.nodes,
            k=args.neighbours,
            seed=seed,
            allowed=allowed,
            num_edges=num_edges,
        )

        print(f"  [{idx:4d}] seed={seed:10d}  edges={num_edges:5d}  -> {filepath}")

    print("\nDone.")


if __name__ == "__main__":
    main()