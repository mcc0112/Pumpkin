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
  5. Write a MiniZinc (.mzn) file matching the tour-design model
     from Figure 1 of the paper (minimise the longest leg of the
     circuit, using only edges present in the transport network).

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

MiniZinc model
--------------
The generated file is self-contained: it embeds the data and includes
the circuit constraint predicate inline (no separate include needed).
It minimises maxleg, the length of the longest leg in the Hamiltonian
circuit, using only edges present in the transport network.
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
        # Sort all other nodes by distance to i
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

    Returns the list of extra (directed) edges that were added so the
    caller can insert them into the edge set.
    """
    start = rng.randrange(n)
    visited = [False] * n
    path = [start]
    visited[start] = True
    added_edges: List[Tuple[int, int]] = []

    current = start
    while len(path) < n:
        # Neighbours reachable from current that are still unvisited
        unvisited_neighbours = [v for v in adj[current] if not visited[v]]

        if unvisited_neighbours:
            # Follow a random existing edge to an unvisited node
            nxt = rng.choice(unvisited_neighbours)
        else:
            # All existing neighbours already visited — add a new edge
            unvisited_all = [v for v in range(n) if not visited[v]]
            nxt = rng.choice(unvisited_all)
            # Register the new edge in both directions
            adj[current].add(nxt)
            adj[nxt].add(current)
            added_edges.append((min(current, nxt), max(current, nxt)))

        visited[nxt] = True
        path.append(nxt)
        current = nxt

    # Close the circuit: add edge from last node back to start if not present
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

    Returns travel_time[i][j]:
      -1  if no direct connection exists between i and j
      >=0 the (integer) travel time otherwise
    """
    # Step 1: k-nearest-neighbour edges
    edge_set = k_nearest_edges(dist, k)

    # Build adjacency sets for the random walk
    adj: List[Set[int]] = [set() for _ in range(n)]
    for (i, j) in edge_set:
        adj[i].add(j)
        adj[j].add(i)

    # Step 2: random walk to guarantee a Hamiltonian circuit
    extra = random_walk_hamiltonian(n, adj, rng)
    edge_set.update(extra)

    # Build the travel_time matrix (-1 = no direct connection)
    travel_time = [[-1] * n for _ in range(n)]
    for (i, j) in edge_set:
        travel_time[i][j] = dist[i][j]
        travel_time[j][i] = dist[i][j]

    return travel_time


# ---------------------------------------------------------------------------
# MiniZinc file writer
# ---------------------------------------------------------------------------

MZN_TEMPLATE = """\
%% Tour-design benchmark instance
%% Generated by generate_instances.py
%% Method: Francis & Stuckey (2014) "Explaining circuit propagation"
%%
%% Parameters
%%   n = {n}  (number of locations)
%%   k = {k}  (nearest-neighbour degree used during generation)
%%   seed = {seed}
%%   scale = {scale}  (distances are Euclidean * scale, rounded to int)

include "circuit.mzn";

int: n = {n};
set of int: Locations = 1..n;

%% Maximum possible distance (used as upper bound for maxleg)
int: maxLegLen = {max_leg_len};

%% travel_time[i,j] = travel time from i to j;  -1 means no direct link
array[Locations, Locations] of int: travelTime = array2d(Locations, Locations, [
{travel_time_rows}
]);

%% Successor variables: succ[i] = next location after i in the tour
array[Locations] of var Locations: succ;

%% Only use edges that exist in the transport network
constraint forall(loc1, loc2 in Locations)(
  travelTime[loc1, loc2] < 0 -> succ[loc1] != loc2
);

%% Successors must form a Hamiltonian circuit
constraint circuit(succ);

%% Variable for the length of the longest leg
var 1..maxLegLen: maxleg;

%% maxleg >= travel time of every used leg
constraint forall(loc1, loc2 in Locations)(
  succ[loc1] == loc2 -> maxleg >= travelTime[loc1, loc2]
);

solve minimize maxleg;

output [
  "succ = ", show(succ), "\\n",
  "maxleg = ", show(maxleg), "\\n"
];
"""


def format_travel_time(travel_time: List[List[int]], n: int) -> str:
    """Format the 2-D travel-time array as a flat MiniZinc array literal."""
    lines = []
    for i in range(n):
        row = ", ".join(str(travel_time[i][j]) for j in range(n))
        comma = "," if i < n - 1 else ""
        lines.append(f"  {row}{comma}  %% from location {i + 1}")
    return "\n".join(lines)


def write_mzn(
    filepath: str,
    n: int,
    k: int,
    seed: int,
    scale: int,
    travel_time: List[List[int]],
) -> None:
    # Largest finite travel time as the upper bound for maxleg
    max_leg_len = max(
        travel_time[i][j]
        for i in range(n)
        for j in range(n)
        if travel_time[i][j] >= 0
    )

    travel_time_rows = format_travel_time(travel_time, n)

    content = MZN_TEMPLATE.format(
        n=n,
        k=k,
        seed=seed,
        scale=scale,
        max_leg_len=max_leg_len,
        travel_time_rows=travel_time_rows,
    )

    with open(filepath, "w") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate circuit-constraint benchmark instances "
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
) -> List[List[int]]:
    """Generate one instance and return its travel_time matrix."""
    rng = random.Random(seed)

    # Random locations in the unit square
    coords = [(rng.random(), rng.random()) for _ in range(n)]

    # Integer distance matrix
    dist = build_distance_matrix(coords, scale)

    # Transport network with guaranteed Hamiltonian circuit
    travel_time = build_graph(n, k, dist, rng)

    return travel_time


def main() -> None:
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine the base seed
    base_seed = args.seed if args.seed is not None else random.randrange(2 ** 32)

    print(f"Generating {args.count} instance(s):")
    print(f"  n={args.nodes}, k={args.neighbours}, scale={args.scale}")
    print(f"  base seed={base_seed}, output dir='{args.outdir}'")
    print()

    for idx in range(args.count):
        seed = base_seed + idx  # deterministic per-instance seed

        travel_time = generate_instance(
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
            scale=args.scale,
            travel_time=travel_time,
        )

        # Count edges for a quick sanity-check summary
        edges = sum(
            1
            for i in range(args.nodes)
            for j in range(i + 1, args.nodes)
            if travel_time[i][j] >= 0
        )
        print(f"  [{idx:4d}] seed={seed:10d}  edges={edges:5d}  -> {filepath}")

    print("\nDone.")


if __name__ == "__main__":
    main()