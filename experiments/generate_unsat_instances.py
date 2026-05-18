"""
generate_unsat_instances.py  –  UNSAT circuit benchmark instance generator
===========================================================================

Generates MiniZinc instances that are guaranteed UNSAT by construction,
by omitting the random-walk Hamiltonicity guarantee used in the SAT generator.
The graph is built identically (k-nearest geographic neighbours) but no edges
are added to ensure a Hamiltonian circuit exists.  A fraction of these will
naturally lack any Hamiltonian circuit; we verify this at generation time by
running a quick reachability check, and only keep instances confirmed UNSAT.

Two construction strategies are provided:

  Strategy A  --unsat-mode=random  (default)
    Simply skip the random walk.  Works well at low k where the graph is
    sparse enough that Hamiltonicity is unlikely.  Instances that happen
    to be SAT are discarded and regenerated with a fresh seed.

  Strategy B  --unsat-mode=forced
    After building the k-nearest graph, partition nodes into two groups of
    roughly equal size and remove all edges crossing from group A to group B.
    This creates a directed Hall-set violation: nodes in group A can only
    reach other nodes in group A, so no Hamiltonian circuit can exist.
    The violation is detectable immediately by matching-based reasoning but
    requires exhaustive search by a decomposed AllDifferent.

Usage
-----
python generate_unsat_instances.py [options]

Options
-------
  -n, --nodes        INT    Number of locations        (default: 20)
  -k, --neighbours   INT    k-nearest degree           (default: 5)
  -c, --count        INT    Number of instances        (default: 5)
  -s, --seed         INT    Base random seed
  -o, --outdir       PATH   Output directory
  --unsat-mode       STR    'random' or 'forced'       (default: random)
  --scale            INT    Distance scale factor      (default: 1000)
  --prefix           STR    Filename prefix            (default: 'unsat')
  --max-attempts     INT    Max seed attempts before giving up (default: 200)
"""

import argparse
import math
import os
import random
import sys
from typing import List, Set, Tuple, Optional


# ---------------------------------------------------------------------------
# Distance helpers  (identical to SAT generator)
# ---------------------------------------------------------------------------

def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def build_distance_matrix(
    coords: List[Tuple[float, float]],
    scale: int,
) -> List[List[int]]:
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = round(euclidean(coords[i], coords[j]) * scale)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def k_nearest_edges(dist: List[List[int]], k: int) -> Set[Tuple[int, int]]:
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


# ---------------------------------------------------------------------------
# UNSAT verification  –  necessary (not sufficient) check via out-degree
# ---------------------------------------------------------------------------

def has_any_hamiltonian_circuit_heuristic(allowed: List[List[bool]], n: int) -> bool:
    """
    Quick necessary-condition check: if any node has out-degree 0 the
    instance is trivially UNSAT.  This is not a full Hamiltonian check
    but is fast and catches most cases at low k.

    For strategy B (forced partition), the instance is UNSAT by construction
    so we skip this check there.
    """
    for i in range(n):
        if not any(allowed[i][j] for j in range(n) if j != i):
            return False   # node i has no valid successor -> UNSAT
    return True   # might still be UNSAT but not trivially so


def reachability_unsat(allowed: List[List[bool]], n: int) -> bool:
    """
    Stronger check: from every node, can we reach all other nodes?
    If not, no Hamiltonian circuit exists.  Uses BFS.
    Returns True if the instance is UNSAT (some node cannot reach all others).
    """
    for start in range(n):
        visited = [False] * n
        visited[start] = True
        queue = [start]
        while queue:
            cur = queue.pop()
            for nxt in range(n):
                if nxt != cur and allowed[cur][nxt] and not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        if not all(visited):
            return True   # start cannot reach some node -> UNSAT
    return False   # full reachability exists (may still be UNSAT but harder to tell)


# ---------------------------------------------------------------------------
# Strategy A: random k-nearest without Hamiltonicity guarantee
# ---------------------------------------------------------------------------

def build_random_unsat(
    n: int,
    k: int,
    dist: List[List[int]],
) -> Optional[List[List[bool]]]:
    """
    Build a k-nearest graph without the random walk.
    Returns the allowed matrix if reachability check says UNSAT, else None.
    """
    edge_set = k_nearest_edges(dist, k)

    allowed = [[False] * n for _ in range(n)]
    for (i, j) in edge_set:
        allowed[i][j] = True
        allowed[j][i] = True

    if reachability_unsat(allowed, n):
        return allowed
    return None


# ---------------------------------------------------------------------------
# Strategy B: forced Hall-set partition
# ---------------------------------------------------------------------------

def build_forced_unsat(
    n: int,
    k: int,
    dist: List[List[int]],
    rng: random.Random,
) -> List[List[bool]]:
    """
    Build a k-nearest graph then remove all edges from group A -> group B,
    where group A = nodes 0..split-1 and group B = nodes split..n-1.
    Group A nodes can only reach other group A nodes, creating a Hall-set
    violation detectable by matching-based reasoning in one propagation step.

    The split is chosen randomly around n//2 to vary the violation size.
    """
    edge_set = k_nearest_edges(dist, k)

    # Random split size between n//3 and 2*n//3 to vary Hall set size
    split = rng.randint(n // 3, 2 * n // 3)
    group_a = set(range(split))

    allowed = [[False] * n for _ in range(n)]
    for (i, j) in edge_set:
        # Remove edges from group A -> group B (but keep B -> A and A -> A)
        if i in group_a and j not in group_a:
            # Only add the reverse direction (B -> A)
            allowed[j][i] = True
        elif j in group_a and i not in group_a:
            allowed[i][j] = True
        else:
            # Both in same group: keep both directions
            allowed[i][j] = True
            allowed[j][i] = True

    return allowed, split


# ---------------------------------------------------------------------------
# MiniZinc writer
# ---------------------------------------------------------------------------

MZN_TEMPLATE = """\
%% Circuit-constraint UNSAT benchmark instance
%% Generated by generate_unsat_instances.py
%%
%% Parameters
%%   n       = {n}
%%   k       = {k}
%%   seed    = {seed}
%%   mode    = {mode}
%%   edges   = {num_edges}
%%   {extra_comment}

include "circuit.mzn";

int: n = {n};
set of int: Locations = 1..n;

array[Locations, Locations] of bool: allowed = array2d(Locations, Locations, [
{allowed_rows}
]);

array[Locations] of var Locations: succ;

constraint forall(i in Locations)(
  succ[i] in {{j | j in Locations where allowed[i, j]}}
);

constraint circuit(succ);

solve satisfy;

output [
  "succ = ", show(succ), "\\n"
];
"""


def format_allowed(allowed: List[List[bool]], n: int) -> str:
    lines = []
    for i in range(n):
        row = ", ".join("true" if allowed[i][j] else "false" for j in range(n))
        comma = "," if i < n - 1 else ""
        lines.append(f"  {row}{comma}  %% from location {i + 1}")
    return "\n".join(lines)


def count_edges(allowed: List[List[bool]], n: int) -> int:
    return sum(1 for i in range(n) for j in range(n) if i < j and allowed[i][j])


def write_mzn(filepath, n, k, seed, mode, allowed, extra_comment=""):
    num_edges = count_edges(allowed, n)
    content = MZN_TEMPLATE.format(
        n=n,
        k=k,
        seed=seed,
        mode=mode,
        num_edges=num_edges,
        extra_comment=extra_comment,
        allowed_rows=format_allowed(allowed, n),
    )
    with open(filepath, "w") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate UNSAT circuit benchmark instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", "--nodes",       type=int, default=20)
    p.add_argument("-k", "--neighbours",  type=int, default=5)
    p.add_argument("-c", "--count",       type=int, default=5)
    p.add_argument("-s", "--seed",        type=int, default=None)
    p.add_argument("-o", "--outdir",      type=str, default=".")
    p.add_argument("--unsat-mode",        type=str, default="random",
                   choices=["random", "forced"],
                   help="UNSAT construction strategy")
    p.add_argument("--scale",             type=int, default=1000)
    p.add_argument("--prefix",            type=str, default="unsat")
    p.add_argument("--max-attempts",      type=int, default=200,
                   help="Max seed attempts per instance (random mode only)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base_seed = args.seed if args.seed is not None else random.randrange(2 ** 32)
    print(f"Generating {args.count} UNSAT instance(s):")
    print(f"  n={args.nodes}, k={args.neighbours}, mode={args.unsat_mode}")
    print(f"  base seed={base_seed}, outdir='{args.outdir}'")
    print()

    generated = 0
    attempt = 0

    while generated < args.count:
        if attempt > args.max_attempts:
            print(
                f"WARNING: gave up after {attempt} attempts; "
                f"only generated {generated}/{args.count} instances.\n"
                f"Try reducing k or using --unsat-mode=forced.",
                file=sys.stderr,
            )
            break

        seed = base_seed + attempt
        attempt += 1
        rng = random.Random(seed)

        coords = [(rng.random(), rng.random()) for _ in range(args.nodes)]
        dist = build_distance_matrix(coords, args.scale)

        if args.unsat_mode == "random":
            allowed = build_random_unsat(args.nodes, args.k, dist)
            if allowed is None:
                continue   # this seed happened to be SAT-reachable, skip it
            extra = "No Hamiltonicity guarantee (random walk omitted)"

        else:  # forced
            allowed, split = build_forced_unsat(args.nodes, args.k, dist, rng)
            extra = (
                f"Forced Hall violation: nodes 1..{split} form isolated group "
                f"(no outgoing edges to nodes {split+1}..{args.nodes})"
            )

        filename = (
            f"{args.prefix}_n{args.nodes}_k{args.neighbours}"
            f"_{args.unsat_mode}_{generated:04d}.mzn"
        )
        filepath = os.path.join(args.outdir, filename)
        write_mzn(filepath, args.nodes, args.k, seed, args.unsat_mode, allowed, extra)

        print(
            f"  [{generated:4d}] seed={seed:10d}  attempt={attempt:4d}  "
            f"-> {filename}"
        )
        generated += 1

    print("\nDone.")


if __name__ == "__main__":
    main()