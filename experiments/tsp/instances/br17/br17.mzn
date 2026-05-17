%% TSP benchmark instance
%% Source   : br17
%% Comment  : 17 city problem (Repetto)
%% Type     : ATSP (asymmetric)  (EXPLICIT)
%% Nodes    : 17
%% Mode     : satisfy (first good circuit)
%% Bounds   : lb=0  ub=87 (greedy nearest-neighbour tour)
%% Converted by tsp_to_mzn.py

include "circuit.mzn";

int: n = 17;
set of int: Cities = 1..n;

%% lb = sum of each city's cheapest outgoing edge  (valid lower bound)
%% ub = greedy nearest-neighbour tour cost         (valid upper bound)
int: lb = 0;
int: ub = 87;

%% dist[i,j] = travel cost from city i to city j  (1-indexed)
array[Cities, Cities] of int: dist = array2d(Cities, Cities, [
  9999, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5,  %% from city 1
  3, 9999, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5,  %% from city 2
  5, 3, 9999, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24,  %% from city 3
  48, 48, 74, 9999, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12,  %% from city 4
  48, 48, 74, 0, 9999, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12,  %% from city 5
  8, 8, 50, 6, 6, 9999, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8,  %% from city 6
  8, 8, 50, 6, 6, 0, 9999, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8,  %% from city 7
  5, 5, 26, 12, 12, 8, 8, 9999, 0, 5, 5, 5, 5, 26, 8, 8, 0,  %% from city 8
  5, 5, 26, 12, 12, 8, 8, 0, 9999, 5, 5, 5, 5, 26, 8, 8, 0,  %% from city 9
  3, 0, 3, 48, 48, 8, 8, 5, 5, 9999, 0, 3, 0, 3, 8, 8, 5,  %% from city 10
  3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 9999, 3, 0, 3, 8, 8, 5,  %% from city 11
  0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 9999, 3, 5, 8, 8, 5,  %% from city 12
  3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 9999, 3, 8, 8, 5,  %% from city 13
  5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 9999, 48, 48, 24,  %% from city 14
  8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 9999, 0, 8,  %% from city 15
  8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 9999, 8,  %% from city 16
  5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 9999  %% from city 17
]);

%% succ[i] = next city after i in the Hamiltonian circuit
array[Cities] of var Cities: succ;

%% Hamiltonian circuit
constraint circuit(succ);

%% Total tour length, bounded tightly on both sides
var lb..ub: tourLength;
constraint tourLength = sum(i in Cities)(dist[i, succ[i]]);

%% No symmetry breaking for ATSP (reversals change cost).

%% SATISFY MODE
%% The bound  tourLength <= ub  is already enforced by the variable domain.
%% The solver stops as soon as it finds the first feasible assignment —
%% no optimality proof required.  This makes failures/noGoods a clean
%% measure of pure circuit propagator search reduction.
solve satisfy;

output [
  "succ = ",       show(succ),       "\n",
  "tourLength = ", show(tourLength), "\n"
];
