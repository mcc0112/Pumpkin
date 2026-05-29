%% TSP benchmark instance
%% Source   : gr17
%% Comment  : 17-city problem (Groetschel)
%% Type     : TSP (symmetric)  (EXPLICIT)
%% Nodes    : 17
%% Mode     : satisfy (first good circuit)
%% Bounds   : lb=1258  ub=2187 (greedy nearest-neighbour tour)
%% Converted by tsp_to_mzn.py

include "circuit.mzn";

int: n = 17;
set of int: Cities = 1..n;

%% lb = sum of each city's cheapest outgoing edge  (valid lower bound)
%% ub = greedy nearest-neighbour tour cost         (valid upper bound)
int: lb = 1258;
int: ub = 2187;

%% dist[i,j] = travel cost from city i to city j  (1-indexed)
array[Cities, Cities] of int: dist = array2d(Cities, Cities, [
  0, 633, 257, 91, 412, 150, 80, 134, 259, 505, 353, 324, 70, 211, 268, 246, 121,  %% from city 1
  633, 0, 390, 661, 227, 488, 572, 530, 555, 289, 282, 638, 567, 466, 420, 745, 518,  %% from city 2
  257, 390, 0, 228, 169, 112, 196, 154, 372, 262, 110, 437, 191, 74, 53, 472, 142,  %% from city 3
  91, 661, 228, 0, 383, 120, 77, 105, 175, 476, 324, 240, 27, 182, 239, 237, 84,  %% from city 4
  412, 227, 169, 383, 0, 267, 351, 309, 338, 196, 61, 421, 346, 243, 199, 528, 297,  %% from city 5
  150, 488, 112, 120, 267, 0, 63, 34, 264, 360, 208, 329, 83, 105, 123, 364, 35,  %% from city 6
  80, 572, 196, 77, 351, 63, 0, 29, 232, 444, 292, 297, 47, 150, 207, 332, 29,  %% from city 7
  134, 530, 154, 105, 309, 34, 29, 0, 249, 402, 250, 314, 68, 108, 165, 349, 36,  %% from city 8
  259, 555, 372, 175, 338, 264, 232, 249, 0, 495, 352, 95, 189, 326, 383, 202, 236,  %% from city 9
  505, 289, 262, 476, 196, 360, 444, 402, 495, 0, 154, 578, 439, 336, 240, 685, 390,  %% from city 10
  353, 282, 110, 324, 61, 208, 292, 250, 352, 154, 0, 435, 287, 184, 140, 542, 238,  %% from city 11
  324, 638, 437, 240, 421, 329, 297, 314, 95, 578, 435, 0, 254, 391, 448, 157, 301,  %% from city 12
  70, 567, 191, 27, 346, 83, 47, 68, 189, 439, 287, 254, 0, 145, 202, 289, 55,  %% from city 13
  211, 466, 74, 182, 243, 105, 150, 108, 326, 336, 184, 391, 145, 0, 57, 426, 96,  %% from city 14
  268, 420, 53, 239, 199, 123, 207, 165, 383, 240, 140, 448, 202, 57, 0, 483, 153,  %% from city 15
  246, 745, 472, 237, 528, 364, 332, 349, 202, 685, 542, 157, 289, 426, 483, 0, 336,  %% from city 16
  121, 518, 142, 84, 297, 35, 29, 36, 236, 390, 238, 301, 55, 96, 153, 336, 0  %% from city 17
]);

%% succ[i] = next city after i in the Hamiltonian circuit
array[Cities] of var Cities: succ;

%% Hamiltonian circuit
constraint circuit(succ);

%% Total tour length, bounded tightly on both sides
var lb..ub: tourLength;
constraint tourLength = sum(i in Cities)(dist[i, succ[i]]);

%% Symmetry breaking: pred[i] is the predecessor of city i.
%% Requiring succ[1] < pred[1] eliminates all n cyclic
%% rotations of every tour, keeping one canonical direction.
array[Cities] of var Cities: pred;
constraint forall(i in Cities)(pred[succ[i]] = i);
constraint succ[1] < pred[1];
%% SATISFY MODE
%% The bound  tourLength <= ub  is already enforced by the variable domain.
%% The solver stops as soon as it finds the first feasible assignment —
%% no optimality proof required.  This makes failures/noGoods a clean
%% measure of pure circuit propagator search reduction.
solve :: int_search(succ, input_order, indomain_min) satisfy;

output [
  "succ = ",       show(succ),       "\n",
  "tourLength = ", show(tourLength), "\n"
];
