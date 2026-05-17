%% TSP benchmark instance
%% Source   : burma14
%% Comment  : 14-Staedte in Burma (Zaw Win)
%% Type     : TSP (symmetric)  (GEO)
%% Nodes    : 14
%% Mode     : satisfy (first good circuit)
%% Bounds   : lb=2022  ub=4048 (greedy nearest-neighbour tour)
%% Converted by tsp_to_mzn.py

include "circuit.mzn";

int: n = 14;
set of int: Cities = 1..n;

%% lb = sum of each city's cheapest outgoing edge  (valid lower bound)
%% ub = greedy nearest-neighbour tour cost         (valid upper bound)
int: lb = 2022;
int: ub = 4048;

%% dist[i,j] = travel cost from city i to city j  (1-indexed)
array[Cities, Cities] of int: dist = array2d(Cities, Cities, [
  0, 153, 510, 706, 966, 581, 455, 70, 160, 372, 157, 567, 342, 398,  %% from city 1
  153, 0, 422, 664, 997, 598, 507, 197, 311, 479, 310, 581, 417, 376,  %% from city 2
  510, 422, 0, 289, 744, 390, 437, 491, 645, 880, 618, 374, 455, 211,  %% from city 3
  706, 664, 289, 0, 491, 265, 410, 664, 804, 1070, 768, 259, 499, 310,  %% from city 4
  966, 997, 744, 491, 0, 400, 514, 902, 990, 1261, 947, 418, 635, 636,  %% from city 5
  581, 598, 390, 265, 400, 0, 168, 522, 634, 910, 593, 19, 284, 239,  %% from city 6
  455, 507, 437, 410, 514, 168, 0, 389, 482, 757, 439, 163, 124, 232,  %% from city 7
  70, 197, 491, 664, 902, 522, 389, 0, 154, 406, 133, 508, 273, 355,  %% from city 8
  160, 311, 645, 804, 990, 634, 482, 154, 0, 276, 43, 623, 358, 498,  %% from city 9
  372, 479, 880, 1070, 1261, 910, 757, 406, 276, 0, 318, 898, 633, 761,  %% from city 10
  157, 310, 618, 768, 947, 593, 439, 133, 43, 318, 0, 582, 315, 464,  %% from city 11
  567, 581, 374, 259, 418, 19, 163, 508, 623, 898, 582, 0, 275, 221,  %% from city 12
  342, 417, 455, 499, 635, 284, 124, 273, 358, 633, 315, 275, 0, 247,  %% from city 13
  398, 376, 211, 310, 636, 239, 232, 355, 498, 761, 464, 221, 247, 0  %% from city 14
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
solve satisfy;

output [
  "succ = ",       show(succ),       "\n",
  "tourLength = ", show(tourLength), "\n"
];
