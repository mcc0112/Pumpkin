include "circuit.mzn";

int: n;
set of int: Cities = 1..n;

int: lb;
int: ub;

array[Cities, Cities] of int: dist;

array[Cities] of var Cities: succ;

constraint circuit(succ);

var lb..ub: tourLength;

constraint
    tourLength =
        sum(i in Cities)(
            dist[i, succ[i]]
        );

array[Cities] of var Cities: pred;

constraint
    forall(i in Cities)(
        pred[succ[i]] = i
    );

constraint succ[1] < pred[1];

solve :: int_search(
    succ,
    input_order,
    indomain_min
) satisfy;

output [
    "succ = ", show(succ), "\n",
    "tourLength = ", show(tourLength), "\n"
];