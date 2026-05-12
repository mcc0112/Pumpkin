include "globals.mzn";

int: n = 10;
array[1..n, 1..n] of int: dist = array2d(1..10, 1..10, [
  99,29,82,46,68,52,72,42,51,55,
  29,99,55,46,42,43,43,23,23,31,
  82,55,99,68,46,55,23,43,41,29,
  46,46,68,99,82,15,72,31,62,42,
  68,42,46,82,99,74,23,52,21,46,
  52,43,55,15,74,99,61,23,55,31,
  72,43,23,72,23,61,99,42,23,31,
  42,23,43,31,52,23,42,99,33,15,
  51,23,41,62,21,55,23,33,99,29,
  55,31,29,42,46,31,31,15,29,99
]);
array[1..n] of var 1..n: next;
var int: total_dist = sum(i in 1..n)(dist[i, next[i]]);

constraint circuit(next);

solve :: int_search(next, input_order, indomain_min, complete)
  minimize total_dist;

output ["next = ", show(next), "\ntotal = ", show(total_dist), "\n"];