

use fixedbitset::FixedBitSet;
use pumpkin_checking::AtomicConstraint;
use pumpkin_checking::CheckerVariable;
use pumpkin_checking::InferenceChecker;
use pumpkin_checking::VariableState;


const VALUE_OFFSET: i32 = 1;

#[inline]
fn domain_value_to_index(domain_value: i32) -> usize {
    (domain_value - VALUE_OFFSET) as usize
}

#[inline]
fn index_to_domain_value(index: usize) -> i32 {
    index as i32 + VALUE_OFFSET
}


#[derive(Debug, Clone)]
pub struct CircuitChecker<Var> {
    pub successors: Box<[Var]>,
}


impl<Var, Atomic> InferenceChecker<Atomic> for CircuitChecker<Var>
where
    Var: CheckerVariable<Atomic>,
    Atomic: AtomicConstraint,
{
    fn check(
        &self,
        state: VariableState<Atomic>,
        premises: &[Atomic],
        consequent: Option<&Atomic>,
    ) -> bool {
        match consequent {
            None => check_conflict(&self.successors, &state, premises),
            Some(c) => check_pruning(&self.successors, &state, premises, c),
        }
    }
}

// ─── Conflict verification ────────────────────────────────────────────────────

/// Verify a conflict inference.
///
/// Returns `true` if either:
///   - The premise-induced domains admit no perfect bipartite matching
///     (AllDifferent Hall violation), or
///   - The fixed edges form a sub-cycle shorter than n (circuit conflict).
/// 
/// TODO: change to cehck for hall set
fn check_conflict<Var, Atomic>(
    successors: &[Var],
    state: &VariableState<Atomic>,
    _premises: &[Atomic],
) -> bool
where
    Var: CheckerVariable<Atomic>,
    Atomic: AtomicConstraint,
{
    let (n_vars, n_vals, adj) = build_bipartite(successors, state, None);
    if hopcroft_karp_size(n_vars, n_vals, &adj) < n_vars {
        return true;
    }
    has_subcycle(successors, state)
}

/// Follow fixed edges from every start looking for a cycle shorter than n.
fn has_subcycle<Var, Atomic>(
    successors: &[Var],
    state: &VariableState<Atomic>,
) -> bool
where
    Var: CheckerVariable<Atomic>,
    Atomic: AtomicConstraint,
{
    let n = successors.len();
    for start_idx in 0..n {
        let Some(first_val) = successors[start_idx].induced_fixed_value(state) else {
            continue;
        };
        let mut visited = FixedBitSet::with_capacity(n);
        visited.insert(start_idx);
        let mut current_idx = domain_value_to_index(first_val);
        loop {
            if current_idx >= n {
                break;
            }
            if visited.contains(current_idx) {
                // Closed a cycle — valid only if it covers all n nodes.
                if visited.count_ones(..) == n {
                    return false;
                }
                return true; // proper sub-cycle
            }
            visited.insert(current_idx);
            let Some(next_val) = successors[current_idx].induced_fixed_value(state) else {
                break;
            };
            current_idx = domain_value_to_index(next_val);
        }
    }
    false
}

// ─── Pruning verification ─────────────────────────────────────────────────────

/// Verify a pruning inference `xi ≠ v`.
///
/// Returns `true` if any of:
///   - Branch A: Pinning xi = v makes a perfect matching impossible
///     (AllDiff GAC justification).
///   - Branch B: The premises describe a fixed chain ending at xi whose head
///     is v, with chain length < n (circuit nocycle justification).
///   - Branch C: There exists a tight Hall set H where xi is the unique entry
///     and v is the unique exit (Theorem 4.2 Hall-circuit justification).
fn check_pruning<Var, Atomic>(
    successors: &[Var],
    state: &VariableState<Atomic>,
    _premises: &[Atomic],
    consequent: &Atomic,
) -> bool
where
    Var: CheckerVariable<Atomic>,
    Atomic: AtomicConstraint,
{
    let pruned_val = consequent.value();
    let pinned_var_idx = successors
        .iter()
        .position(|var| var.does_atomic_constrain_self(consequent));

    // Branch A: AllDiff via matching.

    let (n_vars, n_vals, adj) = build_bipartite(
        successors,
        state,
        pinned_var_idx.map(|idx| (idx, pruned_val)),
    );
    if hopcroft_karp_size(n_vars, n_vals, &adj) < n_vars {
        return true;
    }

    if let Some(tail_idx) = pinned_var_idx {
        // Branch B: circuit nocycle prevention.
        if would_close_premature_chain(successors, state, tail_idx, pruned_val) {
            return true;
        }

        // Branch C: Theorem 4.2 Hall-circuit pruning.
        if is_hall_circuit_pruning(successors, state, tail_idx, pruned_val) {
            return true;
        }
    }

    false
}

/// Returns `true` when there is a fixed chain starting at `head_val`
fn would_close_premature_chain<Var, Atomic>(
    successors: &[Var],
    state: &VariableState<Atomic>,
    tail_idx: usize,
    head_val: i32,
) -> bool
where
    Var: CheckerVariable<Atomic>,
    Atomic: AtomicConstraint,
{
    let n = successors.len();
    let head_idx = domain_value_to_index(head_val);
    if head_idx >= n {
        return false;
    }

    let mut visited = FixedBitSet::with_capacity(n);
    visited.insert(head_idx);
    let mut current = head_idx;

    loop {
        let Some(next_val) = successors[current].induced_fixed_value(state) else {
            break;
        };
        let next_idx = domain_value_to_index(next_val);
        if next_idx >= n || visited.contains(next_idx) {
            break;
        }
        visited.insert(next_idx);
        current = next_idx;
    }

    // Valid nocycle prune iff chain ends at tail_idx and is shorter than n.
    current == tail_idx && visited.count_ones(..) < n
}

/// Verify Theorem 4.2 by enumerating all proper non-empty subsets of variables
/// and checking whether any tight Hall set has `pruned_var` as its unique entry
/// and `pruned_val` as its unique exit.

fn is_hall_circuit_pruning<Var, Atomic>(
    successors: &[Var],
    state: &VariableState<Atomic>,
    pruned_var: usize,
    pruned_val: i32,
) -> bool
where
    Var: CheckerVariable<Atomic>,
    Atomic: AtomicConstraint,
{
    let n = successors.len();

    // Collect induced domain per variable.
    // Unconstrained variables (no premises) get the full 1..=n domain,
    // meaning "no information restricts this variable".
    let domains: Vec<Vec<i32>> = successors
        .iter()
        .map(|var| {
            let induced: Vec<i32> = var
                .iter_induced_domain(state)
                .into_iter()
                .flatten()
                .collect();
            if induced.is_empty() {
                (VALUE_OFFSET..=(n as i32 + VALUE_OFFSET - 1)).collect()
            } else {
                induced
            }
        })
        .collect();

    // Enumerate all non-empty proper subsets of {0, ..., n-1}.
    let n_subsets = 1usize << n;
    for mask in 1..n_subsets {
        // Skip the full set — Theorem 4.2 requires a strict subset.
        if mask == n_subsets - 1 {
            continue;
        }

        let hall_vars: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();

        // Union of domains of variables in this subset.
        let mut domain_union: Vec<i32> = hall_vars
            .iter()
            .flat_map(|&i| domains[i].iter().copied())
            .collect();
        domain_union.sort_unstable();
        domain_union.dedup();

        // Tight Hall condition: |vars| == |domain union|.
        if hall_vars.len() != domain_union.len() {
            continue;
        }

        // I (entries): variables in H whose 1-indexed node id is NOT in the
        // domain union.  `index_to_domain_value` converts 0-indexed position
        // to 1-indexed node id, consistent with the MiniZinc convention.
        let i_set: Vec<usize> = hall_vars
            .iter()
            .copied()
            .filter(|&i| !domain_union.contains(&index_to_domain_value(i)))
            .collect();

        // O (exits): domain values in the union whose corresponding node
        // (domain_value_to_index) is NOT in H.
        let o_set: Vec<i32> = domain_union
            .iter()
            .copied()
            .filter(|&dv| {
                let zero_idx = domain_value_to_index(dv);
                zero_idx >= n || !hall_vars.contains(&zero_idx)
            })
            .collect();

        // Theorem 4.2 fires only when there is exactly one entry and one exit.
        if i_set.len() == 1
            && o_set.len() == 1
            && i_set[0] == pruned_var
            && o_set[0] == pruned_val
        {
            return true;
        }
    }

    false
}

// ─── Bipartite graph construction ─────────────────────────────────────────────


fn build_bipartite<Var, Atomic>(
    successors: &[Var],
    state: &VariableState<Atomic>,
    pin: Option<(usize, i32)>,
) -> (usize, usize, Vec<Vec<usize>>)
where
    Var: CheckerVariable<Atomic>,
    Atomic: AtomicConstraint,
{
    let n_vars = successors.len();

    let domains: Vec<Vec<i32>> = successors
        .iter()
        .enumerate()
        .map(|(i, var)| {
            if let Some((pin_idx, pin_val)) = pin {
                if i == pin_idx {
                    return vec![pin_val];
                }
            }
            let induced: Vec<i32> = var
                .iter_induced_domain(state)
                .into_iter()
                .flatten()
                .collect();
            if induced.is_empty() {
                (VALUE_OFFSET..=(n_vars as i32 + VALUE_OFFSET - 1)).collect()
            } else {
                induced
            }
        })
        .collect();

    // Compress values into a dense 0-indexed range.
    let mut all_vals: Vec<i32> = domains.iter().flatten().copied().collect();
    all_vals.sort_unstable();
    all_vals.dedup();
    let n_vals = all_vals.len();
    let val_index = |v: i32| all_vals.partition_point(|&x| x < v);

    let adj: Vec<Vec<usize>> = domains
        .iter()
        .map(|dom| dom.iter().map(|&v| val_index(v)).collect())
        .collect();

    (n_vars, n_vals, adj)
}

// ─── Hopcroft-Karp (size only) ────────────────────────────────────────────────

const UNMATCHED: usize = usize::MAX;
const INF_DIST: usize = usize::MAX;

fn hopcroft_karp_size(n_vars: usize, n_vals: usize, adj: &[Vec<usize>]) -> usize {
    let mut match_var = vec![UNMATCHED; n_vars];
    let mut match_val = vec![UNMATCHED; n_vals];
    let mut size = 0;

    loop {
        let mut dist = vec![INF_DIST; n_vars];
        let mut queue = std::collections::VecDeque::new();

        for i in 0..n_vars {
            if match_var[i] == UNMATCHED {
                dist[i] = 0;
                queue.push_back(i);
            }
        }

        let mut found_augmenting = false;
        while let Some(i) = queue.pop_front() {
            for &v in &adj[i] {
                let next = match_val[v];
                if next == UNMATCHED {
                    found_augmenting = true;
                } else if dist[next] == INF_DIST {
                    dist[next] = dist[i] + 1;
                    queue.push_back(next);
                }
            }
        }

        if !found_augmenting {
            break;
        }

        for i in 0..n_vars {
            if match_var[i] == UNMATCHED
                && dfs_augment(i, adj, &mut match_var, &mut match_val, &mut dist)
            {
                size += 1;
            }
        }
    }

    size
}

fn dfs_augment(
    i: usize,
    adj: &[Vec<usize>],
    match_var: &mut Vec<usize>,
    match_val: &mut Vec<usize>,
    dist: &mut [usize],
) -> bool {
    for &v in &adj[i] {
        let next = match_val[v];
        let admissible =
            next == UNMATCHED || (dist[next] != INF_DIST && dist[next] == dist[i] + 1);

        if admissible {
            let augmented =
                next == UNMATCHED || dfs_augment(next, adj, match_var, match_val, dist);
            if augmented {
                match_var[i] = v;
                match_val[v] = i;
                dist[i] = INF_DIST;
                return true;
            }
        }
    }
    dist[i] = INF_DIST;
    false
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use pumpkin_checking::Comparison;
    use pumpkin_checking::TestAtomic;
    use pumpkin_checking::VariableState;

    use super::*;

    fn eq(name: &'static str, value: i32) -> TestAtomic {
        TestAtomic { name, comparison: Comparison::Equal, value }
    }

    fn neq(name: &'static str, value: i32) -> TestAtomic {
        TestAtomic { name, comparison: Comparison::NotEqual, value }
    }

    fn ge(name: &'static str, value: i32) -> TestAtomic {
        TestAtomic { name, comparison: Comparison::GreaterEqual, value }
    }

    fn le(name: &'static str, value: i32) -> TestAtomic {
        TestAtomic { name, comparison: Comparison::LessEqual, value }
    }

    fn make_checker(vars: Vec<&'static str>) -> CircuitChecker<&'static str> {
        CircuitChecker { successors: vars.into() }
    }

    // =========================================================================
    // CONFLICT — AllDifferent Hall violation
    // =========================================================================

    #[test]
    fn conflict_alldiff_two_vars_same_singleton() {
        let premises = [eq("x1", 2), eq("x2", 2)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, None));
    }

    #[test]
    fn conflict_alldiff_three_vars_two_values() {
        let premises = [
            ge("x1", 2), le("x1", 3),
            ge("x2", 2), le("x2", 3),
            ge("x3", 2), le("x3", 3),
        ];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, None));
    }

    #[test]
    fn conflict_alldiff_all_fixed_to_same() {
        let premises = [eq("x1", 5), eq("x2", 5), eq("x3", 5)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, None));
    }

    // =========================================================================
    // CONFLICT — circuit sub-cycle
    // =========================================================================

    #[test]
    fn conflict_circuit_simple_subcycle() {
        // 1→2→1 in a 3-node graph
        let premises = [eq("x1", 2), eq("x2", 1)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, None));
    }

    #[test]
    fn conflict_circuit_self_loop() {
        let premises = [eq("x1", 1)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, None));
    }

    #[test]
    fn conflict_circuit_three_node_subcycle_in_four_node_graph() {
        // 1→2→3→1 in a 4-node graph
        let premises = [eq("x1", 2), eq("x2", 3), eq("x3", 1)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3", "x4"]);
        assert!(checker.check(state, &premises, None));
    }

    // =========================================================================
    // NO CONFLICT — valid states
    // =========================================================================

    #[test]
    fn no_conflict_full_hamiltonian_cycle() {
        // 1→2→3→1 in a 3-node graph — valid
        let premises = [eq("x1", 2), eq("x2", 3), eq("x3", 1)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(!checker.check(state, &premises, None));
    }

    #[test]
    fn no_conflict_all_distinct_fixed_with_free_node() {
        let premises = [eq("x1", 2), eq("x2", 3), eq("x3", 4)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3", "x4"]);
        assert!(!checker.check(state, &premises, None));
    }

    #[test]
    fn no_conflict_incomplete_chain() {
        let premises = [eq("x1", 2), eq("x2", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(!checker.check(state, &premises, None));
    }

    #[test]
    fn no_conflict_two_node_hamiltonian() {
        let premises = [eq("x1", 2), eq("x2", 1)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2"]);
        assert!(!checker.check(state, &premises, None));
    }

    // =========================================================================
    // PRUNING — AllDiff GAC
    // =========================================================================

    #[test]
    fn pruning_alldiff_fixed_var_removes_value_from_peer() {
        let premises = [eq("x2", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, Some(&neq("x1", 3))));
    }

    #[test]
    fn pruning_alldiff_tight_pair_forces_third_away_from_2() {
        let premises = [ge("x1", 2), le("x1", 3), ge("x2", 2), le("x2", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, Some(&neq("x3", 2))));
    }

    #[test]
    fn pruning_alldiff_tight_pair_forces_third_away_from_3() {
        let premises = [ge("x1", 2), le("x1", 3), ge("x2", 2), le("x2", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, Some(&neq("x3", 3))));
    }

    #[test]
    fn pruning_alldiff_two_fixed_vars_force_third() {
        let premises = [eq("x1", 2), eq("x2", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state.clone(), &premises, Some(&neq("x3", 2))));
        assert!(checker.check(state, &premises, Some(&neq("x3", 3))));
    }

    // =========================================================================
    // PRUNING — circuit nocycle
    // =========================================================================

    #[test]
    fn pruning_circuit_nocycle_chain_of_one_in_three_nodes() {
        // Chain 1→2, length 1 < 3; x2≠1 is premature
        let premises = [eq("x1", 2)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, Some(&neq("x2", 1))));
    }

    #[test]
    fn pruning_circuit_nocycle_chain_of_two_in_four_nodes() {
        // Chain 1→2→3, length 2 < 4; x3≠1 is premature
        let premises = [eq("x1", 2), eq("x2", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3", "x4"]);
        assert!(checker.check(state, &premises, Some(&neq("x3", 1))));
    }

    #[test]
    fn pruning_circuit_nocycle_does_not_prune_completing_edge() {
        // Chain 1→2→3 in a 3-node graph; x3→1 completes the cycle — invalid prune
        let premises = [eq("x1", 2), eq("x2", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(!checker.check(state, &premises, Some(&neq("x3", 1))));
    }

    // =========================================================================
    // PRUNING — HallCircuit Theorem 4.2
    // =========================================================================

    #[test]
    fn pruning_hall_circuit_entry_cannot_take_internal_value() {
        // Tight Hall set {x2,x3} over {2,3}; pinning x1=2 → 3 vars, 2 values.
        let premises = [ge("x2", 2), le("x2", 3), ge("x3", 2), le("x3", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, Some(&neq("x1", 2))));
    }

    #[test]
    fn pruning_hall_circuit_entry_cannot_take_other_internal_value() {
        let premises = [ge("x2", 2), le("x2", 3), ge("x3", 2), le("x3", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, Some(&neq("x1", 3))));
    }

    #[test]
    fn pruning_hall_circuit_combined_chain_and_hall() {
        let premises = [
            eq("x1", 2),
            ge("x3", 3), le("x3", 4),
            ge("x4", 3), le("x4", 4),
        ];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3", "x4"]);

        // Nocycle: chain 1→2, length 1 < 4.
        assert!(checker.check(state.clone(), &premises, Some(&neq("x2", 1))));
        // AllDiff: tight pair {x3,x4} over {3,4}.
        assert!(checker.check(state.clone(), &premises, Some(&neq("x2", 3))));
        assert!(checker.check(state, &premises, Some(&neq("x2", 4))));
    }

    // =========================================================================
    // INVALID PRUNINGS
    // =========================================================================

    #[test]
    fn invalid_pruning_value_outside_tight_set() {
        // Tight pair {x1,x2} over {2,3}; pinning x3=4 → matching succeeds.
        let premises = [ge("x1", 2), le("x1", 3), ge("x2", 2), le("x2", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(!checker.check(state, &premises, Some(&neq("x3", 4))));
    }

    #[test]
    fn invalid_pruning_completing_hamiltonian_edge() {
        // Chain 1→2→3 in a 3-node graph; x3→1 completes the cycle.
        let premises = [eq("x1", 2), eq("x2", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(!checker.check(state, &premises, Some(&neq("x3", 1))));
    }

    #[test]
    fn invalid_pruning_no_premises() {
        let premises: [TestAtomic; 0] = [];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(!checker.check(state, &premises, Some(&neq("x1", 2))));
    }

    #[test]
    fn invalid_pruning_wrong_value_for_fixed_var() {
        // x1=2; x2≠3 has no justification — 3 is not taken by anyone.
        let premises = [eq("x1", 2)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(!checker.check(state, &premises, Some(&neq("x2", 3))));
    }
}