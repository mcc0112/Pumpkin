use pumpkin_checking::AtomicConstraint;
use pumpkin_checking::CheckerVariable;
use pumpkin_checking::InferenceChecker;
use pumpkin_checking::VariableState;

#[derive(Debug, Clone)]
pub struct AllDifferentChecker<Var> {
    pub successors: Box<[Var]>,
}

impl<Var, Atomic> InferenceChecker<Atomic> for AllDifferentChecker<Var>
where
    Var: CheckerVariable<Atomic>,
    Atomic: AtomicConstraint,
{
    fn check(
        &self,
        state: VariableState<Atomic>,
        _premises: &[Atomic],
        _consequent: Option<&Atomic>,
    ) -> bool {
        let n_vars = self.successors.len();

        // Collect all values reachable across all induced domains.
        // iter_induced_domain returns Option — None means the domain is empty,
        // which itself would be a conflict, so we treat it as no values.
        let mut all_vals: Vec<i32> = self
            .successors
            .iter()
            .flat_map(|var| {
                var.iter_induced_domain(&state)
                    .into_iter()
                    .flatten()
            })
            .collect();
        all_vals.sort_unstable();
        all_vals.dedup();

        let n_vals = all_vals.len();
        let val_index = |v: i32| all_vals.partition_point(|&x| x < v);

        let adj: Vec<Vec<usize>> = self
            .successors
            .iter()
            .map(|var| {
                var.iter_induced_domain(&state)
                    .into_iter()
                    .flatten()
                    .map(|v| val_index(v))
                    .collect()
            })
            .collect();

        hopcroft_karp_size(n_vars, n_vals, &adj) < n_vars
    }
}
// ---------------------------------------------------------------------------
// Hopcroft-Karp — returns only the matching size (no need for the full
// Matching struct here, but we keep the same algorithmic structure as the
// propagator for consistency and ease of review).
// ---------------------------------------------------------------------------

const UNMATCHED: usize = usize::MAX;
const INF_DIST: usize = usize::MAX;

fn hopcroft_karp_size(n_vars: usize, n_vals: usize, adj: &[Vec<usize>]) -> usize {
    let mut match_var = vec![UNMATCHED; n_vars];
    let mut match_val = vec![UNMATCHED; n_vals];
    let mut size = 0;

    loop {
        // BFS: build layered graph of shortest augmenting paths.
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

        // DFS: augment along vertex-disjoint shortest paths.
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
        let admissible = next == UNMATCHED
            || (dist[next] != INF_DIST && dist[next] == dist[i] + 1);

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


#[cfg(test)]
mod tests {
    use pumpkin_checking::TestAtomic;
    use pumpkin_checking::VariableState;
    use pumpkin_checking::Comparison;

    use super::*;

    fn eq(name: &'static str, value: i32) -> TestAtomic {
        TestAtomic { name, comparison: Comparison::Equal, value }
    }

    fn neq(name: &'static str, value: i32) -> TestAtomic {
        TestAtomic { name, comparison: Comparison::NotEqual, value }
    }

    fn make_checker(vars: Vec<&'static str>) -> AllDifferentChecker<&'static str> {
        AllDifferentChecker { successors: vars.into() }
    }

    //CONFLICT CASEs - check -> true
    #[test]
    fn conflict_two_vars_same_fixed_value() {
        let premises = [eq("x1", 2), eq("x2", 2)];
        let state = VariableState::prepare_for_conflict_check(premises, None).expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2"]);
        assert!(checker.check(state, &premises, None));
    }

    #[test]
    fn conflict_three_vars_two_values_hall_violation() {
        // x1, x2, x3 all fixed into {1, 2} — 3 vars, only 2 distinct values
        let premises = [eq("x1", 1), eq("x2", 2), eq("x3", 1)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, None));
    }
    #[test]
    fn conflict_all_fixed_to_same_value() {
        let premises = [eq("x1", 7), eq("x2", 7), eq("x3", 7)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(checker.check(state, &premises, None));
    }

    //NO CONFLICT -> false

    #[test]
    fn no_conflict_all_distinct_fixed() {
        // x1 == 1, x2 == 2, x3 == 3 — perfect matching exists
        let premises = [eq("x1", 1), eq("x2", 2), eq("x3", 3)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");
        let checker = make_checker(vec!["x1", "x2", "x3"]);
        assert!(!checker.check(state, &premises, None));
    }
    #[test]
    fn no_conflict_single_fixed_variable() {
        let premises = [eq("x1", 42)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");
        let checker = make_checker(vec!["x1"]);
        assert!(!checker.check(state, &premises, None));
    }




}