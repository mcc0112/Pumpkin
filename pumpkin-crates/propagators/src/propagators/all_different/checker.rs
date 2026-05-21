use pumpkin_checking::AtomicConstraint;
use pumpkin_checking::CheckerVariable;
use pumpkin_checking::InferenceChecker;
use pumpkin_checking::VariableState;

#[derive(Debug, Clone)]
pub struct AllDifferentChecker<Var> {
    pub successors: Box<[Var]>,
}

/// Core verification logic for AllDifferent.
///
/// The checker is called with:
///   - `state`:      the domain state induced by the *premises* (the explanation).
///   - `premises`:   the literals that form the explanation.
///   - `consequent`: None  → verifying a conflict;
///                   Some(a) → verifying the pruning `xi ≠ v`.
///
/// ## Conflict verification  (consequent = None)
///
/// The premises describe a set S of variables confined to a neighbourhood N(S).
/// We must confirm |max_matching(S, N(S))| < |S|, i.e. Hall's condition is
/// violated. The `state` already encodes each variable's domain as induced by
/// the premises, so we just run Hopcroft-Karp and check the matching size.
///
/// ## Pruning verification  (consequent = Some(xi ≠ v))
///
/// The explanation is a tight Hall set T_vars whose neighbourhood T_vals
/// satisfies |T_vals| = |T_vars| (a saturated matching).  We need to prove
/// that under the premises, xi *cannot* take v.
///
/// The standard LCG verification rule is:
///   premises  ∧  ¬(consequent)  →  conflict
///
/// ¬(xi ≠ v)  is  xi = v.  So we add xi = v on top of the premise-induced
/// state and check whether a Hall violation (no perfect matching) follows.
/// If matching size < n_vars, the inference is valid.
///
/// Concretely: the premises pin T_vars to T_vals; adding xi = v means xi
/// competes for a value already fully consumed by T_vars, causing the
/// violation.
impl<Var, Atomic> InferenceChecker<Atomic> for AllDifferentChecker<Var>
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
            // ── Conflict case ────────────────────────────────────────────────
            None => {
                // Under the premise-induced domain, confirm Hall violation.
                let (n_vars, n_vals, adj) =
                    build_bipartite::<Var, Atomic>(&self.successors, &state, None);
                hopcroft_karp_size(n_vars, n_vals, &adj) < n_vars
            }

            // ── Pruning case: verify  premises ∧ (xi = v)  →  conflict ──────
            Some(consequent_atom) => {
                // Find which variable xi the consequent constrains.
                let pinned_var_idx = self
                    .successors
                    .iter()
                    .position(|var| var.does_atomic_constrain_self(consequent_atom));

                // The value that the consequent claims to remove.
                // For a `xi ≠ v` consequent, its negation is `xi = v`.
                // We pin xi to v and check for a Hall violation.
                let pruned_val = consequent_atom.value();

                let (n_vars, n_vals, adj) = build_bipartite::<Var, Atomic>(
                    &self.successors,
                    &state,
                    // Pin xi to pruned_val (negate the consequent).
                    pinned_var_idx.map(|idx| (idx, pruned_val)),
                );

                // If no perfect matching exists, the inference is valid.
                hopcroft_karp_size(n_vars, n_vals, &adj) < n_vars
            }
        }
    }
}

/// Build a bipartite variable/value adjacency list from the premise-induced
/// domain state, with an optional pin that forces one variable to a single value.
///
/// The pin implements  ¬(consequent):
///   - For pruning `xi ≠ v`, the negation is `xi = v`, so we pin variable xi
///     to the singleton domain {v}.
///
/// For all other variables, `iter_induced_domain` returns the domain as
/// narrowed by the premises.
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

    // Collect each variable's domain under the premise state, applying
    // the pin to the target variable.
    let domains: Vec<Vec<i32>> = successors
        .iter()
        .enumerate()
        .map(|(i, var)| {
            if let Some((pin_idx, pin_val)) = pin {
                if i == pin_idx {
                    // Negated consequent: restrict xi to exactly {pin_val}.
                    return vec![pin_val];
                }
            }
            // All other variables: use domain induced by premises.
            var.iter_induced_domain(state)
                .into_iter()
                .flatten()
                .collect()
        })
        .collect();

    // Build a compact value index (only values that actually appear).
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

// ---------------------------------------------------------------------------
// Hopcroft-Karp — returns only the matching size.
// ---------------------------------------------------------------------------

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

// ============================
// Tests
// ============================

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

    fn ge(name: &'static str, value: i32) -> TestAtomic {
        TestAtomic { name, comparison: Comparison::GreaterEqual, value }
    }

    fn le(name: &'static str, value: i32) -> TestAtomic {
        TestAtomic { name, comparison: Comparison::LessEqual, value }
    }

    fn make_checker(vars: Vec<&'static str>) -> AllDifferentChecker<&'static str> {
        AllDifferentChecker { successors: vars.into() }
    }

    // =========================================================
    // CONFLICT CASES  (consequent = None, should return true)
    // =========================================================

    #[test]
    fn conflict_two_vars_same_fixed_value() {
        let premises = [eq("x1", 2), eq("x2", 2)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atoms");
        let checker = make_checker(vec!["x1", "x2"]);
        assert!(checker.check(state, &premises, None));
    }

    #[test]
    fn conflict_three_vars_two_values_hall_violation() {
        // x1=1, x2=2, x3=1 — 3 vars, only 2 distinct values
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

    // =========================================================
    // NO-CONFLICT CASES  (consequent = None, should return false)
    // =========================================================

    #[test]
    fn no_conflict_all_distinct_fixed() {
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

    // =========================================================
    // PRUNING CASES  (consequent = Some(xi ≠ v), should return true)
    //
    // The premises describe a tight Hall set T_vars ⊆ {vars} \ {xi} whose
    // neighbourhood T_vals is exactly |T_vars| values.  The consequent says
    // xi ≠ v for some v ∈ T_vals.
    //
    // Verification: premises ∧ (xi = v) → Hall violation → return true.
    // =========================================================

    /// x0 ∈ {1,2}, x1 = 1 (fixed).
    /// Tight set = {x1}, T_vals = {1}.
    /// Pruning: x0 ≠ 1.
    /// Check: premises = [x1 = 1], pin x0 = 1 → both compete for 1 → conflict.
    #[test]
    fn pruning_fixed_var_removes_value_from_peer() {
        // Premises: x1 is fixed to 1 (describes the tight set {x1}).
        let premises = [eq("x1", 1)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");
        // Consequent: x0 ≠ 1.
        let consequent = neq("x0", 1);
        let checker = make_checker(vec!["x0", "x1"]);
        assert!(
            checker.check(state, &premises, Some(&consequent)),
            "x1=1 is a tight Hall set; pinning x0=1 should create a conflict"
        );
    }

    /// x0, x1 ∈ {1,2} — tight pair forming Hall set over {1,2}.
    /// Pruning: x2 ≠ 1  (and separately x2 ≠ 2).
    /// Premises: x0 ∈ {1,2} (lb/ub), x1 ∈ {1,2} (lb/ub) — domain confined.
    /// Check: pin x2 = 1 → x0, x1, x2 all fight for {1,2} → only 2 values for 3 vars.
    #[test]
    fn pruning_hall_pair_forces_third_var_away_from_1() {
        // Describe the tight pair {x0, x1} confined to {1, 2}.
        let premises = [ge("x0", 1), le("x0", 2), ge("x1", 1), le("x1", 2)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");
        let consequent = neq("x2", 1);
        let checker = make_checker(vec!["x0", "x1", "x2"]);
        assert!(
            checker.check(state, &premises, Some(&consequent)),
            "tight pair x0x1 over 1,2 means x2 cannot be 1"
        );
    }

    #[test]
    fn pruning_hall_pair_forces_third_var_away_from_2() {
        let premises = [ge("x0", 1), le("x0", 2), ge("x1", 1), le("x1", 2)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");
        let consequent = neq("x2", 2);
        let checker = make_checker(vec!["x0", "x1", "x2"]);
        assert!(
            checker.check(state, &premises, Some(&consequent)),
            "tight pair over means x2 cannot be 2"
        );
    }

    /// x0=1, x1=2 fixed.  Tight set = {x0, x1}, T_vals = {1, 2}.
    /// Pruning: x2 ≠ 1 and x2 ≠ 2.
    #[test]
    fn pruning_two_fixed_vars_force_third_away() {
        // Two fixed variables form the tight set.
        let premises = [eq("x0", 1), eq("x1", 2)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");

        let checker = make_checker(vec!["x0", "x1", "x2"]);

        let c1 = neq("x2", 1);
        assert!(
            checker.check(state.clone(), &premises, Some(&c1)),
            "x0=1,x1=2 fixed → x2 cannot be 1"
        );

        let c2 = neq("x2", 2);
        assert!(
            checker.check(state, &premises, Some(&c2)),
            "x0=1,x1=2 fixed → x2 cannot be 2"
        );
    }

    // =========================================================
    // INVALID PRUNING CASES  (should return false — the inference
    // is NOT justified by the given premises)
    // =========================================================

    /// x0 ∈ {1,2} and x1 ∈ {1,2} don't justify pruning value 3 from x2.
    #[test]
    fn invalid_pruning_value_not_in_tight_set() {
        let premises = [ge("x0", 1), le("x0", 2), ge("x1", 1), le("x1", 2)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");
        // x2 ≠ 3 is NOT supported — 3 is not in the tight set.
        let consequent = neq("x2", 3);
        let checker = make_checker(vec!["x0", "x1", "x2"]);
        assert!(
            !checker.check(state, &premises, Some(&consequent)),
            "value 3 is not in the tight set — pinning x2=3 should not create a conflict"
        );
    }

    /// Single fixed variable x0=1 does NOT justify removing 2 from x1.
    #[test]
    fn invalid_pruning_different_value() {
        let premises = [eq("x0", 1)];
        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");
        let consequent = neq("x1", 2);
        let checker = make_checker(vec!["x0", "x1"]);
        assert!(
            !checker.check(state, &premises, Some(&consequent)),
            "x0=1 does not prevent x1 from taking 2"
        );
    }
}