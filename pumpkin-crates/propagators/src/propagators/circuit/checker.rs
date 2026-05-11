use fixedbitset::FixedBitSet;
use pumpkin_checking::AtomicConstraint;
use pumpkin_checking::CheckerVariable;
use pumpkin_checking::InferenceChecker;

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
        state: pumpkin_checking::VariableState<Atomic>,
        _premises: &[Atomic],
        _consequent: Option<&Atomic>,
    ) -> bool {
        // Try all the successors as possible starting points
        for successor in self.successors.iter() {
            // Skip if successor is not yet fixed
            let Some(next_node) = successor.induced_fixed_value(&state) else {
                continue;
            };

            // circuit is 1-indexed
            let mut next_idx = usize::try_from(next_node).unwrap() - 1;

            let mut visited = FixedBitSet::with_capacity(self.successors.len());

            loop {
                if visited.contains(next_idx) {
                    if visited.count_ones(..) < self.successors.len() {
                        return true; // proper subcycle = conflict
                    } else {
                        return false; // full Hamiltonian cycle = no conflict
                    }
                }

                visited.insert(next_idx);

                // Move on to the next node if there is one
                let Some(next_node) = self.successors[next_idx].induced_fixed_value(&state) else {
                    break;
                };

                next_idx = usize::try_from(next_node).unwrap() - 1;
            }
        }

        false
    }
}

// Tests

#[cfg(test)]
mod tests {
    use pumpkin_checking::TestAtomic;
    use pumpkin_checking::VariableState;

    use super::*;

    fn eq(name: &'static str, value: i32) -> TestAtomic {
        TestAtomic {
            name,
            comparison: pumpkin_checking::Comparison::Equal,
            value,
        }
    }

    #[test]
    fn conflict_simple_subcycle() {
        // 1 -> 2 -> 1, node 3 outside
        let premises = [eq("x1", 2), eq("x2", 1)];

        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");

        let checker = CircuitChecker {
            successors: vec!["x1", "x2", "x3"].into(),
        };

        assert!(checker.check(state, &premises, None));
    }

    #[test]
    fn no_conflict_full_hamiltonian_cycle() {
        // 1 -> 2 -> 3 -> 1
        let premises = [eq("x1", 2), eq("x2", 3), eq("x3", 1)];

        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");

        let checker = CircuitChecker {
            successors: vec!["x1", "x2", "x3"].into(),
        };

        assert!(!checker.check(state, &premises, None));
    }

    #[test]
    fn no_conflict_incomplete_path() {
        // 1 -> 2 -> 3, but x3 is not fixed
        let premises = [eq("x1", 2), eq("x2", 3)];

        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");

        let checker = CircuitChecker {
            successors: vec!["x1", "x2", "x3"].into(),
        };

        assert!(!checker.check(state, &premises, None));
    }

    #[test]
    fn conflict_fixed_self_loop() {
        // 1 -> 1, with more than one node
        let premises = [eq("x1", 1)];

        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");

        let checker = CircuitChecker {
            successors: vec!["x1", "x2", "x3"].into(),
        };

        assert!(checker.check(state, &premises, None));
    }

    #[test]
    fn no_conflict_two_variable_cycle() {
        // 1 -> 2 -> 1
        let premises = [eq("x1", 2), eq("x2", 1)];

        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");

        let checker = CircuitChecker {
            successors: vec!["x1", "x2"].into(),
        };

        assert!(!checker.check(state, &premises, None));
    }

    #[test]
    fn conflict_three_node_subcycle_with_four_nodes() {
        // 1 -> 2 -> 3 -> 1, node 4 outside
        let premises = [
            eq("x1", 2),
            eq("x2", 3),
            eq("x3", 1),
        ];

        let state = VariableState::prepare_for_conflict_check(premises, None)
            .expect("no conflicting atomics");

        let checker = CircuitChecker {
            successors: vec!["x1", "x2", "x3", "x4"].into(),
        };

        assert!(checker.check(state, &premises, None));
    }
}