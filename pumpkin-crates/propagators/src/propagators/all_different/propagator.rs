use pumpkin_core::declare_inference_label;
use pumpkin_core::proof::ConstraintTag;
use pumpkin_core::proof::InferenceCode;
use pumpkin_core::propagation::InferenceCheckers;
use pumpkin_core::propagation::PropagationContext;
use pumpkin_core::propagation::Propagator;
use pumpkin_core::propagation::PropagatorConstructor;
use pumpkin_core::propagation::PropagatorConstructorContext;
use pumpkin_core::propagation::ReadDomains;
use pumpkin_core::variables::IntegerVariable;
use pumpkin_core::propagation::DomainEvents;
use pumpkin_core::propagation::LocalId;
use pumpkin_core::propagation::Domains;
use pumpkin_core::state::PropagationStatusCP;

use crate::all_different::AllDifferentChecker;

#[derive(Debug, Clone)]
pub struct AllDifferentConstructor<Var> {
    pub sucs: Box<[Var]>,
    pub constraint_tag: ConstraintTag,
}
declare_inference_label!(AllDifferent);

impl<Var: IntegerVariable + 'static> PropagatorConstructor for AllDifferentConstructor<Var> {
    type PropagatorImpl = AllDifferentPropagator<Var>;

    fn create(self, mut context: PropagatorConstructorContext) -> Self::PropagatorImpl {
        self.sucs
            .iter()
            .enumerate()
            .for_each(|(index, successor)| {
                context.register(
                    successor.clone(),
                    DomainEvents::ASSIGN,
                    LocalId::from(index as u32),
                );
                context.register_backtrack(
                    successor.clone(),
                    DomainEvents::ASSIGN,
                    LocalId::from(index as u32),
                );
            });
        AllDifferentPropagator {
            sucs: self.sucs,
            inference_code: InferenceCode::new(self.constraint_tag, AllDifferent),
        }
    }

    fn add_inference_checkers(&self, mut checkers: InferenceCheckers<'_>) {
        // checkers.add_inference_checker(
        //     InferenceCode::new(self.constraint_tag, AllDifferent),
        //     Box::new(AllDifferentChecker {
        //         successors: self.sucs.clone(),
        //     }),
        // );
    }
}

#[derive(Debug, Clone)]
pub struct AllDifferentPropagator<Var> {
    sucs: Box<[Var]>,
    inference_code: InferenceCode,
}

impl<Var: IntegerVariable + 'static> Propagator for AllDifferentPropagator<Var> {
    fn name(&self) -> &str {
        "AllDifferent"
    }
    fn propagate(&mut self, mut context: PropagationContext) -> pumpkin_core::state::PropagationStatusCP {
        self.check_matching_conflict(context.domains())
    }

    fn propagate_from_scratch(
        &self,
        mut context: PropagationContext,
    ) -> pumpkin_core::state::PropagationStatusCP {
        self.check_matching_conflict(context.domains())
    }
}
struct BipartiteGraph {
    n_vars: usize,
    n_vals: usize,
    /// adj[var_index] = list of value-indices (0-indexed) in domain of var i.
    adj: Vec<Vec<usize>>,
    /// Shift so that domain values map to 0-indexed value-nodes.
    /// For MiniZinc 1-indexed successors this is always 1.
    val_offset: i32,
}
 
impl BipartiteGraph {
    fn debug_print(&self) {
        println!("BipartiteGraph:");
        println!("  n_vars = {}", self.n_vars);
        println!("  n_vals = {}", self.n_vals);
        println!("  val_offset = {}", self.val_offset);
        for (i, neighbors) in self.adj.iter().enumerate() {
            print!("  var {} ->", i);
            for &idx in neighbors {
                let val = index_to_domain_value(idx);
                print!(" {}(idx={})", val, idx);
            }
            println!();
        }
    }
    fn build<Var: IntegerVariable>(successors: &[Var], domains: Domains) -> Self {
        // Find global min/max over all domains to size the value array.
        let val_offset = successors
            .iter()
            .map(|v| domains.lower_bound(v))
            .min()
            .unwrap_or(1);
 
        let max_val = successors
            .iter()
            .map(|v| domains.upper_bound(v))
            .max()
            .unwrap_or(val_offset);
 
        let n_vars = successors.len();
        let n_vals = (max_val - val_offset + 1) as usize;
        let mut adj = vec![Vec::new(); n_vars];
 
        for (i, var) in successors.iter().enumerate() {
            for val in domains.iterate_domain(var) {
                adj[i].push((val - val_offset) as usize);
            }
        }
 
        BipartiteGraph { n_vars, n_vals, adj, val_offset }
    }
}

impl<Var: IntegerVariable + 'static> AllDifferentPropagator<Var> {
    fn check_matching_conflict(&self, domains: Domains) -> PropagationStatusCP {
        //Step 1 : build bipartite graphs
        //Step 2 : find the mazimu bipartite matching
        //Step 3 : check if matching = n = All diff satificable
        //Step 4 : Derive hall violation
        //Step 5 build explanation + report conflict.
        todo!()
    }
}




 
const VALUE_OFFSET: usize = 1;
 
#[inline]
fn domain_value_to_index(domain_value: i32) -> usize {
    domain_value as usize - VALUE_OFFSET
}
 
#[inline]
fn index_to_domain_value(index: usize) -> i32 {
    index as i32 + VALUE_OFFSET as i32
}




#[cfg(test)]
mod tests { 
    use super::*;
    use pumpkin_checking::CheckerVariable;
use pumpkin_core::state::State;
    use pumpkin_core::variables::IntegerVariable;

    #[test]
    fn test_bipartite_graph_simple() {
        let mut state = State::default();

        // Create 3 variables with initial domain [1..3]
        let x = state.new_interval_variable(1, 2, None);
        let y = state.new_interval_variable(2, 3, None);
        let z = state.new_interval_variable(1, 3, None);

        let successors = vec![x, y, z].into_boxed_slice();

        // Build the graph directly (no propagation needed)
        let graph = BipartiteGraph::build(&successors, state.get_domains());
        graph.debug_print();
        // Check graph structure
        assert_eq!(graph.n_vars, 3);
        assert_eq!(graph.val_offset, 1);
        assert_eq!(graph.n_vals, 3); // values 1,2,3 → indices 0,1,2

        // Convert adjacency lists back to domain values
        let neighbors_as_values: Vec<Vec<i32>> = graph.adj
            .iter()
            .map(|ns| ns.iter().map(|&idx| index_to_domain_value(idx)).collect())
            .collect();

        assert_eq!(neighbors_as_values[0], vec![1, 2]); // x
        assert_eq!(neighbors_as_values[1], vec![2, 3]); // y
        assert_eq!(neighbors_as_values[2], vec![1, 2, 3]); // z
    }
}