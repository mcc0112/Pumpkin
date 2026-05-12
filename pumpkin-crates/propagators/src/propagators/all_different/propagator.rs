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
use pumpkin_core::state::PropagatorConflict;
use pumpkin_core::predicate;
use pumpkin_core::predicates::PropositionalConjunction;
use pumpkin_core::state::Conflict;

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
                    DomainEvents::ANY_INT,
                    LocalId::from(index as u32),
                );
                context.register_backtrack(
                    successor.clone(),
                    DomainEvents::ANY_INT,
                    LocalId::from(index as u32),
                );
            });
        AllDifferentPropagator {
            sucs: self.sucs,
            inference_code: InferenceCode::new(self.constraint_tag, AllDifferent),
        }
    }

    fn add_inference_checkers(&self, mut checkers: InferenceCheckers<'_>) {
        checkers.add_inference_checker(
            InferenceCode::new(self.constraint_tag, AllDifferent),
            Box::new(AllDifferentChecker {
                successors: self.sucs.clone(),
            }),
        );
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
///////     GRAPH BUILDING
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
                let val = idx as i32 + 1;
                print!(" {}(idx={})", val, idx);
            }
            println!();
        }
    }
    fn build<Var: IntegerVariable>(successors: &[Var], domains: &Domains) -> Self {
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


///////     STEP 2: RUN MATCHING
const UNMATCHED: usize = usize::MAX;
const INF_DIST: usize = usize::MAX;
 
struct Matching {
    /// match_var[i] = value-index matched to variable i, or UNMATCHED.
    match_var: Vec<usize>,
    /// match_val[v] = variable-index matched to value v, or UNMATCHED.
    match_val: Vec<usize>,
    size: usize,
}
 
impl Matching {
    fn new(n_vars: usize, n_vals: usize) -> Self {
        Matching {
            match_var: vec![UNMATCHED; n_vars],
            match_val: vec![UNMATCHED; n_vals],
            size: 0,
        }
    }
}

fn hopcroft_karp(graph: &BipartiteGraph) -> Matching {
    let mut m = Matching::new(graph.n_vars, graph.n_vals);
 
    loop {
        // ---- BFS phase: build layered graph of shortest augmenting paths ----
        //
        // dist[i] = distance of variable-node i from the set of free variable-
        // nodes, following alternating (free, matched, free, ...) arcs.
        // We only store distances for variable-nodes; value-nodes are implicit.
        let mut dist = vec![INF_DIST; graph.n_vars];
        let mut queue = std::collections::VecDeque::new();
 
        for i in 0..graph.n_vars {
            if m.match_var[i] == UNMATCHED {
                dist[i] = 0;
                queue.push_back(i);
            }
        }
 
        let mut found_augmenting = false;
 
        while let Some(i) = queue.pop_front() {
            for &v in &graph.adj[i] {
                // Free arc: var i -> val v (edge not in matching).
                // Matching arc: val v -> var next (follow the matching back).
                let next = m.match_val[v];
                if next == UNMATCHED {
                    // val v is free: augmenting path endpoint reachable.
                    found_augmenting = true;
                } else if dist[next] == INF_DIST {
                    dist[next] = dist[i] + 1;
                    queue.push_back(next);
                }
            }
        }
 
        if !found_augmenting {
            break; // Maximum matching reached.
        }
 
        // ---- DFS phase: augment along vertex-disjoint shortest paths ----
        for i in 0..graph.n_vars {
            if m.match_var[i] == UNMATCHED && dfs_augment(i, graph, &mut m, &mut dist) {
                m.size += 1;
            }
        }
    }
 
    m
}
 
fn dfs_augment(
    i: usize,
    graph: &BipartiteGraph,
    m: &mut Matching,
    dist: &mut [usize],
) -> bool {
    for &v in &graph.adj[i] {
        let next = m.match_val[v];
        // Only follow edges that respect the layered structure.
        let admissible = next == UNMATCHED
            || (dist[next] != INF_DIST && dist[next] == dist[i] + 1);
 
        if admissible {
            let augmented = next == UNMATCHED || dfs_augment(next, graph, m, dist);
            if augmented {
                m.match_var[i] = v;
                m.match_val[v] = i;
                dist[i] = INF_DIST; // consumed; block re-use in this DFS phase
                return true;
            }
        }
    }
    dist[i] = INF_DIST; // dead end
    false
}

//// STEP 3 - FIND HALL SET
fn find_hall_set(graph: &BipartiteGraph, m: &Matching) -> (Vec<usize>, Vec<usize>) {
    let mut var_visited = vec![false; graph.n_vars];
    let mut val_visited = vec![false; graph.n_vals];
    let mut queue = std::collections::VecDeque::new();
 
    // Seed: all unmatched variable-nodes.
    for i in 0..graph.n_vars {
        if m.match_var[i] == UNMATCHED {
            var_visited[i] = true;
            queue.push_back(i);
        }
    }
 
    while let Some(i) = queue.pop_front() {
        for &v in &graph.adj[i] {
            if !val_visited[v] {
                val_visited[v] = true; // reached this value-node via a free arc
                // Follow the matching arc back to a variable-node.
                let matched_var = m.match_val[v];
                if matched_var != UNMATCHED && !var_visited[matched_var] {
                    var_visited[matched_var] = true;
                    queue.push_back(matched_var);
                }
            }
        }
    }
 
    let hall_vars: Vec<usize> = (0..graph.n_vars).filter(|&i| var_visited[i]).collect();
    let hall_vals: Vec<usize> = (0..graph.n_vals).filter(|&v| val_visited[v]).collect();
 
    debug_assert!(
        hall_vals.len() < hall_vars.len(),
        "Bug in Hall extraction: |N(S)|={} >= |S|={}",
        hall_vals.len(),
        hall_vars.len()
    );
 
    (hall_vars, hall_vals)
}


impl<Var: IntegerVariable + 'static> AllDifferentPropagator<Var> {
    fn check_matching_conflict(&self, domains: Domains) -> PropagationStatusCP {
        //Step 1 : build bipartite graphs
        let graph = BipartiteGraph::build(&self.sucs, &domains);

        //Step 2 : find the mazimu bipartite matching
        let matching = hopcroft_karp(&graph);
        //Step 3 : check if matching = n = All diff satificable
        if matching.size == graph.n_vars {
            return Ok(());
        }
        
        //Step 4 : Derive hall violation
        let (hall_vars, hall_vals) = find_hall_set(&graph, &matching);

        //Step 5 build explanation + report conflict.
        let conjunction =
            self.make_hall_explanation(domains, &graph, &hall_vars, &hall_vals);
 
        Err(Conflict::Propagator(PropagatorConflict {
            conjunction,
            inference_code: self.inference_code.clone(),
        }))
    }
    fn make_hall_explanation(
        &self,
        domains: Domains,
        graph: &BipartiteGraph,
        hall_vars: &[usize],
        hall_vals: &[usize],
    ) -> PropositionalConjunction {
        hall_vars
            .iter()
            .flat_map(|&i| {
                let var = &self.sucs[i];
 
                if let Some(fixed_val) = domains.fixed_value(var) {
                    // Variable is fully assigned: one literal covers everything.
                    // `succ[i] == fixed_val` implies `succ[i] != v` for all v != fixed_val,
                    // so it subsumes every individual exclusion literal.
                    vec![predicate!(var == fixed_val)]
                } else {
                    // Variable is not fixed: explain which values in N(S) are
                    // excluded from its domain, since those exclusions are what
                    // made N(S) too small.
                    hall_vals
                        .iter()
                        .filter_map(|&v_idx| {
                            let domain_val = v_idx as i32 + graph.val_offset;
                            if !domains.contains(var, domain_val) {
                                // This value is in N(S) but was removed from
                                // succ[i]'s domain — that removal is on the trail
                                // and is part of the Hall violation's cause.
                                Some(predicate!(var != domain_val))
                            } else {
                                None
                            }
                        })
                        .collect()
                }
            })
            .collect()
    }

}











#[cfg(test)]
mod tests { 
    use super::*;
    use pumpkin_core::state::State;
    
    fn make_state(domains: &[(i32, i32)]) -> State {
        let mut state = State::default();
        let vars: Box<[_]> = domains
            .iter()
            .map(|&(lo, hi)| state.new_interval_variable(lo, hi, None))
            .collect();
        let tag = state.new_constraint_tag();
        let _ = state.add_propagator(AllDifferentConstructor {
            sucs: vars,
            constraint_tag: tag,
        });
        state
    }
 
    #[test]
    fn no_conflict_all_distinct_fixed() {
        let mut state = make_state(&[(1, 1), (2, 2), (3, 3)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }
 
    #[test]
    fn conflict_two_vars_same_fixed_value() {
        let mut state = make_state(&[(2, 2), (2, 2), (3, 3)]);
        assert!(state.propagate_to_fixed_point().is_err());
    }
 
    #[test]
    fn conflict_hall_violation_unfixed_vars() {
        let mut state = make_state(&[(1, 2), (1, 2), (1, 2)]);
        assert!(
            state.propagate_to_fixed_point().is_err(),
            "3 vars constrained to only 2 values is a Hall violation"
        );
    }
 
    #[test]
    fn no_conflict_nothing_fixed() {
        let mut state = make_state(&[(1, 3), (1, 3), (1, 3)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }
 
    #[test]
    fn single_variable_ok() {
        let mut state = make_state(&[(1, 1)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }
 
    #[test]
    fn no_conflict_two_vars_two_vals() {
        let mut state = make_state(&[(1, 2), (1, 2)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }
 
    #[test]
    fn no_conflict_partial_assignment_ok() {
        let mut state = make_state(&[(1, 1), (2, 2), (1, 4)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }
 
    #[test]
    fn conflict_four_vars_two_vals() {
        let mut state = make_state(&[(1, 2), (1, 2), (1, 2), (1, 2)]);
        assert!(state.propagate_to_fixed_point().is_err());
    }

    #[test]
    fn no_conflict_five_distinct_singletons() {
        let mut state = make_state(&[(1,1),(2,2),(3,3),(4,4),(5,5)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }

    #[test]
    fn conflict_five_vars_four_vals() {
        let mut state = make_state(&[(1,4),(1,4),(1,4),(1,4),(1,4)]);
        assert!(state.propagate_to_fixed_point().is_err());
    }

    #[test]
    fn no_conflict_staircase_domains() {
        let mut state = make_state(&[(1,2),(2,3),(3,4)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }

    #[test]
    fn conflict_staircase_tail_clash() {
        let mut state = make_state(&[(1,2),(2,3),(3,3),(3,3)]);
        assert!(state.propagate_to_fixed_point().is_err());
    }

    #[test]
    fn conflict_hidden_hall_four_vars_three_vals() {
        let mut state = make_state(&[(1,3),(1,3),(1,3),(1,3)]);
        assert!(state.propagate_to_fixed_point().is_err());
    }

   
    #[test]
    fn conflict_subset_hall_violation() {
        let mut state = make_state(&[(1,2),(1,2),(1,2),(1,10)]);
        assert!(state.propagate_to_fixed_point().is_err());
    }

    #[test]
    fn no_conflict_two_vars_confined_ok() {
        let mut state = make_state(&[(1,2),(1,2),(3,4),(5,6)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }
    #[test]
    fn no_conflict_one_fixed_rest_wide() {
        let mut state = make_state(&[(3,3),(1,5),(1,5),(1,5)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }

    #[test]
    fn conflict_fixed_vars_exhaust_values() {
        let mut state = make_state(&[(1,1),(2,2),(1,2)]);
        assert!(state.propagate_to_fixed_point().is_err());
    }

    #[test]
    fn conflict_two_identical_singletons() {
        let mut state = make_state(&[(5,5),(5,5)]);
        assert!(state.propagate_to_fixed_point().is_err());
    }

    #[test]
    fn no_conflict_large_domains() {
        let mut state = make_state(&[(1,100),(1,100),(1,100),(1,100),(1,100)]);
        assert!(state.propagate_to_fixed_point().is_ok());
    }

    #[test]
    fn conflict_all_vars_forced_to_one() {
        let mut state = make_state(&[(7,7),(7,7),(7,7)]);
        assert!(state.propagate_to_fixed_point().is_err());
    }
  
}