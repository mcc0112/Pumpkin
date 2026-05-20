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
        self.check_conflict_and_propgate(context)
    }

    fn propagate_from_scratch(
        &self,
        mut context: PropagationContext,
    ) -> pumpkin_core::state::PropagationStatusCP {
        self.check_conflict_and_propgate(context)
    }
}

// ============================
// STEP 1 - Build Bipartite Graph from domains
// Creates an adjacency list where adj[i] is the list of value-indices
// reachable from variable i.
// ============================

struct BipartiteGraph {
    n_vars: usize,
    n_vals: usize,
    /// adj[var_index] = list of value-indices (0-indexed) in domain of var i.
    adj: Vec<Vec<usize>>,
    /// Shift so that domain values map to 0-indexed value-nodes.
    val_offset: i32,
}

impl BipartiteGraph {
    fn build<Var: IntegerVariable>(successors: &[Var], domains: &Domains) -> Self {
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

        // OPTIMISATION 1: pre-size each inner Vec to the variable's domain
        // width so it never needs to reallocate as values are pushed.
        let mut adj: Vec<Vec<usize>> = successors
            .iter()
            .map(|v| {
                let capacity =
                    (domains.upper_bound(v) - domains.lower_bound(v) + 1) as usize;
                Vec::with_capacity(capacity)
            })
            .collect();

        for (i, var) in successors.iter().enumerate() {
            for val in domains.iterate_domain(var) {
                adj[i].push((val - val_offset) as usize);
            }
        }

        BipartiteGraph { n_vars, n_vals, adj, val_offset }
    }
}

// ============================
// STEP 2 - HOPCROFT-KARP MATCHING
// ============================

const UNMATCHED: usize = usize::MAX;
const INF_DIST: usize = usize::MAX;

struct Matching {
    /// match_var[i] = value-index matched to variable i, or UNMATCHED.
    match_var: Vec<usize>,
    /// OPTIMISATION 3: use HashMap instead of a dense Vec so that wide but
    /// sparse domains (e.g. domain {1, 1000}) don't allocate 1000 slots.
    match_val: std::collections::HashMap<usize, usize>,
    size: usize,
}

impl Matching {
    fn new(n_vars: usize) -> Self {
        Matching {
            match_var: vec![UNMATCHED; n_vars],
            // Pre-size the map to the number of variables — the maximum
            // number of matched value-nodes can never exceed n_vars.
            match_val: std::collections::HashMap::with_capacity(n_vars),
            size: 0,
        }
    }

    #[inline]
    fn get_match_val(&self, v: usize) -> usize {
        self.match_val.get(&v).copied().unwrap_or(UNMATCHED)
    }

    #[inline]
    fn set_match_val(&mut self, v: usize, var: usize) {
        self.match_val.insert(v, var);
    }
}

fn hopcroft_karp(graph: &BipartiteGraph) -> Matching {
    // OPTIMISATION 3: Matching no longer takes n_vals.
    let mut m = Matching::new(graph.n_vars);

    loop {
        // ---- BFS phase: build layered graph of shortest augmenting paths ----
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
                let next = m.get_match_val(v);
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

        // ---- DFS phase: augmentation ----
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
        let next = m.get_match_val(v);
        let admissible = next == UNMATCHED
            || (dist[next] != INF_DIST && dist[next] == dist[i] + 1);

        if admissible {
            let augmented = next == UNMATCHED || dfs_augment(next, graph, m, dist);
            if augmented {
                m.match_var[i] = v;
                m.set_match_val(v, i);
                dist[i] = INF_DIST;
                return true;
            }
        }
    }
    dist[i] = INF_DIST;
    false
}

// ============================
// STEP 3 - FIND HALL SET
// ============================

fn find_hall_set(graph: &BipartiteGraph, m: &Matching) -> (Vec<usize>, Vec<usize>) {
    let mut var_visited = vec![false; graph.n_vars];
    let mut val_visited = vec![false; graph.n_vals];
    let mut queue = std::collections::VecDeque::new();

    for i in 0..graph.n_vars {
        if m.match_var[i] == UNMATCHED {
            var_visited[i] = true;
            queue.push_back(i);
        }
    }

    while let Some(i) = queue.pop_front() {
        for &v in &graph.adj[i] {
            if !val_visited[v] {
                val_visited[v] = true;
                let matched_var = m.get_match_val(v);
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

// ============================
// STEP 4 - Residual Graph + Tarjan's SCC
// ============================

struct ResidualGraph {
    n_nodes: usize,
    adj: Vec<Vec<usize>>,
}

impl ResidualGraph {
    fn build(graph: &BipartiteGraph, m: &Matching) -> Self {
        let n_vars = graph.n_vars;
        let n_vals = graph.n_vals;
        let n_nodes = n_vars + n_vals;

        // OPTIMISATION 2: pre-size var-nodes to their domain size; val-nodes
        // to 1 since each can only have one matched back-arc.
        let mut adj: Vec<Vec<usize>> = (0..n_nodes)
            .map(|i| {
                if i < n_vars {
                    Vec::with_capacity(graph.adj[i].len())
                } else {
                    Vec::with_capacity(1)
                }
            })
            .collect();

        for i in 0..n_vars {
            for &v in &graph.adj[i] {
                let var_node = i;
                let val_node = n_vars + v;
                if m.match_var[i] == v {
                    // Matched edge: value → variable
                    adj[val_node].push(var_node);
                } else {
                    // Unmatched edge: variable → value
                    adj[var_node].push(val_node);
                }
            }
        }
        ResidualGraph { n_nodes, adj }
    }
}

// ============================
// STEP 5 - Tarjan's SCC
// ============================

// OPTIMISATION 4: replace Vec<Option<usize>> (2 words per slot) with a plain
// Vec<usize> using usize::MAX as the "unvisited" sentinel, halving the memory
// cost of the indices array.
const UNVISITED: usize = usize::MAX;

struct TarjanState {
    index:    usize,
    stack:    Vec<usize>,
    on_stack: Vec<bool>,
    /// OPTIMISATION 4: flat sentinel-based vec instead of Vec<Option<usize>>.
    indices:  Vec<usize>,
    lowlinks: Vec<usize>,
    scc_id:   Vec<usize>,
    next_id:  usize,
}

impl TarjanState {
    fn new(n: usize) -> Self {
        TarjanState {
            index:    0,
            stack:    Vec::new(),
            on_stack: vec![false; n],
            indices:  vec![UNVISITED; n],
            lowlinks: vec![0; n],
            scc_id:   vec![0; n],
            next_id:  0,
        }
    }
}

fn tarjan_scc(graph: &ResidualGraph) -> Vec<usize> {
    let n = graph.n_nodes;
    let mut state = TarjanState::new(n);
    for start in 0..n {
        if state.indices[start] == UNVISITED {
            tarjan_visit(start, &graph.adj, &mut state);
        }
    }
    state.scc_id
}

fn tarjan_visit(start: usize, adj: &[Vec<usize>], state: &mut TarjanState) {
    // OPTIMISATION 4: use sentinel comparisons in place of Option methods.
    state.indices[start]  = state.index;
    state.lowlinks[start] = state.index;
    state.index += 1;
    state.stack.push(start);
    state.on_stack[start] = true;

    let mut call_stack: Vec<(usize, usize)> = Vec::new();
    call_stack.push((start, 0));

    while let Some(_) = call_stack.last() {
        let (v, ei) = *call_stack.last().unwrap();
        let ei_ref = &mut call_stack.last_mut().unwrap().1;

        if *ei_ref < adj[v].len() {
            let w = adj[v][*ei_ref];
            *ei_ref += 1;

            if state.indices[w] == UNVISITED {
                // Tree edge: recurse into w
                state.indices[w]  = state.index;
                state.lowlinks[w] = state.index;
                state.index += 1;
                state.stack.push(w);
                state.on_stack[w] = true;
                call_stack.push((w, 0));
            } else if state.on_stack[w] {
                // Back edge: update lowlink
                let w_idx = state.indices[w];
                if w_idx < state.lowlinks[v] {
                    state.lowlinks[v] = w_idx;
                }
            }
        } else {
            // All neighbours of v processed — pop v
            call_stack.pop();

            if let Some(&(parent, _)) = call_stack.last() {
                if state.lowlinks[v] < state.lowlinks[parent] {
                    state.lowlinks[parent] = state.lowlinks[v];
                }
            }

            // Check if v is the root of an SCC
            if state.lowlinks[v] == state.indices[v] {
                loop {
                    let w = state.stack.pop().unwrap();
                    state.on_stack[w] = false;
                    state.scc_id[w] = state.next_id;
                    if w == v { break; }
                }
                state.next_id += 1;
            }
        }
    }
}

// ============================
// Core propagation logic
// ============================

impl<Var: IntegerVariable + 'static> AllDifferentPropagator<Var> {
    fn check_conflict_and_propgate(&self, mut context: PropagationContext) -> PropagationStatusCP {
        let domains = context.domains();

        // Step 1: build bipartite graph
        let graph = BipartiteGraph::build(&self.sucs, &domains);

        // Step 2: maximum matching
        let matching = hopcroft_karp(&graph);

        // Step 3: conflict check — if no perfect matching, find Hall set and raise conflict
        if matching.size < graph.n_vars {
            let (hall_vars, hall_vals) = find_hall_set(&graph, &matching);
            let conjunction = self.make_hall_explanation(
                domains, &graph, &hall_vars, &hall_vals,
            );
            return Err(Conflict::Propagator(PropagatorConflict {
                conjunction,
                inference_code: self.inference_code.clone(),
            }));
        }

        // Step 4: build directed residual graph
        let residual = ResidualGraph::build(&graph, &matching);

        // Step 5: compute SCCs
        let scc_id = tarjan_scc(&residual);

        // Step 6: pruning — collect then apply
        // OPTIMISATION: pre-size with a lower-bound capacity to avoid repeated growth.
        let mut prunings: Vec<(usize, i32, PropositionalConjunction)> =
            Vec::with_capacity(graph.n_vars);

        for i in 0..graph.n_vars {
            let var_node = i;
            let matched_val = matching.match_var[i];

            for &v in &graph.adj[i] {
                if v == matched_val {
                    continue;
                }
                let val_node = graph.n_vars + v;

                if scc_id[var_node] != scc_id[val_node] {
                    let domain_val = v as i32 + graph.val_offset;
                    let explanation = self.make_pruning_explanation(
                        &domains,
                        &graph,
                        &scc_id,
                        i,
                        v,
                        &matching,
                    );
                    prunings.push((i, domain_val, explanation));
                }
            }
        }

        for (var_idx, domain_val, reason) in prunings {
            let var = &self.sucs[var_idx];
            if context.contains(var, domain_val) {
                context.post(
                    predicate!(var != domain_val),
                    reason,
                    &self.inference_code,
                )?;
            }
        }

        Ok(())
    }

    fn make_hall_explanation(
        &self,
        domains: Domains,
        graph: &BipartiteGraph,
        hall_vars: &[usize],
        hall_vals: &[usize],
    ) -> PropositionalConjunction {
        let hall_val_set: std::collections::HashSet<usize> =
            hall_vals.iter().copied().collect();

        hall_vars
            .iter()
            .flat_map(|&i| {
                let var = &self.sucs[i];
                let lb = domains.lower_bound(var);
                let ub = domains.upper_bound(var);

                if let Some(fixed_val) = domains.fixed_value(var) {
                    vec![predicate!(var == fixed_val)]
                } else {
                    let mut lits = vec![
                        predicate!(var >= lb),
                        predicate!(var <= ub),
                    ];
                    for v_idx in 0..graph.n_vals {
                        if hall_val_set.contains(&v_idx) {
                            continue;
                        }
                        let domain_val = v_idx as i32 + graph.val_offset;
                        if domain_val <= lb || domain_val >= ub {
                            continue;
                        }
                        if !domains.contains(var, domain_val) {
                            lits.push(predicate!(var != domain_val));
                        }
                    }
                    lits
                }
            })
            .collect()
    }

    fn make_pruning_explanation(
        &self,
        domains: &Domains,
        graph: &BipartiteGraph,
        scc_id: &[usize],
        var_idx: usize,
        pruned_val_idx: usize,
        matching: &Matching,
    ) -> PropositionalConjunction {
        let xi_scc = scc_id[var_idx];

        let scc_vars: Vec<usize> = (0..graph.n_vars)
            .filter(|&j| scc_id[j] == xi_scc)
            .collect();

        let n_s: std::collections::HashSet<usize> = (0..graph.n_vals)
            .filter(|&v| scc_id[graph.n_vars + v] == xi_scc)
            .collect();

        let lits: Vec<_> = scc_vars
            .iter()
            .flat_map(|&j| {
                let var = &self.sucs[j];
                let lb = domains.lower_bound(var);
                let ub = domains.upper_bound(var);

                if let Some(fv) = domains.fixed_value(var) {
                    vec![predicate!(var == fv)]
                } else {
                    let mut v_lits = vec![
                        predicate!(var >= lb),
                        predicate!(var <= ub),
                    ];
                    for v_idx in 0..graph.n_vals {
                        if n_s.contains(&v_idx) {
                            continue;
                        }
                        let dv = v_idx as i32 + graph.val_offset;
                        if dv <= lb || dv >= ub {
                            continue;
                        }
                        if !domains.contains(var, dv) {
                            v_lits.push(predicate!(var != dv));
                        }
                    }
                    v_lits
                }
            })
            .collect();

        lits.into_iter().collect()
    }
}

// ============================
// Tests
// ============================

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

    #[test]
    fn conflict_subset_hall_violation_five_vars() {
        let mut state = make_state(&[(1, 2), (1, 2), (1, 2), (1, 10), (1, 10)]);
        assert!(
            state.propagate_to_fixed_point().is_err(),
            "subset of 3 vars crowding 2 values is a Hall violation even with other vars present"
        );
    }
}