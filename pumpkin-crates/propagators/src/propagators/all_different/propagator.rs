/// A domain-consistent AllDifferent propagator based on bipartite matching. 
/// 
/// This propagator enforces generalized arc ocnsitncy (GAC) for the 
/// AllDifferent constraint using the classical Regin algorithm
/// 




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
}//
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

/// STEP 1 
/// 
/// Build the bipartite graph Var <-> Val used by Regin's filtering
/// 
/// Each variable node i connects to value nodes represenitng the integers
/// currently in its domain. Values are normalized to a 0-indexed range using 
/// val_offset to map bakc to actual domain values. 
/// 
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
        //safety check just to make sure the min/max never operate on an empty iterator
        if successors.is_empty() {
            return BipartiteGraph { n_vars: 0, n_vals: 0, adj: Vec::new(), val_offset: 0 }
        }
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

/// STEP 2 
/// Compute a maximum matching between variables and values 
/// 
/// A perfect amtching corresponds to a feasible assignemt satisfying 
/// AllDiffernet. If no perfect matching exists, the constraint is already violated 
/// and we must extract a Hall set explaining the conflict
/// 
/// DFS phase is implemented iteratively to avoid recursion depth issues on large domains

const UNMATCHED: usize = usize::MAX;
const INF_DIST: usize = usize::MAX;

struct Matching {
    /// match_var[i] = value-index matched to variable i, or UNMATCHED.
    match_var: Vec<usize>,
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

    #[inline]
    fn get_match_val(&self, v: usize) -> usize {
        self.match_val[v]
    }

    #[inline]
    fn set_match_val(&mut self, v: usize, var: usize) {
        self.match_val[v] = var;
    }
}

fn hopcroft_karp(graph: &BipartiteGraph) -> Matching {
    let mut m = Matching::new(graph.n_vars, graph.n_vals);

    loop {
        // BFS phase: build layered graph of shortest augmenting paths 
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

        // DFS phase: augmentation 
        for i in 0..graph.n_vars {
            if m.match_var[i] == UNMATCHED && dfs_augment_iterative(i, graph, &mut m, &mut dist) {
                m.size +=1;
            }
        }
    }

    m
}

fn dfs_augment_iterative (
    start: usize, 
    graph: &BipartiteGraph, 
    m: &mut Matching, 
    dist: &mut [usize],
) -> bool {
    let mut call_stack: Vec<(usize, usize)> = vec![(start, 0)];
    let mut result_stack: Vec<bool> = vec![false];
    while let Some(frame) = call_stack.last_mut() {
        let (i, ei) = *frame;
        if ei >= graph.adj[i].len() {
            call_stack.pop();
            result_stack.pop();
            if let Some(parent_result) = result_stack.last_mut() {
                let _ = parent_result;
            }
            dist[i] = INF_DIST;
            continue;
        }
        let v = graph.adj[i][ei];
        frame.1 += 1;
        let next = m.get_match_val(v);
        let admissible = next == UNMATCHED ||
            (dist[next] != INF_DIST && dist[next] == dist[i] +1);
        
        if !admissible {
            continue;
        }
        if next == UNMATCHED {
            m.match_var[i] = v;
            m.set_match_val(v, i);
            dist[i] = INF_DIST;
            *result_stack.last_mut().unwrap() = true;
            call_stack.pop();
        
            while let Some (&(pi, _)) = call_stack.last() {
                let pv = m.match_var[pi];
                let _ = pv; 
                let parent_ei = call_stack.last().unwrap().1 -1;
                let taken_v = graph.adj[pi][parent_ei];
                m.match_var[pi] = taken_v;
                m.set_match_val(taken_v, pi);
                dist[pi] = INF_DIST;
                call_stack.pop();
                result_stack.pop();

            }
            return true;
        }
        call_stack.push((next, 0));
        result_stack.push(false);
    }
    false

    
}

/// Extracts a Hall set when the matching is not perfect 
/// 
/// A Hall set S is a set of variable whos combined domain valeus N(S)
/// are too few -> Hall violation
/// 
/// The BFS explores alternating paths starting from unmatched variables. 
/// All visited nodes form S and the visited value node form N(S)
/// 
/// The resulting explanation states taht the vairalbe in S are collectleively restiricted to N(S)
/// which is insuffiecient -> thus conflict
/// 
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
    //if no unmatched variable 0 essentailly a check:
    if queue.is_empty() {
        return (Vec::new(), Vec::new());
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


/// For each varaible value pair (i, v) not in the maximum mathcing,
/// determines whether assignemnd x_i = v would break all perfect amtching. 
/// 
/// This is done by 
///     - Building the residual graph of alternating pathsi
///     - COmputing SCCs 
///     - Identifying tight Hall sets (SCCs where |H| = |V|)
/// 
/// if v belongs to such a tight Hall set but i doe not - then i cannot take v 
/// without violating the Hall condition. The vlaue is pruned. 
/// 
/// Explanatiosn are constructed usign only infromation that awas tru tat the start of propagaiton, ensuring trail ordering
fn find_pruning_hall_set(
    graph: &BipartiteGraph,
    m: &Matching,
    pruned_var: usize,
    pruned_val: usize,
) -> (Vec<usize>, Vec<usize>) {
    let old_m = m.match_val[pruned_val];
    debug_assert!(old_m != UNMATCHED, "pruned value must be matched");

    let mut var_visited = vec![false; graph.n_vars];
    let mut val_visited = vec![false; graph.n_vals];
    let mut queue = std::collections::VecDeque::new();

    // Start BFS from the variable currently matched to pruned_val
    var_visited[old_m] = true;
    queue.push_back(old_m);

    while let Some(h) = queue.pop_front() {
        for &v in &graph.adj[h] {
            if val_visited[v] {
                continue;
            }
            val_visited[v] = true;

            // Never route through pruned_val back to pruned_var.
            // pruned_val is in V (it gets collected below) but the BFS
            // must not follow it to pruned_var, which is outside H.
            if v == pruned_val {
                continue;
            }

            let next = m.match_val[v];
            if next != UNMATCHED && !var_visited[next] {
                var_visited[next] = true;
                queue.push_back(next);
            }
        }
    }

    // H = variables visited (excludes pruned_var by construction)
    // V = values visited (includes pruned_val since val_visited[pruned_val] = true)
    let hall_vars: Vec<usize> = (0..graph.n_vars)
        .filter(|&h| var_visited[h])
        .collect();
    let hall_vals: Vec<usize> = (0..graph.n_vals)
        .filter(|&v| val_visited[v])
        .collect();

    debug_assert!(
        hall_vars.len() == hall_vals.len(),
        "tight Hall set must have |H| = |V|, got |H|={} |V|={}",
        hall_vars.len(), hall_vals.len()
    );

    (hall_vars, hall_vals)
}

/// Build the directed residual graph used for SCC detection
/// 
/// Edges represnt alternating paths:
///     - Unmatched edges; Var -> Val 
///     - Matchign edge: Val -> Var 
/// 
/// Additioanlly, all free valeus connect to a special node T and 
/// T connects back to all varaibles. This models the ability to start 
/// an augmenting path from any free value. 
/// 
/// SCCs in this graph correspond to strongly connected alternating components
/// Tight SCCs revela Hall sets taht justify pruning

struct ResidualGraph {
    n_nodes: usize,  // n_vars + n_vals + 1 (the +1 is T)
    adj: Vec<Vec<usize>>,
}

impl ResidualGraph {
    fn build(graph: &BipartiteGraph, m: &Matching) -> Self {
        let n_vars = graph.n_vars;
        let n_vals = graph.n_vals;
        let t_node = n_vars + n_vals;
        let n_nodes = n_vars + n_vals + 1;

        let mut adj: Vec<Vec<usize>> = (0..n_nodes)
            .map(|i| {
                if i < n_vars {
                    Vec::with_capacity(graph.adj[i].len())
                } else if i == t_node {
                    Vec::with_capacity(n_vars)
                } else {
                    Vec::with_capacity(1)
                }
            })
            .collect();

        // Build matched/unmatched residual edges
        for i in 0..n_vars {
            for &v in &graph.adj[i] {
                let var_node = i;
                let val_node = n_vars + v;
                if m.match_var[i] == v {
                    // Matched edge reversed: value -> variable
                    adj[val_node].push(var_node);
                } else {
                    // Unmatched edge: variable -> value
                    adj[var_node].push(val_node);
                }
            }
        }

        // Connect free (unmatched) values through T
        //   free_val_node -> T       (any path reaching a free value can continue)
        //   T ->  every variable      (T can re-enter the variable layer anywhere)
        for v in 0..n_vals {
            if m.get_match_val(v) == UNMATCHED {
                let val_node = n_vars + v;
                adj[val_node].push(t_node);   
            }
        }
        let has_free_vals = (0..n_vals).any(|v: usize| m.get_match_val(v) == UNMATCHED);
        if has_free_vals {
            for i in 0..n_vars {
                adj[t_node].push(i);          
            }
        }

        ResidualGraph { n_nodes, adj }
    }
}


/// Tarjan's SCC
/// 
/// Computes strongly connected components of the residual grpah
/// 
/// The SCC strcuture paritioans variable sand values into components that
/// share alteranting reachability. A component where teh number of variable 
/// nodes equals teh number of value ndoe corresponds to a tight Hall st
/// 
/// The implementaiton is once again iterative
const UNVISITED: usize = usize::MAX;

struct TarjanState {
    index:    usize,
    stack:    Vec<usize>,
    on_stack: Vec<bool>,
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

/// Main propagation entry point.
///
/// 1. Build bipartite graph from current domains.
/// 2. Compute maximum matching.
/// 3. If matching is not perfect → extract Hall set → raise conflict.
/// 4. Otherwise:
///      a. Build residual graph.
///      b. Compute SCCs.
///      c. For each variable–value pair not in the matching:
///           - If they lie in different SCCs → prune value.
///           - Generate minimal explanation using Hall structure.
/// 5. Apply all prunings.
///
/// This ensures full domain consistency for `AllDifferent`.

impl<Var: IntegerVariable + 'static> AllDifferentPropagator<Var> {
    fn check_conflict_and_propgate(&self, mut context: PropagationContext) -> PropagationStatusCP {
        let domains = context.domains();

        //Check
        if self.sucs.is_empty() {
            return Ok(());
        }

        // Step 1: build bipartite graph
        let graph = BipartiteGraph::build(&self.sucs, &domains);

        // Step 2: maximum matching
        let matching = hopcroft_karp(&graph);

        // Step 3: conflict check — if no perfect matching, find Hall set and raise conflict
        if matching.size < graph.n_vars {
            let (hall_vars, hall_vals) = find_hall_set(&graph, &matching);

            let conjunction = self.make_hall_explanation(
                &domains, &graph, &hall_vars, &hall_vals,
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
        let mut prunings: Vec<(usize, i32, PropositionalConjunction)> =
            Vec::with_capacity(graph.n_vars);
        for i in 0..graph.n_vars {
            let matched_val = matching.match_var[i];
            for &v in &graph.adj[i] {
                if v == matched_val { continue; }
                let val_node = graph.n_vars + v;
                if scc_id[i] == scc_id[val_node] { continue; }

                let domain_val = v as i32 + graph.val_offset;

                // find_pruning_hall_set gives us H (hall_vars) and V (hall_vals)
                // H is confined to V, v is in V, so xi cannot take v
                let (hall_vars, hall_vals) = find_pruning_hall_set(
                    &graph, &matching, i, v
                );

                let explanation = self.make_pruning_explanation_from_hall(
                    &domains,
                    &graph,
                    &hall_vars,
                    &hall_vals,
                );
                prunings.push((i, domain_val, explanation));
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
        domains: &Domains,
        graph: &BipartiteGraph,
        hall_vars: &[usize],
        hall_vals: &[usize],
    ) -> PropositionalConjunction {
        let in_hall_vals: Vec<bool> = {
            let mut v = vec![false; graph.n_vals];
            for &val_idx in hall_vals {v[val_idx] = true;}
            v
        };

        hall_vars
            .iter()
            .flat_map(|&h| {
                let var = &self.sucs[h];
                if let Some (fixed) = domains.fixed_value(var) {
                    return vec![predicate!(var == fixed)];
                }

                let lb = domains.lower_bound(var);
                let ub = domains.upper_bound(var);

                let mut lits = vec![predicate!(var >= lb), predicate!(var <= ub)];

                for d_idx in 0..graph.n_vals {
                    if in_hall_vals[d_idx] {continue;}
                    let d = d_idx as i32 + graph.val_offset;
                    if d <= lb || d >= ub {continue;}

                    if !domains.contains(var, d) {
                        lits.push(predicate!(var != d));
                    }

                }
                lits
            })
            .collect()
    }
    
    /// Build a minimal explanation for the pruning of value `pruned_val_idx`
    /// from variable `var_idx`.
    // fn make_pruning_explanation(
    //     &self,
    //     domains: &Domains,
    //     graph: &BipartiteGraph,
    //     scc_id: &[usize],
    //     val_node: usize,
    // ) -> PropositionalConjunction {
    //     let target_scc = scc_id[val_node];
    //     let n_vars = graph.n_vars;

    //     // Collect which value indices are in the target SCC
    //     let in_val_scc: Vec<bool> = {
    //         let mut v = vec![false; graph.n_vals];
    //         for val_idx in 0..graph.n_vals {
    //             let node = n_vars + val_idx;
    //             if scc_id[node] == target_scc {
    //                 v[val_idx] = true;
    //             }
    //         }
    //         v
    //     };

    //     // For each variable h in the tight SCC, describe its confinement
    //     // using only bounds and holes that existed at propagation entry.
    //     (0..n_vars)
    //         .filter(|&h| scc_id[h] == target_scc)
    //         .flat_map(|h| {
    //             let var = &self.sucs[h];
    //             let lb = domains.lower_bound(var);
    //             let ub = domains.upper_bound(var);

    //             let mut lits = vec![
    //                 predicate!(var >= lb),
    //                 predicate!(var <= ub),
    //             ];
    //             for val_idx in 0..graph.n_vals {
    //                 if in_val_scc[val_idx] {
    //                     continue; // value is in the tight set, skip
    //                 }
    //                 let d = val_idx as i32 + graph.val_offset;
    //                 if d <= lb || d >= ub {
    //                     continue; // already excluded by bounds
    //                 }
    //                 if !domains.contains(var, d) {
    //                     lits.push(predicate!(var != d));
    //                 }
    //             }
    //             lits
    //         })
    //         .collect()
    // }

    fn make_pruning_explanation_from_hall(
        &self,
        domains: &Domains,
        graph: &BipartiteGraph,
        hall_vars: &[usize],
        hall_vals: &[usize],
    ) -> PropositionalConjunction {
        // Mark which value indices are in V (the tight neighbourhood)
        let in_hall_vals: Vec<bool> = {
            let mut v = vec![false; graph.n_vals];
            for &val_idx in hall_vals {
                v[val_idx] = true;
            }
            v
        };

        // For each variable h in H, describe its confinement to V.

        hall_vars
            .iter()
            .flat_map(|&h| {
                let var = &self.sucs[h];
                let lb = domains.lower_bound(var);
                let ub = domains.upper_bound(var);

                // If fixed, a single equality literal suffices
                if let Some(fixed) = domains.fixed_value(var) {
                    return vec![predicate!(var == fixed)];
                }

                let mut lits = vec![
                    predicate!(var >= lb),
                    predicate!(var <= ub),
                ];

                // Emit hole literals only for values OUTSIDE V that fall
                // within [lb, ub] and are already absent from the domain.
                // These holes explain why xh cannot escape V.
                for d_idx in 0..graph.n_vals {
                    if in_hall_vals[d_idx] {
                        continue; // inside V, not a hole we need to explain
                    }
                    let d = d_idx as i32 + graph.val_offset;
                    if d <= lb || d >= ub {
                        continue; // already excluded by bounds
                    }
                    if !domains.contains(var, d) {
                        lits.push(predicate!(var != d));
                    }
                }
                lits
            })
            .collect()
    }

}

// ============================
// Tests
// ============================

#[cfg(test)]
mod tests {
    use super::*;
    use pumpkin_core::{state::State, variables::DomainId};

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
    fn make_state_get_vars(domains: &[(i32, i32)]) -> (State, Box<[DomainId]>) {
        let mut state = State::default();
        let vars: Box<[_]> = domains
            .iter()
            .map(|&(lo, hi)| state.new_interval_variable(lo, hi, None))
            .collect();
        let tag = state.new_constraint_tag();
        let _ = state.add_propagator(AllDifferentConstructor {
            sucs: vars.clone(),
            constraint_tag: tag,
        });
        (state, vars)
    }

    // ── Test 1 ──────────────────────────────────────────────────────────────────
    // y is fixed to 1, so 1 must be pruned from x. Value 3 in z is untouched.
    #[test]
    fn alldiff_prune_fixed_var_removes_value_from_peer() {
        let (mut state, vars) = make_state_get_vars(&[(1, 2), (1, 1), (1, 3)]);
        let _ = state.propagate_to_fixed_point();
        let domains = state.get_domains();

        assert!(
            !domains.contains(&vars[0], 1),
            "y is fixed to 1, so 1 must be pruned from x"
        );
        assert!(
            domains.contains(&vars[2], 3),
            "value 3 is not used by any fixed var and must stay in z"
        );
    }


    #[test]
    fn alldiff_prune_symmetric_pair_no_pruning() {
        let (mut state, vars) = make_state_get_vars(&[(1, 2), (1, 2)]);
        let _ = state.propagate_to_fixed_point();
        let domains = state.get_domains();

        for val in [1, 2] {
            assert!(
                domains.contains(&vars[0], val),
                "x must still contain {val}: it appears in a valid matching"
            );
            assert!(
                domains.contains(&vars[1], val),
                "y must still contain {val}: it appears in a valid matching"
            );
        }
    }

    #[test]
    fn alldiff_prune_staircase_no_over_pruning() {
        let (mut state, vars) = make_state_get_vars(&[(1, 2), (2, 3), (3, 4)]);
        let _ = state.propagate_to_fixed_point();
        let domains = state.get_domains();

        assert!(
            domains.contains(&vars[0], 2),
            "value 2 is in x's domain and lies on an augmenting path — must not be pruned"
        );
        assert!(
            domains.contains(&vars[2], 4),
            "value 4 is exclusively reachable by z — must not be pruned"
        );
    }

    #[test]
    fn alldiff_prune_hall_pair_forces_third_var() {
        let (mut state, vars) = make_state_get_vars(&[(1, 2), (1, 2), (1, 3)]);
        let _ = state.propagate_to_fixed_point();
        let domains = state.get_domains();

        assert!(
            !domains.contains(&vars[2], 1),
            "x and y exhaust value 1 must be pruned from z"
        );
        assert!(
            !domains.contains(&vars[2], 2),
            "x and y exhaust : value 2 must be pruned from z"
        );
        assert!(
            domains.contains(&vars[2], 3),
            "value 3 is the only remaining option for z and must be preserved"
        );
    }


    #[test]
    fn alldiff_prune_exclusive_value_in_wide_domain_preserved() {
        let (mut state, vars) = make_state_get_vars(&[(1, 2), (2, 3), (1, 4)]);
        let _ = state.propagate_to_fixed_point();
        let domains = state.get_domains();

        assert!(
            domains.contains(&vars[2], 4),
            "value 4 is only reachable by z — it must not be pruned even though z's domain is wide"
        );
        assert!(
            domains.contains(&vars[2], 3),
            "value 3 is also reachable by z via an augmenting path — must be kept"
        );
    }


    #[test]
    fn alldiff_prune_fixed_var_removes_its_value_from_peer_domain() {
        let (mut state, vars) = make_state_get_vars(&[(3, 3), (1, 3), (1, 2)]);
        let _ = state.propagate_to_fixed_point();
        let domains = state.get_domains();

        assert!(
            !domains.contains(&vars[1], 3),
            "x is fixed to 3, so 3 must be pruned from y"
        );
        assert!(
            domains.contains(&vars[1], 1),
            "value 1 is not taken by any fixed var and must remain in y"
        );
    }
    #[test]
    fn alldiff_prune_full_latin_square_no_pruning() {
        let (mut state, vars) = make_state_get_vars(&[(1, 3), (1, 3), (1, 3)]);
        let _ = state.propagate_to_fixed_point();
        let domains = state.get_domains();

        for (i, var) in vars.iter().enumerate() {
            for val in 1..=3 {
                assert!(
                    domains.contains(var, val),
                    "var[{i}] should still contain {val}: every assignment is part of some matching"
                );
            }
        }
    }

    #[test]
    fn alldiff_prune_only_completing_value_preserved() {
        let (mut state, vars) = make_state_get_vars(&[(1, 1), (2, 2), (1, 3)]);
        let _ = state.propagate_to_fixed_point();
        let domains = state.get_domains();

        assert!(
            !domains.contains(&vars[2], 1),
            "value 1 is consumed by x=1 and must be pruned from z"
        );
        assert!(
            !domains.contains(&vars[2], 2),
            "value 2 is consumed by y=2 and must be pruned from z"
        );
        assert!(
            domains.contains(&vars[2], 3),
            "value 3 is the only completing assignment for z — must NOT be pruned, \
            analogous to the closing Hamiltonian edge in the circuit test"
        );
    }
}