use fixedbitset::FixedBitSet;
use pumpkin_core::conjunction;
use pumpkin_core::create_statistics_struct;
use pumpkin_core::declare_inference_label;
use pumpkin_core::predicate;
use pumpkin_core::predicates::PropositionalConjunction;
use pumpkin_core::proof::ConstraintTag;
use pumpkin_core::proof::InferenceCode;
use pumpkin_core::propagation::DomainEvents;
use pumpkin_core::propagation::Domains;
use pumpkin_core::propagation::LocalId;
use pumpkin_core::propagation::PropagationContext;
use pumpkin_core::propagation::Propagator;
use pumpkin_core::propagation::PropagatorConstructor;
use pumpkin_core::propagation::ReadDomains;
use pumpkin_core::state::Conflict;
use pumpkin_core::state::PropagationStatusCP;
use pumpkin_core::state::PropagatorConflict;
use pumpkin_core::statistics::Statistic;
use pumpkin_core::variables::IntegerVariable;
use pumpkin_core::propagation::InferenceCheckers;

use crate::circuit::CircuitChecker;


// constructor for the propagator. ConstraintTag is for proof logging
#[derive(Debug, Clone)]
pub struct CircuitConstructor<Var> {
    pub successors: Box<[Var]>,
    pub constraint_tag: ConstraintTag,
}

// Propagator struct. Contains propagator info and inference code (latter for explanations)
#[derive(Debug, Clone)]
pub struct CircuitPropagator<Var> {
    first_iteration: bool, 
    pub successors: Box<[Var]>,
    // fields (and maybe extra ones)
    inference_code: InferenceCode,
    statistics: CircuitBaseStatistics,
}

// The whole propagator constructor itself
impl<Var> PropagatorConstructor 
    for CircuitConstructor<Var> 
where 
    // define the type of the variable
    Var : IntegerVariable + 'static, 
{
    type PropagatorImpl = CircuitPropagator<Var>; //associated type; specifies this constructor produces a CircuitConstructor when the solver instantiates it.

    fn create(
        self,
        mut context: pumpkin_core::propagation::PropagatorConstructorContext,
    ) -> Self::PropagatorImpl {
        // registering for domain events; when should our propagator be enqueued. The flag DomainEvents: ASSIGN determines that it should be queued on 'assignment'. 
        // LocalId is to internally indicate to what variable changes occur; unique for each var
        self.successors
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

        // create the actual propagator and generate new inference code
        CircuitPropagator {
            // set variables to base values
            first_iteration: true, 
            successors: self.successors,
            inference_code: InferenceCode::new(self.constraint_tag, CircuitPrevent),
            statistics: CircuitBaseStatistics::default(),
        }
    }
    
    
    // inference checker
    fn add_inference_checkers(&self, mut checkers: InferenceCheckers<'_>) {
        checkers.add_inference_checker(
            InferenceCode::new(self.constraint_tag, CircuitPrevent),
            Box::new(CircuitChecker {
                successors: self.successors.clone(),
            }),
        );
    }
}

declare_inference_label!(CircuitPrevent);

create_statistics_struct!(CircuitBaseStatistics {
    //how many propgation calls were triggered by ASSIGN evnet vs other domain changes
    propagations_that_found_conflict: usize,
    //Failure depth: total fixed edges at time of each conflic
    //Dividing by the number of conflict (post processing - average dept)
    total_fixed_edges_at_conflict: usize,
    number_of_conflicts: usize,
    //Did propgation do useful work 
    propagations_that_pruned: usize, 
    propagations_total: usize,

});

// here comes an implementation of Propagator which has some basic functions (like defining the name) but also important functions propagate() and propagate_from_scratch()
impl<Var: IntegerVariable + 'static> Propagator for CircuitPropagator<Var> {
    fn name(&self) -> &str {
    "Circuit"
    }
     fn propagate_from_scratch(&self, mut context: PropagationContext) -> PropagationStatusCP {
        self.remove_self_loops(&mut context)?;
        self.check(context.domains())?;
        self.prevent(&mut context)
    }

    fn log_statistics(&self, statistic_logger: pumpkin_core::statistics::StatisticLogger) {
        self.statistics.log(statistic_logger);
    }
    fn propagate(&mut self, mut context: PropagationContext) -> PropagationStatusCP {
        self.statistics.propagations_total += 1;
        // self.statistics.propagations_triggered_by_assign +=1;

        if self.first_iteration {
            self.first_iteration = false;
            self.remove_self_loops(&mut context)?;
        }

        let pruned_before = self.count_fixed(&mut context);  // see Step 6
        
        let check_result = self.check_with_stats(&mut context);  // see Step 5
        if check_result.is_err() {
            return check_result;
        }

         self.prevent(&mut context)?;

        let pruned_after = self.count_fixed( &mut context);
        if pruned_after > pruned_before {
            self.statistics.propagations_that_pruned += 1;
        }

        Ok(())
    }
    
   


    // defining name, but also priority (?), notify (?), notify__backtrack (?), propagate (when is this called), propagate_from_scratch (and when this? and why was this not implemented in reference)
}

impl<Var: IntegerVariable + 'static> CircuitPropagator<Var> {
    fn remove_self_loops(&self, context: &mut PropagationContext) -> PropagationStatusCP {
        for (zero_indexed_node, domain_of_one_indexed_node) in self.successors.iter().enumerate() {
            context.post(
                predicate!(domain_of_one_indexed_node != index_to_domain_value(zero_indexed_node)),
                conjunction!(),
                &self.inference_code,
            )?;
        }
        Ok(())
    }
}

impl<Var: IntegerVariable + 'static> CircuitPropagator<Var> {
    fn prevent(&self, context:&mut PropagationContext) -> PropagationStatusCP {
        // collect all nodes that have an incoming enforced/fixed edge, these cannot be start of possible chains
        let mut has_incoming_edge = FixedBitSet::with_capacity(self.successors.len());
        // for every fixed edge we find, we follow it and add the resulting node to the list
        for successor in self.successors.iter() {
            if let Some(fixed_value) = context.fixed_value(successor) {
                has_incoming_edge.insert(domain_value_to_index(fixed_value));
            }
        }



        // For every node that has no fixed incoming edge, we try to create a chain.
        for unmarked in has_incoming_edge.zeroes() {
            // If the node has no fixed outgoing edge, we cannot create a chain and go to the next possible node to start a chain.
            let Some(fixed_value) = context.fixed_value(&self.successors[unmarked]) else {
                continue;
            };

            // If it does have an outgoing fixed edge, we can start creating a chain with our starting node.
            let mut chain = vec![unmarked];

            // Now we keep up extending our chain as long as we reach nodes that have a fixed outgoing edge.
            // We already know the upcoming node as we checked if the first node had a fixed outgoing edge;
            let mut next = domain_value_to_index(fixed_value);
            // Set to keep track which nodes have we already visited
            let mut seen = FixedBitSet::with_capacity(self.successors.len());
            seen.insert(unmarked);
            // And then we keep on looping until we end up in a node with no fixed outgoing edge.
            while let Some(fixed_value_next) = context.fixed_value(&self.successors[next]) {
                 // If we arrive at a node that we have seen before we break the loop
                if seen.contains(next) {
                    break;
                }
                seen.insert(next);
                // We add the next value to the chain
                chain.push(next);
                // And continue to unfold the chain from there. As the domains themselves are 1-indexed, we need to transform them to 0-indexed for our own array.
                next = domain_value_to_index(fixed_value_next);
            }

            // We have found a chain. If the last node in the chain has a possible edge to the starting node, we prune that edge only if
            // the length of the chain is not n: if we have not visited all nodes yet we cannot return to the starting node already.
            if context.contains(&self.successors[next], index_to_domain_value(unmarked)) && chain.len() + 1< self.successors.len() {
                let reason = self.create_prevent_explanation(context.domains(), &chain);
                context.post(
                    predicate!(self.successors[next] != index_to_domain_value(unmarked)),
                    reason,
                    &self.inference_code,
                )?;
            }
        }

        Ok(())
    }

    fn create_prevent_explanation(
        &self,
        context: Domains,
        path: &[usize],
    ) -> PropositionalConjunction {
        path.iter()
            .map(|&index| {
                let var = &self.successors[index];

                predicate!(
                    var == context
                        .fixed_value(var)
                        .expect("Expected every variable in the chain to be assigned")
                )
            })
            .collect()
    }
}    

impl<Var: IntegerVariable + 'static> CircuitPropagator<Var> {
    fn check(&self, context: Domains) -> PropagationStatusCP {
        let n = self.successors.len();

        for start in 0..n {
            let mut visited = FixedBitSet::with_capacity(n);
            let mut cycle_path = Vec::new();

            let mut current = start;

            loop {
                // get the domain of the current node
                let domain = &self.successors[current];

                // check if the node already has an enforced edge
                let Some(fixed_value) = context.fixed_value(domain) else {
                    // if no edge is enforced, we stop following the cycle
                    break;
                };

                // if we already visited this node before
                if visited.contains(current) {
                    // check if we visited all nodes in this iteration and whether we would go to the starting node,
                    // creating a Hamiltonian cycle
                    if visited.count_ones(..) == n &&  current == start {
                        return Ok(());
                    }
                    
                    // Otherwise, we raise a conflict
                    return Err(Conflict::Propagator(PropagatorConflict {
                        conjunction: self.create_check_explanation(context, &cycle_path),
                        inference_code: self.inference_code.clone(),
                    }));
                }

                visited.insert(current);
                cycle_path.push(current);

                let next_index = domain_value_to_index(fixed_value);
                if next_index >= n {
                    break;
                    // should not happen; nodes should not be able to refer outside of range
                }
                current = next_index;
            }

        }
        Ok(())
    }
    fn check_with_stats(&mut self, context: &mut PropagationContext) -> PropagationStatusCP {
        let result = self.check(context.domains());
        if result.is_err() {
            // Count fixed edges as depth proxy
            let fixed_count = self.successors.iter()
                .filter(|s| context.fixed_value(*s).is_some())
                .count();
            self.statistics.total_fixed_edges_at_conflict += fixed_count;
            self.statistics.number_of_conflicts += 1;
            self.statistics.propagations_that_found_conflict += 1;
        }
        result
    }
    fn count_fixed(&self, context: &mut PropagationContext) -> usize {
        self.successors.iter()
            .filter(|s| context.fixed_value(*s).is_some())
            .count()
    }

    fn create_check_explanation(
        &self,
        context: Domains,
        cycle: &[usize],
    ) -> PropositionalConjunction {
        cycle
            .iter()
            .map(|&index| {
                let var = &self.successors[index];

                predicate!(
                    var == context
                        .fixed_value(var)
                        .expect("Found a subcycle")
                )
            })
            .collect()
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
    use pumpkin_core::{propagation::ReadDomains, state::State};

    use crate::circuit::CircuitConstructor;

    //VALID FULL HAMILTONIAN PATH (NO CONFLICT)
    #[test]
    fn circuit_hamiltonian_path_conflict_detection() {
        let mut state = State::default();

        let x = state.new_interval_variable(2, 2, None);
        let y = state.new_interval_variable(3, 3, None);
        let z = state.new_interval_variable(1, 1, None);

        let constraint_tag = state.new_constraint_tag();

        let _ = state.add_propagator(CircuitConstructor {
            successors: vec![x, y, z].into(),
            constraint_tag,
        });

        let result = state.propagate_to_fixed_point();

        assert!(
            result.is_ok(),
            "If there is a cycle concerning all variables, then no conflict should be reported"
        )
    }
    // SIMPLE SUBCYCLE (SHOULD CONFLICT)
    #[test]
    fn circuit_conflict_detection_simple() {
        let mut state = State::default();

        let x = state.new_interval_variable(2, 2, None);
        let y = state.new_interval_variable(1, 1, None);
        let z = state.new_interval_variable(1, 3, None);

        let constraint_tag = state.new_constraint_tag();

        let _ = state.add_propagator(CircuitConstructor {
            successors: vec![x, y, z].into(),
            constraint_tag,
        });

        let result = state.propagate_to_fixed_point();

        assert!(
            result.is_err(),
            "If there is a cycle concerning all variables, then no conflict should be reported"
        )
    }

    // SELF LOOP REMOVAL 
    #[test]
    fn circuit_removes_self_loops() {
        let mut state = State::default();
        let x = state.new_interval_variable(1, 3, None);
        let y = state.new_interval_variable(1, 3, None);
        let z = state.new_interval_variable(1, 3, None);

        let constraint_tag = state.new_constraint_tag();
        let _ = state.add_propagator(CircuitConstructor {
            successors: vec![x, y, z].into(),
            constraint_tag,
        });

        let _ = state.propagate_to_fixed_point();
        assert!(
            !state.get_domains().contains(&x, 1), 
            "Self-loop x=1 mst be removed"
        );
    }

    //SELF LOOP FIXED (CONFLICT)
    #[test]
    fn circuit_self_loop_fixed() {
        let mut state = State::default();
        let x = state.new_interval_variable(1, 1, None);
        let y = state.new_interval_variable(1, 3, None);
        let z = state.new_interval_variable(1, 3, None);

        let constraint_tag = state.new_constraint_tag();
        let _ = state.add_propagator(CircuitConstructor {
            successors: vec![x, y, z].into(),
            constraint_tag,
        });

        let result = state.propagate_to_fixed_point();
        assert!(result.is_err(), "Forced self-loop = conflict");
        
    }

    //Prevent should not prune closing hamilton cycle edge
    #[test]
    fn circuit_prevent_not_prune_closing_edge() {
        let mut state = State::default();
        let x = state.new_interval_variable(2, 2, None); 
        let y = state.new_interval_variable(3, 3, None); 
        let z = state.new_interval_variable(1, 3, None); 

        let constraint_tag = state.new_constraint_tag();
        let _ = state.add_propagator(CircuitConstructor {
            successors: vec![x, y, z].into(),
            constraint_tag,
        });

        let _ = state.propagate_to_fixed_point();

        assert!(
        state.get_domains().contains(&z, 1),
        "Closing edge z-x completes a full Hamiltonian cycle and must NOT be pruned"
    );
    }

    //Edge case: single variable must conflict
    #[test]
    fn circuit_single_variable_conflict() {
        let mut state = State::default();

        let x = state.new_interval_variable(1, 1, None);

       let constraint_tag = state.new_constraint_tag();
        let _ = state.add_propagator(CircuitConstructor {
            successors: vec![x].into(),
            constraint_tag,
        });

        let result = state.propagate_to_fixed_point();
        assert!(result.is_err(), "Single node with self-loop must conflict");
    }

    //test two variables okey
    #[test]
    fn circuit_two_variable_cycle_ok() {
        let mut state = State::default();

        let x = state.new_interval_variable(2, 2, None);
        let y = state.new_interval_variable(1, 1, None);

        let constraint_tag = state.new_constraint_tag();
        let _ = state.add_propagator(CircuitConstructor {
            successors: vec![x, y].into(),
            constraint_tag,
        });

        let result = state.propagate_to_fixed_point();
        assert!(result.is_ok(), "2-cycle is a valid Hamiltonian cycle");
    }
        #[test]
    fn circuit_articulation_point_conflict() {
        let mut state = State::default();

        let x1 = state.new_interval_variable(2, 2, None);
        let x2 = state.new_interval_variable(1, 4, None);
        let x3 = state.new_interval_variable(2, 2, None);
        let x4 = state.new_interval_variable(2, 2, None);

        let constraint_tag = state.new_constraint_tag();

        let _ = state.add_propagator(CircuitConstructor {
            successors: vec![x1, x2, x3, x4].into(),
            constraint_tag,
        });

        let result = state.propagate_to_fixed_point();

        assert!(
            result.is_err(),
            "A possible graph with an articulation point cannot contain a Hamiltonian circuit"
        );
    }
}