use pumpkin_core::declare_inference_label;
use pumpkin_core::proof::ConstraintTag;
use pumpkin_core::proof::InferenceCode;
use pumpkin_core::propagation::InferenceCheckers;
use pumpkin_core::propagation::PropagationContext;
use pumpkin_core::propagation::Propagator;
use pumpkin_core::propagation::PropagatorConstructor;
use pumpkin_core::propagation::PropagatorConstructorContext;
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