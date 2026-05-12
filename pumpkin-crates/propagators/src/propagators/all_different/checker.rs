#[derive(Debug, Clone)]
pub struct AllDifferentChecker<Var> {
    sucs: Box<[Var]>,
}

