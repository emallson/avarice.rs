/// Validate that the assumed properties of the `Objective` trait hold for type `$t`.
///
/// To use this macro, your type must implement `Arbitrary` from the `quickcheck` crate.
///
/// # Examples
///
/// ```rust,ignore
/// #[derive(Clone, Debug)]
/// pub struct MyObjective {
///     // ...
/// }
///
/// impl Objective for MyObjective {
///     // ...
/// }
///
/// #[cfg(test)]
/// mod test {
///     use quickcheck::Arbitrary;
///
///     impl Arbitrary for MyObjective {
///         // ...
///     }
///     validate(MyObjective);
/// }
/// ```
#[macro_export]
macro_rules! validate {
    ($t:ty) => {
        use quickcheck::TestResult;
        use std::iter::FromIterator;

        type Obj = $t;
        type Element = <$t as Objective>::Element;
        type State = <$t as Objective>::State;

        fn valid_element(obj: &Obj, u: Element) -> bool {
            obj.elements().position(|x| x == u).is_none()
        }

        fn valid_solution(obj: &Obj, sol: &Set<Element>) -> bool {
            let all_els = Set::from_iter(obj.elements());

            sol.difference(&all_els).count() == 0
        }

        fn state(obj: &Obj, sol: &Set<Element>) -> State {
            let mut state = State::default();
            for &el in sol {
                obj.insert_mut(el, &mut state).unwrap();
            }
            state
        }

        fn independent(obj: &Obj, st: &State, u: Element, v: Element) -> bool {
            obj.depends(u, st).unwrap().position(|x| x == v).is_none()
        }

        quickcheck! {
            /// check that Δ(u | S \cup {u}) == 0
            fn delta_zero_contained(obj: $t, u: <$t as Objective>::Element, sol: Set<<$t as Objective>::Element>) -> TestResult {
                if !valid_element(&obj, u) || !valid_solution(&obj, &sol) {
                    return TestResult::discard();
                }

                let mut sol = sol.clone();
                sol.insert(u);
                let st = state(&obj, &sol);
                TestResult::from_bool(obj.delta(u, &sol, &st).unwrap() == 0.0)
            }

            /// check that ∇(u, v | S \cup {u}) == 1
            fn nabla_one_u_contained(obj: Obj, u: <$t as Objective>::Element, v: <$t as Objective>::Element, sol: Set<<$t as Objective>::Element>) -> TestResult {
                if !valid_element(&obj, u) || !valid_element(&obj, v) || !valid_solution(&obj, &sol) {
                    return TestResult::discard();
                }

                let mut sol = sol.clone();
                sol.insert(u);
                let st = state(&obj, &sol);
                TestResult::from_bool(obj.nabla(u, v, &sol, &st).unwrap() == 1.0)
            }

            /// check that ∇(u, v | S \cup {v}) == 1
            fn nabla_one_v_contained(obj: Obj, u: <$t as Objective>::Element, v: <$t as Objective>::Element, sol: Set<<$t as Objective>::Element>) -> TestResult {
                if !valid_element(&obj, u) || !valid_element(&obj, v) || !valid_solution(&obj, &sol) {
                    return TestResult::discard();
                }

                let mut sol = sol.clone();
                sol.insert(v);
                let st = state(&obj, &sol);
                TestResult::from_bool(obj.nabla(u, v, &sol, &st).unwrap() == 1.0)
            }

            /// check that ∇(u, v | S \cup {v}) == 1
            fn nabla_one_independent(obj: Obj, u: <$t as Objective>::Element, v: <$t as Objective>::Element, sol: Set<<$t as Objective>::Element>) -> TestResult {
                if u == v || !valid_element(&obj, u) || !valid_element(&obj, v) || !valid_solution(&obj, &sol) {
                    return TestResult::discard();
                }

                let st = state(&obj, &sol);
                if !independent(&obj, &st, u, v) {
                    return TestResult::discard();
                }
                TestResult::from_bool(obj.nabla(u, v, &sol, &st).unwrap() == 1.0)
            }

            /// check that the dependency relation is symmetric
            fn dependency_symmetric(obj: Obj, u: <$t as Objective>::Element, sol: Set<<$t as Objective>::Element>) -> TestResult {
                if !valid_element(&obj, u) || !valid_solution(&obj, &sol) {
                    return TestResult::discard();
                }

                let st = state(&obj, &sol);
                let depends = Vec::from_iter(obj.depends(u, &st).unwrap());

                for &dep in &depends {
                    if independent(&obj, &st, dep, u) {
                        return TestResult::failed();
                    }
                }
                TestResult::passed()
            }
        }
    }
}

#[cfg(test)]
mod test {
    use objective::*;
    use errors::*;

    use quickcheck::{Gen, Arbitrary};

    #[derive(Clone, Debug)]
    struct DummyObjective {
        contents: Vec<u8>,
    }

    impl Objective for DummyObjective {
        type State = ();
        type Element = u8;

        fn elements(&self) -> ElementIterator<Self> {
            Box::new(self.contents.iter().cloned())
        }

        fn depends(&self, u: u8, _st: &Self::State) -> Result<ElementIterator<Self>> {
            Ok(Box::new(self.contents
                .iter()
                .cloned()
                .filter(move |&v| v == u)))
        }

        fn benefit(&self, sol: &Set<u8>, _st: &Self::State) -> Result<f64> {
            Ok(sol.len() as f64)
        }

        fn insert_mut(&self, _u: u8, _st: &mut Self::State) -> Result<()> {
            Ok(())
        }
    }

    impl Arbitrary for DummyObjective {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            DummyObjective { contents: Arbitrary::arbitrary(g) }
        }
    }

    validate!(DummyObjective);
}
