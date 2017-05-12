//! A robust, efficient, and generic implementation of the common greedy approximation algorithm
//! for solving the set maximization problem.
extern crate rand;
#[macro_use]
extern crate error_chain;
extern crate rayon;
#[macro_use]
extern crate slog;
extern crate slog_stdlog;
extern crate fnv;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;
#[cfg(test)]
#[macro_use]
extern crate avarice_derive;

pub mod greedy;
pub mod objective;
pub mod ratio;
pub mod errors;
pub mod macros;

#[cfg(test)]
mod test {
    use objective::*;
    use objective::curvature::Bounded;
    use errors::*;
    use std::iter;

    macro_rules! obj {
        ($tr:ident, $name:ident) => {
            #[derive($tr, Debug, Clone)]
            struct $name {}

            impl Objective for $name {
                type Element = ();
                type State = ();

                fn elements(&self) -> ElementIterator<Self> {
                    Box::new(iter::empty())
                }

                fn depends(&self, _u: Self::Element, _st: &Self::State) -> Result<ElementIterator<Self>> {
                    Ok(Box::new(iter::empty()))
                }

                fn benefit(&self, _s: &Set<Self::Element>, _st: &Self::State) -> Result<f64> {
                    Ok(0.0)
                }

                fn insert_mut(&self, _u: Self::Element, _st: &mut Self::State) -> Result<()> {
                    Ok(())
                }
            }

        }
    }

    obj!(Submodular, SubObj);

    #[test]
    fn submod_derived_bounds() {
        assert!(SubObj::bounds() == (None, Some(1.0)));
    }

    obj!(Supermodular, SupObj);

    #[test]
    fn supmod_derived_bounds() {
        assert!(SupObj::bounds() == (Some(1.0), None));
    }

    obj!(Modular, ModObj);

    #[test]
    fn mod_derived_bounds() {
        assert!(ModObj::bounds() == (Some(1.0), Some(1.0)));
    }
}
