extern crate avarice;
#[macro_use]
extern crate avarice_derive;

#[cfg(test)]
mod test {
    use avarice::objective::*;
    use avarice::objective::curvature::Bounded;
    use avarice::errors::*;
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
