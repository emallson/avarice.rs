//! This module contains a trait representing the (real-valued) objective associated with a
//! discrete optimization problem. The greedy algorithm contained in this crate is implemented for
//! any struct implementing `Objective`.
use std::fmt::Debug;
use std::hash::Hash;
use errors::*;
use fnv::FnvHashSet;
use setlike::Setlike;

pub type Set<E> = FnvHashSet<E>;

pub type ElementIterator<'a, O> = Box<Iterator<Item = <O as Objective>::Element> + 'a>;

pub mod curvature {
    //! Marker traits for Curvature bounds.
    //! These allow compile-time verification that an algorithm called on an `Objective` has its
    //! curvature constraints satisfied.
    //!
    //! As a concrete example, the `greedy::lazier_greedy` method requires a `Submodular`
    //! objective. Previously, this was done by checking an enum and panicking if it failed. Marker
    //! traits move the error from runtime to compilation.

    /// An objective with fixed bounds that are not the (0, 1) of `Submodular` or (1, ∞) of
    /// `Supermodular`.
    pub trait Bounded: super::Objective {
        fn bounds() -> (Option<f64>, Option<f64>);
    }

    /// A submodular objective, which satisfies `∀ S ⊆ T: f(S ∪ {e}) - f(S) ≥ f(T ∪ {e}) - f(T)` or (equivalently) has curvature bounded above by `1`.
    ///
    /// The `Bounded` implementation should return a fixed `(None, Some(1.0))`.
    pub trait Submodular: Bounded {}

    /// A supermodular objective, which satisfies `∀ S ⊆ T: f(S ∪ {e}) - f(S) ≤ f(T ∪ {e}) - f(T)` or (equivalently) has curvature bounded below by `1`.
    ///
    /// The `Bounded` implementation should return a fixed `(Some(1.0), None)`.
    pub trait Supermodular: Bounded {}

    /// A modular objective. The `Bounded` implementation should return a fixed `(Some(1.0),
    /// Some(1.0))`.
    ///
    /// This trait is the least common. The semantics of this type of objective are best understood
    /// by the equation `f(S) = Σf(e)`. That is, the objective is exactly the sum of the weights of
    /// the elements of the set `S`.
    pub trait Modular: Bounded {}
    impl<T: Modular> Submodular for T {}
    impl<T: Modular> Supermodular for T {}
}

/// An objective to be optimized.
///
/// The functions are assumed to be 'internally-stateless'. That is, they do not modify the problem
/// instance, and instead track any relevant information on the associated `State` type. This
/// allows snapshotting of the state and allows safe parallel computation.
pub trait Objective: Sized {
    type Element: Eq + Hash + Clone + Copy + Debug;
    /// Holds any state that is tracked either to assist in correctness or to provide e.g.
    /// cacheing.
    type State: Default + Clone;

    /// The domain of the solution.
    fn elements(&self) -> ElementIterator<Self>;

    /// The implementation of the function `f`. This is the eponymous `Objective`.
    fn benefit<S>(&self, s: &S, state: &Self::State) -> Result<f64>
        where S: Setlike<Self::Element> + IntoIterator<Item = Self::Element>;

    /// The marginal gain $\delta(u \mid S) = f(S \cup u) - f(S)$
    ///
    /// Should return `0` for $u \in S$.
    ///
    /// The default implementation is incredibly inefficient (two calls to `benefit`), so
    /// overriding this is *strongly* encouraged.
    fn delta<S>(&self, u: Self::Element, s: &S, state: &Self::State) -> Result<f64>
        where S: Setlike<Self::Element> + IntoIterator<Item = Self::Element>;

    /// The primal curvature $\nabla(u, v\mid S) = \frac{\delta(u \mid S \cup v)}{\delta(u \mid
    /// S)}$.
    ///
    /// Should return `1` for $u \in S$ or when `depends(u, S)` does not contain `v`.
    ///
    /// The default implementation panics with `unimplemented!()`
    #[allow(unused_variables)]
    fn nabla<S: Setlike<Self::Element>>(&self,
                                        u: Self::Element,
                                        v: Self::Element,
                                        s: &S,
                                        state: &Self::State)
                                        -> Result<f64> {
        unimplemented!()
    }

    /// The elements on which the marginal gain of `u` depends. The technical definition I have
    /// been using is based on the primal curvature (`nabla`): if `u` depends on `v`, then
    /// `nabla(u, v, sol, state) != 1`.
    fn depends(&self, u: Self::Element, state: &Self::State) -> Result<ElementIterator<Self>>;
    /// Same as `depends`, except that `nabla(u, v, sol, state) > 1`.
    ///
    /// Not all objectives need to implement this. If it is not implemented, calling it panics (via
    /// `unimplemented!()`)
    #[allow(unused_variables)]
    fn supermodular_depends(&self,
                            u: Self::Element,
                            state: &Self::State)
                            -> Result<ElementIterator<Self>> {
        unimplemented!()
    }

    /// Update the state tracked in `s` to reflect the insertion of `u` into the solution.
    fn insert_mut(&self, u: Self::Element, s: &mut Self::State) -> Result<()>;

    /// Produce a new state reflecting the insertion of `u` into the solution.
    fn insert(&self, u: Self::Element, s: &Self::State) -> Result<Self::State> {
        let mut state = s.clone();
        self.insert_mut(u, &mut state)?;
        Ok(state)
    }
}

/// Lazy extension of `Objective`. While the same state type is used, state updates are handled
/// independently. If you need two different kinds of state for each kind of update, consider using
/// an enum-struct. For example:
///
/// ```
/// #[derive(Clone, Debug)]
/// pub enum State {
///     Empty,
///     Strict {
///         foo: usize,
///         bar: Vec<usize>,
///     },
///     Lazy {
///         baz: f64,
///     }
/// }
///
/// impl Default for State {
///     fn default() -> Self {
///         State::Empty
///     }
/// }
/// ```
pub trait LazyObjective: Objective {
    /// Updates the marginal gain of `element`, returning it value after updating the state to
    /// reflect the insertion of `previous` elements.
    ///
    /// Returns `Ok(None)` if the marginal gain does not need to be updated.
    fn update_lazy_mut<S: Setlike<Self::Element>>(&self,
                                                  element: Self::Element,
                                                  previous: &S,
                                                  state: &mut Self::State)
                                                  -> Result<Option<f64>>;

    /// Update the state to preserve invariants upon insertion of `element`.
    ///
    /// This should do the least amount of work possible without making `update_lazy_mut` too
    /// complex.
    fn insert_lazy_mut(&self, element: Self::Element, state: &mut Self::State) -> Result<()>;
}

/// An objective with further constraints than mere cardinality.
pub trait ConstrainedObjective: Objective {
    /// Returns true if adding `el` to `sol` would create a valid set.
    fn valid_addition(&self,
                      el: Self::Element,
                      sol: &Vec<Self::Element>,
                      state: &Self::State)
                      -> bool;
}
