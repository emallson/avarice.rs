//! This module contains a trait representing the (real-valued) objective associated with a
//! discrete optimization problem. The greedy algorithm contained in this crate is implemented for
//! any struct implementing `Objective`.
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use rand::{Rng, thread_rng};
use rand::distributions::{Range, IndependentSample};
use errors::*;

#[derive(Copy, Clone, Debug)]
pub enum SampleElement<E> {
    Dependent(E),
    NonDependent,
}

use self::SampleElement::*;

pub type Set<E> = HashSet<E>;

pub type ElementIterator<'a, O> = Box<Iterator<Item = <O as Objective>::Element> + 'a>;

/// An objective to be optimized.
///
/// The functions are assumed to be 'internally-stateless'. That is, they do not modify the problem
/// instance, and instead track any relevant information on the associated `State` type. This
/// allows snapshotting of the state and allows safe parallel computation.
///
/// Any implementation must at a minimum implement `elements`, `benefit`, and `insert_mut`, though
/// at least implementing `delta` is strongly encouraged.  `nabla` may be omitted if your
/// application does not need it.
pub trait Objective: Sized {
    type Element: Eq + Hash + Clone + Copy + Debug;
    /// Holds any state that is tracked either to assist in correctness or to provide e.g.
    /// cacheing.
    type State: Default + Clone;

    /// The domain of the solution.
    fn elements(&self) -> ElementIterator<Self>;

    /// The implementation of the function `f`. This is the eponymous `Objective`.
    fn benefit(&self, s: &Set<Self::Element>, state: &Self::State) -> Result<f64>;

    /// The marginal gain $\delta(u \mid S) = f(S \cup u) - f(S)$
    ///
    /// Should return `0` for $u \in S$.
    ///
    /// The default implementation is incredibly inefficient (two calls to `benefit`), so
    /// overriding this is *strongly* encouraged.
    fn delta(&self, u: Self::Element, s: &Set<Self::Element>, state: &Self::State) -> Result<f64> {
        let next_state = self.insert(u, state)?;
        let mut next_set = s.clone();
        next_set.insert(u);

        Ok(self.benefit(&next_set, &next_state)? - self.benefit(s, state)?)
    }

    /// The primal curvature $\nabla(u, v\mid S) = \frac{\delta(u \mid S \cup v)}{\delta(u \mid
    /// S)}$.
    ///
    /// Should return `1` for $u \in S$ or when `depends(u, S)` does not contain `v`.
    ///
    /// The default implementation should be avoided, as it computes this result directly by
    /// calling `delta` twice.
    fn nabla(&self,
             u: Self::Element,
             v: Self::Element,
             s: &Set<Self::Element>,
             state: &Self::State)
             -> Result<f64> {
        let next_state = self.insert(v, state)?;
        let mut next_set = s.clone();
        next_set.insert(v);

        let lower = self.delta(u, s, state)?;
        let upper = self.delta(u, &next_set, &next_state)?;
        if lower == 0.0 && upper == 0.0 {
            Ok(1.0)
        } else if lower == 0.0 {
            Err(ErrorKind::DivideByZero(upper, lower).into())
        } else {
            Ok(upper / lower)
        }
    }

    /// The elements on which the marginal gain of `u` depends. The technical definition I have
    /// been using is based on the primal curvature (`nabla`): if `u` depends on `v`, then
    /// `nabla(u, v, sol, state) == 1`.
    fn depends(&self, u: Self::Element, state: &Self::State) -> Result<ElementIterator<Self>>;

    /// Update the state tracked in `s` to reflect the insertion of `u` into the solution.
    fn insert_mut(&self, u: Self::Element, s: &mut Self::State) -> Result<()>;

    /// Produce a new state reflecting the insertion of `u` into the solution.
    fn insert(&self, u: Self::Element, s: &Self::State) -> Result<Self::State> {
        let mut state = s.clone();
        self.insert_mut(u, &mut state)?;
        Ok(state)
    }

    /// Uniformly sample a sequence of `k` elements from about element `u` consistent with initial
    /// state `state`. *The state is not updated, so if the dependent set is state-dependent,
    /// future elements may not be valid selections.*
    fn sample_sequence(&self,
                       u: Self::Element,
                       k: usize,
                       bias: Option<f64>,
                       sol: &Set<Self::Element>,
                       state: &Self::State)
                       -> Result<Vec<SampleElement<Self::Element>>> {
        let mut sample = Vec::with_capacity(k);
        let mut deps = self.depends(u, state)?
            .filter(|&v| !sol.contains(&v))
            .collect::<Vec<_>>();
        let mut rng = thread_rng();
        rng.shuffle(&mut deps);

        let bias = bias.unwrap_or_else(|| {
            deps.len() as f64 / (self.elements().count() - sol.len()) as f64
        });

        let uniform = Range::new(0.0, 1.0);

        for _ in 0..k {
            if uniform.ind_sample(&mut rng) <= bias && !deps.is_empty() {
                // take a dependent element
                sample.push(Dependent(deps.pop().unwrap()));
            } else {
                sample.push(NonDependent);
            }
        }

        Ok(sample)
    }

    /// Compute the total primal curvature of an element `u` after the first `k` elements of
    /// `sequence` have been added. If the `sequence` does not have at least `k` elements,
    /// `ErrorKind::SampleTooSmall` is returned.
    fn gamma(&self,
             u: Self::Element,
             k: usize,
             mut sol: Set<Self::Element>,
             sequence: Vec<SampleElement<Self::Element>>,
             mut state: Self::State)
             -> Result<(f64, Set<Self::Element>, Self::State)> {
        if sequence.len() < k {
            return Err(ErrorKind::SampleTooSmall(sequence.len(), k).into());
        }
        let mut prod = 1f64;
        for e in sequence.iter().take(k) {
            prod *= match e {
                &Dependent(v) => self.nabla(u, v, &sol, &state)?,
                // proof of this relation is in the 2017 notebook, pg. 8-9
                //
                // the gist of it is that the non-dependence of x implies a pair of f_u(S \cup {x})
                // = f_u(S)-style relations, which are used to show ∇(u, v | S) = ∇(u, v | S \cup
                // {x}).
                &NonDependent => 1f64,
            };

            if let &Dependent(v) = e {
                self.insert_mut(v, &mut state)?;
                sol.insert(v);
            }
        }

        Ok((prod, sol, state))
    }

    /// Compute the sequence of total primal curvature values for `1..k` elements of `sequence`.
    /// Unlike `gamma`, this does not return the final solution or state. `gamma(0|S)` is omitted
    /// as it is constant `1`.
    fn gamma_seq(&self,
                 u: Self::Element,
                 k: usize,
                 mut sol: Set<Self::Element>,
                 sequence: Vec<SampleElement<Self::Element>>,
                 mut state: Self::State)
                 -> Result<Vec<f64>> {
        if sequence.len() < k {
            return Err(ErrorKind::SampleTooSmall(sequence.len(), k).into());
        }
        let mut prod = 1f64;
        let mut seq = Vec::with_capacity(k);
        for e in sequence.iter().take(k) {
            prod *= match e {
                &Dependent(v) => self.nabla(u, v, &sol, &state)?,
                // proof of this relation is in the 2017 notebook, pg. 8-9
                //
                // the gist of it is that the non-dependence of x implies a pair of f_u(S \cup {x})
                // = f_u(S)-style relations, which are used to show ∇(u, v | S) = ∇(u, v | S \cup
                // {x}).
                &NonDependent => 1f64,
            };

            seq.push(prod);

            if let &Dependent(v) = e {
                self.insert_mut(v, &mut state)?;
                sol.insert(v);
            }
        }
        Ok(seq)
    }
}
