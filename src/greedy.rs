//! This module contains a (reasonably) optimized implementation of the greedy algorithm, which is
//! also generic over any implementor of the `Objective` trait.
//!
//! The greedy algorithm is exceptionally simple in concept: to construct a solution `S` with `k`
//! elements maximizing `f`, iteratively select the element `e` that maximizes `Δ(e, S) =
//! f(S ∪ {e}) - f(S)` and add that to the solution.
//!
//! The performance of this algorithm depends *heavily* on the implementation of the `Objective`.
//! In particular, `depends` should be as fast as possible (preferably no more than a copy), and
//! the dependent sets should be made as small as possible while preserving correctness.
//!
//! Further, any `Objective` to be optimized with the greedy algorithm should override the `delta`
//! method to improve performance, as even when the `benefit` function is trivial the default
//! implementation still includes an avoidable copy.
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use slog::{DrainExt, Logger};
use slog_stdlog::StdLog;
use objective::*;
use errors::*;

struct WeightedNode<O: Objective> {
    pub weight: f64,
    pub node: O::Element,
}

impl<O: Objective> PartialEq for WeightedNode<O> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<O: Objective> Eq for WeightedNode<O> {}

impl<O: Objective> PartialOrd for WeightedNode<O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.weight.partial_cmp(&other.weight)
    }
}

impl<O: Objective> Ord for WeightedNode<O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight.partial_cmp(&other.weight).unwrap()
    }
}

/// A max-heap-based state-tracking generic implementation of the greedy algorithm finding a
/// set `S` with `k` elements that approximately maximizes the objective `f` (represented by
/// `obj`). If the primal curvature (`obj.nabla`) can be bounded above by 1 and the set of valid
/// solutions is equal `{S | |S| ≤ k, S ⊆ X}` (where `X` is all elements), then this solution is
/// at least `(1 - 1/e) ≈ 63%` as good as any optimal solution.
///
/// The performance of this algorithm depends *heavily* on the implementation of the `Objective`.
/// In particular, `depends` should be as fast as possible (preferably no more than a copy), and
/// the dependent sets should be made as small as possible while preserving correctness.
///
/// Further, any `Objective` to be optimized with the greedy algorithm should override the `delta`
/// method to improve performance, as even when the `benefit` function is trivial the default
/// implementation still includes an avoidable copy.
pub fn greedy<O: Objective + Sync>(obj: &O,
                                   k: usize,
                                   log: Option<Logger>)
                                   -> Result<(f64, Vec<O::Element>, O::State)>
    where O::Element: Send + Sync,
          O::State: Sync
{
    let log = log.unwrap_or_else(|| Logger::root(StdLog.fuse(), o!()));
    let mut state = O::State::default();
    let mut solset = Set::default();
    fn reheap<O: Objective>(sol: &Set<O::Element>,
                            obj: &O,
                            state: &O::State,
                            prior: BinaryHeap<WeightedNode<O>>,
                            elements: Set<O::Element>)
                            -> Result<BinaryHeap<WeightedNode<O>>> {
        prior.into_iter()
            .filter(|ref we| !elements.contains(&we.node))
            .map(|we| Ok(we))
            .chain(elements.iter().cloned().map(|e| {
                let w = obj.delta(e, &sol, &state)?;
                Ok(WeightedNode {
                    node: e,
                    weight: w,
                })
            }))
            .collect()
    }

    let elements = obj.elements().collect::<Set<_>>();
    debug!(log, "initializing heap");
    let mut heap = reheap(&solset,
                          obj,
                          &state,
                          BinaryHeap::with_capacity(elements.len()),
                          elements)?;
    debug!(log, "done initializing heap");

    let mut f = 0.0;
    let mut sol = Vec::with_capacity(k);
    let mut i = 0;
    while let Some(top) = heap.pop() {
        sol.push(top.node);
        solset.insert(top.node.into());
        f += top.weight;
        trace!(log, "selected element"; "node" => format!("{:?}", top.node), "objective value" => f);
        obj.insert_mut(top.node, &mut state)?;
        // NOTE: this assumes a symmetric dependency relationship,
        // i.e. v ∈ D(u) ⇐⇒ u ∈ D(v) ∀ u,v ∈ E
        heap = reheap(&solset,
                      obj,
                      &state,
                      heap,
                      obj.depends(top.node, &state)?
                          .filter(|&u| !solset.contains(&u))
                          .collect())?;

        // can't && i < k in a while let, unfortunately
        i += 1;
        if i >= k {
            break;
        }
    }

    debug!(log, "greedy solution"; "f" => f);
    Ok((f, sol, state))
}

pub fn lazy_greedy<O: Objective>(obj: &O,
                                 k: usize,
                                 log: Option<Logger>)
                                 -> Result<(f64, Vec<O::Element>, O::State)> {
    assert!(O::curv_bounds() == Curvature::Submodular);
    let log = log.unwrap_or_else(|| Logger::root(StdLog.fuse(), o!()));
    let mut state = O::State::default();
    let mut solset = Set::default();
    // an insertion is *incomplete* if it has been added to the solution, but it `insert` has not
    // been called for it to update the state.
    let mut incomplete_insertions: Set<O::Element> = Set::default();
    let mut time_since_reheap = 0;
    let mut times = Vec::with_capacity(k);
    let mut deps_size = Vec::new();
    fn reheap<O: Objective>(sol: &Set<O::Element>,
                            obj: &O,
                            state: &O::State,
                            prior: BinaryHeap<WeightedNode<O>>,
                            elements: Set<O::Element>)
                            -> Result<BinaryHeap<WeightedNode<O>>> {
        prior.into_iter()
            .filter(|ref we| !elements.contains(&we.node))
            .map(|we| Ok(we))
            .chain(elements.iter().cloned().map(|e| {
                let w = obj.delta(e, &sol, &state)?;
                Ok(WeightedNode {
                    node: e,
                    weight: w,
                })
            }))
            .collect()
    }

    let elements = obj.elements().collect::<Set<_>>();
    debug!(log, "initializing heap");
    let mut heap = reheap(&solset,
                          obj,
                          &state,
                          BinaryHeap::with_capacity(elements.len()),
                          elements)?;
    debug!(log, "done initializing heap");

    let mut f = 0.0;
    let mut sol = Vec::with_capacity(k);
    let mut i = 0;
    while let Some(top) = heap.pop() {
        let deps = obj.depends(top.node, &state)?.collect::<Set<_>>();
        let incomplete_deps = deps.iter()
            .filter(|&u| incomplete_insertions.contains(&u))
            .cloned()
            .collect::<Set<_>>();
        let mut todos = Vec::new();
        for &dep in &incomplete_deps {
            obj.insert_mut(dep, &mut state)?;
            incomplete_insertions.remove(&dep);
            todos.extend(obj.depends(dep, &state)?);
        }

        if incomplete_deps.len() > 0 {
            // this may no longer be the top node, put it back on the heap and reheap
            heap.push(top);
            heap = reheap(&solset, obj, &state, heap, todos.into_iter().collect())?;
            deps_size.push(incomplete_deps.len());
            times.push(time_since_reheap);
            time_since_reheap = 0;
            continue;
        }

        time_since_reheap += 1;

        sol.push(top.node);
        solset.insert(top.node.into());
        incomplete_insertions.insert(top.node);
        f += top.weight;
        trace!(log, "selected element"; "node" => format!("{:?}", top.node), "objective value" => f);

        // can't && i < k in a while let, unfortunately
        i += 1;
        if i >= k {
            break;
        }
    }

    info!(log, "greedy solution"; 
          "f" => f,
          "lazy speedup" => incomplete_insertions.len() as f64 / k as f64,
          "insertions skipped" => incomplete_insertions.len(),
          "avg steps between reheaps" => times.iter().sum::<usize>() as f64 / times.len() as f64,
          "avg incomplete deps at reheap" => deps_size.iter().sum::<usize>() as f64 / deps_size.len() as f64);
    Ok((f, sol, state))
}
