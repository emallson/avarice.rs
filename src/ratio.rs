use errors::*;
use objective::*;
use fnv::FnvHashSet;
use rand::{Rng, thread_rng, sample};
use rand::distributions::{IndependentSample, Range};
use slog::{Drain, Logger};
use slog_stdlog;
use rayon::prelude::*;
use std::f64;
use std::iter::empty;

#[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub enum SampleElement<Element> {
    Independent(Element),
    Dependent(Element),
}

/// Compute the total primal curvature between two sets `S` and `T`. The input `ST` should be `S ∪ T`.
///
/// # Errors
/// If $f_{el}(S) = 0$ but $f_{el}(S \cup T) \neq 0$ then `Err(DivideByZero)` is returned.
#[allow(non_snake_case)]
pub fn gamma<O: Objective>(obj: &O,
                           el: O::Element,
                           ST: &FnvHashSet<O::Element>,
                           S: &FnvHashSet<O::Element>,
                           state: &O::State,
                           joint_state: &O::State)
                           -> Result<f64> {
    let lower = obj.delta(el, S, state)?;
    let upper = obj.delta(el, ST, joint_state)?;

    if lower == upper {
        Ok(1.0)
    } else if lower == 0.0 {
        Err(ErrorKind::DivideByZero(upper, lower).into())
    } else {
        Ok(upper / lower)
    }
}

/// Compute `Λ(T, S)`. The ordering of elements `σ` is given by the ordering of `T`.
///
/// # Errors
/// If `T` contains duplicate elements, `Err(DuplicateElements)` is returned.
/// If any `Γ` returns `Err(DivideByZero)`, that is propagated.
#[allow(non_snake_case)]
pub fn lambda<O: Objective>(obj: &O,
                            T: &Vec<SampleElement<O::Element>>,
                            S: &FnvHashSet<O::Element>,
                            state: &O::State)
                            -> Result<f64> {
    let mut Ti = S.clone();
    let mut tstate = state.clone();
    let mut sum = 0.0;
    for t in T {
        if let &SampleElement::Dependent(t) = t {
            if S.contains(&t) {
                continue;
            }
            obj.insert_mut(t, &mut tstate)?;
            sum += gamma(obj, t, &Ti, S, state, &tstate)?;
            Ti.insert(t);
        } else {
            sum += 1.0; // gamma is 1 for independent elements by definition
        }
    }
    Ok(sum)
}

/// Construct a sample `T ~ D_{p,k}` where `p = bias` and `k` is given.
///
/// For worst case, bias should be 1.
#[allow(non_snake_case)]
pub fn biased_dependency_sample<O: Objective>(obj: &O,
                                              bias: f64,
                                              k: usize)
                                              -> Result<Vec<SampleElement<O::Element>>> {
    use ratio::SampleElement::*;
    let mut rng = thread_rng();
    let mut T = Vec::with_capacity(k);
    let uniform = Range::new(0.0, 1.0);

    fn choose<'a, T: 'a + Copy, I: IntoIterator<Item = &'a T>, R: Rng>(rng: &mut R,
                                                                       iter: I)
                                                                       -> Option<T> {
        let mut iter = iter.into_iter();
        if iter.size_hint().1.unwrap() == 0 {
            return None;
        }
        let range = Range::new(0, iter.size_hint().1.unwrap());
        iter.nth(range.ind_sample(rng)).cloned()
    }

    T.push(Dependent(sample(&mut rng, obj.elements(), 1)[0]));

    let deps = |t, supermodular| if let Dependent(t) = t {
        if supermodular {
            obj.supermodular_depends(t, &O::State::default())
        } else {
            obj.depends(t, &O::State::default())
        }
    } else {
        Ok(Box::new(empty::<O::Element>()) as ElementIterator<O>)
    };

    let element_count = obj.elements().count();
    let mut independent = obj.elements().collect::<FnvHashSet<_>>();
    for dep in deps(T[0], false)? {
        independent.remove(&dep);
    }
    let mut superdepends = deps(T[0], true)?.collect::<FnvHashSet<_>>();

    for i in 1..k {
        let x = if uniform.ind_sample(&mut rng) <= bias {
            // dependent
            match choose(&mut rng, &superdepends) {
                None => {
                    if superdepends.len() + T.len() == element_count {
                        // no more independent elements either
                        return Err(ErrorKind::InsufficientElements(k, i - 1).into());
                    } else {
                        Independent(choose(&mut rng, &independent).unwrap())
                    }
                }
                Some(x) => Dependent(x),
            }
        } else {
            // independent
            if independent.len() == 0 {
                // no more independent elements
                Dependent(choose(&mut rng, &superdepends).ok_or(Error::from(ErrorKind::InsufficientElements(k, i - 1)))?)
            } else {
                Independent(choose(&mut rng, &independent).unwrap())
            }
        };

        for dep in deps(x, false)? {
            independent.remove(&dep);
        }
        superdepends.extend(deps(x, true)?);

        if let Dependent(_) = x {
            assert!(!T.contains(&x));
        }
        T.push(x);

        for &t in &T {
            let t = match t {
                Dependent(t) => t,
                Independent(t) => t,
            };
            superdepends.remove(&t);
            independent.remove(&t);
        }
    }

    assert!(T.len() == k);
    Ok(T)
}

/// Find the value `x` such that `Pr[X ≤ x] ≥ δ` for random variable `X` with `n` i.i.d samples.
///
/// Uses a binary search on the `X_i` to locate a suitable value. The `gap` parameter controls how
/// precise the value found must be: once `max - min <= gap` the binary search will terminate.
///
/// TODO: this could be *much* more efficient than recomputing the cdf for every x tested.
fn empirical_cdf(samples: &Vec<f64>, delta: f64, gap: f64, log: Option<Logger>) -> Result<f64> {
    let log = log.unwrap_or_else(|| Logger::root(slog_stdlog::StdLog.fuse(), o!()));
    const ITERS_MAX: usize = 100;
    if delta > 1.0 - gap as f64 {
        warn!(log, "δ set to a higher precision than η, binary search may never converge"; "δ" => delta, "η" => gap);
    }
    let n = samples.len();
    let mut max =
        samples.iter().filter(|&f| !f.is_nan()).fold(f64::NEG_INFINITY,
                                                     |max, &cur| if cur > max { cur } else { max });
    let mut min = samples.iter()
        .filter(|&f| !f.is_nan())
        .fold(f64::INFINITY, |min, &cur| if cur < min { cur } else { min });
    assert!(max.is_finite());
    assert!(min.is_finite());

    let cdf = |x| samples.iter().filter(|&xi| *xi <= x).count() as f64 / n as f64;
    let mut iters = 0;

    loop {
        let x = (max + min) / 2.0;
        assert!(x.is_finite());
        debug!(log, "testing"; "x" => x, "gap" => max - min, "max" => max, "min" => min);

        let p = cdf(x);

        if p >= delta && max - min <= gap {
            info!(log, "found x with gap satisfied"; "x" => x, "gap" => max - min, "p" => p);
            return Ok(x);
        } else if max - min <= gap {
            // gotten small enough
            let p = cdf(max);
            assert!(p >= delta);
            info!(log, "gap satisfied, but x not found. using max"; "x" => max, "gap" => max - min, "p" => p);
            return Ok(max);
        }

        if p >= delta {
            debug!(log, "p >= δ"; "p" => p);
            // decrease max
            max = x;
        } else {
            debug!(log, "p < δ"; "p" => p);
            // increase min
            min = x;
        }
        iters += 1;
        if iters > ITERS_MAX {
            crit!(log, "binary search failed to converge (likely due to loss of precision from an abnormal number of ζ events)"; "min" => min, "max" => max, "gap" => max - min, "η" => gap);
            return Err(ErrorKind::NoConvergence(min, max, iters).into());
        }
    }
}

pub fn sample_lambda<O: Objective + Sync>(obj: &O,
                                          sol: &FnvHashSet<O::Element>,
                                          state: O::State,
                                          bias: f64,
                                          k: usize,
                                          num_samples: usize)
                                          -> Result<Vec<f64>>
    where O::Element: Send + Sync,
          O::State: Sync
{
    let mut sample_vec = Vec::with_capacity(num_samples);

    (0..num_samples)
        .into_par_iter()
        .map(|_| {
            let sample = biased_dependency_sample(obj, bias, k)?;
            lambda(obj, &sample, sol, &state)
        })
        .collect_into(&mut sample_vec);

    sample_vec.into_iter().collect::<Result<Vec<_>>>()
}

pub fn estimate_lambda<O: Objective + Sync>(obj: &O,
                                            sol: &FnvHashSet<O::Element>,
                                            state: O::State,
                                            bias: f64,
                                            k: usize,
                                            eps: f64,
                                            delta: f64,
                                            delta2: f64,
                                            eta: f64,
                                            log: Option<Logger>)
                                            -> Result<f64>
    where O::Element: Send + Sync,
          O::State: Sync
{
    let log = log.unwrap_or_else(|| Logger::root(slog_stdlog::StdLog.fuse(), o!()));

    let num_samples = (1.0 - delta2).ln() / (-2.0 * eps.powi(2));
    // assert!(eps >= (1.0 / (2.0 * num_samples) * 2f64.ln()).sqrt()); // sufficiency condition for the bound.
    assert!((-2.0 * eps.powi(2) * num_samples).exp() <= 0.5);

    let num_samples = num_samples.ceil() as usize;
    info!(log, "constructing samples"; "num_samples" => num_samples);

    let samples = sample_lambda(obj, sol, state, bias, k, num_samples)?;
    assert!(samples.len() == num_samples);
    let num_nans: usize = samples.iter().filter(|&f| f.is_nan() || f.is_infinite()).count();

    if num_nans > 0 {
        crit!(log, "samples contain NaN / infinite values"; "count" => num_nans);
        panic!("samples contain NaN / infinite values");
    }

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() /
                   (samples.len() as f64 - 1.0);
    let min = samples.iter().fold(f64::INFINITY, |min, xi| xi.min(min));
    let max = samples.iter().fold(0.0, |max, xi| xi.max(max));

    info!(log, "done sampling"; "mean" => mean, "variance" => variance, "min" => min, "max" => max);

    info!(log, "estimating cdf"; "η" => eta);
    match empirical_cdf(&samples, delta, eta, Some(log.new(o!("section" => "cdf")))) {
        Ok(f) => Ok(f),
        Err(Error(ErrorKind::NoConvergence(_, max, _), _)) => Ok(max),
        err => err,
    }
}


pub fn estimate_lambda_chebyshev<O: Objective + Sync>(obj: &O,
                                                      sol: &FnvHashSet<O::Element>,
                                                      state: O::State,
                                                      bias: f64,
                                                      k: usize,
                                                      delta: f64,
                                                      num_samples: usize,
                                                      log: Option<Logger>)
                                                      -> Result<f64>
    where O::Element: Send + Sync,
          O::State: Sync
{
    let log = log.unwrap_or_else(|| Logger::root(slog_stdlog::StdLog.fuse(), o!()));

    info!(log, "constructing samples"; "num_samples" => num_samples);

    let samples = sample_lambda(obj, sol, state, bias, k, num_samples)?;
    assert!(samples.len() == num_samples);
    info!(log, #"no_term", "sampled Λ"; "samples" => format!("{:?}", samples));
    let num_nans: usize = samples.iter().filter(|&f| f.is_nan() || f.is_infinite()).count();

    if num_nans > 0 {
        crit!(log, "samples contain NaN / infinite values"; "count" => num_nans);
        panic!("samples contain NaN / infinite values");
    }

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() /
                   (samples.len() as f64 - 1.0);
    let stddev = variance.sqrt();
    let min = samples.iter().fold(f64::INFINITY, |min, xi| xi.min(min));
    let max = samples.iter().fold(0.0, |max, xi| xi.max(max));

    info!(log, "done sampling"; "mean" => mean, "variance" => variance, "std. dev." => stddev, "min" => min, "max" => max);
    let factor = (1.0 / (1.0 - delta)).sqrt(); // d = 1 - 1/k^2 -> 1/k^2 = 1 - d -> k^2 = 1/(1 - d) -> sqrt(1/(1 - d))
    Ok(mean + factor * stddev)
}

pub fn estimate_lambda_saw<O: Objective + Sync>(obj: &O,
                                                sol: &FnvHashSet<O::Element>,
                                                state: O::State,
                                                bias: f64,
                                                k: usize,
                                                delta: f64,
                                                eta: f64,
                                                num_samples: usize,
                                                log: Option<Logger>)
                                                -> Result<f64>
    where O::Element: Send + Sync,
          O::State: Sync
{
    let log = log.unwrap_or_else(|| Logger::root(slog_stdlog::StdLog.fuse(), o!()));

    info!(log, "constructing samples"; "num_samples" => num_samples);

    let samples = sample_lambda(obj, sol, state, bias, k, num_samples)?;
    assert!(samples.len() == num_samples);
    info!(log, #"no_term", "sampled Λ"; "samples" => format!("{:?}", samples));
    let num_nans: usize = samples.iter().filter(|&f| f.is_nan() || f.is_infinite()).count();

    if num_nans > 0 {
        crit!(log, "samples contain NaN / infinite values"; "count" => num_nans);
        panic!("samples contain NaN / infinite values");
    }

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() /
                   (samples.len() as f64 - 1.0);
    let stddev = variance.sqrt();
    let min = samples.iter().fold(f64::INFINITY, |min, xi| xi.min(min));
    let max = samples.iter().fold(0.0, |max, xi| xi.max(max));

    info!(log, "done sampling"; "mean" => mean, "variance" => variance, "std. dev." => stddev, "min" => min, "max" => max);
    let r = num_samples as f64;
    let factor = binary_search(|lambda| {
                                   1.0 -
                                   1.0 / (r + 1.0) *
                                   ((r + 1.0) / r * ((r - 1.0) / lambda.powi(2) + 1.0)).floor()
                               },
                               (1.0, 100.0),
                               delta,
                               eta,
                               Some(log));
    let factor = match factor {
        Ok(f) => Ok(f),
        Err(Error(ErrorKind::NoConvergence(_, max, _), _)) => Ok(max),
        err => err,
    }?;
    Ok(mean + factor * ((num_samples as f64 + 1.0) / num_samples as f64 * variance).sqrt())
}

/// Does binary search to find the smallest `x` such that f(x) >= goal.
///
/// Terminates when the bounds of the binary search (initially the `(min, max)` given) satisfy `max
/// - min ≤ η`.
fn binary_search<F>(f: F,
                    (mut min, mut max): (f64, f64),
                    goal: f64,
                    gap: f64,
                    log: Option<Logger>)
                    -> Result<f64>
    where F: Fn(f64) -> f64
{
    let log = log.unwrap_or_else(|| Logger::root(slog_stdlog::StdLog.fuse(), o!()));
    const ITERS_MAX: usize = 100;
    assert!(max.is_finite());
    assert!(min.is_finite());

    let mut iters = 0;
    loop {
        let x = (max + min) / 2.0;
        assert!(x.is_finite());
        debug!(log, "testing"; "x" => x, "gap" => max - min, "max" => max, "min" => min);

        let p = f(x);

        if p >= goal && max - min <= gap {
            info!(log, "found x with gap satisfied"; "x" => x, "gap" => max - min, "p" => p);
            return Ok(x);
        } else if max - min <= gap {
            // gotten small enough
            let p = f(max);
            if p < goal {
                crit!(log, "max does not satisfy f(max) >= goal"; "max" => max, "f(max)" => p, "goal" => goal);
                assert!(p >= goal);
            }
            info!(log, "gap satisfied, but x not found. using max"; "x" => max, "gap" => max - min, "p" => p);
            return Ok(max);
        }

        if p >= goal {
            debug!(log, "p >= δ"; "p" => p);
            // decrease max
            max = x;
        } else {
            debug!(log, "p < δ"; "p" => p);
            // increase min
            min = x;
        }
        iters += 1;
        if iters > ITERS_MAX {
            crit!(log, "binary search failed to converge (likely due to loss of precision from an abnormal number of ζ events)"; "min" => min, "max" => max, "gap" => max - min, "η" => gap);
            return Err(ErrorKind::NoConvergence(min, max, iters).into());
        }
    }
}
