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
extern crate setlike;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;
#[cfg(test)]
#[macro_use]
extern crate avarice_derive;

pub mod greedy;
pub mod objective;
pub mod errors;
pub mod macros;
