use std::fmt::Debug;

error_chain! {
    types {
        Error, ErrorKind, ResultExt, Result;
    }

    links {
    }

    foreign_links {

    }

    errors {
        NoCandidates(el: Box<Debug + Send>) {
            description("no candidates for selection about element")
            display("no candidates for selection about element {:?}", el)
        }
        SampleTooSmall(given: usize, requested: usize) {
            description("provided sample too small")
            display("provided sample too small (given: {}, requested: {})", given, requested)
        }
        DivideByZero(upper: f64, lower: f64) {
            description("attempted division by zero")
            display("attempted division by zero (upper: {}, lower: {})", upper, lower)
        }
        DuplicateElements(element: String) {
            description("duplicate elements in ordered 'set'")
            display("duplicate element {} found in ordered 'set'", element)
        }
        InsufficientElements(k: usize, l: usize) {
            description("not enough elements to construct set")
            display("not enough elements to construct set of size {}; only found {}", k, l)
        }
        NoConvergence(min: f64, max: f64, iters: usize) {
            description("binary search failed to converge")
            display("binary search failed to converge ([min, max] = [{}, {}], iters = {})", min, max, iters)
        }
    }
}
