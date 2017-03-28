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
    }
}
