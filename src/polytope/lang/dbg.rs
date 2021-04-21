//! A language we can use for debugging.

use super::{Language, Options, Prefix};

pub struct Dbg;

impl Prefix for Dbg {}

impl Language for Dbg {
    fn suffix(d: usize, _options: Options) -> String {
        format!("({}-elements)", d - 1)
    }

    fn pyramid(base: &super::Name, _options: Options) -> String {
        format!("({}) pyramid", Self::parse(base, Options::default()))
    }

    fn prism(base: &super::Name, _options: Options) -> String {
        format!("({}) prism", Self::parse(base, Options::default()))
    }

    fn tegum(base: &super::Name, _options: Options) -> String {
        format!("({}) tegum", Self::parse(base, Options::default()))
    }

    fn simplex(rank: isize, _options: Options) -> String {
        format!("{}-simplex", rank)
    }

    fn hypercube(rank: isize, _options: Options) -> String {
        format!("{}-hypercube", rank)
    }

    fn orthoplex(rank: isize, _options: Options) -> String {
        format!("{}-orthoplex", rank)
    }
}
