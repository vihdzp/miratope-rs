//! A language we can use for debugging.

use crate::translation::name::NameType;

use super::super::{Language, Name, Options, Prefix};

pub struct Dbg;

impl Prefix for Dbg {}

impl Language for Dbg {
    fn suffix(d: usize, _options: Options) -> String {
        format!("({}D)", d)
    }

    fn pyramid<T: NameType>(base: &Name<T>, _options: Options) -> String {
        format!("({}) pyramid", Self::parse(base, Options::default()))
    }

    fn prism<T: NameType>(base: &Name<T>, _options: Options) -> String {
        format!("({}) prism", Self::parse(base, Options::default()))
    }

    fn tegum<T: NameType>(base: &Name<T>, _options: Options) -> String {
        format!("({}) tegum", Self::parse(base, Options::default()))
    }

    fn simplex<T: NameType>(_regular: T, rank: usize, _options: Options) -> String {
        format!("{}-simplex", rank)
    }

    fn hypercube<T: NameType>(regular: T, rank: usize, _options: Options) -> String {
        if regular.is_regular() {
            format!("{}-hypercube", rank)
        } else {
            format!("{}-hypercuboid", rank)
        }
    }

    fn orthoplex<T>(_regular: T, rank: usize, _options: Options) -> String {
        format!("{}-orthoplex", rank)
    }

    fn multiproduct<T: NameType>(name: &Name<T>, _options: Options) -> String {
        let (bases, kind) = match name {
            Name::Multipyramid(bases) => (bases, "pyramid"),
            Name::Multiprism(bases) => (bases, "prism"),
            Name::Multitegum(bases) => (bases, "tegum"),
            Name::Multicomb(bases) => (bases, "comb"),
            _ => panic!("Not a product!"),
        };

        let mut str_bases = String::new();

        let (last, bases) = bases.split_last().unwrap();
        for base in bases {
            str_bases.push_str(&format!("({})", Self::parse(base, _options)));
            str_bases.push_str(", ");
        }
        str_bases.push_str(&format!("({})", Self::parse(last, _options)));

        format!("({}) {}-{}", str_bases, bases.len(), kind)
    }
}
