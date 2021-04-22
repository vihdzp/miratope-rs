//! A language we can use for debugging.

use super::super::{Language, Name, Options, Prefix};

pub struct Dbg;

impl Prefix for Dbg {}

impl Language for Dbg {
    fn suffix(d: usize, _options: Options) -> String {
        format!("({}-elements)", d - 1)
    }

    fn pyramid(base: &Name, _options: Options) -> String {
        format!("({}) pyramid", Self::parse(base, Options::default()))
    }

    fn prism(base: &Name, _options: Options) -> String {
        format!("({}) prism", Self::parse(base, Options::default()))
    }

    fn tegum(base: &Name, _options: Options) -> String {
        format!("({}) tegum", Self::parse(base, Options::default()))
    }

    fn simplex(rank: usize, _options: Options) -> String {
        format!("{}-simplex", rank)
    }

    fn hypercube(rank: usize, _options: Options) -> String {
        format!("{}-hypercube", rank)
    }

    fn orthoplex(rank: usize, _options: Options) -> String {
        format!("{}-orthoplex", rank)
    }

    fn multiproduct(name: &Name, _options: Options) -> String {
        let (bases, kind) = match name {
            Name::Multipyramid(bases) => (bases, "multipyramid"),
            Name::Multiprism(bases) => (bases, "multiprism"),
            Name::Multitegum(bases) => (bases, "multitegum"),
            Name::Multicomb(bases) => (bases, "multicomb"),
            _ => panic!("Not a product!"),
        };

        let mut str_bases = String::new();

        let (last, bases) = bases.split_last().unwrap();
        for base in bases {
            str_bases.push_str(&format!("({})", Self::parse(base, _options)));
            str_bases.push_str(", ");
        }
        str_bases.push_str(&format!("({})", Self::parse(last, _options)));

        format!("({}) {}", str_bases, kind)
    }
}
