//! A language we can use for debugging.

use super::{
    name::{Name, NameType},
    Language, Options, Prefix,
};
use crate::polytope::r#abstract::rank::Rank;

pub struct Dbg;

impl Prefix for Dbg {}

impl Language for Dbg {
    fn suffix(rank: Rank, _options: Options) -> String {
        format!("({}D)", rank)
    }

    fn pyramid_of<T: NameType>(base: &Name<T>, _options: Options) -> String {
        format!("({}) pyramid", Self::parse(base, Options::default()))
    }

    fn prism_of<T: NameType>(base: &Name<T>, _options: Options) -> String {
        format!("({}) prism", Self::parse(base, Options::default()))
    }

    fn tegum_of<T: NameType>(base: &Name<T>, _options: Options) -> String {
        format!("({}) tegum", Self::parse(base, Options::default()))
    }

    fn simplex(rank: Rank, _options: Options) -> String {
        format!("{}-simplex", rank)
    }

    fn hyperblock(rank: Rank, _options: Options) -> String {
        format!("{}-hyperblock", rank)
    }

    fn hypercube(rank: Rank, _options: Options) -> String {
        format!("{}-hypercube", rank)
    }

    fn orthoplex(rank: Rank, _options: Options) -> String {
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
