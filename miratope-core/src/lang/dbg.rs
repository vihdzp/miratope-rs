//! A language we can use for debugging.

use super::{
    name::{Name, NameType},
    Language, Options, Prefix,
};
use crate::abs::rank::Rank;

pub struct Dbg;

impl Prefix for Dbg {}

impl Language for Dbg {
    type Gender = super::Agender;

    fn suffix(rank: Rank, _options: Options<Self::Gender>) -> String {
        format!("({}D)", rank)
    }

    fn pyramid_of<T: NameType>(base: &Name<T>, _options: Options<Self::Gender>) -> String {
        format!("({}) pyramid", Self::parse_with(base, Default::default()))
    }

    fn prism_of<T: NameType>(base: &Name<T>, _options: Options<Self::Gender>) -> String {
        format!("({}) prism", Self::parse_with(base, Default::default()))
    }

    fn tegum_of<T: NameType>(base: &Name<T>, _options: Options<Self::Gender>) -> String {
        format!("({}) tegum", Self::parse_with(base, Default::default()))
    }

    fn simplex(rank: Rank, _options: Options<Self::Gender>) -> String {
        format!("{}-simplex", rank)
    }

    fn hyperblock(rank: Rank, _options: Options<Self::Gender>) -> String {
        format!("{}-hyperblock", rank)
    }

    fn hypercube(rank: Rank, _options: Options<Self::Gender>) -> String {
        format!("{}-hypercube", rank)
    }

    fn orthoplex(rank: Rank, _options: Options<Self::Gender>) -> String {
        format!("{}-orthoplex", rank)
    }

    fn multiproduct<T: NameType>(name: &Name<T>, _options: Options<Self::Gender>) -> String {
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
            str_bases.push_str(&format!("({})", Self::parse_with(base, _options)));
            str_bases.push_str(", ");
        }
        str_bases.push_str(&format!("({})", Self::parse_with(last, _options)));

        format!("({}) {}-{}", str_bases, bases.len(), kind)
    }
}
