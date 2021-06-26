//! French translation. Credits to Blaxapate.

use super::{Bigender, Language, Options, Prefix};

pub struct Fr;

impl Prefix for Fr {}

impl Language for Fr {
    type Gender = Bigender;

    fn nullitope(options: Options<Self::Gender>) -> String {
        format!("nullitope{}", options.two("", "s"))
    }

    fn point(options: Options<Self::Gender>) -> String {
        format!("point{}", options.two("", "s"))
    }

    fn dyad(options: Options<Self::Gender>) -> String {
        format!("dyad{}", options.four("e", "es", "ique", "iques"))
    }

    fn triangle(options: Options<Self::Gender>) -> String {
        format!("triang{}", options.four("le", "les", "ulaire", "ulaires"))
    }

    fn rectangle(options: Options<Self::Gender>) -> String {
        format!("rectangle{}", options.two("", "s"))
    }

    fn square(options: Options<Self::Gender>) -> String {
        format!("carré{}", options.six("", "s", "", "s", "e", "es"))
    }

    fn pyramid(options: Options<Self::Gender>) -> String {
        format!("pyramid{}", options.two("", "s"))
    }

    fn prism(options: Options<Self::Gender>) -> String {
        format!("prisme{}", options.two("", "s"))
    }

    fn tegum(options: Options<Self::Gender>) -> String {
        format!("tégume{}", options.four("", "s", "aire", "aires"))
    }

    fn dual(options: Options<Self::Gender>) -> String {
        format!("dua{}", options.six("l", "ux", "l", "ux", "le", "les"))
    }
}
