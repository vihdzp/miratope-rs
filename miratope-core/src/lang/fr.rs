//! French translation. Credits to Blaxapate.

use super::{Bigender, Language, Options, Prefix};

pub struct Fr;

impl Prefix for Fr {}

impl Language for Fr {
    type Count = super::Plural;
    type Gender = Bigender;

    fn nullitope(options: Options<Self::Count, Self::Gender>) -> String {
        format!("nullitope{}", options.two("", "s"))
    }

    fn point(options: Options<Self::Count, Self::Gender>) -> String {
        format!("point{}", options.two("", "s"))
    }

    fn dyad(options: Options<Self::Count, Self::Gender>) -> String {
        format!("dyad{}", options.four("e", "es", "ique", "iques"))
    }

    fn triangle(options: Options<Self::Count, Self::Gender>) -> String {
        format!("triang{}", options.four("le", "les", "ulaire", "ulaires"))
    }

    fn rectangle(options: Options<Self::Count, Self::Gender>) -> String {
        format!("rectangle{}", options.two("", "s"))
    }

    fn square(options: Options<Self::Count, Self::Gender>) -> String {
        format!("carré{}", options.six("", "s", "", "s", "e", "es"))
    }

    fn pyramid(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.two("pyramid", "pyramids")
    }

    fn prism(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.two("prisme", "prismes")
    }

    fn tegum(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.four("tégume", "tégumes", "tégumeaire", "tégumeaires")
    }

    fn dual(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six("dual", "duaux", "dual", "duaux", "duale", "duales")
    }
}
