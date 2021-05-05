//! French translation. Credits to Blaxapate.

use super::{Language, Options, Position, Prefix};

pub struct Fr;

impl Prefix for Fr {}

impl Language for Fr {
    fn adj_pos() -> Position {
        Position::After
    }

    fn nullitope(options: Options) -> String {
        format!("nullitope{}", options.two("", "s"))
    }

    fn point(options: Options) -> String {
        format!("point{}", options.two("", "s"))
    }

    fn dyad(options: Options) -> String {
        format!("dyad{}", options.four("e", "es", "ique", "iques"))
    }

    fn triangle(options: Options) -> String {
        format!("triang{}", options.four("le", "les", "ulaire", "ulaires"))
    }

    fn rectangle(options: Options) -> String {
        format!("rectangle{}", options.two("", "s"))
    }

    fn square(options: Options) -> String {
        format!("carré{}", options.six("", "s", "", "s", "e", "es"))
    }

    fn pyramid(options: Options) -> String {
        format!("pyramid{}", options.two("", "s"))
    }

    fn prism(options: Options) -> String {
        format!("prisme{}", options.two("", "s"))
    }

    fn tegum(options: Options) -> String {
        format!("tégume{}", options.four("", "s", "aire", "aires"))
    }

    fn dual(options: Options) -> String {
        format!("dua{}", options.six("l", "ux", "l", "ux", "le", "les"))
    }
}
