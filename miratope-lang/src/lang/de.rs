//! Implements the German language.
use crate::{
    name::{Name, NameType},
    options::{Gender, Options, Plural},
    GreekPrefix, Language, Position, Prefix,
};

use miratope_core::abs::rank::Rank;

/// The gender system for German.
#[derive(Clone, Copy)]
pub enum Trigender {
    /// Male gender.
    Male,

    /// Female gender.
    Female,

    /// Neuter gender.
    Neuter,
}

impl Default for Trigender {
    fn default() -> Self {
        Self::Neuter
    }
}

impl Gender for Trigender {}

/// This implementation uses the fact, specific to German, that plurals for all
/// genders are the same, and that female and plural declensions are the same
/// (in the nominative case).
impl Options<Plural, Trigender> {
    /// Chooses a suffix for an adjective from three options:
    ///
    /// * A singular adjective (male).
    /// * A singular adjective (female).
    /// * A singular adjective (neuter).
    ///
    /// Assumes that the word will be used as an adjective, regardless of
    /// `self.adjective`.
    fn three_adj<'a>(&self, adj_m: &'a str, adj_f: &'a str, adj_n: &'a str) -> &'a str {
        if self.count.is_one() {
            match self.gender {
                Trigender::Male => adj_m,
                Trigender::Female => adj_f,
                Trigender::Neuter => adj_n,
            }
        } else {
            adj_f
        }
    }

    /// Chooses a suffix from five options:
    ///
    /// * Base form.
    /// * A plural.
    /// * A singular adjective (male).
    /// * A singular adjective (female).
    /// * A singular adjective (neuter).
    fn five<'a>(
        &self,
        base: &'a str,
        plural: &'a str,
        adj_m: &'a str,
        adj_f: &'a str,
        adj_n: &'a str,
    ) -> &'a str {
        if self.adjective {
            self.three_adj(adj_m, adj_f, adj_n)
        } else if self.count.is_one() {
            base
        } else {
            plural
        }
    }
}

/// Calls `options.three`, autocompleting the declension rules of German.
macro_rules! three {
    ($options: ident, $adj: literal) => {
        $options.three(concat!($adj, "er"), concat!($adj, "e"), concat!($adj, "es"))
    };
}

/// Calls `options.five`, autocompleting the declension rules of German.
macro_rules! five {
    ($options: ident, $base: literal, $plural: literal, $adj: literal) => {
        $options.five(
            $base,
            $plural,
            concat!($adj, "er"),
            concat!($adj, "e"),
            concat!($adj, "es"),
        )
    };
}

/// The German language.
pub struct De;

impl GreekPrefix for De {}

impl Prefix for De {
    fn prefix(n: usize) -> String {
        Self::greek_prefix(n)
    }
}

impl Language for De {
    type Count = Plural;
    type Gender = Trigender;

    /// The default position to place adjectives. This will be used for the
    /// default implementations, but it can be overridden in any specific case.
    fn default_pos() -> Position {
        Position::Before
    }

    /// Returns the suffix for a d-polytope. Only needs to work up to d = 20, we
    /// won't offer support any higher than that.
    fn suffix(d: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        const SUFFIXES: [&str; 21] = [
            "mon", "tel", "gon", "edr", "cor", "ter", "pet", "ex", "zet", "yot", "xen", "dac",
            "hendac", "doc", "tradac", "teradac", "petadac", "exdac", "zetadac", "yotadac",
            "xendac",
        ];

        SUFFIXES[d.into_usize()].to_owned() + options.four("o", "os", "al", "ales")
    }

    fn nullitope(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        todo!()
    }

    fn point(options: Options<Self::Count, Self::Gender>) -> &'static str {
        five!(options, "Punkt", "Punkte", "punktuell")
    }

    fn dyad(options: Options<Self::Count, Self::Gender>) -> &'static str {
        five!(options, "Dyade", "Dyaden", "dyadisch")
    }

    fn triangle(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        todo!()
    }

    fn square(options: Options<Self::Count, Self::Gender>) -> &'static str {
        five!(options, "Quadrat", "Quadrate", "quadratisch")
    }

    fn rectangle(options: Options<Self::Count, Self::Gender>) -> &'static str {
        five!(options, "Rechteck", "Rechtecke", "rechteckig")
    }

    fn pyramid(options: Options<Self::Count, Self::Gender>) -> &'static str {
        five!(options, "Pyramide", "Pyramiden", "pyramidal")
    }

    fn pyramid_gender() -> Self::Gender {
        Trigender::Female
    }

    fn prism(options: Options<Self::Count, Self::Gender>) -> &'static str {
        five!(options, "Prisma", "Prismen", "prismatisch")
    }

    fn tegum(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        todo!()
    }

    fn comb(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        todo!()
    }

    fn cuboid(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        todo!()
    }

    fn cube(options: Options<Self::Count, Self::Gender>) -> &'static str {
        five!(options, "Würfel", "Würfel", "würfelig")
    }

    fn hyperblock(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        let mut result = Self::prefix(rank.into()) + "block";

        if !options.adjective {
            super::uppercase_mut(&mut result);
        }

        result
    }

    fn hypercube(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        let mut result = Self::hypercube_prefix(rank.into()) + "rakt";

        if !options.adjective {
            super::uppercase_mut(&mut result);
        }

        result
    }

    fn dual(options: Options<Self::Count, Self::Gender>) -> &'static str {
        three!(options, "dual")
    }

    fn antiprism(options: Options<Self::Count, Self::Gender>) -> &'static str {
        five!(options, "Antiprisma", "Antiprismen", "antiprismatisch")
    }

    fn antitegum(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        todo!()
    }

    fn hosotope(_rank: Rank, _options: Options<Self::Count, Self::Gender>) -> String {
        todo!()
    }

    fn petrial(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        todo!()
    }

    fn great(options: Options<Self::Count, Self::Gender>) -> &'static str {
        three!(options, "groß")
    }

    fn small(options: Options<Self::Count, Self::Gender>) -> &'static str {
        three!(options, "klein")
    }

    fn stellated(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "stern"
    }

    fn stellated_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::parse_with(base, options) + Self::stellated(options)
    }
}
