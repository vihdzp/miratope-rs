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

impl Trigender {
    /// Chooses one of three arguments lazily depending on the gender of `self`.
    fn choose_lazy<T, F1, F2, F3>(self, m: F1, f: F2, n: F3) -> T
    where
        F1: FnOnce() -> T,
        F2: FnOnce() -> T,
        F3: FnOnce() -> T,
    {
        match self {
            Self::Male => m(),
            Self::Female => f(),
            Self::Neuter => n(),
        }
    }

    /// Chooses one of three arguments depending on the gender of `self`.
    fn choose<T>(self, m: T, f: T, n: T) -> T {
        self.choose_lazy(|| m, || f, || n)
    }

    /// Conjugates a string according to the usual German declension rules.
    fn auto(self, str: String) -> String {
        self.choose_lazy(|| str + "er", || str + "e", || str + "es")
    }
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
    type Gender = Trigender;

    fn default_pos() -> Position {
        Position::Before
    }

    fn suffix_noun_str(d: Rank) -> String {
        todo!()
    }

    fn suffix_adj(d: Rank, _: Self::Gender) -> String {
        todo!()
    }

    fn nullitope_noun_str() -> String {
        "Nullitop".to_owned()
    }

    fn nullitope_gender() -> Self::Gender {
        Trigender::Neuter
    }

    fn nullitope_adj(gender: Self::Gender) -> String {
        gender.auto("nullitopisch".to_owned())
    }

    fn point_noun_str() -> String {
        "Punkt".to_owned()
    }

    fn point_gender() -> Self::Gender {
        Trigender::Male
    }

    fn point_adj(gender: Self::Gender) -> String {
        gender.auto("punktuell".to_owned())
    }

    fn dyad_noun_str() -> String {
        "Dyade".to_owned()
    }

    fn dyad_gender() -> Self::Gender {
        Trigender::Female
    }

    fn dyad_adj(gender: Self::Gender) -> String {
        gender.auto("dyadisch")
    }

    fn triangle_noun_str() ->String {
        "Dreieck".to_owned()
    }

    fn triangle_gender() ->Self::Gender {
        Trigender::Neuter
    }

    fn triangle_adj(gender:Self::Gender) ->String {
        gender.auto("dreieckig")
    }

    fn square_noun_str() ->String {
        "Quadrat".to_owned()
    }

    fn square_gender() ->Self::Gender {
        Trigender::Neuter
    }

    fn square_adj(gender:Self::Gender) ->String {
        gender.auto("quadratisch")
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
