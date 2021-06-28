//! Implements the English language.
use crate::{options::Options, GreekPrefix, Language, Position, Prefix};

use miratope_core::abs::rank::Rank;

/// The English language.
pub struct En;

impl GreekPrefix for En {}

impl Prefix for En {
    /// Converts a number into its Greek prefix equivalent.
    fn prefix(n: usize) -> String {
        Self::greek_prefix(n)
    }

    /// The same as the usual Greek prefix, except that we use "duo" instead of
    /// "di" and "trio" instead of "tri". Not sure why.
    fn multi_prefix(n: usize) -> String {
        match n {
            2 => String::from("duo"),
            3 => String::from("trio"),
            _ => Self::prefix(n),
        }
    }

    /// In the case `n == 4`, we return "tesse". Otherwise, this is the same as
    /// the usual Greek prefix, except that we apply the following
    /// transformation rules:
    /// - If the last letter is a vowel, we convert it into an 'e'.
    ///   - If the next to last letter is a 'c', we convert it into a 'k'.
    /// - Otherwise, we add an 'e'.
    fn hypercube_prefix(n: usize) -> String {
        if n == 4 {
            return "tesse".to_owned();
        }

        let mut prefix = Self::prefix(n);
        let mut chars = prefix.char_indices().rev();

        if let Some((idx_last, c_last)) = chars.next() {
            if !super::is_vowel(c_last) {
                // Append an 'e' if it doesn't end with a vowel.
                prefix.push('e');
            } else {
                if let Some((idx_prev, 'c')) = chars.next() {
                    prefix.replace_range(idx_prev..idx_last, "k");
                }

                // Change the final vowel to an 'e'.
                prefix.pop();
                prefix.push('e');
            }
        }

        prefix
    }
}

impl Language for En {
    type Count = crate::options::Plural;
    type Gender = crate::options::Agender;

    /// The default position to place adjectives. This will be used for the
    /// default implementations, but it can be overridden in any specific case.
    fn default_pos() -> Position {
        Position::Before
    }

    fn suffix(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        const SUFFIXES: [&str; 25] = [
            "mon", "tel", "gon", "hedr", "chor", "ter", "pet", "ex", "zett", "yott", "xenn", "dak",
            "hend", "dok", "tradak", "tedak", "pedak", "exdak", "zedak", "yodak", "nedak", "ik",
            "iken", "ikod", "iktr",
        ];

        SUFFIXES[rank.into_usize()].to_owned()
            + if rank == Rank::new(2) {
                options.three("", "s", "al")
            } else if rank == Rank::new(3) {
                options.three("on", "a", "al")
            } else {
                options.three("on", "a", "ic")
            }
    }

    /// The name of a nullitope.
    fn nullitope(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("nullitope", "nullitopes", "nullitopic")
    }

    /// The name of a point.
    fn point(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.two("point", "points")
    }

    /// The name of a dyad.
    fn dyad(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("dyad", "dyads", "dyadic")
    }

    /// The name of a triangle.
    fn triangle(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("triangle", "triangles", "triangular")
    }

    /// The name of a square.
    fn square(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.two("square", "squares")
    }

    /// The name of a rectangle.
    fn rectangle(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("rectangle", "rectangles", "rectangular")
    }

    /// The name of a pyramid.
    fn pyramid(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("pyramid", "pyramids", "pyramidal")
    }

    /// The name of a prism.
    fn prism(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("prism", "prisms", "prismatic")
    }

    /// The name of a tegum.
    fn tegum(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("tegum", "tegums", "tegmatic")
    }

    /// The name of a comb.
    fn comb(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.two("comb", "combs")
    }

    /// The name for an antiprism.
    fn antiprism(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("antiprism", "antiprisms", "antiprismatic")
    }

    /// The name for an antitegum.
    fn antitegum(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("antitegum", "antitegums", "antitegmatic")
    }

    fn hosotope(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        "hoso".to_owned() + &Self::suffix(rank, options)
    }

    /// The name for a Petrial. This word can't ever be used as a noun.
    fn petrial(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "Petrial"
    }

    /// The name for a cuboid.
    fn cuboid(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("cuboid", "cuboids", "cuboidal")
    }

    /// The name for a cube.
    fn cube(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.three("cube", "cubes", "cubic")
    }

    /// The name for a hyperblock with a given rank.
    fn hyperblock(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        format!("{}block{}", Self::prefix(rank.into()), options.two("", "s"))
    }

    /// The name for a hypercube with a given rank.
    fn hypercube(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        Self::hypercube_prefix(rank.into()) + options.three("ract", "racts", "ractic")
    }

    /// The adjective for a "great" version of a polytope.
    fn great(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "great"
    }

    /// The adjective for a "small" version of a polytope.
    fn small(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "small"
    }

    /// The adjective for a "stellated" version of a polytope.
    fn stellated(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "stellated"
    }

    /// The adjective for a "dual" polytope.
    fn dual(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "dual"
    }
}

#[cfg(test)]
mod tests {
    use crate::name::{Abs, Name};

    use super::*;

    #[test]
    /// TODO: expand this test.
    fn names() {
        assert_eq!(
            En::parse(&Name::<Abs>::polygon(Default::default(), 5)),
            "pentagon"
        );

        assert_eq!(
            En::parse(&Name::<Abs>::simplex(Default::default(), Rank::new(5)).prism()),
            "hexateric prism"
        );
    }
}
