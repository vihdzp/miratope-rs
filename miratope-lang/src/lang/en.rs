//! Implements the English language.
use crate::{GreekPrefix, Language, Position, Prefix};

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
            if super::is_vowel(c_last) {
                if let Some((idx_prev, 'c')) = chars.next() {
                    prefix.replace_range(idx_prev..idx_last, "k");
                }

                // Change the final vowel to an 'e' by first popping before appending.
                prefix.pop();
            }
            // Append an 'e'.
            prefix.push('e');
        }

        prefix
    }
}

const SUFFIXES: [&str; 25] = [
    "mon", "tel", "gon", "hedr", "chor", "ter", "pet", "ex", "zett", "yott", "xenn", "dak", "hend",
    "dok", "tradak", "tedak", "pedak", "exdak", "zedak", "yodak", "nedak", "ik", "iken", "ikod",
    "iktr",
];

impl Language for En {
    type Gender = crate::gender::Agender;

    /// The default position to place adjectives. This will be used for the
    /// default implementations, but it can be overridden in any specific case.
    fn default_pos() -> Position {
        Position::Before
    }

    fn suffix_noun_str(rank: usize) -> String {
        let suffix = SUFFIXES[rank].to_owned();

        match rank {
            3 => suffix,
            _ => suffix + "on",
        }
    }

    fn suffix_adj(_: Self::Gender, rank: usize) -> String {
        let suffix = SUFFIXES[rank].to_owned();

        match rank {
            3 | 4 => suffix + "al",
            _ => suffix + "ic",
        }
    }

    fn nullitope_noun_str() -> String {
        "nullitope".to_owned()
    }

    fn nullitope_adj(_: Self::Gender) -> String {
        "nullitopic".to_owned()
    }

    fn point_noun_str() -> String {
        "point".to_owned()
    }

    fn point_adj(_: Self::Gender) -> String {
        "point".to_owned()
    }

    fn dyad_noun_str() -> String {
        "dyad".to_owned()
    }

    fn dyad_adj(_: Self::Gender) -> String {
        "dyadic".to_owned()
    }

    fn triangle_noun_str() -> String {
        "triangle".to_owned()
    }

    fn triangle_adj(_: Self::Gender) -> String {
        "triangular".to_owned()
    }

    fn square_noun_str() -> String {
        "square".to_owned()
    }

    fn square_adj(_: Self::Gender) -> String {
        "square".to_owned()
    }

    fn rectangle_noun_str() -> String {
        "rectangle".to_owned()
    }

    fn rectangle_adj(_: Self::Gender) -> String {
        "rectangular".to_owned()
    }

    fn pyramid_noun_str() -> String {
        "pyramid".to_owned()
    }

    fn pyramid_adj(_: Self::Gender) -> String {
        "pyramidal".to_owned()
    }

    fn prism_noun_str() -> String {
        "prism".to_owned()
    }

    fn prism_adj(_: Self::Gender) -> String {
        "prismatic".to_owned()
    }

    fn tegum_noun_str() -> String {
        "tegum".to_owned()
    }

    fn tegum_adj(_: Self::Gender) -> String {
        "tegmatic".to_owned()
    }

    fn comb_noun_str() -> String {
        "comb".to_owned()
    }

    fn comb_adj(_: Self::Gender) -> String {
        "comb".to_owned()
    }

    fn antiprism_noun_str() -> String {
        "antiprism".to_owned()
    }

    fn antiprism_adj(_: Self::Gender) -> String {
        "antiprismatic".to_owned()
    }

    fn antitegum_noun_str() -> String {
        "antitegum".to_owned()
    }

    fn antitegum_adj(_: Self::Gender) -> String {
        "antitegmatic".to_owned()
    }

    fn hosotope_noun_str(rank: usize) -> String {
        "hoso".to_owned() + &Self::suffix_noun_str(rank)
    }

    fn hosotope_adj(gender: Self::Gender, rank: usize) -> String {
        "hoso".to_owned() + &Self::suffix_adj(gender, rank)
    }

    fn ditope_noun_str(rank: usize) -> String {
        "hoso".to_owned() + &Self::suffix_noun_str(rank)
    }

    fn ditope_adj(gender: Self::Gender, rank: usize) -> String {
        "di".to_owned() + &Self::suffix_adj(gender, rank)
    }

    fn petrial_adj(_: Self::Gender) -> String {
        "Petrial".to_owned()
    }

    fn simplex_noun_str(rank: usize) -> String {
        Self::generic_noun_str(rank, rank)
    }

    fn simplex_adj(gender: Self::Gender, rank: usize) -> String {
        Self::generic_adj(gender, rank, rank)
    }

    fn cuboid_noun_str() -> String {
        "cuboid".to_owned()
    }

    fn cuboid_adj(_: Self::Gender) -> String {
        "cuboidal".to_owned()
    }

    fn cube_noun_str() -> String {
        "cube".to_owned()
    }

    fn cube_adj(_: Self::Gender) -> String {
        "cubic".to_owned()
    }

    fn hyperblock_noun_str(rank: usize) -> String {
        Self::greek_prefix(rank) + "block"
    }

    fn hyperblock_adj(_: Self::Gender, rank: usize) -> String {
        Self::greek_prefix(rank) + "block"
    }

    fn hypercube_noun_str(rank: usize) -> String {
        Self::greek_prefix(rank) + "ract"
    }

    fn hypercube_adj(_: Self::Gender, rank: usize) -> String {
        Self::greek_prefix(rank) + "ractic"
    }

    fn orthoplex_noun_str(rank: usize) -> String {
        Self::generic_noun_str(1 << rank, rank)
    }

    fn orthoplex_adj(gender: Self::Gender, rank: usize) -> String {
        Self::generic_adj(gender, 1 << rank, rank)
    }

    fn great_adj(_: Self::Gender) -> String {
        "great".to_owned()
    }

    fn small_adj(_: Self::Gender) -> String {
        "small".to_owned()
    }

    fn stellated_adj(_: Self::Gender) -> String {
        "stellated".to_owned()
    }

    fn dual_adj(_: Self::Gender) -> String {
        "dual".to_owned()
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
            En::parse(&Name::<Abs>::simplex(Default::default(), 6).prism()),
            "hexateric prism"
        );
    }
}
