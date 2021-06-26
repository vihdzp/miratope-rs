use crate::abs::rank::Rank;

use super::{GreekPrefix, Language, Options, Prefix};

/// The English language.
pub struct En;

impl GreekPrefix for En {}

impl Language for En {
    type Count = super::Plural;
    type Gender = super::Agender;

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

    /// The name for a Petrial. This word can't ever be used as a noun.
    fn petrial(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "Petrial"
    }

    /// The name for a hyperblock with a given rank.
    fn hyperblock(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        match rank.into_usize() {
            3 => format!("cuboid{}", options.three("", "s", "al")),
            n => {
                format!("{}block{}", Self::prefix(n), options.two("", "s"))
            }
        }
    }

    /// The name for a hypercube with a given rank.
    fn hypercube(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        match rank.into_usize() {
            3 => format!("cub{}", options.three("e", "s", "ic")),
            4 => format!("tesseract{}", options.three("", "s", "ic")),
            n => {
                let prefix = Self::prefix(n).chars().collect::<Vec<_>>();

                // Penta -> Pente, or Deca -> Deke
                let (_, str0) = prefix.split_last().unwrap();
                let (c1, str1) = str0.split_last().unwrap();

                let suffix = options.three("", "s", "ic");
                if *c1 == 'c' {
                    format!("{}keract{}", str1.iter().collect::<String>(), suffix)
                } else {
                    format!("{}eract{}", str0.iter().collect::<String>(), suffix)
                }
            }
        }
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
