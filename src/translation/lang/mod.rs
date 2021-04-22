mod dbg;
mod en;
mod es;

pub use dbg::Dbg;
pub use en::En;
pub use es::Es;

use super::{Name, Options, Prefix};

/// A convenience method for declensing nouns in English.
///
/// In English, there's only three different ways a word can be modified:
///
/// * It can be made into an adjective.
/// * It can be made into a plural noun.
/// * It can remain as a singular noun.
///
/// This method reads the options and returns whichever string applies in the
/// specific case.
fn adj_or_plural<'a>(options: Options, adj: &'a str, plural: &'a str, none: &'a str) -> &'a str {
    if options.adjective {
        adj
    } else if options.count > 1 {
        plural
    } else {
        none
    }
}

/// The trait shared by all languages. Defaults to English.
pub trait Language: Prefix {
    /// Parses the [`Name`] in the specified language, with the given [`Options`].
    fn parse(name: &Name, options: Options) -> String {
        debug_assert!(name.is_valid());

        match name {
            Name::Nullitope => Self::nullitope(options),
            Name::Point => Self::point(options),
            Name::Dyad => Self::dyad(options),
            Name::Triangle => Self::triangle(options),
            Name::Square => Self::square(options),
            Name::Generic(n, d) => Self::generic(*n, *d, options),
            Name::Pyramid(base) => Self::pyramid(base, options),
            Name::Prism(base) => Self::prism(base, options),
            Name::Tegum(base) => Self::tegum(base, options),
            Name::Multipyramid(_)
            | Name::Multiprism(_)
            | Name::Multitegum(_)
            | Name::Multicomb(_) => Self::multiproduct(name, options),
            Name::Simplex(rank) => Self::simplex(*rank, options),
            Name::Hypercube(rank) => Self::hypercube(*rank, options),
            Name::Orthoplex(rank) => Self::orthoplex(*rank, options),
            _ => Self::unknown(),
        }
    }

    /// Returns the suffix for a d-polytope. Only needs to work up to d = 20, we
    /// won't offer support any higher than that.
    fn suffix(d: usize, options: Options) -> String {
        const SUFFIXES: [&str; 21] = [
            "mon", "tel", "gon", "hedr", "chor", "ter", "pet", "ex", "zett", "yott", "xenn", "dak",
            "hendak", "dok", "tradak", "teradak", "petadak", "exdak", "zettadak", "yottadak",
            "xendak",
        ];

        format!(
            "{}{}",
            SUFFIXES.get(d).unwrap_or(&"top"),
            if d == 2 {
                adj_or_plural(options, "al", "s", "")
            } else if d == 3 {
                adj_or_plural(options, "al", "a", "on")
            } else {
                adj_or_plural(options, "ic", "a", "on")
            }
        )
    }

    /// The name of a nullitope.
    fn nullitope(options: Options) -> String {
        format!("nullitop{}", adj_or_plural(options, "ic", "es", "e"))
    }

    /// The name of a point.
    fn point(options: Options) -> String {
        format!("point{}", adj_or_plural(options, "", "s", ""))
    }

    /// The name of a dyad.
    fn dyad(options: Options) -> String {
        format!("dyad{}", adj_or_plural(options, "ic", "s", ""))
    }

    /// The name of a triangle.
    fn triangle(options: Options) -> String {
        format!("triang{}", adj_or_plural(options, "ular", "les", "le"))
    }

    /// The name of a square.
    fn square(options: Options) -> String {
        format!("square{}", adj_or_plural(options, "", "s", ""))
    }

    /// The generic name for a polytope with `n` facets in `d` dimensions.
    fn generic(n: usize, d: usize, options: Options) -> String {
        format!("{}{}", Self::prefix(n), Self::suffix(d, options))
    }

    /// The name for a pyramid with a given base.
    fn pyramid(base: &Name, options: Options) -> String {
        format!(
            "{} pyramid{}",
            Self::parse(
                base,
                Options {
                    adjective: true,
                    gender: options.gender,
                    count: 1
                }
            ),
            adj_or_plural(options, "al", "s", "")
        )
    }

    /// The name for a prism with a given base.
    fn prism(base: &Name, options: Options) -> String {
        format!(
            "{} prism{}",
            Self::parse(
                base,
                Options {
                    adjective: true,
                    gender: options.gender,
                    count: 1
                }
            ),
            adj_or_plural(options, "atic", "s", "")
        )
    }

    /// The name for a tegum with a given base.
    fn tegum(base: &Name, options: Options) -> String {
        format!(
            "{} teg{}",
            Self::parse(
                base,
                Options {
                    adjective: true,
                    gender: options.gender,
                    count: 1
                }
            ),
            adj_or_plural(options, "matic", "a", "um")
        )
    }

    fn multiproduct(name: &Name, options: Options) -> String {
        // Gets the bases and the kind of multiproduct.
        let (bases, kind) = match name {
            Name::Multipyramid(bases) => (
                bases,
                format!("pyramid{}", adj_or_plural(options, "al", "s", "")),
            ),
            Name::Multiprism(bases) => (
                bases,
                format!("prism{}", adj_or_plural(options, "atic", "s", "")),
            ),
            Name::Multitegum(bases) => (
                bases,
                format!("teg{}", adj_or_plural(options, "matic", "ums", "um")),
            ),
            Name::Multicomb(bases) => (
                bases,
                format!("comb{}", adj_or_plural(options, "", "s", "")),
            ),
            _ => panic!("Not a product!"),
        };

        let n = bases.len();
        let prefix = match n {
            2 => String::from("duo"),
            3 => String::from("trio"),
            _ => Self::prefix(n),
        };
        let kind = format!("{}{}", prefix, kind);

        let mut str_bases = String::new();
        let parse_base = |base| {
            format!(
                "{}",
                Self::parse(
                    base,
                    Options {
                        adjective: true,
                        ..Default::default()
                    }
                )
            )
        };

        let (last, bases) = bases.split_last().unwrap();
        for base in bases {
            str_bases.push_str(&parse_base(base));
            str_bases.push_str("-");
        }
        str_bases.push_str(&parse_base(last));

        format!("{} {}", str_bases, kind)
    }

    /// The name for a simplex with a given rank.
    fn simplex(rank: usize, options: Options) -> String {
        let n = rank as usize;
        Self::generic(n + 1, n, options)
    }

    /// The name for a hypercube with a given rank.
    fn hypercube(rank: usize, options: Options) -> String {
        let n = rank as usize;
        Self::generic(2 * n, n, options)
    }

    /// The name for an orthoplex with a given rank.
    fn orthoplex(rank: usize, options: Options) -> String {
        let n = rank as usize;
        Self::generic(2u32.pow(n as u32) as usize, n, options)
    }

    /// A placeholder name for a polytope whose name is not known.
    fn unknown() -> String {
        String::from("unknown")
    }
}
