mod dbg;
mod en;
mod es;

pub use dbg::Dbg;
pub use en::En;
pub use es::Es;

use super::{name::NameType, Name, Options, Prefix};

pub fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

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
    fn parse<T: NameType>(name: &Name<T>, options: Options) -> String {
        debug_assert!(name.is_valid(), "Invalid name {:?}.", name);

        match name {
            Name::Nullitope => Self::nullitope(options),
            Name::Point => Self::point(options),
            Name::Dyad => Self::dyad(options),
            Name::Triangle(regular) => Self::triangle(*regular, options),
            Name::Square => Self::square(options),
            Name::Rectangle => Self::rectangle(options),
            Name::Generic(n, d) => Self::generic(*n, *d, options),
            Name::Pyramid(base) => Self::pyramid(base, options),
            Name::Prism(base) => Self::prism(base, options),
            Name::Tegum(base) => Self::tegum(base, options),
            Name::Multipyramid(_)
            | Name::Multiprism(_)
            | Name::Multitegum(_)
            | Name::Multicomb(_) => Self::multiproduct(name, options),
            Name::Simplex(regular, rank) => Self::simplex(*regular, *rank, options),
            Name::Hypercube(regular, rank) => Self::hypercube(*regular, *rank, options),
            Name::Orthoplex(regular, rank) => Self::orthoplex(*regular, *rank, options),
            Name::Dual(base) => Self::dual(base, options),
            Name::Unknown => Self::unknown(),
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
    fn triangle<T: NameType>(_regular: T, options: Options) -> String {
        format!("triang{}", adj_or_plural(options, "ular", "les", "le"))
    }

    /// The name of a square.
    fn square(options: Options) -> String {
        format!("square{}", adj_or_plural(options, "", "s", ""))
    }

    /// The name of a rectangle.
    fn rectangle(options: Options) -> String {
        format!("rectang{}", adj_or_plural(options, "ular", "les", "le"))
    }

    /// The generic name for a polytope with `n` facets in `d` dimensions.
    fn generic(n: usize, d: usize, options: Options) -> String {
        format!("{}{}", Self::prefix(n), Self::suffix(d, options))
    }

    /// The name for a pyramid with a given base.
    fn pyramid<T: NameType>(base: &Name<T>, options: Options) -> String {
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
    fn prism<T: NameType>(base: &Name<T>, options: Options) -> String {
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
    fn tegum<T: NameType>(base: &Name<T>, options: Options) -> String {
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

    fn multiproduct<T: NameType>(name: &Name<T>, options: Options) -> String {
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
            Self::parse(
                base,
                Options {
                    adjective: true,
                    ..Default::default()
                },
            )
        };

        let (last, bases) = bases.split_last().unwrap();
        for base in bases {
            str_bases.push_str(&parse_base(base));
            str_bases.push('-');
        }
        str_bases.push_str(&parse_base(last));

        format!("{} {}", str_bases, kind)
    }

    /// The name for a simplex with a given rank.
    fn simplex<T: NameType>(_regular: T, rank: usize, options: Options) -> String {
        Self::generic(rank + 1, rank, options)
    }

    /// The name for a hypercube with a given rank.
    fn hypercube<T: NameType>(_regular: T, rank: usize, options: Options) -> String {
        match rank {
            3 => format!("cub{}", adj_or_plural(options, "ic", "s", "e")),
            4 => format!("tesseract{}", adj_or_plural(options, "ic", "s", "")),
            _ => {
                let mut prefix = Self::prefix(rank).chars().collect::<Vec<_>>();
                let len = prefix.len();

                // Penta -> Pente, or Deca -> Deke
                if let Some(c) = prefix.last_mut() {
                    if is_vowel(*c) {
                        *c = 'e';
                    }
                }
                if let Some(c) = prefix.get_mut(len - 2) {
                    if *c == 'c' {
                        *c = 'k';
                    }
                }

                format!(
                    "{}ract{}",
                    prefix.into_iter().collect::<String>(),
                    adj_or_plural(options, "ic", "s", "")
                )
            }
        }
    }

    /// The name for an orthoplex with a given rank.
    fn orthoplex<T: NameType>(_regular: T, rank: usize, options: Options) -> String {
        Self::generic(2u32.pow(rank as u32) as usize, rank, options)
    }

    /// The name for the dual of another polytope.
    fn dual<T: NameType>(base: &Name<T>, options: Options) -> String {
        format!("dual {}", Self::parse(base, options))
    }

    /// A placeholder name for a polytope whose name is not known.
    fn unknown() -> String {
        String::from("unknown")
    }
}
