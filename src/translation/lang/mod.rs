mod dbg;
mod en;
mod es;

pub use dbg::Dbg;
pub use en::En;
pub use es::Es;

use super::{name::NameType, Gender, Name, Options, Prefix};

pub fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

pub fn parentheses(str: String, paren: bool) -> String {
    if paren {
        format!("({})", str)
    } else {
        str
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
            Name::Orthodiagonal => Self::orthodiagonal(options),
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
            SUFFIXES[d],
            if d == 2 {
                Self::three(options, "", "s", "al")
            } else if d == 3 {
                Self::three(options, "on", "a", "al")
            } else {
                Self::three(options, "on", "a", "ic")
            }
        )
    }

    /// The name of a nullitope.
    fn nullitope(options: Options) -> String {
        format!("nullitop{}", Self::three(options, "e", "es", "ic"))
    }

    /// The name of a point.
    fn point(options: Options) -> String {
        format!("point{}", Self::two(options, "", "s"))
    }

    /// The name of a dyad.
    fn dyad(options: Options) -> String {
        format!("dyad{}", Self::three(options, "", "s", "ic"))
    }

    /// The name of a triangle.
    fn triangle<T: NameType>(_regular: T, options: Options) -> String {
        format!("triang{}", Self::three(options, "le", "les", "ular"))
    }

    /// The name of a square.
    fn square(options: Options) -> String {
        format!("square{}", Self::two(options, "", "s"))
    }

    /// The name of a rectangle.
    fn rectangle(options: Options) -> String {
        format!("rectang{}", Self::three(options, "le", "les", "ular"))
    }

    /// The name of an orthodiagonal quadrilateral. You should probably just
    /// default this one to "tetragon," as it only exists for tracking purposes.
    fn orthodiagonal(options: Options) -> String {
        Self::generic(4, 2, options)
    }

    /// The generic name for a polytope with `n` facets in `d` dimensions.
    fn generic(n: usize, d: usize, options: Options) -> String {
        format!("{}{}", Self::prefix(n), Self::suffix(d, options))
    }

    fn base<T: NameType>(base: &Name<T>, options: Options) -> String {
        parentheses(Self::parse(base, options), options.parentheses)
    }

    fn base_adj<T: NameType>(base: &Name<T>, options: Options) -> String {
        parentheses(
            Self::parse(
                base,
                Options {
                    adjective: true,
                    ..options
                },
            ),
            options.parentheses,
        )
    }

    fn pyramidal(options: Options) -> String {
        format!("pyramid{}", Self::three(options, "", "s", "al"))
    }

    /// The name for a pyramid with a given base.
    fn pyramid<T: NameType>(base: &Name<T>, options: Options) -> String {
        format!(
            "{} {}",
            Self::base_adj(base, options),
            Self::pyramidal(options)
        )
    }

    fn prismatic(options: Options) -> String {
        format!("prism{}", Self::three(options, "", "s", "atic"))
    }

    /// The name for a prism with a given base.
    fn prism<T: NameType>(base: &Name<T>, options: Options) -> String {
        format!(
            "{} {}",
            Self::base_adj(base, options),
            Self::prismatic(options)
        )
    }

    fn tegmatic(options: Options) -> String {
        format!("teg{}", Self::three(options, "um", "ums", "matic"))
    }

    /// The name for a tegum with a given base.
    fn tegum<T: NameType>(base: &Name<T>, options: Options) -> String {
        format!(
            "{} {}",
            Self::base_adj(base, options),
            Self::tegmatic(options)
        )
    }

    fn multiproduct<T: NameType>(name: &Name<T>, options: Options) -> String {
        // Gets the bases and the kind of multiproduct.
        let (bases, kind) = match name {
            Name::Multipyramid(bases) => (bases, Self::pyramidal(options)),
            Name::Multiprism(bases) => (bases, Self::prismatic(options)),
            Name::Multitegum(bases) => (bases, Self::tegmatic(options)),
            Name::Multicomb(bases) => (bases, format!("comb{}", Self::two(options, "", "s"))),
            _ => panic!("Not a product!"),
        };
        dbg!(&kind);

        let n = bases.len();
        let prefix = match n {
            2 => String::from("duo"),
            3 => String::from("trio"),
            _ => Self::prefix(n),
        };
        let kind = format!("{}{}", prefix, kind);

        let mut str_bases = String::new();

        let (last, bases) = bases.split_last().unwrap();
        for base in bases {
            str_bases.push_str(&Self::base_adj(base, options));
            str_bases.push('-');
        }
        str_bases.push_str(&Self::base_adj(last, options));

        format!("{} {}", str_bases, kind)
    }

    /// The name for a simplex with a given rank.
    fn simplex<T: NameType>(_regular: T, rank: usize, options: Options) -> String {
        Self::generic(rank + 1, rank, options)
    }

    /// The name for a hypercube with a given rank.
    fn hypercube<T: NameType>(regular: T, rank: usize, options: Options) -> String {
        if regular.is_regular() {
            match rank {
                3 => format!("cub{}", Self::three(options, "e", "s", "ic")),
                4 => format!("tesseract{}", Self::three(options, "", "s", "ic")),
                _ => {
                    let prefix = Self::prefix(rank).chars().collect::<Vec<_>>();

                    // Penta -> Pente, or Deca -> Deke
                    let (_, str0) = prefix.split_last().unwrap();
                    let (c1, str1) = str0.split_last().unwrap();

                    let suffix = Self::three(options, "", "s", "ic");
                    if *c1 == 'c' {
                        format!("{}keract{}", str1.iter().collect::<String>(), suffix)
                    } else {
                        format!("{}eract{}", str0.iter().collect::<String>(), suffix)
                    }
                }
            }
        } else {
            match rank {
                3 => format!("cuboid{}", Self::three(options, "", "s", "al")),
                _ => {
                    format!("{}block{}", Self::prefix(rank), Self::two(options, "", "s"))
                }
            }
        }
    }

    /// The name for an orthoplex with a given rank.
    fn orthoplex<T: NameType>(_regular: T, rank: usize, options: Options) -> String {
        Self::generic(2u32.pow(rank as u32) as usize, rank, options)
    }

    /// The name for the dual of another polytope.
    fn dual<T: NameType>(base: &Name<T>, options: Options) -> String {
        format!("dual {}", Self::base(base, options))
    }

    /// A placeholder name for a polytope whose name is not known.
    fn unknown() -> String {
        String::from("unknown")
    }

    /// Chooses a suffix from two options:
    ///
    /// * Base form.
    /// * A plural.
    ///
    /// Assumes that plurals are from 2 onwards.
    fn two<'a>(options: Options, base: &'a str, plural: &'a str) -> &'a str {
        if options.count > 1 {
            plural
        } else {
            base
        }
    }

    /// Chooses a suffix from three options:
    ///
    /// * Base form.
    /// * A plural.
    /// * An adjective for both the singular and plural.
    ///
    /// Assumes that plurals are from 2 onwards.
    fn three<'a>(options: Options, base: &'a str, plural: &'a str, adj: &'a str) -> &'a str {
        if options.adjective {
            adj
        } else if options.count > 1 {
            plural
        } else {
            base
        }
    }

    /// Chooses a suffix from four options:
    ///
    /// * Base form.
    /// * A plural.
    /// * A singular adjective.
    /// * A plural adjective.
    ///
    /// Assumes that plurals are from 2 onwards.
    fn four<'a>(
        options: Options,
        base: &'a str,
        plural: &'a str,
        adj: &'a str,
        plural_adj: &'a str,
    ) -> &'a str {
        if options.adjective {
            if options.count == 1 {
                adj
            } else {
                plural_adj
            }
        } else if options.count == 1 {
            base
        } else {
            plural
        }
    }

    /// Chooses a suffix from six options:
    ///
    /// * Base form.
    /// * A plural.
    /// * A singular adjective (male).
    /// * A plural adjective (male).
    /// * A singular adjective (female).
    /// * A plural adjective (female).
    ///
    /// Assumes that plurals are from 2 onwards.
    fn six<'a>(
        options: Options,
        base: &'a str,
        plural: &'a str,
        adj_m: &'a str,
        plural_adj_m: &'a str,
        adj_f: &'a str,
        plural_adj_f: &'a str,
    ) -> &'a str {
        if options.adjective {
            if options.count == 1 {
                match options.gender {
                    Gender::Male => adj_m,
                    Gender::Female => adj_f,
                    _ => panic!("Unexpected gender!"),
                }
            } else {
                match options.gender {
                    Gender::Male => plural_adj_m,
                    Gender::Female => plural_adj_f,
                    _ => panic!("Unexpected gender!"),
                }
            }
        } else if options.count == 1 {
            base
        } else {
            plural
        }
    }
}
