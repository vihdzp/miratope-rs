mod en;
mod es;
pub use en::En;
pub use es::Es;

const NULLITOPE: &str = "nullitope";

/// Represents the lowercase name of a polytope, in a syntax-tree structure.
#[derive(Debug, Clone)]
pub enum Name {
    /// A pyramid based on some polytope.
    Pyramid(Box<Name>),

    /// A prism based on some polytope.
    Prism(Box<Name>),

    /// A tegum based on some polytope.
    Tegum(Box<Name>),

    /// A simplex of a given dimension.
    Simplex(isize),

    /// A hypercube of a given dimension.
    Hypercube(isize),

    /// An orthoplex of a given dimension.
    Orthoplex(isize),

    /// A convex polygon with a given amount of sides.
    Polygon(usize),

    /// The name of the polytope is unknown.
    Unknown,
}

impl Name {
    pub fn pyramid(&self) -> Self {
        Self::Pyramid(Box::new(self.clone()))
    }

    pub fn prism(&self) -> Self {
        Self::Prism(Box::new(self.clone()))
    }

    pub fn tegum(&self) -> Self {
        Self::Tegum(Box::new(self.clone()))
    }
}

/// The different grammatical genders.
#[derive(Clone, Copy)]
pub enum Gender {
    Male,
    Female,
    Neutral,
}

/// Represents the different modifiers that can be applied to a term.
#[derive(Clone, Copy)]
pub struct Options {
    /// Is the polytope being used as an adjective?
    adjective: bool,

    /// How many of this polytope are there?
    count: usize,

    /// What's the grammatical gender
    gender: Gender,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            adjective: false,
            count: 1,
            gender: Gender::Male,
        }
    }
}

fn adj_or_plural<'a>(options: Options, adj: &'a str, plural: &'a str, none: &'a str) -> &'a str {
    if options.adjective {
        adj
    } else if options.count > 1 {
        plural
    } else {
        none
    }
}

/// Trait shared by languages that allow for greek prefixes. Defaults to English.
pub trait GreekPrefix {
    /// The prefix for 0.
    fn zero(alone: bool) -> &'static str {
        if alone {
            "nulli"
        } else {
            ""
        }
    }

    /// The prefix for 1.
    fn one(alone: bool) -> &'static str {
        if alone {
            "mono"
        } else {
            "hena"
        }
    }

    /// The adder for 10.
    fn plus_ten() -> &'static str {
        "deca"
    }

    /// The multiplier for 10.
    fn tenfold() -> &'static str {
        "conta"
    }

    /// The prefix for 11.
    fn eleven() -> &'static str {
        "hendeca"
    }

    /// The prefix for 12.
    fn twelve() -> &'static str {
        "dodeca"
    }

    /// The prefix for 20.
    fn twenty() -> &'static str {
        "icosa"
    }

    /// The prefix for 30.
    fn thirty() -> &'static str {
        "triaconta"
    }

    /// The prefix for 100.
    fn hundred() -> &'static str {
        "hecto"
    }

    /// The adder for 100.
    fn plus_hundred() -> &'static str {
        "hecaton"
    }

    /// The multiplier for 100.
    fn hundredfold() -> &'static str {
        "cosa"
    }

    /// The prefix for 200.
    fn two_hundred() -> &'static str {
        "diacosa"
    }

    /// The prefix for 300.
    fn three_hundred() -> &'static str {
        "triacosa"
    }

    /// The prefix for 1000.
    fn thousand() -> &'static str {
        "chilia"
    }

    /// The prefix for 2000.
    fn two_thousand() -> &'static str {
        "dischilia"
    }

    /// The prefix for 3000.
    fn three_thousand() -> &'static str {
        "trischilia"
    }

    /// The prefix for 10000.
    fn ten_thousand() -> &'static str {
        "myria"
    }

    /// The prefix for 20000.
    fn twenty_thousand() -> &'static str {
        "dismyria"
    }

    /// The prefix for 30000.
    fn thirty_thousand() -> &'static str {
        "trismyria"
    }

    /// Should return the empty string for zero.
    fn units(n: usize) -> &'static str {
        const UNITS: [&str; 10] = [
            "", "hena", "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "ennea",
        ];

        UNITS[n]
    }

    fn prefix(n: usize, _options: Options) -> String {
        Self::prefix_trailing(n, _options, true)
    }

    fn prefix_trailing(n: usize, _options: Options, alone: bool) -> String {
        match n {
            0 => Self::zero(alone).to_string(),
            1 => Self::one(alone).to_string(),
            2..=9 => Self::units(n).to_string(),
            11 => Self::eleven().to_string(),
            12 => Self::twelve().to_string(),
            10 | 13..=19 => format!("{}{}", Self::units(n % 10), Self::plus_ten()),
            20..=29 => format!("{}{}", Self::twenty(), Self::units(n % 10)),
            30..=39 => format!("{}{}", Self::thirty(), Self::units(n % 10)),
            40..=99 => format!(
                "{}{}{}",
                Self::units(n / 10),
                Self::tenfold(),
                Self::units(n % 10)
            ),
            100 => Self::hundred().to_string(),
            101..=199 => format!(
                "{}{}",
                Self::plus_hundred(),
                Self::prefix_trailing(n % 100, _options, false)
            ),
            200..=299 => format!(
                "{}{}",
                Self::two_hundred(),
                Self::prefix_trailing(n % 100, _options, false)
            ),
            300..=399 => format!(
                "{}{}",
                Self::three_hundred(),
                Self::prefix_trailing(n % 100, _options, false)
            ),
            400..=999 => format!(
                "{}{}{}",
                Self::units(n / 100),
                Self::hundredfold(),
                Self::prefix_trailing(n % 100, _options, false)
            ),
            1000..=1999 => format!(
                "{}{}",
                Self::thousand(),
                Self::prefix_trailing(n % 1000, _options, false)
            ),
            2000..=2999 => format!(
                "{}{}",
                Self::two_thousand(),
                Self::prefix_trailing(n % 1000, _options, false)
            ),
            3000..=3999 => format!(
                "{}{}",
                Self::three_thousand(),
                Self::prefix_trailing(n % 1000, _options, false)
            ),
            4000..=9999 => format!(
                "{}{}{}",
                Self::units(n / 1000),
                Self::thousand(),
                Self::prefix_trailing(n % 1000, _options, false)
            ),
            10000..=19999 => format!(
                "{}{}",
                Self::ten_thousand(),
                Self::prefix_trailing(n % 10000, _options, false)
            ),
            20000..=29999 => format!(
                "{}{}",
                Self::twenty_thousand(),
                Self::prefix_trailing(n % 10000, _options, false)
            ),
            30000..=39999 => format!(
                "{}{}",
                Self::thirty_thousand(),
                Self::prefix_trailing(n % 10000, _options, false)
            ),
            40000..=99999 => format!(
                "{}{}{}",
                Self::units(n / 10000),
                Self::ten_thousand(),
                Self::prefix_trailing(n % 10000, _options, false)
            ),
            _ => format!("{}-", n),
        }
    }
}

/// The trait shared by all languages. Defaults to English.
pub trait Language: GreekPrefix {
    /// Parses the [`Name`] in the specified language, with the given [`Options`].
    fn parse(name: &Name, options: Options) -> String {
        match name {
            Name::Pyramid(base) => Self::pyramid(base, options),
            Name::Prism(base) => Self::prism(base, options),
            Name::Tegum(base) => Self::tegum(base, options),
            Name::Simplex(rank) => Self::simplex(*rank, options),
            Name::Hypercube(rank) => Self::hypercube(*rank, options),
            Name::Orthoplex(rank) => Self::orthoplex(*rank, options),
            Name::Polygon(n) => Self::polygon(*n, options),
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

    fn simple(n: usize, d: usize, options: Options) -> String {
        format!("{}{}", Self::prefix(n, options), Self::suffix(d, options))
    }

    fn polygon(n: usize, options: Options) -> String {
        Self::simple(n, 2, options)
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

    /// The name for a simplex with a given rank.
    fn simplex(rank: isize, options: Options) -> String {
        if rank == -1 {
            NULLITOPE.to_string()
        } else {
            let n = rank as usize;
            Self::simple(n + 1, n, options)
        }
    }

    /// The name for a hypercube with a given rank.
    fn hypercube(rank: isize, options: Options) -> String {
        if rank == -1 {
            NULLITOPE.to_string()
        } else {
            let n = rank as usize;
            Self::simple(2 * n, n, options)
        }
    }

    /// The name for an orthoplex with a given rank.
    fn orthoplex(rank: isize, options: Options) -> String {
        if rank == -1 {
            NULLITOPE.to_string()
        } else {
            let n = rank as usize;
            Self::simple(2u32.pow(n as u32) as usize, n, options)
        }
    }

    /// A placeholder name for a polytope whose name is not known.
    fn unknown() -> String {
        String::from("unknown")
    }
}
