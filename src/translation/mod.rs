//! A module dedicated to parsing the names of polytopes into different
//! languages.
//!
//! A great part of the terms we use to describe polytopes are recently coined
//! neologisms and words that haven't entered the wider mathematical sphere.
//! Furthermore, there are some rather large families of words (like those for
//! polygons) that must be translated into the target language. This makes
//! translating Miratope much harder than translating most other software would
//! be. In what follows, we've left extensive documentation, in the hope that it
//! makes the work of anyone trying to translate Miratope much easier.
//!
//! # How does translation work?
//! Every polytope in Miratope is stored alongside its [`Name`]. Names can be
//! thought of as nodes in a tree, which represents how the polytope has been
//! built up. For instance, a pentagonal-cubic duoprism would have a name like
//! this:
//!
//! ```
//! let pecube = Name::Multiprism(Box::new([
//!     Name::Polygon(5),  // 5-gon
//!     Name::Hypercube(3) // 3-hypercube
//! ]));
//! ```
//!
//! To parse a name into a target language, one must specify the following set
//! of options:
//!
//! * `adjective`: Does the polytope act like an adjective?
//! * `count`: How many of the polytope are there?
//! * `gender`: What (if applicable) is the grammatical gender of the polytope?
//!
//! The [`parse`](Language::parse) function takes in this name and arguments,
//! and uses the corresponding methods to parse and combine each of its parts:
//!
//! ```
//! assert_eq!(En::parse(pecube, Options {
//!     adjective: false,
//!     count: 1,
//!     gender: Gender::Male,
//! }), "pentagonal-cubic duoprism");
//! ```
//!
//! # What do I need to code?
//! Though the [`parse`](Language::parse) function is the main way to convert
//! polytopes into their names, in reality, it's just a big `match` statement
//! that calls specific functions to parse every specific polytope type. These
//! are the functions that need to be coded in the target language.

pub mod lang;

const NULLITOPE: &str = "nullitope";

/// A language-independent representation of a polytope name, in a syntax
/// tree-like structure structure.
#[derive(Debug, Clone)]
pub enum Name {
    Nullitope,
    Point,
    Dyad,
    Triangle,
    Square,

    /// A pyramid based on some polytope.
    Pyramid(Box<Name>),

    /// A prism based on some polytope.
    Prism(Box<Name>),

    /// A tegum based on some polytope.
    Tegum(Box<Name>),

    /// A multipyramid based on a list of polytopes.
    Multipyramid(Vec<Name>),

    /// A multiprism based on a list of polytopes.
    Multiprism(Vec<Name>),

    /// A multitegum based on a list of polytopes.
    Multitegum(Vec<Name>),

    /// A multicomb based on a list of polytopes.
    Multicomb(Vec<Name>),

    /// The dual of a specified polytope.
    Dual(Box<Name>),

    /// A simplex of a given dimension.
    Simplex(isize),

    /// A hypercube of a given dimension.
    Hypercube(isize),

    /// An orthoplex of a given dimension.
    Orthoplex(isize),

    /// A convex polygon, not necessarily regular, with a given amount of sides.
    Polygon(usize),

    /// The name of the polytope is unknown.
    Unknown,
}

impl Name {
    /// Simplifies a [`Name`] by taking advantage of equivalences between
    /// constructions. In general, if two things have equivalent names, they'll
    /// get converted into a common name.
    ///
    /// The conversions this method performs are as follows:
    /// * Duals of duals get converted into the original polytope.
    /// * 3-gons and 4-gons get converted into triangles and squares.
    /// * The simplest simplices, hypercubes and orthoplices get turned into
    /// nullitopes, points, dyads, triangles, or squares.
    /// * Multiproducts of multiproducts get turned into a single multiproduct.
    ///
    /// # Assumptions
    /// * This method assumes that only the top layer requires simplifying, and
    /// that any other `Name` that is contained is already simplified.
    pub fn simplify(self) -> Self {
        match self {
            // The dual of a dual is the original polytope.
            Self::Dual(base) => {
                if let Self::Dual(original) = *base {
                    *original
                } else {
                    Self::Dual(base)
                }
            }
            // The simplest polygons have their own names.
            Self::Polygon(n) => match n {
                3 => Self::Triangle,
                4 => Self::Square,
                _ => self,
            },
            // The simplest simplices have their own names.
            Self::Simplex(n) => match n {
                -1 => Self::Nullitope,
                0 => Self::Point,
                1 => Self::Dyad,
                2 => Self::Triangle,
                _ => self,
            },
            // The simplest hypercubes and orthoplices have their own names.
            Self::Hypercube(n) | Self::Orthoplex(n) => match n {
                -1 => Self::Nullitope,
                0 => Self::Point,
                1 => Self::Dyad,
                2 => Self::Square,
                _ => self,
            },
            // Any multipyramid inside of the multipyramid is taken out of the
            // product.
            Self::Multipyramid(bases) => {
                if bases.is_empty() {
                    return Self::Nullitope;
                }

                let mut new_bases = Vec::new();
                let mut pyramid_count = 0;

                // Figures out which bases of the multipyramid are multipyramids
                // themselves, and accounts for them accordingly.
                for base in bases.into_iter() {
                    match base {
                        Self::Nullitope => {}
                        Self::Point => pyramid_count += 1,
                        Self::Dyad => pyramid_count += 2,
                        Self::Simplex(n) => pyramid_count += n + 1,
                        Self::Multipyramid(mut extra_bases) => new_bases.append(&mut extra_bases),
                        _ => new_bases.push(base),
                    }
                }

                let mut multipyramid = Self::Multipyramid(new_bases);
                for _ in 0..pyramid_count {
                    multipyramid = multipyramid.pyramid();
                }
                multipyramid
            }
            _ => self,
        }
    }

    /// Builds a pyramid name from a given name.
    pub fn pyramid(self) -> Self {
        Self::Pyramid(Box::new(self))
    }

    /// Builds a prism name from a given name.
    pub fn prism(self) -> Self {
        Self::Prism(Box::new(self))
    }

    /// Builds a tegum name from a given name.
    pub fn tegum(self) -> Self {
        Self::Tegum(Box::new(self))
    }

    /// Returns the name for a regular polygon of `n` sides.
    pub fn reg_polygon(n: usize) -> Self {
        match n {
            3 => Self::Simplex(2),
            4 => Self::Hypercube(2),
            _ => Self::Polygon(n),
        }
    }

    fn rank_product(&self) -> Option<isize> {
        let (bases, offset) = match self {
            Self::Multipyramid(bases) => (bases, -1),
            Self::Multiprism(bases) | Self::Multitegum(bases) => (bases, 0),
            Self::Multicomb(bases) => (bases, 1),
            _ => return None,
        };

        let mut rank = offset;
        for base in bases.iter() {
            rank += base.rank()? - offset;
        }
        Some(rank)
    }

    /// Returns the rank of the polytope that the name describes, or `None` if
    /// it's unable to figure it out.
    pub fn rank(&self) -> Option<isize> {
        match self {
            Name::Pyramid(base) | Name::Prism(base) | Name::Tegum(base) => Some(base.rank()? + 1),
            Name::Simplex(rank) | Name::Hypercube(rank) | Name::Orthoplex(rank) => Some(*rank),
            Name::Dual(base) => base.rank(),
            Name::Polygon(_) => Some(2),
            Name::Multipyramid(_)
            | Name::Multiprism(_)
            | Name::Multitegum(_)
            | Name::Multicomb(_) => self.rank_product(),
            _ => None,
        }
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
    /// Determines whether the polytope acts as an adjective.
    adjective: bool,

    /// The number of the polytope there are.
    count: usize,

    /// The grammatical gender of the polytope.
    gender: Gender,
}

impl Default for Options {
    /// The options default to a single polytope, as a noun, in neutral gender.
    fn default() -> Self {
        Options {
            adjective: false,
            count: 1,
            gender: Gender::Neutral,
        }
    }
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

/// Trait that allows one to build a prefix from any natural number. Every
/// [`Language`] must implement this trait. If the language implements a
/// Greek-like system for prefixes (e.g. "penta", "hexa"), you should implement
/// this trait via [`GreekPrefix`] instead.
///
/// Defaults to just using `n-` as prefixes.
pub trait Prefix {
    fn prefix(n: usize) -> String {
        format!("{}-", n)
    }
}

/// Trait shared by languages that allow for greek prefixes or anything similar.
/// Every `struct` implementing this trait automatically implements [`Prefix`]
/// as well.
///
/// Defaults to the English ["Wikipedian system."](https://polytope.miraheze.org/wiki/Nomenclature#Wikipedian_system)
pub trait GreekPrefix {
    /// The prefixes for a single digit number.
    const UNITS: [&'static str; 10] = [
        "", "hena", "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "ennea",
    ];

    /// Represents the number 10 for numbers between 10 and 19.
    const DECA: &'static str = "deca";

    /// Represents a factor of 10.
    const CONTA: &'static str = "conta";

    /// The prefix for 11.
    const HENDECA: &'static str = "hendeca";

    /// The prefix for 12.
    const DODECA: &'static str = "dodeca";

    /// The prefix for 20.
    const ICOSA: &'static str = "icosa";

    /// The prefix for 30.
    const TRIACONTA: &'static str = "triaconta";

    /// The prefix for 100.
    const HECTO: &'static str = "hecto";

    /// Represents the number 100 for numbers between 101 and 199.
    const HECATON: &'static str = "hecaton";

    /// Represents a factor of 100.
    const COSA: &'static str = "cosa";

    /// The prefix for 200.
    const DIACOSA: &'static str = "diacosa";

    /// The prefix for 300.
    const TRIACOSA: &'static str = "triacosa";

    /// The prefix for 1000.    
    const CHILIA: &'static str = "chilia";

    /// The prefix for 2000.
    const DISCHILIA: &'static str = "dischilia";

    /// The prefix for 3000.
    const TRISCHILIA: &'static str = "trischilia";

    /// The prefix for 10000.  
    const MYRIA: &'static str = "myria";

    /// The prefix for 20000.
    const DISMYRIA: &'static str = "dismyria";

    /// The prefix for 30000.
    const TRISMYRIA: &'static str = "trismyria";

    /// Converts a number into its Greek prefix equivalent.
    fn greek_prefix(n: usize) -> String {
        match n {
            2..=9 => Self::UNITS[n].to_string(),
            11 => Self::HENDECA.to_string(),
            12 => Self::DODECA.to_string(),
            10 | 13..=19 => format!("{}{}", Self::UNITS[n % 10], Self::DECA),
            20..=29 => format!("{}{}", Self::ICOSA, Self::UNITS[n % 10]),
            30..=39 => format!("{}{}", Self::TRIACONTA, Self::UNITS[n % 10]),
            40..=99 => format!(
                "{}{}{}",
                Self::UNITS[n / 10],
                Self::CONTA,
                Self::UNITS[n % 10]
            ),
            100 => Self::HECTO.to_string(),
            101..=199 => format!("{}{}", Self::HECATON, Self::greek_prefix(n % 100)),
            200..=299 => format!("{}{}", Self::DIACOSA, Self::greek_prefix(n % 100)),
            300..=399 => format!("{}{}", Self::TRIACOSA, Self::greek_prefix(n % 100)),
            400..=999 => format!(
                "{}{}{}",
                Self::UNITS[n / 100],
                Self::COSA,
                Self::greek_prefix(n % 100)
            ),
            1000..=1999 => format!("{}{}", Self::CHILIA, Self::greek_prefix(n % 1000)),
            2000..=2999 => format!("{}{}", Self::DISCHILIA, Self::greek_prefix(n % 1000)),
            3000..=3999 => format!("{}{}", Self::TRISCHILIA, Self::greek_prefix(n % 1000)),
            4000..=9999 => format!(
                "{}{}{}",
                Self::UNITS[n / 1000],
                Self::CHILIA,
                Self::greek_prefix(n % 1000)
            ),
            10000..=19999 => format!("{}{}", Self::MYRIA, Self::greek_prefix(n % 10000)),
            20000..=29999 => format!("{}{}", Self::DISMYRIA, Self::greek_prefix(n % 10000)),
            30000..=39999 => format!("{}{}", Self::TRISMYRIA, Self::greek_prefix(n % 10000)),
            40000..=99999 => format!(
                "{}{}{}",
                Self::UNITS[n / 10000],
                Self::MYRIA,
                Self::greek_prefix(n % 10000)
            ),
            _ => format!("{}-", n),
        }
    }
}

impl<T: GreekPrefix> Prefix for T {
    fn prefix(n: usize) -> String {
        T::greek_prefix(n)
    }
}

/// The trait shared by all languages. Defaults to English.
pub trait Language: Prefix {
    /// Parses the [`Name`] in the specified language, with the given [`Options`].
    fn parse(name: &Name, options: Options) -> String {
        match name {
            Name::Nullitope => Self::nullitope(options),
            Name::Point => Self::point(options),
            Name::Dyad => Self::dyad(options),
            Name::Triangle => Self::triangle(options),
            Name::Square => Self::square(options),
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

    /// The name for a polytope with `n` facets in `d` dimensions.
    fn basic(n: usize, d: usize, options: Options) -> String {
        format!("{}{}", Self::prefix(n), Self::suffix(d, options))
    }

    fn polygon(n: usize, options: Options) -> String {
        Self::basic(n, 2, options)
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

    fn multiproduct(&self) -> String {
        todo!()
    }

    /// The name for a simplex with a given rank.
    fn simplex(rank: isize, options: Options) -> String {
        if rank == -1 {
            NULLITOPE.to_string()
        } else {
            let n = rank as usize;
            Self::basic(n + 1, n, options)
        }
    }

    /// The name for a hypercube with a given rank.
    fn hypercube(rank: isize, options: Options) -> String {
        if rank == -1 {
            NULLITOPE.to_string()
        } else {
            let n = rank as usize;
            Self::basic(2 * n, n, options)
        }
    }

    /// The name for an orthoplex with a given rank.
    fn orthoplex(rank: isize, options: Options) -> String {
        if rank == -1 {
            NULLITOPE.to_string()
        } else {
            let n = rank as usize;
            Self::basic(2u32.pow(n as u32) as usize, n, options)
        }
    }

    /// A placeholder name for a polytope whose name is not known.
    fn unknown() -> String {
        String::from("unknown")
    }
}
