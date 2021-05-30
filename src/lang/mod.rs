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
//! For more information, see the [`Name`] module's documentation.
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
//!
//! The list of functions you'll definitely need to translate are the following:
//!
//! - [`nullitope`](Language::nullitope)
//! - [`point`](Language::point)
//! - [`dyad`](Language::dyad)
//! - [`triangle`](Language::triangle)
//! - [`square`](Language::square)
//! - [`rectangle`](Language::rectangle)
//! - [`generic`](Language::generic)
//! - [`pyramid`](Language::pyramid)
//! - [`prism`](Language::prism)
//! - [`tegum`](Language::tegum)
//! - [`hyperblock`](Language::hyperblock)
//! - [`hypercube`](Language::hypercube)
//! - [`dual`](Language::dual)

pub mod dbg;
pub mod en;
pub mod es;
mod fr;
mod ja;
pub mod name;
mod pii;

pub use dbg::Dbg;
pub use en::En;
pub use es::Es;
pub use fr::Fr;
pub use ja::Ja;
pub use pii::Pii;

use crate::{
    lang::name::{Name, NameData, NameType},
    polytope::r#abstract::rank::Rank,
};

use strum_macros::EnumIter;

/// The different grammatical genders.
#[derive(Clone, Copy, Debug)]
pub enum Gender {
    Male,
    Female,
    None,
}

impl std::ops::BitOr for Gender {
    type Output = Gender;

    fn bitor(self, rhs: Self) -> Self::Output {
        match self {
            Self::None => rhs,
            _ => self,
        }
    }
}

impl Default for Gender {
    fn default() -> Self {
        Gender::None
    }
}

/// Represents the different modifiers that can be applied to a term.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Determines whether the polytope acts as an adjective.
    pub adjective: bool,

    /// The number of the polytope there are.
    pub count: usize,

    /// The grammatical gender of the polytope.
    pub gender: Gender,

    /// Whether we use parentheses for non-ambiguity.
    pub parentheses: bool,
}

impl Default for Options {
    /// The options default to a single polytope, as a noun, in neutral gender.
    fn default() -> Self {
        Options {
            adjective: false,
            count: 1,
            gender: Default::default(),
            parentheses: false,
        }
    }
}

impl Options {
    /// Chooses a suffix from two options:
    ///
    /// * Base form.
    /// * A plural.
    ///
    /// Assumes that plurals are from 2 onwards.
    fn two<'a>(&self, base: &'a str, plural: &'a str) -> &'a str {
        if self.count > 1 {
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
    fn three<'a>(&self, base: &'a str, plural: &'a str, adj: &'a str) -> &'a str {
        if self.adjective {
            adj
        } else if self.count > 1 {
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
        &self,
        base: &'a str,
        plural: &'a str,
        adj: &'a str,
        plural_adj: &'a str,
    ) -> &'a str {
        if self.adjective {
            if self.count == 1 {
                adj
            } else {
                plural_adj
            }
        } else if self.count == 1 {
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
        &self,
        base: &'a str,
        plural: &'a str,
        adj_m: &'a str,
        plural_adj_m: &'a str,
        adj_f: &'a str,
        plural_adj_f: &'a str,
    ) -> &'a str {
        if self.adjective {
            if self.count == 1 {
                match self.gender {
                    Gender::Male => adj_m,
                    Gender::Female => adj_f,
                    _ => panic!("Unexpected gender!"),
                }
            } else {
                match self.gender {
                    Gender::Male => plural_adj_m,
                    Gender::Female => plural_adj_f,
                    _ => panic!("Unexpected gender!"),
                }
            }
        } else if self.count == 1 {
            base
        } else {
            plural
        }
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

    /// Represents the number 20 for numbers between 21 and 29.
    const ICOSI: &'static str = "icosi";

    /// The prefix for 30.
    const TRIACONTA: &'static str = "triaconta";

    /// The prefix for 100.
    const HECTO: &'static str = "hecto";

    /// Represents the number 100 for numbers between 101 and 199.
    const HECATON: &'static str = "hecaton";

    /// Represents a factor of 100.
    const COSA: &'static str = "cosa";

    /// The prefix for 200.
    const DIACOSI: &'static str = "diacosi";

    /// The prefix for 300.
    const TRIACOSI: &'static str = "triacosi";

    /// Represents the number 100 for numbers between 400 and 999.
    const COSI: &'static str = "cosi";

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
            0..=9 => Self::UNITS[n].to_string(),
            11 => Self::HENDECA.to_string(),
            12 => Self::DODECA.to_string(),
            10 | 13..=19 => format!("{}{}", Self::UNITS[n % 10], Self::DECA),
            20 => Self::ICOSA.to_string(),
            21..=29 => format!("{}{}", Self::ICOSI, Self::UNITS[n % 10]),
            30..=39 => format!("{}{}", Self::TRIACONTA, Self::UNITS[n % 10]),
            40..=99 => format!(
                "{}{}{}",
                Self::UNITS[n / 10],
                Self::CONTA,
                Self::UNITS[n % 10]
            ),
            100 => Self::HECTO.to_string(),
            101..=199 => format!("{}{}", Self::HECATON, Self::greek_prefix(n % 100)),
            200..=299 => format!("{}{}", Self::DIACOSI, Self::greek_prefix(n % 100)),
            300..=399 => format!("{}{}", Self::TRIACOSI, Self::greek_prefix(n % 100)),
            400..=999 => format!(
                "{}{}{}",
                Self::UNITS[n / 100],
                Self::COSI,
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

pub fn parentheses(str: String, paren: bool) -> String {
    if paren {
        format!("({})", str)
    } else {
        str
    }
}

/// Whether an adjective *usually* goes before or after a word in the language.
pub enum Position {
    Before,
    After,
}

/// The trait shared by all languages. Defaults to English.
pub trait Language: Prefix {
    /// The usual position for adjectives.
    fn adj_pos() -> Position {
        Position::Before
    }

    /// Parses the [`Name`] in the specified language, with the given [`Options`].
    fn parse<T: NameType>(name: &Name<T>, options: Options) -> String {
        debug_assert!(name.is_valid(), "Invalid name {:?}.", name);

        match name {
            // Basic shapes
            Name::Nullitope => Self::nullitope(options),
            Name::Point => Self::point(options),
            Name::Dyad => Self::dyad(options),

            // 2D shapes
            Name::Triangle { regular: _ } => Self::triangle(options),
            Name::Square => Self::square(options),
            Name::Rectangle => Self::rectangle(options),
            Name::Orthodiagonal => Self::generic(4, Rank::new(2), options),
            Name::Polygon { regular: _, n } => Self::generic(*n, Rank::new(2), options),

            // Regular families
            Name::Simplex { regular: _, rank } => Self::simplex(*rank, options),
            Name::Hyperblock { regular, rank } => {
                if regular.satisfies(|r| r.is_yes()) {
                    Self::hypercube(*rank, options)
                } else {
                    Self::hyperblock(*rank, options)
                }
            }
            Name::Orthoplex { regular: _, rank } => Self::orthoplex(*rank, options),

            // Modifiers
            Name::Pyramid(base) => Self::pyramid_of(base, options),
            Name::Prism(base) => Self::prism_of(base, options),
            Name::Tegum(base) => Self::tegum_of(base, options),
            Name::Antiprism {
                base,
                facet_count: _,
            } => Self::antiprism_of(base, options),
            Name::Antitegum {
                base,
                vertex_count: _,
                center: _,
            } => Self::antitegum_of(base, options),
            Name::Petrial {
                base,
                facet_count: _,
            } => Self::petrial_of(base, options),

            // Multimodifiers
            Name::Multipyramid(_)
            | Name::Multiprism(_)
            | Name::Multitegum(_)
            | Name::Multicomb(_) => Self::multiproduct(name, options),

            // Single adjectives
            Name::Small(base) => Self::small_of(base, options),
            Name::Great(base) => Self::great_of(base, options),
            Name::Stellated(base) => Self::stellated_of(base, options),

            Name::Generic { facet_count, rank } => Self::generic(*facet_count, *rank, options),
            Name::Dual { base, center: _ } => Self::dual_of(base, options),
            Name::Compound(components) => Self::compound(components, options),
        }
    }

    fn parse_uppercase<T: NameType>(name: &Name<T>, options: Options) -> String {
        let result = Self::parse(name, options);
        let mut c = result.chars();

        match c.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        }
    }

    /// Returns the suffix for a d-polytope. Only needs to work up to d = 20, we
    /// won't offer support any higher than that.
    fn suffix(rank: Rank, options: Options) -> String {
        const SUFFIXES: [&str; 25] = [
            "mon", "tel", "gon", "hedr", "chor", "ter", "pet", "ex", "zett", "yott", "xenn", "dak",
            "hend", "dok", "tradak", "tedak", "pedak", "exdak", "zedak", "yodak", "nedak", "ik",
            "iken", "ikod", "iktr",
        ];

        format!(
            "{}{}",
            SUFFIXES[rank.usize()],
            if rank == Rank::new(2) {
                options.three("", "s", "al")
            } else if rank == Rank::new(3) {
                options.three("on", "a", "al")
            } else {
                options.three("on", "a", "ic")
            }
        )
    }

    /// The name of a nullitope.
    fn nullitope(options: Options) -> String {
        format!("nullitop{}", options.three("e", "es", "ic"))
    }

    /// The name of a point.
    fn point(options: Options) -> String {
        format!("point{}", options.two("", "s"))
    }

    /// The name of a dyad.
    fn dyad(options: Options) -> String {
        format!("dyad{}", options.three("", "s", "ic"))
    }

    /// The name of a triangle.
    fn triangle(options: Options) -> String {
        format!("triang{}", options.three("le", "les", "ular"))
    }

    /// The name of a square.
    fn square(options: Options) -> String {
        format!("square{}", options.two("", "s"))
    }

    /// The name of a rectangle.
    fn rectangle(options: Options) -> String {
        format!("rectang{}", options.three("le", "les", "ular"))
    }

    /// The generic name for a polytope with a given facet count and rank.
    fn generic(facet_count: usize, rank: Rank, options: Options) -> String {
        format!(
            "{}{}",
            Self::prefix(facet_count),
            Self::suffix(rank, options)
        )
    }

    fn combine(adj: String, noun: String) -> String {
        match Self::adj_pos() {
            Position::Before => format!("{} {}", adj, noun),
            Position::After => format!("{} {}", noun, adj),
        }
    }

    /// The name of a pyramid.
    fn pyramid(options: Options) -> String {
        format!("pyramid{}", options.three("", "s", "al"))
    }

    /// The name for a pyramid with a given base.
    fn pyramid_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(Self::base_adj(base, options), Self::pyramid(options))
    }

    /// The name for a prism.
    fn prism(options: Options) -> String {
        format!("prism{}", options.three("", "s", "atic"))
    }

    /// The name for a prism with a given base.
    fn prism_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(Self::base_adj(base, options), Self::prism(options))
    }

    /// The name for a tegum.
    fn tegum(options: Options) -> String {
        format!("teg{}", options.three("um", "ums", "matic"))
    }

    /// The name for a tegum with a given base.
    fn tegum_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(Self::base_adj(base, options), Self::tegum(options))
    }

    /// The name for an antiprism.
    fn antiprism(options: Options) -> String {
        format!("antiprism{}", options.three("", "s", "atic"))
    }

    /// The name for an antiprism with a given base.
    fn antiprism_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(Self::base_adj(base, options), Self::antiprism(options))
    }

    /// The name for an antitegum.
    fn antitegum(options: Options) -> String {
        format!("antiteg{}", options.three("um", "ums", "matic"))
    }

    /// The name for an antitegum with a given base.
    fn antitegum_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(Self::base_adj(base, options), Self::antitegum(options))
    }

    /// The name for a Petrial.
    fn petrial(options: Options) -> String {
        format!("Petrial{}", options.three("", "s", ""))
    }

    /// The name for a Petrial with a given base.
    fn petrial_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(
            Self::petrial(Options {
                adjective: true,
                ..options
            }),
            Self::parse(base, options),
        )
    }

    fn multiproduct<T: NameType>(name: &Name<T>, options: Options) -> String {
        // Gets the bases and the kind of multiproduct.
        let (bases, kind) = match name {
            Name::Multipyramid(bases) => (bases, Self::pyramid(options)),
            Name::Multiprism(bases) => (bases, Self::prism(options)),
            Name::Multitegum(bases) => (bases, Self::tegum(options)),
            Name::Multicomb(bases) => (bases, format!("comb{}", options.two("", "s"))),
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

        let (last, bases) = bases.split_last().unwrap();
        for base in bases {
            str_bases.push_str(&Self::base_adj(base, options));
            str_bases.push('-');
        }
        str_bases.push_str(&Self::base_adj(last, options));

        format!("{} {}", str_bases, kind)
    }

    /// The name for a simplex with a given rank.
    fn simplex(rank: Rank, options: Options) -> String {
        Self::generic(rank.plus_one_usize(), rank, options)
    }

    /// The name for a hyperblock with a given rank.
    fn hyperblock(rank: Rank, options: Options) -> String {
        match rank.usize() {
            3 => format!("cuboid{}", options.three("", "s", "al")),
            n => {
                format!("{}block{}", Self::prefix(n), options.two("", "s"))
            }
        }
    }

    /// The name for a hypercube with a given rank.
    fn hypercube(rank: Rank, options: Options) -> String {
        match rank.usize() {
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

    /// The name for an orthoplex with a given rank.
    fn orthoplex(rank: Rank, options: Options) -> String {
        Self::generic(2u32.pow(rank.u32()) as usize, rank, options)
    }

    fn great(_options: Options) -> String {
        String::from("great")
    }

    fn great_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(Self::great(options), Self::base(base, options))
    }

    fn small(_options: Options) -> String {
        String::from("small")
    }

    fn small_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(Self::small(options), Self::base(base, options))
    }

    fn stellated(_options: Options) -> String {
        String::from("stellated")
    }

    fn stellated_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(Self::stellated(options), Self::base(base, options))
    }

    /// The name for the dual of another polytope.
    fn dual(_options: Options) -> String {
        String::from("dual")
    }

    /// The name for the dual of another polytope.
    fn dual_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        Self::combine(Self::dual(options), Self::base(base, options))
    }

    fn compound<T: NameType>(components: &[(usize, Name<T>)], options: Options) -> String {
        let ((last_rep, last_component), first_components) = components.split_last().unwrap();
        let mut str = String::new();

        let parse_component = |rep, component| {
            Self::parse(
                component,
                Options {
                    count: rep,
                    ..Default::default()
                },
            )
        };

        let comma = if components.len() == 2 { "" } else { "," };
        for (rep, component) in first_components {
            str.push_str(&format!(
                "{} {}{} ",
                rep,
                parse_component(*rep, component),
                comma
            ));
        }

        str.push_str(&format!(
            "and {} {} compound{}",
            last_rep,
            parse_component(*last_rep, last_component),
            options.two("", "s")
        ));

        str
    }

    /// The name of a base of some other construction, parenthesized if
    /// necessary.
    fn base<T: NameType>(base: &Name<T>, options: Options) -> String {
        parentheses(Self::parse(base, options), options.parentheses)
    }

    /// The name of a base of some other construction, as an adjective,
    /// parenthesized if necessary.
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
}

/// We should maybe make `dyn Language` work eventually.
#[derive(Clone, Copy, Debug, EnumIter)]
pub enum SelectedLanguage {
    En,
    Es,
    Fr,
    Ja,
    Pii,
}

impl SelectedLanguage {
    pub fn parse<T: NameType>(&self, name: &Name<T>, options: Options) -> String {
        match self {
            Self::En => En::parse(name, options),
            Self::Es => Es::parse(name, options),
            Self::Fr => Fr::parse(name, options),
            Self::Ja => Ja::parse(name, options),
            Self::Pii => Pii::parse(name, options),
        }
    }

    pub fn parse_uppercase<T: NameType>(&self, name: &Name<T>, options: Options) -> String {
        match self {
            Self::En => En::parse_uppercase(name, options),
            Self::Es => Es::parse_uppercase(name, options),
            Self::Fr => Fr::parse_uppercase(name, options),
            Self::Ja => Ja::parse(name, options),
            Self::Pii => Pii::parse_uppercase(name, options),
        }
    }

    pub fn to_string(&self) -> &str {
        match self {
            Self::En => "English",
            Self::Es => "Spanish",
            Self::Fr => "French",
            Self::Ja => "Japanese",
            Self::Pii => "Proto Indo-Iranian",
        }
    }
}

impl Default for SelectedLanguage {
    fn default() -> Self {
        Self::En
    }
}
