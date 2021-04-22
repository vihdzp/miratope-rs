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
pub mod name;

pub use lang::Language;
pub use name::Name;

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
