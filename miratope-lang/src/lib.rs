//! A module dedicated to parsing the names of polytopes into different
//! languages.
//!
//! A great part of the terms we use to describe polytopes are recently coined
//! neologisms and words that haven't entered the wider mathematical sphere.
//! Furthermore, there are some rather large families of words (like those for
//! polygons) that must be algorithmically translated into the target language.
//! This makes translating Miratope much harder than translating most other
//! software would be.
//!
//! In what follows, we've left extensive documentation, in the hope that it
//! makes the work of anyone trying to translate Miratope much easier.
//!
//! # How does translation work?
//! Every polytope in Miratope is stored alongside its [`Name`]. Names can be
//! thought of as nodes in a tree, which represents how the polytope has been
//! built up. For instance, an (abstract) pentagonal-cubic duoprism would have a
//! name like this:
//!
//! ```
//! # use miratope_core::abs::rank::Rank;
//! # use miratope_lang::{lang::En, Language, name::{Abs, Name}};
//! let pecube: Name<Abs> = Name::multiprism(vec![
//!     Name::polygon(Default::default(), 5), // 5-gon
//!     Name::hyperblock(Default::default(), Rank::new(3)) // 3-hypercube
//! ]);
//! # assert_eq!(En::parse(&pecube), "pentagonal-cubic duoprism");
//! ```
//!
//! For more information, see the [`Name`] struct's documentation.
//!
//! The [`parse`](Language::parse) function takes in this name, and uses the
//! corresponding methods to parse and combine each of its parts:
//!
//! ```
//! # use miratope_core::abs::rank::Rank;
//! # use miratope_lang::{lang::En, Language, name::{Abs, Name}};
//! # let pecube: Name<Abs> = Name::multiprism(vec![
//! #     Name::polygon(Default::default(), 5), // abstract 5-gon
//! #     Name::hyperblock(Default::default(), Rank::new(3)) // abstract 3-hypercube
//! # ]);
//! assert_eq!(En::parse(&pecube), "pentagonal-cubic duoprism");
//! ```
//!
//! # What do I need to code?
//! You'll first need to implement the [`Prefix`] trait for your language. This
//! will specify how numerical prefixes will be translated. If your language
//! uses Greek-like prefixes (as in **penta**-gon or **hexe**-ract), you'll want
//! to implement this trait via [`GreekPrefix`].
//!
//! You'll then need to fully implement the [`Language`] trait for your
//! language. This trait will by default assume the default [`Position`] and
//! [`Gender`] of all adjectives, for convenience with languages such as English
//! (which has no grammatical gender, and always put adjectives before nouns).
//! However, these can be overriden by manually implementing the methods ending
//! in `_gender` and `_pos`.

pub mod lang;
pub mod name;
pub mod options;
pub mod poly;

use crate::options::Options;
use name::{Name, NameData, NameType, Regular};

use miratope_core::abs::rank::Rank;
use options::{Count, Gender};
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

/// The link to the [Polytope Wiki](https://polytope.miraheze.org/wiki/).
pub const WIKI_LINK: &str = "https://polytope.miraheze.org/wiki/";

/// Trait that allows one to build a prefix from any natural number. Every
/// [`Language`] must implement this trait. If the language implements a
/// Greek-like system for prefixes (e.g. "penta", "hexa"), you should implement
/// this trait via [`GreekPrefix`] instead.
///
/// In some instances, we require small variations on the main prefix system.
/// These fallback to the main prefix system by default, but can be overridden
/// if needed.
///
/// Defaults to just using `n-` as prefixes.
pub trait Prefix {
    /// Returns the prefix that stands for n-.
    fn prefix(n: usize) -> String {
        format!("{}-", n)
    }

    /// Returns the prefix that stands for n- in a polygon, whenever we aren't
    /// writing an adjective.
    fn polygon_prefix(n: usize) -> String {
        Self::prefix(n)
    }

    /// Returns the prefix that stands for n- in a multiproduct.
    fn multi_prefix(n: usize) -> String {
        Self::prefix(n)
    }

    /// Returns the prefix that stands for n- in an n-eract (hypercube).
    fn hypercube_prefix(n: usize) -> String {
        Self::prefix(n)
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
            // Units.
            0..=9 => Self::UNITS[n].to_owned(),

            // Two digit numbers.
            11 => Self::HENDECA.to_owned(),
            12 => Self::DODECA.to_owned(),
            10 | 13..=19 => Self::UNITS[n % 10].to_owned() + Self::DECA,
            20 => Self::ICOSA.to_owned(),
            21..=29 => Self::ICOSI.to_owned() + Self::UNITS[n % 10],
            30..=39 => Self::TRIACONTA.to_owned() + Self::UNITS[n % 10],
            40..=99 => Self::UNITS[n / 10].to_owned() + Self::CONTA + Self::UNITS[n % 10],

            // Three digit numbers.
            100 => Self::HECTO.to_owned(),
            101..=199 => Self::HECATON.to_owned() + &Self::greek_prefix(n % 100),
            200..=299 => Self::DIACOSI.to_owned() + &Self::greek_prefix(n % 100),
            300..=399 => Self::TRIACOSI.to_owned() + &Self::greek_prefix(n % 100),
            400..=999 => {
                Self::UNITS[n / 100].to_owned() + Self::COSI + &Self::greek_prefix(n % 100)
            }

            // Four digit numbers.
            1000..=1999 => Self::CHILIA.to_owned() + &Self::greek_prefix(n % 1000),
            2000..=2999 => Self::DISCHILIA.to_owned() + &Self::greek_prefix(n % 1000),
            3000..=3999 => Self::TRISCHILIA.to_owned() + &Self::greek_prefix(n % 1000),
            4000..=9999 => {
                Self::UNITS[n / 1000].to_owned() + Self::CHILIA + &Self::greek_prefix(n % 1000)
            }

            // Five digit numbers.
            10000..=19999 => Self::MYRIA.to_owned() + &Self::greek_prefix(n % 10000),
            20000..=29999 => Self::DISMYRIA.to_owned() + &Self::greek_prefix(n % 10000),
            30000..=39999 => Self::TRISMYRIA.to_owned() + &Self::greek_prefix(n % 10000),
            40000..=99999 => {
                Self::UNITS[n / 10000].to_owned() + Self::MYRIA + &Self::greek_prefix(n % 10000)
            }

            // We default to n-.
            _ => format!("{}-", n),
        }
    }
}

/// The position at which an adjective will go with respect to a noun.
#[derive(Clone, Copy, Debug)]
pub enum Position {
    /// The adjective goes before the noun.
    Before,

    /// The adjective goes after the noun.
    After,
}

impl std::ops::Not for Position {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::Before => Self::After,
            Self::After => Self::Before,
        }
    }
}

/// The trait shared by all languages.
///
/// We strived to make this trait as general as possible while still making code
/// reuse possible. However, there may be an implicit bias towards European
/// languages due to the profile of the people who worked on this. Any
/// suggestions towards making this more general are welcome.
pub trait Language: Prefix {
    /// Whichever grammatical number system the language uses.
    type Count: Count;

    /// Whichever grammatical gender system the language uses.
    type Gender: Gender;

    /// Parses the [`Name`] in the specified language.
    fn parse<T: NameType>(name: &Name<T>) -> String {
        Self::parse_with(name, Default::default())
    }

    /// Parses the [`Name`] in the specified language, with the given [`Options`].
    fn parse_with<T: NameType>(
        name: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        debug_assert!(name.is_valid(), "Invalid name {:?}.", name);

        match name {
            // Basic shapes
            Name::Nullitope => Self::nullitope(options).to_owned(),
            Name::Point => Self::point(options).to_owned(),
            Name::Dyad => Self::dyad(options).to_owned(),

            // 2D shapes
            Name::Triangle { .. } => Self::triangle(options).to_owned(),
            Name::Square => Self::square(options).to_owned(),
            Name::Rectangle => Self::rectangle(options).to_owned(),
            Name::Orthodiagonal => Self::generic(4, Rank::new(2), options),
            Name::Polygon { n, .. } => Self::generic(*n, Rank::new(2), options),

            // Regular families
            Name::Simplex { rank, .. } => Self::simplex(*rank, options),
            Name::Cuboid { regular } => {
                if regular.satisfies(Regular::is_yes) {
                    Self::cube(options).to_owned()
                } else {
                    Self::cuboid(options).to_owned()
                }
            }
            Name::Hyperblock { regular, rank } => {
                if regular.satisfies(Regular::is_yes) {
                    Self::hypercube(*rank, options)
                } else {
                    Self::hyperblock(*rank, options)
                }
            }
            Name::Orthoplex { rank, .. } => Self::orthoplex(*rank, options),

            // Modifiers
            Name::Pyramid(base) => Self::pyramid_of(base, options),
            Name::Prism(base) => Self::prism_of(base, options),
            Name::Tegum(base) => Self::tegum_of(base, options),
            Name::Antiprism { base, .. } => Self::antiprism_of(base, options),
            Name::Antitegum { base, .. } => Self::antitegum_of(base, options),
            Name::Ditope { base, rank } => Self::ditope_of(base, *rank, options),
            Name::Hosotope { base, rank } => Self::hosotope_of(base, *rank, options),
            Name::Petrial { base, .. } => Self::petrial_of(base, options),

            // Multimodifiers
            Name::Multipyramid(_)
            | Name::Multiprism(_)
            | Name::Multitegum(_)
            | Name::Multicomb(_) => Self::multiproduct(name, options),

            // Single adjectives
            Name::Small(base) => Self::small_of(base, options),
            Name::Great(base) => Self::great_of(base, options),
            Name::Stellated(base) => Self::stellated_of(base, options),

            &Name::Generic { facet_count, rank } => Self::generic(facet_count, rank, options),
            Name::Dual { base, .. } => Self::dual_of(base, options),
        }
    }

    /// Parses the [`Name`] in the specified language. If the first character is
    /// ASCII, it makes it uppercase.
    fn parse_uppercase<T: NameType>(name: &Name<T>) -> String {
        let mut result = Self::parse(name);
        lang::uppercase_mut(&mut result);
        result
    }

    /// The default position to place adjectives. This will be used for the
    /// default implementations, but it can be overridden in any specific case.
    fn default_pos() -> Position;

    /// Combines an adjective and a noun, placing the adjective in a given
    /// [`Position`] with respect to the noun.
    fn combine(adj: &str, noun: &str, pos: Position) -> String {
        match pos {
            Position::Before => format!("{} {}", adj, noun),
            Position::After => format!("{} {}", noun, adj),
        }
    }

    /// Converts a name into an adjective. The options passed are those for the
    /// term this adjective is modifying, which might either be a noun or
    /// another adjective.
    ///
    /// If the options don't specify an adjective, the specified gender is used.
    /// Otherwise, the adjective inherits the gender from the options.
    fn to_adj<T: NameType>(
        base: &Name<T>,
        mut options: Options<Self::Count, Self::Gender>,
        gender: Self::Gender,
    ) -> String {
        let adj = options.adjective;
        options.adjective = true;

        // This is a noun, so we use the specified gender.
        if !adj {
            options.gender = gender;
        }

        Self::parse_with(base, options)
    }

    /// Returns the suffix for a d-polytope. Only needs to work up to d = 20, we
    /// won't offer support any higher than that.
    fn suffix(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String;

    /// The generic name for a polytope with a given facet count and rank.
    fn generic(
        facet_count: usize,
        rank: Rank,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::prefix(facet_count) + &Self::suffix(rank, options)
    }

    /// The name of a nullitope.
    fn nullitope(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The name of a point.
    fn point(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The name of a dyad.
    fn dyad(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The name of a triangle.
    fn triangle(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The name of a square.
    fn square(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The name of a rectangle.
    fn rectangle(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The name of a pyramid.
    fn pyramid(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position at which the "pyramid" adjective goes. We assume this is
    /// shared by "multipyramid".
    fn pyramid_pos() -> Position {
        Self::default_pos()
    }

    /// The gender of the "pyramid" noun. We assume this is shared by
    /// "multipyramid".
    fn pyramid_gender() -> Self::Gender {
        Default::default()
    }

    /// The name for a pyramid with a given base.
    fn pyramid_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            &Self::to_adj(base, options, Self::pyramid_gender()),
            Self::pyramid(options),
            Self::pyramid_pos(),
        )
    }

    /// The name for a prism.
    fn prism(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position at which the "prism" adjective goes. We assume this is
    /// shared by "multiprism".
    fn prism_pos() -> Position {
        Self::default_pos()
    }

    /// The gender of the "prism" noun. We assume this is shared by
    /// "multiprism".
    fn prism_gender() -> Self::Gender {
        Default::default()
    }

    /// The name for a prism with a given base.
    fn prism_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            &Self::to_adj(base, options, Self::prism_gender()),
            &Self::prism(options),
            Self::prism_pos(),
        )
    }

    /// The name for a tegum.
    fn tegum(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position at which the "tegum" adjective goes. We assume this is
    /// shared by "multitegum".
    fn tegum_pos() -> Position {
        Self::default_pos()
    }

    /// The gender of the "tegum" noun. We assume this is shared by
    /// "multitegum".
    fn tegum_gender() -> Self::Gender {
        Default::default()
    }

    /// The name for a tegum with a given base.
    fn tegum_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            &Self::to_adj(base, options, Self::tegum_gender()),
            Self::tegum(options),
            Self::tegum_pos(),
        )
    }

    /// The name for a comb.
    fn comb(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position at which the "multicomb" adjective goes.
    fn comb_pos() -> Position {
        Self::default_pos()
    }

    /// The gender of the "multicomb" noun.
    fn comb_gender() -> Self::Gender {
        Default::default()
    }

    // A comb can't be used as a standalone. Instead, it must be used as part of
    // the word "multicomb."

    /// Makes the name for a general multiproduct
    fn multiproduct<T: NameType>(
        name: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        // Gets the bases and the kind of multiproduct.
        let (bases, kind, gender, pos) = match name {
            Name::Multipyramid(bases) => (
                bases,
                Self::pyramid(options),
                Self::pyramid_gender(),
                Self::pyramid_pos(),
            ),
            Name::Multiprism(bases) => (
                bases,
                Self::prism(options),
                Self::prism_gender(),
                Self::prism_pos(),
            ),
            Name::Multitegum(bases) => (
                bases,
                Self::tegum(options),
                Self::tegum_gender(),
                Self::tegum_pos(),
            ),
            Name::Multicomb(bases) => (
                bases,
                Self::comb(options),
                Self::comb_gender(),
                Self::comb_pos(),
            ),

            // This method shouldn't be called in any other case.
            _ => unreachable!(),
        };

        // Prepends a multi_prefix.
        let kind = Self::multi_prefix(bases.len()) + kind;

        // Concatenates the bases as adjectives, adding hyphens between them.
        let mut str_bases = String::new();
        let (last, bases) = bases.split_last().unwrap();
        for base in bases {
            str_bases.push_str(&Self::to_adj(base, options, gender));
            str_bases.push('-');
        }
        str_bases.push_str(&Self::to_adj(last, options, gender));

        Self::combine(&str_bases, &kind, pos)
    }

    /// The name for a simplex with a given rank.
    fn simplex(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        Self::generic(rank.plus_one_usize(), rank, options)
    }

    /// The name for a cuboid.
    fn cuboid(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The name for a cube.
    fn cube(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The name for a hyperblock with a given rank.
    fn hyperblock(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String;

    /// The name for a hypercube with a given rank.
    fn hypercube(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String;

    /// The name for an orthoplex with a given rank.
    fn orthoplex(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        Self::generic(1 << rank.into_usize(), rank, options)
    }

    /// The adjective for a "dual" polytope.
    fn dual(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position at which the "dual" adjective goes.
    fn dual_pos() -> Position {
        Self::default_pos()
    }

    /// The name for the dual of another polytope.
    fn dual_of<T: NameType>(base: &Name<T>, options: Options<Self::Count, Self::Gender>) -> String {
        Self::combine(
            &Self::dual(options),
            &Self::parse_with(base, options),
            Self::dual_pos(),
        )
    }

    /// The name for an antiprism.
    fn antiprism(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position at which the "antiprism" adjective goes.
    fn antiprism_pos() -> Position {
        Self::prism_pos()
    }

    /// The gender of the "antiprism" noun.
    fn antiprism_gender() -> Self::Gender {
        Self::prism_gender()
    }

    /// The name for an antiprism with a given base.
    fn antiprism_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            &Self::to_adj(base, options, Self::antiprism_gender()),
            &Self::antiprism(options),
            Self::antiprism_pos(),
        )
    }

    /// The name for an antitegum.
    fn antitegum(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position at which the "antitegum" adjective goes.
    fn antitegum_pos() -> Position {
        Self::tegum_pos()
    }

    /// The gender of the "antitegum" noun.
    fn antitegum_gender() -> Self::Gender {
        Self::tegum_gender()
    }

    /// The name for an antitegum with a given base.
    fn antitegum_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            &Self::to_adj(base, options, Self::antitegum_gender()),
            &Self::antitegum(options),
            Self::antitegum_pos(),
        )
    }

    /// The name for a ditope.
    fn ditope(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        Self::generic(2, rank, options)
    }

    /// The position at which the "ditope" noun goes.
    fn ditope_pos() -> Position {
        !Self::default_pos()
    }

    /// The gender of the "ditope" noun.
    fn ditope_gender(_rank: Rank) -> Self::Gender {
        Default::default()
    }

    /// The name for a ditope with a given base.
    fn ditope_of<T: NameType>(
        base: &Name<T>,
        rank: Rank,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            &Self::to_adj(base, options, Self::ditope_gender(rank)),
            &Self::ditope(rank, options),
            !Self::ditope_pos(),
        )
    }

    /// The name for a hosotope.
    fn hosotope(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String;

    /// The position at which the "hosotope" noun goes.
    fn hosotope_pos() -> Position {
        !Self::default_pos()
    }

    /// The gender of the "hosotope" noun.
    fn hosotope_gender(_rank: Rank) -> Self::Gender {
        Default::default()
    }

    /// The name for a hosotope with a given base.
    fn hosotope_of<T: NameType>(
        base: &Name<T>,
        rank: Rank,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            &Self::to_adj(base, options, Self::hosotope_gender(rank)),
            &Self::hosotope(rank, options),
            !Self::hosotope_pos(),
        )
    }

    /// The adjective for a Petrial.
    fn petrial(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position at which the "Petrial" adjective goes.
    fn petrial_pos() -> Position {
        Self::default_pos()
    }

    /// The name for a Petrial with a given base.
    fn petrial_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            Self::petrial(options),
            &Self::parse_with(base, options),
            Self::petrial_pos(),
        )
    }

    /// The adjective for a "great" version of a polytope.
    fn great(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position of the "great" adjective.
    fn great_pos() -> Position {
        Self::default_pos()
    }

    /// The name for a great polytope with a given base.
    fn great_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            Self::great(options),
            &Self::parse_with(base, options),
            Self::great_pos(),
        )
    }

    /// The adjective for a "small" version of a polytope.
    fn small(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position of the "small" adjective.
    fn small_pos() -> Position {
        Self::default_pos()
    }

    /// The name for a small polytope with a given base.
    fn small_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            Self::small(options),
            &Self::parse_with(base, options),
            Self::small_pos(),
        )
    }

    /// The adjective for a "stellated" version of a polytope.
    fn stellated(options: Options<Self::Count, Self::Gender>) -> &'static str;

    /// The position of the "small" adjective.
    fn stellated_pos() -> Position {
        Self::default_pos()
    }

    /// The name for a stellated polytope from a given base.
    fn stellated_of<T: NameType>(
        base: &Name<T>,
        options: Options<Self::Count, Self::Gender>,
    ) -> String {
        Self::combine(
            &Self::stellated(options),
            &Self::parse_with(base, options),
            Self::stellated_pos(),
        )
    }
}

/// We should maybe make `dyn Language` work eventually.
#[derive(Clone, Copy, Debug, EnumIter, Serialize, Deserialize)]
pub enum SelectedLanguage {
    /// English
    En,

    /// Spanish
    Es,

    /// German
    De,
    // French
    // Fr,

    // Japanese
    // Ja,

    // Proto Indo-Iranian
    // Pii,
}

impl std::fmt::Display for SelectedLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::En => "English",
            Self::Es => "Spanish",
            Self::De => "German",
            // Self::Fr => "French",
            // Self::Ja => "Japanese",
            // Self::Pii => "Proto Indo-Iranian",
        })
    }
}

impl SelectedLanguage {
    pub fn parse<T: NameType>(&self, name: &Name<T>) -> String {
        use crate::lang::*;

        match self {
            Self::En => En::parse_uppercase(name),
            Self::Es => Es::parse_uppercase(name),
            Self::De => De::parse_uppercase(name),
            // Self::Fr => Fr::parse_uppercase(name),
            // Self::Ja => Ja::parse(name),
            // Self::Pii => Pii::parse_uppercase(name),
        }
    }
}

impl Default for SelectedLanguage {
    fn default() -> Self {
        Self::En
    }
}
