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

pub mod gender;
pub mod lang;
pub mod name;
pub mod poly;

use name::{Name, NameData, NameType, Quadrilateral, Regular};

use gender::Gender;
use miratope_core::abs::rank::Rank;
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

#[macro_use]
extern crate doc_comment;

use paste::paste;

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

#[macro_export]
/// Allows for easier implementation of the [`GreekPrefix`] trait.
macro_rules! greek_prefixes {
    ($(#[$um:meta])* UNITS = $u:expr; $($(#[$idm:meta])* $id:ident = $e:expr;)*) => {
        $(#[$um])*
        const UNITS: [&'static str; 10] = $u;
        $(
            $(#[$idm])*
            const $id: &'static str = $e;
        )*
    };
}

/// Trait shared by languages that allow for greek prefixes or anything similar.
/// Every `struct` implementing this trait automatically implements [`Prefix`]
/// as well.
///
/// Defaults to the English ["Wikipedian system."](https://polytope.miraheze.org/wiki/Nomenclature#Wikipedian_system)
pub trait GreekPrefix {
    greek_prefixes! {
        /// The prefixes for a single digit number.
        UNITS = [
            "", "hena", "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "ennea",
        ];

        /// Represents the number 10 for numbers between 10 and 19.
        DECA = "deca";

        /// Represents a factor of 10.
        CONTA = "conta";

        /// The prefix for 11.
        HENDECA = "hendeca";

        /// The prefix for 12.
        DODECA = "dodeca";

        /// The prefix for 20.
        ICOSA = "icosa";

        /// Represents the number 20 for numbers between 21 and 29.
        ICOSI = "icosi";

        /// The prefix for 30.
        TRIACONTA = "triaconta";

        /// The prefix for 100.
        HECTO = "hecto";

        /// Represents the number 100 for numbers between 101 and 199.
        HECATON = "hecaton";

        /// Represents a factor of 100.
        COSA = "cosa";

        /// The prefix for 200.
        DIACOSI = "diacosi";

        /// The prefix for 300.
        TRIACOSI = "triacosi";

        /// Represents the number 100 for numbers between 400 and 999.
        COSI = "cosi";

        /// The prefix for 1000.
        CHILIA = "chilia";

        /// The prefix for 2000.
        DISCHILIA = "dischilia";

        /// The prefix for 3000.
        TRISCHILIA = "trischilia";

        /// The prefix for 10000.
        MYRIA = "myria";

        /// The prefix for 20000.
        DISMYRIA = "dismyria";

        /// The prefix for 30000.
        TRISMYRIA = "trismyria";
    }

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

impl Position {
    /// Combines an adjective and a noun, placing the adjective in a given
    /// position with respect to the noun.
    fn combine(self, mut adj: String, mut noun: String) -> String {
        match self {
            Self::Before => {
                adj.push(' ');
                adj += &noun;
                adj
            }
            Self::After => {
                noun.push(' ');
                noun += &adj;
                noun
            }
        }
    }
}

/// The output of parsing a noun.
pub struct ParseOutput<G: Gender> {
    /// The grammatical gender of the output.
    gender: G,

    /// The parsed noun.
    output: String,
}

impl<G: Gender> ParseOutput<G> {
    /// Initializes a new `ParseOutput` from its fields.
    pub fn new(gender: G, output: String) -> Self {
        Self { gender, output }
    }

    /// Applies a function to the output of `self` to create a `ParseOutput`
    /// with the same gender.
    pub fn map_output<F: FnOnce(G, String) -> String>(self, f: F) -> Self {
        Self::new(self.gender, f(self.gender, self.output))
    }
}

/// Declares the necessary trait methods for parsing a terminal noun name. If
/// you want to declare the trait methods for a non-terminal noun name, see
/// [`impl_operator`].
///
/// # Example
///
/// ```
/// impl_noun!(
///     nullitope,
///     polygon(n: u32),
///     simplex(rank: Rank)
/// );
/// ```
macro_rules! decl_noun {
    ($($name:ident $(($($args:ident: $ty:ty),*))?),*) => {
        $(
            paste! {
                doc_comment! {
                    concat!("The string corresponding to the noun for \"", stringify!($name), "\"."),
                    #[allow(unused_variables)]
                    fn [<$name _noun_str>]($($($args: $ty),*)?) -> String;
                }

                doc_comment! {
                    concat!("The grammatical gender corresponding to the noun \"", stringify!($name), "\"."),
                    #[allow(unused_variables)]
                    fn [<$name _gender>]($($($args: $ty),*)?) -> Self::Gender {
                        Default::default()
                    }
                }

                doc_comment! {
                    concat!("Parses the name corresponding to the noun for \"", stringify!($name), "\"."),
                    #[allow(unused_variables)]
                    fn [<$name _noun>]($($($args: $ty),*)?) -> ParseOutput<Self::Gender> {
                        ParseOutput::new(
                            Self::[<$name _gender>]($($($args),*)?),
                            Self::[<$name _noun_str>]($($($args),*)?)
                        )
                    }
                }

                doc_comment! {
                    concat!("Parses the name corresponding to the adjective for \"", stringify!($name), "\"."),
                    #[allow(unused_variables)]
                    fn [<$name _adj>](gender: Self::Gender, $($($args: $ty),*)?) -> String;
                }
            }
        )*
    };
}

/// Declares the necessary trait methods for parsing a non-terminal noun name.
/// This will automatically call [`impl_name`] to implement the base methods.
/// If your name is also a multiproduct, use [`impl_multiproduct`] instead.
///
/// # Example
///
/// ```
/// impl_operator!(
///     antiprism,
///     antitegum,
///     ditope(rank: Rank)
/// );
/// ```
macro_rules! decl_operator {
    ($($name:ident $(($($args:ident: $ty:ty),*))?),*) => {
        $(
            decl_noun!($name $(($($args: $ty),*))?);

            paste! {
                doc_comment! {
                    concat!("The position corresponding to the adjective for \"", stringify!($name), "\"."),
                    #[allow(unused_variables)]
                    fn [<$name _pos>]($($($args: $ty),*)?) -> Position {
                        Self::default_pos()
                    }
                }

                doc_comment! {
                    concat!("Parses the ", stringify!($name), " of a name as a noun."),
                    #[allow(unused_variables)]
                    fn [<$name _of_noun>]<T: NameType>(base: &Name<T>, $($($args: $ty),*)?) -> ParseOutput<Self::Gender> {
                        Self::[<$name _noun>]($($($args),*)?).map_output(|gender, output|
                            Self::[<$name _pos>]($($($args),*)?).combine(
                                Self::parse_adj(base, gender),
                                output,
                            )
                        )
                    }
                }

                doc_comment! {
                    concat!("Parses the ", stringify!($name), " of a name as an adjective."),
                    #[allow(unused_variables)]
                    fn [<$name _of_adj>]<T: NameType>(gender: Self::Gender, base: &Name<T>, $($($args: $ty),*)?) -> String {
                        Self::[<$name _pos>]($($($args),*)?).combine(
                            Self::parse_adj(base, gender),
                            Self::[<$name _adj>](gender, $($($args),*)?)
                        )
                    }
                }
            }
        )*
    };
}

/// Declares the necessary trait methods for parsing a multiprismatic name. This
/// will automatically call [`impl_operator`] to implement the base methods.
///
/// # Example
///
/// ```
/// impl_multiprism!(
///     pyramid,
///     prism,
///     tegum
/// );
/// ```
macro_rules! decl_multiproduct {
    ($($name:ident),*) => {
        $(
            decl_operator!($name);

            paste! {
                doc_comment! {
                    concat!("Makes the name for a general ", stringify!($name), " product as a noun."),
                    fn [<$name _product_noun>]<T: NameType>(bases: &[Name<T>]) -> ParseOutput<Self::Gender> {
                        // The kind of the product.
                        let kind = Self::[<$name _noun>]();

                        kind.map_output(|gender, output|
                            Self::[<$name _pos>]().combine(
                                Self::hyphenate(bases.iter().map(|base| Self::parse_adj(base, gender))),
                                (Self::multi_prefix(bases.len()) + &output)
                            )
                        )
                    }
                }

                doc_comment! {
                    concat!("Makes the name for a general ", stringify!($name), " product as an adjective."),
                    fn [<$name _product_adj>]<T: NameType>(gender: Self::Gender, bases: &[Name<T>]) -> String {
                        Self::[<$name _pos>]().combine(
                            Self::hyphenate(bases.iter().map(|base| Self::parse_adj(base, gender))),
                            (Self::multi_prefix(bases.len()) + &Self::[<$name _adj>](gender))
                        )
                    }
                }
            }
        )*
    }
}

/// Declares the necessary trait methods for parsing an adjetive name.
///
/// # Example
///
/// ```
/// impl_adj!(
///     small,
///     great,
///     stellated
/// );
/// ```
macro_rules! decl_adj {
    ($($name:ident $(($($args:ident: $ty:ty),*))?),*) => {
        $(
            paste! {
                doc_comment! {
                    concat!("The position corresponding to the adjective \"", stringify!($name), "\"."),
                    fn [<$name _pos>]($($($args: $ty),*)?) -> Position {
                        Self::default_pos()
                    }
                }

                doc_comment! {
                    concat!("Parses the name corresponding to the adjective for \"", stringify!($name), "\"."),
                    fn [<$name _adj>](gender: Self::Gender, $($($args: $ty),*)?) -> String;
                }

                doc_comment! {
                    concat!("Parses a name with the modifier \"", stringify!($name), "\" as a noun."),
                    fn [<$name _of_noun>]<T: NameType>(base: &Name<T>, $($($args: $ty),*)?) -> ParseOutput<Self::Gender> {
                        Self::parse_noun(base).map_output(|gender, output|
                            Self::[<$name _pos>]().combine(
                                Self::[<$name _adj>](gender, $($($args),*)?),
                                output,
                            )
                        )
                    }
                }

                doc_comment! {
                    concat!("Parses a name with the modifier \"", stringify!($name), "\" as an adjective."),
                    fn [<$name _of_adj>]<T: NameType>(gender: Self::Gender, base: &Name<T>, $($($args: $ty),*)?) -> String {
                        Self::[<$name _pos>]($($($args),*)?).combine(
                            Self::[<$name _adj>](gender, $($($args),*)?),
                            Self::parse_adj(base, gender)
                        )
                    }
                }
            }
        )*
    };
}

/// An macro that helps implement the [`Language::parse_noun`] and the
/// [`Language::parse_adj`] methods.
macro_rules! impl_parse {
    ($type:ident $(,$gender:ident)? -> $ty:ty) => {
        paste! {
            /// Parses the [`Name`] in the specified language as a noun.
            fn [<parse_ $type>]<T: NameType>(name: &Name<T>, $($gender: Self::Gender)?) -> $ty {
                use Name::*;

                debug_assert!(name.is_valid(), "Invalid name {:?}.", name);

                match name {
                    // Basic shapes
                    Nullitope => Self::[<nullitope_ $type>]($($gender)?),
                    Point => Self::[<point_ $type>]($($gender)?),
                    Dyad => Self::[<dyad_ $type>]($($gender)?),

                    // 2D shapes
                    Triangle { .. } => Self::[<triangle_ $type>]($($gender)?),

                    // TODO: merge these into one.
                    Quadrilateral { quad } => Self::[<generic_quadrilateral_ $type>]::<T>($($gender,)? *quad),

                    Polygon { n, .. } => Self::[<generic_ $type>]($($gender,)? *n, Rank::new(2)),

                    // Regular families
                    Simplex { rank, .. } => Self::[<simplex_ $type>]($($gender,)? *rank),
                    Cuboid { regular } => Self::[<generic_cuboid_ $type>]::<T>($($gender,)? regular),
                    Hyperblock { regular, rank } => Self::[<generic_hyperblock_ $type>]::<T>($($gender,)? regular, *rank),
                    Orthoplex { rank, .. } => Self::[<orthoplex_ $type>]($($gender,)? *rank),

                    // Modifiers
                    Pyramid(base) => Self::[<pyramid_of_ $type>]($($gender,)? base),
                    Prism(base) => Self::[<prism_of_ $type>]($($gender,)? base),
                    Tegum(base) => Self::[<tegum_of_ $type>]($($gender,)? base),
                    Antiprism { base, .. } => Self::[<antiprism_of_ $type>]($($gender,)? base),
                    Antitegum { base, .. } => Self::[<antitegum_of_ $type>]($($gender,)? base),
                    Ditope { base, rank } => Self::[<ditope_of_ $type>]($($gender,)? base, *rank),
                    Hosotope { base, rank } => Self::[<hosotope_of_ $type>]($($gender,)? base, *rank),
                    Petrial { base, .. } => Self::[<petrial_of_ $type>]($($gender,)? base),

                    // Multimodifiers
                    Multipyramid(bases) => Self::[<pyramid_product_ $type>]($($gender,)? bases),
                    Multiprism(bases) => Self::[<prism_product_ $type>]($($gender,)? bases),
                    Multitegum(bases) => Self::[<tegum_product_ $type>]($($gender,)? bases),
                    Multicomb(bases) => Self::[<comb_product_ $type>]($($gender,)? bases),

                    // Single adjectives
                    Small(base) => Self::[<small_of_ $type>]($($gender,)? base),
                    Great(base) => Self::[<great_of_ $type>]($($gender,)? base),
                    Stellated(base) => Self::[<stellated_of_ $type>]($($gender,)? base),

                    &Generic { facet_count, rank } => Self::[<generic_ $type>]($($gender,)? facet_count, rank),
                    Dual { base, .. } => Self::[<dual_of_ $type>]($($gender,)? base),
                }
            }
        }
    };

    () => {
        impl_parse!(noun -> ParseOutput<Self::Gender>);
        impl_parse!(adj, gender -> String);
    };
}

/// The trait shared by all languages. Its one and only goal is to take a
/// [`Name`] and parse it into a `String` in the given language.
///
/// We strived to make this trait as general as possible while still making code
/// reuse possible. However, there may be an implicit bias towards European
/// languages due to the profile of the people who worked on this. Any
/// suggestions towards making this more general are welcome.
///
/// In what follows, we describe how parsing works in general.
///
/// # Nouns & Adjectives
///
/// Every name acts as either a noun or an adjective. That is, it must either
/// describe some standalone object, or it must modify such an object. Any noun
/// must have a corresponding adjective, but the opposite is not required of
/// adjectives.
///
/// Examples of nouns include nullitope, polygon, prism. Note the corresponding
/// adjectives nullitopic, polygonal, prismatic. Examples of adjectives include
/// great, stellated, truncated.
///
/// For translation to work, the target language will need to have categories of
/// words serving these broad functions.
///
/// # Word agreement
///
/// Many languages require the various words that describe a single object to
/// match up according to certain rules. This is known as **agreement**. To
/// account for this, every parsed name takes in and returns some [`ParseInfo`],
/// including grammatical count and gender.
///
/// A noun doesn't require any `ParseInfo`, but it must output it and propagate
/// it into its argument and any surrounding adjective (if any). On the other
/// hand, an adjective requires `ParseInfo`, but won't need to return any.
/// Because of this, methods that generate the nouns and adjectives for a given
/// name are separate.
///
/// # What needs to be implemented?
///
/// For convenience, many trait methods have default values. However, this
/// doesn't mean that these methods should always be left unimplemented. What
/// needs to be implemented will depend on the type of name and some
/// characteristics of the language.
///
/// For terminal nouns, you should implement
/// - `_str_noun`
/// - `_adj`
/// - `_gender`*
///
/// For non-terminal nouns, you should also implement
/// - `_pos`**
///
/// For adjectives, you should implement
/// - `_pos`**
/// - `_adj`
///
/// \* in gendered languages \*\* if the language does not place adjectives in a
/// unique position
///
/// If any other name does not behave as expected, the implementation may always
/// be overridden.
pub trait Language: Prefix {
    /// Whichever grammatical gender system the language uses.
    type Gender: Gender;

    /// The default position to place adjectives. This will be used for the
    /// default implementations, but it can be overridden in any specific case.
    fn default_pos() -> Position;

    decl_noun!(
        nullitope,
        point,
        dyad,
        triangle,
        square,
        rectangle,
        simplex(rank: Rank),
        cuboid,
        cube,
        hyperblock(rank: Rank),
        hypercube(rank: Rank),
        orthoplex(rank: Rank),
        suffix(rank: Rank)
    );

    decl_operator!(
        antiprism,
        antitegum,
        ditope(rank: Rank),
        hosotope(rank: Rank)
    );

    decl_multiproduct!(pyramid, prism, tegum, comb);

    decl_adj!(dual, petrial, great, small, stellated);

    /// Hyphenates various adjectives together.
    fn hyphenate<T: AsRef<str>, U: Iterator<Item = T>>(mut bases: U) -> String {
        let mut res = bases.next().unwrap().as_ref().to_owned();

        for name in bases {
            res.push('-');
            res += name.as_ref();
        }

        res
    }

    // Defaults for generic names.

    /// The string corresponding to the noun for a generic polytope.
    fn generic_noun_str(facet_count: usize, rank: Rank) -> String {
        Self::prefix(facet_count) + &Self::suffix_noun_str(rank)
    }

    /// The grammatical gender corresponding to the noun for a generic polytope.
    fn generic_gender(_: usize, rank: Rank) -> Self::Gender {
        Self::suffix_gender(rank)
    }

    /// The generic name for a polytope with a given facet count and rank as a
    /// noun.
    fn generic_noun(facet_count: usize, rank: Rank) -> ParseOutput<Self::Gender> {
        Self::suffix_noun(rank).map_output(|_, output| Self::prefix(facet_count) + &output)
    }

    /// The generic name for a polytope with a given facet count and rank as an
    /// adjective.
    fn generic_adj(gender: Self::Gender, facet_count: usize, rank: Rank) -> String {
        Self::prefix(facet_count) + &Self::suffix_adj(gender, rank)
    }

    // Some auxiliary functions for parsing.

    fn generic_quadrilateral_noun<T: NameType>(
        quad: T::DataQuadrilateral,
    ) -> ParseOutput<Self::Gender> {
        match quad.unwrap_or_default() {
            Quadrilateral::Square => Self::square_noun(),
            Quadrilateral::Rectangle => Self::rectangle_noun(),
            Quadrilateral::Orthodiagonal => Self::generic_noun(4, Rank::new(2)),
        }
    }

    fn generic_quadrilateral_adj<T: NameType>(
        gender: Self::Gender,
        quad: T::DataQuadrilateral,
    ) -> String {
        match quad.unwrap_or_default() {
            Quadrilateral::Square => Self::square_adj(gender),
            Quadrilateral::Rectangle => Self::rectangle_adj(gender),
            Quadrilateral::Orthodiagonal => Self::generic_adj(gender, 4, Rank::new(2)),
        }
    }

    /// Parses a generic cuboid (regular or irregular) as a noun.
    fn generic_cuboid_noun<T: NameType>(regular: &T::DataRegular) -> ParseOutput<Self::Gender> {
        if regular.satisfies(Regular::is_yes) {
            Self::cube_noun()
        } else {
            Self::cuboid_noun()
        }
    }

    /// Parses a generic cuboid (regular or irregular) as a noun.
    fn generic_cuboid_adj<T: NameType>(gender: Self::Gender, regular: &T::DataRegular) -> String {
        if regular.satisfies(Regular::is_yes) {
            Self::cube_adj(gender)
        } else {
            Self::cuboid_adj(gender)
        }
    }

    /// Parses a generic hyperblock (regular or irregular) as a noun.
    fn generic_hyperblock_noun<T: NameType>(
        regular: &T::DataRegular,
        rank: Rank,
    ) -> ParseOutput<Self::Gender> {
        if regular.satisfies(Regular::is_yes) {
            Self::hypercube_noun(rank)
        } else {
            Self::hyperblock_noun(rank)
        }
    }

    /// Parses a generic hyperblock (regular or irregular) as an adjective.
    fn generic_hyperblock_adj<T: NameType>(
        gender: Self::Gender,
        regular: &T::DataRegular,
        rank: Rank,
    ) -> String {
        if regular.satisfies(Regular::is_yes) {
            Self::hypercube_adj(gender, rank)
        } else {
            Self::hyperblock_adj(gender, rank)
        }
    }

    impl_parse!();

    /// Parses the [`Name`] in the specified language.
    fn parse<T: NameType>(name: &Name<T>) -> String {
        Self::parse_noun(name).output
    }

    /// Parses the [`Name`] in the specified language. If the first character is
    /// ASCII, it makes it uppercase.
    fn parse_uppercase<T: NameType>(name: &Name<T>) -> String {
        let mut result = Self::parse(name);
        lang::uppercase_mut(&mut result);
        result
    }
}

#[derive(Clone, Copy, Debug, EnumIter, Serialize, Deserialize)]
pub enum SelectedLanguage {
    /// English
    En,
    // Spanish
    //  Es,

    // German
    //  De,
    // French
    // Fr,

    // Japanese
    // Ja,

    // Proto Indo-Iranian
    // Pii,
}

impl std::fmt::Display for SelectedLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Self::En => "English",
            //        Self::Es => "Spanish",
            //       Self::De => "German",
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
            //      Self::Es => Es::parse_uppercase(name),
            //       Self::De => De::parse_uppercase(name),
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
