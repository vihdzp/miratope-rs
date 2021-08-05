//! Module that defines a language-independent representation of polytope names.

use std::{fmt::Debug, fs, marker::PhantomData, mem};

use miratope_core::{geometry::Point, Consts, Float};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// The trait for a type marker that determines whether a name describes an
/// abstract or concrete polytope.
///
/// For some reason, adding `DeserializeOwned` causes the trait to jank out.
/// This means we need to keep typing it everywhere.
pub trait NameType: Debug + Clone + PartialEq + Serialize {
    /// Either `AbsData<Point>` or `ConData<Point>`. Workaround until generic
    /// associated types are stable.
    type DataPoint: NameData<Point> + DeserializeOwned;

    /// Either `AbsData<Regular>` or `ConData<Regular>`. Workaround until generic
    /// associated types are stable.
    type DataRegular: NameData<Regular> + Default + DeserializeOwned;

    /// Either `AbsData<Quadrilateral>` or `ConData<Quadrilateral>`. Workaround until generic
    /// associated types are stable.
    type DataQuadrilateral: NameData<Quadrilateral> + Default + DeserializeOwned + Copy;

    /// Whether the name marker is for an abstract polytope.
    fn is_abstract() -> bool;
}

/// A trait for data associated to a name. It can either be [`AbsData`], which
/// is zero size and compares `true` with anything, or [`ConData`], which stores
/// an actual value of type `T` which is used for comparisons.
///
/// The idea is that `NameData` should be used to store whichever conditions on
/// concrete polytopes always hold on abstract polytopes.
pub trait NameData<T>: PartialEq + Debug + Clone + Serialize {
    /// Initializes a new `NameData` with a given value.
    fn new(value: T) -> Self;

    /// Determines whether `self` contains a given value.
    fn is(&self, value: &T) -> bool;

    /// Determines whether `self` satisfies a given predicate.
    fn satisfies<F: Fn(&T) -> bool>(&self, f: F) -> bool;

    /// Retrieves the wrapped value, or a specified value if none.
    fn unwrap_or(self, value: T) -> T;

    /// Retrieves the wrapped value, or the default value if none.
    fn unwrap_or_default(self) -> T
    where
        T: Default,
    {
        #[allow(clippy::or_fun_call)]
        self.unwrap_or(Default::default())
    }
}

/// Phantom data associated with an abstract polytope.
///
/// Will compare as equal to anything else, and will satisfy any predicate.
#[derive(Copy, Debug, Serialize, Deserialize)]
pub struct AbsData<T>(PhantomData<T>);

/// The default value is the only possible value.
impl<T> Default for AbsData<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Any two `AbsData` compare as equal to one another.
impl<T> PartialEq for AbsData<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Since any `AbsData` stores the exact same info, cloning is trivial.
impl<T> Clone for AbsData<T> {
    fn clone(&self) -> Self {
        Default::default()
    }
}

impl<T: Debug> NameData<T> for AbsData<T> {
    /// Initializes a new `AbsData` that pretends to hold a given value.
    fn new(_: T) -> Self {
        Default::default()
    }

    /// Returns `true` no matter what, as if `self` actually held the given value.
    fn is(&self, _: &T) -> bool {
        true
    }

    /// Returns `true` no matter what, as if `self` actually satisfied the given predicate.
    fn satisfies<F: Fn(&T) -> bool>(&self, _: F) -> bool {
        true
    }

    /// Returns the specified value verbatim.
    fn unwrap_or(self, value: T) -> T {
        value
    }
}

/// A type marker for a name representing an abstract polytope.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Abs;

impl NameType for Abs {
    type DataPoint = AbsData<Point>;
    type DataRegular = AbsData<Regular>;
    type DataQuadrilateral = AbsData<Quadrilateral>;

    fn is_abstract() -> bool {
        true
    }
}

/// Data associated with a concrete polytope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConData<T>(T);

impl<T: PartialEq + Debug + Clone + Serialize + DeserializeOwned> NameData<T> for ConData<T> {
    /// Initializes a new `ConData` that holds a given value.
    fn new(value: T) -> Self {
        Self(value)
    }

    /// Determines whether `self` contains a given value.
    fn is(&self, value: &T) -> bool {
        &self.0 == value
    }

    /// Determines whether `self` satisfies the given predicate.
    fn satisfies<F: Fn(&T) -> bool>(&self, f: F) -> bool {
        f(&self.0)
    }

    /// Retrieves the wrapped value, ignores the argument.
    fn unwrap_or(self, _: T) -> T {
        self.0
    }
}

/// A type marker for a name representing a concrete polytope.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Con;

impl NameType for Con {
    type DataPoint = ConData<Point>;
    type DataRegular = ConData<Regular>;
    type DataQuadrilateral = ConData<Quadrilateral>;

    fn is_abstract() -> bool {
        false
    }
}

/// Determines whether a `Name` refers to a concrete regular polytope. This is
/// often indirectly stored as a `NameData<Regular>`, so that all abstract
/// polytopes behave as regular.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Regular {
    /// The polytope is indeed regular.
    Yes {
        /// The center of the polytope.
        // TODO: make into an Option to take the nullitope into account?
        center: Point,
    },

    /// The polytope is not regular.
    No,
}

/// We don't treat something as regular by default.
impl Default for Regular {
    fn default() -> Self {
        Self::No
    }
}

impl Regular {
    /// Returns whether `self` matches `Regular::Yes { .. }`.
    pub fn is_yes(&self) -> bool {
        matches!(self, Regular::Yes { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quadrilateral {
    /// A square.
    Square,

    /// A rectangle.
    Rectangle,

    /// An orthodiagonal quadrilateral (a dyadic duotegum).
    Orthodiagonal,
}

impl Default for Quadrilateral {
    fn default() -> Self {
        Self::Square
    }
}

impl Quadrilateral {
    pub fn is_square(self) -> bool {
        matches!(self, Self::Square)
    }
}

/// A language-independent representation of a polytope name, in a syntax
/// tree-like structure.
///
/// Many of the variants are subject to complicated invariants which help keep
/// the translation code more modular by separation of concerns. If you
/// instanciate a `Name` directly, **you ought to guarantee that these
/// invariants hold.** Convenience methods are provided, often with the same
/// name as their respective variants, which will guarantee these invariants for
/// you.
///
/// For more info on concrete usage of this type, see the [`parse`] function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Name<T: NameType> {
    /// A nullitope.
    Nullitope,

    /// A point.
    Point,

    /// A dyad.
    Dyad,

    /// A triangle.
    Triangle {
        /// Stores whether the triangle is regular, and its center if it is.
        regular: T::DataRegular,
    },

    /// A quadrilateral.
    Quadrilateral { quad: T::DataQuadrilateral },

    /// A polygon with **at least 4** sides if irregular, or **at least 5**
    /// sides if regular.
    Polygon {
        /// Stores whether the polygon is regular, and its center if it is.
        regular: T::DataRegular,

        /// The facet count of the polygon.
        n: usize,
    },

    /// A pyramid based on some polytope.
    Pyramid(Box<Name<T>>),

    /// A prism based on some polytope.
    Prism(Box<Name<T>>),

    /// A tegum based on some polytope.
    Tegum(Box<Name<T>>),

    /// A multipyramid based on a list of polytopes. The list must contain **at
    /// least two** elements, and contain nothing that can be interpreted as a
    /// multipyramid.
    Multipyramid(Vec<Name<T>>),

    /// A multiprism based on a list of polytopes. The list must contain **at
    /// least two** elements, and contain nothing that can be interpreted as a
    /// multiprism.
    Multiprism(Vec<Name<T>>),

    /// A multitegum based on a list of polytopes. The list must contain **at
    /// least two** elements, and contain nothing that can be interpreted as a
    /// multitegum.
    Multitegum(Vec<Name<T>>),

    /// A multicomb based on a list of polytopes. The list must contain **at
    /// least two** elements, and contain nothing that can be interpreted as a
    /// multicomb.
    Multicomb(Vec<Name<T>>),

    /// An antiprism based on a polytope.
    Antiprism {
        /// The polytope the antiprism is based upon.
        base: Box<Name<T>>,
    },

    /// An antitegum based on a polytope.
    Antitegum {
        /// The polytope the antitegum is based upon.
        base: Box<Name<T>>,

        /// The center used for dualizing.
        center: T::DataPoint,
    },

    /// The Petrial of a polyhedron.
    Petrial {
        /// The polytope the Petrial is based upon.
        base: Box<Name<T>>,
    },

    /// The dual of a specified polytope.
    Dual {
        /// The polytope the dual is based upon.
        base: Box<Name<T>>,

        /// The center used for dualizing.
        center: T::DataPoint,
    },

    /// A ditope based on a specified polytope.
    Ditope {
        /// The polytope the ditope is based upon.
        base: Box<Name<T>>,

        /// The rank of the ditope.
        rank: usize,
    },

    /// A hosotope based on a specified polytope.
    Hosotope {
        /// The polytope the hosotope is based upon.
        base: Box<Name<T>>,

        /// The rank of the hosotope.
        rank: usize,
    },

    /// A simplex of a given dimension, **at least 3.**
    Simplex {
        /// Stores whether the simplex is regular, and its center if it is.
        regular: T::DataRegular,

        /// The rank of the simplex.
        rank: usize,
    },

    /// A cuboid.
    Cuboid {
        /// Stores whether the cuboid is regular, and its center if it is.        
        regular: T::DataRegular,
    },

    /// A hyperblock of a given rank, **at least 4.**
    Hyperblock {
        /// Stores whether the hyperblock is regular, and its center if it is.        
        regular: T::DataRegular,

        /// The rank of the hyperblock.
        rank: usize,
    },

    /// An orthoplex (polytope whose opposite vertices form an orthogonal basis)
    /// of a given dimension, **at least 3.**
    Orthoplex {
        /// Stores whether the orthoplex is regular, and its center if it is.        
        regular: T::DataRegular,

        /// The rank of the orthoplex.
        rank: usize,
    },

    /// A polytope with a given facet count and rank, in that order. The facet
    /// count must be **at least 2,** and the dimension must be **at least 3**
    /// and **at most 20.**
    Generic {
        /// The number of facets of the polytope.
        facet_count: usize,

        /// The rank of the polytope.
        rank: usize,
    },

    /// A smaller variant of a polytope.
    Small(Box<Name<T>>),

    /// A greater variant of a polytope.
    Great(Box<Name<T>>),

    /// A stellation of a polytope.
    Stellated(Box<Name<T>>),
}

impl<T: NameType> Default for Name<T> {
    fn default() -> Self {
        Self::Nullitope
    }
}

impl<T: NameType> Name<T> {
    /// Gets the name from the first line of an OFF file.
    pub fn from_src(first_line: &str) -> Option<Self>
    where
        T: DeserializeOwned,
    {
        let mut fl_iter = first_line.char_indices();

        if let Some((_, '#')) = fl_iter.next() {
            let (idx, _) = fl_iter.next()?;
            if let Ok(new_name) = ron::from_str(&first_line[idx..]) {
                return Some(new_name);
            }
        }

        None
    }

    /// Reads a name, serialized from the first line of an OFF file.
    pub fn from_off<U: AsRef<std::path::Path>>(path: U) -> Option<Self>
    where
        T: DeserializeOwned,
    {
        use std::io::{BufRead, BufReader};

        let file = BufReader::new(fs::File::open(path).ok()?);
        let first_line = file.lines().next()?.ok()?;

        Self::from_src(&first_line)
    }

    /// Determines whether a `Name` is valid, that is, all of the conditions
    /// specified on its variants hold. Used for debugging.
    pub fn is_valid(&self) -> bool {
        match self {
            // Polygons must not be interpretable as triangles or squares.
            Self::Polygon { regular, n } => match *n {
                2 | 5..=usize::MAX => true,
                4 => !regular.satisfies(Regular::is_yes),
                _ => false,
            },

            // Petrials must always be 3D, but we have no way to check this.

            // Simplices and orthoplices must be at least 3D, otherwise they
            // have other names.
            Self::Simplex { rank, .. } | Self::Orthoplex { rank, .. } => *rank >= 4,

            // Hyperblocks can't be 3D, since Cuboids are a separate thing.
            Self::Hyperblock { rank, .. } => *rank >= 5,

            // Multioperations must contain at least two bases and nothing nested.
            Self::Multipyramid(bases)
            | Self::Multiprism(bases)
            | Self::Multitegum(bases)
            | Self::Multicomb(bases) => {
                // Any multiproduct must have at least two bases.
                if bases.len() < 2 {
                    return false;
                }

                // No base should have the same variant as self.
                for base in bases {
                    if mem::discriminant(base) == mem::discriminant(self) {
                        return false;
                    }
                }

                true
            }

            // Generic polytopes must have at least 2 facets, and rank between
            // 3 and 20.
            &Self::Generic { facet_count, rank } => facet_count >= 2 && rank >= 4 && rank <= 21,

            // For lack of info, we return true otherwise.
            _ => true,
        }
    }

    /// The name for a generic polytope with a given number of facets, and a
    /// given rank.
    pub fn generic(n: usize, rank: usize) -> Self {
        match rank {
            // Hardcoded names.
            0 => Self::Nullitope,
            1 => Self::Point,
            2 => Self::Dyad,

            // We use the same scheme as for irregular polygons in the 2D case.
            3 => Self::polygon(Default::default(), n),

            // Otherwise, we use a generic name.
            _ => Self::Generic {
                facet_count: n,
                rank,
            },
        }
    }

    /// Builds a pyramid name from a given name.
    pub fn pyramid(self) -> Self {
        match self {
            // Hardcoded cases.
            Self::Nullitope => Self::Point,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::Triangle {
                regular: Default::default(),
            },

            // We make irregular triangles into irregular simplices, and regular
            // triangles into triangular pyramids.
            Self::Triangle { regular } => {
                if regular.is(&Regular::No) {
                    Self::Simplex { regular, rank: 4 }
                } else {
                    Self::Pyramid(Box::new(Self::Triangle { regular }))
                }
            }

            // We make irregular simplices into irregular simplices, and regular
            // simplices into simplicial pyramids.
            Self::Simplex { regular, rank } => {
                if regular.is(&Regular::No) {
                    Self::Simplex {
                        regular,
                        rank: rank + 1,
                    }
                } else {
                    Self::Pyramid(Box::new(Self::Simplex { regular, rank }))
                }
            }

            // We integrate pyramids into a single multipyramid.
            Self::Pyramid(base) => Self::multipyramid(vec![Self::Dyad, *base]),

            // We integrate multipyramids into a single multipyramid.
            Self::Multipyramid(mut bases) => {
                bases.push(Self::Point);
                Self::multipyramid(bases)
            }

            // We default to just making a pyramid out of the base.
            _ => Self::Pyramid(Box::new(self)),
        }
    }

    /// Builds a prism name from a given name.
    pub fn prism(self) -> Self {
        match self {
            // Hardcoded cases.
            Self::Nullitope => Self::Nullitope,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::rectangle(),
            Self::Quadrilateral { quad } => {
                if quad.is(&Quadrilateral::Orthodiagonal) {
                    Self::Prism(Box::new(self))
                } else {
                    Self::Cuboid {
                        regular: Default::default(),
                    }
                }
            }

            // We make an irregular cuboid into an irregular cuboid, and
            // a regular one into a cubic prism.
            Self::Cuboid { ref regular } => {
                if regular.is(&Regular::No) {
                    Self::Hyperblock {
                        regular: Default::default(),
                        rank: 5,
                    }
                } else {
                    Self::Prism(Box::new(self))
                }
            }

            // We make an irregular hyperblock into an irregular hyperblock, and
            // a regular one into a hypercube prism.
            Self::Hyperblock { regular, rank } => {
                if regular.is(&Regular::No) {
                    Self::Hyperblock {
                        regular,
                        rank: rank + 1,
                    }
                } else {
                    Self::Prism(Box::new(Self::Hyperblock { regular, rank }))
                }
            }

            // We integrate prisms into a single multiprism.
            Self::Prism(base) => Self::multiprism(vec![Self::rectangle(), *base]),

            // We integrate multiprisms into a single multiprism.
            Self::Multiprism(mut bases) => {
                bases.push(Self::Dyad);
                Self::multiprism(bases)
            }

            // We default to just making a prism out of the base.
            _ => Self::Prism(Box::new(self)),
        }
    }

    /// Builds a tegum name from a given name.
    pub fn tegum(self) -> Self {
        match self {
            // Hardcoded cases.
            Self::Nullitope => Self::Nullitope,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::orthodiagonal(),
            Self::Quadrilateral { quad } => {
                if quad.is(&Quadrilateral::Rectangle) {
                    Self::Tegum(Box::new(self))
                } else {
                    Self::Orthoplex {
                        regular: Default::default(),
                        rank: 4,
                    }
                }
            }

            // We make an irregular orthoplex into an irregular orthoplex, and
            // a regular one into an orthoplex prism.
            Self::Orthoplex { regular, rank } => {
                if regular.is(&Regular::No) {
                    Self::Orthoplex {
                        regular,
                        rank: rank + 1,
                    }
                } else {
                    Self::Tegum(Box::new(Self::Orthoplex { regular, rank }))
                }
            }

            // We integrate tegums into a single multitegum.
            Self::Tegum(base) => Self::multitegum(vec![Self::orthodiagonal(), *base]),

            // We integrate multitegums into a single multitegum.
            Self::Multitegum(mut bases) => {
                bases.push(Self::Dyad);
                Self::multitegum(bases)
            }

            // We default to just making a tegum out of the base.
            _ => Self::Tegum(Box::new(self)),
        }
    }

    /// Builds an antiprism name from a given name.
    pub fn antiprism(self) -> Self {
        match self {
            // Hardcoded cases.
            Self::Nullitope => Self::Point,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::orthodiagonal(),

            // Simplices become irregular orthoplices.
            Self::Simplex { rank, .. } => Self::Orthoplex {
                rank: rank + 1,
                regular: Default::default(),
            },

            // We default to just making an antiprism out of the base.
            _ => Self::Antiprism {
                base: Box::new(self),
            },
        }
    }

    /// Builds a dual name from a given name. You must specify the facet count
    /// of the polytope this dual refers to.
    pub fn dual(self, center: T::DataPoint, facet_count: usize, rank: usize) -> Self {
        /// Constructs a regular dual from a regular polytope.
        macro_rules! regular_dual {
            ($regular: ident, $dual: ident $(, $other_fields: ident)*) => {
                if $regular.satisfies(|r| match r {
                    Regular::Yes {
                        center: original_center,
                    } => center.satisfies(|c| (c - original_center).norm() < Float::EPS),
                    Regular::No => true,
                }) {
                    Self::$dual {
                        $regular,
                        $($other_fields)*
                    }
                } else {
                    Self::$dual {
                        regular: Default::default(),
                        $($other_fields)*
                    }
                }
            };
        }

        /// Constructs a regular dual from a pyramid, prism, or tegum.
        macro_rules! modifier_dual {
            ($base: ident, $modifier: ident, $dual: ident) => {
                if T::is_abstract() {
                    Self::$dual(Box::new($base.dual(center, facet_count, rank)))
                } else {
                    Self::Dual {
                        base: Box::new(Self::$modifier($base)),
                        center,
                    }
                }
            };
        }

        /// Constructs a regular dual from a multipyramid, multiprism,
        /// multitegum, or multicomb.
        macro_rules! multimodifier_dual {
            ($bases: ident, $modifier: ident, $dual: ident) => {
                if T::is_abstract() {
                    Self::$dual(
                        $bases
                            .into_iter()
                            .map(|base| base.dual(center.clone(), facet_count, rank))
                            .collect(),
                    )
                } else {
                    Self::Dual {
                        base: Box::new(Self::$modifier($bases)),
                        center,
                    }
                }
            };
        }

        match self {
            // Self-dual polytopes.
            Self::Nullitope | Self::Point | Self::Dyad => self,

            // Other hardcoded cases.
            Self::Triangle { regular } => regular_dual!(regular, Triangle),
            Self::Quadrilateral { quad } => {
                if quad.is(&Quadrilateral::Orthodiagonal) {
                    Self::polygon(Default::default(), 4)
                } else {
                    Self::orthodiagonal()
                }
            }

            // Duals of duals become the original polytopes if possible, and
            // default to generic names otherwise.
            Self::Dual {
                base,
                center: original_center,
            } => {
                if center == original_center {
                    *base
                } else {
                    Self::Generic { facet_count, rank }
                }
            }

            // Regular duals.
            Self::Polygon { regular, n } => regular_dual!(regular, Polygon, n),
            Self::Simplex { regular, rank } => regular_dual!(regular, Simplex, rank),
            Self::Hyperblock { regular, rank } => regular_dual!(regular, Orthoplex, rank),
            Self::Orthoplex { regular, rank } => regular_dual!(regular, Hyperblock, rank),

            // Duals of modifiers.
            Self::Pyramid(base) => modifier_dual!(base, Pyramid, Pyramid),
            Self::Prism(base) => modifier_dual!(base, Prism, Tegum),
            Self::Tegum(base) => modifier_dual!(base, Tegum, Prism),
            Self::Antiprism { base } => Self::Antitegum { base, center },
            Self::Antitegum {
                base,
                center: original_center,
            } => {
                if center == original_center {
                    Self::Antiprism { base }
                } else {
                    Self::Dual {
                        base: Box::new(Self::Antitegum {
                            base,
                            center: original_center,
                        }),
                        center,
                    }
                }
            }

            // Duals of multi-modifiers.
            Self::Multipyramid(bases) => multimodifier_dual!(bases, Multipyramid, Multipyramid),
            Self::Multiprism(bases) => multimodifier_dual!(bases, Multiprism, Multitegum),
            Self::Multitegum(bases) => multimodifier_dual!(bases, Multitegum, Multiprism),
            Self::Multicomb(bases) => multimodifier_dual!(bases, Multicomb, Multicomb),

            // Defaults to just adding a dual before the name.
            _ => Self::Dual {
                base: Box::new(self),
                center,
            },
        }
    }

    /// Makes a ditope out of the name.
    pub fn ditope(self, rank: usize) -> Self {
        match self {
            // We do nothing in the case of the nullitope.
            Self::Nullitope => Self::Nullitope,

            // Hardcoded cases.
            Self::Point => Self::Dyad,
            Self::Dyad => Self::polygon(Default::default(), 2),

            // In any other case, we just box the polytope.
            _ => Self::Ditope {
                base: Box::new(self),
                rank,
            },
        }
    }

    /// Makes a ditope out of the name.
    pub fn hosotope(self, rank: usize) -> Self {
        match self {
            // We do nothing in the case of the nullitope.
            Self::Nullitope => Self::Nullitope,

            // Hardcoded cases.
            Self::Point => Self::Dyad,
            Self::Dyad => Self::polygon(Default::default(), 2),

            // In any other case, we just box the polytope.
            _ => Self::Hosotope {
                base: Box::new(self),
                rank,
            },
        }
    }

    /// Makes a Petrial out of the name.
    pub fn petrial(self) -> Self {
        match self {
            // Petrials are involutions.
            Self::Petrial { base } => *base,

            // In any other case, we just box the polytope.
            _ => Self::Petrial {
                base: Box::new(self),
            },
        }
    }

    /// Returns the name for a square.
    pub fn square() -> Self {
        Self::Quadrilateral {
            quad: Default::default(),
        }
    }

    /// Returns the name for a rectangle, depending on whether it's abstract or
    /// not.
    pub fn rectangle() -> Self {
        Self::Quadrilateral {
            quad: T::DataQuadrilateral::new(Quadrilateral::Rectangle),
        }
    }

    /// Returns the name for an orthodiagonal quadrilateral, depending on
    /// whether it's abstract or not.
    pub fn orthodiagonal() -> Self {
        Self::Quadrilateral {
            quad: T::DataQuadrilateral::new(Quadrilateral::Orthodiagonal),
        }
    }

    /// The name for an *n*-simplex, regular or not.
    pub fn simplex(regular: T::DataRegular, rank: usize) -> Self {
        match rank {
            0 => Self::Nullitope,
            1 => Self::Point,
            2 => Self::Dyad,
            3 => Self::Triangle { regular },
            _ => Self::Simplex { regular, rank },
        }
    }

    /// The name for an *n*-block, regular or not.
    pub fn hyperblock(regular: T::DataRegular, rank: usize) -> Self {
        match rank {
            0 => Self::Nullitope,
            1 => Self::Point,
            2 => Self::Dyad,
            3 => {
                if regular.satisfies(Regular::is_yes) {
                    Self::square()
                } else {
                    Self::rectangle()
                }
            }
            4 => Self::Cuboid { regular },
            _ => Self::Hyperblock { regular, rank },
        }
    }

    /// The name for an *n*-orthoplex.
    pub fn orthoplex(regular: T::DataRegular, rank: usize) -> Self {
        match rank {
            0 => Self::Nullitope,
            1 => Self::Point,
            2 => Self::Dyad,
            3 => Self::orthodiagonal(),
            _ => Self::Orthoplex { regular, rank },
        }
    }

    /// Returns the name for a polygon (not necessarily regular) of `n` sides.
    pub fn polygon(regular: T::DataRegular, n: usize) -> Self {
        match n {
            3 => Self::Triangle { regular },
            4 => {
                if regular.satisfies(Regular::is_yes) {
                    Self::square()
                } else {
                    Self::Polygon { regular, n }
                }
            }
            _ => Self::Polygon { regular, n },
        }
    }

    /// Makes a multipyramid out of a set of names. Uses the names in roughly
    /// the same order as were given.
    pub fn multipyramid(bases: Vec<Name<T>>) -> Self {
        let mut new_bases = Vec::new();
        let mut pyramid_count = 0;

        // Figures out which bases of the multipyramid are multipyramids
        // themselves, and accounts for them accordingly.
        for base in bases {
            match base {
                Self::Nullitope => {}
                Self::Point => pyramid_count += 1,
                Self::Dyad => pyramid_count += 2,
                Self::Triangle { .. } => pyramid_count += 3,
                Self::Simplex { rank, .. } => pyramid_count += rank,
                Self::Multipyramid(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one pyramid, we combine all of them into a
        // single simplex.
        if pyramid_count >= 2 {
            new_bases.push(Self::simplex(
                Default::default(),
                usize::from(pyramid_count - 1),
            ));
        }

        // Either the final name, or the single base.
        let multipyramid = match new_bases.len() {
            0 => Self::Nullitope,
            1 => new_bases.swap_remove(0),
            _ => Self::Multipyramid(new_bases),
        };

        // If we take exactly one pyramid, we apply it at the end.
        if pyramid_count == 1 {
            Self::Pyramid(Box::new(multipyramid))
        }
        // Otherwise, we already combined them.
        else {
            multipyramid
        }
    }

    /// Makes a multiprism out of a set of names. Uses the names in roughly
    /// the same order as were given.
    pub fn multiprism(bases: Vec<Name<T>>) -> Self {
        let mut new_bases = Vec::new();
        let mut prism_count = 0;

        // Figures out which bases of the multiprism are multiprisms themselves,
        // and accounts for them accordingly.
        for base in bases {
            match base {
                Self::Nullitope => {
                    return Self::Nullitope;
                }
                Self::Point => {}
                Self::Dyad => prism_count += 1,
                Self::Quadrilateral { quad } => {
                    if quad.is(&Quadrilateral::Orthodiagonal) {
                        new_bases.push(base);
                    } else {
                        prism_count += 2;
                    }
                }
                Self::Cuboid { .. } => prism_count += 3,
                Self::Hyperblock { rank, .. } => prism_count += rank,
                Self::Multiprism(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one prism, we combine all of them into a
        // single hyperblock.
        if prism_count >= 2 {
            new_bases.push(Self::hyperblock(
                Default::default(),
                usize::from(prism_count),
            ));
        }

        // Either the final name, or the single base.
        let multiprism = match new_bases.len() {
            0 => Self::Point,
            1 => new_bases.swap_remove(0),
            _ => Self::Multiprism(new_bases),
        };

        // If we take exactly one prism, we apply it at the end.
        if prism_count == 1 {
            Self::Prism(Box::new(multiprism))
        }
        // Otherwise, we already combined them.
        else {
            multiprism
        }
    }

    /// Makes a multitegum out of a set of names. Uses the names in roughly
    /// the same order as were given.
    pub fn multitegum(bases: Vec<Name<T>>) -> Self {
        let mut new_bases = Vec::new();
        let mut tegum_count = 0;

        // Figures out which bases of the multitegum are multitegums themselves,
        // and accounts for them accordingly.
        for base in bases {
            match base {
                Self::Nullitope => {
                    return Self::Nullitope;
                }
                Self::Point => {}
                Self::Dyad => tegum_count += 1,
                Self::Quadrilateral { quad } => {
                    if quad.is(&Quadrilateral::Rectangle) {
                        new_bases.push(base);
                    } else {
                        tegum_count += 2;
                    }
                }
                Self::Orthoplex { rank, .. } => tegum_count += rank,
                Self::Multitegum(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one tegum, we combine all of them into a
        // single orthoplex.
        if tegum_count >= 2 {
            new_bases.push(Self::orthoplex(
                Default::default(),
                usize::from(tegum_count),
            ));
        }

        // Either the final name, or the single base.
        let multitegum = match new_bases.len() {
            0 => Self::Point,
            1 => new_bases.swap_remove(0),
            _ => Self::Multitegum(new_bases),
        };

        // If we take exactly one tegum, we apply it at the end.
        if tegum_count == 1 {
            Self::Tegum(Box::new(multitegum))
        }
        // Otherwise, we already combined them.
        else {
            multitegum
        }
    }

    /// Makes a multicomb out of a set of names. Uses the names in roughly
    /// the same order as were given.
    pub fn multicomb(bases: Vec<Self>) -> Self {
        let mut new_bases = Vec::new();

        // Figures out which bases of the multicomb are multicombs themselves,
        // and accounts for them accordingly.
        for base in bases {
            if let Self::Multicomb(mut extra_bases) = base {
                new_bases.append(&mut extra_bases);
            } else {
                new_bases.push(base);
            }
        }

        // Either the final name, or the single base.
        match new_bases.len() {
            0 => Self::Point,
            1 => new_bases.swap_remove(0),
            _ => Self::Multicomb(new_bases),
        }
    }
}
