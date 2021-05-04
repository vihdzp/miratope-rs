use std::{fmt::Debug, marker::PhantomData};

use crate::geometry::Point;

/// A type marker that determines whether a name describes an abstract or
/// concrete polytope.
pub trait NameType: Debug + Clone + PartialEq {
    /// Either `AbsData<Point>` or `ConData<Point>`. Workaround until generic
    /// associated types are stable.
    type DataPoint: NameData<Point>;

    /// Whether the name marker is for an abstract polytope.
    fn is_abstract() -> bool;
}

/// A trait for data associated to a name. It can either be [`AbsData`], which
/// is zero size and compares `true` with anything, or [`ConData`], which stores
/// an actual value which is used for comparisons.
pub trait NameData<T>: PartialEq + Debug + Clone {
    fn new(value: T) -> Self;
}

#[derive(Debug)]
/// Phantom data associated with an abstract polytope. Internally stores nothing,
/// and compares as `true` with anything else.
pub struct AbsData<T>(PhantomData<T>);

impl<T> Default for AbsData<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

/// Any two `AbsData` compare as equal to one another.
impl<T> PartialEq for AbsData<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> Clone for AbsData<T> {
    fn clone(&self) -> Self {
        Default::default()
    }
}

impl<T> Copy for AbsData<T> {}

impl<T: Debug> NameData<T> for AbsData<T> {
    fn new(_value: T) -> Self {
        Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// A name representing an abstract polytope.
pub struct Abs;

impl NameType for Abs {
    type DataPoint = AbsData<Point>;

    fn is_abstract() -> bool {
        true
    }
}

#[derive(Debug, Clone)]
/// Data associated with a concrete polytope.
pub struct ConData<T>(T);

impl<T: PartialEq> PartialEq for ConData<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Copy> Copy for ConData<T> {}

impl<T: PartialEq + Debug + Clone> NameData<T> for ConData<T> {
    fn new(value: T) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// A name representing a concrete polytope.
pub struct Con(bool);

impl NameType for Con {
    type DataPoint = ConData<Point>;

    fn is_abstract() -> bool {
        false
    }
}

/// A language-independent representation of a polytope name, in a syntax
/// tree-like structure structure.
///
/// Many of the variants are subject to complicated invariants which help keep
/// the translation code more modular by separation of concerns. If you
/// instanciate a `Name` directly, **you ought to guarantee that these
/// invariants hold.** Convenience methods are provided, which will guarantee these
/// invariants for you.
#[derive(Debug, Clone, PartialEq)]
pub enum Name<T: NameType> {
    /// A nullitope.
    Nullitope,

    /// A point.
    Point,

    /// A dyad.
    Dyad,

    /// An irregular triangle.
    Triangle,

    /// An equilateral triangle.
    RegularTriangle,

    /// A square.
    Square,

    /// An irregular rectangle.
    Rectangle,

    /// A polygon with **at least 2** sides.
    Polygon(usize),

    /// A regular polygon with **at least 2** sides.
    RegularPolygon(usize),

    /// An orthodiagonal quadrilateral.
    Orthodiagonal,

    /// A pyramid based on some polytope.
    Pyramid(Box<Name<T>>),

    /// A prism based on some polytope.
    Prism(Box<Name<T>>),

    /// A tegum based on some polytope.
    Tegum(Box<Name<T>>),

    /// A multipyramid based on a list of polytopes. The list must contain **at
    /// least 2** elements, and contain nothing that can be interpreted as a
    /// multipyramid.
    Multipyramid(Vec<Name<T>>),

    /// A multiprism based on a list of polytopes. The list must contain at
    /// least two elements, be "sorted", and contain nothing that can be
    /// interpreted as a multiprism.
    Multiprism(Vec<Name<T>>),

    /// A multitegum based on a list of polytopes.
    Multitegum(Vec<Name<T>>),

    /// A multicomb based on a list of polytopes.
    Multicomb(Vec<Name<T>>),

    /// The dual of a specified polytope.
    Dual {
        base: Box<Name<T>>,
        center: T::DataPoint,
    },

    /// A simplex of a given dimension, **at least 3.**
    Simplex(usize),

    /// A regular simplex of a given dimension, **at least 3.**
    RegularSimplex(usize),

    /// A hyperblock of a given rank, **at least 3.**
    Hyperblock(usize),

    /// A regular hypercube of a given rank, **at least 3.**
    Hypercube(usize),

    /// An orthoplex of a given dimension, **at least 3.**
    Orthoplex(usize),

    /// A regular orthoplex of a given dimension, **at least 3.**
    RegularOrthoplex(usize),

    /// A polytope with a given facet count and rank, in that order. The facet
    /// count must be **at least 2,** and the dimension must be **at least 3**
    /// and **at most 20.**
    Generic { facet_count: usize, rank: usize },

    /// A compound of some polytopes.
    Compound(Vec<(usize, Name<T>)>),
}

impl<T: NameType> Name<T> {
    /// Auxiliary function to get the rank of a multiproduct.
    fn rank_product(&self) -> Option<isize> {
        // The bases of the product, and the difference between the rank of a
        // product of two polytopes and the sum of their ranks.
        let (bases, offset) = match self {
            Self::Multipyramid(bases) => (bases, 1),
            Self::Multiprism(bases) | Self::Multitegum(bases) => (bases, 0),
            Self::Multicomb(bases) => (bases, -1),
            _ => return None,
        };

        let mut rank = -offset;
        for base in bases.iter() {
            rank += base.rank()? + offset;
        }
        Some(rank)
    }

    /// Returns the rank of the polytope that the name describes, or `None` if
    /// it's unable to figure it out.
    ///
    /// # Todo
    /// We need to embed enough metadata in the name for this to always be able
    /// to figure out a rank.
    pub fn rank(&self) -> Option<isize> {
        match self {
            Name::Nullitope => Some(-1),
            Name::Point => Some(0),
            Name::Dyad => Some(1),
            Name::Triangle
            | Name::RegularTriangle
            | Name::Square
            | Name::Rectangle
            | Name::Orthodiagonal
            | Name::Polygon(_)
            | Name::RegularPolygon(_) => Some(2),
            Name::Simplex(rank)
            | Name::RegularSimplex(rank)
            | Name::Hyperblock(rank)
            | Name::Hypercube(rank)
            | Name::Orthoplex(rank)
            | Name::RegularOrthoplex(rank) => Some(*rank as isize),
            Name::Dual { base, center: _ } => base.rank(),
            Name::Generic {
                facet_count: _,
                rank,
            } => Some(*rank as isize),
            Name::Pyramid(base) | Name::Prism(base) | Name::Tegum(base) => Some(base.rank()? + 1),
            Name::Multipyramid(_)
            | Name::Multiprism(_)
            | Name::Multitegum(_)
            | Name::Multicomb(_) => self.rank_product(),
            Name::Compound(components) => components[0].1.rank(),
        }
    }

    /// Returns the number of facets of the polytope that the name describes, or
    /// `None` if it's unable to figure it out.
    ///
    /// # Todo
    /// We need to embed enough metadata in the name for this to always be able
    /// to figure out a facet count.
    pub fn facet_count(&self) -> Option<usize> {
        Some(match self {
            Name::Nullitope => 0,
            Name::Point => 1,
            Name::Dyad => 2,
            Name::Triangle | Name::RegularTriangle => 3,
            Name::Square | Name::Rectangle | Name::Orthodiagonal => 4,
            Name::Polygon(n)
            | Name::RegularPolygon(n)
            | Name::Generic {
                facet_count: n,
                rank: _,
            } => *n,
            Name::Simplex(rank) | Name::RegularSimplex(rank) => *rank + 1,
            Name::Hyperblock(rank) | Name::Hypercube(rank) => *rank * 2,
            Name::Orthoplex(rank) | Name::RegularOrthoplex(rank) => 2u32.pow(*rank as u32) as usize,
            Name::Multipyramid(bases) | Name::Multitegum(bases) => {
                let mut facet_count = 1;
                for base in bases {
                    facet_count *= base.facet_count()?;
                }
                facet_count
            }
            Name::Multiprism(bases) | Name::Multicomb(bases) => {
                let mut facet_count = 0;
                for base in bases {
                    facet_count += base.facet_count()?;
                }
                facet_count
            }
            Name::Pyramid(base) => base.facet_count()? + 1,
            Name::Prism(base) => 2 * (base.facet_count()? + 1),
            Name::Tegum(base) => 2 * base.facet_count()?,
            Name::Compound(components) => {
                let mut facet_count = 0;
                for (rep, name) in components.iter() {
                    facet_count += rep * name.facet_count()?;
                }
                facet_count
            }
            Name::Dual { base: _, center: _ } => return None,
        })
    }

    /// Determines whether a `Name` is valid, that is, all of the conditions
    /// specified on its variants hold. Used for debugging.
    pub fn is_valid(&self) -> bool {
        match self {
            Self::Simplex(rank)
            | Self::RegularSimplex(rank)
            | Self::Hyperblock(rank)
            | Self::Hypercube(rank)
            | Self::Orthoplex(rank)
            | Self::RegularOrthoplex(rank) => *rank >= 3,
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
                    if std::mem::discriminant(base) == std::mem::discriminant(self) {
                        return false;
                    }
                }

                // We should check that the bases are sorted somehow.

                true
            }
            Self::Generic { facet_count, rank } => *facet_count >= 2 && *rank >= 3 && *rank <= 20,
            _ => true,
        }
    }

    pub fn generic(n: usize, d: isize) -> Self {
        match d {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Polygon(n),
            _ => Self::Generic {
                facet_count: n,
                rank: d as usize,
            },
        }
    }

    /// Builds a pyramid name from a given name.
    pub fn pyramid(self) -> Self {
        match self {
            Self::Nullitope => Self::Nullitope,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::triangle(T::is_abstract()),
            Self::Triangle => {
                if T::is_abstract() {
                    Self::RegularSimplex(3)
                } else {
                    Self::Simplex(3)
                }
            }
            Self::RegularTriangle => {
                if T::is_abstract() {
                    Self::RegularSimplex(3)
                } else {
                    Self::Pyramid(Box::new(self))
                }
            }
            Self::Simplex(rank) => Self::Simplex(rank + 1),
            Self::RegularSimplex(rank) => {
                if T::is_abstract() {
                    Self::RegularSimplex(rank + 1)
                } else {
                    Self::Pyramid(Box::new(self))
                }
            }
            Self::Pyramid(base) => Self::multipyramid(vec![*base, Self::Dyad]),
            Self::Multipyramid(mut bases) => {
                bases.push(Self::Point);
                Self::multipyramid(bases)
            }
            _ => Self::Pyramid(Box::new(self)),
        }
    }

    /// Builds a prism name from a given name.
    pub fn prism(self) -> Self {
        match self {
            Self::Nullitope => Self::Nullitope,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::rectangle(T::is_abstract()),
            Self::Rectangle => Self::Hyperblock(3),
            Self::Hyperblock(rank) => Self::Hyperblock(rank + 1),
            Self::Hypercube(_) => Self::Prism(Box::new(self)),
            Self::Prism(base) => Self::multiprism(vec![*base, Self::rectangle(T::is_abstract())]),
            Self::Multiprism(mut bases) => {
                bases.push(Self::Dyad);
                Self::multiprism(bases)
            }
            _ => Self::Prism(Box::new(self)),
        }
    }

    /// Builds a tegum name from a given name.
    pub fn tegum(self) -> Self {
        match self {
            Self::Nullitope => Self::Nullitope,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::orthodiagonal(T::is_abstract()),
            Self::Orthodiagonal => Self::Orthoplex(3),
            Self::Orthoplex(rank) => Self::Orthoplex(rank + 1),
            Self::RegularOrthoplex(rank) => {
                if T::is_abstract() {
                    Self::RegularOrthoplex(rank + 1)
                } else {
                    Self::Tegum(Box::new(self))
                }
            }
            Self::Tegum(base) => Self::multitegum(vec![*base, Self::Orthodiagonal]),
            Self::Multitegum(mut bases) => {
                bases.push(Self::Dyad);
                Self::multitegum(bases)
            }
            _ => Self::Tegum(Box::new(self)),
        }
    }

    /// Builds a dual name from a given name.
    pub fn dual(self, center: T::DataPoint) -> Self {
        match self {
            Self::Nullitope | Self::Point | Self::Dyad => self,
            Self::Dual {
                base,
                center: original_center,
            } => {
                if center == original_center {
                    *base
                } else {
                    // Instead of stacking duals, we just default to generic
                    // names after two duals.
                    Self::Generic {
                        facet_count: base.facet_count().unwrap(),
                        rank: base.rank().unwrap() as usize,
                    }
                }
            }
            Self::Square | Self::Rectangle => Self::orthodiagonal(T::is_abstract()),
            Self::Orthodiagonal => Self::polygon(4),
            Self::Simplex(_) => self,
            Self::RegularSimplex(rank) => {
                if T::is_abstract() {
                    self
                } else {
                    Self::Simplex(rank)
                }
            }
            Self::Hyperblock(rank) => Self::Orthoplex(rank),
            Self::Hypercube(rank) => {
                if T::is_abstract() {
                    Self::RegularOrthoplex(rank)
                } else {
                    Self::Orthoplex(rank)
                }
            }
            Self::Orthoplex(rank) => Self::Hyperblock(rank),
            Self::RegularOrthoplex(rank) => {
                if T::is_abstract() {
                    Self::Hypercube(rank)
                } else {
                    Self::Hyperblock(rank)
                }
            }
            Self::Generic {
                facet_count: _,
                rank,
            } => {
                if rank <= 2 {
                    self
                } else {
                    Self::Dual {
                        base: Box::new(self),
                        center,
                    }
                }
            }
            Self::Pyramid(base) => {
                if T::is_abstract() {
                    Self::Pyramid(Box::new(base.dual(center)))
                } else {
                    Self::Dual {
                        base: Box::new(Self::Prism(base)),
                        center,
                    }
                }
            }
            Self::Prism(base) => {
                if T::is_abstract() {
                    Self::Tegum(Box::new(base.dual(center)))
                } else {
                    Self::Dual {
                        base: Box::new(Self::Prism(base)),
                        center,
                    }
                }
            }
            Self::Tegum(base) => {
                if T::is_abstract() {
                    Self::Prism(Box::new(base.dual(center)))
                } else {
                    Self::Dual {
                        base: Box::new(Self::Prism(base)),
                        center,
                    }
                }
            }
            Self::Multipyramid(bases) => {
                // I don't know if this relation actually holds in concrete polytopes.
                Self::Multipyramid(
                    bases
                        .into_iter()
                        .map(|base| base.dual(center.clone()))
                        .collect(),
                )
            }
            Self::Multiprism(bases) => {
                if T::is_abstract() {
                    Self::Multitegum(
                        bases
                            .into_iter()
                            .map(|base| base.dual(center.clone()))
                            .collect(),
                    )
                } else {
                    Self::Dual {
                        base: Box::new(Self::Multiprism(bases)),
                        center,
                    }
                }
            }
            Self::Multitegum(bases) => {
                if T::is_abstract() {
                    Self::Multiprism(
                        bases
                            .into_iter()
                            .map(|base| base.dual(center.clone()))
                            .collect(),
                    )
                } else {
                    Self::Dual {
                        base: Box::new(Self::Multitegum(bases)),
                        center,
                    }
                }
            }
            Self::Multicomb(bases) => {
                if T::is_abstract() {
                    Self::Multicomb(
                        bases
                            .into_iter()
                            .map(|base| base.dual(center.clone()))
                            .collect(),
                    )
                } else {
                    Self::Dual {
                        base: Box::new(Self::Multicomb(bases)),
                        center,
                    }
                }
            }
            _ => Self::Dual {
                base: Box::new(self),
                center,
            },
        }
    }

    pub fn triangle(abs: bool) -> Self {
        if abs {
            Self::RegularTriangle
        } else {
            Self::Triangle
        }
    }

    pub fn rectangle(abs: bool) -> Self {
        if abs {
            Self::Square
        } else {
            Self::Rectangle
        }
    }

    pub fn orthodiagonal(abs: bool) -> Self {
        if abs {
            Self::Square
        } else {
            Self::Orthodiagonal
        }
    }

    /// The name for an *n*-simplex.
    pub fn simplex(rank: isize) -> Self {
        match rank {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Triangle,
            _ => Self::Simplex(rank as usize),
        }
    }

    pub fn regular_simplex(rank: isize) -> Self {
        match rank {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::RegularTriangle,
            _ => Self::RegularSimplex(rank as usize),
        }
    }

    pub fn cuboid(rank: isize) -> Self {
        match rank {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Rectangle,
            _ => Self::Hyperblock(rank as usize),
        }
    }

    /// The name for an *n*-hypercube.
    pub fn hypercube(rank: isize) -> Self {
        match rank {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Square,
            _ => Self::Hypercube(rank as usize),
        }
    }

    /// The name for an *n*-orthoplex.
    pub fn orthoplex(rank: isize) -> Self {
        match rank {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Orthodiagonal,
            _ => Self::Orthoplex(rank as usize),
        }
    }

    pub fn regular_orthoplex(rank: isize) -> Self {
        match rank {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Square,
            _ => Self::RegularOrthoplex(rank as usize),
        }
    }

    /// Returns the name for a polygon (not necessarily regular) of `n` sides.
    pub fn polygon(n: usize) -> Self {
        match n {
            3 => Self::Triangle,

            _ => Self::Generic {
                facet_count: n,
                rank: 2,
            },
        }
    }

    pub fn regular_polygon(n: usize) -> Self {
        match n {
            3 => Self::RegularTriangle,
            4 => Self::Square,
            _ => Self::Generic {
                facet_count: n,
                rank: 2,
            },
        }
    }

    /// Sorts the bases of a multiproduct according to their rank, and then
    /// their facet count.
    fn base_cmp(base0: &Self, base1: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // Returns an Ordering if it's not equal to Ordering::Equal.
        macro_rules! return_if_ne {
            ($x:expr) => {
                let macro_x = $x;

                if macro_x != Ordering::Equal {
                    return macro_x;
                }
            };
        }

        // Names are firstly compared by rank.
        return_if_ne!(base0.rank().unwrap_or(-2).cmp(&base1.rank().unwrap_or(-2)));

        // If we know the facet count of the names, a name with less facets
        // compares as less to one with more facets.
        return_if_ne!(base0
            .facet_count()
            .unwrap_or(0)
            .cmp(&base1.facet_count().unwrap_or(0)));

        // The names are equal for all we care about.
        Ordering::Equal
    }

    pub fn multipyramid(bases: Vec<Name<T>>) -> Self {
        let mut new_bases = Vec::new();
        let mut pyramid_count = 0;

        // Figures out which bases of the multipyramid are multipyramids
        // themselves, and accounts for them accordingly.
        for base in bases.into_iter() {
            match base {
                Self::Nullitope => {}
                Self::Point => pyramid_count += 1,
                Self::Dyad => pyramid_count += 2,
                Self::Triangle | Self::RegularTriangle => pyramid_count += 2,
                Self::Simplex(rank) | Self::RegularSimplex(rank) => pyramid_count += rank + 1,
                Self::Multipyramid(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one pyramid, we combine all of them into a
        // single simplex.
        if pyramid_count >= 2 {
            new_bases.push(Name::simplex(pyramid_count as isize - 1));
        }

        // Sorts the bases by convention.
        new_bases.sort_by(&Self::base_cmp);

        let multipyramid = match new_bases.len() {
            0 => Self::Nullitope,
            1 => new_bases.swap_remove(0),
            _ => Self::Multipyramid(new_bases),
        };

        // If we take exactly one pyramid, we apply it at the end.
        if pyramid_count == 1 {
            multipyramid.pyramid()
        }
        // Otherwise, we already combined them.
        else {
            multipyramid
        }
    }

    pub fn multiprism(bases: Vec<Name<T>>) -> Self {
        let mut new_bases = Vec::new();
        let mut prism_count = 0;

        // Figures out which bases of the multipyramid are multipyramids
        // themselves, and accounts for them accordingly.
        for base in bases.into_iter() {
            match base {
                Self::Nullitope => {
                    return Self::Nullitope;
                }
                Self::Point => {}
                Self::Dyad => prism_count += 1,
                Self::Square | Self::Rectangle => prism_count += 2,
                Self::Hyperblock(rank) | Self::Hypercube(rank) => prism_count += rank,
                Self::Multiprism(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one prism, we combine all of them into a
        // single hyperblock.
        if prism_count >= 2 {
            new_bases.push(Name::cuboid(prism_count as isize));
        }

        // Sorts the bases by convention.
        new_bases.sort_by(&Self::base_cmp);

        let multiprism = match new_bases.len() {
            0 => Self::Point,
            1 => new_bases.swap_remove(0),
            _ => Self::Multiprism(new_bases),
        };

        // If we take exactly one prism, we apply it at the end.
        if prism_count == 1 {
            multiprism.prism()
        }
        // Otherwise, we already combined them.
        else {
            multiprism
        }
    }

    pub fn multitegum(bases: Vec<Name<T>>) -> Self {
        let mut new_bases = Vec::new();
        let mut tegum_count = 0;

        // Figures out which bases of the multipyramid are multipyramids
        // themselves, and accounts for them accordingly.
        for base in bases.into_iter() {
            match base {
                Self::Nullitope => {
                    return Self::Nullitope;
                }
                Self::Point => {}
                Self::Dyad => tegum_count += 1,
                Self::Square => tegum_count += 2,
                Self::Orthoplex(rank) | Self::RegularOrthoplex(rank) => tegum_count += rank,
                Self::Multitegum(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one tegum, we combine all of them into a
        // single orthoplex.
        if tegum_count >= 2 {
            new_bases.push(if T::is_abstract() {
                Name::RegularOrthoplex(tegum_count)
            } else {
                Name::Orthoplex(tegum_count)
            });
        }

        // Sorts the bases by convention.
        new_bases.sort_by(&Self::base_cmp);

        let multitegum = match new_bases.len() {
            0 => Self::Point,
            1 => new_bases.swap_remove(0),
            _ => Self::Multitegum(new_bases),
        };

        // If we take exactly one tegum, we apply it at the end.
        if tegum_count == 1 {
            multitegum.tegum()
        }
        // Otherwise, we already combined them.
        else {
            multitegum
        }
    }

    pub fn multicomb(bases: Vec<Self>) -> Self {
        let mut new_bases = Vec::new();

        // Figures out which bases of the multipyramid are multipyramids
        // themselves, and accounts for them accordingly.
        for base in bases.into_iter() {
            if let Self::Multicomb(mut extra_bases) = base {
                new_bases.append(&mut extra_bases);
            } else {
                new_bases.push(base);
            }
        }

        // Sorts the bases by convention.
        new_bases.sort_by(&Self::base_cmp);

        match new_bases.len() {
            0 => Self::Point,
            1 => new_bases.swap_remove(0),
            _ => Self::Multicomb(new_bases),
        }
    }

    pub fn compound(mut components: Vec<(usize, Self)>) -> Self {
        use itertools::Itertools;

        components.sort_by(|(_, name0), (_, name1)| Self::base_cmp(name0, name1));

        let mut new_components: Vec<(usize, _)> = Vec::new();
        for (rep, name) in components {
            if let Self::Compound(mut extra_components) = name {
                new_components.append(&mut extra_components);
            } else {
                new_components.push((rep, name));
            }
        }

        new_components.sort_by(|(_, name0), (_, name1)| Self::base_cmp(name0, name1));
        let mut components = Vec::new();

        for (name, group) in &new_components
            .into_iter()
            .group_by(|(_, name)| name.clone())
        {
            if let Self::Compound(mut extra_components) = name {
                components.append(&mut extra_components);
            } else {
                components.push((group.map(|(rep, _)| rep).sum(), name));
            }
        }

        if components.len() == 1 {
            components.swap_remove(0).1
        } else {
            Self::Compound(components)
        }
    }
}
