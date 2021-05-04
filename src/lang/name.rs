use std::{fmt::Debug, marker::PhantomData};

/// A type marker that determines whether a name describes an abstract or
/// concrete polytope.
pub trait NameType: Debug + Clone + PartialEq {
    /// Either `AbsData<bool>` or `ConData<bool>`. Workaround until generic
    /// associated types are stable.
    type DataBool: NameData<bool> + Copy;

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

impl<T> PartialEq for AbsData<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> Clone for AbsData<T> {
    fn clone(&self) -> Self {
        Self(Default::default())
    }
}

impl<T> Copy for AbsData<T> {}

impl<T: Debug> NameData<T> for AbsData<T> {
    fn new(_value: T) -> Self {
        Self(Default::default())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// A name representing an abstract polytope.
pub struct Abs;

impl NameType for Abs {
    type DataBool = AbsData<bool>;

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
    type DataBool = ConData<bool>;

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

    /// A triangle, which stores whether it's regular.
    Triangle(T::DataBool),

    /// A square.
    Square,

    /// An irregular rectangle.
    Rectangle,

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
    Dual(Box<Name<T>>),

    /// A simplex of a given dimension, **at least 3.** The boolean stores
    /// whether it's regular, the integer stores its rank.
    Simplex(T::DataBool, usize),

    /// A regular hypercube of a given dimension, **at least 3.** The boolean
    /// stores whether it's regular, the integer stores its rank.
    Hypercube(T::DataBool, usize),

    /// A regular orthoplex of a given dimension, **at least 2.** The boolean
    /// stores whether it's regular, the integer stores its rank.
    Orthoplex(T::DataBool, usize),

    /// A polytope with a given facet count and rank, in that order. The facet
    /// count must be **at least 2,** and the dimension must be **at most 20.**
    Generic(usize, usize),

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
    pub fn rank(&self) -> Option<isize> {
        match self {
            Name::Nullitope => Some(-1),
            Name::Point => Some(0),
            Name::Dyad => Some(1),
            Name::Triangle(_) | Name::Square | Name::Rectangle | Name::Orthodiagonal => Some(2),
            Name::Simplex(_, rank) | Name::Hypercube(_, rank) | Name::Orthoplex(_, rank) => {
                Some(*rank as isize)
            }
            Name::Dual(base) => base.rank(),
            Name::Generic(_, d) => Some(*d as isize),
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
    pub fn facet_count(&self) -> Option<usize> {
        Some(match self {
            Name::Nullitope => 0,
            Name::Point => 1,
            Name::Dyad => 2,
            Name::Triangle(_) => 3,
            Name::Square | Name::Rectangle | Name::Orthodiagonal => 4,
            Name::Generic(n, _) => *n,
            Name::Simplex(_, n) => *n + 1,
            Name::Hypercube(_, n) => *n * 2,
            Name::Orthoplex(_, n) => 2u32.pow(*n as u32) as usize,
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
            Name::Dual(_) => return None,
        })
    }

    /// Determines whether a `Name` is valid, that is, all of the conditions
    /// specified on its variants hold. Used for debugging.
    pub fn is_valid(&self) -> bool {
        match self {
            Self::Simplex(_, n) | Self::Hypercube(_, n) | Self::Orthoplex(_, n) => *n >= 3,
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
            _ => true,
        }
    }

    pub fn rectangle(regular: T::DataBool) -> Self {
        if regular == T::DataBool::new(true) {
            Self::Square
        } else {
            Self::Rectangle
        }
    }

    pub fn orthodiagonal(regular: T::DataBool) -> Self {
        if regular == T::DataBool::new(true) {
            Self::Square
        } else {
            Self::Orthodiagonal
        }
    }

    pub fn generic(n: usize, d: isize) -> Self {
        match d {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            _ => Self::Generic(n, d as usize),
        }
    }

    /// Builds a pyramid name from a given name.
    pub fn pyramid(self) -> Self {
        match self {
            Self::Nullitope => Self::Nullitope,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::Generic(3, 2),
            Self::Triangle(regular) => {
                if regular == T::DataBool::new(true) {
                    Self::Pyramid(Box::new(self))
                } else {
                    Self::Simplex(T::DataBool::new(false), 3)
                }
            }
            Self::Simplex(regular, n) => {
                if regular == T::DataBool::new(true) {
                    Self::Pyramid(Box::new(self))
                } else {
                    Self::Simplex(T::DataBool::new(false), n + 1)
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
            Self::Dyad => Self::rectangle(T::DataBool::new(false)),
            Self::Rectangle => Self::Hypercube(T::DataBool::new(false), 3),
            Self::Hypercube(regular, n) => {
                if regular == T::DataBool::new(true) {
                    Self::Prism(Box::new(self))
                } else {
                    Self::Hypercube(T::DataBool::new(false), n + 1)
                }
            }
            Self::Prism(base) => {
                Self::multiprism(vec![*base, Self::rectangle(T::DataBool::new(false))])
            }
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
            Self::Dyad => Self::orthoplex(T::DataBool::new(false), 2),
            Self::Orthodiagonal => Self::Orthoplex(T::DataBool::new(false), 3),
            Self::Orthoplex(regular, n) => {
                if regular == T::DataBool::new(true) {
                    Self::Tegum(Box::new(self))
                } else {
                    Self::Orthoplex(T::DataBool::new(false), n + 1)
                }
            }
            Self::Tegum(base) => {
                Self::multitegum(vec![*base, Self::Orthoplex(T::DataBool::new(false), 2)])
            }
            Self::Multitegum(mut bases) => {
                bases.push(Self::Dyad);
                Self::multitegum(bases)
            }
            _ => Self::Tegum(Box::new(self)),
        }
    }

    /// Builds a dual name from a given name.
    pub fn dual(self) -> Self {
        match self {
            Self::Nullitope | Self::Point | Self::Dyad => self,
            Self::Dual(base) => {
                // Abstractly, duals of duals give back the original polytope.
                if T::is_abstract() {
                    if let Self::Dual(original) = *base {
                        *original
                    } else {
                        Self::Dual(base)
                    }
                }
                // Geometrically, we have no guarantees.
                else {
                    Self::Dual(base)
                }
            }
            Self::Square | Self::Rectangle => Self::orthodiagonal(T::DataBool::new(false)),
            Self::Orthodiagonal => Self::polygon(T::DataBool::new(false), 4),
            Self::Simplex(_, n) => Self::Simplex(T::DataBool::new(false), n),
            Self::Hypercube(_, n) => Self::Orthoplex(T::DataBool::new(false), n),
            Self::Orthoplex(_, n) => Self::Hypercube(T::DataBool::new(false), n),
            Self::Generic(_, d) => {
                if d <= 2 {
                    self
                } else {
                    Self::Dual(Box::new(self))
                }
            }
            Self::Pyramid(base) => {
                if T::is_abstract() {
                    Self::Pyramid(Box::new(base.dual()))
                } else {
                    Self::Dual(Box::new(Self::Prism(base)))
                }
            }
            Self::Prism(base) => {
                if T::is_abstract() {
                    Self::Tegum(Box::new(base.dual()))
                } else {
                    Self::Dual(Box::new(Self::Prism(base)))
                }
            }
            Self::Tegum(base) => {
                if T::is_abstract() {
                    Self::Prism(Box::new(base.dual()))
                } else {
                    Self::Dual(Box::new(Self::Prism(base)))
                }
            }
            Self::Multipyramid(bases) => {
                // I don't know if this relation actually holds in concrete polytopes.
                Self::Multipyramid(bases.into_iter().map(|base| base.dual()).collect())
            }
            Self::Multiprism(bases) => {
                if T::is_abstract() {
                    Self::Multitegum(bases.into_iter().map(|base| base.dual()).collect())
                } else {
                    Self::Dual(Box::new(Self::Multiprism(bases)))
                }
            }
            Self::Multitegum(bases) => {
                if T::is_abstract() {
                    Self::Multiprism(bases.into_iter().map(|base| base.dual()).collect())
                } else {
                    Self::Dual(Box::new(Self::Multitegum(bases)))
                }
            }
            Self::Multicomb(bases) => {
                if T::is_abstract() {
                    Self::Multicomb(bases.into_iter().map(|base| base.dual()).collect())
                } else {
                    Self::Dual(Box::new(Self::Multicomb(bases)))
                }
            }
            _ => Self::Dual(Box::new(self)),
        }
    }

    /// The name for an *n*-simplex.
    pub fn simplex(regular: T::DataBool, n: isize) -> Self {
        match n {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Triangle(regular),
            _ => Self::Simplex(regular, n as usize),
        }
    }

    /// The name for an *n*-hypercube.
    pub fn hypercube(regular: T::DataBool, n: isize) -> Self {
        match n {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => {
                if regular == T::DataBool::new(true) {
                    Self::Square
                } else {
                    Self::Rectangle
                }
            }
            _ => Self::Hypercube(regular, n as usize),
        }
    }

    /// The name for an *n*-orthoplex.
    pub fn orthoplex(regular: T::DataBool, n: isize) -> Self {
        match n {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => {
                if regular == T::DataBool::new(true) {
                    Self::Square
                } else {
                    Self::Orthodiagonal
                }
            }
            _ => Self::Orthoplex(regular, n as usize),
        }
    }

    /// Returns the name for a polygon (not necessarily regular) of `n` sides.
    pub fn polygon(regular: T::DataBool, n: usize) -> Self {
        match n {
            3 => Self::Triangle(regular),
            4 => {
                if regular == T::DataBool::new(true) {
                    Self::Square
                } else {
                    Self::Generic(4, 2)
                }
            }
            _ => Self::Generic(n, 2),
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
                Self::Triangle(_) => pyramid_count += 2,
                Self::Simplex(_, n) => pyramid_count += n + 1,
                Self::Multipyramid(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one pyramid, we combine all of them into a
        // single simplex.
        if pyramid_count >= 2 {
            new_bases.push(Name::simplex(
                T::DataBool::new(false),
                pyramid_count as isize - 1,
            ));
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
                Self::Hypercube(_, n) => prism_count += n,
                Self::Multiprism(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one prism, we combine all of them into a
        // single hypercube.
        if prism_count >= 2 {
            new_bases.push(Name::hypercube(
                T::DataBool::new(false),
                prism_count as isize,
            ));
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
                Self::Orthoplex(_, n) => tegum_count += n,
                Self::Multitegum(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one tegum, we combine all of them into a
        // single orthoplex.
        if tegum_count >= 2 {
            new_bases.push(Name::orthoplex(
                T::DataBool::new(false),
                tegum_count as isize,
            ));
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
