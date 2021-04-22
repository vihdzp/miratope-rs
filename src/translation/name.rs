use std::fmt::Debug;

/// Determines whether a name is to be treated as the name for an abstract
/// polytope. Doubles as a way to mark some name variants as coming from a
/// regular polytope or not.
pub trait NameType: Debug + Clone + PartialEq + Copy {
    fn is_abstract() -> bool;

    fn regular(x: bool) -> Self;

    fn is_regular(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Abs;

impl NameType for Abs {
    fn is_abstract() -> bool {
        true
    }

    fn regular(_: bool) -> Self {
        Self
    }

    fn is_regular(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Con(bool);

impl NameType for Con {
    fn is_abstract() -> bool {
        false
    }

    fn regular(x: bool) -> Self {
        Self(x)
    }

    fn is_regular(&self) -> bool {
        self.0
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
    Triangle(T),

    /// A square.
    Square,

    /// A rectangle.
    Rectangle,

    /// A pyramid based on some polytope. Don't instanciate this directly, use
    /// [`Name::pyramid`] instead.
    Pyramid(Box<Name<T>>),

    /// A prism based on some polytope. Don't instanciate this directly, use
    /// [`Name::prism`] instead.
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
    Simplex(T, usize),

    /// A regular hypercube of a given dimension, **at least 3.** The boolean stores
    /// whether it's regular, the integer stores its rank.
    Hypercube(T, usize),

    /// A regular orthoplex of a given dimension, **at least 2.** The boolean stores
    /// whether it's regular, the integer stores its rank.
    Orthoplex(T, usize),

    /// A polytope with a given facet count and rank, in that order. The facet
    /// count must be **at least 2,** and the dimension must be **at most 20.**
    Generic(usize, usize),

    /// The name of the polytope is unknown.
    Unknown,
}

impl<T: NameType> Default for Name<T> {
    fn default() -> Self {
        Self::Unknown
    }
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
            Name::Triangle(_) | Name::Square | Name::Rectangle => Some(2),
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
            _ => None,
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
            Name::Square | Name::Rectangle => 4,
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
            _ => return None,
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
                match bases.len() {
                    0..=1 => return false,
                    2 => {
                        if self != &Self::Multitegum(vec![]) {
                            return false;
                        }
                    }
                    _ => {}
                }

                // No base should have the same variant as self.
                for base in bases {
                    if base == self {
                        return false;
                    }
                }

                // We should check that the bases are sorted somehow.

                true
            }
            _ => true,
        }
    }

    /// Builds a pyramid name from a given name.
    pub fn pyramid(self) -> Self {
        match self {
            Self::Nullitope => Self::Nullitope,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::Generic(3, 2),
            Self::Triangle(regular) => {
                if regular.is_regular() {
                    Self::Pyramid(Box::new(self))
                } else {
                    Self::Simplex(T::regular(false), 3)
                }
            }
            Self::Simplex(regular, n) => {
                if regular.is_regular() {
                    Self::Pyramid(Box::new(self))
                } else {
                    Self::Simplex(T::regular(false), n + 1)
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
            Self::Dyad => Self::Square,
            Self::Square => Self::Hypercube(T::regular(false), 3),
            Self::Hypercube(regular, n) => {
                if regular.is_regular() {
                    Self::Prism(Box::new(self))
                } else {
                    Self::Hypercube(T::regular(false), n + 1)
                }
            }
            Self::Prism(base) => Self::multiprism(vec![*base, Self::Rectangle]),
            Self::Multiprism(mut bases) => {
                bases.push(Self::Dyad);
                Self::multipyramid(bases)
            }
            _ => Self::Prism(Box::new(self)),
        }
    }

    /// Builds a tegum name from a given name.
    pub fn tegum(self) -> Self {
        match self {
            Self::Nullitope => Self::Nullitope,
            Self::Point => Self::Dyad,
            Self::Dyad => Self::Square,
            Self::Square => Self::Orthoplex(T::regular(false), 3),
            Self::Orthoplex(regular, n) => {
                if regular.is_regular() {
                    Self::Tegum(Box::new(self))
                } else {
                    Self::Orthoplex(T::regular(false), n + 1)
                }
            }
            Self::Tegum(base) => {
                Self::multitegum(vec![*base, Self::Orthoplex(T::regular(false), 2)])
            }
            Self::Multitegum(mut bases) => {
                bases.push(Self::Dyad);
                Self::multitegum(bases)
            }
            _ => Self::Tegum(Box::new(self)),
        }
    }

    /// Builds a dual name from a given name.
    pub fn dual(self, abs: bool) -> Self {
        match self {
            Self::Dual(base) => {
                // Abstractly, duals of duals give back the original polytope.
                if abs {
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
            Self::Generic(_, d) => {
                if d <= 2 {
                    self
                } else {
                    Self::default()
                }
            }
            Self::Multipyramid(_) => self,
            _ => self,
        }
    }

    /// The name for an *n*-simplex.
    pub fn simplex(regular: T, n: isize) -> Self {
        match n {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Triangle(regular),
            _ => Self::Simplex(regular, n as usize),
        }
    }

    /// The name for an *n*-hypercube.
    pub fn hypercube(regular: T, n: isize) -> Self {
        match n {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => {
                if regular.is_regular() {
                    Self::Square
                } else {
                    Self::Rectangle
                }
            }
            _ => Self::Hypercube(regular, n as usize),
        }
    }

    /// The name for an *n*-orthoplex.
    pub fn orthoplex(regular: T, n: isize) -> Self {
        match n {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => {
                if regular.is_regular() {
                    Self::Square
                } else {
                    Self::Orthoplex(regular, 2)
                }
            }
            _ => Self::Orthoplex(regular, n as usize),
        }
    }

    /// Returns the name for a regular polygon of `n` sides.
    pub fn reg_polygon(n: usize) -> Self {
        match n {
            3 => Self::Triangle(T::regular(true)),
            4 => Self::Square,
            _ => Self::Generic(n, 2),
        }
    }

    /// Returns the name for a polygon (not necessarily regular) of `n` sides.
    pub fn polygon(n: usize) -> Self {
        if n == 3 {
            Self::Triangle(T::regular(false))
        } else {
            Self::Generic(n, 2)
        }
    }

    /// Sorts the bases of a multiproduct according to their rank, and then
    /// their facet count.
    fn sort_bases(bases: &mut Vec<Name<T>>) {
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

        bases.sort_unstable_by(|base0, base1| {
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
        });
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
            new_bases.push(Name::simplex(T::regular(false), pyramid_count as isize - 1));
        }

        // Sorts the bases by convention.
        Self::sort_bases(&mut new_bases);

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
            new_bases.push(Name::hypercube(T::regular(false), prism_count as isize));
        }

        // Sorts the bases by convention.
        Self::sort_bases(&mut new_bases);

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
            new_bases.push(Name::orthoplex(T::regular(false), tegum_count as isize));
        }

        // Sorts the bases by convention.
        Self::sort_bases(&mut new_bases);

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

    pub fn multicomb(bases: Vec<Name<T>>) -> Self {
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
        Self::sort_bases(&mut new_bases);

        match new_bases.len() {
            0 => Self::Point,
            1 => new_bases.swap_remove(0),
            _ => Self::Multicomb(new_bases),
        }
    }
}
