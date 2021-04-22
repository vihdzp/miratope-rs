/// A language-independent representation of a polytope name, in a syntax
/// tree-like structure structure.
#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum Name {
    /// A nullitope.
    Nullitope,

    /// A point.
    Point,

    /// A dyad.
    Dyad,

    /// A triangle.
    Triangle,

    /// A square.
    Square,

    /// A pyramid based on some polytope. Don't instanciate this directly, use
    /// [`Name::pyramid`] instead.
    Pyramid(Box<Name>),

    /// A prism based on some polytope. Don't instanciate this directly, use
    /// [`Name::prism`] instead.
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

    /// A simplex of a given dimension, **at least 3.** Use [`Nullitope`](Name::Nullitope),
    /// [`Point`](Name::Point), [`Dyad`](Name::Dyad), or [`Triangle`](Name::Triangle)
    /// for the simplices of lower rank.
    Simplex(usize),

    /// A hypercube of a given dimension, **at least 3.** Use [`Nullitope`](Name::Nullitope),
    /// [`Point`](Name::Point), [`Dyad`](Name::Dyad), or [`Square`](Name::Square)
    /// for the simplices of lower rank.
    Hypercube(usize),

    /// An orthoplex of a given dimension, **at least 3.** Use [`Nullitope`](Name::Nullitope),
    /// [`Point`](Name::Point), [`Dyad`](Name::Dyad), or [`Square`](Name::Square)
    /// for the orthoplices of lower rank.
    Orthoplex(usize),

    /// A polytope with a given facet count and rank, in that order.
    Generic(usize, usize),

    /// The name of the polytope is unknown.
    Unknown,
}

impl Name {
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
            Name::Triangle => Some(2),
            Name::Square => Some(2),
            Name::Simplex(rank) | Name::Hypercube(rank) | Name::Orthoplex(rank) => {
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
            Name::Triangle => 3,
            Name::Square => 4,
            Name::Simplex(n) => *n + 1,
            Name::Hypercube(n) => *n * 2,
            Name::Orthoplex(n) => 2u32.pow(*n as u32) as usize,
            Name::Generic(n, _) => *n,
            _ => return None,
        })
    }

    pub fn is_valid(&self) -> bool {
        match self {
            Self::Simplex(n) | Self::Hypercube(n) | Self::Orthoplex(n) => *n >= 3,
            Self::Multicomb(bases) => !bases.is_empty(),
            _ => true,
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

    /// Builds a dual name from a given name. Assumes that the polytope is
    /// geometric, and hence that relations like the dual being an involution
    /// don't generally hold. If you want these relations to be taken into
    /// account, use [`abstract_dual`].
    pub fn dual(self) -> Self {
        match self {
            Self::Generic(_, _) => Self::Unknown,
            _ => self,
        }
    }

    /// The name for an *n*-simplex.
    pub fn simplex(n: isize) -> Self {
        match n {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Triangle,
            _ => Self::Simplex(n as usize),
        }
    }

    /// The name for an *n*-hypercube.
    pub fn hypercube(n: isize) -> Self {
        match n {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Square,
            _ => Self::Hypercube(n as usize),
        }
    }

    /// The name for an *n*-orthoplex.
    pub fn orthoplex(n: isize) -> Self {
        match n {
            -1 => Self::Nullitope,
            0 => Self::Point,
            1 => Self::Dyad,
            2 => Self::Square,
            _ => Self::Orthoplex(n as usize),
        }
    }

    /// Returns the name for a regular polygon of `n` sides.
    pub fn reg_polygon(n: usize) -> Self {
        match n {
            3 => Self::Triangle,
            4 => Self::Square,
            _ => Self::Generic(n, 2),
        }
    }

    /// Returns the name for a polygon (not necessarily regular) of `n` sides.
    pub fn polygon(n: usize) -> Self {
        if n == 3 {
            Self::Triangle
        } else {
            Self::Generic(n, 2)
        }
    }

    fn sort_bases(bases: &mut Vec<Name>) {
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

            // Names are then compared by their variant names in an arbitrary
            // manner, so that polytopes of the same type are grouped together.
            return_if_ne!(base0.cmp(&base1));

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

    pub fn multipyramid(mut bases: Vec<Name>) -> Self {
        if bases.is_empty() {
            return Self::Nullitope;
        } else if bases.len() == 1 {
            return bases.swap_remove(0);
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
                Self::Triangle => pyramid_count += 2,
                Self::Simplex(n) => pyramid_count += n + 1,
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
        Self::sort_bases(&mut new_bases);

        let multipyramid = Self::Multipyramid(new_bases);
        if pyramid_count == 1 {
            multipyramid.pyramid()
        } else {
            multipyramid
        }
    }

    pub fn multiprism(mut bases: Vec<Name>) -> Self {
        if bases.is_empty() {
            return Self::Point;
        } else if bases.len() == 1 {
            return bases.swap_remove(0);
        }

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
                Self::Square => prism_count += 2,
                Self::Hypercube(n) => prism_count += n,
                Self::Multipyramid(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one pyramid, we combine all of them into a
        // single simplex.
        if prism_count >= 2 {
            new_bases.push(Name::hypercube(prism_count as isize));
        }

        // Sorts the bases by convention.
        Self::sort_bases(&mut new_bases);

        let multiprism = Self::Multiprism(new_bases);
        if prism_count == 1 {
            multiprism.prism()
        } else {
            multiprism
        }
    }

    pub fn multitegum(mut bases: Vec<Name>) -> Self {
        if bases.is_empty() {
            return Self::Point;
        } else if bases.len() == 1 {
            return bases.swap_remove(0);
        }

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
                Self::Hypercube(n) => tegum_count += n,
                Self::Multipyramid(mut extra_bases) => new_bases.append(&mut extra_bases),
                _ => new_bases.push(base),
            }
        }

        // If we're taking more than one pyramid, we combine all of them into a
        // single simplex.
        if tegum_count >= 2 {
            new_bases.push(Name::orthoplex(tegum_count as isize));
        }

        // Sorts the bases by convention.
        Self::sort_bases(&mut new_bases);

        let multitegum = Self::Multitegum(new_bases);
        if tegum_count == 1 {
            multitegum.tegum()
        } else {
            multitegum
        }
    }

    pub fn multicomb(mut bases: Vec<Name>) -> Self {
        if bases.is_empty() {
            return Self::Nullitope;
        } else if bases.len() == 1 {
            return bases.swap_remove(0);
        }

        // Sorts the bases by convention.
        Self::sort_bases(&mut bases);

        Self::Multicomb(bases)
    }
}
