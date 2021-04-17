//! Contains the definitions of the different traits and structs for
//! [polytopes](https://polytope.miraheze.org/wiki/Polytope), as well as some
//! basic methods to operate on them.

use std::hash::Hash;

use derive_deref::{Deref, DerefMut};

use self::{geometry::Point, ranked_poset::RankVec};
pub use types::{concrete::*, r#abstract::*, renderable::*};

pub mod cd;
pub mod convex;
pub mod cox;
pub mod geometry;
pub mod group;
pub mod off;
pub mod ranked_poset;
pub mod shapes;
pub mod types;

/// The names for 0-elements, 1-elements, 2-elements, and so on.
const ELEMENT_NAMES: [&str; 11] = [
    "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta", "Xenna", "Daka",
];

/// The word "Components".
const COMPONENTS: &str = "Components";

/// The trait for methods common to all polytopes.
pub trait Polytope: Sized + Clone {
    /// The [rank](https://polytope.miraheze.org/wiki/Rank) of the polytope.
    fn rank(&self) -> isize;

    /// The number of elements of a given rank.
    fn el_count(&self, rank: isize) -> usize;

    /// The element counts of the polytope.
    fn el_counts(&self) -> RankVec<usize>;

    /// Whether the polytope is
    /// [orientable](https://polytope.miraheze.org/wiki/Orientability).
    fn orientable(&self) -> bool;

    /// Returns an instance of the
    /// [nullitope](https://polytope.miraheze.org/wiki/Nullitope), the unique
    /// polytope of rank &minus;1.
    fn nullitope() -> Self;

    /// Returns an instance of the
    /// [point](https://polytope.miraheze.org/wiki/Point), the unique polytope
    /// of rank 0.
    fn point() -> Self;

    /// Returns an instance of the
    /// [dyad](https://polytope.miraheze.org/wiki/Dyad), the unique polytope of
    /// rank 1.
    fn dyad() -> Self;

    /// Returns an instance of a
    /// [polygon](https://polytope.miraheze.org/wiki/Polygon) with a given
    /// amount of sides.
    fn polygon(n: usize) -> Self;

    /// Builds a
    /// [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product) from
    /// two polytopes.
    fn duopyramid(p: &Self, q: &Self) -> Self;

    /// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
    /// from two polytopes.
    fn duoprism(p: &Self, q: &Self) -> Self;

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum(p: &Self, q: &Self) -> Self;

    /// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
    /// from two polytopes.
    fn duocomb(p: &Self, q: &Self) -> Self;

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope.
    fn ditope(&self) -> Self;

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope in place.
    fn ditope_mut(&mut self);

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope.
    fn hosotope(&self) -> Self;

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope in place.
    fn hosotope_mut(&mut self);

    /// Builds a [pyramid](https://polytope.miraheze.org/wiki/Pyramid) from a
    /// given base.
    fn pyramid(&self) -> Self {
        Self::duopyramid(self, &Self::point())
    }

    /// Builds a [prism](https://polytope.miraheze.org/wiki/Prism) from a
    /// given base.
    fn prism(&self) -> Self {
        Self::duoprism(self, &Self::dyad())
    }

    /// Builds a [tegum](https://polytope.miraheze.org/wiki/Bipyramid) from a
    /// given base.
    fn tegum(&self) -> Self {
        Self::duotegum(self, &Self::dyad())
    }

    /// Takes the
    /// [pyramid product](https://polytope.miraheze.org/wiki/Pyramid_product) of
    /// a set of polytopes.
    fn multipyramid(factors: &[&Self]) -> Self {
        Self::multiproduct(&Self::duopyramid, factors, Self::nullitope())
    }

    /// Takes the
    /// [prism product](https://polytope.miraheze.org/wiki/Prism_product) of a
    /// set of polytopes.
    fn multiprism(factors: &[&Self]) -> Self {
        Self::multiproduct(&Self::duoprism, factors, Self::point())
    }

    /// Takes the
    /// [tegum product](https://polytope.miraheze.org/wiki/Tegum_product) of a
    /// set of polytopes.
    fn multitegum(factors: &[&Self]) -> Self {
        Self::multiproduct(&Self::duotegum, factors, Self::point())
    }

    /// Takes the
    /// [comb product](https://polytope.miraheze.org/wiki/Comb_product) of a set
    /// of polytopes.
    fn multicomb(factors: &[&Self]) -> Self {
        Self::multiproduct(&Self::duocomb, factors, Self::point())
    }

    /// Helper method for applying an associative binary function on a list of
    /// entries.
    fn multiproduct(
        product: &dyn Fn(&Self, &Self) -> Self,
        factors: &[&Self],
        identity: Self,
    ) -> Self {
        match factors.len() {
            // An empty product just evaluates to the identity element.
            0 => identity,

            // A product of one entry is just equal to the entry itself.
            1 => factors[0].clone(),

            // Evaluates larger products recursively.
            _ => {
                let (&first, factors) = factors.split_first().unwrap();

                product(first, &Self::multiproduct(&product, factors, identity))
            }
        }
    }

    fn antiprism(&self) -> Self;

    // The basic, regular polytopes.
    fn simplex(rank: isize) -> Self {
        Self::multipyramid(&vec![&Self::point(); (rank + 1) as usize])
    }

    fn hypercube(rank: isize) -> Self {
        if rank == -1 {
            Self::nullitope()
        } else {
            Self::multiprism(&vec![&Self::dyad(); rank as usize])
        }
    }

    fn orthoplex(rank: isize) -> Self {
        if rank == -1 {
            Self::nullitope()
        } else {
            Self::multitegum(&vec![&Self::dyad(); rank as usize])
        }
    }
}

/// The indices of the subelements of a polytope.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Deref, DerefMut)]
pub struct Subelements(pub Vec<usize>);

impl Subelements {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn count(count: usize) -> Self {
        let mut vec = Vec::new();

        for i in 0..count {
            vec.push(i);
        }

        Self(vec)
    }
}

/// The indices of the superelements of a polytope.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Deref, DerefMut)]
pub struct Superelements(pub Vec<usize>);

impl Superelements {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn count(count: usize) -> Self {
        let mut vec = Vec::new();

        for i in 0..count {
            vec.push(i);
        }

        Self(vec)
    }
}

/// The subelements and superlements of a polytope.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Elements {
    pub subs: Subelements,
    pub sups: Superelements,
}

impl Elements {
    /// Initializes a new element with no subelements.
    pub fn new() -> Self {
        Self::min(0)
    }

    /// Builds a minimal element for a polytope.
    pub fn min(vertex_count: usize) -> Self {
        Self {
            subs: Subelements::new(),
            sups: Superelements::count(vertex_count),
        }
    }

    /// Builds a maximal element adjacent to a given number of facets.
    pub fn max(facet_count: usize) -> Self {
        let mut subs = Subelements::with_capacity(facet_count);

        for i in 0..facet_count {
            subs.push(i);
        }

        Self {
            subs,
            sups: Superelements::new(),
        }
    }

    pub fn from_subs(subs: Subelements) -> Self {
        Self {
            subs,
            sups: Superelements(Vec::new()),
        }
    }

    pub fn swap_mut(&mut self) {
        std::mem::swap(&mut self.subs.0, &mut self.sups.0)
    }
}

/// A list of [`Elements`](Element) of the same
/// [rank](https://polytope.miraheze.org/wiki/Rank).
#[derive(Debug, Clone, Deref, DerefMut)]
pub struct ElementList(pub Vec<Elements>);

impl ElementList {
    /// Initializes an empty element list.
    pub fn new() -> Self {
        ElementList(Vec::new())
    }

    /// Initializes an empty element list with a given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        ElementList(Vec::with_capacity(capacity))
    }

    /// Returns the element list for the nullitope in a polytope with a given
    /// vertex count.
    pub fn min(vertex_count: usize) -> Self {
        Self(vec![Elements::min(vertex_count)])
    }

    /// Returns the element list for the maximal element in a polytope with a
    /// given facet count.
    pub fn max(facet_count: usize) -> Self {
        Self(vec![Elements::max(facet_count)])
    }

    /// Returns the element list for a set number of vertices in a polytope.
    /// **Does not include any superelements.**
    pub fn vertices(vertex_count: usize) -> Self {
        let mut els = ElementList::with_capacity(vertex_count);

        for _ in 0..vertex_count {
            els.push(Elements::from_subs(Subelements(vec![0])));
        }

        els
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a bunch of varied polytopes to run general tests on. Use only
    /// for tests that should work on **everything** you give it!
    fn test_polytopes() -> [Abstract; 19] {
        [
            Abstract::nullitope(),
            Abstract::point(),
            Abstract::dyad(),
            Abstract::polygon(2),
            Abstract::polygon(3),
            Abstract::polygon(4),
            Abstract::polygon(5),
            Abstract::polygon(10),
            Abstract::hypercube(3),
            Abstract::hypercube(4),
            Abstract::hypercube(5),
            Abstract::simplex(3),
            Abstract::simplex(4),
            Abstract::simplex(5),
            Abstract::orthoplex(3),
            Abstract::orthoplex(4),
            Abstract::orthoplex(5),
            Abstract::duoprism(&Abstract::polygon(6), &Abstract::polygon(7)),
            Abstract::dyad().ditope().ditope().ditope().ditope(),
        ]
    }

    #[test]
    /// Checks that a nullitope is generated correctly.
    fn nullitope_check() {
        let nullitope = Abstract::nullitope();

        assert_eq!(
            nullitope.el_counts().0,
            vec![1],
            "Nullitope element counts don't match expected value."
        );
        assert!(nullitope.full_check(), "Nullitope is invalid.");
    }

    #[test]
    /// Checks that a point is generated correctly.
    fn point_check() {
        let point = Abstract::point();

        assert_eq!(
            point.el_counts().0,
            vec![1, 1],
            "Point element counts don't match expected value."
        );
        assert!(point.full_check(), "Point is invalid.");
    }

    #[test]
    /// Checks that a dyad is generated correctly.
    fn dyad_check() {
        let dyad = Abstract::dyad();

        assert_eq!(
            dyad.el_counts().0,
            vec![1, 2, 1],
            "Dyad element counts don't match expected value."
        );
        assert!(dyad.full_check(), "Dyad is invalid.");
    }

    #[test]
    /// Checks that polygons are generated correctly.
    fn polygon_check() {
        for n in 2..=10 {
            let polygon = Abstract::polygon(n);

            assert_eq!(
                polygon.el_counts().0,
                vec![1, n, n, 1],
                "{}-gon element counts don't match expected value.",
                n
            );
            assert!(polygon.full_check(), "{}-gon is invalid.", n);
        }
    }

    #[test]
    /// Checks that polygonal duopyramids are generated correctly.
    fn duopyramid_check() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                let duopyramid = Abstract::duopyramid(&polygons[m - 2], &polygons[n - 2]);

                assert_eq!(
                    duopyramid.el_counts().0,
                    vec![
                        1,
                        m + n,
                        m + n + m * n,
                        2 * m * n + 2,
                        m + n + m * n,
                        m + n,
                        1
                    ],
                    "{}-{} duopyramid element counts don't match expected value.",
                    m,
                    n
                );
                assert!(
                    duopyramid.full_check(),
                    "{}-{} duopyramid are invalid.",
                    m,
                    n
                );
            }
        }
    }

    #[test]
    /// Checks that polygonal duoprisms are generated correctly.
    fn duoprism_check() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                let duoprism = Abstract::duoprism(&polygons[m - 2], &polygons[n - 2]);

                assert_eq!(
                    duoprism.el_counts().0,
                    vec![1, m * n, 2 * m * n, m + n + m * n, m + n, 1],
                    "{}-{} duoprism element counts don't match expected value.",
                    m,
                    n
                );
                assert!(duoprism.full_check(), "{}-{} duoprism is invalid.", m, n);
            }
        }
    }

    #[test]
    /// Checks that polygonal duotegums are generated correctly.
    fn duotegum_check() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                let duotegum = Abstract::duotegum(&polygons[m - 2], &polygons[n - 2]);

                assert_eq!(
                    duotegum.el_counts().0,
                    vec![1, m + n, m + n + m * n, 2 * m * n, m * n, 1],
                    "{}-{} duotegum element counts don't match expected value.",
                    m,
                    n
                );
                assert!(duotegum.full_check(), "{}-{} duotegum is invalid.", m, n);
            }
        }
    }

    #[test]
    /// Checks that polygonal duocombs are generated correctly.
    fn duocomb_check() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                let duocomb = Abstract::duocomb(&polygons[m - 2], &polygons[n - 2]);

                assert_eq!(
                    duocomb.el_counts().0,
                    vec![1, m * n, 2 * m * n, m * n, 1],
                    "{}-{} duocomb element counts don't match expected value.",
                    m,
                    n
                );
                assert!(duocomb.full_check(), "{}-{} duocomb is invalid.", m, n);
            }
        }
    }

    /// Calculates `n` choose `k`.
    fn choose(n: usize, k: usize) -> usize {
        let mut res = 1;

        for r in 0..k {
            res *= n - r;
            res /= r + 1;
        }

        res
    }

    #[test]
    /// Checks that simplices are generated correctly.
    fn simplex_check() {
        for n in -1..=5 {
            let simplex = Abstract::simplex(n);

            for k in -1..=n {
                assert_eq!(
                    simplex.el_count(k),
                    choose((n + 1) as usize, (k + 1) as usize),
                    "{}-simplex {}-element counts don't match up",
                    n,
                    k
                );
            }

            assert!(simplex.full_check(), "{}-simplex is invalid.", n)
        }
    }

    #[test]
    /// Checks that hypercubes are generated correctly.
    fn hypercube_check() {
        for n in -1..=5 {
            let hypercube = Abstract::hypercube(n);

            for k in 0..=n {
                assert_eq!(
                    hypercube.el_count(k),
                    choose(n as usize, k as usize) * 2u32.pow((n - k) as u32) as usize,
                    "{}-hypercube {}-element counts don't match up",
                    n,
                    k
                );
            }

            assert!(hypercube.full_check(), "{}-hypercube is invalid.", n)
        }
    }

    #[test]
    /// Checks that orthoplices are generated correctly.
    fn orthoplex_check() {
        for n in -1..=5 {
            let orthoplex = Abstract::orthoplex(n);

            for k in -1..n {
                assert_eq!(
                    orthoplex.el_count(k),
                    choose(n as usize, (k + 1) as usize) * 2u32.pow((k + 1) as u32) as usize,
                    "{}-orthoplex {}-element counts don't match up",
                    n,
                    k
                );
            }

            assert!(orthoplex.full_check(), "{}-orthoplex is invalid.", n)
        }
    }

    #[test]
    /// Checks that duals are generated correctly.
    fn dual_check() {
        let mut polytopes = test_polytopes();

        for (idx, poly) in polytopes.iter_mut().enumerate() {
            let el_counts = poly.el_counts();

            poly.dual_mut();

            // The element counts of the dual should be the same as the reversed
            // element counts of the original.
            let mut du_el_counts_rev = poly.el_counts();
            du_el_counts_rev.reverse();
            assert_eq!(
                el_counts.0, du_el_counts_rev.0,
                "Dual element counts of test polytope #{} don't match expected value.",
                idx
            );
            assert!(
                poly.full_check(),
                "Dual of test polytope #{} is invalid.",
                idx
            );
        }
    }
}
