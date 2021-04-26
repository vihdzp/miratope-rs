//! Contains the definitions of the different traits and structs for
//! [polytopes](https://polytope.miraheze.org/wiki/Polytope), as well as some
//! basic methods to operate on them.

use crate::{
    lang::{name::NameType, Name},
    polytope::flag::FlagIter,
};
use std::hash::Hash;

use derive_deref::{Deref, DerefMut};

use self::{
    flag::{Flag, FlagEvent},
    geometry::Point,
    rank::RankVec,
};
pub use types::{concrete::*, r#abstract::*, renderable::*};

pub mod cd;
pub mod convex;
pub mod cox;
pub mod flag;
pub mod geometry;
pub mod ggb;
pub mod group;
pub mod off;
pub mod rank;
pub mod types;

/// The names for 0-elements, 1-elements, 2-elements, and so on.
const ELEMENT_NAMES: [&str; 11] = [
    "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta", "Xenna", "Daka",
];

/// The word "Components".
const COMPONENTS: &str = "Components";

/// The trait for methods common to all polytopes.
pub trait Polytope<T: NameType>: Sized + Clone {
    /// The [rank](https://polytope.miraheze.org/wiki/Rank) of the polytope.
    fn rank(&self) -> isize;

    fn name(&self) -> &Name<T>;

    fn name_mut(&mut self) -> &mut Name<T>;

    fn with_name(mut self, name: Name<T>) -> Self {
        *self.name_mut() = name;
        self
    }

    /// The number of elements of a given rank.
    fn el_count(&self, rank: isize) -> usize;

    /// The element counts of the polytope.
    fn el_counts(&self) -> RankVec<usize>;

    fn vertex_count(&self) -> usize {
        self.el_count(0)
    }

    fn facet_count(&self) -> usize {
        self.el_count(self.rank() - 1)
    }

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

    /// Returns an instance of a [polygon](https://polytope.miraheze.org/wiki/Polygon)
    /// with a given number of sides.
    fn polygon(n: usize) -> Self;

    /// Returns the dual of a polytope.
    fn _dual(&self) -> Option<Self>;

    /// Builds the dual of a polytope in place.
    fn _dual_mut(&mut self) -> Result<(), ()>;

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks.
    fn append(&mut self, p: Self) -> Result<(), ()>;

    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, rank: isize, idx: usize) -> Option<Self>;

    /// Gets the element figure with a given rank and index as a polytope.
    fn element_fig(&self, rank: isize, idx: usize) -> Option<Self> {
        let mut element_fig = self._dual()?.element(self.rank() - rank - 1, idx)?;

        if element_fig._dual_mut().is_ok() {
            Some(element_fig)
        } else {
            None
        }
    }

    /// Gets the section defined by two elements with given ranks and indices as
    /// a polytope, or returns `None` in case no section is defined by these
    /// elements.
    fn section(
        &self,
        rank_lo: isize,
        idx_lo: usize,
        rank_hi: isize,
        idx_hi: usize,
    ) -> Option<Self> {
        self.element(rank_hi, idx_hi)?.element_fig(rank_lo, idx_lo)
    }

    fn facet(&self, idx: usize) -> Option<Self> {
        self.element(self.rank() - 1, idx)
    }

    fn verf(&self, idx: usize) -> Option<Self> {
        self.element_fig(0, idx)
    }

    /// Builds a compound polytope from a set of components.
    fn compound(components: Vec<Self>) -> Option<Self> {
        Self::compound_iter(components.into_iter())
    }

    /// Builds a compound polytope from an iterator over components.
    fn compound_iter<U: Iterator<Item = Self>>(mut components: U) -> Option<Self> {
        Some(if let Some(mut p) = components.next() {
            for q in components {
                if p.append(q).is_err() {
                    return None;
                }
            }

            p
        } else {
            Self::nullitope()
        })
    }

    /// Returns an iterator over all "flag events" of a polytope. For more info,
    /// see [`FlagIter`].
    fn flag_events(&self) -> FlagIter;

    /// Returns an iterator over all flags of a polytope.
    fn flags(&self) -> Box<dyn Iterator<Item = Flag>> {
        Box::new(
            self.flag_events()
                .filter(|event| event.is_flag())
                .map(|event| {
                    if let FlagEvent::Flag(flag) = event {
                        flag
                    } else {
                        panic!("Non-flag somehow slipped through!")
                    }
                }),
        )
    }

    /// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// from two polytopes.
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

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope.
    fn antiprism(&self) -> Self;

    /// Determines whether a given polytope is
    /// [orientable](https://polytope.miraheze.org/wiki/Orientability).
    fn orientable(&self) -> bool;

    /// Builds a [pyramid](https://polytope.miraheze.org/wiki/Pyramid) from a
    /// given base.
    fn pyramid(&self) -> Self {
        Self::duopyramid(self, &Self::point()).with_name(self.name().clone().pyramid())
    }

    /// Builds a [prism](https://polytope.miraheze.org/wiki/Prism) from a
    /// given base.
    fn prism(&self) -> Self {
        Self::duoprism(self, &Self::dyad()).with_name(self.name().clone().prism())
    }

    /// Builds a [tegum](https://polytope.miraheze.org/wiki/Bipyramid) from a
    /// given base.
    fn tegum(&self) -> Self {
        Self::duotegum(self, &Self::dyad()).with_name(self.name().clone().tegum())
    }

    /// Takes the [pyramid product](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// of a set of polytopes.
    fn multipyramid(factors: &[&Self]) -> Self {
        Self::multipyramid_iter(factors.iter().copied())
    }

    /// Takes the [pyramid product](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// of an iterator over polytopes.
    fn multipyramid_iter<'a, U: Iterator<Item = &'a Self>>(factors: U) -> Self
    where
        Self: 'a,
    {
        factors.fold(Self::nullitope(), |p, q| Self::duopyramid(&p, q))
    }

    /// Takes the [prism product](https://polytope.miraheze.org/wiki/Prism_product)
    /// of a set of polytopes.
    fn multiprism(factors: &[&Self]) -> Self {
        Self::multiprism_iter(factors.iter().copied())
    }

    /// Takes the [prism product](https://polytope.miraheze.org/wiki/Prism_product)
    /// of an iterator over polytopes.
    fn multiprism_iter<'a, U: Iterator<Item = &'a Self>>(factors: U) -> Self
    where
        Self: 'a,
    {
        factors.fold(Self::point(), |p, q| Self::duoprism(&p, q))
    }

    /// Takes the [tegum product](https://polytope.miraheze.org/wiki/Tegum_product)
    /// of a set of polytopes.
    fn multitegum(factors: &[&Self]) -> Self {
        Self::multitegum_iter(factors.iter().copied())
    }

    /// Takes the [tegum product](https://polytope.miraheze.org/wiki/Tegum_product)
    /// of an iterator over polytopes.
    fn multitegum_iter<'a, U: Iterator<Item = &'a Self>>(factors: U) -> Self
    where
        Self: 'a,
    {
        factors.fold(Self::point(), |p, q| Self::duotegum(&p, q))
    }

    /// Takes the [comb product](https://polytope.miraheze.org/wiki/Comb_product)
    /// of a set of polytopes.
    fn multicomb(factors: &[&Self]) -> Self {
        Self::multicomb_iter(factors.iter().copied())
    }

    /// Takes the [comb product](https://polytope.miraheze.org/wiki/Comb_product)
    /// of an iterator over polytopes.
    fn multicomb_iter<'a, U: Iterator<Item = &'a Self>>(mut factors: U) -> Self
    where
        Self: 'a,
    {
        if let Some(init) = factors.next().cloned() {
            factors.fold(init, |p, q| Self::duocomb(&p, q))
        }
        // There's no sensible way to take an empty comb product, so we just
        // make it a nullitope for simplicity.
        else {
            Self::nullitope()
        }
    }

    /// Builds a [simplex](https://polytope.miraheze.org/wiki/Simplex) with a
    /// given rank.
    fn simplex(rank: isize) -> Self {
        Self::multipyramid(&vec![&Self::point(); (rank + 1) as usize])
            .with_name(Name::simplex(T::regular(true), rank))
    }

    /// Builds a [hypercube](https://polytope.miraheze.org/wiki/Hypercube) with
    /// a given rank.
    fn hypercube(rank: isize) -> Self {
        if rank == -1 {
            Self::nullitope()
        } else {
            Self::multiprism(&vec![&Self::dyad(); rank as usize])
                .with_name(Name::hypercube(T::regular(true), rank))
        }
    }

    /// Builds an [orthoplex](https://polytope.miraheze.org/wiki/Orthoplex) with
    /// a given rank.
    fn orthoplex(rank: isize) -> Self {
        if rank == -1 {
            Self::nullitope()
        } else {
            Self::multitegum(&vec![&Self::dyad(); rank as usize])
                .with_name(Name::orthoplex(T::regular(true), rank))
        }
    }
}

/// Common boilerplate code for subelements and superelements.
trait Subsupelements: Sized {
    /// Builds a list of either subelements or superelements from a vector.
    fn from_vec(vec: Vec<usize>) -> Self;

    /// Constructs a new, empty subelement or superelement list.
    fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    /// Constructs a new, empty subelement list with the capacity to store
    /// elements up to the specified rank.
    fn with_capacity(rank: usize) -> Self {
        Self::from_vec(Vec::with_capacity(rank))
    }

    /// Constructs a subelement list consisting of the indices from `0` to
    /// `count`.
    fn count(count: usize) -> Self {
        let mut vec = Vec::new();

        for i in 0..count {
            vec.push(i);
        }

        Self::from_vec(vec)
    }
}

/// The indices of the subelements of a polytope.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Deref, DerefMut)]
pub struct Subelements(pub Vec<usize>);

impl Subsupelements for Subelements {
    /// Builds a list of subelements from a vector.
    fn from_vec(vec: Vec<usize>) -> Self {
        Self(vec)
    }
}

/// The indices of the superelements of a polytope.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Deref, DerefMut)]
pub struct Superelements(pub Vec<usize>);

impl Subsupelements for Superelements {
    /// Builds a list of superelements from a vector.
    fn from_vec(vec: Vec<usize>) -> Self {
        Self(vec)
    }
}

/// An element in a polytope, which stores the indices of both its subelements
/// and superlements.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Element {
    /// The indices of the subelements of the element.
    pub subs: Subelements,

    /// The indices of the superelements of the element.
    pub sups: Superelements,
}

impl Element {
    /// Initializes a new element with no subelements and no superelements.
    pub fn new() -> Self {
        Self {
            subs: Subelements::new(),
            sups: Superelements::new(),
        }
    }

    /// Builds a minimal element adjacent to a given amount of vertices.
    pub fn min(vertex_count: usize) -> Self {
        Self {
            subs: Subelements::new(),
            sups: Superelements::count(vertex_count),
        }
    }

    /// Builds a maximal element adjacent to a given number of facets.
    pub fn max(facet_count: usize) -> Self {
        Self {
            subs: Subelements::count(facet_count),
            sups: Superelements::new(),
        }
    }

    /// Builds an element from a given set of subelements and an empty
    /// superelement list.
    pub fn from_subs(subs: Subelements) -> Self {
        Self {
            subs,
            sups: Superelements::new(),
        }
    }

    /// Swaps the subelements and superelements of the element.
    pub fn swap_mut(&mut self) {
        std::mem::swap(&mut self.subs.0, &mut self.sups.0)
    }

    /// Sorts the subelements and superelements by index.
    pub fn sort(&mut self) {
        self.subs.sort_unstable();
        self.sups.sort_unstable();
    }
}

/// A list of [`Elements`](Element) of the same
/// [rank](https://polytope.miraheze.org/wiki/Rank).
#[derive(Debug, Clone, Deref, DerefMut)]
pub struct ElementList(pub Vec<Element>);

impl ElementList {
    /// Initializes an empty element list.
    pub fn new() -> Self {
        ElementList(Vec::new())
    }

    /// Initializes an empty element list with a given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        ElementList(Vec::with_capacity(capacity))
    }

    /// Returns an element list with a single, empty element. Often used as the
    /// element list for the nullitopes when a polytope is built in layers.
    pub fn single() -> Self {
        Self(vec![Element::new()])
    }

    /// Returns the element list for the nullitope in a polytope with a given
    /// vertex count.
    pub fn min(vertex_count: usize) -> Self {
        Self(vec![Element::min(vertex_count)])
    }

    /// Returns the element list for the maximal element in a polytope with a
    /// given facet count.
    pub fn max(facet_count: usize) -> Self {
        Self(vec![Element::max(facet_count)])
    }

    /// Returns the element list for a set number of vertices in a polytope.
    /// **Does not include any superelements.**
    pub fn vertices(vertex_count: usize) -> Self {
        let mut els = ElementList::with_capacity(vertex_count);

        for _ in 0..vertex_count {
            els.push(Element::from_subs(Subelements(vec![0])));
        }

        els
    }

    pub fn into_iter(self) -> std::vec::IntoIter<Element> {
        self.0.into_iter()
    }
}

impl IntoIterator for ElementList {
    type Item = Element;

    type IntoIter = std::vec::IntoIter<Element>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// Maybe move these tests to the individual files?
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
                    "{}-{} duopyramid is invalid.",
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
