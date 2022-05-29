#![deny(
    missing_docs,
    nonstandard_style,
    unused_parens,
    unused_qualifications,
    rust_2018_idioms,
    rust_2018_compatibility,
    future_incompatible,
    missing_copy_implementations
)]
/*
#![warn(
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    missing_docs
)]*/

//! This is the main dependency of
//! [Miratope](https://github.com/vihdzp/miratope-rs). It contains all code
//! to build and name [`Abstract`] and [`Concrete`](conc::Concrete) polytopes
//! alike.
//!
//! If you're interested in actually rendering polytopes, you might want to take
//! a look at the [`miratope`](https://crates.io/crates/miratope) crate instead.

pub mod abs;
pub mod conc;
pub mod cox;
pub mod file;
pub mod float;
pub mod geometry;
pub mod group;

use std::{collections::HashSet, error::Error, iter, ops::IndexMut};

use abs::{
    flag::{Flag, FlagIter, OrientedFlag, OrientedFlagIter},
    ranked::Ranks,
    Abstract, Element, ElementList, ElementMap, Ranked,
};

use vec_like::VecLike;

/// The names for 0-elements, 1-elements, 2-elements, and so on.
const ELEMENT_NAMES: [&str; 12] = [
    "", "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta", "Xenna",
    "Daka",
];

/// The word "Components".
const COMPONENTS: &str = "Components";

/// Represents an error in a concrete dual, in which a facet with a given index
/// passes through the inversion center.
#[derive(Clone, Copy, Debug)]
pub struct DualError(usize);

impl std::fmt::Display for DualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "facet {} passes through inversion center", self.0)
    }
}

impl Error for DualError {}

/// Gets the precalculated value for n!.
fn factorial(n: usize) -> u32 {
    /// Precalculated factorials from 0! to 13!.
    const FACTORIALS: [u32; 13] = [
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600,
    ];

    FACTORIALS[n]
}

/// The trait for methods common to all polytopes.
///
/// Some of these methods must be implemented manually. Others are just
/// shorthands for calling methods on the contained [`Abstract`] polytopes.
pub trait Polytope:
    Clone + IndexMut<usize, Output = ElementList> + IndexMut<(usize, usize), Output = Element>
{
    /// The error type of taking a dual.
    type DualError: Error;

    /// Returns a reference to the underlying abstract polytope.
    fn abs(&self) -> &Abstract;

    /// Returns a mutable reference to the underlying abstract polytope.
    fn abs_mut(&mut self) -> &mut Abstract;

    /// Returns the underlying abstract polytope.
    fn into_abs(self) -> Abstract;

    /// Returns a mutable reference to the [`Ranks`] of the polytope.
    ///
    /// # Safety
    /// The user must certify that any modification done to the polytope
    /// ultimately results in a valid [`Abstract`].
    unsafe fn ranks_mut(&mut self) -> &mut Ranks {
        self.abs_mut().ranks_mut()
    }

    /// Sorts the subelements and superelements of the entire polytope. This is
    /// usually called before iterating over the flags of the polytope.
    ///
    /// This will do nothing if the polytope is already sorted.
    fn element_sort(&mut self) {
        if !self.abs().sorted() {
            // Safety: changing the order of the indices in an element does not
            // change whether the polytope is valid.
            unsafe {
                self.ranks_mut().element_sort();
                self.abs_mut().set_sorted(true);
            }
        }
    }

    /// Returns an instance of the
    /// [nullitope](https://polytope.miraheze.org/wiki/Nullitope), the unique
    /// polytope of rank &minus;1.
    fn nullitope() -> Self;

    /// Returns whether this is a nullitope.
    fn is_nullitope(&self) -> bool {
        self.rank() == 0
    }

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

    /// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// from two polytopes.
    fn duopyramid(&self, p: &Self) -> Self;

    /// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
    /// from two polytopes.
    fn duoprism(&self, p: &Self) -> Self;

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum(&self, p: &Self) -> Self;

    /// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
    /// from two polytopes.
    fn duocomb(&self, p: &Self) -> Self;

    /// Builds a [pyramid](https://polytope.miraheze.org/wiki/Pyramid) from a
    /// given base.
    fn pyramid(&self) -> Self {
        self.duopyramid(&Self::point())
    }

    /// Builds a [prism](https://polytope.miraheze.org/wiki/Prism) from a
    /// given base.
    fn prism(&self) -> Self {
        self.duoprism(&Self::dyad())
    }

    /// Builds a [tegum](https://polytope.miraheze.org/wiki/Bipyramid) from a
    /// given base.
    fn tegum(&self) -> Self {
        self.duotegum(&Self::dyad())
    }

    /// Takes the [pyramid product](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// of an iterator over polytopes.
    fn multipyramid<'a, I: IntoIterator<Item = &'a Self>>(factors: I) -> Self
    where
        Self: 'a,
    {
        multiproduct(factors, Self::duopyramid, Self::nullitope)
    }

    /// Takes the [prism product](https://polytope.miraheze.org/wiki/Prism_product)
    /// of an iterator over polytopes.
    fn multiprism<'a, I: IntoIterator<Item = &'a Self>>(factors: I) -> Self
    where
        Self: 'a,
    {
        multiproduct(factors, Self::duoprism, Self::point)
    }

    /// Takes the [tegum product](https://polytope.miraheze.org/wiki/Tegum_product)
    /// of an iterator over polytopes.
    fn multitegum<'a, I: IntoIterator<Item = &'a Self>>(factors: I) -> Self
    where
        Self: 'a,
    {
        multiproduct(factors, Self::duotegum, Self::point)
    }

    /// Takes the [comb product](https://polytope.miraheze.org/wiki/Comb_product)
    /// of an iterator over polytopes.
    fn multicomb<'a, I: IntoIterator<Item = &'a Self>>(factors: I) -> Self
    where
        Self: 'a,
    {
        multiproduct(factors, Self::duocomb, Self::nullitope)
    }

    /// Builds a [simplex](https://polytope.miraheze.org/wiki/Simplex) with a
    /// given rank.
    fn simplex(rank: usize) -> Self {
        Self::multipyramid(iter::repeat(&Self::point()).take(rank))
    }

    /// Builds a regular [tetrahedron](https://polytope.miraheze.org/wiki/Tetrahedron).
    fn tetrahedron() -> Self {
        Self::simplex(4)
    }

    /// Builds a [hypercube](https://polytope.miraheze.org/wiki/Hypercube) with
    /// a given rank.
    fn hypercube(rank: usize) -> Self {
        if rank == 0 {
            Self::nullitope()
        } else {
            Self::multiprism(iter::repeat(&Self::dyad()).take(rank - 1))
        }
    }

    /// Builds a regular [cube](https://polytope.miraheze.org/wiki/Cube).
    fn cube() -> Self {
        Self::hypercube(4)
    }

    /// Builds an [orthoplex](https://polytope.miraheze.org/wiki/Orthoplex) with
    /// a given rank.
    fn orthoplex(rank: usize) -> Self;

    /// Builds a regular [octahedron](https://polytope.miraheze.org/wiki/Octahedron).
    fn octahedron() -> Self {
        Self::orthoplex(4)
    }

    /// Returns the dual of a polytope. Never fails for an abstract polytope. In
    /// case of failing on a concrete polytope, returns the index of a facet
    /// through the inversion center.
    fn try_dual(&self) -> Result<Self, Self::DualError>;

    /// Builds the dual of a polytope in place. Never fails for an abstract
    /// polytope. In case of failing on a concrete polytope, returns the index
    /// of a facet through the inversion center and does nothing.
    fn try_dual_mut(&mut self) -> Result<(), Self::DualError>;

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks.
    fn comp_append(&mut self, p: Self);

    /// Returns a map from the elements in a polytope to the index of one of its
    /// vertices. Does not map the minimal element anywhere.
    fn vertex_map(&self) -> ElementMap<usize> {
        self.abs().vertex_map()
    }

    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, rank: usize, idx: usize) -> Option<Self>;

    /// Gets the element figure with a given rank and index as a polytope.
    fn element_fig(&self, rank: usize, idx: usize) -> Result<Option<Self>, Self::DualError>;

    /// Gets the section defined by two elements with given ranks and indices as
    /// a polytope, or returns `None` in case no section is defined by these
    /// elements.
    fn section(
        &self,
        lo_rank: usize,
        lo_idx: usize,
        hi_rank: usize,
        hi_idx: usize,
    ) -> Result<Option<Self>, Self::DualError> {
        if let Some(el) = self.element(hi_rank, hi_idx) {
            el.element_fig(lo_rank, lo_idx)
        } else {
            Ok(None)
        }
    }

    /// Gets the facet associated to the element of a given index as a polytope.
    fn facet(&self, idx: usize) -> Option<Self> {
        let r = self.rank();
        (r != 0).then(|| self.element(r - 1, idx)).flatten()
    }

    /// Gets the verf associated to the element of a given index as a polytope.
    fn verf(&self, idx: usize) -> Result<Option<Self>, Self::DualError> {
        self.element_fig(1, idx)
    }

    /// Builds a compound polytope from an iterator over components.
    fn compound<U: Iterator<Item = Self>>(mut components: U) -> Self {
        if let Some(mut p) = components.next() {
            for q in components {
                p.comp_append(q);
            }
            p
        } else {
            Self::nullitope()
        }
    }

    /// Makes a polytope strongly connected. Splits compounds into their components.
    fn defiss(&self) -> Vec<Self>;

    /// Builds a Petrial in place. Returns `true` if successful. Does not modify
    /// the original polytope otherwise.
    fn petrial_mut(&mut self) -> bool;

    /// Builds the Petrial of a polytope. Returns `None` if the polytope is not
    /// 3D, or if its Petrial is not a valid polytope.
    fn petrial(&self) -> Option<Self> {
        let mut clone = self.clone();
        clone.petrial_mut().then(|| clone)
    }

    /// Returns the indices of the vertices of a Petrie polygon in cyclic
    /// order, or `None` if it self-intersects.
    ///
    /// # Panics
    /// Panics if the polytope is not sorted.
    fn petrie_polygon_vertices(&self, flag: Flag) -> Option<Vec<usize>> {
        let rank = self.rank();
        let mut new_flag = flag.clone();
        let first_vertex = flag[1];
        let mut vertices = Vec::new();
        let mut vertex_hash = HashSet::new();

        assert!(self.abs().sorted());

        loop {
            // Applies 1-changes up to (rank-1)-changes in order.
            for idx in 1..rank {
                new_flag.change_mut(self.abs(), idx);
            }

            // If we just hit a previous vertex, we return.
            let new_vertex = new_flag[1];
            if vertex_hash.contains(&new_vertex) {
                return None;
            }

            // Adds the new vertex.
            vertices.push(new_vertex);
            vertex_hash.insert(new_vertex);

            // If we're back to the beginning, we break out of the loop.
            if new_vertex == first_vertex {
                break;
            }
        }

        // We returned to precisely the initial flag.
        (flag == new_flag).then(|| vertices)
    }

    /// Builds a Petrie polygon from a given flag of the polytope. Returns
    /// `None` if this Petrie polygon is invalid.
    fn petrie_polygon_with(&mut self, flag: Flag) -> Option<Self>;

    /// Returns the first [`Flag`] of a polytope. This is the flag built when we
    /// start at the maximal element and repeatedly take the first subelement.
    fn first_flag(&self) -> Flag {
        let rank = self.rank();
        let mut flag = Flag::with_capacity(rank + 1);
        let mut idx = 0;
        flag.push(0);

        for r in (1..=rank).rev() {
            idx = self[(r, idx)].subs[0];
            flag.push(idx);
        }

        flag.reverse();

        flag
    }

    /// Returns the first [`OrientedFlag`] of a polytope. This is the flag built
    /// when we start at the maximal element and repeatedly take the first
    /// subelement.
    fn first_oriented_flag(&self) -> OrientedFlag {
        self.first_flag().into()
    }

    /// Returns an iterator over all [`Flag`]s of a polytope.
    fn flags(&self) -> FlagIter<'_> {
        FlagIter::new(self.abs())
    }

    /// Returns an iterator over all [`OrientedFlag`]s of a polytope.
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
    fn flag_events(&self) -> OrientedFlagIter<'_> {
        OrientedFlagIter::new(self.abs())
    }

    /// Returns the omnitruncate of a polytope.
    fn omnitruncate(&self) -> Self;

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope.
    fn ditope(&self) -> Self {
        let mut clone = self.clone();
        clone.ditope_mut();
        clone
    }

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope in place.
    fn ditope_mut(&mut self);

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope.
    fn hosotope(&self) -> Self {
        let mut clone = self.clone();
        clone.hosotope_mut();
        clone
    }

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope in place.
    fn hosotope_mut(&mut self);

    /// Attempts to build an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. If it fails, it returns the index of a facet
    /// through the inversion center.
    fn try_antiprism(&self) -> Result<Self, Self::DualError>;

    /// Splits compound faces into their components.
    fn untangle_faces(&mut self);

    /// Determines whether a given polytope is
    /// [orientable](https://polytope.miraheze.org/wiki/Orientability).
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
    fn orientable(&self) -> bool {
        self.flag_events().all(|f| f.orientable())
    }

    /// Determines whether a given polytope is
    /// [orientable](https://polytope.miraheze.org/wiki/Orientability).
    fn orientable_mut(&mut self) -> bool {
        self.element_sort();
        self.orientable()
    }
}

/// Returns the multiproduct (i.e. multipyramid, multiprism, multitegum, or
/// multicomb) of an iterator over polytopes. This is built by repeatedly
/// applying the product on two polytopes. In case the iterator is empty, this
/// builds a polytope using the specified default function.
fn multiproduct<'a, T, I, P, D>(factors: I, product: P, default: D) -> T
where
    T: Clone + 'a,
    I: IntoIterator<Item = &'a T>,
    P: Fn(&T, &T) -> T,
    D: Fn() -> T,
{
    let mut factors = factors.into_iter();

    match (factors.next(), factors.next()) {
        (None, _) => default(),
        (Some(factor), None) => factor.clone(),
        (Some(first), Some(second)) => factors.fold(product(first, second), |p, q| product(&p, q)),
    }
}

/// Tests whether a polytope's element counts match the expected element counts,
/// and whether a polytope is valid.
#[cfg(test)]
pub(crate) fn test<T: Polytope, I: IntoIterator<Item = usize>>(poly: &T, element_counts: I) {
    let mut element_counts = element_counts.into_iter();
    for r in 0..=poly.rank() {
        assert_eq!(
            poly.el_count(r),
            element_counts.next().unwrap_or_default(),
            "element count mismatch at rank {}",
            r
        )
    }

    poly.assert_valid();
}
