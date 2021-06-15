//! A bunch of wrappers and helper structs that make dealing with abstract
//! polytopes much less confusing.

use crate::{impl_veclike, vec_like::VecLike};
use std::{collections::HashMap, iter::IntoIterator};

use super::{
    rank::{Rank, RankVec},
    Abstract,
};
use crate::polytope::Polytope;

/// A bundled rank and index, which can be used to refer to an element in an
/// abstract polytope.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ElementRef {
    /// The rank of the element.
    pub rank: Rank,

    /// The index of the element in its corresponding element list.
    pub idx: usize,
}

impl std::fmt::Display for ElementRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "element at rank {}, index {}", self.rank, self.idx)
    }
}

impl ElementRef {
    /// Creates a new element reference with a given rank and index.
    pub fn new(rank: Rank, idx: usize) -> Self {
        Self { rank, idx }
    }
}

/// Common boilerplate code for [`Subelements`] and [`Superelements`].
pub trait Subsupelements<'a>: Sized + VecLike<'a, VecItem = usize> {
    /// Constructs a subelement or superelement list consisting of the indices
    /// from `0` to `n - 1`.
    fn count(n: usize) -> Self {
        let mut vec = Vec::new();

        for i in 0..n {
            vec.push(i);
        }

        vec.into()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Subelements(pub Vec<usize>);
impl_veclike!(Subelements, usize, usize);
impl<'a> Subsupelements<'a> for Subelements {}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Superelements(pub Vec<usize>);
impl_veclike!(Superelements, usize, usize);
impl<'a> Subsupelements<'a> for Superelements {}

/// An element in a polytope, which stores the indices of both its subelements
/// and superlements. These make up the entries of an [`ElementList`].
#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub struct Element {
    /// The indices of the subelements of the previous rank.
    pub subs: Subelements,

    /// The indices of the superelements of the next rank.
    pub sups: Superelements,
}

impl Element {
    /// Initializes a new element with no subelements and no superelements.
    pub fn new() -> Self {
        Default::default()
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

    /// Sorts both the subelements and superelements by index.
    pub fn sort(&mut self) {
        self.subs.sort_unstable();
        self.sups.sort_unstable();
    }
}

/// A list of [`Elements`](Element) of the same
/// [rank](https://polytope.miraheze.org/wiki/Rank).
///
/// If you only want to deal with the subelements of a polytope, use
/// [`SubelementList`] instead.
#[derive(Debug, Clone)]
pub struct ElementList(Vec<Element>);
impl_veclike!(ElementList, Element, usize);

impl ElementList {
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
}

/// A list of subelements in a polytope.
#[derive(Debug)]
pub struct SubelementList(Vec<Subelements>);
impl_veclike!(SubelementList, Subelements, usize);

impl SubelementList {
    /// Returns the subelement list for the minimal element in a polytope.
    pub fn min() -> Self {
        Self(vec![Subelements::new()])
    }

    /// Returns the subelement list for a set number of vertices in a polytope.
    pub fn vertices(vertex_count: usize) -> Self {
        let mut els = SubelementList::with_capacity(vertex_count);

        for _ in 0..vertex_count {
            els.push(Subelements(vec![0]));
        }

        els
    }

    /// Returns the subelement list for the maximal element in a polytope with a
    /// given facet count.
    pub fn max(facet_count: usize) -> Self {
        Self(vec![Subelements::count(facet_count)])
    }
}

/// A structure used to build a polytope from the bottom up.
pub struct AbstractBuilder(Abstract);

impl AbstractBuilder {
    pub fn new() -> Self {
        Self(Abstract::new())
    }

    pub fn with_capacity(rank: Rank) -> Self {
        Self(Abstract::with_rank_capacity(rank))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional)
    }

    pub fn push(&mut self, subelements: SubelementList) {
        self.0.push_subs(subelements)
    }

    /// Pushes an element list with a single empty element into the polytope.
    pub fn push_min(&mut self) {
        // If you're using this method, the polytope should be empty.
        debug_assert!(self.is_empty());
        self.push(SubelementList::min());
    }

    ///  To be
    /// used in circumstances where the elements are built up in layers.
    pub fn push_vertices(&mut self, vertex_count: usize) {
        // If you're using this method, the polytope should consist of a single
        // minimal element.
        debug_assert_eq!(self.0.rank(), Rank::new(-1));
        self.push(SubelementList::vertices(vertex_count))
    }

    pub fn push_max(&mut self) {
        self.0.push_max();
    }

    pub fn build(self) -> Abstract {
        self.0
    }
}

/// Maps each recursive subelement of an abstract polytope's element to a
/// `usize`, representing its index in a new polytope. This is used to build the
/// elements of polytopes (as polytopes), or to find their vertices.
pub struct ElementHash(RankVec<HashMap<usize, usize>>);

impl ElementHash {
    /// Returns a map from elements on a polytope to elements on a new polytope
    /// representing a particular element (as a polytope). If the element
    /// doesn't exist, we return `None`.
    pub fn new(poly: &Abstract, el: ElementRef) -> Option<Self> {
        poly.get_element(el)?;

        // A vector of HashMaps. The k-th entry is a map from k-elements of the
        // original polytope into k-elements in a new polytope.
        let mut hashes = RankVec::with_rank_capacity(el.rank);
        for _ in Rank::range_inclusive_iter(-1, el.rank) {
            hashes.push(HashMap::new());
        }
        hashes[el.rank].insert(el.idx, 0);

        // Gets subindices of subindices, until reaching the vertices.
        for r in Rank::range_inclusive_iter(0, el.rank).rev() {
            let (left_slice, right_slice) = hashes.split_at_mut(r);
            let prev_hash = left_slice.last_mut().unwrap();
            let hash = right_slice.first().unwrap();

            for &idx in hash.keys() {
                for &sub in &poly[r][idx].subs {
                    let len = prev_hash.len();
                    prev_hash.entry(sub).or_insert(len);
                }
            }
        }

        Some(Self(hashes))
    }

    /// Gets the `HashMap` corresponding to elements of a given rank.
    fn get(&self, idx: Rank) -> Option<&HashMap<usize, usize>> {
        self.0.get(idx)
    }

    /// Gets the indices of the elements of a given rank in the original
    /// polytope.
    fn to_elements(&self, rank: Rank) -> Vec<usize> {
        if let Some(elements) = self.get(rank) {
            let mut new_elements = Vec::new();
            new_elements.resize(elements.len(), 0);

            for (&sub, &idx) in elements {
                new_elements[idx] = sub;
            }

            new_elements
        } else {
            Vec::new()
        }
    }

    /// Gets the indices of the vertices in the original polytope.
    pub fn to_vertices(&self) -> Vec<usize> {
        self.to_elements(Rank::new(0))
    }

    /// Gets the indices of the vertices of a given element in a polytope.
    pub fn to_polytope(&self, poly: &Abstract) -> Abstract {
        let rank = self.0.rank();
        let mut abs = Abstract::with_rank_capacity(rank);

        // For every rank stored in the element map.
        for r in Rank::range_inclusive_iter(-1, rank) {
            let mut elements = ElementList::new();
            let hash = &self.0[r];

            for _ in 0..hash.len() {
                elements.push(Element::new());
            }

            // For every element of rank r in the hash element list.
            for (&idx, &new_idx) in hash {
                // We take the corresponding element in the original polytope
                // and use the hash map to get its sub and superelements in the
                // new polytope.
                let el = poly.get_element(ElementRef::new(r, idx)).unwrap();
                let mut new_el = Element::new();

                // Gets the subelements.
                if let Some(r_minus_one) = r.try_minus_one() {
                    if let Some(prev_hash) = self.get(r_minus_one) {
                        for sub in &el.subs {
                            if let Some(&new_sub) = prev_hash.get(sub) {
                                new_el.subs.push(new_sub);
                            }
                        }
                    }
                }

                // Gets the superelements.
                if let Some(next_hash) = self.get(r.plus_one()) {
                    for sup in &el.sups {
                        if let Some(&new_sup) = next_hash.get(sup) {
                            new_el.sups.push(new_sup);
                        }
                    }
                }

                elements[new_idx] = new_el;
            }

            abs.push(elements);
        }

        abs
    }
}

/// A pair of indices in a polytope.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Indices(pub usize, pub usize);

/// Represents the lowest and highest element of a section of an abstract
/// polytope. Not to be confused with a cross-section.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SectionRef {
    /// The lowest element in the section.
    pub lo: ElementRef,

    /// The highest element in the section.
    pub hi: ElementRef,
}

impl std::fmt::Display for SectionRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "section between ({}) and ({})", self.lo, self.hi)
    }
}

impl SectionRef {
    /// Initializes a new section between two elements.
    pub fn new(lo: ElementRef, hi: ElementRef) -> Self {
        Self { lo, hi }
    }

    /// Returns the indices stored in the section.
    pub fn indices(&self) -> Indices {
        Indices(self.lo.idx, self.hi.idx)
    }
}

/// Maps the sections of a polytope with the same height to indices in a new
/// polytope. Organizes the sections first by their lowest rank, then by their
/// hash.
#[derive(Default, Debug)]
pub struct SectionHash(HashMap<SectionRef, usize>);

impl SectionHash {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn into_iter(self) -> std::collections::hash_map::IntoIter<SectionRef, usize> {
        self.0.into_iter()
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<SectionRef, usize> {
        self.0.iter()
    }

    pub fn insert(&mut self, section: SectionRef) -> Option<usize> {
        let len = self.len();
        self.0.insert(section, len)
    }

    /// Returns all singleton sections of a polytope.
    pub fn singletons(poly: &Abstract) -> Self {
        let mut section_hash = Self::new();

        for (rank, elements) in poly.ranks.rank_iter().rank_enumerate() {
            for idx in 0..elements.len() {
                let el = ElementRef::new(rank, idx);
                section_hash.insert(SectionRef::new(el, el));
            }
        }

        section_hash
    }

    /// Gets the index of a section in the hash, inserting it if necessary.
    pub fn get(&mut self, section: SectionRef) -> usize {
        use std::collections::hash_map::Entry;

        let len = self.len();

        // We organize by lowest rank, then by hash.
        match self.0.entry(section) {
            // Directly returns the index of the section.
            Entry::Occupied(idx) => *idx.get(),

            // Adds the section, increases the length by 1, then returns its index.
            Entry::Vacant(entry) => {
                entry.insert(len);
                len
            }
        }
    }
}
