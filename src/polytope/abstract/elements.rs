//! A bunch of wrappers and helper structs that make dealing with abstract
//! polytopes much less confusing.

use std::{
    collections::HashMap,
    iter::IntoIterator,
    ops::{Index, IndexMut},
};

use super::{
    rank::{Rank, RankVec},
    Abstract,
};
use crate::polytope::Polytope;

/// A bundled rank and index, which can be used to refer to an element in an
/// abstract polytope.
#[derive(Clone, Debug, Hash)]
pub struct ElementRef {
    pub rank: Rank,
    pub idx: usize,
}

impl std::fmt::Display for ElementRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "element at rank {}, index {}", self.rank, self.idx)
    }
}

impl ElementRef {
    pub fn new(rank: Rank, idx: usize) -> Self {
        Self { rank, idx }
    }
}

/// Common boilerplate code for [`Subelements`] and [`Superelements`].
pub trait Subsupelements: Sized + Index<usize> + IndexMut<usize> + IntoIterator {
    /// Builds a list of either subelements or superelements from a vector.
    fn from_vec(vec: Vec<usize>) -> Self;

    /// Returns a reference to the internal vector.
    fn as_vec(&self) -> &Vec<usize>;

    /// Returns a mutable reference to the internal vector.
    fn as_vec_mut(&mut self) -> &mut Vec<usize>;

    /// Constructs a new, empty subelement or superelement list.
    fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    /// Returns `true` if the vector contains no elements.
    fn is_empty(&self) -> bool {
        self.as_vec().is_empty()
    }

    /// Returns the number of indices stored.
    fn len(&self) -> usize {
        self.as_vec().len()
    }

    /// Returns an iterator over the element indices.
    fn iter(&self) -> std::slice::Iter<usize> {
        self.as_vec().iter()
    }

    /// Returns an iterator that allows modifying each index.
    fn iter_mut(&mut self) -> std::slice::IterMut<usize> {
        self.as_vec_mut().iter_mut()
    }

    /// Sorts the indices.
    fn sort(&mut self) {
        self.as_vec_mut().sort_unstable()
    }

    /// Sorts the indices by a specified comparison function.
    fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&usize, &usize) -> std::cmp::Ordering,
    {
        self.as_vec_mut().sort_unstable_by(compare)
    }

    /// Sorts the indices with a key extraction function. The sorting is
    /// unstable.
    fn sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&usize) -> K,
        K: Ord,
    {
        self.as_vec_mut().sort_unstable_by_key(f)
    }

    /// Appends an index to the back of the list of indices.
    fn push(&mut self, value: usize) {
        self.as_vec_mut().push(value)
    }

    /// Returns `true` if the list of indices contains an index with the given
    /// value.
    fn contains(&self, x: &usize) -> bool {
        self.as_vec().contains(x)
    }

    /// Constructs a new, empty subelement or superelement list with the
    /// capacity to store a given amount of indices.
    fn with_capacity(capacity: usize) -> Self {
        Self::from_vec(Vec::with_capacity(capacity))
    }

    /// Constructs a subelement or superelement list consisting of the indices
    /// from `0` to `n - 1`.
    fn count(n: usize) -> Self {
        let mut vec = Vec::new();

        for i in 0..n {
            vec.push(i);
        }

        Self::from_vec(vec)
    }
}

/// Implements all remaining common code between `Subelements` and `Superelements`.
macro_rules! impl_sub_sups {
    ($T: ident, $name: expr) => {
        /// The indices of the
        #[doc = $name]
        /// of a polytope, which make up the entries of an [`Element`].
        #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $T(pub Vec<usize>);

        /// Allows indexing by an `usize`.
        impl Index<usize> for $T {
            type Output = usize;

            fn index(&self, index: usize) -> &Self::Output {
                &self.as_vec()[index]
            }
        }

        /// Allows mutable indexing by an `usize`.
        impl IndexMut<usize> for $T {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_vec_mut()[index]
            }
        }

        /// Iterates over the slice while moving out.
        impl IntoIterator for $T {
            type Item = usize;

            type IntoIter = std::vec::IntoIter<usize>;

            fn into_iter(self) -> Self::IntoIter {
                self.0.into_iter()
            }
        }

        /// Iterates over references in the slice.
        impl<'a> IntoIterator for &'a $T {
            type Item = &'a usize;

            type IntoIter = std::slice::Iter<'a, usize>;

            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        impl Subsupelements for $T {
            /// Builds a list of
            #[doc = $name]
            /// from a vector. Provided only for the [`Subsupelements`] trait.
            /// Use `Self` instead.
            fn from_vec(vec: Vec<usize>) -> Self {
                Self(vec)
            }

            /// Returns a reference to the internal vector. Provided only for
            /// the [`Subsupelements`] trait. Use `.0` instead.
            fn as_vec(&self) -> &Vec<usize> {
                &self.0
            }

            /// Returns a mutable reference to the internal vector. Provided
            /// only for the [`Subsupelements`] trait. Use `.0` instead.
            fn as_vec_mut(&mut self) -> &mut Vec<usize> {
                &mut self.0
            }
        }
    };
}

// Blanket implementations wouldn't work here.
impl_sub_sups!(Subelements, "subelements");
impl_sub_sups!(Superelements, "superelements");

/// An element in a polytope, which stores the indices of both its subelements
/// and superlements. These make up the entries of an [`ElementList`].
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

    /// Sorts both the subelements and superelements by index.
    pub fn sort(&mut self) {
        self.subs.sort();
        self.sups.sort();
    }
}

/// A list of [`Elements`](Element) of the same
/// [rank](https://polytope.miraheze.org/wiki/Rank).
///
/// If you only want to deal with the subelements of a
/// polytope, use [`SubelementList`] instead.
#[derive(Debug, Clone)]
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

    /// Moves all the elements of `other` into `Self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut ElementList) {
        self.0.append(&mut other.0)
    }

    /// Returns the number of elements in the element list.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns an iterator over the element list.
    pub fn iter(&self) -> std::slice::Iter<Element> {
        self.0.iter()
    }

    /// Returns an iterator that allows modifying each value.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<Element> {
        self.0.iter_mut()
    }

    /// Returns a reference to the element at a given index.
    pub fn get(&self, idx: usize) -> Option<&Element> {
        self.0.get(idx)
    }

    /// Returns a mutable reference to the element at a given index.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Element> {
        self.0.get_mut(idx)
    }

    /// Determines whether the element list contains a given element.
    pub fn contains(&self, x: &Element) -> bool {
        self.0.contains(x)
    }

    /// Resizes the `ElementList` in-place so that `len` is equal to `new_len`.
    /// Fills all new empty slots with `value`.
    pub fn resize(&mut self, new_len: usize, value: Element) {
        self.0.resize(new_len, value)
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

    /// Pushes a value into the element list.
    pub fn push(&mut self, value: Element) {
        self.0.push(value)
    }

    pub fn subelements(&self) -> SubelementList {
        let mut subelements = SubelementList::with_capacity(self.len());

        for el in self.iter() {
            subelements.push(el.subs.clone());
        }

        subelements
    }
}

impl IntoIterator for ElementList {
    type Item = Element;

    type IntoIter = std::vec::IntoIter<Element>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Index<usize> for ElementList {
    type Output = Element;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for ElementList {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// A list of subelements in a polytope.
pub struct SubelementList(Vec<Subelements>);

impl SubelementList {
    /// Initializes an empty subelement list.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Initializes an empty subelement list with a given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Pushes a value into the subelement list.
    pub fn push(&mut self, value: Subelements) {
        self.0.push(value);
    }

    /// Returns the number of elements in the subelement list.
    pub fn len(&self) -> usize {
        self.0.len()
    }

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

impl IntoIterator for SubelementList {
    type Item = Subelements;

    type IntoIter = std::vec::IntoIter<Subelements>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Index<usize> for SubelementList {
    type Output = Subelements;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for SubelementList {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// A structure used to build a polytope from the bottom up.
pub struct AbstractBuilder(Abstract);

impl AbstractBuilder {
    pub fn new() -> Self {
        Self(Abstract::new())
    }

    pub fn with_capacity(rank: Rank) -> Self {
        Self(Abstract::with_capacity(rank))
    }

    pub fn push(&mut self, subelements: SubelementList) {
        self.0.push_subs(subelements)
    }

    /// Pushes an element list with a single empty element into the polytope.
    pub fn push_min(&mut self) {
        // If you're using this method, the polytope should be empty.
        debug_assert!(self.0.ranks.is_empty());

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

/// As a byproduct of calculating either the vertices or the entire polytope
/// corresponding to a given section, we generate a map from ranks and indices
/// in the original polytope to ranks and indices in the section. This struct
/// encodes such a map as a ranked vector of hash maps.
pub struct ElementHash(RankVec<HashMap<usize, usize>>);

impl ElementHash {
    /// Gets the hashmap corresponding to elements of a given rank.
    pub fn get(&self, idx: Rank) -> Option<&HashMap<usize, usize>> {
        self.0.get(idx)
    }

    /// Returns a map from elements on the polytope to elements in an element.
    /// If the element doesn't exist, we return `None`.
    pub fn from_element(poly: &Abstract, el: &ElementRef) -> Option<Self> {
        poly.get_element(el)?;

        // A vector of HashMaps. The k-th entry is a map from k-elements of the
        // original polytope into k-elements in a new polytope.
        let mut hashes = RankVec::with_capacity(el.rank);
        for _ in Rank::range_inclusive_iter(Rank::new(-1), el.rank) {
            hashes.push(HashMap::new());
        }
        hashes[el.rank].insert(el.idx, 0);

        // Gets subindices of subindices, until reaching the vertices.
        for r in Rank::range_inclusive_iter(Rank::new(0), el.rank).rev() {
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

    /// Gets the indices of the elements of a given rank in a polytope.
    pub fn to_elements(&self, rank: Rank) -> Vec<usize> {
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

    /// Gets the indices of the vertices of a given element in a polytope.
    pub fn to_polytope(&self, poly: &Abstract) -> Abstract {
        let rank = self.0.rank();
        let mut abs = Abstract::with_capacity(rank);

        for r in Rank::range_inclusive_iter(Rank::new(-1), rank) {
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
                let el = poly.get_element(&ElementRef::new(r, idx)).unwrap();
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

/// A section of an abstract polytope, not to be confused with a cross-section.
#[derive(Hash)]
pub struct Section {
    /// The lowest element in the section.
    pub lo: ElementRef,

    /// The highest element in the section.
    pub hi: ElementRef,
}

impl std::fmt::Display for Section {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "section between ({}) and ({})", self.lo, self.hi)
    }
}

impl Section {
    pub fn new(lo: ElementRef, hi: ElementRef) -> Self {
        Self { lo, hi }
    }

    /// The maximal section of a polytope.
    pub fn max(rank: Rank) -> Self {
        Self {
            lo: ElementRef::new(Rank::new(-1), 0),
            hi: ElementRef::new(rank, 0),
        }
    }

    /// The height of a section, i.e. the rank of the polytope it describes.
    pub fn height(&self) -> Rank {
        (self.hi.rank - self.lo.rank).minus_one()
    }

    /// Returns the indices stored in the section.
    pub fn indices(&self) -> Indices {
        Indices(self.lo.idx, self.hi.idx)
    }
}

/// Maps the sections of a polytope with the same height to indices in a new
/// polytope. Organizes the sections first by their lowest rank, then by their
/// hash.
#[derive(Debug)]
pub struct SectionHash {
    /// A map from sections in a polytope to indices.
    pub rank_vec: RankVec<HashMap<Indices, usize>>,

    /// The total amount of elements in the map.
    pub len: usize,
}

impl SectionHash {
    /// Initializes a new, empty `SectionHash` for sections of a given height
    /// in a polytope with a given rank.
    pub fn new(rank: Rank, height: Rank) -> Self {
        let mut rank_vec = RankVec::new();
        for _ in Rank::range_iter(Rank::new(-1), rank - height) {
            rank_vec.push(HashMap::new());
        }

        Self { rank_vec, len: 0 }
    }

    /// Returns all singleton sections of a polytope.
    pub fn singletons(poly: &Abstract) -> Self {
        let rank = poly.rank();
        let mut section_hash = Self::new(rank, Rank::new(-1));
        let mut len = 0;

        for (rank, elements) in poly.ranks.iter().rank_enumerate() {
            for i in 0..elements.len() {
                section_hash.rank_vec[rank].insert(Indices(i, i), len);
                len += 1;
            }
        }

        section_hash.len = len;
        section_hash
    }

    /// Gets the index of a section in the hash, inserting it if necessary.
    pub fn get(&mut self, section: Section) -> usize {
        use std::collections::hash_map::Entry;

        // We organize by lowest rank, then by hash.
        match self.rank_vec[section.lo.rank].entry(section.indices()) {
            // Directly returns the index of the section.
            Entry::Occupied(idx) => *idx.get(),

            // Adds the section, increases the length by 1, then returns its index.
            Entry::Vacant(entry) => {
                let len = self.len;
                entry.insert(len);
                self.len += 1;
                len
            }
        }
    }
}
