//! A bunch of wrappers and helper structs that make dealing with abstract
//! polytopes much less confusing.

use std::{collections::HashMap, iter::IntoIterator};

use super::Abstract;
use crate::Polytope;

use vec_like::*;

/// A bundled rank and index, which can be used as coordinates to refer to an
/// element in an abstract polytope.

pub trait ElementRef: Copy + Eq {
    /// The rank of the element.
    fn rank(self) -> usize {
        self.rank_idx().0
    }

    /// The index of the element in its corresponding element list.
    fn idx(self) -> usize {
        self.rank_idx().1
    }

    /// Returns the bundled rank and index, in that order.
    fn rank_idx(self) -> (usize, usize);
}

impl ElementRef for (usize, usize) {
    fn rank_idx(self) -> (usize, usize) {
        self
    }
}

/// Common boilerplate code for [`Subelements`] and [`Superelements`].
pub trait Subsupelements: Sized + VecLike<VecItem = usize> {
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

/// Represents a list of subelements in a polytope. Each element is represented
/// as its index in the [`ElementList`] of the previous rank. Is used as one of
/// the fields in an [`Element`].
///
/// Internally, this is just a wrapper around a `Vec<usize>`.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Subelements(pub Vec<usize>);
impl_veclike!(Subelements, Item = usize, Index = usize);
impl Subsupelements for Subelements {}

/// Represents a list of superelements in a polytope. Each element is
/// represented as its index in the [`ElementList`] of the next rank. Is used as
/// one of the fields in an [`Element`].
///
/// Internally, this is just a wrapper around a `Vec<usize>`.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Superelements(pub Vec<usize>);
impl_veclike!(Superelements, Item = usize, Index = usize);
impl Subsupelements for Superelements {}

/// Represents an element in a polytope (also known as a face), which stores the
/// indices of both its [`Subelements`] and its [`Superelements`]. These make up
/// the entries of an [`ElementList`].
///
/// Even though one of these fields would suffice to precisely define an
/// element in an abstract polytope, we often are in need to use both of them.
/// To avoid recalculating them every single time, we just store them both.
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
/// [rank](https://polytope.miraheze.org/wiki/Rank). An [`Abstract`] is built
/// out of a [`RankVec`] of these.
///
/// If you only want to deal with the subelements of a polytope, consider using
/// a [`SubelementList`] instead.
///
/// Internally, this is just a wrapper around `Vec<Element>`.
#[derive(Debug, Clone)]
pub struct ElementList(Vec<Element>);
impl_veclike!(ElementList, Item = Element, Index = usize);

impl<'a> rayon::iter::IntoParallelIterator for &'a mut ElementList {
    type Iter = rayon::slice::IterMut<'a, Element>;

    type Item = &'a mut Element;

    fn into_par_iter(self) -> Self::Iter {
        self.as_mut().into_par_iter()
    }
}

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

/// A list of [`Subelements`] in a polytope. Can be used by an
/// [`AbstractBuilder`] to build the [`Elements`](Element) of a polytope one
/// rank at a time.
#[derive(Debug)]
pub struct SubelementList(Vec<Subelements>);
impl_veclike!(SubelementList, Item = Subelements, Index = usize);

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

#[derive(Debug, Clone)]
pub struct Ranks(Vec<ElementList>);
impl_veclike!(Ranks, Item = ElementList, Index = usize);

impl Ranks {
    pub fn with_rank_capacity(rank: usize) -> Self {
        Self::with_capacity(rank + 1)
    }

    pub fn rank(&self) -> usize {
        self.len() - 1
    }
}

/// A structure used to build a polytope from the bottom up.
///
/// To operate on polytopes, we often need both the [`Subelements`] and
/// [`Superelements`] of each [`Element`]. However, if we only had one of these
/// for every `Element`, the other would be uniquely specified. This struct
/// allows us to build a polytope rank by rank by specifying the
/// [`SubelementLists`](SubelementList) of each succesive rank. It also has a
/// few convenienece methods to build these lists in specific cases.
#[derive(Default)]
pub struct AbstractBuilder(Abstract);

impl AbstractBuilder {
    /// Initializes a new empty abstract builder.
    pub fn new() -> Self {
        Default::default()
    }

    /// Initializes a new empty abstract builder with a capacity to store
    /// elements up and until a given [`Rank`].
    pub fn with_capacity(rank: usize) -> Self {
        Self(Abstract::with_rank_capacity(rank))
    }

    /// Returns `true` if we haven't added any elements to the polytope.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Reserves capacity for at least `additional` more element lists to be
    /// inserted in the polytope.
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional)
    }

    /// Pushes a new [`SubelementList`] onto the polytope.
    pub fn push(&mut self, subelements: SubelementList) {
        self.0.push_subs(subelements)
    }

    /// Pushes an element list with a single empty element into the polytope.
    ///
    /// This method should only be used when the polytope is empty.
    pub fn push_min(&mut self) {
        // If you're using this method, the polytope should be empty.
        debug_assert!(self.is_empty());
        self.push(SubelementList::min());
    }

    /// Pushes an element list with a set number of vertices into the polytope.
    ///
    /// This method should only be used when a single, minimal element list has
    /// been inserted into the polytope.
    pub fn push_vertices(&mut self, vertex_count: usize) {
        // If you're using this method, the polytope should consist of a single
        // minimal element.
        debug_assert_eq!(self.0.rank(), 0);
        self.push(SubelementList::vertices(vertex_count))
    }

    /// Pushes a maximal element list into the polytope.
    ///
    /// This should be the last push operation that you apply to a polytope.
    pub fn push_max(&mut self) {
        self.0.push_max();
    }

    /// Returns the built polytope, consuming the builder in the process.
    pub fn build(self) -> Abstract {
        self.0
    }
}

/// Maps each recursive subelement of an abstract polytope's element to a
/// `usize`, representing its index in a new polytope. This is used to build the
/// elements figures of polytopes, or to find their vertices.
pub struct ElementHash(Vec<HashMap<usize, usize>>);

impl ElementHash {
    /// Returns a map from elements on a polytope to elements on a new polytope
    /// representing a particular element (as a polytope). If the element
    /// doesn't exist, we return `None`.
    pub fn new(poly: &Abstract, rank: usize, idx: usize) -> Option<Self> {
        poly.get_element(rank, idx)?;

        // A vector of HashMaps. The k-th entry is a map from k-elements of the
        // original polytope into k-elements in a new polytope.
        let mut hashes = Vec::with_capacity(rank + 1);
        for _ in 0..=rank {
            hashes.push(HashMap::new());
        }
        hashes[rank].insert(idx, 0);

        // Gets subindices of subindices, until reaching the vertices.
        for r in (1..=rank).rev() {
            let (left_slice, right_slice) = hashes.split_at_mut(r);
            let prev_hash = left_slice.last_mut().unwrap();
            let hash = right_slice.first().unwrap();

            for &idx in hash.keys() {
                for &sub in &poly[(r, idx)].subs {
                    let len = prev_hash.len();
                    prev_hash.entry(sub).or_insert(len);
                }
            }
        }

        Some(Self(hashes))
    }

    fn rank(&self) -> usize {
        self.0.len() - 1
    }

    /// Gets the `HashMap` corresponding to elements of a given rank.
    fn get(&self, idx: usize) -> Option<&HashMap<usize, usize>> {
        self.0.get(idx)
    }

    /// Gets the indices of the elements of a given rank in the original
    /// polytope.
    fn to_elements(&self, rank: usize) -> Vec<usize> {
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
        self.to_elements(1)
    }

    /// Gets the indices of the vertices of a given element in a polytope.
    pub fn to_polytope(&self, poly: &Abstract) -> Abstract {
        let rank = self.rank();
        let mut abs = Abstract::with_rank_capacity(rank);

        // For every rank stored in the element map.
        for r in 0..=rank {
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
                let el = poly.get_element(r, idx).unwrap();
                let mut new_el = Element::new();

                // Gets the subelements.
                if r >= 1 {
                    if let Some(prev_hash) = self.get(r - 1) {
                        for sub in &el.subs {
                            if let Some(&new_sub) = prev_hash.get(sub) {
                                new_el.subs.push(new_sub);
                            }
                        }
                    }
                }

                // Gets the superelements.
                if let Some(next_hash) = self.get(r + 1) {
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

/// Represents the lowest and highest element of a section of an abstract
/// polytope. Not to be confused with a cross-section.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SectionRef {
    /// The rank of the lowest element in the section.
    pub lo_rank: usize,

    pub lo_idx: usize,

    /// The highest element in the section.
    pub hi_rank: usize,

    pub hi_idx: usize,
}

impl std::fmt::Display for SectionRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "section between ({:?}) and ({:?})", self.lo(), self.hi())
    }
}

impl SectionRef {
    /// Initializes a new section between two elements.
    pub fn new(lo_rank: usize, lo_idx: usize, hi_rank: usize, hi_idx: usize) -> Self {
        Self {
            lo_rank,
            lo_idx,
            hi_rank,
            hi_idx,
        }
    }

    pub fn from_els(lo: (usize, usize), hi: (usize, usize)) -> Self {
        Self::new(lo.rank(), lo.idx(), hi.rank(), hi.idx())
    }

    pub fn singleton(rank: usize, idx: usize) -> Self {
        let el = (rank, idx);
        Self::from_els(el, el)
    }

    pub fn lo(self) -> (usize, usize) {
        (self.lo_rank, self.lo_idx)
    }

    pub fn hi(self) -> (usize, usize) {
        (self.hi_rank, self.hi_idx)
    }
}

/// Represents a map from sections in a polytope to their indices in a new
/// polytope (its [antiprism](Abstract::antiprism)). Exists only to make the
/// antiprism code a bit easier to understand.
///
/// In practice, all of the sections we store have a common height, which means
/// that we could save some memory by using a representation of [`SectionRef`]
/// with three arguments instead of four. This probably isn't worth the hassle,
/// though.
#[derive(Default, Debug)]
pub(crate) struct SectionHash(HashMap<SectionRef, usize>);

impl IntoIterator for SectionHash {
    type Item = (SectionRef, usize);

    type IntoIter = std::collections::hash_map::IntoIter<SectionRef, usize>;

    /// Returns an iterator over the stored section index pairs.
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl SectionHash {
    /// Initializes a new section hash.
    pub fn new() -> Self {
        Default::default()
    }

    /// Returns the number of stored elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns all singleton sections of a polytope.
    pub fn singletons(poly: &Abstract) -> Self {
        let mut section_hash = Self::new();

        for (rank, elements) in poly.ranks.iter().enumerate() {
            for idx in 0..elements.len() {
                section_hash
                    .0
                    .insert(SectionRef::singleton(rank, idx), section_hash.len());
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
