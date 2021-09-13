//! A bunch of wrappers and helper structs that make dealing with abstract
//! polytopes much less confusing.

use std::{
    collections::HashMap,
    iter::{self, FromIterator, IntoIterator},
    ops::{Index, IndexMut},
    slice, vec,
};

use super::Abstract;

use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use vec_like::*;

/// Represents a map from ranks and indices into elements of a given type.
/// This struct used whenever we want to associate some value to every element
/// of an abstract polytope.
///
/// Internally, this is just a wrapper around a `Vec<Vec<T>>`.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct ElementMap<T>(Vec<Vec<T>>);
impl_veclike!(@for [T] ElementMap<T>, Item = Vec<T>);

impl<T> Index<(usize, usize)> for ElementMap<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self[index.0][index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for ElementMap<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self[index.0][index.1]
    }
}

/// Represents a list of subelements in a polytope. Each element is represented
/// as its index in the [`ElementList`] of the previous rank. This is used as
/// one of the fields in an [`Element`].
///
/// Internally, this is just a wrapper around a `Vec<usize>`.
///
/// # Note on notation
/// Throughout the code, and unless specified otherwise, we use the word
/// **subelement** to refer to the elements of rank `r - 1` incident to an
/// element of rank `r`. This is contrary to mathematical use, where it just
/// refers to any element that's incident and of lesser rank than another. We
/// instead use the term **recursive subelement** for the standard mathematical
/// notion.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct Subelements(Vec<usize>);
impl_veclike!(Subelements, Item = usize);

/// Represents a list of superelements in a polytope. Each element is
/// represented as its index in the [`ElementList`] of the previous rank. This
/// is used as  one of the fields in an [`Element`].
///
/// Internally, this is just a wrapper around a `Vec<usize>`.
///
/// # Note on notation
/// Throughout the code, and unless specified otherwise, we use the word
/// **superelement** to refer to the elements of rank `r + 1` incident to an
/// element of rank `r`. This is contrary to mathematical use, where it just
/// refers to any element that's incident and of greater rank than another. We
/// instead use the term **recursive superelement** for the standard
/// mathematical notion.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct Superelements(Vec<usize>);
impl_veclike!(Superelements, Item = usize);

/// Represents an element in a polytope (also known as a face). Each element
/// stores only the indices of its [`Subelements`] and its [`Superelements`].
///
/// Even though one of these fields would suffice to precisely define an
/// element in an abstract polytope, we're often are in need of both of them. To
/// avoid recalculating them every single time, we just store them both.
#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub struct Element {
    /// The indices of the subelements of the previous rank.
    pub subs: Subelements,

    /// The indices of the superelements of the next rank.
    pub sups: Superelements,
}

impl From<Subelements> for Element {
    fn from(subs: Subelements) -> Self {
        Self {
            subs,
            sups: Superelements::new(),
        }
    }
}

impl From<Superelements> for Element {
    fn from(sups: Superelements) -> Self {
        Self {
            subs: Subelements::new(),
            sups,
        }
    }
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
            sups: (0..vertex_count).collect(),
        }
    }

    /// Builds a maximal element adjacent to a given number of facets.
    pub fn max(facet_count: usize) -> Self {
        Self {
            subs: (0..facet_count).collect(),
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

/// A list of [`Elements`](Element) of the same rank.
///
/// Internally, this is just a wrapper around a `Vec<Element>`.
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct ElementList(Vec<Element>);
impl_veclike!(ElementList, Item = Element);

impl<'a> rayon::iter::IntoParallelIterator for &'a mut ElementList {
    type Iter = rayon::slice::IterMut<'a, Element>;
    type Item = &'a mut Element;

    fn into_par_iter(self) -> Self::Iter {
        self.0.par_iter_mut()
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

/// A list of [`Subelements`] corresponding to [`Elements`](Element) of the same
/// rank.
///
/// Internally, this is just a wrapper around a `Vec<Subelements>`.
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct SubelementList(Vec<Subelements>);
impl_veclike!(SubelementList, Item = Subelements);

impl SubelementList {
    /// Returns the subelement list for the minimal element in a polytope.
    pub fn min() -> Self {
        Self(vec![Subelements::new()])
    }

    /// Returns the subelement list for a set number of vertices in a polytope.
    pub fn vertices(vertex_count: usize) -> Self {
        iter::repeat(Subelements(vec![0]))
            .take(vertex_count)
            .collect()
    }

    /// Returns the subelement list for the maximal element in a polytope with a
    /// given facet count.
    pub fn max(facet_count: usize) -> Self {
        Self(vec![(0..facet_count).collect()])
    }
}

/// The signature of the function that turns an `&ElementList` into an iterator.
type IterFn = for<'r> fn(&'r ElementList) -> slice::Iter<'r, Element>;

/// The signature of the function that turns an `&mut ElementList` into a
/// mutable iterator.
type IterMutFn = for<'r> fn(&'r mut ElementList) -> slice::IterMut<'r, Element>;

/// The signature of the function that turns an `ElementList` into an owned
/// iterator.
type IntoIterFn = fn(ElementList) -> std::vec::IntoIter<Element>;

/// The signature of the function that returns the length of an `ElementList`.
type LenFn = for<'r> fn(&'r ElementList) -> usize;

/// The signature of an iterator over an `&'a ElementList`.
pub type ElementIter<'a> = iter::Flatten<iter::Map<slice::Iter<'a, ElementList>, IterFn>>;

/// The signature of an iterator over an `&'a mut ElementList`.
pub type ElementIterMut<'a> = iter::Flatten<iter::Map<slice::IterMut<'a, ElementList>, IterMutFn>>;

/// The signature of an owned iterator over an `ElementList`.
pub type ElementIntoIter = iter::Flatten<iter::Map<vec::IntoIter<ElementList>, IntoIterFn>>;

/// Represents the [`ElementLists`](ElementList) of each rank that make up an
/// abstract polytope.
///
/// Contrary to [`Abstract`], there's no requirement that the elements in
/// `Ranks` form a valid polytope.
#[derive(Debug, Clone)]
pub struct Ranks(Vec<ElementList>);
impl_veclike!(Ranks, Item = ElementList);

impl Index<(usize, usize)> for Ranks {
    type Output = Element;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self[index.0][index.1]
    }
}

impl IndexMut<(usize, usize)> for Ranks {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self[index.0][index.1]
    }
}

/// The trait for any structure with an underlying set of [`Ranks`].
///
/// This is meant to provide implementations for the methods common to an
/// abstract polytope and its underlying set of ranks. As such, this trait does
/// not contain any methods to build polytopes.
///
/// Furthermore, none of these methods may assume that it's being called on a
/// valid polytope, though they may panic under certain conditions.
pub trait Ranked:
    Sized + Index<usize, Output = ElementList> + Index<(usize, usize), Output = Element>
{
    /// Returns a reference to the ranks.
    fn ranks(&self) -> &Ranks;

    /// Returns the ranks.
    fn into_ranks(self) -> Ranks;

    /// Asserts that `self` is a valid abstract polytope.
    ///
    /// # Panics
    /// This method will panic if the assertion fails.
    fn assert_valid(&self) {
        self.ranks().is_valid().unwrap();
    }

    /// Returns the rank of the structure, i.e. the length of the `Ranks` minus
    /// one.
    ///
    /// # Panics
    /// This method will panic if it's called on an empty set of ranks.
    fn rank(&self) -> usize {
        self.ranks().len() - 1
    }

    /// Returns the number of elements of a given rank. Returns 0 if the rank is
    /// out of bounds.
    fn el_count(&self, rank: usize) -> usize {
        self.ranks().get(rank).map(ElementList::len).unwrap_or(0)
    }

    /// Returns an iterator over the element counts of the structure.
    fn el_count_iter(&self) -> iter::Map<slice::Iter<'_, ElementList>, LenFn> {
        self.ranks().iter().map(ElementList::len as LenFn)
    }

    /// Returns a reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`Polytope::element`](crate::Polytope::element).
    fn get_element(&self, rank: usize, idx: usize) -> Option<&Element> {
        self.ranks().get(rank)?.get(idx)
    }

    /// Gets a reference to the element list of a given rank.
    fn get_element_list(&self, rank: usize) -> Option<&ElementList> {
        self.ranks().get(rank)
    }

    /// Returns a reference to the minimal element of the polytope.
    ///
    /// # Panics
    /// Panics if the minimal element has not been initialized.
    fn min(&self) -> &Element {
        &self[(0, 0)]
    }

    /// Returns the number of minimal elements. This always equals 1 in a valid
    /// polytope.
    fn min_count(&self) -> usize {
        self.el_count(0)
    }

    /// Returns a reference to the maximal element of the polytope.
    ///
    /// # Panics
    /// Panics if the maximal element has not been initialized.
    fn max(&self) -> &Element {
        &self[(self.rank(), 0)]
    }

    /// Returns the number of maximal elements. This always equals 1 in a valid
    /// polytope.
    fn max_count(&self) -> usize {
        self.el_count(self.rank())
    }

    /// Returns the number of vertices.
    fn vertex_count(&self) -> usize {
        self.el_count(1)
    }

    /// Returns the number of edges.
    fn edge_count(&self) -> usize {
        self.el_count(2)
    }

    /// Returns the number of facets.
    fn facet_count(&self) -> usize {
        self.el_count(self.rank().wrapping_sub(1))
    }

    /// Returns an iterator over the elements.
    fn element_iter(&self) -> ElementIter<'_> {
        self.ranks()
            .iter()
            .map(ElementList::iter as IterFn)
            .flatten()
    }

    /// Returns an owned iterator over the elements.
    fn element_into_iter(self) -> ElementIntoIter {
        self.into_ranks()
            .into_iter()
            .map(ElementList::into_iter as IntoIterFn)
            .flatten()
    }
}

impl Ranked for Ranks {
    fn ranks(&self) -> &Ranks {
        self
    }

    fn into_ranks(self) -> Ranks {
        self
    }
}

impl Ranks {
    /// Initializes a new set of ranks capable of storing elements up to a given
    /// rank.
    pub fn with_rank_capacity(rank: usize) -> Self {
        Self::with_capacity(rank + 1)
    }

    /// Returns a mutable reference to the minimal element of the polytope.
    ///
    /// # Panics
    /// Panics if the minimal element has not been initialized.
    pub fn min_mut(&mut self) -> &mut Element {
        &mut self[(0, 0)]
    }

    /// Returns a mutable reference to the maximal element of the polytope.
    ///
    /// # Panics
    /// Panics if the maximal element has not been initialized.
    pub fn max_mut(&mut self) -> &mut Element {
        let rank = self.rank();
        &mut self[(rank, 0)]
    }

    /// Returns a mutable reference to an element of the polytope.
    pub fn get_element_mut(&mut self, rank: usize, idx: usize) -> Option<&mut Element> {
        self.get_mut(rank)?.get_mut(idx)
    }

    /// Returns a mutable iterator over the elements.
    pub fn element_iter_mut(&mut self) -> ElementIterMut<'_> {
        self.iter_mut()
            .map(ElementList::iter_mut as IterMutFn)
            .flatten()
    }

    /// Applies a function to all elements in parallel.
    pub fn for_each_element_mut<F: Fn(&mut Element) + Sync + Send>(&mut self, f: F) {
        // No use parallelizing over all minimal or maximal elements.
        f(self.min_mut());
        f(self.max_mut());

        let rank = self.rank();
        for elements in self.iter_mut().take(rank).skip(1) {
            elements.par_iter_mut().for_each(&f);
        }
    }

    /// Sorts all of the superelements and subelements by index.
    pub fn element_sort(&mut self) {
        self.for_each_element_mut(Element::sort)
    }
}

/// This struct allows us to build a polytope rank by rank by specifying the
/// [`SubelementLists`](SubelementList) of each succesive rank. It also has a
/// few convenienece methods to build these lists in specific cases.
///
/// # An invariant
/// An `AbstractBuilder` wraps around a set of [`Ranks`]. Every method you call
/// on it will make it so that the subelements and superelements of any rank
/// save for the upper one are consistently filled out. The upper rank's
/// elements will have no elements.
///
/// By calling [`Self::build`], you're asserting that the structure you've built
/// represents a valid [`Abstract`] polytope.
#[derive(Default)]
pub struct AbstractBuilder(Ranks);

impl From<Ranks> for AbstractBuilder {
    fn from(ranks: Ranks) -> Self {
        Self(ranks)
    }
}

impl AbstractBuilder {
    /// Initializes a new empty abstract builder.
    pub fn new() -> Self {
        Default::default()
    }

    /// Returns a reference to the [`Ranks`].
    pub fn ranks(&self) -> &Ranks {
        &self.0
    }

    /// Initializes a new empty abstract builder with a capacity to store
    /// a specified amount of elements.
    fn with_capacity(rank: usize) -> Self {
        Self(Ranks::with_capacity(rank))
    }

    /// Initializes a new empty abstract builder with a capacity to store
    /// elements up and until a given rank.
    pub fn with_rank_capacity(rank: usize) -> Self {
        Self(Ranks::with_rank_capacity(rank))
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

    /// Inserts a new empty [`ElementList`].
    pub fn push_empty(&mut self) {
        self.0.push(ElementList::new())
    }

    /// Inserts a new empty [`ElementList`] with a given capacity.
    pub fn push_with_capacity(&mut self, capacity: usize) {
        self.0.push(ElementList::with_capacity(capacity))
    }

    /// Pushes an element with a given list of subelements into the elements of
    /// a given rank. Updates the superelements of its subelements
    /// automatically.
    pub fn push_subs(&mut self, subs: Subelements) {
        let rank = self.0.rank();
        let el_count = self.0.el_count(rank);

        // Updates superelements of the lower rank.
        for &sub in &subs {
            self.0[(rank - 1, sub)].sups.push(el_count);
        }

        self.0[rank].push(subs.into());
    }

    /// Pushes a new subelement list, assuming that the superelements of the
    /// current maximal rank haven't already been set. If they have already been
    /// set, use [`push`](Self::push) instead.
    pub fn push(&mut self, subelements: SubelementList) {
        self.extend(subelements)
    }

    /// Pushes an element list with a single empty element into the polytope.
    ///
    /// This method should only be used when the polytope is empty.
    pub fn push_min(&mut self) {
        debug_assert!(self.is_empty());
        self.push(SubelementList::min());
    }

    /// Pushes a maximal element into the polytope.
    ///
    /// This should be the last push operation that you apply to a polytope.
    pub fn push_max(&mut self) {
        let facet_count = self.0.max_count();
        self.push(SubelementList::max(facet_count));
    }

    /// Pushes an element list with a set number of vertices into the polytope.
    ///
    /// This method should only be used when a single, minimal element list has
    /// been inserted into the polytope.
    pub fn push_vertices(&mut self, vertex_count: usize) {
        debug_assert_eq!(self.0.len(), 1);
        self.push(SubelementList::vertices(vertex_count))
    }

    /// Returns the built polytope, consuming the builder in the process.
    ///
    /// # Safety
    /// By calling this method, you're asserting that whatever you've built is
    /// a valid [`Abstract`] polytope.
    pub unsafe fn build(self) -> Abstract {
        Abstract::from_ranks(self.0)
    }
}

impl Extend<Subelements> for AbstractBuilder {
    fn extend<T: IntoIterator<Item = Subelements>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.push_with_capacity(iter.size_hint().0);

        for subs in iter {
            self.push_subs(subs);
        }
    }
}

impl Extend<SubelementList> for AbstractBuilder {
    fn extend<T: IntoIterator<Item = SubelementList>>(&mut self, iter: T) {
        for subelements in iter {
            self.push(subelements);
        }
    }
}

impl FromIterator<SubelementList> for AbstractBuilder {
    fn from_iter<T: IntoIterator<Item = SubelementList>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut builder = Self::with_capacity(iter.size_hint().0);
        builder.extend(iter);
        builder
    }
}

// TODO: The rest of the file contains highly specialized structs for specific
// algorithms. We should probably move them elsewhere.

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
        let mut hashes: Vec<_> = iter::repeat_with(HashMap::new).take(rank + 1).collect();
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
        // TODO: use an AbstractBuilder instead, probably.
        let rank = self.rank();
        let mut abs = Ranks::with_rank_capacity(poly.rank());

        // For every rank stored in the element map.
        for r in 0..=rank {
            let hash = &self.0[r];
            let mut elements: ElementList = iter::repeat(Element::new()).take(hash.len()).collect();

            // For every element of rank r in the hash element list.
            for (&idx, &new_idx) in hash {
                // We take the corresponding element in the original polytope
                // and use the hash map to get its sub and superelements in the
                // new polytope.
                let el = &poly[(r, idx)];

                // Gets the subelements.
                let mut subs = Subelements::new();
                if r >= 1 {
                    if let Some(prev_hash) = self.get(r - 1) {
                        for sub in &el.subs {
                            if let Some(&new_sub) = prev_hash.get(sub) {
                                subs.push(new_sub);
                            }
                        }
                    }
                }

                // Gets the superelements. (do I really need to do this?)
                let mut sups = Superelements::new();
                if let Some(next_hash) = self.get(r + 1) {
                    for sup in &el.sups {
                        if let Some(&new_sup) = next_hash.get(sup) {
                            sups.push(new_sup);
                        }
                    }
                }

                elements[new_idx] = Element { subs, sups };
            }

            abs.push(elements);
        }

        // Safety: TODO document
        unsafe { Abstract::from_ranks(abs) }
    }
}
