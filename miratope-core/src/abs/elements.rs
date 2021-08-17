//! A bunch of wrappers and helper structs that make dealing with abstract
//! polytopes much less confusing.

use std::{
    collections::HashMap,
    iter::{self, IntoIterator},
    ops::{Index, IndexMut},
    slice, vec,
};

use super::Abstract;

use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use vec_like::*;

/// Represents a map from ranks and indices into elements of a given type.
/// Is internally stored as a jagged array.
#[derive(Clone, Debug)]
pub struct ElementMap<T>(Vec<Vec<T>>);
impl_veclike!(@for [T] ElementMap<T>, Item = Vec<T>, Index = usize);

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

/// Common boilerplate code for [`Subelements`] and [`Superelements`].
pub trait Subsupelements: VecLike<VecItem = usize> {
    /// Constructs a subelement or superelement list consisting of the indices
    /// from `0` to `n - 1`.
    fn count(n: usize) -> Self {
        let mut vec = Vec::new();

        for i in 0..n {
            vec.push(i);
        }

        Self::from_inner(vec)
    }

    /// Returns a reference to the inner slice.
    fn as_slice(&self) -> &[usize] {
        self.as_inner().as_slice()
    }

    /// Returns a reference to the inner mutable slice.
    fn as_mut_slice(&mut self) -> &mut [usize] {
        self.as_inner_mut().as_mut_slice()
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
#[derive(Debug, Clone)]
pub struct Ranks(Vec<ElementList>);
impl_veclike!(Ranks, Item = ElementList, Index = usize);

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
pub trait Ranked:
    Sized + IndexMut<usize, Output = ElementList> + IndexMut<(usize, usize), Output = Element>
{
    /// Returns a reference to the ranks.
    fn ranks(&self) -> &Ranks;

    /// Returns a mutable reference to the ranks.
    fn ranks_mut(&mut self) -> &mut Ranks;

    /// Returns the ranks.
    fn into_ranks(self) -> Ranks;

    /// Returns the rank of the structure, i.e. the length of the `Ranks` minus
    /// one.
    fn rank(&self) -> usize {
        self.ranks().len() - 1
    }

    /// Returns the number of elements of a given rank.
    fn el_count(&self, rank: usize) -> usize {
        self.ranks().get(rank).map(ElementList::len).unwrap_or(0)
    }

    /// Returns an iterator over the element counts of the structure.
    fn el_count_iter(&self) -> iter::Map<slice::Iter<'_, ElementList>, LenFn> {
        self.ranks().iter().map(ElementList::len as LenFn)
    }

    /// Returns a reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`element`](Self::element).
    fn get_element(&self, rank: usize, idx: usize) -> Option<&Element> {
        self.ranks().get(rank)?.get(idx)
    }

    /// Returns a mutable reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`element`](Self::element).
    fn get_element_mut(&mut self, rank: usize, idx: usize) -> Option<&mut Element> {
        self.ranks_mut().get_mut(rank)?.get_mut(idx)
    }

    /// Gets a reference to the element list of a given rank.
    fn get_element_list(&self, rank: usize) -> Option<&ElementList> {
        self.ranks().get(rank)
    }

    /// Gets a mutable reference to the element list of a given rank.
    fn get_element_list_mut(&mut self, rank: usize) -> Option<&mut ElementList> {
        self.ranks_mut().get_mut(rank)
    }

    /// Returns a reference to the minimal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    fn min(&self) -> &Element {
        &self[(0, 0)]
    }

    /// Returns a mutable reference to the minimal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    fn min_mut(&mut self) -> &mut Element {
        &mut self[(0, 0)]
    }

    /// Returns the number of minimal elements. Unless you call this in the
    /// middle of some other method, this should always be exactly 1.
    fn min_count(&self) -> usize {
        self.el_count(0)
    }

    /// Returns a reference to the maximal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    fn max(&self) -> &Element {
        &self[(self.rank(), 0)]
    }

    /// Returns a mutable reference to the maximal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    fn max_mut(&mut self) -> &mut Element {
        let rank = self.rank();
        &mut self[(rank, 0)]
    }

    /// Returns the number of maximal elements. Unless you call this in the
    /// middle of some other method, this should always be exactly 1.
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

    /// Returns a reference to the facets of the polytope.
    fn get_facets(&self) -> Option<&ElementList> {
        self.get_element_list(self.rank().wrapping_sub(1))
    }

    /// Returns a mutable reference to the facets of the polytope.
    fn get_facets_mut(&mut self) -> Option<&mut ElementList> {
        self.get_element_list_mut(self.rank().wrapping_sub(1))
    }

    /// Returns the number of facets.
    fn facet_count(&self) -> usize {
        self.el_count(self.rank().wrapping_sub(1))
    }

    /// Pushes a list of elements of a given rank.
    fn push_elements(&mut self, elements: ElementList) {
        self.ranks_mut().push(elements);
    }

    /// Pushes a given element into the vector of elements of a given rank.
    fn push_at(&mut self, rank: usize, el: Element) {
        self[rank].push(el);
    }

    /// Pushes a given element into the vector of elements of a given rank.
    /// Updates the superelements of its subelements automatically.
    fn push_subs_at(&mut self, rank: usize, subs: Subelements) {
        if rank != 0 {
            let i = self.el_count(rank);

            // Updates superelements of the lower rank.
            for &sub in &subs {
                self[(rank - 1, sub)].sups.push(i);
            }
        }

        self.push_at(rank, subs.into());
    }

    /// Pushes a new subelement list, assuming that the
    /// superelements of the current maximal rank **haven't** already been set.
    /// If they have already been set, use [`push`](Self::push) instead.
    fn push_subs(&mut self, subelements: SubelementList) {
        self.push_elements(ElementList::with_capacity(subelements.len()));

        for sub_el in subelements {
            self.push_subs_at(self.rank(), sub_el);
        }
    }

    /// Pushes a maximal element into the polytope, with the facets as
    /// subelements. To be used in circumstances where the elements are built up
    /// in layers.
    fn push_max(&mut self) {
        let max_count = self.max_count();
        self.push_subs(SubelementList::max(max_count));
    }

    /// Returns an iterator over the elements.
    fn element_iter(&self) -> ElementIter<'_> {
        self.ranks()
            .iter()
            .map(ElementList::iter as IterFn)
            .flatten()
    }

    /// Returns a mutable iterator over the elements.
    fn element_iter_mut(&mut self) -> ElementIterMut<'_> {
        self.ranks_mut()
            .iter_mut()
            .map(ElementList::iter_mut as IterMutFn)
            .flatten()
    }

    /// Returns an owned iterator over the elements.
    fn element_into_iter(self) -> ElementIntoIter {
        self.into_ranks()
            .into_iter()
            .map(ElementList::into_iter as IntoIterFn)
            .flatten()
    }

    /// Applies a function to all elements in parallel.
    fn for_each_element<F: Fn(&mut Element) + Sync + Send + Clone>(&mut self, f: F) {
        // No use parallelizing over all minimal or maximal elements.
        let rank = self.rank();
        f(self.min_mut());

        if rank != 0 {
            for elements in self.ranks_mut().iter_mut().skip(1).take(rank - 1) {
                elements.par_iter_mut().for_each(f.clone());
            }

            f(self.max_mut());
        }
    }

    // TODO: sugar for the parallel iterator.
}

impl Ranked for Ranks {
    fn ranks(&self) -> &Ranks {
        self
    }

    fn ranks_mut(&mut self) -> &mut Ranks {
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

    /// Sorts all of the superelements and subelements by index.
    pub fn element_sort(&mut self) {
        self.for_each_element(Element::sort)
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

    /// The index of the lowest element in the section.
    pub lo_idx: usize,

    /// The rank of the highest element in the section.
    pub hi_rank: usize,

    /// The index of the highest element in the section.
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

    /// Creates a new singleton section.
    pub fn singleton(rank: usize, idx: usize) -> Self {
        Self::new(rank, idx, rank, idx)
    }

    /// Creates a new section by replacing the lowest element of another.
    pub fn with_lo(mut self, lo_rank: usize, lo_idx: usize) -> Self {
        self.lo_rank = lo_rank;
        self.lo_idx = lo_idx;
        self
    }

    /// Creates a new section by replacing the highest element of another.
    pub fn with_hi(mut self, hi_rank: usize, hi_idx: usize) -> Self {
        self.hi_rank = hi_rank;
        self.hi_idx = hi_idx;
        self
    }

    /// Returns the lowest element of a section.
    pub fn lo(self) -> (usize, usize) {
        (self.lo_rank, self.lo_idx)
    }

    /// Returns the highest element of a section.
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

        for (rank, elements) in poly.iter().enumerate() {
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
