use std::collections::HashMap;

use derive_deref::{Deref, DerefMut};

use super::{rank::RankVec, Abstract};

/// Common boilerplate code for subelements and superelements.
pub trait Subsupelements: Sized {
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

#[derive(Deref, DerefMut)]
/// As a byproduct of calculating either the vertices or the entire polytope
/// corresponding to a given section, we generate a map from ranks and indices
/// in the original polytope to ranks and indices in the section. This struct
/// encodes such a map as a vector of hash maps.
pub struct ElementHash(RankVec<HashMap<usize, usize>>);

impl ElementHash {
    /// Returns a map from elements on the polytope to elements in an element.
    /// If the element doesn't exist, we return `None`.
    pub fn from_element(poly: &Abstract, rank: isize, idx: usize) -> Option<Self> {
        poly.get_element(rank, idx)?;

        // A vector of HashMaps. The k-th entry is a map from k-elements of the
        // original polytope into k-elements in a new polytope.
        let mut hashes = RankVec::with_capacity(rank);
        for _ in -1..=rank {
            hashes.push(HashMap::new());
        }
        hashes[rank].insert(idx, 0);

        // Gets subindices of subindices, until reaching the vertices.
        for r in (0..=rank).rev() {
            let (left_slice, right_slice) = hashes.split_at_mut(r);
            let prev_hash = left_slice.last_mut().unwrap();
            let hash = right_slice.first().unwrap();

            for (&idx, _) in hash.iter() {
                for &sub in poly[r as isize][idx].subs.iter() {
                    let len = prev_hash.len();
                    prev_hash.entry(sub).or_insert(len);
                }
            }
        }

        Some(Self(hashes))
    }

    /// Gets the indices of the elements of a given rank in a polytope.
    pub fn to_elements(&self, rank: isize) -> Vec<usize> {
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
        let rank = self.rank();
        let mut abs = Abstract::with_capacity(rank);

        for r in -1..=rank {
            let mut elements = ElementList::new();
            let hash = &self[r];

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
                if let Some(prev_hash) = self.get(r - 1) {
                    for sub in el.subs.iter() {
                        if let Some(&new_sub) = prev_hash.get(sub) {
                            new_el.subs.push(new_sub);
                        }
                    }
                }

                // Gets the superelements.
                if let Some(next_hash) = self.get(r + 1) {
                    for sup in el.sups.iter() {
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
