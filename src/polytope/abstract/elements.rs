use std::collections::HashMap;

use super::{rank::RankVec, Abstract};

/// Common boilerplate code for subelements and superelements.
pub trait Subsupelements: Sized {
    /// Builds a list of either subelements or superelements from a vector.
    fn from_vec(vec: Vec<usize>) -> Self;

    /// Returns a reference to the internal vector.
    fn as_vec(&self) -> &Vec<usize>;

    /// Returns a mutable reference to the internal vector.
    fn as_vec_mut(&mut self) -> &mut Vec<usize>;

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

    /// Constructs a new, empty subelement or superelement list.
    fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    /// Constructs a new, empty subelement or superelement list with the
    /// capacity to store a given amount of indices.
    fn with_capacity(capacity: usize) -> Self {
        Self::from_vec(Vec::with_capacity(capacity))
    }

    /// Constructs a subelement or superelement list consisting of the indices
    /// from `0` to `count`.
    fn count(count: usize) -> Self {
        let mut vec = Vec::new();

        for i in 0..count {
            vec.push(i);
        }

        Self::from_vec(vec)
    }
}

/// The indices of the subelements of a polytope.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Subelements(pub Vec<usize>);

impl Subsupelements for Subelements {
    /// Builds a list of subelements from a vector.
    fn from_vec(vec: Vec<usize>) -> Self {
        Self(vec)
    }

    /// Returns a reference to the internal vector. Use `.0` instead.
    fn as_vec(&self) -> &Vec<usize> {
        &self.0
    }

    /// Returns a mutable reference to the internal vector. Use `.0` instead.
    fn as_vec_mut(&mut self) -> &mut Vec<usize> {
        &mut self.0
    }
}

impl std::ops::Index<usize> for Subelements {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_vec()[index]
    }
}

impl std::ops::IndexMut<usize> for Subelements {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_vec_mut()[index]
    }
}

/// The indices of the superelements of a polytope.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Superelements(pub Vec<usize>);

impl Subsupelements for Superelements {
    /// Builds a list of superelements from a vector.
    fn from_vec(vec: Vec<usize>) -> Self {
        Self(vec)
    }

    /// Returns a reference to the internal vector. Use `.0` instead.
    fn as_vec(&self) -> &Vec<usize> {
        &self.0
    }

    /// Returns a mutable reference to the internal vector. Use `.0` instead.
    fn as_vec_mut(&mut self) -> &mut Vec<usize> {
        &mut self.0
    }
}

impl std::ops::Index<usize> for Superelements {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_vec()[index]
    }
}

impl std::ops::IndexMut<usize> for Superelements {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_vec_mut()[index]
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

    /// Sorts both the subelements and superelements by index.
    pub fn sort(&mut self) {
        self.subs.sort();
        self.sups.sort();
    }
}

/// A list of [`Elements`](Element) of the same
/// [rank](https://polytope.miraheze.org/wiki/Rank).
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

    pub fn get(&self, idx: usize) -> Option<&Element> {
        self.0.get(idx)
    }

    pub fn contains(&self, x: &Element) -> bool {
        self.0.contains(x)
    }

    pub fn resize(&mut self, new_len: usize, value: Element) {
        self.0.resize(new_len, value)
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

    pub fn push(&mut self, value: Element) {
        self.0.push(value)
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

impl std::ops::Index<usize> for ElementList {
    type Output = Element;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for ElementList {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// As a byproduct of calculating either the vertices or the entire polytope
/// corresponding to a given section, we generate a map from ranks and indices
/// in the original polytope to ranks and indices in the section. This struct
/// encodes such a map as a vector of hash maps.
pub struct ElementHash(RankVec<HashMap<usize, usize>>);

impl ElementHash {
    pub fn get(&self, idx: isize) -> Option<&HashMap<usize, usize>> {
        self.0.get(idx)
    }

    /// Returns a map from elements on the polytope to elements in an element.
    /// If the element doesn't exist, we return `None`.
    pub fn from_element(poly: &Abstract, rank: isize, idx: usize) -> Option<Self> {
        poly.element_ref(rank, idx)?;

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
        let rank = self.0.rank();
        let mut abs = Abstract::with_capacity(rank);

        for r in -1..=rank {
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
                let el = poly.element_ref(r, idx).unwrap();
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

/// A pair of indices in a polytope.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Indices(pub usize, pub usize);

/// A section of an abstract polytope, not to be confused with a cross-section.
#[derive(Hash)]
pub struct Section {
    pub rank_lo: isize,
    pub idx_lo: usize,
    pub rank_hi: isize,
    pub idx_hi: usize,
}

impl Section {
    /// The maximal section of a polytope.
    pub fn max(rank: isize) -> Self {
        Self {
            rank_lo: -1,
            idx_lo: 0,
            rank_hi: rank,
            idx_hi: 0,
        }
    }

    pub fn height(&self) -> isize {
        self.rank_hi - self.rank_lo - 1
    }

    pub fn indices(&self) -> Indices {
        Indices(self.idx_lo, self.idx_hi)
    }
}

/// Maps the sections of a polytope with the same height to indices in a new
/// polytope. Organizes the sections first by their lower rank, then by their hash.
#[derive(Debug)]
pub struct SectionHash {
    pub rank_vec: RankVec<HashMap<Indices, usize>>,
    pub len: usize,
}

impl SectionHash {
    /// Initializes a new, empty `SectionHash` for sections of a given height
    /// in a polytope with a given rank.
    pub fn new(rank: isize, height: isize) -> Self {
        let max_rank = rank - height - 1;
        let mut rank_vec = RankVec::with_capacity(max_rank);

        for _ in -1..=max_rank {
            rank_vec.push(HashMap::new());
        }

        Self { rank_vec, len: 0 }
    }

    /// All singleton sections of a polytope.
    pub fn singletons(poly: &Abstract) -> Self {
        let rank = poly.rank();
        let mut section_hash = Self::new(rank, -1);
        let mut len = 0;

        for (rank, elements) in poly.ranks.iter().enumerate() {
            let rank = rank as isize - 1;

            for i in 0..elements.len() {
                section_hash.rank_vec[rank].insert(Indices(i, i), len);

                len += 1;
            }
        }

        section_hash.len = len;
        section_hash
    }

    /// Gets the index of a section in the hash, and whether it already existed
    /// or was just added.
    pub fn get(&mut self, section: Section) -> usize {
        use std::collections::hash_map::Entry;

        match self.rank_vec[section.rank_lo].entry(section.indices()) {
            Entry::Occupied(idx) => *idx.get(),
            Entry::Vacant(entry) => {
                let len = self.len;
                entry.insert(len);
                self.len += 1;

                len
            }
        }
    }
}
