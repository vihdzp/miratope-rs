use derive_deref::{Deref, DerefMut};
use std::collections::{HashMap, HashSet};

use crate::polytope::{rank::RankVec, Element, ElementList, Polytope, Subelements};

/// The [ranked poset](https://en.wikipedia.org/wiki/Graded_poset) corresponding
/// to an [abstract polytope](https://polytope.miraheze.org/wiki/Abstract_polytope).
/// It stores the indices of both the subelements and superelements of each
/// element.
///
/// # How to use?
/// The fact that we store both subelements and superelements is quite useful
/// for many algorithms. However, it becomes inconvenient when actually building
/// a polytope, since most of the time, we can only easily generate subelements.
///
/// To get around this, we provide a [`push_subs`](Abstract::push_subs) method.
/// Instead of manually having to set the superelements in the polytope, one can
/// instead provide an [`ElementList`] whose elements have their superelements
/// set to empty vectors. This method will automatically set the superelements
/// of the subelements of the previous rank.
#[derive(Debug, Clone, Deref, DerefMut)]
pub struct Abstract(RankVec<ElementList>);

impl Abstract {
    /// Initializes a polytope with an empty element list.
    pub fn new() -> Self {
        Abstract(RankVec::new())
    }

    /// Initializes a new polytope with the capacity needed to store elements up
    /// to a given rank.
    pub fn with_capacity(rank: isize) -> Self {
        Abstract(RankVec::with_capacity(rank))
    }

    /// Initializes a polytope from a vector of element lists.
    pub fn from_vec(vec: Vec<ElementList>) -> Self {
        Abstract(RankVec(vec))
    }

    /// Returns a reference to the minimal element of the polytope.
    pub fn min(&self) -> &Element {
        &self[0][0]
    }

    /// Pushes a new element list, assuming that the superelements of the
    /// maximal rank **have** already been set. If they haven't already been
    /// set, use [`push_subs`](Self::push_subs) instead.    
    pub fn push(&mut self, elements: ElementList) {
        self.0.push(elements);
    }

    /// Pushes a given element into the vector of elements of a given rank.
    pub fn push_at(&mut self, rank: isize, el: Element) {
        let i = self[rank].len();

        if let Some(lower_rank) = self.get_mut(rank - 1) {
            // Updates superelements of the lower rank.
            for &sub in &el.subs.0 {
                lower_rank[sub].sups.push(i);
            }
        }

        self[rank].push(el);
    }

    /// Pushes a new element list, assuming that the superelements of the
    /// maximal rank **haven't** already been set. If they have already been
    /// set, use [`push`](Self::push) instead.    
    pub fn push_subs(&mut self, elements: ElementList) {
        // We assume the superelements of the maximal rank haven't been set.
        if !self.is_empty() {
            for el in self[self.rank()].iter() {
                debug_assert!(el.sups.is_empty(), "The method push_subs can only been used when the superelements of the elements of the maximal rank haven't already been set.");
            }
        }

        self.0.push(ElementList::with_capacity(elements.len()));
        let rank = self.rank();

        for el in elements.0.into_iter() {
            self.push_at(rank, el);
        }
    }

    /// Pushes a minimal element with no superelements into the polytope. To be
    /// used in circumstances where the elements are built up in layers.
    pub fn push_min(&mut self) {
        // If you're using this method, the polytope should be empty.
        debug_assert!(self.is_empty());

        self.push_subs(ElementList::min(0));
    }

    /// Pushes a minimal element with no superelements into the polytope. To be
    /// used in circumstances where the elements are built up in layers.
    pub fn push_vertices(&mut self, vertex_count: usize) {
        // If you're using this method, the polytope should consist of a single
        // minimal element.
        debug_assert_eq!(self.rank(), -1);

        self.push_subs(ElementList::vertices(vertex_count))
    }

    /// Pushes a maximal element into the polytope, with the facets as
    /// subelements. To be used in circumstances where the elements are built up
    /// in layers.
    pub fn push_max(&mut self) {
        let facet_count = self.el_count(self.rank());
        self.push_subs(ElementList::max(facet_count));
    }

    /// Converts a polytope into its dual.
    pub fn dual(&self) -> Self {
        let mut clone = self.clone();
        clone.dual_mut();
        clone
    }

    /// Converts a polytope into its dual in place.
    pub fn dual_mut(&mut self) {
        for elements in self.iter_mut() {
            for el in elements.iter_mut() {
                el.swap_mut();
            }
        }

        self.reverse();
    }

    /// Gets the indices of the vertices of a given element in a polytope.
    pub fn element_vertices(&self, rank: isize, idx: usize) -> Option<Vec<usize>> {
        // A nullitope doesn't have vertices.
        if rank == -1 {
            return None;
        }

        let mut indices = vec![idx];

        // Gets subindices of subindices, until reaching the vertices.
        for r in (1..=rank).rev() {
            let mut hash_subs = HashSet::new();

            for idx in indices {
                for &sub in &self[r][idx].subs.0 {
                    hash_subs.insert(sub);
                }
            }

            indices = hash_subs.into_iter().collect();
        }

        Some(indices)
    }

    /// Gets the element with a given rank and index as a polytope.
    pub fn element(&self, _rank: isize, _idx: usize) -> Option<Self> {
        todo!()
    }

    pub fn section(
        &self,
        _rank_low: isize,
        _idx_low: usize,
        _rank_hi: isize,
        _idx_hi: usize,
    ) -> Self {
        // assert incidence.

        todo!()
    }

    /// Checks whether the polytope is bounded
    pub fn full_check(&self) -> bool {
        self.is_bounded() && self.check_incidences() && self.is_dyadic()
        // && self.is_strongly_connected()
    }

    /// Determines whether the polytope is bounded, i.e. whether it has a single
    /// minimal element and a single maximal element. A valid polytope should
    /// always return `true`.
    pub fn is_bounded(&self) -> bool {
        self.el_count(-1) == 1 && self.el_count(self.rank()) == 1
    }

    /// Checks whether subelements and superelements match up, and whether they
    /// all refer to valid elements in the polytope. If this returns `false`,
    /// then either the polytope hasn't fully built up, or there's something
    /// seriously wrong.
    pub fn check_incidences(&self) -> bool {
        let rank = self.rank();
        if !self[rank][0].sups.is_empty() {
            return false;
        }

        for r in -1..=rank {
            for (idx, el) in self[r].iter().enumerate() {
                for &sub in el.subs.iter() {
                    if !self[r - 1][sub].sups.contains(&idx) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Determines whether the polytope satisfies the diamond property. A valid
    /// non-fissary polytope should always return `true`.
    pub fn is_dyadic(&self) -> bool {
        #[derive(PartialEq)]
        enum Count {
            Once,
            Twice,
        }

        // For every element, by looking through the subelements of its
        // subelements, we need to find each exactly twice.
        for r in 1..self.rank() {
            for el in self[r].iter() {
                let mut hash_sub_subs = HashMap::new();

                for &sub in &el.subs.0 {
                    let sub_el = &self[r - 1][sub];

                    for &sub_sub in &sub_el.subs.0 {
                        match hash_sub_subs.get(&sub_sub) {
                            // Found for the first time.
                            None => hash_sub_subs.insert(sub_sub, Count::Once),

                            // Found for the second time.
                            Some(Count::Once) => hash_sub_subs.insert(sub_sub, Count::Twice),

                            // Found for the third time?! Abort!
                            Some(Count::Twice) => return false,
                        };
                    }
                }

                // If any subsubelement was found only once, this also
                // violates the diamond property.
                for (_, count) in hash_sub_subs.into_iter() {
                    if count == Count::Once {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Determines whether the polytope is connected. A valid non-compound
    /// polytope should always return `true`.
    pub fn is_connected(&self) -> bool {
        todo!()
    }

    /// Determines whether the polytope is strongly connected. A valid
    /// non-compound polytope should always return `true`.
    pub fn is_strongly_connected(&self) -> bool {
        todo!()
    }

    /// Takes the direct product of two polytopes. If the `min` flag is
    /// turned off, it ignores the minimal elements of both of the factors and
    /// adds one at the end. The `max` flag works analogously.
    ///
    /// The elements of this product are in one to one correspondence to pairs
    /// of elements in the set of polytopes. The elements of a specific rank are
    /// sorted first by lexicographic order of the ranks, then by lexicographic
    /// order of the elements.
    pub fn product(p: &Self, q: &Self, min: bool, max: bool) -> Self {
        let p_rank = p.rank();
        let q_rank = q.rank();

        let p_low = -(min as isize);
        let p_hi = p_rank - (!max as isize);

        let q_low = -(min as isize);
        let q_hi = q_rank - (!max as isize);

        let rank = p_rank + q_rank + 1 - (!min as isize) - (!max as isize);

        // Initializes the element lists. These will only contain the
        // subelements as they're generated. When they're complete, we'll call
        // push_subs for each of them into a new Abstract.
        let mut element_lists = RankVec::with_capacity(rank);
        for _ in -1..=rank {
            element_lists.push(ElementList::new());
        }

        // We add the elements of a given rank in lexicographic order of the
        // ranks. This vector memoizes how many elements of the same rank are
        // added by the time we add those of the form (p_rank, q_rank). It
        // stores this value in offset_memo[p_rank - p_low][q_rank - q_hi].
        let mut offset_memo: Vec<Vec<_>> = Vec::new();
        for p_rank in p_low..=p_hi {
            let mut offset_memo_row = Vec::new();

            for q_rank in q_low..=q_hi {
                offset_memo_row.push(
                    if p_rank == p_low || q_rank == q_hi {
                        0
                    } else {
                        offset_memo[(p_rank - p_low - 1) as usize][(q_rank - q_low + 1) as usize]
                    } + p.el_count(p_rank) * q.el_count(q_rank),
                );
            }

            offset_memo.push(offset_memo_row);
        }

        // Gets the value stored in offset_memo[p_rank - p_low][q_rank - q_hi],
        // or returns 0 if the indices are out of range.
        let offset = |p_rank: isize, q_rank: isize| -> _ {
            // The usize casts may overflow, but we really don't care about it.
            if let Some(offset_memo_row) = offset_memo.get((p_rank - p_low) as usize) {
                offset_memo_row
                    .get((q_rank - q_low) as usize)
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            }
        };

        // Every element of the product is in one to one correspondence with
        // a pair of an element from p and an element from q. This function
        // finds the position we placed it in.
        let get_element_index = |p_rank, p_idx, q_rank, q_idx| -> _ {
            offset(p_rank - 1, q_rank + 1) + p_idx * q.el_count(q_rank) + q_idx
        };

        // Adds elements in order of rank.
        for prod_rank in -1..=rank {
            // Adds elements by lexicographic order of the ranks.
            for p_els_rank in p_low..=p_hi {
                let q_els_rank = prod_rank - p_els_rank - (min as isize);
                if q_els_rank < q_low || q_els_rank > q_hi {
                    continue;
                }

                // Takes the product of every element in p with rank p_els_rank,
                // with every element in q with rank q_els_rank.
                for (p_idx, p_el) in p[p_els_rank].iter().enumerate() {
                    for (q_idx, q_el) in q[q_els_rank].iter().enumerate() {
                        let mut subs = Vec::new();

                        // Products of p's subelements with q.
                        if p_els_rank != 0 || min {
                            for &s in &p_el.subs.0 {
                                subs.push(get_element_index(p_els_rank - 1, s, q_els_rank, q_idx))
                            }
                        }

                        // Products of q's subelements with p.
                        if q_els_rank != 0 || min {
                            for &s in &q_el.subs.0 {
                                subs.push(get_element_index(p_els_rank, p_idx, q_els_rank - 1, s))
                            }
                        }

                        element_lists[prod_rank].push(Element::from_subs(Subelements(subs)))
                    }
                }
            }
        }

        // If !min, we have to set a minimal element manually.
        if !min {
            let vertex_count = p.el_count(0) * q.el_count(0);
            element_lists[-1] = ElementList::empty();
            element_lists[0] = ElementList::vertices(vertex_count);
        }

        // If !max, we have to set a maximal element manually.
        if !max {
            element_lists[rank] = ElementList::max(element_lists[rank - 1].len());
        }

        // Uses push_subs to add all of the element lists into a new polytope.
        let mut product = Self::with_capacity(element_lists.rank());

        for elements in element_lists.0.into_iter() {
            product.push_subs(elements);
        }

        product
    }
}

impl Polytope for Abstract {
    /// Returns the rank of the polytope.
    fn rank(&self) -> isize {
        self.len() as isize - 2
    }

    /// Gets the number of elements of a given rank.
    fn el_count(&self, rank: isize) -> usize {
        if let Some(els) = self.get(rank) {
            els.0.len()
        } else {
            0
        }
    }

    /// Gets the number of elements of all ranks.
    fn el_counts(&self) -> RankVec<usize> {
        let mut counts = RankVec::with_capacity(self.rank());

        for r in -1..=self.rank() {
            counts.push(self[r].len())
        }

        counts
    }

    /// Returns the unique polytope of rank −1.
    fn nullitope() -> Self {
        Abstract::from_vec(vec![ElementList::min(0)])
    }

    /// Returns the unique polytope of rank 0.
    fn point() -> Self {
        Abstract::from_vec(vec![ElementList::min(1), ElementList::max(1)])
    }

    /// Returns the unique polytope of rank 1.
    fn dyad() -> Self {
        let mut abs = Abstract::with_capacity(1);
        abs.push(ElementList::min(2));
        abs.push(ElementList::vertices(2));
        abs.push_subs(ElementList::max(2));
        abs
    }

    /// Returns the unique polytope of rank 2 with a given amount of vertices.
    fn polygon(n: usize) -> Self {
        assert!(n >= 2);

        let nullitope = ElementList::min(n);
        let mut vertices = ElementList::with_capacity(n);
        let mut edges = ElementList::with_capacity(n);
        let maximal = ElementList::max(n);

        for i in 0..n {
            vertices.push(Element::from_subs(Subelements(vec![0])));

            edges.push(Element::from_subs(Subelements(vec![i % n, (i + 1) % n])));
        }

        let mut poly = Abstract::with_capacity(2);

        poly.push(nullitope);
        poly.push(vertices);
        poly.push_subs(edges);
        poly.push_subs(maximal);

        poly
    }

    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::product(p, q, true, true)
    }

    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::product(p, q, false, true)
    }

    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::product(p, q, true, false)
    }

    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::product(p, q, false, false)
    }

    fn ditope(&self) -> Self {
        let mut clone = self.clone();
        clone.ditope_mut();
        clone
    }

    fn ditope_mut(&mut self) {
        let rank = self.rank();
        let max = self[rank][0].clone();

        self.push_at(rank, max);
        self.push_subs(ElementList::max(2));
    }

    fn hosotope(&self) -> Self {
        let mut clone = self.clone();
        clone.hosotope_mut();
        clone
    }

    fn hosotope_mut(&mut self) {
        let min = self[-1][0].clone();

        self[-1].push(min);
        self.insert(-1, ElementList::max(2));
    }

    fn antiprism(&self) -> Self {
        todo!()
    }

    fn orientable(&self) -> bool {
        todo!()
    }
}

impl std::ops::Index<isize> for Abstract {
    type Output = ElementList;

    fn index(&self, index: isize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<isize> for Abstract {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
