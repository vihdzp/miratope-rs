use derive_deref::{Deref, DerefMut};
use std::collections::HashMap;

use crate::polytope::{rank::RankVec, Element, ElementList, Polytope, Subelements, Subsupelements};

type ElementHash = RankVec<HashMap<usize, usize>>;

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
/// of the elements of the previous rank.
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
            for &sub in el.subs.iter() {
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

        self.push(ElementList::with_capacity(elements.len()));
        let rank = self.rank();

        for el in elements.into_iter() {
            self.push_at(rank, el);
        }
    }

    /// Pushes an element list with a single empty element into the polytope. To
    /// be used in circumstances where the elements are built up in layers, as
    /// the base element.
    pub fn push_single(&mut self) {
        // If you're using this method, the polytope should be empty.
        debug_assert!(self.is_empty());

        self.push_subs(ElementList::single());
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

    fn get_element(&self, rank: isize, idx: usize) -> Option<&Element> {
        self.0.get(rank)?.get(idx)
    }

    /// Returns an [`ElementHash`], used as an auxiliary data structure for
    /// operations involving elements.
    ///
    /// As a byproduct of calculating either the vertices or the entire polytope
    /// corresponding to a given element, we generate a map from ranks and
    /// indices in the original polytope to ranks and indices in the element.
    /// This function returns such a map, encoded as a vector of hash maps.
    ///
    /// If the element doesn't exist, we return `None`.
    fn element_hash(&self, rank: isize, idx: usize) -> Option<ElementHash> {
        self.get_element(rank, idx)?;

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
                for &sub in self[r as isize][idx].subs.iter() {
                    let len = prev_hash.len();
                    if !prev_hash.contains_key(&sub) {
                        prev_hash.insert(sub, len);
                    }
                }
            }
        }

        Some(hashes)
    }

    /// Gets the indices of the vertices of a given element in a polytope.
    fn vertices_from_element_hash(element_hash: &ElementHash) -> Vec<usize> {
        if let Some(hash_vertices) = element_hash.get(0) {
            let mut vertices = Vec::new();
            vertices.resize(hash_vertices.len(), 0);

            for (&sub, &idx) in hash_vertices {
                vertices[idx] = sub;
            }

            vertices
        } else {
            Vec::new()
        }
    }

    /// Gets the indices of the vertices of a given element in a polytope.
    fn polytope_from_element_hash(&self, element_hash: &ElementHash) -> Self {
        let rank = element_hash.rank();
        let mut abs = Self::with_capacity(rank);

        for r in -1..=rank {
            let mut elements = ElementList::new();
            let hash = &element_hash[r];

            for _ in 0..hash.len() {
                elements.push(Element::new());
            }

            for (&idx, &new_idx) in hash {
                let el = self.get_element(r, idx).unwrap();
                let mut new_el = Element::new();

                if let Some(prev_hash) = element_hash.get(r - 1) {
                    for sub in el.subs.iter() {
                        if let Some(&new_sub) = prev_hash.get(sub) {
                            new_el.subs.push(new_sub);
                        }
                    }
                }

                if let Some(next_hash) = element_hash.get(r + 1) {
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

    pub fn element_vertices(&self, rank: isize, idx: usize) -> Option<Vec<usize>> {
        Some(Self::vertices_from_element_hash(
            &self.element_hash(rank, idx)?,
        ))
    }

    pub fn element_and_vertices(&self, rank: isize, idx: usize) -> Option<(Vec<usize>, Self)> {
        let element_hash = self.element_hash(rank, idx)?;

        Some((
            Self::vertices_from_element_hash(&element_hash),
            self.polytope_from_element_hash(&element_hash),
        ))
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

                for &sub in el.subs.iter() {
                    let sub_el = &self[r - 1][sub];

                    for &sub_sub in sub_el.subs.iter() {
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

    /// Takes the [direct product](https://en.wikipedia.org/wiki/Direct_product#Direct_product_of_binary_relations)
    /// of two polytopes. If the `min` flag is turned off, it ignores the
    /// minimal elements of both of the factors and adds one at the end. The
    /// `max` flag works analogously.
    ///
    /// The elements of this product are in one to one correspondence to pairs
    /// of elements in the set of polytopes. The elements of a specific rank are
    /// sorted first by lexicographic order of the ranks, then by lexicographic
    /// order of the elements.
    pub fn product(p: &Self, q: &Self, min: bool, max: bool) -> Self {
        // The ranks of p and q.
        let p_rank = p.rank();
        let q_rank = q.rank();

        // The lowest and highest ranks we'll use to take products in p and q.
        let p_low = -(min as isize);
        let p_hi = p_rank - (!max as isize);
        let q_low = -(min as isize);
        let q_hi = q_rank - (!max as isize);

        // The rank of the product.
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
                        let mut subs = Subelements::new();

                        // Products of p's subelements with q.
                        if p_els_rank != 0 || min {
                            for &s in p_el.subs.iter() {
                                subs.push(get_element_index(p_els_rank - 1, s, q_els_rank, q_idx))
                            }
                        }

                        // Products of q's subelements with p.
                        if q_els_rank != 0 || min {
                            for &s in q_el.subs.iter() {
                                subs.push(get_element_index(p_els_rank, p_idx, q_els_rank - 1, s))
                            }
                        }

                        element_lists[prod_rank].push(Element::from_subs(subs))
                    }
                }
            }
        }

        // If !min, we have to set a minimal element manually.
        if !min {
            let vertex_count = p.el_count(0) * q.el_count(0);
            element_lists[-1] = ElementList::single();
            element_lists[0] = ElementList::vertices(vertex_count);
        }

        // If !max, we have to set a maximal element manually.
        if !max {
            element_lists[rank] = ElementList::max(element_lists[rank - 1].len());
        }

        // Uses push_subs to add all of the element lists into a new polytope.
        let mut product = Self::with_capacity(element_lists.rank());

        for elements in element_lists.into_iter() {
            product.push_subs(elements);
        }

        product
    }
}

impl Polytope for Abstract {
    /// The return type of [`dual`](Self::dual).
    type Dual = Self;

    /// The return type of [`dual_mut`](Self::dual_mut).
    type DualMut = ();

    /// The [rank](https://polytope.miraheze.org/wiki/Rank) of the polytope.
    fn rank(&self) -> isize {
        self.0.rank()
    }

    /// The number of elements of a given rank.
    fn el_count(&self, rank: isize) -> usize {
        if let Some(els) = self.get(rank) {
            els.len()
        } else {
            0
        }
    }

    /// The element counts of the polytope.
    fn el_counts(&self) -> RankVec<usize> {
        let mut counts = RankVec::with_capacity(self.rank());

        for r in -1..=self.rank() {
            counts.push(self[r].len())
        }

        counts
    }

    /// Returns an instance of the
    /// [nullitope](https://polytope.miraheze.org/wiki/Nullitope), the unique
    /// polytope of rank &minus;1.
    fn nullitope() -> Self {
        Abstract::from_vec(vec![ElementList::min(0)])
    }

    /// Returns an instance of the
    /// [point](https://polytope.miraheze.org/wiki/Point), the unique polytope
    /// of rank 0.
    fn point() -> Self {
        Abstract::from_vec(vec![ElementList::min(1), ElementList::max(1)])
    }

    /// Returns an instance of the
    /// [dyad](https://polytope.miraheze.org/wiki/Dyad), the unique polytope of
    /// rank 1.
    fn dyad() -> Self {
        let mut abs = Abstract::with_capacity(1);

        abs.push(ElementList::min(2));
        abs.push(ElementList::vertices(2));
        abs.push_subs(ElementList::max(2));

        abs
    }

    /// Returns an instance of a [polygon](https://polytope.miraheze.org/wiki/Polygon)
    /// with a given number of sides.
    fn polygon(n: usize) -> Self {
        assert!(n >= 2, "A polygon must have at least 2 sides.");

        let nullitope = ElementList::min(n);
        let vertices = ElementList::vertices(n);
        let mut edges = ElementList::with_capacity(n);
        let maximal = ElementList::max(n);

        for i in 0..n {
            edges.push(Element::from_subs(Subelements(vec![i % n, (i + 1) % n])));
        }

        let mut poly = Abstract::with_capacity(2);

        poly.push(nullitope);
        poly.push(vertices);
        poly.push_subs(edges);
        poly.push_subs(maximal);

        poly
    }

    /// Converts a polytope into its dual.
    fn dual(&self) -> Self::Dual {
        let mut clone = self.clone();
        clone.dual_mut();
        clone
    }

    /// Converts a polytope into its dual in place.
    fn dual_mut(&mut self) -> Self::DualMut {
        for elements in self.iter_mut() {
            for el in elements.iter_mut() {
                el.swap_mut();
            }
        }

        self.reverse();
    }

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks.
    fn append(&mut self, p: Self) -> Result<(), ()> {
        let rank = self.rank();

        // The polytopes must have the same ranks.
        if rank != p.rank() {
            return Err(());
        }

        let el_counts = self.el_counts();

        for (r, elements) in p.into_iter().rank_enumerate() {
            if r == -1 || r == rank {
                continue;
            }

            let sub_offset = el_counts[r - 1];
            let sup_offset = el_counts[r + 1];

            for mut el in elements.into_iter() {
                if r != 0 {
                    for sub in el.subs.iter_mut() {
                        *sub += sub_offset;
                    }
                }

                if r != rank - 1 {
                    for sup in el.sups.iter_mut() {
                        *sup += sup_offset;
                    }
                }

                self.push_at(r, el);
            }
        }

        Ok(())
    }

    fn element(&self, rank: isize, idx: usize) -> Option<Self> {
        Some(self.polytope_from_element_hash(&self.element_hash(rank, idx)?))
    }

    fn element_fig(&self, _rank: isize, _idx: usize) -> Option<Self> {
        todo!()
    }

    fn section(
        &self,
        _rank_lo: isize,
        _idx_lo: usize,
        _rank_hi: isize,
        _idx_hi: usize,
    ) -> Option<Self> {
        todo!()
    }

    /// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// from two polytopes.
    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::product(p, q, true, true)
    }

    /// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
    /// from two polytopes.
    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::product(p, q, false, true)
    }

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::product(p, q, true, false)
    }

    /// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
    /// from two polytopes.
    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::product(p, q, false, false)
    }

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope.
    fn ditope(&self) -> Self {
        let mut clone = self.clone();
        clone.ditope_mut();
        clone
    }

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope in place.
    fn ditope_mut(&mut self) {
        let rank = self.rank();
        let max = self[rank][0].clone();

        self.push_at(rank, max);
        self.push_subs(ElementList::max(2));
    }

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope.
    fn hosotope(&self) -> Self {
        let mut clone = self.clone();
        clone.hosotope_mut();
        clone
    }

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope in place.
    fn hosotope_mut(&mut self) {
        let min = self[-1][0].clone();

        self[-1].push(min);
        self.insert(-1, ElementList::max(2));
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope.
    fn antiprism(&self) -> Self {
        todo!()
    }

    /// Determines whether a given polytope is
    /// [orientable](https://polytope.miraheze.org/wiki/Orientability).
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

impl IntoIterator for Abstract {
    type Item = ElementList;

    type IntoIter = crate::polytope::rank::IntoIter<ElementList>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
