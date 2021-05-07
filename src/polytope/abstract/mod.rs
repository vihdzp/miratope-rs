pub mod elements;
pub mod flag;
pub mod rank;

use std::collections::HashMap;

use self::{
    elements::{
        Element, ElementHash, ElementList, Section, SectionHash, Subelements, Subsupelements,
    },
    flag::{FlagEvent, FlagIter},
    rank::{Rank, RankVec},
};
use super::Polytope;
use crate::lang::name::{Abs, AbsData, Name};

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
#[derive(Debug, Clone)]
pub struct Abstract {
    pub ranks: RankVec<ElementList>,
    name: Name<Abs>,
}

impl Abstract {
    /// Initializes a polytope with an empty element list.
    pub fn new() -> Self {
        Self::from_vec(RankVec::new())
    }

    /// Initializes a new polytope with the capacity needed to store elements up
    /// to a given rank.
    pub fn with_capacity(rank: Rank) -> Self {
        Self::from_vec(RankVec::with_capacity(rank))
    }

    /// Initializes a polytope from a vector of element lists.
    pub fn from_vec(ranks: RankVec<ElementList>) -> Self {
        let name = if ranks.0.len() >= 2 {
            let rank = ranks.rank();
            let n = ranks[rank - Rank::new(1)].len();
            Name::generic(n, rank)
        } else {
            Name::Nullitope
        };

        Self { ranks, name }
    }

    /// Returns the rank of the polytope.
    pub fn rank(&self) -> Rank {
        self.ranks.rank()
    }

    /// Returns a reference to the minimal element of the polytope.
    pub fn min(&self) -> &Element {
        self.element_ref(Rank::new(-1), 0).unwrap()
    }

    /// Returns a reference to the maximal element of the polytope.
    pub fn max(&self) -> &Element {
        self.element_ref(self.rank(), 0).unwrap()
    }

    /// Pushes a new element list, assuming that the superelements of the
    /// maximal rank **have** already been set. If they haven't already been
    /// set, use [`push_subs`](Self::push_subs) instead.    
    pub fn push(&mut self, elements: ElementList) {
        self.ranks.push(elements);
    }

    /// Pushes a given element into the vector of elements of a given rank.
    pub fn push_at(&mut self, rank: Rank, el: Element) {
        self[rank].push(el);
    }

    /// Pushes a given element into the vector of elements of a given rank.
    /// Updates the superelements of its subelements automatically.
    pub fn push_subs_at(&mut self, rank: Rank, el: Element) {
        let i = self[rank].len();

        if rank != Rank::new(-1) {
            if let Some(lower_rank) = self.ranks.get_mut(rank - Rank::new(1)) {
                // Updates superelements of the lower rank.
                for &sub in el.subs.iter() {
                    lower_rank[sub].sups.push(i);
                }
            }
        }

        self.push_at(rank, el);
    }

    /// Pushes a new element list, assuming that the superelements of the
    /// maximal rank **haven't** already been set. If they have already been
    /// set, use [`push`](Self::push) instead.    
    pub fn push_subs(&mut self, elements: ElementList) {
        // We assume the superelements of the maximal rank haven't been set.
        if !self.ranks.is_empty() {
            for el in self[self.rank()].iter() {
                debug_assert!(el.sups.is_empty(), "The method push_subs can only be used when the superelements of the elements of the maximal rank haven't already been set.");
            }
        }

        self.push(ElementList::with_capacity(elements.len()));

        for el in elements.into_iter() {
            self.push_subs_at(self.rank(), el);
        }
    }

    /// Pushes an element list with a single empty element into the polytope. To
    /// be used in circumstances where the elements are built up in layers, as
    /// the base element.
    pub fn push_single(&mut self) {
        // If you're using this method, the polytope should be empty.
        debug_assert!(self.ranks.is_empty());

        self.push_subs(ElementList::single());
    }

    /// Pushes a minimal element with no superelements into the polytope. To be
    /// used in circumstances where the elements are built up in layers.
    pub fn push_vertices(&mut self, vertex_count: usize) {
        // If you're using this method, the polytope should consist of a single
        // minimal element.
        debug_assert_eq!(self.rank(), Rank::new(-1));

        self.push_subs(ElementList::vertices(vertex_count))
    }

    /// Pushes a maximal element into the polytope, with the facets as
    /// subelements. To be used in circumstances where the elements are built up
    /// in layers.
    pub fn push_max(&mut self) {
        let facet_count = self.el_count(self.rank());
        self.push_subs(ElementList::max(facet_count));
    }

    /// Returns a reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`element`](Self::element).
    pub fn element_ref(&self, rank: Rank, idx: usize) -> Option<&Element> {
        self.ranks.get(rank)?.get(idx)
    }

    /// Gets the indices of the vertices of an element in the polytope, if it
    /// exists.
    pub fn element_vertices(&self, rank: Rank, idx: usize) -> Option<Vec<usize>> {
        Some(ElementHash::from_element(self, rank, idx)?.to_elements(Rank::new(0)))
    }

    /// Gets both elements with a given rank and index as a polytope and the
    /// indices of its vertices on the original polytope, if it exists.
    pub fn element_and_vertices(&self, rank: Rank, idx: usize) -> Option<(Vec<usize>, Self)> {
        let element_hash = ElementHash::from_element(self, rank, idx)?;
        Some((
            element_hash.to_elements(Rank::new(0)),
            element_hash.to_polytope(self),
        ))
    }

    /// Checks whether the polytope is bounded
    pub fn full_check(&self) -> bool {
        self.is_bounded() && self.check_incidences() && self.is_dyadic()
        // && self.is_strongly_connected()
    }

    pub fn dual(&self) -> Self {
        self.try_dual().unwrap()
    }

    pub fn dual_mut(&mut self) {
        self.try_dual_mut().unwrap();
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. Also returns the indices of the vertices that
    /// form the base and the dual base, in that order.
    pub fn antiprism_and_vertices(&self) -> (Self, Vec<usize>, Vec<usize>) {
        let rank = self.rank();
        let mut section_hash = SectionHash::singletons(self);

        // We actually build the elements backwards, which is as awkward as it
        // seems. Maybe we should fix that in the future?
        let mut backwards_abs = RankVec::with_capacity(rank + Rank::new(1));
        backwards_abs.push(ElementList::max(section_hash.len));

        // Indices of base.
        let vertex_count = self.vertex_count();
        let mut vertices = Vec::with_capacity(vertex_count);

        // Indices of dual base.
        let facet_count = self.facet_count();
        let mut dual_vertices = Vec::with_capacity(facet_count);

        // Adds all elements corresponding to sections of a given height.
        for height in 0..=rank.isize() + 1 {
            let height = Rank::new(height);
            let mut new_section_hash = SectionHash::new(rank, height);
            let mut elements = ElementList::with_capacity(section_hash.len);

            for _ in 0..section_hash.len {
                elements.push(Element::new());
            }

            // Goes over all sections of the previous height, and builds the
            // sections of the current height by either changing the upper
            // element into one of its superelements, or changing the lower
            // element into one of its subelements.
            for (rank_lo, map) in section_hash.rank_vec.iter().rank_enumerate() {
                // The lower and higher ranks of our OLD sections.
                let rank_hi = rank_lo + height;

                // The indices for the bottom and top elements and the index in
                // the antiprism of the old section.
                for (indices, &idx) in map.iter() {
                    // Finds all of the subelements of our old section's
                    // lowest element.
                    for &idx_lo in self.element_ref(rank_lo, indices.0).unwrap().subs.iter() {
                        // Adds the new sections of the current height, gets
                        // their index, uses that to build the ElementList.
                        let sub = new_section_hash.get(Section {
                            rank_lo: rank_lo - Rank::new(1),
                            idx_lo,
                            rank_hi,
                            idx_hi: indices.1,
                        });

                        elements[idx].subs.push(sub);
                    }

                    // Finds all of the superelements of our old section's
                    // highest element.
                    for &idx_hi in self.element_ref(rank_hi, indices.1).unwrap().sups.iter() {
                        // Adds the new sections of the current height, gets
                        // their index, uses that to build the ElementList.
                        let sub = new_section_hash.get(Section {
                            rank_lo,
                            idx_lo: indices.0,
                            rank_hi: rank_hi + Rank::new(1),
                            idx_hi,
                        });

                        elements[idx].subs.push(sub);
                    }
                }
            }

            // We figure out where the vertices of the base and the dual base
            // were sent.
            if height == rank - Rank::new(1) {
                // We create a map from the base's vertices to the new vertices.
                for v in 0..vertex_count {
                    vertices.push(new_section_hash.get(Section {
                        rank_lo: Rank::new(0),
                        idx_lo: v,
                        rank_hi: rank,
                        idx_hi: 0,
                    }));
                }

                // We create a map from the dual base's vertices to the new vertices.
                for f in 0..facet_count {
                    dual_vertices.push(new_section_hash.get(Section {
                        rank_lo: Rank::new(-1),
                        idx_lo: 0,
                        rank_hi: rank - Rank::new(1),
                        idx_hi: f,
                    }));
                }
            }

            backwards_abs.push(elements);
            section_hash = new_section_hash;
        }

        // We built this backwards, so let's fix it.
        backwards_abs.reverse();
        let mut abs = Abstract::with_capacity(backwards_abs.rank());

        for elements in backwards_abs {
            abs.push_subs(elements);
        }

        (abs, vertices, dual_vertices)
    }

    /// Determines whether the polytope is bounded, i.e. whether it has a single
    /// minimal element and a single maximal element. A valid polytope should
    /// always return `true`.
    pub fn is_bounded(&self) -> bool {
        self.el_count(Rank::new(-1)) == 1 && self.el_count(self.rank()) == 1
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

        for r in Rank::range_inclusive_iter(Rank::new(-1), rank) {
            for (idx, el) in self[r].iter().enumerate() {
                for &sub in el.subs.iter() {
                    if let Some(r_minus_one) = r.try_sub(Rank::new(1)) {
                        if !self[r_minus_one][sub].sups.contains(&idx) {
                            return false;
                        }
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
        for r in 1..self.rank().isize() {
            let r = Rank::new(r);

            for el in self[r].iter() {
                let mut hash_sub_subs = HashMap::new();

                for &sub in el.subs.iter() {
                    let sub_el = &self[r - Rank::new(1)][sub];

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
        let p_low = Rank::new(-(min as isize));
        let p_hi = p_rank - Rank::new(!max as isize);
        let q_low = Rank::new(-(min as isize));
        let q_hi = q_rank - Rank::new(!max as isize);

        // The rank of the product.
        let rank =
            p_rank + q_rank + Rank::new(1) - Rank::new(!min as isize) - Rank::new(!max as isize);

        // Initializes the element lists. These will only contain the
        // subelements as they're generated. When they're complete, we'll call
        // push_subs for each of them into a new Abstract.
        let mut element_lists = RankVec::with_capacity(rank);
        for _ in Rank::range_inclusive_iter(Rank::new(-1), rank) {
            element_lists.push(ElementList::new());
        }

        // We add the elements of a given rank in lexicographic order of the
        // ranks. This vector memoizes how many elements of the same rank are
        // added by the time we add those of the form (p_rank, q_rank). It
        // stores this value in offset_memo[p_rank - p_low][q_rank - q_hi].
        let mut offset_memo: Vec<Vec<_>> = Vec::new();
        for p_rank in Rank::range_inclusive_iter(p_low, p_hi) {
            let mut offset_memo_row = Vec::new();

            for q_rank in Rank::range_inclusive_iter(q_low, q_hi) {
                offset_memo_row.push(
                    if p_rank == p_low || q_rank == q_hi {
                        0
                    } else {
                        offset_memo[(p_rank - p_low - Rank::new(1)).usize()]
                            [(q_rank - q_low + Rank::new(1)).usize()]
                    } + p.el_count(p_rank) * q.el_count(q_rank),
                );
            }

            offset_memo.push(offset_memo_row);
        }

        // Gets the value stored in offset_memo[p_rank - p_low][q_rank - q_hi],
        // or returns 0 if the indices are out of range.
        let offset = |p_rank: Rank, q_rank: Rank| -> _ {
            // The usize casts may overflow, but we really don't care about it.
            if let Some(offset_memo_row) =
                offset_memo.get((p_rank - p_low).try_usize().unwrap_or(usize::MAX))
            {
                offset_memo_row
                    .get((q_rank - q_low).try_usize().unwrap_or(usize::MAX))
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            }
        };

        // Every element of the product is in one to one correspondence with
        // a pair of an element from p and an element from q. This function
        // finds the position we placed it in.
        let get_element_index = |p_rank: Rank, p_idx, q_rank, q_idx| -> _ {
            if let Some(p_rank_minus_one) = p_rank.try_sub(Rank::new(1)) {
                offset(p_rank_minus_one, q_rank + Rank::new(1)) + p_idx * q.el_count(q_rank) + q_idx
            } else {
                q_idx
            }
        };

        // Adds elements in order of rank.
        for prod_rank in Rank::range_inclusive_iter(Rank::new(-1), rank) {
            // Adds elements by lexicographic order of the ranks.
            for p_els_rank in Rank::range_inclusive_iter(p_low, p_hi) {
                if let Some(q_els_rank) = prod_rank.try_sub(p_els_rank + Rank::new(min as isize)) {
                    if q_els_rank < q_low || q_els_rank > q_hi {
                        continue;
                    }

                    // Takes the product of every element in p with rank p_els_rank,
                    // with every element in q with rank q_els_rank.
                    for (p_idx, p_el) in p[p_els_rank].iter().enumerate() {
                        for (q_idx, q_el) in q[q_els_rank].iter().enumerate() {
                            let mut subs = Subelements::new();

                            // Products of p's subelements with q.
                            if p_els_rank != Rank::new(0) || min {
                                for &s in p_el.subs.iter() {
                                    subs.push(get_element_index(
                                        p_els_rank - Rank::new(1),
                                        s,
                                        q_els_rank,
                                        q_idx,
                                    ))
                                }
                            }

                            // Products of q's subelements with p.
                            if q_els_rank != Rank::new(0) || min {
                                for &s in q_el.subs.iter() {
                                    subs.push(get_element_index(
                                        p_els_rank,
                                        p_idx,
                                        q_els_rank - Rank::new(1),
                                        s,
                                    ))
                                }
                            }

                            element_lists[prod_rank].push(Element::from_subs(subs))
                        }
                    }
                }
            }
        }

        // If !min, we have to set a minimal element manually.
        if !min {
            let vertex_count = p.vertex_count() * q.vertex_count();
            element_lists[Rank::new(-1)] = ElementList::single();
            element_lists[Rank::new(0)] = ElementList::vertices(vertex_count);
        }

        // If !max, we have to set a maximal element manually.
        if !max {
            element_lists[rank] = ElementList::max(element_lists[rank - Rank::new(1)].len());
        }

        // Uses push_subs to add all of the element lists into a new polytope.
        let mut product = Self::with_capacity(element_lists.rank());
        for elements in element_lists.into_iter() {
            product.push_subs(elements);
        }

        // Sets the name of the polytope.
        let bases = vec![p.name().clone(), q.name().clone()];

        product.with_name(if min && max {
            Name::multipyramid(bases)
        } else if !min && max {
            Name::multiprism(bases)
        } else if min && !max {
            Name::multitegum(bases)
        } else {
            Name::multicomb(bases)
        })
    }
}

impl Polytope<Abs> for Abstract {
    /// The [rank](https://polytope.miraheze.org/wiki/Rank) of the polytope.
    fn rank(&self) -> Rank {
        self.ranks.rank()
    }

    fn name(&self) -> &Name<Abs> {
        &self.name
    }

    fn name_mut(&mut self) -> &mut Name<Abs> {
        &mut self.name
    }

    /// The number of elements of a given rank.
    fn el_count(&self, rank: Rank) -> usize {
        self.ranks
            .get(rank)
            .map(|elements| elements.len())
            .unwrap_or(0)
    }

    /// The element counts of the polytope.
    fn el_counts(&self) -> RankVec<usize> {
        let mut counts = RankVec::with_capacity(self.rank());

        for r in Rank::range_inclusive_iter(Rank::new(-1), self.rank()) {
            counts.push(self[r].len())
        }

        counts
    }

    /// Returns an instance of the
    /// [nullitope](https://polytope.miraheze.org/wiki/Nullitope), the unique
    /// polytope of rank &minus;1.
    fn nullitope() -> Self {
        Self {
            ranks: RankVec(vec![ElementList::min(0)]),
            name: Name::Nullitope,
        }
    }

    /// Returns an instance of the
    /// [point](https://polytope.miraheze.org/wiki/Point), the unique polytope
    /// of rank 0.
    fn point() -> Self {
        Self {
            ranks: RankVec(vec![ElementList::min(1), ElementList::max(1)]),
            name: Name::Point,
        }
    }

    /// Returns an instance of the
    /// [dyad](https://polytope.miraheze.org/wiki/Dyad), the unique polytope of
    /// rank 1.
    fn dyad() -> Self {
        let mut abs = Abstract::with_capacity(Rank::new(1));

        abs.push(ElementList::min(2));
        abs.push(ElementList::vertices(2));
        abs.push_subs(ElementList::max(2));

        abs.with_name(Name::Dyad)
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

        let mut poly = Abstract::with_capacity(Rank::new(2));

        poly.push(nullitope);
        poly.push(vertices);
        poly.push_subs(edges);
        poly.push_subs(maximal);

        poly.with_name(Name::polygon(AbsData::default(), n))
    }

    /// Converts a polytope into its dual.
    fn try_dual(&self) -> Result<Self, usize> {
        let mut clone = self.clone();
        clone.dual_mut();
        Ok(clone)
    }

    /// Converts a polytope into its dual in place.
    fn try_dual_mut(&mut self) -> Result<(), usize> {
        for elements in self.ranks.iter_mut() {
            for el in elements.iter_mut() {
                el.swap_mut();
            }
        }

        self.ranks.reverse();
        self.name = self.name.clone().dual(AbsData::default());

        Ok(())
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. Use [`Self::antiprism`] instead.
    fn try_antiprism(&self) -> Result<Self, usize> {
        Ok(self.antiprism_and_vertices().0)
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

        for (r, elements) in p.ranks.into_iter().rank_enumerate() {
            if r == Rank::new(-1) || r == rank {
                continue;
            }

            let sub_offset = el_counts[r - Rank::new(1)];
            let sup_offset = el_counts[r + Rank::new(1)];

            for mut el in elements.into_iter() {
                if r != Rank::new(0) {
                    for sub in el.subs.iter_mut() {
                        *sub += sub_offset;
                    }
                }

                if r != rank - Rank::new(1) {
                    for sup in el.sups.iter_mut() {
                        *sup += sup_offset;
                    }
                }

                self.push_at(r, el);
            }
        }

        self.name = Name::compound(vec![(1, self.name().clone()), (1, p.name)]);

        Ok(())
    }

    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, rank: Rank, idx: usize) -> Option<Self> {
        Some(ElementHash::from_element(self, rank, idx)?.to_polytope(self))
    }

    fn flag_events(&self) -> FlagIter {
        FlagIter::new(&self)
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
        let max = self.max().clone();

        self.push_subs_at(rank, max);
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
        let min = self.min().clone();
        self[Rank::new(-1)].push(min);
        self.ranks.insert(Rank::new(-1), ElementList::max(2));
    }

    /// Determines whether a given polytope is
    /// [orientable](https://polytope.miraheze.org/wiki/Orientability).
    fn orientable(&self) -> bool {
        for flag_event in self.flag_events() {
            if flag_event == FlagEvent::NonOrientable {
                return false;
            }
        }

        true
    }
}

impl std::ops::Index<Rank> for Abstract {
    type Output = ElementList;

    fn index(&self, index: Rank) -> &Self::Output {
        &self.ranks[index]
    }
}

impl std::ops::IndexMut<Rank> for Abstract {
    fn index_mut(&mut self, index: Rank) -> &mut Self::Output {
        &mut self.ranks[index]
    }
}

impl IntoIterator for Abstract {
    type Item = ElementList;

    type IntoIter = crate::polytope::r#abstract::rank::IntoIter<ElementList>;

    fn into_iter(self) -> Self::IntoIter {
        self.ranks.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::{super::Polytope, rank::Rank, Abstract};

    /// Returns a bunch of varied polytopes to run general tests on. Use only
    /// for tests that should work on **everything** you give it!
    fn test_polytopes() -> [Abstract; 19] {
        [
            Abstract::nullitope(),
            Abstract::point(),
            Abstract::dyad(),
            Abstract::polygon(2),
            Abstract::polygon(3),
            Abstract::polygon(4),
            Abstract::polygon(5),
            Abstract::polygon(10),
            Abstract::hypercube(Rank::new(3)),
            Abstract::hypercube(Rank::new(4)),
            Abstract::hypercube(Rank::new(5)),
            Abstract::simplex(Rank::new(3)),
            Abstract::simplex(Rank::new(4)),
            Abstract::simplex(Rank::new(5)),
            Abstract::orthoplex(Rank::new(3)),
            Abstract::orthoplex(Rank::new(4)),
            Abstract::orthoplex(Rank::new(5)),
            Abstract::duoprism(&Abstract::polygon(6), &Abstract::polygon(7)),
            Abstract::dyad().ditope().ditope().ditope().ditope(),
        ]
    }

    #[test]
    /// Checks that a nullitope is generated correctly.
    fn nullitope_check() {
        let nullitope = Abstract::nullitope();

        assert_eq!(
            nullitope.el_counts().0,
            vec![1],
            "Nullitope element counts don't match expected value."
        );
        assert!(nullitope.full_check(), "Nullitope is invalid.");
    }

    #[test]
    /// Checks that a point is generated correctly.
    fn point_check() {
        let point = Abstract::point();

        assert_eq!(
            point.el_counts().0,
            vec![1, 1],
            "Point element counts don't match expected value."
        );
        assert!(point.full_check(), "Point is invalid.");
    }

    #[test]
    /// Checks that a dyad is generated correctly.
    fn dyad_check() {
        let dyad = Abstract::dyad();

        assert_eq!(
            dyad.el_counts().0,
            vec![1, 2, 1],
            "Dyad element counts don't match expected value."
        );
        assert!(dyad.full_check(), "Dyad is invalid.");
    }

    #[test]
    /// Checks that polygons are generated correctly.
    fn polygon_check() {
        for n in 2..=10 {
            let polygon = Abstract::polygon(n);

            assert_eq!(
                polygon.el_counts().0,
                vec![1, n, n, 1],
                "{}-gon element counts don't match expected value.",
                n
            );
            assert!(polygon.full_check(), "{}-gon is invalid.", n);
        }
    }

    #[test]
    /// Checks that polygonal duopyramids are generated correctly.
    fn duopyramid_check() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                let duopyramid = Abstract::duopyramid(&polygons[m - 2], &polygons[n - 2]);

                assert_eq!(
                    duopyramid.el_counts().0,
                    vec![
                        1,
                        m + n,
                        m + n + m * n,
                        2 * m * n + 2,
                        m + n + m * n,
                        m + n,
                        1
                    ],
                    "{}-{} duopyramid element counts don't match expected value.",
                    m,
                    n
                );
                assert!(
                    duopyramid.full_check(),
                    "{}-{} duopyramid is invalid.",
                    m,
                    n
                );
            }
        }
    }

    #[test]
    /// Checks that polygonal duoprisms are generated correctly.
    fn duoprism_check() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                let duoprism = Abstract::duoprism(&polygons[m - 2], &polygons[n - 2]);

                assert_eq!(
                    duoprism.el_counts().0,
                    vec![1, m * n, 2 * m * n, m + n + m * n, m + n, 1],
                    "{}-{} duoprism element counts don't match expected value.",
                    m,
                    n
                );
                assert!(duoprism.full_check(), "{}-{} duoprism is invalid.", m, n);
            }
        }
    }

    #[test]
    /// Checks that polygonal duotegums are generated correctly.
    fn duotegum_check() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                let duotegum = Abstract::duotegum(&polygons[m - 2], &polygons[n - 2]);

                assert_eq!(
                    duotegum.el_counts().0,
                    vec![1, m + n, m + n + m * n, 2 * m * n, m * n, 1],
                    "{}-{} duotegum element counts don't match expected value.",
                    m,
                    n
                );
                assert!(duotegum.full_check(), "{}-{} duotegum is invalid.", m, n);
            }
        }
    }

    #[test]
    /// Checks that polygonal duocombs are generated correctly.
    fn duocomb_check() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                let duocomb = Abstract::duocomb(&polygons[m - 2], &polygons[n - 2]);

                assert_eq!(
                    duocomb.el_counts().0,
                    vec![1, m * n, 2 * m * n, m * n, 1],
                    "{}-{} duocomb element counts don't match expected value.",
                    m,
                    n
                );
                assert!(duocomb.full_check(), "{}-{} duocomb is invalid.", m, n);
            }
        }
    }

    /// Calculates `n` choose `k`.
    fn choose(n: usize, k: usize) -> usize {
        let mut res = 1;

        for r in 0..k {
            res *= n - r;
            res /= r + 1;
        }

        res
    }

    #[test]
    /// Checks that simplices are generated correctly.
    fn simplex_check() {
        for n in Rank::range_inclusive_iter(Rank::new(-1), Rank::new(5)) {
            let simplex = Abstract::simplex(n);

            for k in Rank::range_inclusive_iter(Rank::new(-1), n) {
                assert_eq!(
                    simplex.el_count(k),
                    choose(n.0, k.0),
                    "{}-simplex {}-element counts don't match up",
                    n,
                    k
                );
            }

            assert!(simplex.full_check(), "{}-simplex is invalid.", n)
        }
    }

    #[test]
    /// Checks that hypercubes are generated correctly.
    fn hypercube_check() {
        for n in Rank::range_inclusive_iter(Rank::new(-1), Rank::new(5)) {
            let hypercube = Abstract::hypercube(n);

            for k in Rank::range_inclusive_iter(Rank::new(0), n) {
                assert_eq!(
                    hypercube.el_count(k),
                    choose(n.usize(), k.usize()) * 2u32.pow((n - k).u32()) as usize,
                    "{}-hypercube {}-element counts don't match up",
                    n,
                    k
                );
            }

            assert!(hypercube.full_check(), "{}-hypercube is invalid.", n)
        }
    }

    #[test]
    /// Checks that orthoplices are generated correctly.
    fn orthoplex_check() {
        for n in Rank::range_inclusive_iter(Rank::new(-1), Rank::new(5)) {
            let orthoplex = Abstract::orthoplex(n);

            for k in Rank::range_iter(Rank::new(-1), n) {
                assert_eq!(
                    orthoplex.el_count(k),
                    choose(n.usize(), k.0) * 2u32.pow(k.0 as u32) as usize,
                    "{}-orthoplex {}-element counts don't match up",
                    n,
                    k
                );
            }

            assert!(orthoplex.full_check(), "{}-orthoplex is invalid.", n)
        }
    }

    #[test]
    /// Checks that duals are generated correctly.
    fn dual_check() {
        let mut polytopes = test_polytopes();

        for (idx, poly) in polytopes.iter_mut().enumerate() {
            let el_counts = poly.el_counts();

            poly.dual_mut();

            // The element counts of the dual should be the same as the reversed
            // element counts of the original.
            let mut du_el_counts_rev = poly.el_counts();
            du_el_counts_rev.reverse();
            assert_eq!(
                el_counts.0, du_el_counts_rev.0,
                "Dual element counts of test polytope #{} don't match expected value.",
                idx
            );
            assert!(
                poly.full_check(),
                "Dual of test polytope #{} is invalid.",
                idx
            );
        }
    }
}
