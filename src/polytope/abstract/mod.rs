pub mod elements;
pub mod flag;
pub mod rank;

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    mem,
};

use self::{
    elements::{
        AbstractBuilder, Element, ElementHash, ElementList, ElementRef, Section, SectionHash,
        SubelementList, Subelements, Subsupelements, Superelements,
    },
    flag::{Flag, FlagEvent},
    rank::{Rank, RankVec},
};
use super::Polytope;
use crate::lang::name::{Abs, AbsData, Name};

use strum_macros::Display;

#[derive(Display)]
pub enum IncidenceType {
    #[strum(serialize = "subelement")]
    Subelement,

    #[strum(serialize = "superelement")]
    Superelement,
}

/// Represents an error in an abstract polytope.
pub enum AbstractError {
    /// The polytope is not bounded, i.e. it doesn't have a single minimal and
    /// maximal element.
    Bounded { min_count: usize, max_count: usize },

    /// The polytope has some invalid index, i.e. some element points to another
    /// non-existent element.
    Index {
        el: ElementRef,
        incidence_type: IncidenceType,
        index: usize,
    },

    /// The polytope has a consistency error, i.e. some element is incident to
    /// another but not viceversa.
    Consistency {
        el: ElementRef,
        incidence_type: IncidenceType,
        index: usize,
    },

    /// The polytope is not ranked, i.e. some element that's not minimal or not
    /// maximal lacks a subelement or superelement, respectively.
    Ranked {
        el: ElementRef,
        incidence_type: IncidenceType,
    },

    /// The polytope is not dyadic, i.e. some section of height 1 does not have
    /// exactly 4 elements.
    Dyadic { section: Section, more: bool },

    /// The polytope is not strictly connected, i.e. some section's flags don't
    /// form a connected graph under flag changes.
    Connected(Section),
}

impl std::fmt::Debug for AbstractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // The polytope is not bounded.
            AbstractError::Bounded {
                min_count: min,
                max_count: max,
            } => write!(
                f,
                "Polytope is unbounded: found {} minimal elements and {} maximal elements",
                min, max
            ),

            // The polytope has an invalid index.
            AbstractError::Index {
                el,
                incidence_type,
                index,
            } => write!(
                f,
                "Polytope has an invalid index: {} has a {} with index {}, but it doesn't exist",
                el, incidence_type, index
            ),

            AbstractError::Consistency {
                el,
                incidence_type,
                index,
            } => write!(
                f,
                "Polytope has an invalid index: {} has a {} with index {}, but not viceversa",
                el, incidence_type, index
            ),

            // The polytope is not ranked.
            AbstractError::Ranked { el, incidence_type } => write!(
                f,
                "Polytope is not ranked: {} has no {}s",
                el, incidence_type
            ),

            // The polytope is not dyadic.
            AbstractError::Dyadic { section, more } => write!(
                f,
                "Polytope is not dyadic: there are {} than 2 elements between {}",
                if *more { "more" } else { "less" },
                section
            ),

            // The polytope is not strictly connected.
            AbstractError::Connected(section) => write!(
                f,
                "Polytope is not strictly connected: {} is not connected",
                section
            ),
        }
    }
}

/// The return value for [`Abstract::is_valid`].
pub type AbstractResult<T> = Result<T, AbstractError>;

/// The [ranked poset](https://en.wikipedia.org/wiki/Graded_poset) corresponding
/// to an [abstract polytope](https://polytope.miraheze.org/wiki/Abstract_polytope).
/// It stores the indices of both the subelements and superelements of each
/// element.
///
/// # How to use?
/// The fact that we store both subelements and superelements is quite useful
/// for many algorithms. However, it becomes inconvenient when actually building
/// a polytope, since most of the time, we can only easily generate one of them.
///
/// To get around this, we provide an [`AbstractBuilder`] struct. Instead of
/// manually setting the superelements in the polytope, one can provide a
/// [`SubelementList`]. The associated methods to the struct will automatically
/// set the superelements of the polytope.
///
/// If you wish to only set some of the subelements, we provide a
/// [`Abstract::push_subs`] method, which will push a list of subelements and
/// automatically set the superelements of the previous rank, under the
/// assumption that they're empty.
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
            let n = ranks[rank.minus_one()].len();
            Name::generic(n, rank)
        } else {
            Name::Nullitope
        };

        Self { ranks, name }
    }

    /// Returns a reference to the minimal element of the polytope.
    pub fn min(&self) -> &Element {
        self.get_element(&ElementRef::new(Rank::new(-1), 0))
            .unwrap()
    }

    /// Returns a mutable reference to the minimal element of the polytope.
    pub fn min_mut(&mut self) -> &mut Element {
        self.get_element_mut(&ElementRef::new(Rank::new(-1), 0))
            .unwrap()
    }

    /// Returns a reference to the maximal element of the polytope.
    pub fn max(&self) -> &Element {
        self.get_element(&ElementRef::new(self.rank(), 0)).unwrap()
    }

    /// Returns a mutable reference to the maximal element of the polytope.
    pub fn max_mut(&mut self) -> &mut Element {
        self.get_element_mut(&ElementRef::new(self.rank(), 0))
            .unwrap()
    }

    /// Pushes a new element list, assuming that the superelements of the
    /// maximal rank **have** already been correctly set. If they haven't
    /// already been set, use [`push_subs`](Self::push_subs) instead.
    pub fn push(&mut self, elements: ElementList) {
        self.ranks.push(elements);
    }

    /// Pushes a given element into the vector of elements of a given rank.
    pub fn push_at(&mut self, rank: Rank, el: Element) {
        self[rank].push(el);
    }

    /// Pushes a given element into the vector of elements of a given rank.
    /// Updates the superelements of its subelements automatically.
    pub fn push_subs_at(&mut self, rank: Rank, sub_el: Subelements) {
        let i = self[rank].len();

        if rank != Rank::new(-1) {
            if let Some(lower_rank) = self.ranks.get_mut(rank.minus_one()) {
                // Updates superelements of the lower rank.
                for &sub in &sub_el {
                    lower_rank[sub].sups.push(i);
                }
            }
        }

        self.push_at(rank, Element::from_subs(sub_el));
    }

    /// Pushes a new element list without superelements, assuming that the
    /// superelements of the maximal rank **haven't** already been set. If they
    /// have already been set, use [`push`](Self::push) instead.
    pub fn push_subs(&mut self, subelements: SubelementList) {
        self.push(ElementList::with_capacity(subelements.len()));

        for sub_el in subelements.into_iter() {
            self.push_subs_at(self.rank(), sub_el);
        }
    }

    /// Pushes a maximal element into the polytope, with the facets as
    /// subelements. To be used in circumstances where the elements are built up
    /// in layers.
    pub fn push_max(&mut self) {
        let facet_count = self.el_count(self.rank());
        self.push_subs(SubelementList::max(facet_count));
    }

    /// Pops the element list of the largest rank.
    pub fn pop(&mut self) -> Option<ElementList> {
        self.ranks.pop()
    }

    /// Sorts the subelements and superelements of the entire polytope.
    pub fn sort(&mut self) {
        for elements in self.ranks.iter_mut() {
            for el in elements.iter_mut() {
                el.sort();
            }
        }
    }

    /// Returns a reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`element`](Self::element).
    pub fn get_element(&self, el: &ElementRef) -> Option<&Element> {
        self.ranks.get(el.rank)?.get(el.idx)
    }

    /// Returns a mutable reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`element`](Self::element).
    pub fn get_element_mut(&mut self, el: &ElementRef) -> Option<&mut Element> {
        self.ranks.get_mut(el.rank)?.get_mut(el.idx)
    }

    /// Gets the indices of the vertices of an element in the polytope, if it
    /// exists.
    pub fn element_vertices(&self, el: &ElementRef) -> Option<Vec<usize>> {
        Some(ElementHash::from_element(self, el)?.to_elements(Rank::new(0)))
    }

    /// Gets both elements with a given rank and index as a polytope and the
    /// indices of its vertices on the original polytope, if it exists.
    pub fn element_and_vertices(&self, el: &ElementRef) -> Option<(Vec<usize>, Self)> {
        let element_hash = ElementHash::from_element(self, el)?;

        Some((
            element_hash.to_elements(Rank::new(0)),
            element_hash.to_polytope(self),
        ))
    }

    /// Returns the indices of a Petrial polygon in cyclic order, or `None` if
    /// it self-intersects.
    pub fn petrie_polygon_vertices(&self, flag: Flag) -> Option<Vec<usize>> {
        let rank = self.rank().try_usize()?;
        let mut new_flag = flag.clone();
        let first_vertex = flag[0];
        let mut vertices = Vec::new();

        // Gotta love that O(n log(n)).
        let mut vertex_hash = HashSet::new();

        loop {
            // Applies 0-changes up to (rank-1)-changes in order.
            for idx in 0..rank {
                new_flag.change_mut(self, idx);
            }

            // If we just hit a previous vertex, we return.
            let new_vertex = new_flag[0];
            if vertex_hash.contains(&new_vertex) {
                return None;
            }

            // Adds the new vertex.
            vertices.push(new_vertex);
            vertex_hash.insert(new_vertex);

            // If we're back to the beginning, we break out of the loop.
            if new_vertex == first_vertex {
                break;
            }
        }

        // We returned to precisely the initial flag.
        if flag == new_flag {
            Some(vertices)
        }
        // The Petrie polygon self-intersects.
        else {
            None
        }
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. Also returns the indices of the vertices that
    /// form the base and the dual base, in that order.
    pub fn antiprism_and_vertices(&self) -> (Self, Vec<usize>, Vec<usize>) {
        let rank = self.rank();
        let mut section_hash = SectionHash::singletons(self);

        // We actually build the elements backwards, which is as awkward as it
        // seems. Maybe we should fix that in the future?
        let mut backwards_abs = RankVec::with_capacity(rank.plus_one());
        backwards_abs.push(SubelementList::max(section_hash.len));

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
            let mut elements = SubelementList::with_capacity(section_hash.len);

            for _ in 0..section_hash.len {
                elements.push(Subelements::new());
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
                for (indices, &idx) in map {
                    // Finds all of the subelements of our old section's
                    // lowest element.
                    for &idx_lo in &self
                        .get_element(&ElementRef::new(rank_lo, indices.0))
                        .unwrap()
                        .subs
                    {
                        // Adds the new sections of the current height, gets
                        // their index, uses that to build the ElementList.
                        let sub = new_section_hash.get(Section::new(
                            ElementRef::new(rank_lo.minus_one(), idx_lo),
                            ElementRef::new(rank_hi, indices.1),
                        ));

                        elements[idx].push(sub);
                    }

                    // Finds all of the superelements of our old section's
                    // highest element.
                    for &idx_hi in self
                        .get_element(&ElementRef::new(rank_hi, indices.1))
                        .unwrap()
                        .sups
                        .iter()
                    {
                        // Adds the new sections of the current height, gets
                        // their index, uses that to build the ElementList.
                        let sub = new_section_hash.get(Section::new(
                            ElementRef::new(rank_lo, indices.0),
                            ElementRef::new(rank_hi.plus_one(), idx_hi),
                        ));

                        elements[idx].push(sub);
                    }
                }
            }

            // We figure out where the vertices of the base and the dual base
            // were sent.
            if height == rank.minus_one() {
                // We create a map from the base's vertices to the new vertices.
                for v in 0..vertex_count {
                    vertices.push(new_section_hash.get(Section::new(
                        ElementRef::new(Rank::new(0), v),
                        ElementRef::new(rank, 0),
                    )));
                }

                // We create a map from the dual base's vertices to the new vertices.
                for f in 0..facet_count {
                    dual_vertices.push(new_section_hash.get(Section::new(
                        ElementRef::new(Rank::new(-1), 0),
                        ElementRef::new(rank.minus_one(), f),
                    )));
                }
            }

            backwards_abs.push(elements);
            section_hash = new_section_hash;
        }

        // We built this backwards, so let's fix it.
        let mut abs = AbstractBuilder::with_capacity(backwards_abs.rank());

        for subelements in backwards_abs.into_iter().rev() {
            abs.push(subelements);
        }

        // Sets name.
        let mut abs = abs.build();
        let facet_count = abs.facet_count();
        abs = abs.with_name(Name::antiprism(self.name.clone(), facet_count));

        (abs, vertices, dual_vertices)
    }

    /// Checks whether the polytope is valid, i.e. whether the polytope is
    /// bounded, dyadic, and all of its indices refer to valid elements.
    pub fn is_valid(&self) -> AbstractResult<()> {
        self.bounded()?;
        self.check_incidences()?;
        self.is_dyadic()?;

        Ok(())
        // && self.is_strongly_connected()
    }

    /// Determines whether the polytope is bounded, i.e. whether it has a single
    /// minimal element and a single maximal element. A valid polytope should
    /// always return `true`.
    pub fn bounded(&self) -> AbstractResult<()> {
        let min_count = self.el_count(Rank::new(-1));
        let max_count = self.el_count(self.rank());

        if min_count == 1 && max_count == 1 {
            Ok(())
        } else {
            Err(AbstractError::Bounded {
                min_count,
                max_count,
            })
        }
    }

    /// Checks whether subelements and superelements match up, and whether they
    /// all refer to valid elements in the polytope. If this returns `false`,
    /// then either the polytope hasn't fully built up, or there's something
    /// seriously wrong.
    pub fn check_incidences(&self) -> AbstractResult<()> {
        // Iterates over elements of every rank.
        for (r, elements) in self.ranks.iter().rank_enumerate() {
            // Iterates over all such elements.
            for (idx, el) in elements.iter().enumerate() {
                // Only the minimal element can have no subelements.
                if r != Rank::new(-1) && el.subs.len() == 0 {
                    return Err(AbstractError::Ranked {
                        el: ElementRef::new(r, idx),
                        incidence_type: IncidenceType::Subelement,
                    });
                }

                // Iterates over the element's subelements.
                for &sub in &el.subs {
                    // Attempts to get the subelement's superelements.
                    if let Some(r_minus_one) = r.try_sub(Rank::new(1)) {
                        if let Some(sub_el) = self.get_element(&ElementRef::new(r_minus_one, sub)) {
                            if sub_el.sups.contains(&idx) {
                                continue;
                            } else {
                                // The element contains a subelement, but not viceversa.
                                return Err(AbstractError::Consistency {
                                    el: ElementRef::new(r, idx),
                                    index: sub,
                                    incidence_type: IncidenceType::Subelement,
                                });
                            }
                        }
                    }

                    // We got ourselves an invalid index.
                    return Err(AbstractError::Index {
                        el: ElementRef::new(r, idx),
                        index: sub,
                        incidence_type: IncidenceType::Subelement,
                    });
                }

                // Only the maximal element can have no superelements.
                if r != self.rank() && el.sups.len() == 0 {
                    return Err(AbstractError::Ranked {
                        el: ElementRef::new(r, idx),
                        incidence_type: IncidenceType::Superelement,
                    });
                }

                // Iterates over the element's superelements.
                for &sup in &el.sups {
                    // Attempts to get the subelement's superelements.
                    if let Some(sub_el) = self.get_element(&ElementRef::new(r.plus_one(), sup)) {
                        if sub_el.subs.contains(&idx) {
                            continue;
                        } else {
                            // The element contains a superelement, but not viceversa.
                            return Err(AbstractError::Consistency {
                                el: ElementRef::new(r, idx),
                                index: sup,
                                incidence_type: IncidenceType::Superelement,
                            });
                        }
                    }

                    // We got ourselves an invalid index.
                    return Err(AbstractError::Index {
                        el: ElementRef::new(r, idx),
                        index: sup,
                        incidence_type: IncidenceType::Superelement,
                    });
                }
            }
        }

        Ok(())
    }

    /// Determines whether the polytope satisfies the diamond property. A valid
    /// non-fissary polytope should always return `true`.
    pub fn is_dyadic(&self) -> AbstractResult<()> {
        #[derive(PartialEq)]
        enum Count {
            Once,
            Twice,
        }

        // For every element, by looking through the subelements of its
        // subelements, we need to find each exactly twice.
        for r in 1..self.rank().isize() {
            let r = Rank::new(r);

            for (idx, el) in self[r].iter().enumerate() {
                let mut hash_sub_subs = HashMap::new();

                for &sub in &el.subs {
                    let sub_el = &self[r.minus_one()][sub];

                    for &sub_sub in &sub_el.subs {
                        match hash_sub_subs.get(&sub_sub) {
                            // Found for the first time.
                            None => hash_sub_subs.insert(sub_sub, Count::Once),

                            // Found for the second time.
                            Some(Count::Once) => hash_sub_subs.insert(sub_sub, Count::Twice),

                            // Found for the third time?! Abort!
                            Some(Count::Twice) => {
                                return Err(AbstractError::Dyadic {
                                    section: Section::new(
                                        ElementRef::new(r - Rank::new(2), sub_sub),
                                        ElementRef::new(r, idx),
                                    ),
                                    more: true,
                                });
                            }
                        };
                    }
                }

                // If any subsubelement was found only once, this also
                // violates the diamond property.
                for (sub_sub, count) in hash_sub_subs.into_iter() {
                    if count == Count::Once {
                        return Err(AbstractError::Dyadic {
                            section: Section::new(
                                ElementRef::new(r - Rank::new(2), sub_sub),
                                ElementRef::new(r, idx),
                            ),
                            more: false,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Determines whether the polytope is connected. A valid non-compound
    /// polytope should always return `true`.
    pub fn is_connected(&self, _section: Section) -> bool {
        todo!()
        /*
        let section = self.get_section(section).unwrap();
        section.flags().count() == section.oriented_flags().count() */
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
        let rank = p_rank + q_rank.plus_one() - Rank::new(!min as isize) - Rank::new(!max as isize);

        // Initializes the element lists. These will only contain the
        // subelements as they're generated. When they're complete, we'll call
        // push_subs for each of them into a new Abstract.
        let mut element_lists = RankVec::with_capacity(rank);
        for _ in Rank::range_inclusive_iter(Rank::new(-1), rank) {
            element_lists.push(SubelementList::new());
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
                        offset_memo[(p_rank.minus_one() - p_low).usize()]
                            [(q_rank.plus_one() - q_low).usize()]
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
        let get_element_index = |p_rank: Rank, p_idx, q_rank: Rank, q_idx| -> _ {
            if let Some(p_rank_minus_one) = p_rank.try_sub(Rank::new(1)) {
                offset(p_rank_minus_one, q_rank.plus_one()) + p_idx * q.el_count(q_rank) + q_idx
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
                                for &s in &p_el.subs {
                                    subs.push(get_element_index(
                                        p_els_rank.minus_one(),
                                        s,
                                        q_els_rank,
                                        q_idx,
                                    ))
                                }
                            }

                            // Products of q's subelements with p.
                            if q_els_rank != Rank::new(0) || min {
                                for &s in &q_el.subs {
                                    subs.push(get_element_index(
                                        p_els_rank,
                                        p_idx,
                                        q_els_rank.minus_one(),
                                        s,
                                    ))
                                }
                            }

                            element_lists[prod_rank].push(subs)
                        }
                    }
                }
            }
        }

        // If !min, we have to set a minimal element manually.
        if !min {
            let vertex_count = p.vertex_count() * q.vertex_count();
            element_lists[Rank::new(-1)] = SubelementList::min();
            element_lists[Rank::new(0)] = SubelementList::vertices(vertex_count);
        }

        // If !max, we have to set a maximal element manually.
        if !max {
            element_lists[rank] = SubelementList::max(element_lists[rank.minus_one()].len());
        }

        // Uses push_subs to add all of the element lists into a new polytope.
        let mut product = AbstractBuilder::with_capacity(element_lists.rank());
        for elements in element_lists.into_iter() {
            product.push(elements);
        }

        // Sets the name of the polytope.
        let bases = vec![p.name.clone(), q.name.clone()];

        product.build().with_name(if min && max {
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

    /// Returns a reference to the [`Name`] of the polytope. This exists only
    /// for trait purposes.
    fn name(&self) -> &Name<Abs> {
        &self.name
    }

    /// Returns a mutable reference to the [`Name`] of the polytope. This exists
    /// only for trait purposes.
    fn name_mut(&mut self) -> &mut Name<Abs> {
        &mut self.name
    }

    /// Returns a reference to `self`. This exists only for trait purposes.
    fn abs(&self) -> &Abstract {
        self
    }

    /// Returns a mutable reference to `self`. This exists only for trait
    /// purposes.
    fn abs_mut(&mut self) -> &mut Abstract {
        self
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
        let mut abs = AbstractBuilder::with_capacity(Rank::new(1));

        abs.push_min();
        abs.push_vertices(2);
        abs.push_max();

        abs.build().with_name(Name::Dyad)
    }

    /// Returns an instance of a [polygon](https://polytope.miraheze.org/wiki/Polygon)
    /// with a given number of sides.
    fn polygon(n: usize) -> Self {
        assert!(n >= 2, "A polygon must have at least 2 sides.");

        let mut edges = SubelementList::with_capacity(n);

        for i in 0..n {
            edges.push(Subelements(vec![i % n, (i + 1) % n]));
        }

        let mut poly = AbstractBuilder::with_capacity(Rank::new(2));

        poly.push_min();
        poly.push_vertices(n);
        poly.push(edges);
        poly.push_max();

        poly.build().with_name(Name::polygon(AbsData::default(), n))
    }

    /// Converts a polytope into its dual. Use [`Self::dual`] instead, as this method
    /// can never fail.
    fn try_dual(&self) -> Result<Self, usize> {
        let mut clone = self.clone();
        clone.dual_mut();
        Ok(clone)
    }

    /// Converts a polytope into its dual in place. Use [`Self::dual_mut`] instead, as
    /// this method can never fail.
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

    /// Builds the [Petrial](https://polytope.miraheze.org/wiki/Petrial) of a
    /// polyhedron in place.
    fn petrial_mut(&mut self) -> Result<(), ()> {
        // Petrials only really make sense for polyhedra.
        if self.rank() != Rank::new(3) {
            return Err(());
        }

        // Consider a flag in a polytope. It has an associated edge. It turns
        // out that if we repeatedly apply a vertex-change, an edge-change, and
        // a face-change, we get the edges that form the Petrial face.
        //
        // We go through all flags in the polytope. As we build one Petrial
        // face, we mark any other flag that gives the same face as "traversed".
        // Once we've traversed all flags, we got our Petrial's faces.
        let mut traversed_flags = BTreeSet::new();
        let mut faces = SubelementList::new();

        for mut flag in self.flags() {
            // If we've found the face associated to this flag before, we skip.
            if !traversed_flags.insert(flag.clone()) {
                continue;
            }

            let mut face = BTreeSet::new();
            let mut edge = flag[Rank::new(1)];
            let mut loop_continue = true;

            // We apply our flag changes and mark our flags until we reach the
            // original edge. We then intentionally overshoot and do it one more
            // time.
            while loop_continue {
                loop_continue = face.insert(edge);

                flag.change_mut(&self, 0);
                traversed_flags.insert(flag.change(&self, 2));
                flag.change_mut(&self, 1);
                flag.change_mut(&self, 2);
                traversed_flags.insert(flag.clone());

                edge = flag[Rank::new(1)];
            }

            // If the edge we found after we returned to the original edge was
            // not already in the face, this means that the Petrial loop
            // self-intersects, and hence the Petrial is not a valid polytope.
            if !face.contains(&edge) {
                return Err(());
            }

            faces.push(Subelements(face.into_iter().collect()));
        }

        // Removes the faces and maximal polytope from self.
        self.pop();
        self.pop();

        // Clears the current edges' superelements.
        for edge in self[Rank::new(1)].iter_mut() {
            edge.sups = Superelements::new();
        }

        // Pushes the new faces and a new maximal element.
        self.push_subs(faces);
        self.push_max();

        // Builds name.
        let name = mem::replace(&mut self.name, Name::Nullitope);
        self.name = name.petrial(self.facet_count());

        // Checks for dyadicity, since that sometimes fails.
        self.is_dyadic().map_err(|_| ())
    }

    fn petrie_polygon_with(&self, flag: Flag) -> Option<Self> {
        Some(Self::polygon(self.petrie_polygon_vertices(flag)?.len()))
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. Use [`Self::antiprism`] instead, as this method can
    /// never fail.
    fn try_antiprism(&self) -> Result<Self, usize> {
        Ok(self.antiprism_and_vertices().0)
    }

    fn flag_omnitruncate(&self) -> Self {
        todo!()
    }

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks. *Updates neither the name nor the
    /// min/max elements.**
    fn _append(&mut self, p: Self) {
        let rank = self.rank();

        // The polytopes must have the same ranks.
        assert_eq!(rank, p.rank());

        let el_counts = self.el_counts();

        for (r, elements) in p
            .ranks
            .into_iter()
            .rank_enumerate()
            .skip(1)
            .take(rank.usize())
        {
            let sub_offset = el_counts[r.minus_one()];
            let sup_offset = el_counts[r.plus_one()];

            for mut el in elements.into_iter() {
                if r != Rank::new(0) {
                    for sub in el.subs.iter_mut() {
                        *sub += sub_offset;
                    }
                }

                if r != rank.minus_one() {
                    for sup in el.sups.iter_mut() {
                        *sup += sup_offset;
                    }
                }

                self.push_at(r, el);
            }
        }
    }

    fn append(&mut self, p: Self) {
        let name = mem::replace(&mut self.name, Name::Nullitope);
        let new_name = Name::compound(vec![(1, name), (1, p.name.clone())]);

        self._append(p);

        *self.min_mut() = Element::min(self.vertex_count());
        *self.max_mut() = Element::max(self.facet_count());

        self.name = new_name;
    }

    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, el: &ElementRef) -> Option<Self> {
        Some(ElementHash::from_element(self, el)?.to_polytope(self))
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
        self.push_subs_at(rank, self.max().subs.clone());
        self.push_max();
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

/// Permits indexing an abstract polytope by rank.
impl std::ops::Index<Rank> for Abstract {
    type Output = ElementList;

    fn index(&self, index: Rank) -> &Self::Output {
        &self.ranks[index]
    }
}

/// Permits mutably indexing an abstract polytope by rank.
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
        assert!(nullitope.is_valid().is_ok(), "Nullitope is invalid.");
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
        assert!(point.is_valid().is_ok(), "Point is invalid.");
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
        assert!(dyad.is_valid().is_ok(), "Dyad is invalid.");
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
            assert!(polygon.is_valid().is_ok(), "{}-gon is invalid.", n);
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
                    duopyramid.is_valid().is_ok(),
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
                assert!(
                    duoprism.is_valid().is_ok(),
                    "{}-{} duoprism is invalid.",
                    m,
                    n
                );
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
                assert!(
                    duotegum.is_valid().is_ok(),
                    "{}-{} duotegum is invalid.",
                    m,
                    n
                );
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
                assert!(
                    duocomb.is_valid().is_ok(),
                    "{}-{} duocomb is invalid.",
                    m,
                    n
                );
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
                    choose(n.plus_one_usize(), k.plus_one_usize()),
                    "{}-simplex {}-element counts don't match up",
                    n,
                    k
                );
            }

            assert!(simplex.is_valid().is_ok(), "{}-simplex is invalid.", n)
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

            assert!(hypercube.is_valid().is_ok(), "{}-hypercube is invalid.", n)
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
                    choose(n.usize(), k.plus_one_usize()) * 2u32.pow(k.plus_one_u32()) as usize,
                    "{}-orthoplex {}-element counts don't match up",
                    n,
                    k
                );
            }

            assert!(orthoplex.is_valid().is_ok(), "{}-orthoplex is invalid.", n)
        }
    }

    #[test]
    /// Checks that duals are generated correctly.
    fn dual_check() {
        use crate::lang::{En, Language};

        for poly in test_polytopes().iter_mut() {
            let el_counts = poly.el_counts();

            poly.dual_mut();

            // The element counts of the dual should be the same as the reversed
            // element counts of the original.
            let mut du_el_counts_rev = poly.el_counts();
            du_el_counts_rev.reverse();
            assert_eq!(
                el_counts.0,
                du_el_counts_rev.0,
                "Dual element counts of {} don't match expected value.",
                En::parse(poly.name(), Default::default())
            );
            assert!(
                poly.is_valid().is_ok(),
                "Dual of polytope {} is invalid.",
                En::parse(poly.name(), Default::default())
            );
        }
    }
}
