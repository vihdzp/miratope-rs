//! Declares the [`Abstract`] polytope type and all associated data structures.

pub mod elements;
pub mod flag;
pub mod rank;

use std::collections::{BTreeSet, HashMap, HashSet};

use self::{
    elements::{
        AbstractBuilder, Element, ElementHash, ElementList, ElementRef, SectionHash, SectionRef,
        SubelementList, Subelements, Superelements,
    },
    flag::{Flag, FlagSet},
    rank::{Rank, RankVec},
};
use super::{DualResult, Polytope};

use rayon::prelude::*;
use strum_macros::Display;
use vec_like::VecLike;

/// Represents the way in which two elements with one rank of difference are
/// incident to one another. Used as a field in some [`AbstractError`] variants.

#[derive(Debug, Display)]
pub enum IncidenceType {
    /// This element is a subelement of another.
    #[strum(serialize = "subelement")]
    Subelement,

    /// This element is a superelement of another.
    #[strum(serialize = "superelement")]
    Superelement,
}

/// Represents an error in an abstract polytope.
#[derive(Debug)]
pub enum AbstractError {
    /// The polytope is not bounded, i.e. it doesn't have a single minimal and
    /// maximal element.
    Bounded {
        /// The number of minimal elements.
        min_count: usize,

        /// The number of maximal elements.
        max_count: usize,
    },

    /// The polytope has some invalid index, i.e. some element points to another
    /// non-existent element.
    Index {
        /// The coordinates of the element at fault.
        el: ElementRef,

        /// Whether the invalid index is a subelement or a superelement.
        incidence_type: IncidenceType,

        /// The invalid index.
        index: usize,
    },

    /// The polytope has a consistency error, i.e. some element is incident to
    /// another but not viceversa.
    Consistency {
        /// The coordinates of the element at fault.
        el: ElementRef,

        /// Whether the invalid incidence is a subelement or a superelement.
        incidence_type: IncidenceType,

        /// The invalid index.
        index: usize,
    },

    /// The polytope is not ranked, i.e. some element that's not minimal or not
    /// maximal lacks a subelement or superelement, respectively.
    Ranked {
        /// The coordinates of the element at fault.
        el: ElementRef,

        /// Whether the missing incidences are at subelements or superelements.
        incidence_type: IncidenceType,
    },

    /// The polytope is not dyadic, i.e. some section of height 1 does not have
    /// exactly 4 elements.
    Dyadic {
        /// The coordinates of the section at fault.
        section: SectionRef,

        /// Whether there were more than 4 elements in the section (or less).
        more: bool,
    },

    /// The polytope is not strictly connected, i.e. some section's flags don't
    /// form a connected graph under flag changes.
    Connected(SectionRef),
}

impl std::fmt::Display for AbstractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // The polytope is not bounded.
            AbstractError::Bounded {
                min_count,
                max_count,
            } => write!(
                f,
                "Polytope is unbounded: found {} minimal elements and {} maximal elements",
                min_count, max_count
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

impl std::error::Error for AbstractError {}

/// The return value for [`Abstract::is_valid`].
pub type AbstractResult<T> = Result<T, AbstractError>;

/// Encodes the ranked poset corresponding to the abstract polytope. Internally,
/// it wraps around a [`RankVec`] of [`ElementLists`](ElementList).
///
/// # What is an abstract polytope?
/// Mathematically, an abstract polytope is a certain kind of **partially
/// ordered set** (also known as a poset). A partially ordered set is a set P
/// together with a relation ≤ on the set, satisfying three properties.
/// 1. Reflexivity: *a* ≤ *a*.
/// 2. Antisymmetry: *a* ≤ *b* and *b* ≤ *a* implies *a* = *b*.
/// 3. Transitivity: *a* ≤ *b* and *b* ≤ *c* implies *a* ≤ *c*.
///
/// If either *a* ≤ *b* or *b* ≤ *a*, we say that *a* and *b* are incident. If
/// *a* ≤ *c* and there's no *b* other than *a* or *c* such that *a* ≤ *b* ≤
/// *c*, we say that *c* covers *a*.
///
/// An **abstract polytope** must also satisfy these three extra conditions:
/// 1. Bounded: It has a minimal and maximal element.
/// 2. Ranked: Every element *x* can be assigned a rank *r*(*x*) so that
///    *r*(*x*) ≤ *r*(*y*) when *x* ≤ *y*, and *r*(*x*) + 1 = *r*(*y*) when *y*
///    covers *x*.
/// 3. Diamond property: If *a* ≤ *c* and *r*(*a*) + 2 = *r*(*c*), then there
///    must be exactly two elements *b* such that *a* ≤ *b* ≤ *c*.
///
/// Each element in the poset represents an element in the polytope (that is, a
/// vertex, edge, etc.) By convention, we choose the rank function so that the
/// minimal element has rank &minus;1, which makes the rank of an element match
/// its expected dimension (vertices are rank 0, edges are rank 1, etc.) The
/// minimal and maximal element in the poset represent an "empty" element and
/// the entire polytope, and are there mostly for convenience.
///
/// For more info, see [Wikipedia](https://en.wikipedia.org/wiki/Abstract_polytope)
/// or the [Polytope Wiki](https://polytope.miraheze.org/wiki/Abstract_polytope)
///
/// # How are these represented?
/// The core attribute of the `Abstract` type is the `ranks` attribute, which
/// contains a `RankVec<ElementList>`.
///
/// A [`RankVec`] is a wrapper around a `Vec` that allows us to index by a
/// [`Rank`]. That is, indexing starts at `-1`.
///
/// An [`ElementList`] is nothing but a list of [`Elements`](Element). Every
/// element stores the indices of the incident [`Subelements`] of the previous
/// rank, as well as the [`Superelements`] of the next rank.
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
#[derive(Debug, Default, Clone)]
pub struct Abstract {
    /// The list of element lists in the polytope, ordered by [`Rank`].
    pub ranks: RankVec<ElementList>,

    /// Whether every single element's subelements and superelements are sorted.
    pub sorted: bool,
}

impl AsRef<Vec<ElementList>> for Abstract {
    fn as_ref(&self) -> &Vec<ElementList> {
        self.ranks.as_ref()
    }
}

impl AsMut<Vec<ElementList>> for Abstract {
    fn as_mut(&mut self) -> &mut Vec<ElementList> {
        self.ranks.as_mut()
    }
}

impl From<RankVec<ElementList>> for Abstract {
    fn from(ranks: RankVec<ElementList>) -> Self {
        Self {
            ranks,
            sorted: false,
        }
    }
}

impl From<Vec<ElementList>> for Abstract {
    fn from(vec: Vec<ElementList>) -> Self {
        RankVec::from(vec).into()
    }
}

impl VecLike for Abstract {
    type VecItem = ElementList;
    type VecIndex = Rank;
}

impl Abstract {
    /// Initializes a polytope with an empty element list.
    pub fn new() -> Self {
        RankVec::new().into()
    }

    /// Initializes a new polytope with the capacity needed to store elements up
    /// to a given rank.
    pub fn with_rank_capacity(rank: Rank) -> Self {
        RankVec::with_rank_capacity(rank).into()
    }

    /// Returns `true` if we haven't added any elements to the polytope. Note
    /// that such a polytope is considered invalid.
    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    /// Reserves capacity for at least `additional` more element lists to be
    /// inserted in `self`.
    pub fn reserve(&mut self, additional: usize) {
        self.ranks.reserve(additional)
    }

    /// Returns a reference to the minimal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    pub fn min(&self) -> &Element {
        self.get_element(ElementRef::new(Rank::new(-1), 0)).unwrap()
    }

    /// Returns a mutable reference to the minimal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    pub fn min_mut(&mut self) -> &mut Element {
        self.get_element_mut(ElementRef::new(Rank::new(-1), 0))
            .unwrap()
    }

    /// Returns a reference to the maximal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    pub fn max(&self) -> &Element {
        self.get_element(ElementRef::new(self.rank(), 0)).unwrap()
    }

    /// Returns a mutable reference to the maximal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    pub fn max_mut(&mut self) -> &mut Element {
        self.get_element_mut(ElementRef::new(self.rank(), 0))
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
    /// superelements of the current maximal rank **haven't** already been set.
    /// If they have already been set, use [`push`](Self::push) instead.
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

    /// Returns a reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`element`](Self::element).
    pub fn get_element(&self, el: ElementRef) -> Option<&Element> {
        self.ranks.get(el.rank)?.get(el.idx)
    }

    /// Returns a mutable reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`element`](Self::element).
    pub fn get_element_mut(&mut self, el: ElementRef) -> Option<&mut Element> {
        self.ranks.get_mut(el.rank)?.get_mut(el.idx)
    }

    /// Gets the indices of the vertices of an element in the polytope, if it
    /// exists.
    pub fn element_vertices(&self, el: ElementRef) -> Option<Vec<usize>> {
        Some(ElementHash::new(self, el)?.to_vertices())
    }

    /// Gets both elements with a given rank and index as a polytope and the
    /// indices of its vertices on the original polytope, if it exists.
    pub fn element_and_vertices(&self, el: ElementRef) -> Option<(Vec<usize>, Self)> {
        let element_hash = ElementHash::new(self, el)?;
        Some((element_hash.to_vertices(), element_hash.to_polytope(self)))
    }

    /// Returns the indices of a Petrial polygon in cyclic order, or `None` if
    /// it self-intersects.
    pub fn petrie_polygon_vertices(&mut self, flag: Flag) -> Option<Vec<usize>> {
        let rank = self.rank().try_usize()?;
        let mut new_flag = flag.clone();
        let first_vertex = flag[0];

        let mut vertices = Vec::new();
        let mut vertex_hash = HashSet::new();

        self.abs_sort();

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
        let mut backwards_abs = RankVec::with_rank_capacity(rank.plus_one());
        backwards_abs.push(SubelementList::max(section_hash.len()));

        // Indices of base.
        let vertex_count = self.vertex_count();
        let mut vertices = Vec::with_capacity(vertex_count);

        // Indices of dual base.
        let facet_count = self.facet_count();
        let mut dual_vertices = Vec::with_capacity(facet_count);

        // Adds all elements corresponding to sections of a given height.
        for height in 0..=rank.into_isize() + 1 {
            let height = Rank::new(height);
            let mut new_section_hash = SectionHash::new();
            let mut elements = SubelementList::with_capacity(section_hash.len());

            for _ in 0..section_hash.len() {
                elements.push(Subelements::new());
            }

            // Goes over all sections of the previous height, and builds the
            // sections of the current height by either changing the upper
            // element into one of its superelements, or changing the lower
            // element into one of its subelements.
            for (section, idx) in section_hash.into_iter() {
                // Finds all of the subelements of our old section's
                // lowest element.
                for &idx_lo in &self.get_element(section.lo).unwrap().subs {
                    // Adds the new sections of the current height, gets
                    // their index, uses that to build the ElementList.
                    let sub = new_section_hash.get(SectionRef::new(
                        ElementRef::new(section.lo.rank.minus_one(), idx_lo),
                        section.hi,
                    ));

                    elements[idx].push(sub);
                }

                // Finds all of the superelements of our old section's
                // highest element.
                for &idx_hi in &self.get_element(section.hi).unwrap().sups {
                    // Adds the new sections of the current height, gets
                    // their index, uses that to build the ElementList.
                    let sub = new_section_hash.get(SectionRef::new(
                        section.lo,
                        ElementRef::new(section.hi.rank.plus_one(), idx_hi),
                    ));

                    elements[idx].push(sub);
                }
            }

            // We figure out where the vertices of the base and the dual base
            // were sent.
            if height == rank.minus_one() {
                // We create a map from the base's vertices to the new vertices.
                for v in 0..vertex_count {
                    vertices.push(new_section_hash.get(SectionRef::new(
                        ElementRef::new(Rank::new(0), v),
                        ElementRef::new(rank, 0),
                    )));
                }

                // We create a map from the dual base's vertices to the new vertices.
                for f in 0..facet_count {
                    dual_vertices.push(new_section_hash.get(SectionRef::new(
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

        (abs.build(), vertices, dual_vertices)
    }

    /// Returns the omnitruncate of a polytope, along with the flags that make
    /// up its vertices.
    ///
    /// # Panics
    /// This method will panic if the polytope isn't sorted.
    pub fn omnitruncate_and_flags(&self) -> (Self, Vec<Flag>) {
        let mut flag_sets = vec![FlagSet::new(self)];
        let mut new_flag_sets = Vec::new();
        let rank = self.rank();

        // The elements of each rank... backwards.
        let mut ranks = Vec::with_capacity(rank.plus_one_usize());

        // Adds elements of each rank.
        for _ in 0..rank.into() {
            let mut subelements = SubelementList::new();

            // Gets the subelements of each element.
            for flag_set in flag_sets {
                let mut subs = Subelements::new();

                // Each subset represents a new element.
                for subset in flag_set.subsets(self) {
                    // We do a brute-force check to see if we've found this
                    // element before.
                    //
                    // TODO: think of something better?
                    match new_flag_sets
                        .iter()
                        .enumerate()
                        .find(|(_, new_flag_set)| subset == **new_flag_set)
                    {
                        // This is a repeat element.
                        Some((idx, _)) => {
                            subs.push(idx);
                        }

                        // This is a new element.
                        None => {
                            subs.push(new_flag_sets.len());
                            new_flag_sets.push(subset);
                        }
                    }
                }

                subelements.push(subs);
            }

            ranks.push(subelements);
            flag_sets = new_flag_sets;
            new_flag_sets = Vec::new();
        }

        let mut flags = Vec::new();
        for flag_set in flag_sets {
            debug_assert_eq!(flag_set.len(), 1);
            flags.push(flag_set.flags.into_iter().next().unwrap());
        }

        ranks.push(SubelementList::vertices(flags.len()));
        ranks.push(SubelementList::min());

        // TODO: wrap this using an AbstractBuilderRev.
        let mut abs = AbstractBuilder::with_capacity(rank);
        for subelements in ranks.into_iter().rev() {
            abs.push(subelements);
        }

        (abs.build(), flags)
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
        for (r, elements) in self.ranks.rank_iter().rank_enumerate() {
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
                        if let Some(sub_el) = self.get_element(ElementRef::new(r_minus_one, sub)) {
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
                    if let Some(sub_el) = self.get_element(ElementRef::new(r.plus_one(), sup)) {
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
        /// The number of times we've found an element.
        #[derive(PartialEq)]
        enum Count {
            /// We've found an element once.
            Once,

            /// We've found an element twice.
            Twice,
        }

        // For every element, by looking through the subelements of its
        // subelements, we need to find each exactly twice.
        for r in 1..self.rank().into_isize() {
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
                                    section: SectionRef::new(
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
                            section: SectionRef::new(
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
    pub fn is_connected(&self, _section: SectionRef) -> bool {
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
        let mut element_lists = RankVec::with_rank_capacity(rank);
        for _ in Rank::range_inclusive_iter(-1, rank) {
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
                        offset_memo[(p_rank.minus_one() - p_low).into_usize()]
                            [(q_rank.plus_one() - q_low).into_usize()]
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
            if let Some(p_rank_minus_one) = p_rank.try_sub(1) {
                offset(p_rank_minus_one, q_rank.plus_one()) + p_idx * q.el_count(q_rank) + q_idx
            } else {
                q_idx
            }
        };

        // Adds elements in order of rank.
        for prod_rank in Rank::range_inclusive_iter(-1, rank) {
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
                            if min || p_els_rank != Rank::new(0) {
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
                            if min || q_els_rank != Rank::new(0) {
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

        product.build()
    }
}

impl Polytope for Abstract {
    fn abs(&self) -> &Abstract {
        self
    }

    fn abs_mut(&mut self) -> &mut Abstract {
        self
    }

    /// Returns an instance of the
    /// [nullitope](https://polytope.miraheze.org/wiki/Nullitope), the unique
    /// polytope of rank &minus;1.
    fn nullitope() -> Self {
        Self {
            ranks: vec![ElementList::min(0)].into(),
            sorted: true,
        }
    }

    /// Returns an instance of the
    /// [point](https://polytope.miraheze.org/wiki/Point), the unique polytope
    /// of rank 0.
    fn point() -> Self {
        Self {
            ranks: vec![ElementList::min(1), ElementList::max(1)].into(),
            sorted: true,
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

        let mut abs = abs.build();
        abs.sorted = true;
        abs
    }

    /// Returns an instance of a [polygon](https://polytope.miraheze.org/wiki/Polygon)
    /// with a given number of sides.
    fn polygon(n: usize) -> Self {
        assert!(n >= 2, "A polygon must have at least 2 sides.");

        let mut edges = SubelementList::with_capacity(n);

        // We add the edges with their indices sorted.
        for i in 0..(n - 1) {
            edges.push(Subelements(vec![i, (i + 1)]));
        }
        edges.push(Subelements(vec![0, n - 1]));

        let mut poly = AbstractBuilder::with_capacity(Rank::new(2));

        poly.push_min();
        poly.push_vertices(n);
        poly.push(edges);
        poly.push_max();

        let mut poly = poly.build();
        poly.sorted = true;
        poly
    }

    /// Converts a polytope into its dual. Use [`Self::dual`] instead, as this method
    /// can never fail.
    fn try_dual(&self) -> DualResult<Self> {
        let mut clone = self.clone();
        clone.dual_mut();
        Ok(clone)
    }

    /// Converts a polytope into its dual in place. Use [`Self::dual_mut`] instead, as
    /// this method can never fail.
    fn try_dual_mut(&mut self) -> DualResult<()> {
        for elements in self.ranks.iter_mut() {
            elements.par_iter_mut().for_each(Element::swap_mut);
        }

        self.ranks.reverse();
        Ok(())
    }

    /// Builds the [Petrial](https://polytope.miraheze.org/wiki/Petrial) of a
    /// polyhedron in place.
    fn petrial_mut(&mut self) -> bool {
        // Petrials only really make sense for polyhedra.
        if self.rank() != Rank::new(3) {
            return false;
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

        self.abs_sort();
        for mut flag in self.flags() {
            // If we've found the face associated to this flag before, we skip.
            if !traversed_flags.insert(flag.clone()) {
                continue;
            }

            let mut face = BTreeSet::new();
            let mut edge = flag[1];
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

                edge = flag[1];
            }

            // If the edge we found after we returned to the original edge was
            // not already in the face, this means that the Petrial loop
            // self-intersects, and hence the Petrial is not a valid polytope.
            if !face.contains(&edge) {
                return false;
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

        // Checks for dyadicity, since that sometimes fails.
        self.is_dyadic().is_ok()
    }

    fn petrie_polygon_with(&mut self, flag: Flag) -> Option<Self> {
        Some(Self::polygon(self.petrie_polygon_vertices(flag)?.len()))
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. Use [`Self::antiprism`] instead, as this
    /// method can never fail.
    fn try_antiprism(&self) -> DualResult<Self> {
        Ok(self.antiprism())
    }

    /// Returns the flag omnitruncate of a polytope.
    fn omnitruncate(&self) -> Self {
        self.omnitruncate_and_flags().0
    }

    /// "Appends" a polytope into another, creating a compound polytope.
    ///
    /// # Panics
    /// This method will panic if the polytopes have different ranks.
    fn comp_append(&mut self, p: Self) {
        let rank = self.rank();

        // The polytopes must have the same ranks.
        assert_eq!(rank, p.rank());

        let el_counts = self.el_counts();

        for (r, elements) in p
            .ranks
            .rank_into_iter()
            .rank_enumerate()
            .skip(1)
            .take(rank.into())
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

        // We don't need to do this every single time.
        *self.min_mut() = Element::min(self.vertex_count());
        *self.max_mut() = Element::max(self.facet_count());
    }

    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, el: ElementRef) -> Option<Self> {
        Some(ElementHash::new(self, el)?.to_polytope(self))
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

    type IntoIter = crate::abs::rank::IntoIter<ElementList>;

    fn into_iter(self) -> Self::IntoIter {
        self.ranks.rank_into_iter()
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

    /// Tests whether a polytope's element counts match the expected element
    /// counts, and whether a polytope is valid.
    fn test(poly: &Abstract, element_counts: Vec<usize>) {
        assert_eq!(
            poly.el_counts(),
            element_counts.into(),
            "{} element counts don't match expected value.",
            "TBA: name"
        );

        assert!(
            poly.is_valid().is_ok(),
            "{} is not a valid polytope.",
            "TBA: name"
        );
    }

    #[test]
    /// Checks that a nullitope is generated correctly.
    fn nullitope() {
        test(&Abstract::nullitope(), vec![1]);
    }

    #[test]
    /// Checks that a point is generated correctly.
    fn point() {
        test(&Abstract::point(), vec![1, 1]);
    }

    #[test]
    /// Checks that a dyad is generated correctly.
    fn dyad() {
        test(&Abstract::dyad(), vec![1, 2, 1]);
    }

    #[test]
    /// Checks that polygons are generated correctly.
    fn polygon() {
        for n in 2..=10 {
            test(&Abstract::polygon(n), vec![1, n, n, 1]);
        }
    }

    #[test]
    /// Checks that polygonal duopyramids are generated correctly.
    fn duopyramid() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                test(
                    &Abstract::duopyramid(&polygons[m - 2], &polygons[n - 2]),
                    vec![
                        1,
                        m + n,
                        m + n + m * n,
                        2 * m * n + 2,
                        m + n + m * n,
                        m + n,
                        1,
                    ],
                );
            }
        }
    }

    #[test]
    /// Checks that polygonal duoprisms are generated correctly.
    fn duoprism() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                test(
                    &Abstract::duoprism(&polygons[m - 2], &polygons[n - 2]),
                    vec![1, m * n, 2 * m * n, m + n + m * n, m + n, 1],
                );
            }
        }
    }

    #[test]
    /// Checks that polygonal duotegums are generated correctly.
    fn duotegum() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                test(
                    &Abstract::duotegum(&polygons[m - 2], &polygons[n - 2]),
                    vec![1, m + n, m + n + m * n, 2 * m * n, m * n, 1],
                );
            }
        }
    }

    #[test]
    /// Checks that polygonal duocombs are generated correctly.
    fn duocomb() {
        let mut polygons = Vec::new();
        for n in 2..=5 {
            polygons.push(Abstract::polygon(n));
        }

        for m in 2..=5 {
            for n in m..=5 {
                test(
                    &Abstract::duocomb(&polygons[m - 2], &polygons[n - 2]),
                    vec![1, m * n, 2 * m * n, m * n, 1],
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
    fn simplex() {
        for n in Rank::range_inclusive_iter(-1, 5) {
            let simplex = Abstract::simplex(n);
            let mut element_counts = Vec::with_capacity(n.plus_one_usize());

            for k in Rank::range_inclusive_iter(-1, n) {
                element_counts.push(choose(n.plus_one_usize(), k.plus_one_usize()));
            }

            test(&simplex, element_counts);
        }
    }

    #[test]
    /// Checks that hypercubes are generated correctly.
    fn hypercube() {
        for n in Rank::range_inclusive_iter(-1, 5) {
            let hypercube = Abstract::hypercube(n);
            let mut element_counts = Vec::with_capacity(n.plus_one_usize());

            element_counts.push(1);
            for k in Rank::range_inclusive_iter(Rank::new(0), n) {
                element_counts.push(choose(n.into(), k.into()) * (1 << (n - k).into_usize()));
            }

            test(&hypercube, element_counts);
        }
    }

    #[test]
    /// Checks that orthoplices are generated correctly.
    fn orthoplex() {
        for n in Rank::range_inclusive_iter(-1, 5) {
            let orthoplex = Abstract::orthoplex(n);
            let mut element_counts = Vec::with_capacity(n.plus_one_usize());

            for k in Rank::range_inclusive_iter(0, n) {
                element_counts.push(choose(n.into(), (n - k).into()) * (1 << k.into_usize()));
            }
            element_counts.push(1);

            test(&orthoplex, element_counts);
        }
    }

    #[test]
    /// Checks that various polytopes are generated correctly.
    fn general_check() {
        for poly in test_polytopes().iter_mut() {
            assert!(poly.is_valid().is_ok(), "{} is not valid.", "TBA: name");
        }
    }

    #[test]
    /// Checks that duals are generated correctly.
    fn dual_check() {
        use vec_like::VecLike;

        for poly in test_polytopes().iter_mut() {
            let el_counts = poly.el_counts();
            poly.dual_mut();

            // The element counts of the dual should be the same as the reversed
            // element counts of the original.
            let mut du_el_counts_rev = poly.el_counts();
            du_el_counts_rev.reverse();
            assert_eq!(
                el_counts, du_el_counts_rev,
                "Dual element counts of {} don't match expected value.",
                "TBA: name"
            );

            // The duals should also be valid polytopes.
            assert!(
                poly.is_valid().is_ok(),
                "Dual of polytope {} is invalid.",
                "TBA: name"
            );
        }
    }
}
