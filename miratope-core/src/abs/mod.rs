//! Declares the [`Abstract`] polytope type and all associated data structures.

pub mod elements;
pub mod flag;
pub mod product;

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    convert::Infallible,
    ops::{Index, IndexMut},
};

use self::flag::{Flag, FlagSet};
use super::Polytope;

use rayon::prelude::*;
use strum_macros::Display;
use vec_like::VecLike;

pub use elements::*;
pub use product::*;

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
        el: (usize, usize),

        /// Whether the invalid index is a subelement or a superelement.
        incidence_type: IncidenceType,

        /// The invalid index.
        index: usize,
    },

    /// The polytope has a consistency error, i.e. some element is incident to
    /// another but not viceversa.
    Consistency {
        /// The coordinates of the element at fault.
        el: (usize, usize),

        /// Whether the invalid incidence is a subelement or a superelement.
        incidence_type: IncidenceType,

        /// The invalid index.
        index: usize,
    },

    /// The polytope is not ranked, i.e. some element that's not minimal or not
    /// maximal lacks a subelement or superelement, respectively.
    Ranked {
        /// The coordinates of the element at fault.
        el: (usize, usize),

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
                "Polytope has an invalid index: {:?} has a {} with index {}, but it doesn't exist",
                el, incidence_type, index
            ),

            AbstractError::Consistency {
                el,
                incidence_type,
                index,
            } => write!(
                f,
                "Polytope has an invalid index: {:?} has a {} with index {}, but not viceversa",
                el, incidence_type, index
            ),

            // The polytope is not ranked.
            AbstractError::Ranked { el, incidence_type } => write!(
                f,
                "Polytope is not ranked: {:?} has no {}s",
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
/// vertex, edge, etc.) The minimal and maximal element in the poset represent
/// an "empty" element and the entire polytope, and are there mostly for
/// convenience. The usual convention is to choose the rank function so that the
/// minimal element has rank &minus;1, which makes the rank of an element match
/// its expected dimension (vertices are rank 0, edges are rank 1, etc.)
/// However, since indexing is most naturally expressed with an `usize`, we'll
/// internally start indexing by 0.
///
/// For more info, see [Wikipedia](https://en.wikipedia.org/wiki/Abstract_polytope)
/// or the [Polytope Wiki](https://polytope.miraheze.org/wiki/Abstract_polytope)
///
/// # How are these represented?
/// The core attribute of the `Abstract` type is the `ranks` attribute, which
/// contains a `Vec<ElementList>`.
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
#[derive(Debug, Clone, Default)]
pub struct Abstract {
    /// The list of element lists in the polytope.
    pub ranks: Ranks,

    /// Whether every single element's subelements and superelements are sorted.
    pub sorted: bool,
}

impl Index<usize> for Abstract {
    type Output = ElementList;

    fn index(&self, index: usize) -> &Self::Output {
        &self.ranks[index]
    }
}

impl IndexMut<usize> for Abstract {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.ranks[index]
    }
}

// todo: remove?
impl From<Ranks> for Abstract {
    fn from(ranks: Ranks) -> Self {
        Self {
            ranks,
            sorted: false,
        }
    }
}

impl IntoIterator for Abstract {
    type Item = ElementList;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.ranks.into_iter()
    }
}

impl VecLike for Abstract {
    type VecItem = ElementList;
    type VecIndex = usize;

    fn as_inner(&self) -> &Vec<ElementList> {
        self.ranks.as_inner()
    }

    fn as_inner_mut(&mut self) -> &mut Vec<ElementList> {
        self.ranks.as_inner_mut()
    }

    fn into_inner(self) -> Vec<ElementList> {
        self.ranks.into_inner()
    }

    fn from_inner(vec: Vec<ElementList>) -> Self {
        Ranks::from_inner(vec).into()
    }
}

impl Abstract {
    /// Initializes a new polytope with the capacity needed to store elements up
    /// to a given rank.
    pub fn with_rank_capacity(rank: usize) -> Self {
        Ranks::with_rank_capacity(rank).into()
    }

    /// Returns a reference to the minimal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    pub fn min(&self) -> &Element {
        self.get_element(0, 0).unwrap()
    }

    /// Returns a mutable reference to the minimal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    pub fn min_mut(&mut self) -> &mut Element {
        self.get_element_mut(0, 0).unwrap()
    }

    /// Returns a reference to the maximal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    pub fn max(&self) -> &Element {
        self.get_element(self.rank(), 0).unwrap()
    }

    pub fn edges_mut(&mut self) -> &mut ElementList {
        &mut self[2]
    }

    /// Returns a mutable reference to the maximal element of the polytope.
    ///
    /// # Panics
    /// Panics if the polytope has not been initialized.
    pub fn max_mut(&mut self) -> &mut Element {
        self.get_element_mut(self.rank(), 0).unwrap()
    }

    /// Pushes a given element into the vector of elements of a given rank.
    pub fn push_at(&mut self, rank: usize, el: Element) {
        self[rank].push(el);
    }

    /// Pushes a given element into the vector of elements of a given rank.
    /// Updates the superelements of its subelements automatically.
    pub fn push_subs_at(&mut self, rank: usize, sub_el: Subelements) {
        let i = self[rank].len();

        if rank != 0 {
            if let Some(lower_rank) = self.ranks.get_mut(rank - 1) {
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

    /// Returns a reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`element`](Self::element).
    pub fn get_element(&self, rank: usize, idx: usize) -> Option<&Element> {
        self.ranks.get(rank)?.get(idx)
    }

    /// Returns a mutable reference to an element of the polytope. To actually get the
    /// entire polytope it defines, use [`element`](Self::element).
    pub fn get_element_mut(&mut self, rank: usize, idx: usize) -> Option<&mut Element> {
        self.ranks.get_mut(rank)?.get_mut(idx)
    }

    /// Gets the indices of the vertices of an element in the polytope, if it
    /// exists.
    pub fn element_vertices(&self, rank: usize, idx: usize) -> Option<Vec<usize>> {
        Some(ElementHash::new(self, rank, idx)?.to_vertices())
    }

    /// Gets both elements with a given rank and index as a polytope and the
    /// indices of its vertices on the original polytope, if it exists.
    pub fn element_and_vertices(&self, rank: usize, idx: usize) -> Option<(Vec<usize>, Self)> {
        let element_hash = ElementHash::new(self, rank, idx)?;
        Some((element_hash.to_vertices(), element_hash.to_polytope(self)))
    }

    /// Returns a map from the elements in a polytope to the index of one of its
    /// vertices. Does not map the minimal element anywhere.
    pub fn vertex_map(&self) -> ElementMap<usize> {
        // Maps every element of the polytope to one of its vertices.
        let mut vertex_map = ElementMap::new();

        // Vertices map to themselves.
        let vertex_count = self.vertex_count();
        let mut vertex_list = Vec::with_capacity(vertex_count);
        for v in 0..vertex_count {
            vertex_list.push(v);
        }
        vertex_map.push(vertex_list);

        // Every other element maps to the vertex of any subelement.
        for r in 2..=self.rank() {
            let mut element_list = Vec::new();

            for el in &self.ranks()[r] {
                element_list.push(vertex_map[r - 2][el.subs[0]]);
            }

            vertex_map.push(element_list);
        }

        vertex_map
    }

    /// Returns the indices of a Petrial polygon in cyclic order, or `None` if
    /// it self-intersects.
    pub fn petrie_polygon_vertices(&mut self, flag: Flag) -> Option<Vec<usize>> {
        let rank = self.rank();
        let mut new_flag = flag.clone();
        let first_vertex = flag[0];

        let mut vertices = Vec::new();
        let mut vertex_hash = HashSet::new();

        self.element_sort();

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
        let mut backwards_abs = Vec::with_capacity(rank + 1);
        backwards_abs.push(SubelementList::max(section_hash.len()));

        // Indices of base.
        let vertex_count = self.vertex_count();
        let mut vertices = Vec::with_capacity(vertex_count);

        // Indices of dual base.
        let facet_count = self.facet_count();
        let mut dual_vertices = Vec::with_capacity(facet_count);

        // Adds all elements corresponding to sections of a given height.
        for height in 1..=rank + 1 {
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
                for &idx_lo in &self
                    .get_element(section.lo_rank, section.lo_idx)
                    .unwrap()
                    .subs
                {
                    // Adds the new sections of the current height, gets
                    // their index, uses that to build the ElementList.
                    let sub = new_section_hash.get(SectionRef::from_els(
                        (section.lo_rank - 1, idx_lo),
                        section.hi(),
                    ));

                    elements[idx].push(sub);
                }

                // Finds all of the superelements of our old section's
                // highest element.
                for &idx_hi in &self
                    .get_element(section.hi_rank, section.hi_idx)
                    .unwrap()
                    .sups
                {
                    // Adds the new sections of the current height, gets
                    // their index, uses that to build the ElementList.
                    let sub = new_section_hash.get(SectionRef::from_els(
                        section.lo(),
                        (section.hi_rank + 1, idx_hi),
                    ));

                    elements[idx].push(sub);
                }
            }

            // We figure out where the vertices of the base and the dual base
            // were sent.
            if height == rank - 1 {
                // We create a map from the base's vertices to the new vertices.
                for v in 0..vertex_count {
                    vertices.push(new_section_hash.get(SectionRef::new(1, v, rank, 0)));
                }

                // We create a map from the dual base's vertices to the new vertices.
                for f in 0..facet_count {
                    dual_vertices.push(new_section_hash.get(SectionRef::new(0, 0, rank - 1, f)));
                }
            }

            backwards_abs.push(elements);
            section_hash = new_section_hash;
        }

        // We built this backwards, so let's fix it.
        let mut abs = AbstractBuilder::with_capacity(backwards_abs.len() - 1);

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
        let mut ranks = Vec::with_capacity(rank + 1);

        // Adds elements of each rank.
        for _ in 0..rank {
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
        let min_count = self.el_count(0);
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
        for (r, elements) in self.ranks.iter().enumerate() {
            // Iterates over all such elements.
            for (idx, el) in elements.iter().enumerate() {
                // Only the minimal element can have no subelements.
                if r != 0 && el.subs.len() == 0 {
                    return Err(AbstractError::Ranked {
                        el: (r, idx),
                        incidence_type: IncidenceType::Subelement,
                    });
                }

                // Iterates over the element's subelements.
                for &sub in &el.subs {
                    // Attempts to get the subelement's superelements.
                    if r >= 1 {
                        if let Some(sub_el) = self.get_element(r - 1, sub) {
                            if sub_el.sups.contains(&idx) {
                                continue;
                            } else {
                                // The element contains a subelement, but not viceversa.
                                return Err(AbstractError::Consistency {
                                    el: (r, idx),
                                    index: sub,
                                    incidence_type: IncidenceType::Subelement,
                                });
                            }
                        }
                    }

                    // We got ourselves an invalid index.
                    return Err(AbstractError::Index {
                        el: (r, idx),
                        index: sub,
                        incidence_type: IncidenceType::Subelement,
                    });
                }

                // Only the maximal element can have no superelements.
                if r != self.rank() && el.sups.len() == 0 {
                    return Err(AbstractError::Ranked {
                        el: (r, idx),
                        incidence_type: IncidenceType::Superelement,
                    });
                }

                // Iterates over the element's superelements.
                for &sup in &el.sups {
                    // Attempts to get the subelement's superelements.
                    if let Some(sub_el) = self.get_element(r + 1, sup) {
                        if sub_el.subs.contains(&idx) {
                            continue;
                        } else {
                            // The element contains a superelement, but not viceversa.
                            return Err(AbstractError::Consistency {
                                el: (r, idx),
                                index: sup,
                                incidence_type: IncidenceType::Superelement,
                            });
                        }
                    }

                    // We got ourselves an invalid index.
                    return Err(AbstractError::Index {
                        el: (r, idx),
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
        for r in 2..self.rank() {
            for (idx, el) in self[r].iter().enumerate() {
                let mut hash_sub_subs = HashMap::new();

                for &sub in &el.subs {
                    let sub_el = &self[(r - 1, sub)];

                    for &sub_sub in &sub_el.subs {
                        match hash_sub_subs.get(&sub_sub) {
                            // Found for the first time.
                            None => hash_sub_subs.insert(sub_sub, Count::Once),

                            // Found for the second time.
                            Some(Count::Once) => hash_sub_subs.insert(sub_sub, Count::Twice),

                            // Found for the third time?! Abort!
                            Some(Count::Twice) => {
                                return Err(AbstractError::Dyadic {
                                    section: SectionRef::new(r - 2, sub_sub, r, idx),
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
                            section: SectionRef::new(r - 2, sub_sub, r, idx),
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
}

impl Polytope for Abstract {
    type DualError = Infallible;

    fn abs(&self) -> &Abstract {
        self
    }

    fn abs_mut(&mut self) -> &mut Abstract {
        self
    }

    fn into_abs(self) -> Abstract {
        self
    }

    /// Returns an instance of the
    /// [nullitope](https://polytope.miraheze.org/wiki/Nullitope), the unique
    /// polytope of rank &minus;1.
    fn nullitope() -> Self {
        Self {
            ranks: Ranks::from_inner(vec![ElementList::min(0)]),
            sorted: true,
        }
    }

    /// Returns an instance of the
    /// [point](https://polytope.miraheze.org/wiki/Point), the unique polytope
    /// of rank 0.
    fn point() -> Self {
        Self {
            ranks: Ranks::from_inner(vec![ElementList::min(1), ElementList::max(1)]),
            sorted: true,
        }
    }

    /// Returns an instance of the
    /// [dyad](https://polytope.miraheze.org/wiki/Dyad), the unique polytope of
    /// rank 1.
    fn dyad() -> Self {
        let mut abs = AbstractBuilder::with_capacity(2);

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

        let mut poly = AbstractBuilder::with_capacity(3);

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
    fn try_dual(&self) -> Result<Self, Self::DualError> {
        let mut clone = self.clone();
        clone.dual_mut();
        Ok(clone)
    }

    /// Converts a polytope into its dual in place. Use [`Self::dual_mut`] instead, as
    /// this method can never fail.
    fn try_dual_mut(&mut self) -> Result<(), Self::DualError> {
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
        if self.rank() != 4 {
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

        self.element_sort();
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

                flag.change_mut(self, 0);
                traversed_flags.insert(flag.change(self, 2));
                flag.change_mut(self, 1);
                flag.change_mut(self, 2);
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
        for edge in self.edges_mut().iter_mut() {
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
    fn try_antiprism(&self) -> Result<Self, Self::DualError> {
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

        for (r, elements) in p.ranks.into_iter().enumerate().skip(1).take(rank - 1) {
            let sub_offset = el_counts[r - 1];
            let sup_offset = el_counts[r + 1];

            for mut el in elements.into_iter() {
                if r != 1 {
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

        // We don't need to do this every single time.
        *self.min_mut() = Element::min(self.vertex_count());
        *self.max_mut() = Element::max(dbg!(self.facet_count()));
    }

    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, rank: usize, idx: usize) -> Option<Self> {
        Some(ElementHash::new(self, rank, idx)?.to_polytope(self))
    }

    /// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// from two polytopes.
    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::product::<false, false>(p, q)
    }

    /// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
    /// from two polytopes.
    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::product::<true, false>(p, q)
    }

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::product::<false, true>(p, q)
    }

    /// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
    /// from two polytopes.
    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::product::<true, true>(p, q)
    }

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope in place. Does nothing in the case of the nullitope.
    fn ditope_mut(&mut self) {
        let rank = self.rank();
        if rank != 0 {
            self.push_subs_at(rank, self.max().subs.clone());
            self.push_max();
        }
    }

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope in place. Does nothing in case of the nullitope.
    fn hosotope_mut(&mut self) {
        if self.rank() != 0 {
            self.min_mut().subs.push(0);
            let min = self.min().clone();
            self[0].push(min);

            for v in &mut self[1] {
                v.subs.push(1);
            }

            self.insert(0, ElementList::min(2));
        }
    }
}

impl Index<(usize, usize)> for Abstract {
    type Output = Element;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self[index.rank()][index.idx()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Abstract::hypercube(4),
            Abstract::hypercube(5),
            Abstract::hypercube(6),
            Abstract::simplex(4),
            Abstract::simplex(5),
            Abstract::simplex(6),
            Abstract::orthoplex(4),
            Abstract::orthoplex(5),
            Abstract::orthoplex(6),
            Abstract::duoprism(&Abstract::polygon(6), &Abstract::polygon(7)),
            Abstract::dyad().ditope().ditope().ditope().ditope(),
        ]
    }

    /// Tests whether a polytope's element counts match the expected element
    /// counts, and whether a polytope is valid.
    fn test(poly: &Abstract, element_counts: &[usize]) {
        assert_eq!(
            &poly.el_counts(),
            &element_counts,
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
        test(&Abstract::nullitope(), &[1]);
    }

    #[test]
    /// Checks that a point is generated correctly.
    fn point() {
        test(&Abstract::point(), &[1, 1]);
    }

    #[test]
    /// Checks that a dyad is generated correctly.
    fn dyad() {
        test(&Abstract::dyad(), &[1, 2, 1]);
    }

    #[test]
    /// Checks that polygons are generated correctly.
    fn polygon() {
        for n in 2..=10 {
            test(&Abstract::polygon(n), &[1, n, n, 1]);
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
                    &[
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
                    &[1, m * n, 2 * m * n, m + n + m * n, m + n, 1],
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
                    &[1, m + n, m + n + m * n, 2 * m * n, m * n, 1],
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
                    &[1, m * n, 2 * m * n, m * n, 1],
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
        for n in 0..=7 {
            let simplex = Abstract::simplex(n);
            let mut element_counts = Vec::with_capacity(n + 1);

            for k in 0..=n {
                element_counts.push(choose(n, k));
            }

            test(&simplex, &element_counts);
        }
    }

    #[test]
    /// Checks that hypercubes are generated correctly.
    fn hypercube() {
        for n in 0..=6 {
            let hypercube = Abstract::hypercube(n);
            let mut element_counts = Vec::with_capacity(n + 1);

            element_counts.push(1);
            for k in 1..=n {
                element_counts.push(choose(n - 1, k - 1) * (1 << (n - k)));
            }

            test(&hypercube, &element_counts);
        }
    }

    #[test]
    /// Checks that orthoplices are generated correctly.
    fn orthoplex() {
        for n in 0..=6 {
            let orthoplex = Abstract::orthoplex(n);
            let mut element_counts = Vec::with_capacity(n + 1);

            for k in 0..n {
                element_counts.push(choose(n - 1, k) * (1 << k));
            }
            element_counts.push(1);

            test(&orthoplex, &element_counts);
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
