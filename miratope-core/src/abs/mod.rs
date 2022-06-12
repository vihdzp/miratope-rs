//! Declares the [`Abstract`] polytope type and all associated data structures.

pub mod antiprism;
pub mod flag;
pub mod product;
pub mod ranked;
pub mod valid;

use std::{
    collections::{BTreeSet, HashMap},
    convert::Infallible,
    ops::{Index, IndexMut},
    slice, vec, iter,
};

use self::flag::{Flag, FlagSet};
use super::Polytope;

use vec_like::VecLike;

use partitions::{PartitionVec, partition_vec};

pub use ranked::*;
pub use valid::*;

/// Contains some metadata about how a polytope has been built up, which can
/// then be used by methods on polytopes to avoid expensive recomputations.
///
/// This struct is not stable, and its fields are subject to change as we see
/// fit.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Metadata {
    /// Whether every single element's subelements and superelements are sorted
    /// by index. This is a necessary condition for the methods that iterate
    /// over flags.
    sorted: bool,
}

impl Default for Metadata {
    fn default() -> Self {
        Self { sorted: false }
    }
}

impl Metadata {
    /// Resets the metadata to its default state.
    pub fn reset(&mut self) {
        *self = Default::default();
    }
}

/// Encodes the ranked poset corresponding to an abstract polytope. Contains
/// both a `Vec` of [`ElementLists`](ElementList), and some metadata about it.
///
/// # Inner representation
/// An `Abstract` wraps around a list of [`Ranks`], which themselves wrap around
/// a `Vec<ElementList>`. Elements of rank *r* are stored as the *r*-th list.
///
/// An [`ElementList`] wraps around a `Vec` of [`Elements`](Element). Every
/// element stores the indices of the incident [`Subelements`] of the previous
/// rank, as well as the [`Superelements`] of the next rank.
///
/// The other attribute of the struct is its [`Metadata`], which just caches
/// the results of expensive computations.
///
/// # Definition
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
/// convenience.
///
/// The usual convention is to choose the rank function so that the minimal
/// element has rank &minus;1. However, since indexing is most naturally
/// expressed with an unsigned integer, we'll internally start indexing by 0.
/// That is, vertices have rank 1, edges have rank 2, and so on.
///
/// For more info, see [Wikipedia](https://en.wikipedia.org/wiki/Abstract_polytope)
/// or the [Polytope Wiki](https://polytope.miraheze.org/wiki/Abstract_polytope).
///
/// # An invariant
/// Every method you call on an `Abstract` must be able to assume that its input
/// is a valid polytope. Furthermore, every single method that returns an
/// `Abstract` must return a valid polytope. Methods that can break this
/// invariant if used improperly are marked with `unsafe`.
///
/// This restriction allows us to properly optimize these methods without
/// worrying about invalid cases.
///
/// Another thing: the metadata must match the polytope it describes.
///
/// # How to use
/// There are two main ways to build a new `Abstract`. The simplest is via the
/// [`AbstractBuilder`] struct, which allows one to build a polytope layer by
/// layer by providing only the [`SubelementLists`](SubelementList) of each
/// rank. Superelements will be set automatically.
///
/// The other way is to build up the `Ranks` manually and convert them into an
/// `Abstract` via [`Abstract::from_ranks`], although this is much harder and
/// quite prone to mistakes.
#[derive(Debug, Clone)]
pub struct Abstract {
    /// The list of element lists in the polytope.
    ranks: Ranks,

    /// Some metadata about the [`Ranks`].
    meta: Metadata,
}

impl From<Abstract> for Ranks {
    fn from(abs: Abstract) -> Self {
        abs.ranks
    }
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

impl IntoIterator for Abstract {
    type Item = ElementList;
    type IntoIter = vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.ranks.into_iter()
    }
}

/// Every polytope is ranked.
impl<T: Polytope> Ranked for T {
    fn ranks(&self) -> &Ranks {
        &self.abs().ranks
    }

    fn into_ranks(self) -> Ranks {
        self.into_abs().into()
    }
}

impl Abstract {
    /// Initializes a new polytope from a list of [`Ranks`].
    ///
    /// # Safety
    /// You must make sure that the [`Ranks`] satisfy the conditions for an
    /// abstract polytope. These are outlined in the documentation for the
    /// [`Abstract`] type.
    pub unsafe fn from_ranks(ranks: Ranks) -> Self {
        Self {
            ranks,
            meta: Default::default(),
        }
    }

    /// Returns a mutable reference to the [`Ranks`] of the polytope. As a side
    /// effect, this will reset the polytope's metadata.
    ///
    /// # Safety
    /// The user must certify that any modification done to the polytope
    /// ultimately results in a valid [`Abstract`].
    pub unsafe fn ranks_mut(&mut self) -> &mut Ranks {
        self.meta.reset();
        &mut self.ranks
    }

    /// Returns whether the indices of all the subelements and superelements are
    /// sorted. Gets this from the polytope's metadata.
    pub fn sorted(&self) -> bool {
        self.meta.sorted
    }

    /// Sets the metadata of the polytope that stores whether the indices of the
    /// polytope's subelements and superelements are sorted to a given value.
    ///
    /// # Safety
    /// Setting this flag incorrectly will cause algorithms to behave
    /// unpredictably, potentially causing UB.
    pub unsafe fn set_sorted(&mut self, sorted: bool) {
        self.meta.sorted = sorted;
    }

    /// Returns an iterator over the [`ElementLists`](ElementList) of each rank.
    pub fn iter(&self) -> slice::Iter<'_, ElementList> {
        self.ranks.iter()
    }

    /// Takes the dual of an abstract polytope in place. This can never fail.
    pub fn dual_mut(&mut self) {
        // Safety: duals of polytopes are polytopes.
        let sorted = self.sorted();
        let ranks = unsafe { self.ranks_mut() };
        ranks.for_each_element_mut(Element::swap_mut);
        ranks.reverse();

        // Safety: if the original elements were sorted, so will these be.
        unsafe {
            self.set_sorted(sorted);
        }
    }

    /// Takes the dual of an abstract polytope. This can never fail.
    pub fn dual(&self) -> Self {
        let mut clone = self.clone();
        clone.dual_mut();
        clone
    }

    /// Converts an abstract polytope into its dual. This can never fail.
    pub fn into_dual(mut self) -> Self {
        self.dual_mut();
        self
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. Also returns the indices of the vertices that
    /// form the base and the dual base, in that order.
    pub fn antiprism_and_vertices(&self) -> (Self, Vec<usize>, Vec<usize>) {
        antiprism::antiprism_and_vertices(self)
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. This can never fail for an abstract polytope.
    pub fn antiprism(&self) -> Self {
        antiprism::antiprism(self)
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

    /// Returns the omnitruncate of a polytope, along with the flags that make
    /// up its respective vertices.
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
    pub fn omnitruncate_and_flags(&self) -> (Self, Vec<Flag>) {
        let mut flag_sets = vec![FlagSet::new_all(self)];
        let mut new_flag_sets = Vec::new();
        let rank = self.rank();

        // The elements of each rank... backwards.
        let mut ranks = Vec::with_capacity(rank + 1);

        // Adds elements of each rank, except for vertices and the minimal
        // element.
        for _ in (2..=rank).rev() {
            let mut subelements = SubelementList::new();

            // Gets the subelements of each element.
            for flag_set in flag_sets {
                let mut subs = Subelements::new();

                // Each subset represents a new element.
                // todo: just return an iterator here.
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

        // todo: it might be better to just return an iterator.
        let mut flags = Vec::new();
        for flag_set in flag_sets {
            debug_assert_eq!(flag_set.len(), 1);
            flags.push(flag_set.flags.into_iter().next().unwrap());
        }

        ranks.push(SubelementList::vertices(flags.len()));
        ranks.push(SubelementList::min());

        let builder: AbstractBuilder = ranks.into_iter().rev().collect();

        // Safety: we've built an omnitruncate based on the polytope. For a
        // proof that this construction yields a valid abstract polytope, see
        // [TODO: write proof].
        (unsafe { builder.build() }, flags)
    }

    /// Returns an arbitrary truncate as an abstract polytope.
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
    pub fn truncate_and_flags(&self, truncate_type: Vec<usize>) -> (Self, Vec<Vec<usize>>) {
        let omni_and_flags = self.omnitruncate_and_flags();
        let omni = omni_and_flags.0;
        let omni_flags = omni_and_flags.1;
        let mut cd = vec![false; self.rank() - 1];
        for i in &truncate_type {
            cd[*i] = true;
        }

        let mut builder = AbstractBuilder::new();

        // maps omnitruncate elements to new elements
        let mut dict = Vec::new();
        for _i in 1..self.rank() {
            dict.push(HashMap::new());
        }

        // maps subflags to new vertices
        let mut verts_dict = HashMap::new();
        let mut verts_subflags = Vec::new();
        // current vertex index
        let mut c = 0;
        // gets new vertices by identifying all with the same subflag
        for (i, _vert) in omni[1].iter().enumerate() {
            let mut subflag = Vec::new();
            for r in truncate_type.iter() {
                subflag.push(omni_flags[i][*r + 1]);
            }
            match verts_dict.get(&subflag) {
                // existing vertex
                Some(idx) => {
                    dict[0].insert(i, *idx);
                }
                // new vertex
                None => {
                    verts_dict.insert(subflag.clone(), c);
                    verts_subflags.push(subflag);
                    dict[0].insert(i, c);
                    c += 1;
                }
            }
        }
        builder.push_min();
        builder.push_vertices(c);

        for rank in 2..self.rank() {
            let mut sublist = SubelementList::new();
            let mut subs_map = HashMap::new(); // used for removing duplicate elements
            c = 0;
            for (i, el) in omni[rank].iter().enumerate() {
                // checks if degenerate
                let verts_of_el = omni.element_vertices(rank, i).unwrap();
                let mut flags_of_el = Vec::new();
                for vert_of_el in verts_of_el {
                    flags_of_el.push(omni_flags[vert_of_el].clone());
                }
                // TODO: make this a `FlagChanges`? idk what that's used for
                let mut flag_changes_of_el = Vec::<usize>::new();
                for j in 0..self.rank() - 1 {
                    for flag in &flags_of_el {
                        if flag[j + 1] != flags_of_el[0][j + 1] {
                            flag_changes_of_el.push(j);
                            break;
                        }
                    }
                }
                let mut valid = true;
                let mut prev = 123456789; // yes violeta idk how to use None or something
                for j in flag_changes_of_el {
                    // same component
                    if j == prev + 1 {
                        if !valid {
                            valid = cd[j];
                        }
                    }
                    // different component
                    else {
                        if !valid {
                            break;
                        }
                        valid = cd[j];
                    }

                    prev = j;
                }
                if !valid {
                    // is degenerate
                    continue;
                }

                let mut subs = Subelements::new();
                for old_sub in el.subs.clone() {
                    if let Some(idx) = dict[rank - 2].get(&old_sub) {
                        subs.push(*idx)
                    }
                }

                subs.sort();
                if let Some(idx) = subs_map.get(&subs) {
                    dict[rank - 1].insert(i, *idx);
                    continue;
                }

                subs_map.insert(subs.clone(), c);
                dict[rank - 1].insert(i, c);
                c += 1;

                sublist.push(subs);
            }
            builder.push(sublist);
        }
        builder.push_max();

        // Safety: we've built a truncate based on the polytope. For a proof
        // that this construction yields a valid abstract polytope, see [TODO:
        // write proof].
        (unsafe { builder.build() }, verts_subflags)
    }
	
	/// Returns whether a polytope is compound
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
    pub fn is_compound(&self) -> bool {
		let flag_set = FlagSet::new_all(self);
        flag_set.len() != self.flags().count()
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
        // Safety: the nullitope is a valid polytope, and its indices are sorted.
        unsafe {
            let mut poly = Self::from_ranks(vec![ElementList::min(0)].into());
            poly.set_sorted(true);
            poly
        }
    }

    /// Returns an instance of the
    /// [point](https://polytope.miraheze.org/wiki/Point), the unique polytope
    /// of rank 0.
    fn point() -> Self {
        // Safety: the point is a valid polytope, and its indices are sorted.
        unsafe {
            let mut poly = Self::from_ranks(vec![ElementList::min(1), ElementList::max(1)].into());
            poly.set_sorted(true);
            poly
        }
    }

    /// Returns an instance of the
    /// [dyad](https://polytope.miraheze.org/wiki/Dyad), the unique polytope of
    /// rank 1.
    fn dyad() -> Self {
        let mut builder = AbstractBuilder::with_rank_capacity(2);
        builder.push_min();
        builder.push_vertices(2);
        builder.push_max();

        // Safety: the dyad is a valid polytope, and its indices are sorted.
        unsafe {
            let mut poly = builder.build();
            poly.set_sorted(true);
            poly
        }
    }

    /// Returns an instance of a [polygon](https://polytope.miraheze.org/wiki/Polygon)
    /// with a given number of sides.
    fn polygon(n: usize) -> Self {
        assert!(n >= 2, "A polygon must have at least 2 sides.");
        let mut edges = SubelementList::with_capacity(n);

        // We add the edges with their indices sorted.
        for i in 1..n {
            edges.push(vec![i - 1, i].into());
        }
        edges.push(vec![0, n - 1].into());

        let mut builder = AbstractBuilder::with_rank_capacity(3);
        builder.push_min();
        builder.push_vertices(n);
        builder.push(edges);
        builder.push_max();

        // Safety: a polygon is a valid polytope, and its indices are sorted.
        unsafe {
            let mut poly = builder.build();
            poly.set_sorted(true);
            poly
        }
    }

    /// Builds an [orthoplex](https://polytope.miraheze.org/wiki/Orthoplex) with
    /// a given rank.
    fn orthoplex(rank: usize) -> Self {
        if rank == 0 {
            Self::nullitope()
        } else {
            Self::multitegum(iter::repeat(&Self::dyad()).take(rank - 1))
        }
    }

    fn vertex_map(&self) -> ElementMap<usize> {
        // Maps every element of the polytope to one of its vertices.
        let mut vertex_map = ElementMap::new();
        vertex_map.push(Vec::new());

        // Vertices map to themselves.
        if self.rank() != 0 {
            vertex_map.push((0..self.vertex_count()).collect());
        }

        // Every other element maps to the vertex of any subelement.
        for (r, elements) in self.ranks.iter().enumerate().skip(2) {
            vertex_map.push(
                elements
                    .iter()
                    .map(|el| vertex_map[(r - 1, el.subs[0])])
                    .collect(),
            );
        }

        vertex_map
    }

    /// Converts a polytope into its dual.
    fn try_dual(&self) -> Result<Self, Self::DualError> {
        Ok(self.dual())
    }

    /// Converts a polytope into its dual in place. Use [`Self::dual_mut`] instead, as
    /// this method can never fail.
    fn try_dual_mut(&mut self) -> Result<(), Self::DualError> {
        self.dual_mut();
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
            let mut edge = flag[2];
            let mut loop_continue = true;

            // We apply our flag changes and mark our flags until we reach the
            // original edge. We then intentionally overshoot and do it one more
            // time.
            //
            // TODO: this loop is awkward, rewrite.
            while loop_continue {
                loop_continue = face.insert(edge);

                flag.change_mut(self, 1);
                traversed_flags.insert(flag.change(self, 3));
                flag.change_mut(self, 2);
                flag.change_mut(self, 3);
                traversed_flags.insert(flag.clone());

                edge = flag[2];
            }

            // If the edge we found after we returned to the original edge was
            // not already in the face, this means that the Petrial loop
            // self-intersects, and hence the Petrial is not a valid polytope.
            if !face.contains(&edge) {
                return false;
            }

            faces.push(face.into_iter().collect());
        }

        // Safety: TODO we need to define the safety guarantees of this function.
        let ranks = unsafe { self.ranks_mut() };

        // Removes the faces and maximal polytope from self.
        ranks.pop();
        ranks.pop();

        // Pushes the new faces and a new maximal element.
        for el in &mut ranks[2] {
            el.sups.clear();
        }

        let face_count = faces.len();
        let mut new_faces = ElementList::with_capacity(face_count);
        for (idx, face) in faces.into_iter().enumerate() {
            for &sub in &face {
                ranks[(2, sub)].sups.push(idx);
            }

            new_faces.push(Element {
                sups: vec![0].into(),
                subs: face,
            });
        }

        ranks.push(new_faces);
        ranks.push(ElementList::max(face_count));

        // Checks for dyadicity, since that sometimes fails.
        ranks.ranks().is_dyadic().is_ok()

        // TODO MAKE THIS SOUND instead of just returning whether it failed or not!
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
        if rank <= 1 {
            return;
        }

        // The polytopes must have the same ranks.
        assert_eq!(
            rank,
            p.rank(),
            "polytopes in a compound must have the same rank"
        );

        // The element counts of the polytope, except we pretend there's no min
        // or max elements.
        let mut el_counts: Vec<_> = self.el_count_iter().collect();
        el_counts[0] = 0;
        el_counts[rank] = 0;

        for (r, elements) in p.into_iter().enumerate().skip(1).take(rank - 1) {
            for mut el in elements.into_iter() {
                let sub_offset = el_counts[r - 1];
                let sup_offset = el_counts[r + 1];

                for sub in el.subs.iter_mut() {
                    *sub += sub_offset;
                }

                for sup in el.sups.iter_mut() {
                    *sup += sup_offset;
                }

                self[r].push(el);
            }
        }

        // We don't need to do this every single time.
        *self.ranks.min_mut() = Element::min(self.vertex_count());
        *self.ranks.max_mut() = Element::max(self.facet_count());
    }

    /// Makes a polytope strongly connected. Splits compounds into their components.
    fn defiss(&self) -> Vec<Abstract> {
        if self.rank() < 1 {
            return vec![Abstract::nullitope()];
        }

        let mut output = Vec::<Abstract>::new();

        let flags: Vec<Flag> = self.flags().collect();
        let mut flags_map_back = HashMap::new();
        for (idx, flag) in flags.iter().enumerate() {
            flags_map_back.insert(flag, idx);
        }

        let mut partitions: Vec<PartitionVec<()>> = vec![partition_vec![(); flags.len()]; self.rank()];

        for (idx, flag) in flags.iter().enumerate() {
            for change in 1..self.rank() {
                let changed_flag = flag.change(self, change);
                let changed_idx = flags_map_back.get(&changed_flag).unwrap();
                
                for rank in 0..self.rank() {
                    if rank != change {
                        partitions[rank].union(idx, *changed_idx);
                    }
                }
            }
        }

        let components = partitions[0].all_sets();

        for component in components {
            let mut elements = Ranks::with_rank_capacity(self.rank());
            elements.push(ElementList::from(vec![Element::new(Subelements::new(), Superelements::new())]));
            for _ in 1..self.rank() {
                elements.push(ElementList::new());
            }

            let mut idx_in_rank = vec![HashMap::<usize, usize>::new(); self.rank()];
            let mut counts = vec![0; self.rank()];
            for (flag_idx, _) in component {
                let mut sub = 0;

                for rank in 1..self.rank() {
                    match idx_in_rank[rank].get(&flag_idx) {
                        Some(idx) => {
                            if !elements[rank][*idx].subs.contains(&sub) {
                                elements[rank][*idx].subs.push(sub);
                            }

                            sub = *idx;
                        }
                        None => {
                            let set = partitions[rank].set(flag_idx);

                            for (el, _) in set {
                                idx_in_rank[rank].insert(el, counts[rank]);
                            }
                            elements[rank].push(
                                Element{
                                    subs: Subelements::from(vec![sub]),
                                    sups: Superelements::from(vec![]),
                                });

                            sub = counts[rank];
                            counts[rank] += 1;
                        }
                    }
                }
            }
            let mut builder = AbstractBuilder::new();
            for rank in elements {
                builder.push_empty();
                for el in rank {
                    builder.push_subs(el.subs);
                }
            }
            builder.push_max();
            unsafe {
                let polytope = builder.build();
                output.push(polytope);
            }
        }

        output
    }
    
    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, rank: usize, idx: usize) -> Option<Self> {
        Some(ElementHash::new(self, rank, idx)?.to_polytope(self))
    }

    /// Gets the element figure with a given rank and index as a polytope.
    fn element_fig(&self, rank: usize, idx: usize) -> Result<Option<Self>, Self::DualError> {
        if rank <= self.rank() {
            // todo: this is quite inefficient for a small element figure since
            // we take the dual of the entire thing.
            if let Some(mut element_fig) = self.try_dual()?.element(self.rank() - rank, idx) {
                element_fig.try_dual_mut()?;
                return Ok(Some(element_fig));
            }
        }

        Ok(None)
    }

    /// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// from two polytopes.
    ///
    /// The vertices of the result will be those corresponding to the vertices
    /// of `self` in the same order, following those corresponding to `other` in
    /// the same order.
    fn duopyramid(&self, other: &Self) -> Self {
        product::duopyramid(self, other)
    }

    /// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
    /// from two polytopes.
    fn duoprism(&self, other: &Self) -> Self {
        product::duoprism(self, other)
    }

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    ///
    /// The vertices of the result will be those corresponding to the vertices
    /// of `self` in the same order, following those corresponding to `other` in
    /// the same order.
    fn duotegum(&self, other: &Self) -> Self {
        product::duotegum(self, other)
    }

    /// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
    /// from two polytopes.
    fn duocomb(&self, other: &Self) -> Self {
        product::duocomb(self, other)
    }

    /// Builds a [star product](https://en.wikipedia.org/wiki/Star_product)
    /// of two polytopes.
    fn star_product(&self, other: &Self) -> Self {
        let mut product = self.clone();
        product.ranks.pop();
        for r in 1..=other.rank() {
            product.ranks.push(other[r].clone());
        }

        let bottom_facet_count = self.el_count(self.rank()-1);
        let top_vertex_count = self.el_count(1);

        for bottom_facet in &mut product[self.rank()-1] {
            bottom_facet.sups = (0..top_vertex_count).collect();
        }
        for top_vertex in &mut product[self.rank()-0] {
            top_vertex.subs = (0..bottom_facet_count).collect();
        }

        product
    }

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope in place. Does nothing in the case of the nullitope.
    fn ditope_mut(&mut self) {
        if self.rank() != 0 {
            let rank = self.rank();
            let ranks = &mut self.ranks;

            for v in &mut ranks[rank - 1] {
                v.sups.push(1);
            }

            ranks.max_mut().sups.push(0);
            let max = ranks.max().clone();
            ranks[rank].push(max);

            ranks.push(ElementList::max(2));
        }
    }

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope in place. Does nothing in case of the nullitope.
    fn hosotope_mut(&mut self) {
        if self.rank() != 0 {
            let ranks = &mut self.ranks;

            for v in &mut ranks[1] {
                v.subs.push(1);
            }

            ranks.min_mut().subs.push(0);
            let min = ranks.min().clone();
            ranks[0].push(min);

            ranks.insert(0, ElementList::min(2));
        }
    }

    /// Splits compound faces into their components.
    fn untangle_faces(&mut self) {
        if self.rank() < 4 {
            return
        }
        let mut new_faces = ElementList::new();
        let self_3_len = self[3].len();

        for f_i in 0..self_3_len {
            let current_len = new_faces.len();
            let mut map = HashMap::new();
            let mut partition = PartitionVec::new();
            let edge_idxs = &self[3][f_i].subs.clone();
            
            for edge_idx in edge_idxs {
                let edge = &self[2][*edge_idx];

                if edge.subs.len() != 2 { // This shouldn't happen, but apparently it does sometimes when doing cross-sections
                    return
                }
                for i in 0..=1 {
                    if map.get(&edge.subs[i]).is_none() {
                        map.insert(edge.subs[i], map.len());
                        partition.push(edge.subs[i]);
                    }
                }
                partition.union(
                    *map.get(&edge.subs[0]).unwrap(),
                    *map.get(&edge.subs[1]).unwrap()
                );
            }

            let mut set_of_vertex = HashMap::new();
            for (i, set) in partition.all_sets().enumerate() {
                for (_, v) in set {
                    set_of_vertex.insert(v, i);
                }
                if i > 0 {
                    for sup in &self[3][f_i].sups.clone() {
                        self[4][*sup].subs.push(new_faces.len() + self_3_len);
                    }
                    new_faces.push(Element::new(Subelements::new(), self[3][f_i].sups.clone()));
                }
            }

            let mut new_face = self[3][f_i].clone();
            new_face.subs.clear();

            for edge_idx in edge_idxs {
                let set_idx = set_of_vertex.get(&self[2][*edge_idx].subs[0]).unwrap();
                if set_idx > &0 {
                    let idx = current_len + set_idx - 1;
                    new_faces[idx].subs.push(*edge_idx);
                    for sup_i in 0..self[2][*edge_idx].sups.len() {
                        if &self[2][*edge_idx].sups[sup_i] == &f_i {
                            self[2][*edge_idx].sups[sup_i] = idx + self_3_len;
                        }
                    }
                }
                else {
                    new_face.subs.push(*edge_idx);
                }
            }

            self[3][f_i] = new_face;
        }
        self[3].append(&mut new_faces);
    }
}

impl Index<(usize, usize)> for Abstract {
    type Output = Element;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self[index.0][index.1]
    }
}

impl IndexMut<(usize, usize)> for Abstract {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self[index.0][index.1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test;

    /// Checks a nullitope.
    #[test]
    fn nullitope() {
        test(&Abstract::nullitope(), [1]);
    }

    /// Checks a point.
    #[test]
    fn point() {
        test(&Abstract::point(), [1, 1]);
    }

    /// Checks a dyad.
    #[test]
    fn dyad() {
        test(&Abstract::dyad(), [1, 2, 1]);
    }

    /// Checks some polygons.
    #[test]
    fn polygon() {
        for n in 2..=10 {
            test(&Abstract::polygon(n), [1, n, n, 1]);
        }
    }

    /// Checks a tetrahedron.
    #[test]
    fn tetrahedron() {
        test(&Abstract::tetrahedron(), [1, 4, 6, 4, 1])
    }

    /// Checks a cube.
    #[test]
    fn cube() {
        test(&Abstract::cube(), [1, 8, 12, 6, 1])
    }

    /// Checks an octahedron.
    #[test]
    fn octahedron() {
        test(&Abstract::octahedron(), [1, 6, 12, 8, 1])
    }

    /// Returns the values C(*n*, 0), ..., C(*n*, *n*).
    fn choose(n: usize) -> Vec<usize> {
        let mut res = Vec::with_capacity(n + 1);
        res.push(1);

        for k in 0..n {
            res.push(res[k] * (n - k) / (k + 1));
        }

        res
    }

    /// Checks simplices.
    #[test]
    fn simplex() {
        let mut simplex = Abstract::nullitope();

        for n in 1..=7 {
            simplex = simplex.pyramid();
            test(&Abstract::simplex(n), choose(n));
        }
    }

    /// Returns an iterator over the element counts of an n-hypercube.
    fn orthoplex_counts(n: usize) -> impl DoubleEndedIterator<Item = usize> {
        choose(n - 1)
            .into_iter()
            .enumerate()
            .map(|(k, c)| c << k)
            .chain(std::iter::once(1))
    }

    /// Checks hypercubes.
    #[test]
    fn hypercube() {
        let mut hypercube = Abstract::point();

        for n in 2..=6 {
            hypercube = hypercube.prism();
            test(&hypercube, orthoplex_counts(n).rev());
        }
    }

    /// Checks orthoplices.
    #[test]
    fn orthoplex() {
        let mut orthoplex = Abstract::point();

        for n in 2..=6 {
            orthoplex = orthoplex.tegum();
            test(&orthoplex, orthoplex_counts(n))
        }
    }

    /// Tests a few duals.
    #[test]
    fn dual() {
        test(&Abstract::nullitope().into_dual(), [1]);
        test(&Abstract::polygon(6).into_dual(), [1, 6, 6, 1]);
        test(&Abstract::cube().into_dual(), [1, 6, 12, 8, 1]);
    }
}
