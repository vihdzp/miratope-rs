//! Declares the [`Abstract`] polytope type and all associated data structures.

pub mod antiprism;
pub mod elements;
pub mod flag;
pub mod product;
pub mod valid;

use std::{
    collections::{BTreeSet, HashMap},
    convert::Infallible,
    ops::{Index, IndexMut},
    slice, vec,
};

use self::flag::{Flag, FlagSet};
use super::Polytope;

use vec_like::VecLike;

pub use antiprism::*;
pub use elements::*;
pub use product::*;
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
/// expressed with an `usize`, we'll internally start indexing by 0. That is,
/// vertices have rank 1, edges have rank 2, and so on.
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

    /// Sets the metadata of the polytope to be sorted.
    ///
    /// # Safety
    /// All of the indices of all of the subelements and superelements of the
    /// polytope must be sorted. Setting this flag incorrectly will cause
    /// algorithms to behave unpredictably.
    pub unsafe fn set_sorted(&mut self) {
        self.meta.sorted = true;
    }

    /// Returns an iterator over the [`ElementLists`](ElementList) of each rank.
    pub fn iter(&self) -> slice::Iter<'_, ElementList> {
        self.ranks.iter()
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
    /// # Safety
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

        // TODO: wrap this using an AbstractBuilderRev.
        let mut abs = AbstractBuilder::with_rank_capacity(rank);
        for subelements in ranks.into_iter().rev() {
            abs.push(subelements);
        }

        // Safety: we've built an omnitruncate based on the polytope. For a
        // proof that this construction yields a valid abstract polytope, see
        // [TODO: write proof].
        (unsafe { abs.build() }, flags)
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
            poly.set_sorted();
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
            poly.set_sorted();
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
            let mut abs = builder.build();
            abs.set_sorted();
            abs
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
            let mut abs = builder.build();
            abs.set_sorted();
            abs
        }
    }

    /// Converts a polytope into its dual.
    fn try_dual(&self) -> Result<Self, Self::DualError> {
        let mut clone = self.clone();
        clone.dual_mut();
        Ok(clone)
    }

    /// Converts a polytope into its dual in place. Use [`Self::dual_mut`] instead, as
    /// this method can never fail.
    fn try_dual_mut(&mut self) -> Result<(), Self::DualError> {
        // Safety: we'll swap the subelements and superelements in each element,
        // then reverse the ranks, thus building the dual, which is a valid
        // abstract polytope.
        let sorted = self.sorted();
        let ranks = unsafe { self.ranks_mut() };
        ranks.for_each_element_mut(Element::swap_mut);
        ranks.reverse();

        // Safety: if the original elements were sorted, so will these be.
        if sorted {
            unsafe {
                self.set_sorted();
            }
        }

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
            //
            // TODO: this loop is awkward, rewrite.
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

        // TODO MAKE THIS SOUND!
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
        let poly_counts: Vec<_> = poly.el_count_iter().collect();

        assert_eq!(
            &poly_counts, &element_counts,
            "{} element counts don't match expected value.",
            "TBA: name"
        );

        poly.assert_valid();
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
            poly.assert_valid();
        }
    }

    #[test]
    /// Checks that duals are generated correctly.
    fn dual_check() {
        for poly in test_polytopes().iter_mut() {
            // The element counts of the dual should be the same as the reversed
            // element counts of the original.
            let el_counts: Vec<_> = poly.el_count_iter().collect();
            poly.dual_mut();
            let new_el_counts: Vec<_> = poly.el_count_iter().rev().collect();
            assert_eq!(
                el_counts, new_el_counts,
                "Dual element counts of {} don't match expected value.",
                "TBA: name"
            );

            // The duals should also be valid polytopes.
            poly.assert_valid();
        }
    }
}
