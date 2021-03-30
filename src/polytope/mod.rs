//! Contains the definitions of the different traits and structs for
//! [polytopes](https://polytope.miraheze.org/wiki/Polytope), as well as some
//! basic methods to operate on them.

use std::{
    collections::{HashMap, HashSet},
    f64::consts::SQRT_2,
    hash::Hash,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use bevy::prelude::Mesh;
use bevy::render::mesh::Indices;
use bevy::render::pipeline::PrimitiveTopology;
use geometry::{Hyperplane, Point};

use self::geometry::{Hypersphere, Matrix};

pub mod convex;
pub mod geometry;
pub mod group;
pub mod off;
pub mod shapes;

/// The names for 0-elements, 1-elements, 2-elements, and so on.
const ELEMENT_NAMES: [&str; 11] = [
    "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta", "Xenna", "Daka",
];

/// The word "Components".
const COMPONENTS: &str = "Components";

/// The trait for methods common to all polytopes.
pub trait Polytope: Sized + Clone {
    /// The [rank](https://polytope.miraheze.org/wiki/Rank) of the polytope.
    fn rank(&self) -> isize;

    /// The number of elements of a given rank.
    fn el_count(&self, rank: isize) -> usize;

    /// The element counts of the polytope.
    fn el_counts(&self) -> RankVec<usize>;

    /// Whether the polytope is
    /// [orientable](https://polytope.miraheze.org/wiki/Orientability).
    fn orientable(&self) -> bool;

    /// Returns an instance of the
    /// [nullitope](https://polytope.miraheze.org/wiki/Nullitope), the unique
    /// polytope of rank &minus;1.
    fn nullitope() -> Self;

    /// Returns an instance of the
    /// [point](https://polytope.miraheze.org/wiki/Point), the unique polytope
    /// of rank 0.
    fn point() -> Self;

    /// Returns an instance of the
    /// [dyad](https://polytope.miraheze.org/wiki/Dyad), the unique polytope of
    /// rank 1.
    fn dyad() -> Self;

    /// Returns an instance of a
    /// [polygon](https://polytope.miraheze.org/wiki/Polygon) with a given
    /// amount of sides.
    fn polygon(n: usize) -> Self;

    /// Builds a
    /// [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product) from
    /// two polytopes.
    fn duopyramid(p: &Self, q: &Self) -> Self;

    /// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
    /// from two polytopes.
    fn duoprism(p: &Self, q: &Self) -> Self;

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum(p: &Self, q: &Self) -> Self;

    /// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
    /// from two polytopes.
    fn duocomb(p: &Self, q: &Self) -> Self;

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope.
    fn ditope(&self) -> Self;

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope in place.
    fn ditope_mut(&mut self);

    /// Builds a [horotope](https://polytope.miraheze.org/wiki/Horotope) of a
    /// given polytope.
    fn horotope(&self) -> Self;

    /// Builds a [horotope](https://polytope.miraheze.org/wiki/Horotope) of a
    /// given polytope in place.
    fn horotope_mut(&mut self);

    /// Builds a [pyramid](https://polytope.miraheze.org/wiki/Pyramid) from a
    /// given base.
    fn pyramid(&self) -> Self {
        Self::duopyramid(self, &Self::point())
    }

    /// Builds a [prism](https://polytope.miraheze.org/wiki/Prism) from a
    /// given base.
    fn prism(&self) -> Self {
        Self::duoprism(self, &Self::dyad())
    }

    /// Builds a [tegum](https://polytope.miraheze.org/wiki/Bipyramid) from a
    /// given base.
    fn tegum(&self) -> Self {
        Self::duotegum(self, &Self::dyad())
    }

    /// Takes the
    /// [pyramid product](https://polytope.miraheze.org/wiki/Pyramid_product) of
    /// a set of polytopes.
    fn multipyramid(factors: &[&Self]) -> Self {
        Self::multiproduct(&Self::duopyramid, factors, Self::nullitope())
    }

    /// Takes the
    /// [prism product](https://polytope.miraheze.org/wiki/Prism_product) of a
    /// set of polytopes.
    fn multiprism(factors: &[&Self]) -> Self {
        Self::multiproduct(&Self::duoprism, factors, Self::point())
    }

    /// Takes the
    /// [tegum product](https://polytope.miraheze.org/wiki/Tegum_product) of a
    /// set of polytopes.
    fn multitegum(factors: &[&Self]) -> Self {
        Self::multiproduct(&Self::duotegum, factors, Self::point())
    }

    /// Takes the
    /// [comb product](https://polytope.miraheze.org/wiki/Comb_product) of a set
    /// of polytopes.
    fn multicomb(factors: &[&Self]) -> Self {
        Self::multiproduct(&Self::duocomb, factors, Self::point())
    }

    /// Helper method for applying an associative binary function on a list of
    /// entries.
    fn multiproduct(
        product: &dyn Fn(&Self, &Self) -> Self,
        factors: &[&Self],
        identity: Self,
    ) -> Self {
        match factors.len() {
            // An empty product just evaluates to the identity element.
            0 => identity,

            // A product of one entry is just equal to the entry itself.
            1 => factors[0].clone(),

            // Evaluates larger products recursively.
            _ => {
                let (&first, factors) = factors.split_first().unwrap();

                product(first, &Self::multiproduct(&product, factors, identity))
            }
        }
    }

    fn antiprism(&self) -> Self;

    // The basic, regular polytopes.
    fn simplex(rank: isize) -> Self {
        Self::multipyramid(&vec![&Self::point(); (rank + 1) as usize])
    }

    fn hypercube(rank: isize) -> Self {
        if rank == -1 {
            Self::nullitope()
        } else {
            Self::multiprism(&vec![&Self::dyad(); rank as usize])
        }
    }

    fn orthoplex(rank: isize) -> Self {
        if rank == -1 {
            Self::nullitope()
        } else {
            Self::multitegum(&vec![&Self::dyad(); rank as usize])
        }
    }
}

/// A `Vec` indexed by [rank](https://polytope.miraheze.org/wiki/Rank). Wraps
/// around operations that offset by a constant for our own convenience.
#[derive(Debug, Clone)]
pub struct RankVec<T>(Vec<T>);

impl<T> RankVec<T> {
    /// Constructs a new, empty `RankVec<T>`.
    fn new() -> Self {
        RankVec(Vec::new())
    }

    /// Constructs a new, empty `RankVec<T>` with the specified capacity.
    fn with_rank(rank: isize) -> Self {
        RankVec(Vec::with_capacity((rank + 2) as usize))
    }

    /// Returns the greatest rank stored in the array.
    fn rank(&self) -> isize {
        self.len() as isize - 2
    }

    /// Returns a reference to the element at a given position or `None` if out
    /// of bounds.
    fn get(&self, index: isize) -> Option<&T> {
        if index < -1 {
            None
        } else {
            self.0.get((index + 1) as usize)
        }
    }

    /// Divides one mutable slice into two at an index.
    fn split_at_mut(&mut self, mid: isize) -> (&mut [T], &mut [T]) {
        self.0.split_at_mut((mid + 1) as usize)
    }

    /// Returns a mutable reference to an element or `None` if the index is out
    /// of bounds.
    fn get_mut(&mut self, index: isize) -> Option<&mut T> {
        if index < -1 {
            None
        } else {
            self.0.get_mut((index + 1) as usize)
        }
    }

    /// Swaps two elements in the vector.
    fn swap(&mut self, a: isize, b: isize) {
        self.0.swap((a + 1) as usize, (b + 1) as usize);
    }
}

impl<T> Deref for RankVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for RankVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Index<isize> for RankVec<T> {
    type Output = T;

    fn index(&self, index: isize) -> &Self::Output {
        &self.0[(index + 1) as usize]
    }
}

impl<T> IndexMut<isize> for RankVec<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        &mut self.0[(index + 1) as usize]
    }
}

/// Represents a single element in an [`Abstract`].
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Element {
    /// The indices of the subelements of the polytope.
    pub subs: Vec<usize>,
}

impl Element {
    /// Initializes a new element with no subelements.
    fn new() -> Self {
        Self::min()
    }

    /// Builds a minimal element for a polytope.
    fn min() -> Self {
        Self { subs: vec![] }
    }

    /// Builds a maximal element adjacent to a given number of facets.
    fn max(facet_count: usize) -> Self {
        let mut subs = Vec::with_capacity(facet_count);

        for i in 0..facet_count {
            subs.push(i);
        }

        Self { subs }
    }
}

/// Represents a list of [`Elements`](Element) of the same
/// [rank](https://polytope.miraheze.org/wiki/Rank).
#[derive(Debug, Clone)]
pub struct ElementList(Vec<Element>);

impl ElementList {
    /// Initializes an empty element list.
    fn new() -> Self {
        ElementList(Vec::new())
    }

    /// Initializes an empty element list with a given capacity.
    fn with_capacity(capacity: usize) -> Self {
        ElementList(Vec::with_capacity(capacity))
    }

    /// Returns the element list for the nullitope in a polytope with a given
    /// vertex count.
    fn min() -> Self {
        Self(vec![Element::min()])
    }

    /// Returns the element list for the maximal element in a polytope with a
    /// given facet count.
    fn max(facet_count: usize) -> Self {
        Self(vec![Element::max(facet_count)])
    }

    /// Returns the element list for a set number of vertices in a polytope.
    fn vertices(vertex_count: usize) -> Self {
        let mut els = ElementList::with_capacity(vertex_count);

        for _ in 0..vertex_count {
            els.push(Element { subs: vec![0] });
        }

        els
    }
}

impl Deref for ElementList {
    type Target = Vec<Element>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ElementList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
/// Represents the [ranked poset](https://en.wikipedia.org/wiki/Graded_poset)
/// corresponding to an
/// [abstract polytope](https://polytope.miraheze.org/wiki/Abstract_polytope).
pub struct Abstract(RankVec<ElementList>);

impl Abstract {
    /// Initializes a polytope with an empty element list.
    fn new() -> Self {
        Abstract(RankVec::new())
    }

    /// Initializes a new polytope with the capacity needed to store elements up
    /// to a given rank.
    fn with_rank(rank: isize) -> Self {
        Abstract(RankVec::with_rank(rank))
    }

    /// Initializes a polytope from a vector of element lists.
    fn from_vec(vec: Vec<ElementList>) -> Self {
        Abstract(RankVec(vec))
    }

    /// Returns a reference to the minimal element of the polytope.
    fn min(&self) -> &Element {
        &self[0][0]
    }

    /// Pushes a minimal element with no superelements into the polytope. To be
    /// used in circumstances where the elements are built up in layers.
    fn push_min(&mut self) {
        // If you're using this method, the polytope should be empty.
        debug_assert!(self.0.is_empty());

        self.push(ElementList::min());
    }

    /// Pushes a minimal element with no superelements into the polytope. To be
    /// used in circumstances where the elements are built up in layers.
    fn push_vertices(&mut self, vertex_count: usize) {
        // If you're using this method, the polytope should consist of a single
        // minimal element.
        debug_assert_eq!(self.rank(), -1);

        self.push(ElementList::vertices(vertex_count))
    }

    /// Pushes a maximal element into the polytope, with the facets as
    /// subelements. To be used in circumstances where the elements are built up
    /// in layers.
    fn push_max(&mut self) {
        let facet_count = self.el_count(self.rank());
        self.push(ElementList::max(facet_count));
    }

    fn insert(&mut self, index: isize, value: ElementList) {
        self.0.insert((index + 1) as usize, value);
    }

    /// Converts a polytope into its dual.
    fn dual(&self) -> Self {
        let mut clone = self.clone();
        clone.dual_mut();
        clone
    }

    /// Converts a polytope into its dual in place.
    fn dual_mut(&mut self) {
        let rank = self.rank();

        for r in 0..=rank {
            // Clears all subelements of the previous rank.
            for mut el in self[r - 1].iter_mut() {
                el.subs = Vec::new();
            }

            // Gets the elements of the previous and current rank mutably.
            let (part1, part2) = self.split_at_mut(r);
            let prev_rank = part1.last_mut().unwrap();
            let cur_rank = &part2[0];

            // Makes the subelements of the previous rank point to the
            // corresponding superelements of the current rank.
            for (idx, el) in cur_rank.iter().enumerate() {
                for &sub in &el.subs {
                    prev_rank[sub].subs.push(idx);
                }
            }
        }

        // Clears subelements of the maximal element.
        self[rank][0].subs = Vec::new();

        // Now that all of the incidences are backwards, there's only one thing
        // left to do... flip!
        self.reverse();
    }

    /// Gets the indices of the vertices of a given element in a polytope.
    fn get_element_vertices(&self, rank: isize, idx: usize) -> Option<Vec<usize>> {
        // A nullitope doesn't have vertices.
        if rank == -1 {
            return None;
        }

        let mut indices = vec![idx];

        // Gets subindices of subindices, until reaching the vertices.
        for r in (1..=rank).rev() {
            let mut hash_subs = HashSet::new();

            for idx in indices {
                for &sub in &self[r][idx].subs {
                    hash_subs.insert(sub);
                }
            }

            indices = hash_subs.into_iter().collect();
        }

        Some(indices)
    }

    /// Gets the element with a given rank and index as a polytope.
    fn get_element(&self, _rank: isize, _idx: usize) -> Option<Self> {
        todo!()
    }

    fn section(&self, _rank_low: isize, _idx_low: usize, _rank_hi: isize, _idx_hi: usize) -> Self {
        // assert incidence.

        todo!()
    }

    /// Calls [has_min_max_elements](Abstract::has_min_max_elements),
    /// [check_incidences](Abstract::check_incidences),
    /// [is_dyadic](Abstract::is_dyadic), and
    /// [is_strongly_connected](Abstract::is_strongly_connected).
    fn full_check(&self) -> bool {
        self.has_min_max_elements() && self.check_incidences() && self.is_dyadic()
        // && self.is_strongly_connected()
    }

    /// Determines whether the polytope has a single minimal element and a
    /// single maximal element. A valid polytope should always return `true`.
    fn has_min_max_elements(&self) -> bool {
        self.el_count(-1) == 1 && self.el_count(self.rank()) == 1
    }

    /// Checks whether all of the subelements refer to valid elements in the
    /// polytope. If this returns `false`, then either the polytope hasn't been
    /// fully built up, or there's something seriously wrong.
    fn check_incidences(&self) -> bool {
        for r in -1..self.rank() {
            for element in self[r].iter() {
                for &sub in &element.subs {
                    if self[r - 1].get(sub).is_none() {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Determines whether the polytope satisfies the diamond property. A valid
    /// non-fissary polytope should always return `true`.
    fn is_dyadic(&self) -> bool {
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

                for &sub in &el.subs {
                    let sub_el = &self[r - 1][sub];

                    for &sub_sub in &sub_el.subs {
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
    fn is_connected(&self) -> bool {
        todo!()
    }

    /// Determines whether the polytope is strongly connected. A valid
    /// non-compound polytope should always return `true`.
    fn is_strongly_connected(&self) -> bool {
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
    fn product(p: &Self, q: &Self, min: bool, max: bool) -> Self {
        let p_rank = p.rank();
        let q_rank = q.rank();

        let p_low = -(min as isize);
        let p_hi = p_rank - (!max as isize);

        let q_low = -(min as isize);
        let q_hi = q_rank - (!max as isize);

        let rank = p_rank + q_rank + 1 - (!min as isize) - (!max as isize);

        // Initializes the product with empty element lists.
        let mut product = Self::with_rank(rank);
        for _ in -1..=rank {
            product.push(ElementList::new());
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
                            for &s in &p_el.subs {
                                subs.push(get_element_index(p_els_rank - 1, s, q_els_rank, q_idx))
                            }
                        }

                        // Products of q's subelements with p.
                        if q_els_rank != 0 || min {
                            for &s in &q_el.subs {
                                subs.push(get_element_index(p_els_rank, p_idx, q_els_rank - 1, s))
                            }
                        }

                        product[prod_rank].push(Element { subs })
                    }
                }
            }
        }

        // If !min, we have to set a minimal element manually.
        if !min {
            product[-1] = ElementList::min();
            product[0] = ElementList::vertices(p.el_count(0) * q.el_count(0));
        }

        // If !max, we have to set a maximal element manually.
        if !max {
            product[rank] = ElementList::max(product.el_count(rank - 1));
        }

        product
    }
}

impl Polytope for Abstract {
    /// Returns the rank of the polytope.
    fn rank(&self) -> isize {
        self.0.len() as isize - 2
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
        let mut counts = RankVec::with_rank(self.rank());

        for r in -1..=self.rank() {
            counts.push(self[r].len())
        }

        counts
    }

    /// Returns the unique polytope of rank −1.
    fn nullitope() -> Self {
        Abstract::from_vec(vec![ElementList::min()])
    }

    /// Returns the unique polytope of rank 0.
    fn point() -> Self {
        Abstract::from_vec(vec![ElementList::min(), ElementList::max(1)])
    }

    /// Returns the unique polytope of rank 1.
    fn dyad() -> Self {
        Abstract::from_vec(vec![
            ElementList::min(),
            ElementList::vertices(2),
            ElementList::max(2),
        ])
    }

    /// Returns the unique polytope of rank 2 with a given amount of vertices.
    fn polygon(n: usize) -> Self {
        assert!(n >= 2);

        let nullitope = ElementList::min();
        let mut vertices = ElementList::with_capacity(n);
        let mut edges = ElementList::with_capacity(n);
        let maximal = ElementList::max(n);

        for i in 1..=n {
            vertices.push(Element { subs: vec![0] });

            edges.push(Element {
                subs: vec![i % n, (i + 1) % n],
            });
        }

        Abstract::from_vec(vec![nullitope, vertices, edges, maximal])
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

        self[rank].push(max);
        self.push(ElementList::max(2));
    }

    fn horotope(&self) -> Self {
        let mut clone = self.clone();
        clone.horotope_mut();
        clone
    }

    fn horotope_mut(&mut self) {
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

impl Deref for Abstract {
    type Target = RankVec<ElementList>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Abstract {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
/// Represents a
/// [concrete polytope](https://polytope.miraheze.org/wiki/Polytope), which is
/// an [`Abstract`] together with the corresponding vertices.
pub struct Concrete {
    /// The list of vertices as points in Euclidean space.
    pub vertices: Vec<Point>,

    /// The underlying abstract polytope.
    pub abs: Abstract,
}

impl Concrete {
    pub fn new(vertices: Vec<Point>, abs: Abstract) -> Self {
        // There must be as many abstract vertices as concrete ones.
        debug_assert_eq!(vertices.len(), abs.el_count(0));

        if let Some(vertex0) = vertices.get(0) {
            for vertex1 in &vertices {
                debug_assert_eq!(vertex0.len(), vertex1.len());
            }
        }

        Self { vertices, abs }
    }

    /// Returns the rank of the polytope.
    pub fn rank(&self) -> isize {
        self.abs.rank()
    }

    /// Returns the number of dimensions of the space the polytope lives in,
    /// or `None` in the case of the nullitope.
    pub fn dimension(&self) -> Option<usize> {
        Some(self.vertices.get(0)?.len())
    }

    /// Scales a polytope by a given factor.
    pub fn scale(mut self, k: f64) -> Self {
        for v in &mut self.vertices {
            *v *= k;
        }

        self
    }

    /// Shifts all vertices by a given vector.
    pub fn shift(mut self, o: Point) -> Self {
        for v in &mut self.vertices {
            *v -= &o;
        }

        self
    }

    /// Recenters a polytope so that the gravicenter is at the origin.
    pub fn recenter(self) -> Self {
        if let Some(gravicenter) = self.gravicenter() {
            self.shift(gravicenter)
        } else {
            self
        }
    }

    /// Applies a matrix to all vertices of a polytope.
    pub fn apply(mut self, m: &Matrix) -> Self {
        for v in &mut self.vertices {
            *v = m * v.clone();
        }

        self
    }

    /// Calculates the circumsphere of a polytope. Returns it if the polytope
    /// has one, and returns `None` otherwise.
    pub fn circumsphere(&self) -> Option<Hypersphere> {
        let mut vertices = self.vertices.iter();
        const EPS: f64 = 1e-9;

        let v0 = vertices.next().expect("Polytope has no vertices!").clone();
        let mut o: Point = v0.clone();
        let mut h = Hyperplane::new(v0.clone());

        for v in vertices {
            // If the new vertex does not lie on the hyperplane of the others:
            if let Some(b) = h.add(v.clone()) {
                // Calculates the new circumcenter.
                let k = ((&o - v).norm_squared() - (&o - &v0).norm_squared())
                    / (2.0 * (v - &v0).dot(&b));

                o += k * b;
            }
            // If the new vertex lies on the others' hyperplane, but is not at
            // the correct distance from the first vertex:
            else if ((&o - &v0).norm() - (&o - v).norm()).abs() > EPS {
                return None;
            }
        }

        Some(Hypersphere {
            radius: (&o - v0).norm(),
            center: o,
        })
    }

    /// Gets the gravicenter of a polytope, or `None` in the case of the
    /// nullitope.
    pub fn gravicenter(&self) -> Option<Point> {
        let mut g: Point = vec![0.0; self.dimension()? as usize].into();

        for v in &self.vertices {
            g += v;
        }

        Some(g / (self.vertices.len() as f64))
    }

    /// Gets the edge lengths of all edges in the polytope, in order.
    pub fn edge_lengths(&self) -> Vec<f64> {
        let mut edge_lengths = Vec::new();

        // If there are no edges, we just return the empty vector.
        if let Some(edges) = self.abs.get(1) {
            edge_lengths.reserve_exact(edges.len());

            for edge in edges.iter() {
                let sub0 = edge.subs[0];
                let sub1 = edge.subs[1];

                edge_lengths.push((&self.vertices[sub0] - &self.vertices[sub1]).norm());
            }
        }

        edge_lengths
    }

    pub fn is_equilateral_with_len(&self, len: f64) -> bool {
        const EPS: f64 = 1e-9;
        let edge_lengths = self.edge_lengths().into_iter();

        // Checks that every other edge length is equal to the first.
        for edge_len in edge_lengths {
            if (edge_len - len).abs() > EPS {
                return false;
            }
        }

        true
    }

    /// Checks whether a polytope is equilateral to a fixed precision.
    pub fn is_equilateral(&self) -> bool {
        if let Some(edges) = self.abs.get(1) {
            if let Some(edge) = edges.get(0) {
                let vertices = edge
                    .subs
                    .iter()
                    .map(|&v| &self.vertices[v])
                    .collect::<Vec<_>>();
                let (v0, v1) = (vertices[0], vertices[1]);

                return self.is_equilateral_with_len((v0 - v1).norm());
            }
        }

        true
    }

    /// I haven't actually implemented this in the general case.
    pub fn midradius(&self) -> f64 {
        let vertices = &self.vertices;
        let edges = &self[0];
        let edge = &edges[0];

        let sub0 = edge.subs[0];
        let sub1 = edge.subs[1];

        (&vertices[sub0] + &vertices[sub1]).norm() / 2.0
    }

    /// Returns the dual of a polytope, or `None` if any facets pass through the
    /// origin.
    pub fn dual(&self) -> Option<Self> {
        let mut clone = self.clone();
        clone.dual_mut()?;
        Some(clone)
    }

    /// Builds the dual of a polytope in place, or does nothing in case any
    /// facets go through the origin. Returns the dual if successful, and `None`
    /// otherwise.
    pub fn dual_mut(&mut self) -> Option<&mut Self> {
        self.dual_mut_with_sphere(&Hypersphere::unit(self.dimension().unwrap_or(1)))
    }

    /// Returns the dual of a polytope with a given reciprocation sphere, or
    /// `None` if any facets pass through the reciprocation center.
    pub fn dual_with_sphere(&self, sphere: &Hypersphere) -> Option<Self> {
        let mut clone = self.clone();
        clone.dual_mut_with_sphere(sphere)?;
        Some(clone)
    }

    /// Builds the dual of a polytope with a given reciprocation sphere in
    /// place, or does nothing in case any facets go through the reciprocation
    /// center. Returns the dual if successful, and `None` otherwise.
    pub fn dual_mut_with_sphere(&mut self, sphere: &Hypersphere) -> Option<&mut Self> {
        const EPS: f64 = 1e-9;

        // If we're dealing with a nullitope or point, the dual is itself.
        //
        // TODO: maybe also reciprocate the point geometrically?
        let rank = self.rank();
        if rank < 1 {
            return Some(self);
        }

        // We project the sphere's center onto the polytope's hyperplane to
        // avoid skew weirdness.
        let h = Hyperplane::from_points(self.vertices.clone());
        let o = h.project(&sphere.center);

        let mut projections;

        // We project our inversion center onto each of the facets.
        if rank >= 2 {
            let facet_count = self.el_count(rank - 1);
            projections = Vec::with_capacity(facet_count);

            for idx in 0..facet_count {
                projections.push(
                    Hyperplane::from_points(self.get_element_vertices(rank - 1, idx).unwrap())
                        .project(&o),
                );
            }
        }
        // If our polytope is 1D, the vertices themselves are the facets.
        else {
            projections = self.vertices.clone();
        }

        // Reciprocates the projected points.
        for v in projections.iter_mut() {
            *v -= &o;
            let s = v.norm_squared();

            // If any face passes through the dual center, the dual does
            // not exist, and we return early.
            if s < EPS {
                return None;
            }

            *v /= s;
            *v += &o;
        }

        self.vertices = projections;

        // Takes the abstract dual.
        self.abs.dual_mut();

        Some(self)
    }

    /// Gets the (geometric) vertices of an element on the polytope.
    pub fn get_element_vertices(&self, rank: isize, idx: usize) -> Option<Vec<Point>> {
        Some(
            self.abs
                .get_element_vertices(rank, idx)?
                .iter()
                .map(|&v| self.vertices[v].clone())
                .collect(),
        )
    }

    /// Gets an element of a polytope, as its own polytope.
    pub fn get_element(&self, rank: isize, idx: usize) -> Option<Self> {
        Some(Concrete {
            vertices: self.get_element_vertices(rank, idx)?,
            abs: self.abs.get_element(rank, idx)?,
        })
    }

    /// Gets the [vertex figure](https://polytope.miraheze.org/wiki/Vertex_figure)
    /// of a polytope corresponding to a given vertex.
    pub fn verf(&self, idx: usize) -> Option<Self> {
        self.dual()?.get_element(self.rank() - 1, idx)?.dual()
    }

    /// Generates the vertices for either a tegum or a pyramid product with two
    /// given vertex sets and a given height.
    fn duopyramid_vertices(p: &[Point], q: &[Point], height: f64, tegum: bool) -> Vec<Point> {
        let p_dimension = p[0].len();
        let q_dimension = q[0].len();

        let dimension = p_dimension + q_dimension + tegum as usize;

        let mut vertices = Vec::with_capacity(p.len() + q.len());

        // The vertices corresponding to products of p's nullitope with q's
        // vertices.
        for q_vertex in q {
            let mut prod_vertex = Vec::with_capacity(dimension);
            let pad = p_dimension;

            // Pads prod_vertex to the left.
            prod_vertex.resize(pad, 0.0);

            // Copies q_vertex into prod_vertex.
            for &c in q_vertex.iter() {
                prod_vertex.push(c);
            }

            // Adds the height, in case of a pyramid product.
            if !tegum {
                prod_vertex.push(height / 2.0);
            }

            vertices.push(prod_vertex.into());
        }

        // The vertices corresponding to products of q's nullitope with p's
        // vertices.
        for p_vertex in p {
            let mut prod_vertex = Vec::with_capacity(dimension);

            // Copies p_vertex into prod_vertex.
            for &c in p_vertex.iter() {
                prod_vertex.push(c);
            }

            // Pads prod_vertex to the right.
            prod_vertex.resize(p_dimension + q_dimension, 0.0);

            // Adds the height, in case of a pyramid product.
            if !tegum {
                prod_vertex.push(-height / 2.0);
            }

            vertices.push(prod_vertex.into());
        }

        vertices
    }

    /// Generates the vertices for a duoprism with two given vertex sets.
    fn duoprism_vertices(p: &[Point], q: &[Point]) -> Vec<Point> {
        let mut vertices = Vec::with_capacity(p.len() * q.len());

        // Concatenates all pairs of vertices in order.
        for p_vertex in p {
            for q_vertex in q {
                let p_vertex = p_vertex.into_iter();
                let q_vertex = q_vertex.into_iter();

                vertices.push(p_vertex.chain(q_vertex).cloned().collect::<Vec<_>>().into());
            }
        }

        vertices
    }

    fn duopyramid_with_height(p: &Self, q: &Self, height: f64) -> Self {
        Self::new(
            Self::duopyramid_vertices(&p.vertices, &q.vertices, height, false),
            Abstract::duopyramid(&p.abs, &q.abs),
        )
    }
}

impl Polytope for Concrete {
    fn rank(&self) -> isize {
        self.abs.rank()
    }

    fn el_count(&self, rank: isize) -> usize {
        self.abs.el_count(rank)
    }

    fn el_counts(&self) -> RankVec<usize> {
        self.abs.el_counts()
    }

    fn nullitope() -> Self {
        Self {
            abs: Abstract::nullitope(),
            vertices: Vec::new(),
        }
    }

    fn point() -> Self {
        Self::new(vec![vec![].into()], Abstract::point())
    }

    fn dyad() -> Self {
        Self::new(vec![vec![-0.5].into(), vec![0.5].into()], Abstract::dyad())
    }

    fn polygon(n: usize) -> Self {
        Self::reg_polygon(n, 1)
    }

    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::duopyramid_with_height(p, q, 1.0)
    }

    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::new(
            Self::duoprism_vertices(&p.vertices, &q.vertices),
            Abstract::duoprism(&p.abs, &q.abs),
        )
    }

    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::new(
            Self::duopyramid_vertices(&p.vertices, &q.vertices, 0.0, true),
            Abstract::duotegum(&p.abs, &q.abs),
        )
    }

    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::new(
            Self::duoprism_vertices(&p.vertices, &q.vertices),
            Abstract::duocomb(&p.abs, &q.abs),
        )
    }

    fn ditope(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            abs: self.abs.ditope(),
        }
    }

    fn ditope_mut(&mut self) {
        self.abs.ditope_mut();
    }

    fn horotope(&self) -> Self {
        Self {
            vertices: vec![vec![-0.5].into(), vec![0.5].into()],
            abs: self.abs.horotope(),
        }
    }

    fn horotope_mut(&mut self) {
        self.vertices = vec![vec![-0.5].into(), vec![0.5].into()];
        self.abs.horotope_mut();
    }

    fn antiprism(&self) -> Self {
        todo!()
    }

    fn simplex(rank: isize) -> Self {
        if rank == -1 {
            Self::nullitope()
        } else {
            let dim = rank as usize;
            let mut vertices = Vec::with_capacity(dim + 1);

            // Adds all points with a single entry equal to √2/2, and all others
            // equal to 0.
            for i in 0..dim {
                let mut v = vec![0.0; dim];
                v[i] = SQRT_2 / 2.0;
                vertices.push(v.into());
            }

            // Adds the remaining vertex, all of whose coordinates are equal.
            let a = (1.0 - ((dim + 1) as f64).sqrt()) * SQRT_2 / (2.0 * dim as f64);
            vertices.push(vec![a; dim].into());

            Concrete {
                vertices,
                abs: Abstract::simplex(rank),
            }
            .recenter()
        }
    }

    fn orientable(&self) -> bool {
        self.abs.orientable()
    }
}

impl Index<isize> for Concrete {
    type Output = ElementList;

    /// Gets the list of elements with a given rank.
    fn index(&self, rank: isize) -> &Self::Output {
        &self.abs[rank]
    }
}

impl IndexMut<isize> for Concrete {
    /// Gets the list of elements with a given rank.
    fn index_mut(&mut self, rank: isize) -> &mut Self::Output {
        &mut self.abs[rank]
    }
}

/// Represents a [`Concrete`], together with a triangulation used to
/// render it.
#[derive(Debug, Clone)]
pub struct Renderable {
    /// The underlying concrete polytope.
    pub concrete: Concrete,

    /// Extra vertices that might be needed for the triangulation.
    extra_vertices: Vec<Point>,

    /// Indices of the vertices that make up the triangles.
    triangles: Vec<[usize; 3]>,
}

impl Renderable {
    /// Generates the triangulation of a `Concrete`.
    pub fn new(concrete: Concrete) -> Self {
        // let vertices = &concrete.vertices;
        let edges = concrete.abs.get(1).unwrap();
        let faces = concrete.abs.get(2).unwrap();

        let extra_vertices = Vec::new();
        let mut triangles = Vec::new();

        for face in faces.iter() {
            let edge_i = *face.subs.first().expect("no indices in face");
            let vert_i = edges
                .get(edge_i)
                .expect("Index out of bounds: you probably screwed up the polytope's indices.")
                .subs[0];

            for verts in face.subs[1..].iter().map(|&i| {
                let edge = &edges[i];
                assert_eq!(edge.subs.len(), 2, "edges has more than two subelements");
                [edge.subs[0], edge.subs[1]]
            }) {
                let [vert_j, vert_k]: [usize; 2] = verts;
                if vert_i != vert_j && vert_i != vert_k {
                    triangles.push([vert_i, vert_j, vert_k]);
                }
            }
        }

        Renderable {
            concrete,
            extra_vertices,
            triangles,
        }
    }

    /// Gets the coordinates of the vertices, after projecting down into 3D.
    fn get_vertex_coords(&self) -> Vec<[f32; 3]> {
        self.concrete
            .vertices
            .iter()
            .chain(self.extra_vertices.iter())
            .map(|point| {
                // For now, we do a simple orthogonal projection.
                let mut iter = point.iter().copied().take(3);
                let x = iter.next().unwrap_or(0.0);
                let y = iter.next().unwrap_or(0.0);
                let z = iter.next().unwrap_or(0.0);
                [x as f32, y as f32, z as f32]
            })
            .collect()
    }

    /// Generates a mesh from the polytope.
    pub fn get_mesh(&self) -> Mesh {
        let vertices = self.get_vertex_coords();
        let mut indices = Vec::with_capacity(self.triangles.len() * 3);
        for &[i, j, k] in &self.triangles {
            indices.push(i as u16);
            indices.push(j as u16);
            indices.push(k as u16);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.set_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vec![[0.0, 1.0, 0.0]; vertices.len()],
        );
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertices.len()]);
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.set_indices(Some(Indices::U16(indices)));

        mesh
    }

    /// Generates the wireframe for a polytope.
    pub fn get_wireframe(&self) -> Mesh {
        let edges = self.concrete.abs.get(1).unwrap();
        let vertices = self.get_vertex_coords();
        let mut indices = Vec::with_capacity(edges.len() * 2);
        for edge in edges.iter() {
            indices.push(edge.subs[0] as u16);
            indices.push(edge.subs[1] as u16);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::LineList);
        mesh.set_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vec![[0.0, 1.0, 0.0]; vertices.len()],
        );
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertices.len()]);
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.set_indices(Some(Indices::U16(indices)));

        mesh
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
            Abstract::hypercube(3),
            Abstract::hypercube(4),
            Abstract::hypercube(5),
            Abstract::simplex(3),
            Abstract::simplex(4),
            Abstract::simplex(5),
            Abstract::orthoplex(3),
            Abstract::orthoplex(4),
            Abstract::orthoplex(5),
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
                    "{}-{} duopyramid are invalid.",
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
        for n in -1..=5 {
            let simplex = Abstract::simplex(n);

            for k in -1..=n {
                assert_eq!(
                    simplex.el_count(k),
                    choose((n + 1) as usize, (k + 1) as usize),
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
        for n in -1..=5 {
            let hypercube = Abstract::hypercube(n);

            for k in 0..=n {
                assert_eq!(
                    hypercube.el_count(k),
                    choose(n as usize, k as usize) * 2u32.pow((n - k) as u32) as usize,
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
        for n in -1..=5 {
            let orthoplex = Abstract::orthoplex(n);

            for k in -1..n {
                assert_eq!(
                    orthoplex.el_count(k),
                    choose(n as usize, (k + 1) as usize) * 2u32.pow((k + 1) as u32) as usize,
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
