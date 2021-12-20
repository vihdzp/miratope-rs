pub mod ranked_poset;

use crate::element_vec::ElementVec;
use ranked_poset::{Element, RankedPoset};

/// An abstract prepolytope. This is made out of a [`RankedPoset`].
///
/// The ranked poset must satisfy the two other properties required by a
/// prepolytope:
///
/// 1. It must be **bounded** – it has unique minimal and maximal elements.
/// In particular, it must be non-empty.
/// 2. It must satisfy the **diamond property** – every section of height 3
/// must have exactly four elements.
pub struct Abstract(RankedPoset);

impl Abstract {
    /// Returns a reference to the underlying poset.
    pub fn poset(&self) -> &RankedPoset {
        &self.0
    }

    /// Returns a mutable reference to the underlying poset.
    ///
    /// # Safety
    /// This method is marked as unsafe as it allows one to violate the abstract
    /// polytope invariants.
    pub unsafe fn poset_mut(&mut self) -> &mut RankedPoset {
        &mut self.0
    }

    /// Returns the underlying poset.
    pub fn into_poset(self) -> RankedPoset {
        self.0
    }

    /// Returns the height of the partial order, which is the length of the
    /// backing vector. This is equal to the usual "rank" of prepolytopes plus
    /// two.
    ///
    /// This is guaranteed to be at least 1.
    pub fn height(&self) -> usize {
        self.0.height()
    }

    /// Creates an `Abstract` from a ranked poset.
    ///
    /// # Safety
    /// The ranked poset must satisfy the two other properties required by a
    /// prepolytope.
    pub unsafe fn from_poset_unchecked(poset: RankedPoset) -> Self {
        Self(poset)
    }

    /// Returns an instance of the [nullitope](https://polytope.miraheze.org/wiki/Nullitope),
    /// the unique polytope of height 1.
    pub fn nullitope() -> Self {
        Self(RankedPoset(vec![vec![Element::new()]]))
    }

    /// Returns an instance of the [point](https://polytope.miraheze.org/wiki/Point),
    /// the unique polytope of height 2.
    pub fn point() -> Self {
        Self(RankedPoset(vec![
            vec![Element::min(1)],
            vec![Element::max(1)],
        ]))
    }

    /// Returns an instance of the [dyad](https://polytope.miraheze.org/wiki/Dyad),
    /// the unique polytope of height 3.
    pub fn dyad() -> Self {
        // A vertex of the dyad.
        let vertex = Element::vertex(ElementVec::single(0));

        Self(RankedPoset(vec![
            vec![Element::min(2)],
            vec![vertex.clone(), vertex],
            vec![Element::max(2)],
        ]))
    }

    /// Returns an instance of an [*n*-gon](https://polytope.miraheze.org/wiki/Polygon),
    /// the unique polytope of height 4 with *n* vertices.
    pub fn polygon(n: u32) -> Self {
        // A vertex of the dyad.
        let mut vertices = Vec::new();
        let mut edges = Vec::new();

        // Safety: all of these lists are sorted.
        let vec = unsafe { ElementVec::from_sorted(vec![0, n - 1]) };
        edges.push(Element::facet(vec.clone()));

        for i in 1..n {
            let vec = unsafe { ElementVec::from_sorted(vec![i - 1, i]) };
            vertices.push(Element::vertex(vec.clone()));
            edges.push(Element::facet(vec));
        }

        vertices.push(Element::vertex(vec));

        Self(RankedPoset(vec![
            vec![Element::min(n)],
            vertices,
            edges,
            vec![Element::max(n)],
        ]))
    }

    /// Makes a compound out of two prepolytopes.
    ///
    /// # Panics
    /// This method will panic if called on prepolytopes of different rank,
    /// or prepolytopes with height less or equal to 3.
    pub fn compound(&mut self, other: &Self) {
        let height = self.height();
        assert!(
            height > 3,
            "compounds may only be taken for polygons or higher"
        );
        assert_eq!(
            height,
            other.height(),
            "compounds may only be taken for polytopes of the same rank"
        );

        let poset = unsafe { self.poset_mut() };

        let mut offset = Vec::with_capacity(height);
        offset.push(0);
        for i in 1..(height - 1) {
            offset.push(poset[i].len() as u32);
        }
        offset.push(0);

        for i in 1..(height - 1) {
            for el in &other.poset()[i] {
                let mut sub = Vec::new();
                for &j in &el.sub {
                    sub.push(offset[i - 1] + j);
                }

                let mut sup = Vec::new();
                for &j in &el.sup {
                    sup.push(offset[i + 1] + j);
                }

                unsafe {
                    poset[i].push(Element {
                        sub: ElementVec::from_raw_parts(sub, el.sub.sorted()),
                        sup: ElementVec::from_raw_parts(sup, el.sup.sorted()),
                    })
                }
            }
        }
    }

    /// Returns the [join](https://polytope.miraheze.org/wiki/Join) of two
    /// prepolytopes.
    pub fn join(&self, other: &Self) -> Self {
        unsafe { Self::from_poset_unchecked(self.0.product(&other.0)) }
    }

    /// Returns the [Cartesian product](https://polytope.miraheze.org/wiki/Cartesian_product)
    /// of two prepolytopes.
    pub fn cartesian_product(&self, other: &Self) -> Self {
        let mut poset = RankedPoset::_product(&self.0 .0[1..], &other.0 .0[1..]);
        poset.push_min();
        unsafe { Self::from_poset_unchecked(poset) }
    }

    /// Returns the [free join](https://polytope.miraheze.org/wiki/Free_join) of
    /// two prepolytopes.
    pub fn free_join(&self, other: &Self) -> Self {
        let mut poset = RankedPoset::_product(
            &self.0 .0[..self.height() - 1],
            &other.0 .0[..other.height() - 1],
        );
        poset.push_max();
        unsafe { Self::from_poset_unchecked(poset) }
    }

    /// Returns the [topological product](https://polytope.miraheze.org/wiki/Topological_product)
    /// of two prepolytopes.
    pub fn topological_product(&self, other: &Self) -> Self {
        let mut poset = RankedPoset::_product(
            &self.0 .0[1..self.height() - 1],
            &other.0 .0[1..other.height() - 1],
        );
        poset.push_min();
        poset.push_max();
        unsafe { Self::from_poset_unchecked(poset) }
    }
}
