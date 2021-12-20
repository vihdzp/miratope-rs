use std::collections::HashMap;

use crate::element_vec::ElementVec;

/// An element of a [`RankedPoset`].
#[derive(Clone, Default)]
pub struct Element {
    /// The indices of the subelements.
    pub sub: ElementVec,

    /// The indices of the superelements.
    pub sup: ElementVec,
}

impl Element {
    /// Initializes a new empty element.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initializes the minimal element of a polytope with `n` vertices.
    pub fn min(n: u32) -> Self {
        Self {
            sub: ElementVec::new(),
            sup: ElementVec::count_lt(n),
        }
    }

    /// Initializes a vertex of a polytope.
    pub fn vertex(sup: ElementVec) -> Self {
        Self {
            sub: ElementVec::single(0),
            sup,
        }
    }

    /// Initializes a facet of a polytope.
    pub fn facet(sub: ElementVec) -> Self {
        Self {
            sub,
            sup: ElementVec::single(0),
        }
    }

    /// Initializes the maximal element of a polytope with `n` facets.
    pub fn max(n: u32) -> Self {
        Self {
            sub: ElementVec::count_lt(n),
            sup: ElementVec::new(),
        }
    }

    /// Swaps the subelements and superelements.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.sub, &mut self.sup)
    }
}

/// Stores a ranked partial order. It is represented similarly to a Hasse
/// diagram, where every element is linked to various subelements and various
/// superelements.
///
/// We say that a `RankedPoset` is "well-formed" when the following conditions
/// hold:
///
/// 1. Every element points to valid subelements and superelements.
/// 2. Every element but those with least or greatest rank has at least one
/// subelement and one superelement.
///
/// Every function that takes in a `RankedPoset` should be able to assume that
/// it's well-formed, and return a well-formed output. However, this invariant
/// won't be strictly enforced for reasons of code clarity. Please be careful!
#[derive(Clone, Default)]
pub struct RankedPoset(pub Vec<Vec<Element>>);

impl std::ops::Index<usize> for RankedPoset {
    type Output = Vec<Element>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for RankedPoset {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl RankedPoset {
    /// Returns an empty ranked poset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the height of the partial order, which is the length of the
    /// backing vector.
    pub fn height(&self) -> usize {
        self.0.len()
    }

    /// Builds the dual partial order, where the order relation is flipped.
    pub fn dual(&mut self) {
        // To-do: this is trivially parallelizable.
        for rank in &mut self.0 {
            for el in rank {
                el.swap();
            }
        }

        self.0.reverse();
    }

    /// Returns the number of minimal elements.
    pub fn min_count(&self) -> usize {
        match self.0.first() {
            Some(h) => h.len(),
            None => 0,
        }
    }

    /// Returns the number of maximal elements.
    pub fn max_count(&self) -> usize {
        match self.0.last() {
            Some(h) => h.len(),
            None => 0,
        }
    }

    /// Returns whether the ranked poset is bounded.
    pub fn is_bounded(&self) -> bool {
        self.min_count() == 1 && self.max_count() == 1
    }

    /// Pushes a minimal element.
    pub fn push_min(&mut self) {
        let min_count = self.min_count();
        if min_count != 0 {
            for el in &mut self[0] {
                el.sub = ElementVec::single(0);
            }
        }
        self.0.insert(0, vec![Element::min(min_count as u32)]);
    }

    /// Pushes a maximal element.
    pub fn push_max(&mut self) {
        let max_count = self.max_count();
        if max_count != 0 {
            for el in self.0.last_mut().unwrap() {
                el.sup = ElementVec::single(0);
            }
        }
        self.0.push(vec![Element::min(max_count as u32)]);
    }

    /// Adds a new row of elements, whose subelements are given by the passed
    /// `ElementVec`s.
    pub fn push_subs(&mut self, subs: Vec<ElementVec>) {
        let mut new_rank = Vec::new();

        // For every new element:
        for (new_idx, sub) in subs.into_iter().enumerate() {
            // Updates the superelements of its subelements.
            for &idx in sub.vec() {
                self.0.last_mut().unwrap()[idx as usize]
                    .sup
                    .push(new_idx as u32);
            }

            // Adds the new element.
            new_rank.push(Element {
                sub,
                sup: ElementVec::new(),
            });
        }

        self.0.push(new_rank);
    }

    /// Takes the direct product of two contiguous sets of ranks coming from a
    /// ranked poset. The minimal and maximal elements of the least and greatest
    /// ranks are ignored.
    ///
    /// Any empty ranked poset has an empty direct product with any other ranked
    /// poset.
    ///
    /// The elements of each rank are built in lexicographic order of the ranks,
    /// and then in lexicographic order of the indices.
    pub(super) fn _product(_self: &[Vec<Element>], other: &[Vec<Element>]) -> Self {
        if _self.is_empty() || other.is_empty() {
            return Self::new();
        }

        let h1 = _self.len() - 1;
        let h2 = other.len() - 1;

        // Memoizes the index of the first element created from a pair of ranks.
        let mut memo = HashMap::new();

        let mut poset = RankedPoset::new();
        for rank in 0..=(h1 + h2) {
            let mut subs = Vec::new();

            // For each pair of ranks in lexicographic order.
            for r1 in 0..=rank.min(h1) {
                let r2 = rank - r1;
                memo.insert((r1, r2), subs.len() as u32);

                // For each pair of elements of these ranks in lexicographic order.
                for (i, el1) in _self[r1].iter().enumerate() {
                    for (j, el2) in other[r2].iter().enumerate() {
                        let mut new = ElementVec::new();

                        if r1 != 0 {
                            let offset = memo[&(r1 - 1, r2)];
                            let len = _self[r1 - 1].len() as u32;

                            for &s in &el1.sub {
                                new.push(offset + s * len + j as u32);
                            }
                        }

                        if r2 != 0 {
                            let offset = memo[&(r1, r2 - 1)];
                            let len = _self[r1].len() as u32;

                            for &s in &el2.sub {
                                new.push(offset + i as u32 * len + s);
                            }
                        }

                        subs.push(new);
                    }
                }
            }

            poset.push_subs(subs);
        }

        poset
    }

    /// Builds the [direct product](https://en.wikipedia.org/wiki/Direct_product#Direct_product_of_binary_relations)
    /// of two ranked posets.
    ///
    /// Any empty ranked poset has an empty direct product with any other ranked
    /// poset.
    ///
    /// The elements of each rank are built in lexicographic order of the ranks,
    /// and then in lexicographic order of the indices.
    pub fn product(&self, other: &Self) -> Self {
        Self::_product(&self.0, &other.0)
    }
}
