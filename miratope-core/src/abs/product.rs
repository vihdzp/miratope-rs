//! Contains the code for the polytope products.

use super::*;

/// When we compute any polytope product, we add the elements of any given rank
/// in lexicographic order of the ranks of the elements they come from. This
/// struct memoizes how many elements of the same rank are added by the time we
/// add those of the form `(p_rank, q_rank)`. It stores this value in
/// `offset_memo[(p_rank, q_rank)]`.
struct OffsetMemo<const MIN: bool, const MAX: bool> {
    /// The memoized values. We store them in a single `Vec` to avoid multiple
    /// allocations.
    memo: Vec<usize>,

    /// The rank of the polytope `q` used to build this struct.
    q_rank: usize,
}

impl<const MIN: bool, const MAX: bool> OffsetMemo<MIN, MAX> {
    /// Equals `0` if we're considering the minimal element in the product, `1`
    /// otherwise.
    const MIN_U: usize = MIN as usize;

    /// Equals `0` if we're considering the maximal element in the product, `1`
    /// otherwise.
    const MAX_U: usize = MAX as usize;

    /// Returns the rank of the highest element of a polytope that we consider
    /// for the product. This is the rank if the maximal element is considered,
    /// and one less otherwise.
    fn hi(p: &Abstract) -> usize {
        p.rank() - Self::MAX_U
    }

    /// Returns the number of ranks of a polytope we consider for the product.
    fn range_len(p: &Abstract) -> usize {
        Self::hi(p) + 1 - Self::MIN_U
    }

    /// Returns the number of ranks of `q` we consider for the product.
    fn q_range(&self) -> usize {
        self.q_rank + 1 - Self::MAX_U - Self::MIN_U
    }

    /// Initializes a new offset memoizator.
    fn new(p: &Abstract, q: &Abstract) -> Self {
        let memo = Vec::with_capacity(Self::range_len(p) * Self::range_len(q));
        let q_rank = q.rank();
        let mut res = Self { memo, q_rank };
        res.fill_memo(p, q);
        res
    }

    /// Calculates and stores the required values.
    fn fill_memo(&mut self, p: &Abstract, q: &Abstract) {
        let prod_count = |i, j| p.el_count(i) * q.el_count(j);

        // The highest ranks we consider for the product.
        let p_hi = Self::hi(p);
        let q_hi = Self::hi(q);

        for q_el_rank in Self::MIN_U..=q_hi {
            self.memo.push(prod_count(Self::MIN_U, q_el_rank));
        }

        for p_el_rank in (Self::MIN_U + 1)..=p_hi {
            for q_el_rank in Self::MIN_U..q_hi {
                self.memo
                    .push(self[(p_el_rank - 1, q_el_rank + 1)] + prod_count(p_el_rank, q_el_rank));
            }

            self.memo.push(prod_count(p_el_rank, q_hi));
        }
    }

    /// Every element of the product is in correspondence with
    /// a pair of an element from `p` and an element from `q`. This function
    /// finds the position we placed it in.
    fn get_element_index(
        &self,
        p_rank: usize,
        p_idx: usize,
        q: &Abstract,
        q_rank: usize,
        q_idx: usize,
    ) -> usize {
        let idx = p_idx * q.el_count(q_rank) + q_idx;

        if p_rank == Self::MIN_U || q_rank == Self::hi(q) {
            idx
        } else {
            self[(p_rank - 1, q_rank + 1)] + idx
        }
    }
}

impl<const MIN: bool, const MAX: bool> Index<(usize, usize)> for OffsetMemo<MIN, MAX> {
    type Output = usize;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.memo[(index.0 - Self::MIN_U) * self.q_range() + (index.1 - Self::MIN_U)]
    }
}

/// Takes the [direct product](https://en.wikipedia.org/wiki/Direct_product#Direct_product_of_binary_relations)
/// of two polytopes. If the `MIN` flag is turned on, it ignores the
/// minimal elements of both of the factors and adds one at the end. The
/// `MAX` flag works analogously.
///
/// This method takes in `MIN` and `MAX` as type parameters so that each
/// case may be separately optimized. We should probably run some tests to
/// see if this is actually any better, though.
///
/// The elements of this product are in one to one correspondence to pairs
/// of elements in the set of polytopes. The elements of a specific rank are
/// sorted first by lexicographic order of the ranks, then by lexicographic
/// order of the indices of the elements.
fn product<const MIN: bool, const MAX: bool>(p: &Abstract, q: &Abstract) -> Abstract {
    // The ranks of p and q.
    let p_rank = p.rank();
    let q_rank = q.rank();

    // For anything but a pyramid product, we'll just special-case the product
    // with a nullitope as the nullitope.
    if (MIN || MAX) && (p_rank == 0 || q_rank == 0) {
        return Abstract::nullitope();
    }

    // 0 or 1 depending on whether the minimum/maximum elements are in the
    // polytope.
    let min_u = MIN as usize;
    let max_u = MAX as usize;

    // The highest ranks we'll use to take products in p and q.
    // TODO: THIS CURRENTLY FAILS IN THE NULLITOPE
    let p_hi = p_rank - max_u;
    let q_hi = q_rank - max_u;

    // The rank of the product.
    let rank = p_rank + q_rank - min_u - max_u;

    // Initializes the element lists. These will only contain the
    // subelements as they're generated. When they're complete, we'll call
    // push_subs for each of them into a new Abstract.
    let mut builder = AbstractBuilder::with_rank_capacity(rank + 1);
    let offset_memo = OffsetMemo::<MIN, MAX>::new(p, q);

    // If MIN, we have to set a minimal element and the vertices manually.
    if MIN {
        builder.push_min();
        builder.push_vertices(p.vertex_count() * q.vertex_count());
    }

    let lo = 2 * min_u;
    let hi = rank - max_u;

    // Adds elements in order of rank.
    for prod_rank in lo..=hi {
        let lo = (min_u as isize).max((prod_rank + min_u) as isize - q_hi as isize) as usize;
        let hi = p_hi.min(prod_rank);
        let mut subelements = SubelementList::new();

        // Adds elements by lexicographic order of the ranks.
        for p_el_rank in lo..=hi {
            let q_el_rank = prod_rank + min_u - p_el_rank;

            // Takes the product of every element in p with rank p_els_rank,
            // with every element in q with rank q_els_rank.
            for (p_idx, p_el) in p[p_el_rank].iter().enumerate() {
                for (q_idx, q_el) in q[q_el_rank].iter().enumerate() {
                    let mut subs = Subelements::new();

                    // Products of p's subelements with q.
                    if !MIN || p_el_rank != 1 {
                        for &p_sub in &p_el.subs {
                            subs.push(offset_memo.get_element_index(
                                p_el_rank - 1,
                                p_sub,
                                q,
                                q_el_rank,
                                q_idx,
                            ))
                        }
                    }

                    // Products of q's subelements with p.
                    if !MIN || q_el_rank != 1 {
                        for &q_sub in &q_el.subs {
                            subs.push(offset_memo.get_element_index(
                                p_el_rank,
                                p_idx,
                                q,
                                q_el_rank - 1,
                                q_sub,
                            ))
                        }
                    }

                    subelements.push(subs)
                }
            }
        }

        builder.push(subelements);
    }

    // If MAX, we have to set a maximal element manually.
    if MAX {
        builder.push_max();
    }

    // TODO: If `p` and `q` are sorted, this should be too?

    // Safety: we've built one of the four products on polytopes. For a
    // proof that these constructions yield valid abstract polytopes, see
    // [TODO: write proof].
    unsafe { builder.build() }
}

/// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
/// from two polytopes. This is a [`product`] where `!MIN` and `!MAX`.
///
/// The vertices of the result will be those corresponding to the vertices of
/// `p` in the same order, following those corresponding to `q` in the same
/// order.
pub(super) fn duopyramid(p: &Abstract, q: &Abstract) -> Abstract {
    product::<false, false>(q, p)
}

/// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
/// from two polytopes. This is a [`product`] where `MIN` and `!MAX`.
pub(super) fn duoprism(p: &Abstract, q: &Abstract) -> Abstract {
    product::<true, false>(p, q)
}

/// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
/// from two polytopes. This is a [`product`] where `!MIN` and `MAX`.
///
/// The vertices of the result will be those corresponding to the vertices of
/// `p` in the same order, following those corresponding to `q` in the same
/// order.
pub(super) fn duotegum(p: &Abstract, q: &Abstract) -> Abstract {
    product::<false, true>(q, p)
}

/// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
/// from two polytopes. This is a [`product`] where `MIN` and `MAX`.
pub(super) fn duocomb(p: &Abstract, q: &Abstract) -> Abstract {
    product::<true, true>(p, q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test;

    /// Checks that products involving the nullitope are handled correctly.
    #[test]
    fn nullitope_product() {
        let nullitope = Abstract::nullitope();
        let cube = Abstract::cube();

        assert_eq!(nullitope.duopyramid(&cube).rank(), 4);
        assert!(nullitope.duoprism(&cube).is_nullitope());
        assert!(nullitope.duotegum(&cube).is_nullitope());
        assert!(nullitope.duocomb(&cube).is_nullitope());
    }

    /// Tests that polygonal duoproducts (i.e. duopyramids, duoprisms,
    /// duotegums, duocombs) are generated correctly by checking that the
    /// element counts for the product of an m-gon and an n-gon match for a few
    /// values.
    fn test_duoproduct<P, E, const N: usize>(product: P, element_counts: E)
    where
        P: Fn(&Abstract, &Abstract) -> Abstract,
        E: Fn(usize, usize) -> [usize; N],
    {
        let polygons: Vec<_> = (2..=5).into_iter().map(Abstract::polygon).collect();

        for (m, p) in polygons.iter().enumerate() {
            for (n, q) in polygons.iter().enumerate() {
                test(&product(p, q), element_counts(m + 2, n + 2))
            }
        }
    }

    /// Checks polygonal duopyramids.
    #[test]
    fn duopyramid() {
        test_duoproduct(Abstract::duopyramid, |m, n| {
            [
                1,
                m + n,
                m + n + m * n,
                2 * m * n + 2,
                m + n + m * n,
                m + n,
                1,
            ]
        })
    }

    /// Checks polygonal duoprisms.
    #[test]
    fn duoprism() {
        test_duoproduct(Abstract::duoprism, |m, n| {
            [1, m * n, 2 * m * n, m + n + m * n, m + n, 1]
        })
    }

    /// Checks polygonal duotegums.
    #[test]
    fn duotegum() {
        test_duoproduct(Abstract::duotegum, |m, n| {
            [1, m + n, m + n + m * n, 2 * m * n, m * n, 1]
        })
    }

    /// Checks polygonal duocombs.
    #[test]
    fn duocomb() {
        test_duoproduct(Abstract::duocomb, |m, n| [1, m * n, 2 * m * n, m * n, 1])
    }
}
