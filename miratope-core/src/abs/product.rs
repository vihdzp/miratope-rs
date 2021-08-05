use super::*;

/// When we compute any polytope product, we add the elements of any given rank
/// in lexicographic order of the ranks. This struct memoizes how many elements
/// of the same rank are added by the time we add those of the form
/// `(p_rank, q_rank)`. It stores this value in `offset_memo[(p_rank, q_rank)]`.
struct OffsetMemo<'a, const MIN: bool, const MAX: bool> {
    /// The memoized values.
    memo: Vec<usize>,

    /// The first factor of the product.
    p: &'a Abstract,

    /// The second factor of the product.
    q: &'a Abstract,
}

impl<'a, const MIN: bool, const MAX: bool> OffsetMemo<'a, MIN, MAX> {
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

    /// Initializes a new offset memoizator.
    fn new(p: &'a Abstract, q: &'a Abstract) -> Self {
        let memo = Vec::with_capacity(Self::range_len(p) * Self::range_len(q));
        Self { memo, p, q }.fill_memo()
    }

    /// Returns the number of elements in `p` of rank `p_rank`, times the number
    /// of elements in `q` of rank `q_rank`.
    fn count_prod(&self, p_rank: usize, q_rank: usize) -> usize {
        self.p.el_count(p_rank) * self.q.el_count(q_rank)
    }

    /// Calculates and stores the required values.
    fn fill_memo(mut self) -> Self {
        debug_assert!(self.memo.is_empty());

        for q_rank in Self::MIN_U..=Self::hi(self.q) {
            self.memo.push(self.count_prod(Self::MIN_U, q_rank));
        }

        for p_rank in (Self::MIN_U + 1)..=Self::hi(self.p) {
            for q_rank in Self::MIN_U..Self::hi(self.q) {
                self.memo
                    .push(self[(p_rank - 1, q_rank + 1)] + self.count_prod(p_rank, q_rank));
            }

            self.memo.push(self.count_prod(p_rank, Self::hi(self.q)));
        }

        self
    }

    /// Every element of the product is in correspondence with
    /// a pair of an element from `p` and an element from `q`. This function
    /// finds the position we placed it in.
    fn get_element_index(&self, p_rank: usize, p_idx: usize, q_rank: usize, q_idx: usize) -> usize {
        let idx = p_idx * self.q.el_count(q_rank) + q_idx;

        if p_rank == Self::MIN_U || q_rank == Self::hi(self.q) {
            idx
        } else {
            self[(p_rank - 1, q_rank + 1)] + idx
        }
    }
}

impl<'a, const MIN: bool, const MAX: bool> Index<(usize, usize)> for OffsetMemo<'a, MIN, MAX> {
    type Output = usize;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.memo[(index.0 - Self::MIN_U) * Self::range_len(self.q) + (index.1 - Self::MIN_U)]
    }
}

impl Abstract {
    /// Takes the [direct product](https://en.wikipedia.org/wiki/Direct_product#Direct_product_of_binary_relations)
    /// of two polytopes. If the `MIN` flag is turned on, it ignores the
    /// minimal elements of both of the factors and adds one at the end. The
    /// `MAX` flag works analogously.
    ///
    /// This method takes in `MIN` and `MAX` as type parameters so that each
    /// case may be separately optimized.
    ///
    /// The elements of this product are in one to one correspondence to pairs
    /// of elements in the set of polytopes. The elements of a specific rank are
    /// sorted first by lexicographic order of the ranks, then by lexicographic
    /// order of the elements.
    pub fn product<const MIN: bool, const MAX: bool>(p: &Self, q: &Self) -> Self {
        // The ranks of p and q.
        let p_rank = p.rank();
        let q_rank = q.rank();

        // 0 or 1 depending on whether the minimum/maximum elements are in the
        // polytope.
        let min_u = MIN as usize;
        let max_u = MAX as usize;

        // The highest ranks we'll use to take products in p and q.
        let p_hi = p_rank - max_u;
        let q_hi = q_rank - max_u;

        // The rank of the product.
        let rank = p_rank + q_rank - min_u - max_u;

        // Initializes the element lists. These will only contain the
        // subelements as they're generated. When they're complete, we'll call
        // push_subs for each of them into a new Abstract.
        // TODO: use a builder instead.
        let mut builder = AbstractBuilder::with_capacity(rank + 1);
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
            let lo = if prod_rank + min_u >= q_hi {
                min_u.max(prod_rank + min_u - q_hi)
            } else {
                min_u
            };
            let hi = p_hi.min(prod_rank);
            let mut subelements = SubelementList::new();

            // Adds elements by lexicographic order of the ranks.
            for p_els_rank in lo..=hi {
                let q_els_rank = prod_rank + min_u - p_els_rank;

                // Takes the product of every element in p with rank p_els_rank,
                // with every element in q with rank q_els_rank.
                for (p_idx, p_el) in p[p_els_rank].iter().enumerate() {
                    for (q_idx, q_el) in q[q_els_rank].iter().enumerate() {
                        let mut subs = Subelements::new();

                        // Products of p's subelements with q.
                        if !MIN || p_els_rank != 1 {
                            for &s in &p_el.subs {
                                subs.push(offset_memo.get_element_index(
                                    p_els_rank - 1,
                                    s,
                                    q_els_rank,
                                    q_idx,
                                ))
                            }
                        }

                        // Products of q's subelements with p.
                        if !MIN || q_els_rank != 1 {
                            for &s in &q_el.subs {
                                subs.push(offset_memo.get_element_index(
                                    p_els_rank,
                                    p_idx,
                                    q_els_rank - 1,
                                    s,
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
        builder.build()
    }
}
