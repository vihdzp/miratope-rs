use super::*;

/// We add the elements of a given rank in lexicographic order of the
/// ranks. This struct memoizes how many elements of the same rank are
/// added by the time we add those of the form (p_rank, q_rank). It
/// stores this value in offset_memo[p_rank - min_u][q_rank - min_u].
struct OffsetMemo<const MIN: bool, const MAX: bool>(Vec<Vec<usize>>);

impl<const MIN: bool, const MAX: bool> OffsetMemo<MIN, MAX> {
    fn new(p: &Abstract, q: &Abstract) -> Self {
        // The ranks of p and q.
        let p_rank = p.rank();
        let q_rank = q.rank();

        // 0 or 1 depending on whether the minimum/maximum elements are in the
        // polytope.
        let min_u = !MIN as usize;
        let max_u = !MAX as usize;

        // The highest ranks we'll use to take products in p and q.
        let p_hi = p_rank - max_u;
        let q_hi = q_rank - max_u;

        let mut offset_memo: Vec<Vec<_>> = Vec::new();
        for p_rank in 0..=(p_hi - min_u) {
            let mut offset_memo_row = Vec::new();

            for q_rank in 0..=(q_hi - min_u) {
                offset_memo_row.push(
                    if p_rank == 0 || q_rank == q_hi - min_u {
                        0
                    } else {
                        offset_memo[p_rank - 1][q_rank + 1]
                    } + p.el_count(p_rank + min_u) * q.el_count(q_rank + min_u),
                );
            }

            offset_memo.push(offset_memo_row);
        }

        Self(offset_memo)
    }

    // Every element of the product is in one to one correspondence with
    // a pair of an element from p and an element from q. This function
    // finds the position we placed it in.
    fn get_element_index(
        &self,
        p_rank: usize,
        p_idx: usize,
        q: &Abstract,
        q_rank: usize,
        q_idx: usize,
    ) -> usize {
        let idx = p_idx * q.el_count(q_rank) + q_idx;

        if p_rank == !MIN as usize {
            idx
        } else {
            self[(p_rank - 1, q_rank + 1)] + idx
        }
    }
}

impl<const MIN: bool, const MAX: bool> Index<(usize, usize)> for OffsetMemo<MIN, MAX> {
    type Output = usize;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let min_u = !MIN as usize;

        self.0
            .get(index.0 - min_u)
            .map(|row| row.get(index.1 - min_u))
            .flatten()
            .unwrap_or(&0)
    }
}

impl Abstract {
    /// Takes the [direct product](https://en.wikipedia.org/wiki/Direct_product#Direct_product_of_binary_relations)
    /// of two polytopes. If the `min` flag is turned off, it ignores the
    /// minimal elements of both of the factors and adds one at the end. The
    /// `max` flag works analogously.
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
        let min_u = !MIN as usize;
        let max_u = !MAX as usize;

        // The highest ranks we'll use to take products in p and q.
        let p_hi = p_rank - max_u;
        let q_hi = q_rank - max_u;

        // The rank of the product.
        let rank = p_rank + q_rank - min_u - max_u;

        // Initializes the element lists. These will only contain the
        // subelements as they're generated. When they're complete, we'll call
        // push_subs for each of them into a new Abstract.
        // TODO: use a builder instead.
        let mut element_lists = Vec::with_capacity(rank + 1);
        for _ in 0..=rank {
            element_lists.push(SubelementList::new());
        }

        // We add the elements of a given rank in lexicographic order of the
        // ranks. This vector memoizes how many elements of the same rank are
        // added by the time we add those of the form (p_rank, q_rank). It
        // stores this value in offset_memo[p_rank - min_u][q_rank - min_u].

        // TODO: to avoid allocations, we can go a step further and make this
        // into a single float.

        // TODO: make this into its own struct.
        let offset_memo = OffsetMemo::<MIN, MAX>::new(p, q);

        // Adds elements in order of rank.
        for prod_rank in 0..=rank {
            let hi = p_hi.min(prod_rank + min_u);

            // Adds elements by lexicographic order of the ranks.
            for p_els_rank in min_u..=hi {
                let q_els_rank = prod_rank + min_u - p_els_rank;
                if !(min_u..=q_hi).contains(&q_els_rank) {
                    continue;
                }

                // Takes the product of every element in p with rank p_els_rank,
                // with every element in q with rank q_els_rank.
                for (p_idx, p_el) in p[p_els_rank].iter().enumerate() {
                    for (q_idx, q_el) in q[q_els_rank].iter().enumerate() {
                        let mut subs = Subelements::new();

                        // Products of p's subelements with q.
                        if MIN || p_els_rank != 1 {
                            for &s in &p_el.subs {
                                subs.push(offset_memo.get_element_index(
                                    p_els_rank - 1,
                                    s,
                                    q,
                                    q_els_rank,
                                    q_idx,
                                ))
                            }
                        }

                        // Products of q's subelements with p.
                        if MIN || q_els_rank != 1 {
                            for &s in &q_el.subs {
                                subs.push(offset_memo.get_element_index(
                                    p_els_rank,
                                    p_idx,
                                    q,
                                    q_els_rank - 1,
                                    s,
                                ))
                            }
                        }

                        element_lists[prod_rank].push(subs)
                    }
                }
            }
        }

        // If !min, we have to set a minimal element manually.
        if !MIN {
            let vertex_count = p.vertex_count() * q.vertex_count();
            element_lists[0] = SubelementList::min();
            element_lists[1] = SubelementList::vertices(vertex_count);
        }

        // If !max, we have to set a maximal element manually.
        if !MAX {
            element_lists[rank] = SubelementList::max(element_lists[rank - 1].len());
        }

        // Uses push_subs to add all of the element lists into a new polytope.
        let mut product = AbstractBuilder::with_capacity(element_lists.len() + 1);
        for elements in element_lists.into_iter() {
            product.push(elements);
        }

        // If `p` and `q` are sorted, this should be too?
        product.build()
    }
}
