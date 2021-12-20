//! Contains the code to build an antiprism.

use std::collections::HashMap;

use super::{Abstract, AbstractBuilder, Ranked, SubelementList, Subelements};

use vec_like::VecLike;

/// Represents sections in a polytope with a common height that is stored
/// elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PartialSection {
    /// The rank of the lower element of the section.
    rank_lo: usize,

    /// The index of the lower element of the section.
    idx_lo: usize,

    /// The index of the upper element of the section.
    idx_hi: usize,
}

impl PartialSection {
    /// Initializes a new partial section.
    fn new(rank_lo: usize, idx_lo: usize, idx_hi: usize) -> Self {
        Self {
            rank_lo,
            idx_lo,
            idx_hi,
        }
    }

    /// Returns the rank of the highest element, given the height of the
    /// section.
    ///
    /// The height is the difference between the upper and lower ranks.
    fn hi_rank(self, height: usize) -> usize {
        self.rank_lo + height
    }

    /// Returns the singleton partial section with a given element.
    fn singleton(rank: usize, idx: usize) -> Self {
        Self::new(rank, idx, idx)
    }

    /// Returns the rank and index of the lowest element.
    fn lo(self) -> (usize, usize) {
        (self.rank_lo, self.idx_lo)
    }

    /// Returns the rank and index of the highest element, given the height of
    /// the section.
    fn hi(self, height: usize) -> (usize, usize) {
        (self.hi_rank(height), self.idx_hi)
    }
}

/// Represents a map from sections of a common height in a polytope to their
/// indices in its [`antiprism`].
#[derive(Default, Debug)]
struct SectionMap(HashMap<PartialSection, usize>);

impl IntoIterator for SectionMap {
    type Item = (PartialSection, usize);
    type IntoIter = std::collections::hash_map::IntoIter<PartialSection, usize>;

    /// Returns an iterator over the stored section index pairs.
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl SectionMap {
    /// Initializes a new section hash.
    fn new() -> Self {
        Default::default()
    }

    /// Returns the number of stored elements.
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns a map from all singleton sections of a polytope to consecutive
    /// indices.
    fn singletons(poly: &Abstract) -> Self {
        let mut section_hash = Self::new();

        for (rank, elements) in poly.iter().enumerate() {
            for idx in 0..elements.len() {
                section_hash
                    .0
                    .insert(PartialSection::singleton(rank, idx), section_hash.len());
            }
        }

        section_hash
    }

    /// Gets the index of a section in the hash, or `None` if it doesn't exist.
    fn get(&self, section: PartialSection) -> Option<usize> {
        self.0.get(&section).copied()
    }

    /// Gets the index of a section in the hash, inserting it if necessary.
    fn get_insert(&mut self, section: PartialSection) -> usize {
        use std::collections::hash_map::Entry::*;

        // We organize by lowest rank, then by hash.
        let len = self.len();
        match self.0.entry(section) {
            // Directly returns the index of the section.
            Occupied(idx) => *idx.get(),

            // Adds the section, increases the length by 1, then returns its index.
            Vacant(entry) => {
                entry.insert(len);
                len
            }
        }
    }
}

/// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
/// based on a given polytope. Also returns the indices of the vertices that
/// form the base and the dual base, in that order.
pub(super) fn antiprism_and_vertices(p: &Abstract) -> (Abstract, Vec<usize>, Vec<usize>) {
    let rank = p.rank();
    let mut section_map = SectionMap::singletons(p);

    // We build the elements backwards.
    let mut backwards_res = Vec::with_capacity(rank + 1);
    backwards_res.push(SubelementList::max(section_map.len()));

    // Indices of base.
    let vertex_count = p.vertex_count();
    let mut vertices = Vec::with_capacity(vertex_count);

    // Indices of dual base.
    let facet_count = p.facet_count();
    let mut dual_vertices = Vec::with_capacity(facet_count);

    // Adds all elements corresponding to sections of a given height.
    for height in 0..=rank {
        let mut new_section_map = SectionMap::new();
        let mut elements: SubelementList = std::iter::repeat(Subelements::new())
            .take(section_map.len())
            .collect();

        // Goes over all sections of the previous height, and builds the
        // sections of the current height by either changing the upper element
        // into one of its superelements, or changing the lower element into one
        // of its subelements.
        for (section, idx) in section_map.into_iter() {
            for &idx_lo in &p[section.lo()].subs {
                elements[idx].push(new_section_map.get_insert(PartialSection::new(
                    section.rank_lo - 1,
                    idx_lo,
                    section.idx_hi,
                )));
            }

            // Finds all of the superelements of our old section's
            // highest element.
            for &idx_hi in &p[section.hi(height)].sups {
                elements[idx].push(new_section_map.get_insert(PartialSection::new(
                    section.rank_lo,
                    section.idx_lo,
                    idx_hi,
                )));
            }
        }

        // We figure out where the vertices of the base and the dual base
        // were sent.
        if height + 2 == rank {
            // We create a map from the base's vertices to the new vertices.
            for v in 0..vertex_count {
                vertices.push(new_section_map.get(PartialSection::new(1, v, 0)).unwrap());
            }

            // We create a map from the dual base's vertices to the new vertices.
            for f in 0..facet_count {
                dual_vertices.push(new_section_map.get(PartialSection::new(0, 0, f)).unwrap());
            }
        }

        backwards_res.push(elements);
        section_map = new_section_map;
    }

    // We built this backwards, so let's fix it.
    let builder: AbstractBuilder = backwards_res.into_iter().rev().collect();

    // Safety: we've built an antiprism based on the polytope. For a proof
    // that this construction yields a valid abstract polytope, see [TODO:
    // write proof].
    (unsafe { builder.build() }, vertices, dual_vertices)
}

/// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
/// based on a given polytope.
pub(super) fn antiprism(p: &Abstract) -> Abstract {
    antiprism_and_vertices(p).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test, Polytope};

    /// Checks the nullitope antiprism.
    #[test]
    fn nullitope_antiprism() {
        test(&Abstract::nullitope().antiprism(), [1, 1])
    }

    /// Checks some polygonal antiprisms.
    #[test]
    fn polygon_antiprism() {
        for n in 2..=10 {
            test(
                &Abstract::polygon(n).antiprism(),
                [1, 2 * n, 4 * n, 2 * n + 2, 1],
            )
        }
    }

    /// Checks a cubic antiprism.
    #[test]
    fn cubic_antiprism() {
        test(&Abstract::cube().antiprism(), [1, 14, 48, 62, 28, 1])
    }
}
