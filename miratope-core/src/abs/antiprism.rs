//! Contains the code to build an antiprism.

use std::{collections::HashMap, iter};

use super::{Abstract, AbstractBuilder, Ranked, SubelementList, Subelements};

use vec_like::VecLike;

/// Represents the lowest and highest element of a section of an abstract
/// polytope. Not to be confused with a cross-section.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Section {
    /// The rank of the lowest element in the section.
    pub lo_rank: usize,

    /// The index of the lowest element in the section.
    pub lo_idx: usize,

    /// The rank of the highest element in the section.
    pub hi_rank: usize,

    /// The index of the highest element in the section.
    pub hi_idx: usize,
}

impl std::fmt::Display for Section {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "section between ({:?}) and ({:?})", self.lo(), self.hi())
    }
}

impl Section {
    /// Initializes a new section between two elements.
    pub fn new(lo_rank: usize, lo_idx: usize, hi_rank: usize, hi_idx: usize) -> Self {
        Self {
            lo_rank,
            lo_idx,
            hi_rank,
            hi_idx,
        }
    }

    /// Creates a new singleton section.
    pub fn singleton(rank: usize, idx: usize) -> Self {
        Self::new(rank, idx, rank, idx)
    }

    /// Creates a new section by replacing the lowest element of another.
    pub fn with_lo(mut self, lo_rank: usize, lo_idx: usize) -> Self {
        self.lo_rank = lo_rank;
        self.lo_idx = lo_idx;
        self
    }

    /// Creates a new section by replacing the highest element of another.
    pub fn with_hi(mut self, hi_rank: usize, hi_idx: usize) -> Self {
        self.hi_rank = hi_rank;
        self.hi_idx = hi_idx;
        self
    }

    /// Returns the lowest element of a section.
    pub fn lo(self) -> (usize, usize) {
        (self.lo_rank, self.lo_idx)
    }

    /// Returns the highest element of a section.
    pub fn hi(self) -> (usize, usize) {
        (self.hi_rank, self.hi_idx)
    }
}

/// Represents a map from sections in a polytope to their indices in a new
/// polytope (its [antiprism](Abstract::antiprism)). Exists only to make the
/// antiprism code a bit easier to understand.
///
/// In practice, all of the sections we store have a common height, which means
/// that we could save some memory by using a representation of [`SectionRef`]
/// with three arguments instead of four. This probably isn't worth the hassle,
/// though.
#[derive(Default, Debug)]
struct SectionMap(HashMap<Section, usize>);

impl IntoIterator for SectionMap {
    type Item = (Section, usize);

    type IntoIter = std::collections::hash_map::IntoIter<Section, usize>;

    /// Returns an iterator over the stored section index pairs.
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl SectionMap {
    /// Initializes a new section hash.
    pub fn new() -> Self {
        Default::default()
    }

    /// Returns the number of stored elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns all singleton sections of a polytope.
    pub fn singletons(poly: &Abstract) -> Self {
        let mut section_hash = Self::new();

        for (rank, elements) in poly.iter().enumerate() {
            for idx in 0..elements.len() {
                section_hash
                    .0
                    .insert(Section::singleton(rank, idx), section_hash.len());
            }
        }

        section_hash
    }

    /// Gets the index of a section in the hash, inserting it if necessary.
    pub fn get_insert(&mut self, section: Section) -> usize {
        use std::collections::hash_map::Entry;

        // We organize by lowest rank, then by hash.
        let len = self.len();
        match self.0.entry(section) {
            // Directly returns the index of the section.
            Entry::Occupied(idx) => *idx.get(),

            // Adds the section, increases the length by 1, then returns its index.
            Entry::Vacant(entry) => {
                entry.insert(len);
                len
            }
        }
    }
}

/// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
/// based on a given polytope. Also returns the indices of the vertices that
/// form the base and the dual base, in that order.
pub(super) fn antiprism_and_vertices(abs: &Abstract) -> (Abstract, Vec<usize>, Vec<usize>) {
    let rank = abs.rank();
    let mut section_hash = SectionMap::singletons(abs);

    // We actually build the elements backwards, which is as awkward as it
    // seems. Maybe we should fix that in the future?
    let mut backwards_abs = Vec::with_capacity(rank + 1);
    backwards_abs.push(SubelementList::max(section_hash.len()));

    // Indices of base.
    let vertex_count = abs.vertex_count();
    let mut vertices = Vec::with_capacity(vertex_count);

    // Indices of dual base.
    let facet_count = abs.facet_count();
    let mut dual_vertices = Vec::with_capacity(facet_count);

    // Adds all elements corresponding to sections of a given height.
    for height in 1..=rank + 1 {
        let mut new_section_hash = SectionMap::new();
        let mut elements: SubelementList = iter::repeat(Subelements::new())
            .take(section_hash.len())
            .collect();

        // Goes over all sections of the previous height, and builds the
        // sections of the current height by either changing the upper
        // element into one of its superelements, or changing the lower
        // element into one of its subelements.
        for (section, idx) in section_hash.into_iter() {
            for &idx_lo in &abs[section.lo()].subs {
                elements[idx].push(
                    new_section_hash.get_insert(section.with_lo(section.lo_rank - 1, idx_lo)),
                );
            }

            // Finds all of the superelements of our old section's
            // highest element.
            for &idx_hi in &abs[section.hi()].sups {
                elements[idx].push(
                    new_section_hash.get_insert(section.with_hi(section.hi_rank + 1, idx_hi)),
                );
            }
        }

        // We figure out where the vertices of the base and the dual base
        // were sent.
        if height == rank - 1 {
            // We create a map from the base's vertices to the new vertices.
            for v in 0..vertex_count {
                vertices.push(new_section_hash.get_insert(Section::new(1, v, rank, 0)));
            }

            // We create a map from the dual base's vertices to the new vertices.
            for f in 0..facet_count {
                dual_vertices.push(new_section_hash.get_insert(Section::new(0, 0, rank - 1, f)));
            }
        }

        backwards_abs.push(elements);
        section_hash = new_section_hash;
    }

    // We built this backwards, so let's fix it.
    let builder: AbstractBuilder = backwards_abs.into_iter().rev().collect();

    // Safety: we've built an antiprism based on the polytope. For a proof
    // that this construction yields a valid abstract polytope, see [TODO:
    // write proof].
    (unsafe { builder.build() }, vertices, dual_vertices)
}
