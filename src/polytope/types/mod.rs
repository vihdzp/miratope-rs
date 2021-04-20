use ::derive_deref::*;

use std::collections::HashMap;

use super::{rank::RankVec, Abstract, Element, ElementList};

pub mod r#abstract;
pub mod concrete;
pub mod renderable;

#[derive(Deref, DerefMut)]
/// As a byproduct of calculating either the vertices or the entire polytope
/// corresponding to a given section, we generate a map from ranks and indices
/// in the original polytope to ranks and indices in the section. This struct
/// encodes such a map as a vector of hash maps.
struct ElementHash(RankVec<HashMap<usize, usize>>);

impl ElementHash {
    /// Returns a map from elements on the polytope to elements in an element.
    /// If the element doesn't exist, we return `None`.
    fn from_element(poly: &Abstract, rank: isize, idx: usize) -> Option<Self> {
        poly.get_element(rank, idx)?;

        // A vector of HashMaps. The k-th entry is a map from k-elements of the
        // original polytope into k-elements in a new polytope.
        let mut hashes = RankVec::with_capacity(rank);
        for _ in -1..=rank {
            hashes.push(HashMap::new());
        }
        hashes[rank].insert(idx, 0);

        // Gets subindices of subindices, until reaching the vertices.
        for r in (0..=rank).rev() {
            let (left_slice, right_slice) = hashes.split_at_mut(r);
            let prev_hash = left_slice.last_mut().unwrap();
            let hash = right_slice.first().unwrap();

            for (&idx, _) in hash.iter() {
                for &sub in poly[r as isize][idx].subs.iter() {
                    let len = prev_hash.len();
                    prev_hash.entry(sub).or_insert(len);
                }
            }
        }

        Some(Self(hashes))
    }

    /// Gets the indices of the elements of a given rank in a polytope.
    fn to_elements(&self, rank: isize) -> Vec<usize> {
        if let Some(elements) = self.get(rank) {
            let mut new_elements = Vec::new();
            new_elements.resize(elements.len(), 0);

            for (&sub, &idx) in elements {
                new_elements[idx] = sub;
            }

            new_elements
        } else {
            Vec::new()
        }
    }

    /// Gets the indices of the vertices of a given element in a polytope.
    fn to_polytope(&self, poly: &Abstract) -> Abstract {
        let rank = self.rank();
        let mut abs = Abstract::with_capacity(rank);

        for r in -1..=rank {
            let mut elements = ElementList::new();
            let hash = &self[r];

            for _ in 0..hash.len() {
                elements.push(Element::new());
            }

            // For every element of rank r in the hash element list.
            for (&idx, &new_idx) in hash {
                // We take the corresponding element in the original polytope
                // and use the hash map to get its sub and superelements in the
                // new polytope.
                let el = poly.get_element(r, idx).unwrap();
                let mut new_el = Element::new();

                // Gets the subelements.
                if let Some(prev_hash) = self.get(r - 1) {
                    for sub in el.subs.iter() {
                        if let Some(&new_sub) = prev_hash.get(sub) {
                            new_el.subs.push(new_sub);
                        }
                    }
                }

                // Gets the superelements.
                if let Some(next_hash) = self.get(r + 1) {
                    for sup in el.sups.iter() {
                        if let Some(&new_sup) = next_hash.get(sup) {
                            new_el.sups.push(new_sup);
                        }
                    }
                }

                elements[new_idx] = new_el;
            }

            abs.push(elements);
        }

        abs
    }
}
