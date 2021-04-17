use std::collections::HashMap;

use super::{
    geometry::{Hyperplane, Segment},
    Abstract, Concrete, ElementList, Element, Polytope, Subelements,
};

impl Concrete {
    /// Takes the cross-section of a polytope through a given hyperplane.
    ///
    /// # Todo
    /// We should make this function take a general [`Subspace`] instead.
    pub fn slice(&self, slice: Hyperplane) -> Self {
        let mut vertices = Vec::new();

        let mut abs = Abstract::new();

        // We map all indices of k-elements in the original polytope to the
        // indices of the new (k-1)-elements resulting from taking their
        // intersections with the slicing hyperplane.
        let mut hash_element = HashMap::new();

        // Determines the vertices of the cross-section.
        for (idx, edge) in self[1].iter().enumerate() {
            let segment = Segment(
                self.vertices[edge.subs[0]].clone(),
                self.vertices[edge.subs[1]].clone(),
            );

            // If we got ourselves a new vertex:
            if let Some(p) = slice.intersect(segment) {
                hash_element.insert(idx, vertices.len());
                vertices.push(slice.flatten(&p));
            }
        }

        let vertex_count = vertices.len();

        // The slice does not intersect the polytope.
        if vertex_count == 0 {
            return Self::nullitope();
        }

        abs.push(ElementList::min(vertex_count));
        abs.push(ElementList::vertices(vertex_count));

        // Takes care of building everything else.
        for r in 2..self.rank() {
            let mut new_hash_element = HashMap::new();
            let mut new_els = ElementList::new();

            for (idx, el) in self[r].iter().enumerate() {
                let mut new_subs = Vec::new();
                for sub in el.subs.iter() {
                    if let Some(&v) = hash_element.get(sub) {
                        new_subs.push(v);
                    }
                }

                // If we got ourselves a new edge:
                if !new_subs.is_empty() {
                    new_hash_element.insert(idx, new_els.len());
                    new_els.push(Element::from_subs(Subelements(new_subs)));
                }
            }

            abs.push_subs(new_els);
            hash_element = new_hash_element;
        }

        let facet_count = abs.last().unwrap().len();
        abs.push_subs(ElementList::max(facet_count));

        Self { vertices, abs }
    }
}
