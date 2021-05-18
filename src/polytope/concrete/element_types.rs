use std::collections::HashMap;

use crate::polytope::{
    concrete::Concrete,
    r#abstract::{
        elements::ElementRef,
        rank::{Rank, RankVec},
    },
    Polytope,
};

/// The info for any of the "element types" in the polytope. We store metadata
/// about every element of the polytope, and consider these elements as having
/// the same "type" if the metadata matches up.
#[derive(PartialEq, Eq, Hash)]
pub struct ElementData {
    element_counts: RankVec<usize>,
}

impl ElementData {
    pub fn rank(&self) -> Rank {
        self.element_counts.rank()
    }

    pub fn facet_count(&self) -> usize {
        self.element_counts[self.rank() - Rank::new(1)]
    }
}

/// The elements that share a given [`ElementData`].
pub struct ElementType {
    /// The indices of all elements that share the same data.
    indices: Vec<usize>,

    /// The data common to all elements of this type.
    data: ElementData,
}

impl ElementType {
    pub fn count(&self) -> usize {
        self.indices.len()
    }
}

fn get_data(el: &ElementRef, polytope: &Concrete) -> ElementData {
    ElementData {
        element_counts: polytope.element(el).unwrap().el_counts(),
    }
}

impl Concrete {
    /// Gets the element "types" of a polytope.
    fn get_element_types(&self) -> RankVec<Vec<ElementType>> {
        use std::collections::hash_map::Entry;

        let rank = self.rank();
        let mut output = RankVec::with_capacity(rank);

        for (r, elements) in self.abs.ranks.iter().rank_enumerate() {
            // A map from element data to the indices of the elements with such data.
            let mut types = HashMap::new();

            // We build the map.
            for idx in 0..elements.len() {
                let data = get_data(&ElementRef::new(r, idx), self);

                match types.entry(data) {
                    Entry::Vacant(entry) => {
                        entry.insert(vec![idx]);
                    }
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().push(idx);
                    }
                }
            }

            // We create the element types from the map.
            let mut element_types = Vec::with_capacity(types.len());
            for (data, indices) in types {
                element_types.push(ElementType { indices, data });
            }

            output.push(element_types);
        }

        output
    }

    pub fn print_element_types(&self) -> String {
        let types = self.get_element_types();
        let mut output = String::new();

        // We'll move this over to the translation module... some day.
        let el_names = RankVec(vec![
            "", "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta",
            "Xenna", "Daka", "Henda",
        ]);
        let el_suffixes = RankVec(vec![
            "", "", "", "gon", "hedron", "choron", "teron", "peton", "exon", "zetton", "yotton",
            "xennon", "dakon",
        ]);

        output.push_str(&el_names[Rank::new(0)].to_string());
        output.push('\n');
        for t in &types[Rank::new(0)] {
            output.push_str(&t.count().to_string());
            output.push('\n');
        }
        output.push('\n');

        output.push_str(&el_names[Rank::new(1)].to_string());
        output.push('\n');
        for t in &types[Rank::new(1)] {
            output.push_str(&format!("{}\n", t.count()));
        }
        output.push('\n');

        for d in Rank::range_iter(Rank::new(2), self.rank()) {
            output.push_str(&el_names[d].to_string());
            output.push('\n');
            for t in &types[d] {
                output.push_str(&format!(
                    "{} × {}-{}\n",
                    t.count(),
                    t.data.facet_count(),
                    el_suffixes[d]
                ));
            }
            output.push('\n');
        }

        // This doesn't actually print components lol.
        output.push_str("Components:\n");
        for t in &types[self.rank()] {
            output.push_str(&format!(
                "{} × {}-{}\n",
                t.count(),
                t.data.facet_count(),
                el_suffixes[self.rank()]
            ));
        }

        output
    }
}
