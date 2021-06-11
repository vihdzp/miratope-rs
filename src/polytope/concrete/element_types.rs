//! The code used to tally up the "element types" in a polytope.

use std::collections::{BTreeMap, HashMap};

use crate::{
    polytope::{concrete::Concrete, r#abstract::rank::Rank, Polytope},
    Consts, Float, FloatOrd,
};

use approx::abs_diff_eq;

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct ElementCount {
    /// The index of the type of these elements.
    type_index: usize,

    /// The number of elements of a given type.
    count: usize,
}

pub struct ElementCountBuilder(BTreeMap<usize, usize>);

impl ElementCountBuilder {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn insert(&mut self, type_index: usize) {
        if let Some(count) = self.0.get_mut(&type_index) {
            *count += 1;
        } else {
            self.0.insert(type_index, 1);
        }
    }

    pub fn build(self) -> ElementData {
        let mut res = Vec::new();
        for (type_index, count) in self.0 {
            res.push(ElementCount { type_index, count });
        }
        ElementData::ElementCounts(res)
    }
}

/// The info for any of the "element types" in the polytope. We store metadata
/// about every element of the polytope, and consider these elements as having
/// the same "type" if the metadata matches up.
#[derive(PartialEq, Eq, Hash)]
pub enum ElementData {
    /// Every point has the same type for now.
    Point,

    /// An edge's type is given by its length.
    Edge(FloatOrd),

    /// Any other element's type is given by the number of subelements of each
    /// type that it contains.
    ElementCounts(Vec<ElementCount>),
}

impl ElementData {
    pub fn facet_count(&self) -> usize {
        match self {
            ElementData::Point => 1,
            ElementData::Edge(_) => 2,
            ElementData::ElementCounts(element_counts) => {
                element_counts.iter().map(|el_count| el_count.count).sum()
            }
        }
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

    pub fn len(&self) -> Float {
        if let ElementData::Edge(len) = self.data {
            len.0
        } else {
            panic!("Expected edge!")
        }
    }
}

// We'll move this over to the translation module... some day.
const EL_NAMES: [&str; 24] = [
    "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta", "Xenna",
    "Daka", "Henda", "Doka", "Tradaka", "Tedaka", "Pedaka", "Exdaka", "Zedaka", "Yodaka", "Nedaka",
    "Ika", "Ikena", "Ikoda", "Iktra",
];

const EL_SUFFIXES: [&str; 24] = [
    "", "", "gon", "hedron", "choron", "teron", "peton", "exon", "zetton", "yotton", "xennon",
    "dakon", "hendon", "dokon", "tradakon", "tedakon", "pedakon", "exdakon", "zedakon", "yodakon",
    "nedakon", "ikon", "ikenon", "ikodon",
];

impl Concrete {
    /// Gets the element "types" of a polytope.
    fn element_types(&self) -> Vec<Vec<ElementType>> {
        use std::collections::hash_map::Entry;

        let rank = self.rank();
        let mut output;
        if let Some(rank_usize) = rank.try_usize() {
            output = Vec::with_capacity(rank_usize);
        } else {
            return Vec::new();
        }

        // There's only one point type, for now.
        output.push(vec![ElementType {
            indices: (0..self.vertex_count()).collect(),
            data: ElementData::Point,
        }]);

        if rank == Rank::new(0) {
            return output;
        }

        // Gets the edge lengths of all edges in the polytope.
        let mut edge_lengths: BTreeMap<FloatOrd, Vec<usize>> = BTreeMap::new();
        let edge_count = self.el_count(Rank::new(1));

        for edge_idx in 0..edge_count {
            let len = FloatOrd::from(self.edge_len(edge_idx).unwrap());

            // Searches for the greatest length smaller than the current one.
            if let Some((prev_len, indices)) = edge_lengths.range_mut(..len).next_back() {
                if abs_diff_eq!(prev_len.0, len.0, epsilon = Float::EPS) {
                    indices.push(edge_idx);
                    continue;
                }
            }

            // Searches for the smallest length greater than the current one.
            if let Some((next_len, indices)) = edge_lengths.range_mut(len..).next() {
                if abs_diff_eq!(next_len.0, len.0, epsilon = Float::EPS) {
                    indices.push(edge_idx);
                    continue;
                }
            }

            edge_lengths.insert(len, vec![edge_idx]);
        }

        // Creates a map from edge indices to indices of types.
        let mut prev_types = Vec::new();
        let mut edge_types = Vec::new();
        prev_types.resize(edge_count, 0);
        for (type_index, (len, indices)) in edge_lengths.into_iter().enumerate() {
            for &idx in &indices {
                prev_types[idx] = type_index;
            }
            edge_types.push(ElementType {
                indices,
                data: ElementData::Edge(len),
            })
        }
        output.push(edge_types);

        for elements in self
            .abs
            .ranks
            .iter()
            .take(self.rank().plus_one_usize())
            .skip(3)
        {
            // A map from element data to the indices of the elements with such data.
            let mut types = HashMap::new();

            // We build the map.
            for (idx, el) in elements.iter().enumerate() {
                let mut element_count_builder = ElementCountBuilder::new();
                for &sub in &el.subs {
                    element_count_builder.insert(prev_types[sub]);
                }
                let data = element_count_builder.build();

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
            prev_types = Vec::new(); // This can probably be taken out.
            prev_types.resize(elements.len(), 0);
            for (type_index, (data, indices)) in types.into_iter().enumerate() {
                for &idx in &indices {
                    prev_types[idx] = type_index;
                }
                element_types.push(ElementType { indices, data });
            }

            output.push(element_types);
        }

        output
    }

    pub fn print_element_types(&self) {
        // An iterator over the element types of each rank.
        let mut type_iter = self.element_types().into_iter().enumerate();

        // Prints points.
        if let Some((r, types)) = type_iter.next() {
            println!("{}", EL_NAMES[r]);
            for t in types {
                println!("{}", t.count());
            }
            println!();
        } else {
            return;
        }

        // Prints edges.
        if let Some((r, types)) = type_iter.next() {
            println!("{}", EL_NAMES[r]);
            for t in types {
                println!("{} × length {}", t.count(), t.len());
            }
            println!();
        } else {
            return;
        }

        // Prints everything else.
        for (d, types) in type_iter {
            println!("{}", EL_NAMES[d]);
            for t in types {
                println!(
                    "{} × {}-{}",
                    t.count(),
                    t.data.facet_count(),
                    EL_SUFFIXES[d]
                );
            }
            println!();
        }
    }
}
