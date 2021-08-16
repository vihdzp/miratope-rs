//! The code used to tally up the "element types" in a polytope.

use std::collections::HashMap;

use crate::{
    abs::{ElementMap, Ranked},
    conc::Concrete,
};

use crate::Float;
use vec_like::*;

/// Every element in a polytope can be assigned a "type" depending on its
/// attributes. This struct stores a representative of a single type of
/// elements.
#[derive(PartialOrd, Ord, PartialEq, Eq, Hash)]
struct ElementType {
    /// The index of the representative for this element type.
    example: usize,

    /// The number of elements of this type.
    count: usize,
}

/// Stores the metadata associated with an element type.
#[derive(PartialEq, Eq, Hash)]
struct TypeData {
    /// The index of the type that this element held in the last pass.
    prev_index: usize,

    /// The indices of the types of either the subelements or superelements,
    /// depending on what part of the algorithm we're in.
    type_counts: Vec<usize>,
}

// We'll move this over to the translation module... some day.
const EL_NAMES: [&str; 24] = [
    "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta", "Xenna",
    "Daka", "Henda", "Doka", "Tradaka", "Tedaka", "Pedaka", "Exdaka", "Zedaka", "Yodaka", "Nedaka",
    "Ika", "Ikena", "Ikoda", "Iktra",
];

const EL_SUFFIXES: [&str; 24] = [
    "", "telon", "gon", "hedron", "choron", "teron", "peton", "exon", "zetton", "yotton", "xennon",
    "dakon", "hendon", "dokon", "tradakon", "tedakon", "pedakon", "exdakon", "zedakon", "yodakon",
    "nedakon", "ikon", "ikenon", "ikodon",
];

impl<T: Float> Concrete<T> {
    /*  element type of an element is <index>
    - initialize all elements to <0>
    - repeat:
    - iterate over ranks:
        - start an indexed hashmap that'll store metadata
        - iterate over elements:
        - get a vector where indexes are the previous rank's type indexes and values are numbers of subelements of that type
        - get its metadata - <current index, vector^> - could add heuristics like edge lengths to this metadata sometimes
        - if metadata matches one already in the hashmap, give it that index,
            - if not, add a new entry in hashmap and increment index
    - iterate over ranks backwards, use superelements instead of subelements
    - get number of types in total, if it's the same as previous loop, stop
    */
    fn element_types(&self) -> ElementMap<ElementType> {
        let rank = self.rank();

        // Stores the different types, the counts of each, and the indices of
        // the types associated to each element.
        let mut types = ElementMap::new();
        let mut type_counts = Vec::new();
        let mut type_of_element = ElementMap::new();

        // Initializes every element with the zeroth type.
        for el_count in self.el_count_iter() {
            type_of_element.push(vec![0; el_count]);
            types.push(Vec::new());
            type_counts.push(1);
        }

        let mut type_count = rank;

        // To limit the number of passes, we can turn this into a `for` loop.
        loop {
            // We build element types from the bottom up.
            for r in 1..rank {
                // All element types of this rank.
                let mut types_rank: Vec<ElementType> = Vec::new();
                let mut dict = HashMap::new();

                for (i, el) in self[r].iter().enumerate() {
                    let mut sub_type_counts = vec![0; type_counts[r - 1]];

                    for &sub in el.subs.iter() {
                        let sub_type = type_of_element[r - 1][sub];
                        sub_type_counts[sub_type] += 1;
                    }

                    let type_data = TypeData {
                        prev_index: type_of_element[r][i],
                        type_counts: sub_type_counts,
                    };

                    match dict.get(&type_data) {
                        // This is an existing element type.
                        Some(&type_idx) => {
                            type_of_element[r][i] = type_idx;
                            types_rank[type_idx].count += 1;
                        }

                        // This is a new element type.
                        None => {
                            dict.insert(type_data, types_rank.len());
                            type_of_element[r][i] = types_rank.len();
                            types_rank.push(ElementType {
                                example: i,
                                count: 1,
                            });
                        }
                    }
                }

                type_counts[r] = types_rank.len();
                types[r] = types_rank;
            }

            // We do basically the same thing, from the top down.
            for r in (1..rank).rev() {
                // All element types of this rank.
                let mut types_rank: Vec<ElementType> = Vec::new();
                let mut dict = HashMap::new();

                for (i, el) in self[r].iter().enumerate() {
                    let mut sup_type_counts = vec![0; type_counts[r + 1]];

                    for &sup in el.sups.iter() {
                        let sup_type = type_of_element[r + 1][sup];
                        sup_type_counts[sup_type] += 1;
                    }

                    let type_data = TypeData {
                        prev_index: type_of_element[r][i],
                        type_counts: sup_type_counts,
                    };

                    match dict.get(&type_data) {
                        // This is an existing element type.
                        Some(&type_idx) => {
                            type_of_element[r][i] = type_idx;
                            types_rank[type_idx].count += 1;
                        }

                        // This is a new element type.
                        None => {
                            dict.insert(type_data, types_rank.len());
                            type_of_element[r][i] = types_rank.len();
                            types_rank.push(ElementType {
                                example: i,
                                count: 1,
                            });
                        }
                    }
                }

                type_counts[r] = types_rank.len();
                types[r] = types_rank;
            }

            let new_type_count: usize = type_counts.iter().sum();
            if new_type_count == type_count {
                break;
            }

            type_count = new_type_count;
        }

        types
    }

    /// Prints all element types of a polytope into the console.
    pub fn print_element_types(&self) {
        // An iterator over the element types of each rank.
        let type_iter = self.element_types().into_iter().skip(1).enumerate();

        for (r, types) in type_iter {
            if r == self.rank() {
                println!();
                break;
            }

            println!("{}", EL_NAMES[r]);
            for t in types {
                let i = t.example;
                println!(
                    "{} × {}-{}, {}-{}",
                    t.count,
                    self.abs.get_element(r, i).unwrap().subs.len(),
                    EL_SUFFIXES[r],
                    self.abs.get_element(r, i).unwrap().sups.len(),
                    EL_SUFFIXES[self.rank() - r - 1],
                );
            }
            println!();
        }
    }
}
