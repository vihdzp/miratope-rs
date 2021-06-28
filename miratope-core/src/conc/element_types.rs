//! The code used to tally up the "element types" in a polytope.

use std::collections::HashMap;

use crate::{
    abs::{
        elements::ElementRef,
        rank::{Rank, RankVec},
    },
    conc::Concrete,
    Polytope,
};

use vec_like::*;

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash)]

struct ElementType {
    example: usize,
    count: usize,
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
    fn element_types(&self) -> RankVec<Vec<ElementType>> {
        let mut type_of_element = RankVec::<Vec<usize>>::new();
        let mut types = RankVec::<Vec<ElementType>>::new();
        let mut type_counts = RankVec::<usize>::new();
        for r in Rank::range_iter(-1, self.rank()) {
            type_of_element.push(vec![0; self.el_count(r)]);
            types.push(Vec::<ElementType>::new());
            type_counts.push(1);
        }

        let mut _passes = 0;
        let mut number_of_types = 0;
        loop {
            for r in Rank::range_iter(1, self.rank()) {
                let mut types_rank: Vec<ElementType> = Vec::<ElementType>::new();
                let mut dict: HashMap<(usize, Vec<usize>), usize> = HashMap::new();
                let mut c = 0;

                for (i, el) in self[r].iter().enumerate() {
                    let mut sub_types = vec![0; type_counts[r.minus_one()]];

                    for sub in el.subs.iter() {
                        let sub_type = type_of_element[r.minus_one()][*sub];
                        sub_types[sub_type] += 1;
                    }

                    let my_type = type_of_element[r][i];
                    match dict.get(&(my_type, sub_types.clone())) {
                        Some(idx) => {
                            type_of_element[r][i] = *idx;
                            types_rank[*idx].count += 1;
                        }
                        None => {
                            dict.insert((my_type, sub_types), c);
                            type_of_element[r][i] = c;
                            types_rank.push(ElementType {
                                example: i,
                                count: 1,
                            });
                            c += 1;
                        }
                    }
                }
                type_counts[r] = c;
                types[r] = types_rank;
            }

            for r in Rank::range_iter(0, self.rank().minus_one()).rev() {
                let mut types_rank: Vec<ElementType> = Vec::new();
                let mut dict: HashMap<(usize, Vec<usize>), usize> = HashMap::new();
                let mut c = 0;

                for (i, el) in self[r].iter().enumerate() {
                    let mut sup_types = vec![0; type_counts[r.plus_one()]];

                    for sup in el.sups.iter() {
                        let sup_type = type_of_element[r.plus_one()][*sup];
                        sup_types[sup_type] += 1;
                    }

                    let my_type = type_of_element[r][i];
                    match dict.get(&(my_type, sup_types.clone())) {
                        Some(idx) => {
                            type_of_element[r][i] = *idx;
                            types_rank[*idx].count += 1;
                        }
                        None => {
                            dict.insert((my_type, sup_types), c);
                            type_of_element[r][i] = c;
                            types_rank.push(ElementType {
                                example: i,
                                count: 1,
                            });
                            c += 1;
                        }
                    }
                }
                type_counts[r] = c;
                types[r] = types_rank;
            }
            let number_of_types_new: usize = type_counts.iter().sum();
            if number_of_types_new == number_of_types {
                break;
            } else {
                number_of_types = number_of_types_new;
            }
            _passes += 1;
        }
        types
    }

    /// Prints all element types of a polytope into the console.
    pub fn print_element_types(&self) {
        // An iterator over the element types of each rank.
        let type_iter = self.element_types().into_iter().skip(1).enumerate();

        for (r, types) in type_iter {
            println!("{}", EL_NAMES[r]);
            for t in types {
                let i = t.example;
                println!(
                    "{} × {}-{} , {}-{}",
                    t.count,
                    self.abs
                        .get_element(ElementRef {
                            rank: r.into(),
                            idx: i
                        })
                        .unwrap()
                        .subs
                        .len(),
                    EL_SUFFIXES[r],
                    self.abs
                        .get_element(ElementRef {
                            rank: r.into(),
                            idx: i
                        })
                        .unwrap()
                        .sups
                        .len(),
                    EL_SUFFIXES[self.rank().into_usize() - r - 1],
                );
            }
            println!();
        }
    }
}
