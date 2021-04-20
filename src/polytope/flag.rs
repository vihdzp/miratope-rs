use std::collections::{hash_map::Entry, VecDeque};

use std::{cmp::Ordering, collections::HashMap};

use super::Abstract;
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum Orientation {
    Even,
    Odd,
    Either,
}

impl Orientation {
    pub fn flip_mut(&mut self) {
        match self {
            Orientation::Even => *self = Orientation::Odd,
            Orientation::Odd => *self = Orientation::Even,
            Orientation::Either => {}
        }
    }

    pub fn sign(&self) -> f64 {
        match self {
            Orientation::Even => 1.0,
            Orientation::Odd => -1.0,
            Orientation::Either => 0.0,
        }
    }
}

impl Default for Orientation {
    fn default() -> Self {
        Self::Even
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Flag {
    pub elements: Vec<usize>,
    pub orientation: Orientation,
}

/// Gets the common elements of two **sorted** lists.
fn common(vec0: &[usize], vec1: &[usize]) -> Vec<usize> {
    let mut common = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while let Some(&sub0) = vec0.get(i) {
        if let Some(sub1) = vec1.get(j) {
            match sub0.cmp(sub1) {
                Ordering::Equal => {
                    common.push(sub0);
                    i += 1;
                }
                Ordering::Greater => j += 1,
                Ordering::Less => i += 1,
            };
        } else {
            break;
        }
    }

    common
}

impl Flag {
    pub fn with_capacity(rank: isize) -> Self {
        Self {
            elements: Vec::with_capacity(rank as usize),
            orientation: Orientation::default(),
        }
    }

    pub fn get(&self, rank: isize) -> Option<&usize> {
        if rank == -1 || rank == self.elements.len() as isize {
            Some(&0)
        } else {
            self.elements.get(rank as usize)
        }
    }

    pub fn change_mut(&mut self, polytope: &Abstract, idx: usize) {
        assert!(polytope.rank() >= 1);

        let idx = idx as isize;
        let below = polytope.get_element(idx - 1, self[idx - 1]).unwrap();
        let above = polytope.get_element(idx + 1, self[idx + 1]).unwrap();
        let common = common(&below.sups, &above.subs);

        assert_eq!(common.len(), 2);

        if self[idx] == common[0] {
            self[idx] = common[1];
        } else {
            self[idx] = common[0];
        }

        self.orientation.flip_mut();
    }

    pub fn change(&self, polytope: &Abstract, idx: usize) -> Self {
        let mut clone = self.clone();
        clone.change_mut(polytope, idx);
        clone
    }
}

impl std::ops::Index<isize> for Flag {
    type Output = usize;

    fn index(&self, index: isize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl std::ops::IndexMut<isize> for Flag {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        &mut self.elements[index as usize]
    }
}

pub struct FlagIter {
    /// The polytope whose flags are iterated.
    polytope: Abstract,

    /// The flags whose adjacencies are being searched.
    queue: VecDeque<Flag>,

    /// The flag index we need to check next.
    flag_idx: usize,

    /// The flags that have already been found, but whose neighbors haven't all
    /// been found yet.
    found: HashMap<Flag, usize>,
}

pub enum IterResult {
    New(Flag),
    Repeat,
    None,
}

impl FlagIter {
    pub fn new(polytope: &Abstract) -> Self {
        assert_ne!(
            polytope.rank(),
            -1,
            "Can't iterate over the flags of the nullitope!"
        );

        let rank = polytope.rank();

        // The polytope's elements must be sorted! We thus clone the polytope
        // and sort its elements.
        let mut polytope = polytope.clone();
        for elements in polytope.iter_mut() {
            for el in elements.iter_mut() {
                el.subs.sort_unstable();
                el.sups.sort_unstable();
            }
        }

        // Initializes with any flag from the polytope.
        let mut flag = Flag::with_capacity(rank);
        let mut idx = 0;
        flag.elements.push(0);
        for r in 1..rank {
            idx = polytope.get_element(r, idx).unwrap().sups[0];
            flag.elements.push(idx);
        }

        // Initializes found flags.
        let mut found = HashMap::new();
        found.insert(flag.clone(), 0);

        let mut queue = VecDeque::new();
        queue.push_front(flag);

        Self {
            polytope,
            queue,
            flag_idx: 0,
            found,
        }
    }

    pub fn rank(&self) -> usize {
        self.polytope.rank() as usize
    }

    pub fn try_next(&mut self) -> IterResult {
        let rank = self.rank();
        let new_flag;

        if let Some(current) = self.queue.back() {
            new_flag = current.change(&self.polytope, self.flag_idx);

            // Increments flag_idx.
            self.flag_idx = if self.flag_idx + 1 == rank {
                self.queue.pop_back();
                0
            } else {
                self.flag_idx + 1
            };
        } else {
            return IterResult::None;
        }

        // Adds the new flag if not yet found, updates it if found, and removes
        // it if it's been found all of the times it will ever be found.
        match self.found.entry(new_flag.clone()) {
            Entry::Occupied(mut occupied_entry) => {
                *occupied_entry.get_mut() += 1;
                let val = *occupied_entry.get();

                if val == rank {
                    occupied_entry.remove();
                }

                if val == 1 {
                    IterResult::New(new_flag)
                } else {
                    IterResult::Repeat
                }
            }
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(1);
                self.queue.push_front(new_flag.clone());

                IterResult::New(new_flag)
            }
        }
    }
}

impl Iterator for FlagIter {
    type Item = Flag;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.try_next() {
                IterResult::New(f) => return Some(f),
                IterResult::Repeat => {}
                IterResult::None => return None,
            }
        }
    }
}
