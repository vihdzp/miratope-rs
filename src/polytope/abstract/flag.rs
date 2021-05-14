//! Helpful methods and structs for operating on the [flags](https://polytope.miraheze.org/wiki/Flag)
//! of a polytope.

use std::{
    cmp::Ordering,
    collections::{hash_map::Entry, HashMap, VecDeque},
    hash::{Hash, Hasher},
};

use super::{rank::Rank, Abstract};
use crate::Float;

/// The parity of a flag, which flips on any flag change.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Parity {
    /// A flag of even parity.
    Even,

    /// A flag of odd parity.
    Odd,
}

impl Parity {
    /// Flips the parity of a flag.
    pub fn flip_mut(&mut self) {
        match self {
            Parity::Even => *self = Parity::Odd,
            Parity::Odd => *self = Parity::Even,
        }
    }

    /// Returns the "sign" associated with a flag, which is either `1.0` or `-1.0`.
    pub fn sign(&self) -> Float {
        match self {
            Parity::Even => 1.0,
            Parity::Odd => -1.0,
        }
    }
}

impl Default for Parity {
    fn default() -> Self {
        Self::Even
    }
}

#[derive(Clone, Debug)]
pub struct Flag {
    /// The indices of the elements the flag contains, excluding the null and
    /// maximal elements.
    pub elements: Vec<usize>,

    /// The orientation of the flag. If the polytope is non-orientable, this
    /// will contain garbage.
    pub orientation: Parity,
}

impl Hash for Flag {
    /// Returns the hash of the flag. **Does not take orientation into
    /// account.**
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.elements.hash(state);
    }
}

impl PartialEq for Flag {
    /// Determines whether two flags are equal. **Does not take orientation into
    /// account.**
    fn eq(&self, other: &Self) -> bool {
        self.elements.eq(&other.elements)
    }
}

impl PartialOrd for Flag {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.elements.partial_cmp(&other.elements)
    }
}

impl Ord for Flag {
    fn cmp(&self, other: &Self) -> Ordering {
        self.elements.cmp(&other.elements)
    }
}

impl Eq for Flag {}

impl std::ops::Index<usize> for Flag {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl std::ops::IndexMut<usize> for Flag {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl Flag {
    pub fn push(&mut self, value: usize) {
        self.elements.push(value);
    }
}

/// Gets the common elements of two lists. There's definitely a better way.
fn common(vec0: &[usize], vec1: &[usize]) -> Vec<usize> {
    let mut vec0 = vec0.to_owned();
    vec0.sort_unstable();
    let mut vec1 = vec1.to_owned();
    vec1.sort_unstable();

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
    /// Constructs a new, empty `Flag` with the specified capacity.
    pub fn with_capacity(rank: usize) -> Self {
        Self {
            elements: Vec::with_capacity(rank),
            orientation: Parity::default(),
        }
    }

    /// Gets the index of the element stored at a given rank, whilst pretending
    /// that the flag contains a minimal and maximal element.
    pub fn get(&self, rank: Rank) -> Option<&usize> {
        if rank == Rank::new(-1) || rank == Rank::from(self.elements.len()) {
            Some(&0)
        } else {
            self.elements.get(rank.usize())
        }
    }

    /// Applies a specified flag change to the flag in place.
    pub fn change_mut(&mut self, polytope: &Abstract, r: usize) {
        let rank = polytope.rank();
        assert_ne!(
            rank,
            Rank::new(-1),
            "Can't iterate over flags of the nullitope."
        );

        // A flag change is a no-op in a point.
        if rank == Rank::new(0) {
            return;
        }

        let r = Rank::from(r);
        let r_minus_one = r.minus_one();
        let r_plus_one = r.plus_one();

        // Determines the common elements between the subelements of the element
        // above and the superelements of the element below.
        let below = polytope
            .element_ref(r_minus_one, self[r_minus_one])
            .unwrap();
        let above = polytope.element_ref(r_plus_one, self[r_plus_one]).unwrap();
        let common = common(&below.sups.0, &above.subs.0);

        assert_eq!(
            common.len(),
            2,
            "Diamond property fails between rank {}, index {}, and rank {}, index {}.",
            r_minus_one,
            self[r_minus_one],
            r_plus_one,
            self[r_plus_one]
        );

        // Changes the element at idx to the other element in the section
        // determined by the elements above and below.
        if self[r] == common[0] {
            self[r] = common[1];
        } else {
            self[r] = common[0];
        }

        self.orientation.flip_mut();
    }

    /// Applies a specified flag change to the flag.
    pub fn change(&self, polytope: &Abstract, idx: usize) -> Self {
        let mut clone = self.clone();
        clone.change_mut(polytope, idx);
        clone
    }
}

impl std::ops::Index<Rank> for Flag {
    type Output = usize;

    fn index(&self, index: Rank) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl std::ops::IndexMut<Rank> for Flag {
    fn index_mut(&mut self, index: Rank) -> &mut Self::Output {
        self.elements.get_mut(index.usize()).unwrap()
    }
}

/// An iterator over all of the "flag events" of a polytope. A flag event is
/// simply either a [`Flag`], or an event that determines that a polytope is
/// non-orientable.
///
/// The reason we don't iterate over flags directly is that sometimes, we
/// realize a polytope is non-orientable only after traversing every single one
/// of its flags. Hence, we can't bundle the information that the polytope is
/// non-orientable with the flags.
pub struct FlagIter<'a> {
    /// The polytope whose flags are iterated. We must sort all of its elements
    /// before using it for the algorithm to work.
    polytope: &'a Abstract,

    /// The flags whose adjacencies are being searched.
    queue: VecDeque<Flag>,

    flag_changes: Vec<usize>,

    /// The flag index we need to check next.
    flag_idx: usize,

    /// The flags that have already been found, but whose neighbors haven't all
    /// been found yet.
    found: HashMap<Flag, usize>,

    /// Whether all of the flags the iterator has checked so far have a parity.
    orientable: bool,
}

/// The result of trying to get the next flag.
#[derive(Debug)]
pub enum IterResult {
    /// We found a new flag.
    New(Flag),

    /// We found a flag we had already found before.
    Repeat,

    /// We just realized the polytope is non-orientable.
    NonOrientable,

    /// There are no flags left to find.
    None,
}

impl<'a> FlagIter<'a> {
    /// Initializes a new iterator over the flag events of a polytope.
    pub fn new(polytope: &'a Abstract) -> Self {
        let rank = polytope.rank();

        // Initializes with any flag from the polytope.
        let mut first_flag = Flag::with_capacity(rank.usize());
        let mut idx = 0;
        first_flag.elements.push(0);

        for r in Rank::range_iter(Rank::new(1), rank) {
            idx = polytope.element_ref(r.minus_one(), idx).unwrap().sups[0];
            first_flag.elements.push(idx);
        }

        // All flag changes.
        let flag_changes = (0..rank.usize()).collect();

        Self::with_flags(polytope, flag_changes, first_flag)
    }

    pub fn with_flags(polytope: &'a Abstract, flag_changes: Vec<usize>, first_flag: Flag) -> Self {
        assert!(polytope.is_bounded(), "Polytope is not bounded.");
        let rank = polytope.rank();

        // Initializes found flags.
        let mut found = HashMap::new();
        let mut queue = VecDeque::new();

        if rank != Rank::new(-1) {
            found.insert(first_flag.clone(), 0);

            // Initializes queue.
            queue.push_back(first_flag);
        }

        Self {
            polytope,
            queue,
            flag_changes,
            flag_idx: 0,
            found,
            orientable: true,
        }
    }

    pub fn flag_change(&self) -> usize {
        self.flag_changes[self.flag_idx]
    }

    /// Attempts to get the next flag.
    pub fn try_next(&mut self) -> IterResult {
        let rank = self.polytope.rank().usize();
        let new_flag;

        if let Some(current) = self.queue.front() {
            new_flag = current.change(&self.polytope, self.flag_change());

            // Increments flag_idx.
            self.flag_idx = if self.flag_idx + 1 == self.flag_changes.len() {
                self.queue.pop_front();
                0
            } else {
                self.flag_idx + 1
            };
        } else {
            return IterResult::None;
        }

        let new_orientation = new_flag.orientation;
        match self.found.entry(new_flag) {
            // If the flag is already in the found dictionary:
            Entry::Occupied(mut occupied_entry) => {
                *occupied_entry.get_mut() += 1;
                let val = *occupied_entry.get();

                if self.orientable && new_orientation != occupied_entry.key().orientation {
                    self.orientable = false;
                    return IterResult::NonOrientable;
                }

                // In the special case we just found the initial flag again, we
                // return it.
                if val == 1 {
                    let new_flag = occupied_entry.key().clone();

                    // If we've found it all of the times we'll ever find it, no use
                    // in keeping it in the dictionary.
                    if val == rank {
                        occupied_entry.remove();
                    }

                    IterResult::New(new_flag)
                } else {
                    // If we've found it all of the times we'll ever find it, no use
                    // in keeping it in the dictionary.
                    if val == rank {
                        occupied_entry.remove();
                    }

                    // Otherwise, this will always be a repeat.
                    IterResult::Repeat
                }
            }
            // If this flag is new, we just add it and return it.
            Entry::Vacant(vacant_entry) => {
                let new_flag = vacant_entry.key().clone();
                self.queue.push_back(new_flag.clone());
                vacant_entry.insert(1);

                IterResult::New(new_flag)
            }
        }
    }
}

#[derive(PartialEq)]
/// Represents either a new found flag, or the event in which the iterator
/// realizes that the polytope is non-orientable.
pub enum FlagEvent {
    /// We found a new flag.
    Flag(Flag),

    /// We just realized the polytope is non-orientable.
    NonOrientable,
}

impl<'a> Iterator for FlagIter<'a> {
    type Item = FlagEvent;

    fn next(&mut self) -> Option<Self::Item> {
        let rank = self.polytope.rank();

        // A nullitope has no flags.
        if rank == Rank::new(-1) {
            None
        }
        // A point has a single flag.
        else if rank == Rank::new(0) {
            self.queue.pop_front().map(FlagEvent::Flag)
        } else {
            loop {
                match self.try_next() {
                    IterResult::New(f) => {
                        return Some(FlagEvent::Flag(f));
                    }
                    IterResult::NonOrientable => {
                        return Some(FlagEvent::NonOrientable);
                    }
                    IterResult::None => return None,
                    IterResult::Repeat => {}
                }
            }
        }
    }
}
