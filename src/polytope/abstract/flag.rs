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
    // Hopefully this isn't much of a bottleneck.
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
    /// The polytope whose flags are iterated.
    polytope: &'a Abstract,

    /// The flags whose adjacencies are being searched.
    queue: VecDeque<Flag>,

    /// The flag changes we're applying.
    flag_changes: Vec<usize>,

    /// The flag index we need to check next.
    flag_idx: usize,

    /// Have we already returned the first flag?
    first: bool,

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
    /// Returns a dummy iterator that returns `None` every single time.
    pub fn empty(polytope: &'a Abstract) -> Self {
        Self {
            polytope,
            queue: VecDeque::new(), // This is the important bit.
            flag_changes: Vec::new(),
            flag_idx: 0,
            first: true,
            found: HashMap::new(),
            orientable: true,
        }
    }

    /// Initializes a new iterator over the flag events of a polytope.
    pub fn new(polytope: &'a Abstract) -> Self {
        use crate::polytope::Polytope;

        let rank = polytope.rank();

        // Initializes with any flag from the polytope.
        if let Some(first_flag) = polytope.some_flag() {
            // All flag changes.
            let flag_changes = (0..rank.usize()).collect();

            Self::with_flags(polytope, flag_changes, first_flag)
        }
        // A nullitope has no flags.
        else {
            Self::empty(polytope)
        }
    }

    /// Initializes a new iterator over the flag events of a polytope, given an
    /// initial flag and a set of flag changes to apply.
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
            first: rank == Rank::new(-1),
            found,
            orientable: true,
        }
    }

    /// Returns the index of the current flag change to apply.
    pub fn flag_change(&self) -> usize {
        self.flag_changes[self.flag_idx]
    }

    /// Attempts to get the next flag.
    pub fn try_next(&mut self) -> IterResult {
        if let Some(current) = self.queue.front() {
            let rank = self.polytope.rank().usize();
            let new_flag = current.change(&self.polytope, self.flag_change());

            // Increments flag_idx.
            self.flag_idx = if self.flag_idx + 1 == self.flag_changes.len() {
                self.queue.pop_front();
                0
            } else {
                self.flag_idx + 1
            };

            let new_orientation = new_flag.orientation;
            match self.found.entry(new_flag) {
                // If the flag is already in the found dictionary:
                Entry::Occupied(mut occupied_entry) => {
                    *occupied_entry.get_mut() += 1;
                    let val = *occupied_entry.get();

                    // If there's a mismatch between the seen and the expected
                    // orientability, then we know the polytope isn't orientable.
                    if self.orientable && new_orientation != occupied_entry.key().orientation {
                        self.orientable = false;
                        return IterResult::NonOrientable;
                    }

                    // In any case, if we got here, we know this is a repeated flag.
                    //
                    // If we've found it all of the times we'll ever find it,
                    // there's no use in keeping it in the dictionary (profiling
                    // shows this is marginally faster than letting it be).
                    if val == rank {
                        occupied_entry.remove();
                    }

                    IterResult::Repeat
                }
                // If this flag is new, we just add it and return it.
                Entry::Vacant(vacant_entry) => {
                    let new_flag = vacant_entry.key().clone();
                    self.queue.push_back(new_flag.clone());

                    // We've found the flag one (1) time.
                    vacant_entry.insert(1);

                    IterResult::New(new_flag)
                }
            }
        }
        // The queue is empty.
        else {
            return IterResult::None;
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

    /// Gets the next flag event.
    fn next(&mut self) -> Option<Self::Item> {
        let rank = self.polytope.rank();

        // The first flag is a special case.
        if !self.first {
            self.first = true;

            let flag = Some(FlagEvent::Flag(self.found.keys().next().cloned().unwrap()));

            // If we're dealing with a point, this is the only flag.
            if rank == Rank::new(0) {
                self.queue = VecDeque::new();
            }

            return flag;
        }

        // Loops until we get a new flag event.
        loop {
            match self.try_next() {
                // We found a new flag.
                IterResult::New(f) => {
                    return Some(FlagEvent::Flag(f));
                }

                // We just realized the polytope is non-orientable.
                IterResult::NonOrientable => {
                    return Some(FlagEvent::NonOrientable);
                }

                // We already exhausted the flag supply.
                IterResult::None => return None,

                // Repeat flag, try again.
                IterResult::Repeat => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polytope::Polytope;

    #[test]
    fn nullitope() {
        let flag_count = Abstract::nullitope().flags().count();
        assert_eq!(flag_count, 0, "Expected {} flags, found {}.", 0, flag_count);
    }

    #[test]
    fn point() {
        let flag_count = Abstract::point().flags().count();
        assert_eq!(flag_count, 1, "Expected {} flags, found {}.", 1, flag_count);
    }

    #[test]
    fn dyad() {
        let flag_count = Abstract::dyad().flags().count();
        assert_eq!(flag_count, 2, "Expected {} flags, found {}.", 2, flag_count);
    }

    #[test]
    fn polygon() {
        for n in 2..=10 {
            let flag_count = Abstract::polygon(n).flags().count();
            assert_eq!(
                flag_count,
                2 * n,
                "Expected {} flags, found {}.",
                2 * n,
                flag_count
            );
        }
    }

    #[test]
    fn simplex() {
        use factorial::Factorial;

        for n in 0..=5 {
            let flag_count = Abstract::simplex(Rank::new(n as isize)).flags().count();
            let expected = (n + 1).factorial();

            assert_eq!(
                flag_count, expected,
                "Expected {} flags, found {}.",
                expected, flag_count
            );
        }
    }

    #[test]
    fn hypercube() {
        use factorial::Factorial;

        for n in 0..=5 {
            let flag_count = Abstract::hypercube(Rank::new(n as isize)).flags().count();
            let expected = (2u32.pow(n as u32) as usize) * n.factorial();

            assert_eq!(
                flag_count, expected,
                "Expected {} flags, found {}.",
                expected, flag_count
            );
        }
    }

    #[test]
    fn orthoplex() {
        use factorial::Factorial;

        for n in 0..=5 {
            let flag_count = Abstract::orthoplex(Rank::new(n as isize)).flags().count();
            let expected = (2u32.pow(n as u32) as usize) * n.factorial();

            assert_eq!(
                flag_count, expected,
                "Expected {} flags, found {}.",
                expected, flag_count
            );
        }
    }
}
