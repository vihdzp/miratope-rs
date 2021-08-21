//! Helpful methods and structs for operating on the [`Flags`](Flag) of a 
//! polytope.

use std::{
    cmp::Ordering,
    collections::{hash_map::Entry, HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    iter,
    ops::Range,
};

use crate::{
    abs::{elements::Ranked, Abstract},
    Float, Polytope,
};

use vec_like::*;

/// Asserts that the subelements and superelements of a polytope are sorted.
/// This runs only on debug mode, and if the assertion fails, it should be
/// considered a serious bug.
fn assert_sorted(p: &Abstract) {
    debug_assert!(
        p.sorted,
        "a polytope's elements must be sorted before iterating over its flags"
    )
}

/// An auxiliary method for [`Flag::change_mut`]. Gets the two common elements
/// of two **sorted** lists.
///
/// # Panic
/// This method will behave erroneously and might panic if the lists are not
/// sorted. Furthermore, the method will panic if the lists have less than two
/// common elements.
fn common<T: AsRef<[usize]>, U: AsRef<[usize]>>(list1: T, list2: U) -> (usize, usize) {
    let list1 = list1.as_ref();
    let list2 = list2.as_ref();
    let mut i = 0;
    let mut j = 0;
    let mut prev = None;

    loop {
        let sub0 = list1[i];
        let sub1 = list2[j];

        match sub0.cmp(&sub1) {
            Ordering::Equal => {
                if let Some(other) = prev {
                    return (sub0, other);
                } else {
                    prev = Some(sub0);
                }

                i += 1;
            }
            Ordering::Greater => j += 1,
            Ordering::Less => i += 1,
        }
    }
}

/// Represents a [flag](https://polytope.miraheze.org/wiki/Flag) in a polytope.
/// Stores the indices of the elements of each rank.
///
/// The minimal element of the flag must always have index 0. However, we keep
/// it in memory since it allows us to not have to special-case the
/// [`Self::change_mut`] method.
#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Flag(Vec<usize>);
impl_veclike!(Flag, Item = usize, Index = usize);

impl Flag {
    /// Applies a specified flag change to the flag in place.
    ///
    /// Recall that an `i`-flag change sends a flag to another that shares all
    /// elements except for the `i`-th one. In a valid (dyadic) polytope, the
    /// resulting flag always exists and is unique.
    ///
    /// For this flag change to be efficiently applied, we need for all of the
    /// element and subelement lists of the polytope to be sorted. This is
    /// verified via various debug assertions.
    pub fn change_mut(&mut self, polytope: &Abstract, r: usize) {
        assert_sorted(polytope);

        // Determines the common elements between the subelements of the element
        // above and the superelements of the element below.
        let below_idx = self[r - 1];
        let below = polytope.get_element(r - 1, below_idx).unwrap();
        let above_idx = self[r + 1];
        let above = polytope.get_element(r + 1, above_idx).unwrap();
        let (c0, c1) = common(&below.sups, &above.subs);

        // Changes the element at idx to the other element in the section
        // determined by the elements above and below.
        if self[r] == c0 {
            self[r] = c1;
        } else {
            self[r] = c0;
        }
    }

    /// Applies a specified flag change to the flag.
    pub fn change(&self, polytope: &Abstract, idx: usize) -> Self {
        let mut clone = self.clone();
        clone.change_mut(polytope, idx);
        clone
    }
}

/// The parity of a flag, which flips on any flag change.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Orientation {
    /// A flag of even parity.
    Even,

    /// A flag of odd parity.
    Odd,
}

impl Orientation {
    /// Flips the parity of a flag.
    pub fn flip(&self) -> Self {
        match self {
            Self::Even => Self::Odd,
            Self::Odd => Self::Even,
        }
    }

    /// Returns the "sign" associated with a flag, which is either `1.0` or
    /// `-1.0`.
    pub fn sign<T: Float>(&self) -> T {
        match self {
            Self::Even => T::ONE,
            Self::Odd => -T::ONE,
        }
    }
}

/// An arbitrary orientation to serve as the default.
impl Default for Orientation {
    fn default() -> Self {
        Self::Even
    }
}

/// An iterator over all [`Flags`](Flag) of a polytope. This iterator works even
/// if the polytope is a compound polytope. However, it is not able to keep
/// track of orientation.
///
/// Each flag is associated with a sequence whose k-th entry stores the index of
/// the k-th element as a subelement of its superelement. We iterate over flags
/// in the lexicographic order given by these sequences.
///
/// You should use this iterator instead of an [`OrientedFlagIter`] when
/// * you don't care about the [`Orientation`] of the flags,
/// * you want to iterate over all flags.
pub struct FlagIter<'a> {
    /// The polytope whose flags we iterate over.
    polytope: &'a Abstract,

    /// The flag we just found, or `None` if we already went through the entire
    /// iterator.
    flag: Option<Flag>,

    /// The indices of each element of the flag, **as subelements of their
    /// superelements.** These indices **do not** coincide with the actual
    /// indices of the elements in their respective `ElementList`s.
    indices: Vec<usize>,
}

impl<'a> FlagIter<'a> {
    /// Initializes an iterator over all flags of a polytope.
    pub fn new(polytope: &'a Abstract) -> Self {
        assert_sorted(polytope);

        Self {
            polytope,
            flag: Some(polytope.first_flag()),
            indices: vec![0; polytope.rank()],
        }
    }
}

impl<'a> Iterator for FlagIter<'a> {
    type Item = Flag;

    fn next(&mut self) -> Option<Self::Item> {
        let flag = self.flag.as_mut()?;
        let prev_flag = flag.clone();
        let rank = self.polytope.rank();

        // The largest rank of the elements we'll update.
        let mut r = 1;
        loop {
            if r >= rank {
                self.flag = None;
                return Some(prev_flag);
            }

            let element_list = &self.polytope[r + 1];
            let idx = flag[r + 1];

            if element_list[idx].subs.len() == self.indices[r] + 1 {
                self.indices[r] = 0;
                r += 1;
            } else {
                self.indices[r] += 1;
                break;
            }
        }

        // Updates all elements in the flag with ranks r down to 0.
        let idx = flag[r + 1];
        let mut element = &self.polytope[(r + 1, idx)];
        loop {
            let idx = self.indices[r];
            flag[r] = element.subs[idx];

            if r == 1 {
                break;
            }

            element = &self.polytope[(r, flag[r])];
            r -= 1;
        }

        Some(prev_flag)
    }
}

/// A flag together with an orientation. Any flag change flips the orientation.
/// If the polytope associated to the flag is non-orientable, the orientation
/// will be garbage data.
///
/// The implementations for traits like `PartialEq` and `Hash` ignore the
/// orientation of the flag.
#[derive(Clone, Debug, Eq)]
pub struct OrientedFlag {
    /// The indices of the elements the flag contains, excluding the null and
    /// maximal elements.
    pub flag: Flag,

    /// The orientation of the flag. If the polytope is non-orientable, this
    /// will contain garbage.
    pub orientation: Orientation,
}

impl_veclike_field!(OrientedFlag, Item = usize, Index = usize, Field = .flag);

/// Makes an oriented flag from a normal flag.
impl From<Flag> for OrientedFlag {
    fn from(flag: Flag) -> Self {
        Self {
            flag,
            orientation: Default::default(),
        }
    }
}

impl From<Vec<usize>> for OrientedFlag {
    fn from(vec: Vec<usize>) -> Self {
        Flag::from(vec).into()
    }
}

impl Hash for OrientedFlag {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.flag.hash(state);
    }
}

impl PartialEq for OrientedFlag {
    fn eq(&self, other: &Self) -> bool {
        self.flag.eq(&other.flag)
    }
}

impl PartialOrd for OrientedFlag {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.flag.partial_cmp(&other.flag)
    }
}

impl Ord for OrientedFlag {
    fn cmp(&self, other: &Self) -> Ordering {
        self.flag.cmp(&other.flag)
    }
}

impl OrientedFlag {
    /// Applies a specified flag change to the flag.
    pub fn change(&self, polytope: &Abstract, idx: usize) -> Self {
        Self {
            flag: self.flag.change(polytope, idx),
            orientation: self.orientation.flip(),
        }
    }
}

/// Represents a set of flag changes. Each flag change is represented by the
/// rank of the element it modifies.
#[derive(Clone)]
pub struct FlagChanges(Vec<usize>);
impl_veclike!(FlagChanges, Item = usize, Index = usize);

impl FlagChanges {
    /// Returns the set of all possible flag changes in a polytope of a given
    /// rank. These are the flag changes of ranks from 1 up to the rank minus 1.
    pub fn all(rank: usize) -> Self {
        Self((1..rank).collect())
    }

    /// Returns an iterator over all subsets of flag changes created by taking
    /// out a single flag change.
    pub fn subsets(&self) -> iter::Map<Range<usize>, impl FnMut(usize) -> Self + '_> {
        (0..self.len()).map(move |i| {
            let mut subset = self.clone();
            subset.remove(i);
            subset
        })
    }
}

/// An iterator over all of the [`FlagEvent`]s of a polytope. A [`FlagEvent`] is
/// either an [`OrientedFlag`], or an event that determines that a polytope is
/// non-orientable.
///
/// **All methods assume that the polytope has been [sorted](Abstract::sort)
/// beforehand.**
///
/// We store a queue of all [`Flags`](Flag) whose adjacencies need to be
/// searched, together with a `HashSet` which store all of the flags that have
/// been found so far. For each element in the queue, we apply all flag changes
/// in a given set to it. All new flags that we find are then returned and added
/// to the queue.
///
/// The reason we don't iterate over flags directly is that sometimes, we
/// realize that a polytope is non-orientable only after traversing every single
/// one of its flags. Hence, we can't bundle the information that the polytope
/// is non-orientable with the flags.
///
/// You should use this iterator instead of a [`FlagIter`] when
/// * you want to apply a specific set of flag changes,
/// * you care about the orientation of the flags.
pub struct OrientedFlagIter<'a> {
    /// The polytope whose flags we iterate over. For the algorithm that applies
    /// a flag change to work, **this polytope's subelement and superelement
    /// lists must be sorted.**
    ///
    /// Some associated methods will guarantee this condition by sorting the
    /// polytope, while others will assume it.
    polytope: &'a Abstract,

    /// The flags whose adjacencies are being searched.
    queue: VecDeque<OrientedFlag>,

    /// The flag changes we're applying.
    flag_changes: FlagChanges,

    /// The flag index we need to check next.
    flag_idx: usize,

    /// Have we already returned the first flag?
    first: bool,

    /// The flags that have already been found, but whose neighbors haven't all
    /// been found yet.
    found: HashMap<OrientedFlag, usize>,

    /// Whether all of the flags the iterator has checked so far have a parity.
    orientable: bool,
}

/// The result of trying to get the next flag.
#[derive(Debug)]
pub enum FlagNext {
    /// We found a new flag event (either a flag or the non-orientable event).
    New(FlagEvent),

    /// We found a flag we had already found before.
    Repeat,

    /// There are no flags left to find.
    None,
}

impl<'a> OrientedFlagIter<'a> {
    /// Initializes a new iterator over the flag events of a polytope, starting
    /// from an arbitrary flag and applying all flag changes.
    ///
    /// You must [sort](Abstract::sort) the polytope before calling this
    /// method.
    pub fn new(polytope: &'a Abstract) -> Self {
        assert_sorted(polytope);

        // Initializes with any flag from the polytope and all flag changes.
        Self::with_flags(
            polytope,
            FlagChanges::all(polytope.rank()),
            polytope.first_oriented_flag(),
        )
    }

    /// Initializes a new iterator over the flag events of a polytope, starting
    /// from a specified flag and applying a given set of flag changes.
    ///
    /// You must [sort](Abstract::sort) the polytope before calling this
    /// method.
    pub fn with_flags(
        polytope: &'a Abstract,
        flag_changes: FlagChanges,
        first_flag: OrientedFlag,
    ) -> Self {
        assert_sorted(polytope);

        // Initializes found flags.
        let mut found = HashMap::new();
        found.insert(first_flag.clone(), 0);

        // Initializes queue.
        let mut queue = VecDeque::new();
        queue.push_back(first_flag);

        Self {
            polytope,
            queue,
            flag_changes,
            flag_idx: 0,
            first: false,
            found,
            orientable: true,
        }
    }

    /// Returns a new iterator over oriented flags, discarding the
    /// non-orientable event.
    pub fn filter_flags(self) -> impl Iterator<Item = OrientedFlag> + 'a {
        self.filter_map(FlagEvent::flag)
    }

    /// Attempts to get the next flag.
    pub fn try_next(&mut self) -> FlagNext {
        // We get the current flag from the queue.
        if let Some(current) = self.queue.front() {
            let rank = self.polytope.rank();

            // Applies the current flag change to the current flag.
            let flag_change = self.flag_changes[self.flag_idx];
            let new_flag = current.change(self.polytope, flag_change);

            // Increments the flag index.
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
                        return FlagNext::New(FlagEvent::NonOrientable);
                    }

                    // In any case, if we got here, we know this is a repeated
                    // flag.
                    //
                    // If we've found it all of the times we'll ever find it,
                    // there's no use in keeping it in the dictionary (profiling
                    // shows this is marginally faster than letting it be).
                    if val == rank {
                        occupied_entry.remove();
                    }

                    FlagNext::Repeat
                }

                // If this flag is new, we just add it and return it.
                Entry::Vacant(vacant_entry) => {
                    let new_flag = vacant_entry.key().clone();
                    self.queue.push_back(new_flag.clone());

                    // We've found the flag one (1) time.
                    vacant_entry.insert(1);

                    FlagNext::New(FlagEvent::Flag(new_flag))
                }
            }
        }
        // The queue is empty.
        else {
            FlagNext::None
        }
    }
}

/// Represents either a new found flag, or the event in which the iterator
/// realizes that the polytope is non-orientable.
#[derive(Debug)]
pub enum FlagEvent {
    /// We found a new flag.
    Flag(OrientedFlag),

    /// We just realized the polytope is non-orientable.
    NonOrientable,
}

impl FlagEvent {
    /// Returns the flag contained in the event, if any.
    pub fn flag(self) -> Option<OrientedFlag> {
        match self {
            Self::Flag(oriented_flag) => Some(oriented_flag),
            Self::NonOrientable => None,
        }
    }

    /// Returns whether `self` matches `Self::NonOrientable`.
    pub fn non_orientable(&self) -> bool {
        matches!(self, Self::NonOrientable)
    }

    /// Returns whether `self` does not match `Self::NonOrientable`.
    pub fn orientable(&self) -> bool {
        !self.non_orientable()
    }
}

impl<'a> Iterator for OrientedFlagIter<'a> {
    type Item = FlagEvent;

    /// Gets the next flag event.
    fn next(&mut self) -> Option<Self::Item> {
        let rank = self.polytope.rank();

        // The first flag is a special case.
        if !self.first {
            self.first = true;

            let flag = Some(FlagEvent::Flag(self.found.keys().next().cloned().unwrap()));

            // If we're dealing with a point, or if we're performing no flag
            // changes, this is the only flag.
            if rank <= 1 || self.flag_changes.is_empty() {
                self.queue = VecDeque::new();
            }

            return flag;
        }

        // Loops until we get a new flag event.
        loop {
            match self.try_next() {
                // We found a new flag event.
                FlagNext::New(flag_event) => {
                    return Some(flag_event);
                }

                // We already exhausted the flag supply.
                FlagNext::None => return None,

                // Repeat flag, try again.
                FlagNext::Repeat => {}
            }
        }
    }
}

/// Represents a set of flags, created by applying a specific set of flag
/// changes to a flag in a polytope.
pub struct FlagSet {
    /// The flags contained in the set.
    pub flags: HashSet<Flag>,

    /// The flag changes from which these flags were generated.
    pub flag_changes: FlagChanges,
}

// THIS IS ONLY MEANT FOR OMNITRUNCATES!!!
impl PartialEq for FlagSet {
    fn eq(&self, other: &Self) -> bool {
        if self.flag_changes.0 != other.flag_changes.0 {
            return false;
        }

        let flag = self.flags.iter().next().unwrap();
        other.flags.contains(flag)
    }
}

impl Eq for FlagSet {}

impl FlagSet {
    /// Creates a new flag set from any flag of the polytope, using all possible
    /// flag changes.
    pub fn new_all(polytope: &Abstract) -> Self {
        Self::with_flags(
            polytope,
            FlagChanges::all(polytope.rank()),
            polytope.first_flag(),
        )
    }

    /// Creates a new flag set defined by all flags in a polytope that can be
    /// obtained by repeatedly applying any in a given set of flag changes to a
    /// specified flag.
    pub fn with_flags(polytope: &Abstract, flag_changes: FlagChanges, first_flag: Flag) -> Self {
        Self {
            flags: OrientedFlagIter::with_flags(polytope, flag_changes.clone(), first_flag.into())
                .filter_flags()
                .map(|oriented_flag| oriented_flag.flag)
                .collect(),
            flag_changes,
        }
    }

    /// Returns `true` if the flag set is empty.
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }

    /// Returns the number of flags contained in the flag set.
    pub fn len(&self) -> usize {
        self.flags.len()
    }

    /// Returns the set of all flag sets obtained from this one after removing
    /// exactly one element.
    // TODO: make into an iterator instead.
    pub fn subsets(&self, polytope: &Abstract) -> Vec<Self> {
        let mut subsets = Vec::new();

        for flag_changes in self.flag_changes.subsets() {
            let mut flags = HashSet::new();

            for flag in &self.flags {
                if flags.insert(flag.clone()) {
                    let subset = Self::with_flags(polytope, flag_changes.clone(), flag.clone());

                    for new_flag in &subset.flags {
                        flags.insert(new_flag.clone());
                    }

                    subsets.push(subset);
                }
            }
        }

        subsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Polytope;

    /// Tests that a polytope has an expected number of flags, oriented or not.
    fn test(polytope: &mut Abstract, expected: usize) {
        let flag_count = polytope.flags_mut().count();
        assert_eq!(
            expected, flag_count,
            "Expected {} flags, found {}.",
            expected, flag_count
        );

        let flag_count = polytope.flag_events().filter_flags().count();
        assert_eq!(
            expected, flag_count,
            "Expected {} oriented flags, found {}.",
            expected, flag_count
        );
    }

    #[test]
    fn nullitope() {
        test(&mut Abstract::nullitope(), 1)
    }

    #[test]
    fn point() {
        test(&mut Abstract::point(), 1)
    }

    #[test]
    fn dyad() {
        test(&mut Abstract::dyad(), 2)
    }

    #[test]
    fn polygon() {
        for n in 2..=10 {
            test(&mut Abstract::polygon(n), 2 * n);
        }
    }

    #[test]
    fn simplex() {
        for n in 1..=8 {
            test(&mut Abstract::simplex(n), crate::factorial(n) as usize);
        }
    }

    #[test]
    fn hypercube() {
        for n in 1..=7 {
            test(
                &mut Abstract::hypercube(n),
                (1 << (n - 1)) * crate::factorial(n - 1) as usize,
            );
        }
    }

    #[test]
    fn orthoplex() {
        for n in 1..=7 {
            test(
                &mut Abstract::orthoplex(n),
                (1 << (n - 1)) * crate::factorial(n - 1) as usize,
            );
        }
    }
}
