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
    abs::{ranked::Ranked, Abstract},
    Polytope,
};

use vec_like::*;

/// Asserts that the subelements and superelements of a polytope are sorted.
///
/// This is not an expensive test since this metadata is stored in the polytope,
/// but should still be avoided in methods that will be repeatedly called.
fn assert_sorted(p: &Abstract) {
    assert!(
        p.sorted(),
        "a polytope's elements must be sorted before iterating over its flags"
    )
}

/// An auxiliary method for [`Flag::change_mut`]. Gets the two common elements
/// of two **sorted** lists.
///
/// This algorithm is basically a modified version of the merging step in
/// mergesort.
///
/// # Panics
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
        // We could make these unchecked if we knew for a fact that there are
        // two common elements.
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Flag(Vec<usize>);
impl_veclike!(Flag, Item = usize);

impl Flag {
    /// Applies a specified flag change to the flag in place.
    ///
    /// Recall that an `i`-flag change sends a flag to another that shares all
    /// elements except for the `i`-th one. In a valid (dyadic) polytope, the
    /// resulting flag always exists and is unique.
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
    /// Further, flag changes only really make sense from rank 1 up to the rank
    /// of the polytope minus 1.
    pub fn change_mut(&mut self, polytope: &Abstract, r: usize) {
        // This is running in a hot loop. Maybe get rid of it?
        if cfg!(debug_assertions) {
            assert_sorted(polytope);
        }
        debug_assert!(r >= 1);

        // Determines the common elements between the subelements of the element
        // above and the superelements of the element below.
        let below_idx = self[r - 1];
        let below = &polytope[(r - 1, below_idx)];
        let above_idx = self[r + 1];
        let above = &polytope[(r + 1, above_idx)];
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
    pub fn sign(&self) -> f64 {
        match self {
            Self::Even => 1.0,
            Self::Odd => -1.0,
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

impl_veclike_field!(OrientedFlag, Item = usize, Field = .flag);

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
impl_veclike!(FlagChanges, Item = usize);

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
/// **All methods assume that the polytope has been [sorted](Abstract::element_sort)
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
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
    pub fn new(polytope: &'a Abstract) -> Self {
        Self::with_flags(
            polytope,
            FlagChanges::all(polytope.rank()),
            polytope.first_oriented_flag(),
        )
    }

    /// Initializes a new iterator over the flag events of a polytope, starting
    /// from a specified flag and applying a given set of flag changes.
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
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

    /// Returns whether `self` does not match `Self::NonOrientable`.
    pub fn orientable(&self) -> bool {
        !matches!(self, Self::NonOrientable)
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
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
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
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
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
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
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
    use crate::{Polytope, conc::Concrete, file::FromFile};

    /// Tests that a polytope has an expected number of flags, oriented or not.
    fn test_flags(polytope: &mut Abstract, expected: usize) {
        polytope.element_sort();

        assert_eq!(expected, polytope.flags().count(), "flag count mismatch");

        assert_eq!(
            expected,
            polytope.flag_events().filter_flags().count(),
            "oriented flag count mismatch"
        );
    }

    /// Checks a nullitope's flags.
    #[test]
    fn nullitope() {
        test_flags(&mut Abstract::nullitope(), 1)
    }

    /// Checks a point's flags.
    #[test]
    fn point() {
        test_flags(&mut Abstract::point(), 1)
    }

    /// Checks a dyad's flags.
    #[test]
    fn dyad() {
        test_flags(&mut Abstract::dyad(), 2)
    }

    /// Checks some polygons' flags.
    #[test]
    fn polygon() {
        for n in 2..=10 {
            test_flags(&mut Abstract::polygon(n), 2 * n);
        }
    }

    /// Checks some simplexes' flags.
    #[test]
    fn simplex() {
        let mut simplex = Abstract::nullitope();

        for n in 1..=8 {
            simplex = simplex.pyramid();
            test_flags(&mut Abstract::simplex(n), crate::factorial(n) as usize);
        }
    }

    /// The expected number of flags in an *n*-hypercube.
    fn hypercube_expected(n: usize) -> usize {
        (crate::factorial(n - 1) as usize) << (n - 1)
    }

    /// Checks some hypercubes' flags.
    #[test]
    fn hypercube() {
        let mut hypercube = Abstract::point();

        for n in 2..=7 {
            hypercube = hypercube.prism();
            test_flags(&mut hypercube, hypercube_expected(n));
        }
    }

    /// Checks some orthoplices' flags.
    #[test]
    fn orthoplex() {
        let mut orthoplex = Abstract::point();

        for n in 2..=7 {
            orthoplex = orthoplex.tegum();
            test_flags(&mut orthoplex, hypercube_expected(n));
        }
    }

    /// Checks some polyhedra's flags.
    #[test]
    fn polyhedra() {
        let mut thah = Concrete::from_off("
            OFF
            6 7 12

            # Vertices
            0.7071067811865475 0.0 0.0
            -0.7071067811865475 0.0 0.0
            0.0 0.7071067811865475 0.0
            0.0 -0.7071067811865475 0.0
            0.0 0.0 0.7071067811865475
            0.0 0.0 -0.7071067811865475

            # Faces
            3 0 2 4
            3 3 4 1
            3 2 1 5
            3 5 3 0
            4 0 3 1 2
            4 4 0 5 1
            4 2 4 3 5
            ").unwrap().abs;

        let mut co = Concrete::from_off("
            OFF
            12 14 24

            # Vertices
            0.7071067811865475 0.7071067811865475 0.0
            -0.7071067811865475 0.7071067811865475 0.0
            0.7071067811865475 -0.7071067811865475 0.0
            -0.7071067811865475 -0.7071067811865475 0.0
            0.7071067811865475 0.0 0.7071067811865475
            0.7071067811865475 0.0 -0.7071067811865475
            -0.7071067811865475 0.0 0.7071067811865475
            -0.7071067811865475 0.0 -0.7071067811865475
            0.0 0.7071067811865475 0.7071067811865475
            0.0 0.7071067811865475 -0.7071067811865475
            0.0 -0.7071067811865475 0.7071067811865475
            0.0 -0.7071067811865475 -0.7071067811865475

            # Faces
            4 10 4 8 6
            3 4 0 8
            4 4 2 5 0
            4 8 1 9 0
            3 9 0 5
            4 10 3 11 2
            3 4 2 10
            4 1 6 3 7
            3 8 6 1
            3 10 3 6
            3 9 7 1
            3 5 11 2
            3 7 11 3
            4 9 5 11 7
            ").unwrap().abs;

        let mut snic = Concrete::from_off("
            OFF
            24 38 60
            
            # Vertices
            -0.6212264105565853 0.3377539738137524 1.1426135089259621
            0.6212264105565853 -0.3377539738137524 1.1426135089259621
            0.6212264105565853 0.3377539738137524 -1.1426135089259621
            -0.6212264105565853 -0.3377539738137524 -1.1426135089259621
            -0.3377539738137524 1.1426135089259621 0.6212264105565853
            0.3377539738137524 -1.1426135089259621 0.6212264105565853
            0.3377539738137524 1.1426135089259621 -0.6212264105565853
            -0.3377539738137524 -1.1426135089259621 -0.6212264105565853
            -1.1426135089259621 0.6212264105565853 0.3377539738137524
            1.1426135089259621 -0.6212264105565853 0.3377539738137524
            1.1426135089259621 0.6212264105565853 -0.3377539738137524
            -1.1426135089259621 -0.6212264105565853 -0.3377539738137524
            0.3377539738137524 0.6212264105565853 1.1426135089259621
            -0.3377539738137524 -0.6212264105565853 1.1426135089259621
            -0.3377539738137524 0.6212264105565853 -1.1426135089259621
            0.3377539738137524 -0.6212264105565853 -1.1426135089259621
            0.6212264105565853 1.1426135089259621 0.3377539738137524
            -0.6212264105565853 -1.1426135089259621 0.3377539738137524
            -0.6212264105565853 1.1426135089259621 -0.3377539738137524
            0.6212264105565853 -1.1426135089259621 -0.3377539738137524
            1.1426135089259621 0.3377539738137524 0.6212264105565853
            -1.1426135089259621 -0.3377539738137524 0.6212264105565853
            -1.1426135089259621 0.3377539738137524 -0.6212264105565853
            1.1426135089259621 -0.3377539738137524 -0.6212264105565853
            
            # Faces
            3 1 12 20
            3 12 16 20
            3 4 16 12
            4 1 13 0 12
            3 4 0 12
            3 10 20 16
            4 16 6 18 4
            3 6 10 16
            3 4 8 0
            3 8 18 4
            3 13 0 21
            3 21 8 0
            3 8 22 18
            4 8 21 11 22
            3 21 17 11
            3 17 21 13
            3 13 5 1
            3 17 5 13
            3 1 9 20
            3 5 9 1
            4 9 23 10 20
            4 17 5 19 7
            3 17 7 11
            3 11 3 22
            3 7 11 3
            3 6 14 18
            3 22 14 18
            3 14 3 22
            3 2 10 6
            3 14 2 6
            3 5 19 9
            3 9 23 19
            3 7 15 19
            3 3 15 7
            4 14 2 15 3
            3 15 23 19
            3 23 2 10
            3 15 23 2
            ").unwrap().abs;
        
        let mut ti = Concrete::from_off("
            OFF
            60 32 90
            
            # Vertices
            2.4270509831248424 0.0 0.5
            -2.4270509831248424 0.0 0.5
            2.4270509831248424 0.0 -0.5
            -2.4270509831248424 0.0 -0.5
            0.0 0.5 2.4270509831248424
            0.0 0.5 -2.4270509831248424
            0.0 -0.5 2.4270509831248424
            0.0 -0.5 -2.4270509831248424
            0.5 2.4270509831248424 0.0
            0.5 -2.4270509831248424 0.0
            -0.5 2.4270509831248424 0.0
            -0.5 -2.4270509831248424 0.0
            1.0 2.118033988749895 0.8090169943749475
            -1.0 2.118033988749895 0.8090169943749475
            1.0 -2.118033988749895 0.8090169943749475
            1.0 2.118033988749895 -0.8090169943749475
            -1.0 -2.118033988749895 0.8090169943749475
            -1.0 2.118033988749895 -0.8090169943749475
            1.0 -2.118033988749895 -0.8090169943749475
            -1.0 -2.118033988749895 -0.8090169943749475
            2.118033988749895 0.8090169943749475 1.0
            2.118033988749895 -0.8090169943749475 1.0
            -2.118033988749895 0.8090169943749475 1.0
            2.118033988749895 0.8090169943749475 -1.0
            2.118033988749895 -0.8090169943749475 -1.0
            -2.118033988749895 -0.8090169943749475 1.0
            -2.118033988749895 0.8090169943749475 -1.0
            -2.118033988749895 -0.8090169943749475 -1.0
            0.8090169943749475 1.0 2.118033988749895
            -0.8090169943749475 1.0 2.118033988749895
            0.8090169943749475 -1.0 2.118033988749895
            0.8090169943749475 1.0 -2.118033988749895
            -0.8090169943749475 -1.0 2.118033988749895
            -0.8090169943749475 1.0 -2.118033988749895
            0.8090169943749475 -1.0 -2.118033988749895
            -0.8090169943749475 -1.0 -2.118033988749895
            0.5 1.8090169943749475 1.618033988749895
            -0.5 1.8090169943749475 1.618033988749895
            0.5 -1.8090169943749475 1.618033988749895
            0.5 1.8090169943749475 -1.618033988749895
            -0.5 -1.8090169943749475 1.618033988749895
            -0.5 1.8090169943749475 -1.618033988749895
            0.5 -1.8090169943749475 -1.618033988749895
            -0.5 -1.8090169943749475 -1.618033988749895
            1.8090169943749475 1.618033988749895 0.5
            1.8090169943749475 1.618033988749895 -0.5
            1.8090169943749475 -1.618033988749895 0.5
            -1.8090169943749475 1.618033988749895 0.5
            1.8090169943749475 -1.618033988749895 -0.5
            -1.8090169943749475 -1.618033988749895 0.5
            -1.8090169943749475 1.618033988749895 -0.5
            -1.8090169943749475 -1.618033988749895 -0.5
            1.618033988749895 0.5 1.8090169943749475
            -1.618033988749895 0.5 1.8090169943749475
            1.618033988749895 -0.5 1.8090169943749475
            1.618033988749895 0.5 -1.8090169943749475
            -1.618033988749895 -0.5 1.8090169943749475
            -1.618033988749895 0.5 -1.8090169943749475
            1.618033988749895 -0.5 -1.8090169943749475
            -1.618033988749895 -0.5 -1.8090169943749475
            
            # Faces
            6 29 4 6 32 56 53
            6 28 52 54 30 6 4
            5 4 28 36 37 29
            5 32 40 38 30 6
            6 40 16 11 9 14 38
            6 56 25 49 16 40 32
            5 1 25 56 53 22
            6 47 22 53 29 37 13
            6 13 10 8 12 36 37
            6 44 12 36 28 52 20
            5 20 52 54 21 0
            6 21 46 14 38 30 54
            5 5 31 39 41 33
            6 7 5 31 55 58 34
            5 35 43 42 34 7
            6 47 50 26 3 1 22
            5 10 17 50 47 13
            6 1 25 49 51 27 3
            6 8 15 39 41 17 10
            5 45 15 8 12 44
            6 0 20 44 45 23 2
            6 24 2 0 21 46 48
            5 24 2 23 55 58
            6 55 31 39 15 45 23
            5 19 11 16 49 51
            6 9 18 42 43 19 11
            6 27 59 35 43 19 51
            6 33 57 59 35 7 5
            6 57 26 50 17 41 33
            5 26 3 27 59 57
            5 18 48 46 14 9
            6 18 42 34 58 24 48
            ").unwrap().abs;

        test_flags(&mut thah, 48);
        test_flags(&mut co, 96);
        test_flags(&mut snic, 240);
        test_flags(&mut ti, 360);
    }
}
