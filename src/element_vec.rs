use std::cmp::Ordering::*;

/// A vector of integers that stores whether it's sorted. Used to store the
/// subelements or superelements of an element.
#[derive(Clone)]
pub struct ElementVec {
    /// Stores `true` only when the vector is sorted.
    sorted: bool,

    /// The backing vector.
    vec: Vec<u32>,
}

impl Default for ElementVec {
    fn default() -> Self {
        Self {
            sorted: true,
            vec: Vec::new(),
        }
    }
}

impl From<Vec<u32>> for ElementVec {
    fn from(vec: Vec<u32>) -> Self {
        Self { sorted: false, vec }
    }
}

impl std::ops::Index<usize> for ElementVec {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec[index]
    }
}

impl<'a> IntoIterator for &'a ElementVec {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter()
    }
}

impl ElementVec {
    /// Initializes a new empty `ElementVec`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a reference to the underlying vector.
    pub fn vec(&self) -> &Vec<u32> {
        &self.vec
    }

    /// Returns the length of the vector.
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    /// Returns whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    /// Returns an iterator over the vector.
    pub fn iter(&self) -> impl Iterator<Item = &u32> {
        self.vec.iter()
    }

    /// Initializes a `ElementVec` from a sorted vector.
    ///
    /// # Safety
    /// The passed vector must be sorted.
    pub unsafe fn from_sorted(vec: Vec<u32>) -> Self {
        Self { sorted: true, vec }
    }

    /// Initializes a `ElementVec` from its raw parts.
    ///
    /// # Safety
    /// If `sorted` is set as `true`, then the passed vector must actually be sorted.
    pub unsafe fn from_raw_parts(vec: Vec<u32>, sorted: bool) -> Self {
        Self { vec, sorted }
    }

    pub fn sorted(&self) -> bool {
        self.sorted
    }

    /// Initializes a singleton `ElementVec`.
    pub fn single(value: u32) -> Self {
        // Safety: all singletons are sorted.
        unsafe { Self::from_sorted(vec![value]) }
    }

    /// Initializes a `ElementVec` with the numbers from `0` in `n - 1` in order.
    pub fn count_lt(n: u32) -> Self {
        // Safety: the vector is sorted.
        unsafe { Self::from_sorted((0..n).collect()) }
    }

    /// Pushes an element into the `ElementVec`. Assumes that it will become unsorted.
    pub fn push(&mut self, value: u32) {
        self.sorted = false;
        self.vec.push(value);
    }

    /// Sorts `self`. Does nothing if already sorted.
    pub fn sort(&mut self) {
        if !self.sorted {
            self.vec.sort_unstable();
            self.sorted = true;
        }
    }

    /// Checks that two `ElementVec`s have exactly two elements in common.
    pub fn check_diamond(&mut self, other: &mut Self) -> bool {
        let (mut i, mut j) = (0, 0);
        let mut count: u8 = 0;

        loop {
            match self[i].cmp(&other[j]) {
                Less => {
                    i += 1;
                    if i == self.len() {
                        break;
                    }
                }

                Greater => {
                    j += 1;
                    if j == other.len() {
                        break;
                    }
                }

                Equal => {
                    count += 1;
                    if count > 2 {
                        return false;
                    }

                    i += 1;
                    j += 1;
                    if i == self.len() || j == other.len() {
                        break;
                    }
                }
            }
        }

        count == 2
    }

    /// Gets the two common elements between two `ElementVec`s.
    ///
    /// # Safety
    /// This method optimizes heavily by assuming that these two elements
    /// actually exist. In the specific use cases for this method, this is a
    /// consequence of the diamond property of polytopes.
    pub fn common(&mut self, other: &mut Self) -> (u32, u32) {
        self.sort();
        other.sort();

        let (mut i, mut j) = (0, 0);
        let mut first = None;

        // This is a hot loop, so it's worth optimizing.
        loop {
            // Safety: the algorithm will finish before we go out of bounds.
            let (&lval, rval) =
                unsafe { (self.vec().get_unchecked(i), other.vec().get_unchecked(j)) };

            match lval.cmp(rval) {
                Less => i += 1,
                Greater => j += 1,
                Equal => {
                    if let Some(first) = first {
                        return (first, lval);
                    } else {
                        first = Some(lval);
                    }

                    i += 1;
                    j += 1;
                }
            }
        }
    }
}
