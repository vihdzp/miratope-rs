use std::fmt::Display;

use serde::{Deserialize, Serialize};

/// Represents the [rank](https://polytope.miraheze.org/w/index.php?title=Rank)
/// of a polytope.
///
/// Externally, it behaves as a number from -1 onwards. Internally, it contains
/// a signed integer, representing the rank plus 1.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash, Serialize, Deserialize)]
pub struct Rank(pub usize);

impl Rank {
    /// Initializes a `Rank` from an `isize`.
    pub const fn new(rank: isize) -> Self {
        Self((rank + 1) as usize)
    }

    /// Casts the `Rank` into an `usize`, or panics if `self` is `-1`. This
    /// value is **not** the same as the internal value. Use `.0` for that.
    pub const fn usize(&self) -> usize {
        self.0 - 1
    }

    /// Casts the `Rank` into an `usize`, or returns `None` if `self` is `-1`.
    pub const fn try_usize(&self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0 - 1)
        }
    }

    /// Casts the `Rank` into an `isize`.
    pub const fn isize(&self) -> isize {
        self.0 as isize - 1
    }

    /// Casts the `Rank` into an `u32`, or panics if `self` is `-1`.
    pub const fn u32(&self) -> u32 {
        self.usize() as u32
    }

    /// Casts the `Rank` into an `f64`.
    pub const fn f64(&self) -> f64 {
        self.isize() as f64
    }

    /// Adds one to the rank.
    pub const fn plus_one(&self) -> Self {
        Self(self.0 + 1)
    }

    /// Subtracts one from the rank.
    pub const fn minus_one(&self) -> Self {
        Self(self.0 - 1)
    }

    pub const fn try_minus_one(&self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(Self(self.0 - 1))
        }
    }

    /// Returns an iterator over `lo..hi`. A workaround until `Step` is
    /// stabilized.
    pub fn range_iter(
        lo: Rank,
        hi: Rank,
    ) -> std::iter::Map<std::ops::Range<usize>, impl FnMut(usize) -> Rank> {
        (lo.0..hi.0).into_iter().map(Rank)
    }

    /// Returns an iterator over `lo..=hi`. A workaround until `Step` is
    /// stabilized.
    pub fn range_inclusive_iter(
        lo: Rank,
        hi: Rank,
    ) -> std::iter::Map<std::ops::RangeInclusive<usize>, impl FnMut(usize) -> Rank> {
        (lo.0..=hi.0).into_iter().map(Rank)
    }

    /// Subtraction with bounds checking.
    pub const fn try_sub(&self, rhs: Self) -> Option<Self> {
        let lhs = self.0 + 1;
        let rhs = rhs.0;

        if lhs < rhs {
            None
        } else {
            Some(Self(lhs - rhs))
        }
    }
}

/// Adds two ranks.
impl std::ops::Add for Rank {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0 - 1)
    }
}

/// Adds a rank to another.
impl std::ops::AddAssign for Rank {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.0 -= 1;
    }
}

/// Subtracts two ranks.
impl std::ops::Sub for Rank {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 + 1 - rhs.0)
    }
}

/// Converts a `usize` into a `Rank`.
impl From<usize> for Rank {
    fn from(rank: usize) -> Self {
        Self(rank + 1)
    }
}

/// Displays a rank as its `isize` value.
impl Display for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.isize().fmt(f)
    }
}

/// Allows for `Rank` to be used in sliders.
impl bevy_egui::egui::emath::Numeric for Rank {
    const INTEGRAL: bool = true;

    const MIN: Self = Self(0);

    const MAX: Self = Self(usize::MAX);

    fn to_f64(self) -> f64 {
        self.f64()
    }

    fn from_f64(num: f64) -> Self {
        Self::new(num as isize)
    }
}

/// A `Vec` indexed by [rank](https://polytope.miraheze.org/wiki/Rank). Wraps
/// around operations that offset by a constant for our own convenience.
#[derive(Debug, Clone)]
pub struct RankVec<T>(pub Vec<T>);

impl<T> RankVec<T> {
    /// Constructs a new, empty `RankVec<T>`.
    pub fn new() -> Self {
        RankVec(Vec::new())
    }

    /// Determines if the `RankVec<T>` is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Constructs a new, empty `RankVec<T>` with the capacity.
    pub fn with_capacity(rank: Rank) -> Self {
        RankVec(Vec::with_capacity(rank.0 + 1))
    }

    /// Returns the greatest rank stored in the array.
    ///
    /// # Panics
    /// Panics if the `RankVec<T>` is empty.
    pub fn rank(&self) -> Rank {
        Rank(self.0.len() - 1)
    }

    /// Pushes a value onto the `RankVec<T>`.
    pub fn push(&mut self, value: T) {
        self.0.push(value)
    }

    /// Pops a value from the `RankVec<T>`.
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    /// Returns a reference to the element at a given position or `None` if out
    /// of bounds.
    pub fn get(&self, index: Rank) -> Option<&T> {
        self.0.get(index.0)
    }

    /// Returns a mutable reference to an element or `None` if the index is out
    /// of bounds.
    pub fn get_mut(&mut self, index: Rank) -> Option<&mut T> {
        self.0.get_mut(index.0)
    }

    /// Returns the last element of the `RankVec<T>`, or `None` if it is empty.
    pub fn last(&self) -> Option<&T> {
        self.0.last()
    }

    /// Returns an iterator over the `RankVec<T>`.
    pub fn iter(&self) -> Iter<T> {
        Iter(self.0.iter())
    }

    /// Returns an iterator that allows modifying each value.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut(self.0.iter_mut())
    }

    /// Reverses the order of elements in the `RankVec<T>`, in place.
    pub fn reverse(&mut self) {
        self.0.reverse()
    }

    /// Divides one mutable slice into two at an index.
    pub fn split_at_mut(&mut self, mid: Rank) -> (&mut [T], &mut [T]) {
        self.0.split_at_mut(mid.0)
    }

    /// Swaps two elements in the `RankVec<T>`.
    pub fn swap(&mut self, a: Rank, b: Rank) {
        self.0.swap(a.0, b.0);
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    pub fn insert(&mut self, index: Rank, element: T) {
        self.0.insert(index.0, element)
    }
}

/// Allows indexing a `RankVec<T>` by a `Rank`.
impl<T> std::ops::Index<Rank> for RankVec<T> {
    type Output = T;

    fn index(&self, index: Rank) -> &Self::Output {
        let rank = self.rank();
        self.0.get(index.0).unwrap_or_else(|| {
            panic!(
                "index out of bounds: the rank is {} but the index is {}",
                rank, index
            )
        })
    }
}

/// Allows mutably indexing a `RankVec<T>` by a `Rank`.
impl<T> std::ops::IndexMut<Rank> for RankVec<T> {
    fn index_mut(&mut self, index: Rank) -> &mut Self::Output {
        let rank = self.rank();
        self.0.get_mut(index.0).unwrap_or_else(|| {
            panic!(
                "index out of bounds: the rank is {} but the index is {}",
                rank, index
            )
        })
    }
}

impl<T> IntoIterator for RankVec<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.0.into_iter())
    }
}

/// A wrapper around a usual iterator over vectors, which implements a
/// [`rank_enumerate`](IntoIter::rank_enumerate) convenience method.
pub struct Iter<'a, T>(std::slice::Iter<'a, T>);

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, T> Iter<'a, T> {
    /// Wraps around the usual `enumerate` method, offsetting the first entry by 1.
    pub fn rank_enumerate(
        self,
    ) -> std::iter::Map<
        std::iter::Enumerate<std::slice::Iter<'a, T>>,
        impl FnMut((usize, &'a T)) -> (Rank, &'a T),
    > {
        self.0.enumerate().map(|(idx, t)| (Rank(idx), t))
    }
}

/// A wrapper around a usual mutable iterator over vectors, which implements a
/// [`rank_enumerate`](IntoIter::rank_enumerate) convenience method.
pub struct IterMut<'a, T>(std::slice::IterMut<'a, T>);

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, T> IterMut<'a, T> {
    /// Wraps around the usual `enumerate` method, offsetting the first entry by 1.
    pub fn rank_enumerate(
        self,
    ) -> std::iter::Map<
        std::iter::Enumerate<std::slice::IterMut<'a, T>>,
        impl FnMut((usize, &'a mut T)) -> (Rank, &'a mut T),
    > {
        self.0.enumerate().map(|(idx, t)| (Rank(idx), t))
    }
}

/// A wrapper around a usual iterator over vectors, which implements a
/// [`rank_enumerate`](IntoIter::rank_enumerate) convenience method.
pub struct IntoIter<T>(std::vec::IntoIter<T>);

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<T> IntoIter<T> {
    /// Wraps around the usual `enumerate` method, offsetting the first entry by 1.
    pub fn rank_enumerate(
        self,
    ) -> std::iter::Map<
        std::iter::Enumerate<std::vec::IntoIter<T>>,
        impl FnMut((usize, T)) -> (Rank, T),
    > {
        self.0.enumerate().map(|(idx, t)| (Rank(idx), t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Checks that rank arithmetic is in order.
    fn rank() {
        assert_eq!(Rank::new(2) + Rank::new(3), Rank::new(5));
        assert_eq!(Rank::new(7) - Rank::new(4), Rank::new(3));
        assert_eq!(Rank::new(-1).try_sub(Rank::new(1)), None);
        assert_eq!(Rank::new(6).plus_one(), Rank::new(7));
        assert_eq!(Rank::new(0).minus_one(), Rank::new(-1));
    }
}
