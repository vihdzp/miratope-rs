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

    pub fn range_iter(
        lo: Rank,
        hi: Rank,
    ) -> std::iter::Map<std::ops::Range<usize>, impl FnMut(usize) -> Rank> {
        (lo.0..hi.0).into_iter().map(Rank)
    }

    pub fn range_inclusive_iter(
        lo: Rank,
        hi: Rank,
    ) -> std::iter::Map<std::ops::RangeInclusive<usize>, impl FnMut(usize) -> Rank> {
        (lo.0..=hi.0).into_iter().map(Rank)
    }

    /// Subtraction with bounds checking.
    pub fn try_sub(&self, rhs: Self) -> Option<Self> {
        let lhs = self.0 + 1;
        let rhs = rhs.0;

        if lhs < rhs {
            None
        } else {
            Some(Self(lhs - rhs))
        }
    }
}

impl std::ops::Add for Rank {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0 - 1)
    }
}

impl std::ops::AddAssign for Rank {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.0 -= 1;
    }
}

impl std::ops::Sub for Rank {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 + 1 - rhs.0)
    }
}

impl Display for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.isize().fmt(f)
    }
}

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

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Constructs a new, empty `RankVec<T>` with the capacity.
    pub fn with_capacity(rank: Rank) -> Self {
        RankVec(Vec::with_capacity(rank.0 + 1))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns the greatest rank stored in the array. Panics if the `RankVec`
    /// is empty.
    pub fn rank(&self) -> Rank {
        Rank(self.len() - 1)
    }

    pub fn push(&mut self, value: T) {
        self.0.push(value)
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

    pub fn last(&self) -> Option<&T> {
        self.0.last()
    }

    pub fn iter(&self) -> Iter<T> {
        Iter(self.0.iter())
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut(self.0.iter_mut())
    }

    pub fn reverse(&mut self) {
        self.0.reverse()
    }

    /// Divides one mutable slice into two at an index.
    pub fn split_at_mut(&mut self, mid: Rank) -> (&mut [T], &mut [T]) {
        self.0.split_at_mut(mid.0)
    }

    /// Swaps two elements in the vector.
    pub fn swap(&mut self, a: Rank, b: Rank) {
        self.0.swap(a.0, b.0);
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    pub fn insert(&mut self, index: Rank, element: T) {
        self.0.insert(index.0, element)
    }
}

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
