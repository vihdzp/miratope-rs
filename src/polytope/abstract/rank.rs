use std::{fmt::Display, hash::Hash, iter, slice, vec};

use serde::{Deserialize, Serialize};

/// Represents the [rank](https://polytope.miraheze.org/w/index.php?title=Rank)
/// of a polytope.
///
/// Externally, it behaves as a number from -1 onwards. Internally, it contains
/// a signed integer, representing the rank plus 1.
#[derive(
    PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Debug, Hash, Serialize, Deserialize,
)]
pub struct Rank(usize);

impl Rank {
    /// Initializes a `Rank` from an `isize`.
    pub fn new<T: Into<Rank>>(num: T) -> Self {
        num.into()
    }

    /// Casts the `Rank` into an `usize`, or panics if `self` is `-1`. This
    /// value is **not** the same as the internal value. Use `.0` for that.
    pub fn into_usize(self) -> usize {
        self.into()
    }

    /// Casts the `Rank` into an `isize`.
    pub fn into_isize(self) -> isize {
        self.into()
    }

    /// Casts the `Rank` into an `u32`, or panics if `self` is `-1`.
    pub fn into_u32(self) -> u32 {
        self.into()
    }

    /// Casts the `Rank` into an `f64`.
    pub fn into_f64(self) -> f64 {
        self.into_isize() as f64
    }

    /// Casts the `Rank` into an `usize`, or returns `None` if `self` is `-1`.
    pub const fn try_usize(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0 - 1)
        }
    }

    /// Adds one to the rank.
    pub const fn plus_one(self) -> Self {
        Self(self.0 + 1)
    }

    /// Adds one to the rank, returns it as a `usize`. This is equivalent to
    /// simply getting the internal value.
    pub const fn plus_one_usize(self) -> usize {
        self.0
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
    pub fn range_iter<T: Into<Rank>, U: Into<Rank>>(
        lo: T,
        hi: U,
    ) -> std::iter::Map<std::ops::Range<usize>, impl FnMut(usize) -> Rank> {
        (lo.into().0..hi.into().0).into_iter().map(Rank)
    }

    /// Returns an iterator over `lo..=hi`. A workaround until `Step` is
    /// stabilized.
    pub fn range_inclusive_iter<T: Into<Rank>, U: Into<Rank>>(
        lo: T,
        hi: U,
    ) -> std::iter::Map<std::ops::RangeInclusive<usize>, impl FnMut(usize) -> Rank> {
        (lo.into().0..=hi.into().0).into_iter().map(Rank)
    }

    /// Subtraction with bounds checking.
    pub fn try_sub<T: Into<Rank>>(&self, rhs: T) -> Option<Self> {
        let lhs = self.0 + 1;
        let rhs = rhs.into().0;

        if lhs < rhs {
            None
        } else {
            Some(Self(lhs - rhs))
        }
    }
}

impl std::str::FromStr for Rank {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(i32::from_str(s)?.into())
    }
}

macro_rules! impl_rank {
    ($T:ty) => {
        impl From<$T> for Rank {
            fn from(n: $T) -> Self {
                Self((n + 1) as usize)
            }
        }

        impl From<Rank> for $T {
            fn from(rank: Rank) -> Self {
                rank.0 as $T - 1
            }
        }
    };
}

impl_rank!(u8);
impl_rank!(u16);
impl_rank!(u32);
impl_rank!(u64);
impl_rank!(u128);
impl_rank!(usize);

impl_rank!(i8);
impl_rank!(i16);
impl_rank!(i32);
impl_rank!(i64);
impl_rank!(i128);
impl_rank!(isize);

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

/// Displays a rank as its `isize` value.
impl Display for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.into_isize().fmt(f)
    }
}

/// Allows for `Rank` to be used in sliders.
impl bevy_egui::egui::emath::Numeric for Rank {
    const INTEGRAL: bool = true;

    const MIN: Self = Self(0);

    const MAX: Self = Self(usize::MAX);

    fn to_f64(self) -> f64 {
        self.into_f64()
    }

    fn from_f64(num: f64) -> Self {
        Self::new(num as isize)
    }
}

/// A `Vec` indexed by [rank](https://polytope.miraheze.org/wiki/Rank). Wraps
/// around operations that offset by a constant for our own convenience.
#[derive(PartialEq, Eq, Hash, Debug, Default, Clone)]
pub struct RankVec<T>(Vec<T>);

impl<T> From<Vec<T>> for RankVec<T> {
    fn from(vec: Vec<T>) -> Self {
        Self(vec)
    }
}

impl<T> AsRef<Vec<T>> for RankVec<T> {
    fn as_ref(&self) -> &Vec<T> {
        &self.0
    }
}

impl<T> AsMut<Vec<T>> for RankVec<T> {
    fn as_mut(&mut self) -> &mut Vec<T> {
        &mut self.0
    }
}

impl<T> RankVec<T> {
    /// Constructs a new, empty `RankVec<T>`.
    pub fn new() -> Self {
        RankVec(Vec::new())
    }

    /// Determines if the `RankVec<T>` is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Determines if the `RankVec<T>` is empty.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Constructs a new, empty `RankVec<T>` with the capacity.
    pub fn with_capacity(rank: Rank) -> Self {
        RankVec(Vec::with_capacity(rank.plus_one_usize() + 1))
    }

    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional)
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
impl<T, U: Into<Rank>> std::ops::Index<U> for RankVec<T> {
    type Output = T;

    fn index(&self, index: U) -> &Self::Output {
        let rank = self.rank();
        let index = index.into();
        self.0.get(index.0).unwrap_or_else(|| {
            panic!(
                "index out of bounds: the rank is {} but the index is {}",
                rank, index
            )
        })
    }
}

/// Allows mutably indexing a `RankVec<T>` by a `Rank`.
impl<T, U: Into<Rank>> std::ops::IndexMut<U> for RankVec<T> {
    fn index_mut(&mut self, index: U) -> &mut Self::Output {
        let rank = self.rank();
        let index = index.into();
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
pub struct Iter<'a, T>(slice::Iter<'a, T>);

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
    ) -> iter::Map<iter::Enumerate<slice::Iter<'a, T>>, impl FnMut((usize, &'a T)) -> (Rank, &'a T)>
    {
        self.0.enumerate().map(|(idx, t)| (Rank(idx), t))
    }
}

/// A wrapper around a usual mutable iterator over vectors, which implements a
/// [`rank_enumerate`](IntoIter::rank_enumerate) convenience method.
pub struct IterMut<'a, T>(slice::IterMut<'a, T>);

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
    ) -> iter::Map<
        iter::Enumerate<slice::IterMut<'a, T>>,
        impl FnMut((usize, &'a mut T)) -> (Rank, &'a mut T),
    > {
        self.0.enumerate().map(|(idx, t)| (Rank(idx), t))
    }
}

/// A wrapper around a usual iterator over vectors, which implements a
/// [`rank_enumerate`](IntoIter::rank_enumerate) convenience method.
pub struct IntoIter<T>(vec::IntoIter<T>);

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
    ) -> iter::Map<iter::Enumerate<vec::IntoIter<T>>, impl FnMut((usize, T)) -> (Rank, T)> {
        self.0.enumerate().map(|(idx, t)| (Rank(idx), t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Checks that rank arithmetic is in order.
    fn rank_arithmetic() {
        assert_eq!(Rank::new(2) + Rank::new(3), Rank::new(5));
        assert_eq!(Rank::new(7) - Rank::new(4), Rank::new(3));
        assert_eq!(Rank::new(-1).try_sub(Rank::new(1)), None);
        assert_eq!(Rank::new(6).plus_one(), Rank::new(7));
        assert_eq!(Rank::new(0).minus_one(), Rank::new(-1));
        assert_eq!(Rank::new(-1).plus_one_usize(), 0);
    }
}
