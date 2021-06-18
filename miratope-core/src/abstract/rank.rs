use std::{fmt::Display, hash::Hash, iter, slice, vec};

use serde::{Deserialize, Serialize};

use crate::{
    impl_veclike,
    vec_like::{VecIndex, VecLike},
};

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
    pub const fn new(num: isize) -> Self {
        Self((num + 1) as usize)
    }

    /// Casts the `Rank` into an `usize`, or panics if `self` is `-1`. This
    /// value is **not** the same as the internal value. Use `.0` for that.
    pub const fn into_usize(self) -> usize {
        self.0 as usize - 1
    }

    /// Casts the `Rank` into an `isize`.
    pub const fn into_isize(self) -> isize {
        self.0 as isize - 1
    }

    /// Casts the `Rank` into an `u32`, or panics if `self` is `-1`.
    pub const fn into_u32(self) -> u32 {
        self.0 as u32 - 1
    }

    /// Casts the `Rank` into an `f64`.
    pub const fn into_f64(self) -> f64 {
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
    pub const fn minus_one(self) -> Self {
        Self(self.0 - 1)
    }

    pub const fn try_minus_one(self) -> Option<Self> {
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
#[cfg(feature = "bevy_egui")]
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
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct RankVec<T>(Vec<T>);
impl_veclike!(@for [T] RankVec<T>, T, Rank);

impl<T> RankVec<T> {
    /// Returns the greatest rank stored in the array.
    ///
    /// # Panics
    /// Panics if the `RankVec<T>` is empty.
    pub fn rank(&self) -> Rank {
        (self.0.len() as isize - 2).into()
    }

    pub fn with_rank_capacity(rank: Rank) -> Self {
        Self::with_capacity(rank.plus_one_usize() + 1)
    }

    pub fn rank_into_iter(self) -> IntoIter<T> {
        IntoIter(self.into_iter())
    }

    pub fn rank_iter(&self) -> Iter<T> {
        Iter(self.iter())
    }

    pub fn rank_iter_mut(&mut self) -> IterMut<T> {
        IterMut(self.iter_mut())
    }
}

impl VecIndex for Rank {
    fn index(self) -> usize {
        self.plus_one_usize()
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
