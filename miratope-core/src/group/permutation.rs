//! Deals with permutations and permutation groups.

use std::ops::{Index, Mul, MulAssign};

use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, Dynamic, OVector};

/// Represents a permutation on `n` elements. The index `i` is mapped into the
/// number `self[i]`.
pub struct Permutation<N: Dim>(OVector<usize, N>)
where
    DefaultAllocator: Allocator<usize, N>;

/// A statically sized permutation.
pub type SPermutation<const N: usize> = Permutation<Const<N>>;

/// A dynamically sized permutation.
pub type DPermutation = Permutation<Dynamic>;

impl<N: Dim> Index<usize> for Permutation<N>
where
    DefaultAllocator: Allocator<usize, N>,
{
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<N: Dim> Permutation<N>
where
    DefaultAllocator: Allocator<usize, N>,
{
    /// Builds a permutation from the entries.
    ///
    /// # Safety
    /// The user must certify that all of these values are distinct and less
    /// than the length of the permutation.
    pub unsafe fn from_data<I: Into<OVector<usize, N>>>(data: I) -> Self {
        Self(data.into())
    }

    /// Builds a permutation from an iterator. This method works for both static
    /// and dynamically-sized permutations.
    ///
    /// # Safety
    /// The user must certify that all of these values are distinct and less
    /// than the length of the permutation.
    pub unsafe fn from_iterator_generic<I: IntoIterator<Item = usize>>(
        len: usize,
        iter: I,
    ) -> Self {
        Self(OVector::from_iterator_generic(
            N::from_usize(len),
            Const::<1>,
            iter,
        ))
    }

    /// Returns the length of the permutation.
    pub fn len(&self) -> usize {
        self.0.nrows()
    }

    /// Returns `true` if the permutation has no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Swaps two entries in the permutation.
    pub fn swap(&mut self, i: usize, j: usize) {
        self.0.swap((i, 1), (j, 1));
    }

    /// Returns an iterator over the copied entries in `self`.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().copied()
    }

    /// Returns a mutable iterator over the entries in `self`.
    ///
    /// # Safety
    /// The user must certify that once this iterator is exhausted, all of these
    /// values are distinct and less than the length of the permutation.
    pub unsafe fn iter_mut(&mut self) -> impl Iterator<Item = &mut usize> {
        self.0.iter_mut()
    }
}

impl<const N: usize> Default for Permutation<Const<N>> {
    fn default() -> Self {
        Self::id()
    }
}

impl<const N: usize> SPermutation<N> {
    /// Builds a permutation from an iterator.
    ///
    /// # Safety
    /// The user must certify that all of these values are distinct and less
    /// than the length of the permutation.
    pub unsafe fn from_iterator<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self::from_iterator_generic(N, iter)
    }

    /// Returns the identity permutation.
    pub fn id() -> Self {
        // Safety: all of these elements are different and less than `N`.
        unsafe { Self::from_iterator(0..N) }
    }
}

impl DPermutation {
    /// Builds a permutation from an iterator.
    ///
    /// # Safety
    /// The user must certify that all of these values are distinct and less
    /// than the length of the permutation.
    pub unsafe fn from_iterator<I: IntoIterator<Item = usize>>(iter: I, len: usize) -> Self {
        Self::from_iterator_generic(len, iter)
    }

    /// Returns the identity permutation.
    pub fn id(n: usize) -> Self {
        // Safety: all of these elements are different and less than `n`.
        unsafe { Self::from_iterator(0..n, n) }
    }
}

impl<N: Dim> Mul for Permutation<N>
where
    DefaultAllocator: Allocator<usize, N>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Safety: the composition of two permutations is a permutation.
        unsafe { Self::from_iterator_generic(self.len(), self.iter().map(|i| rhs[i])) }
    }
}

impl<N: Dim> MulAssign for Permutation<N>
where
    DefaultAllocator: Allocator<usize, N>,
{
    fn mul_assign(&mut self, rhs: Self) {
        // Safety: the composition of two permutations is a permutation.
        for i in unsafe { self.iter_mut() } {
            *i = rhs[*i];
        }
    }
}
