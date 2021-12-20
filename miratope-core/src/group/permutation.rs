//! Deals with permutations and permutation groups.

use std::{
    collections::BTreeMap,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Index, Mul, MulAssign},
};

use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, Dynamic, OVector, UninitVector,
};

use super::group_item::{GroupItem, Wrapper};

/// Represents a permutation on `n` elements. The index `i` is mapped into the
/// number `self[i]`.
#[derive(Clone, PartialEq, Debug)]
pub struct Permutation<N: Dim>(OVector<usize, N>)
where
    DefaultAllocator: Allocator<usize, N>;

impl<N: Dim> Eq for Permutation<N> where DefaultAllocator: Allocator<usize, N> {}

impl<N: Dim> PartialOrd for Permutation<N>
where
    DefaultAllocator: Allocator<usize, N>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<N: Dim> Ord for Permutation<N>
where
    DefaultAllocator: Allocator<usize, N>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

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

impl<const N: usize> Default for SPermutation<N> {
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

impl<'a, 'b, N: Dim> Mul<&'b Permutation<N>> for &'a Permutation<N>
where
    DefaultAllocator: Allocator<usize, N>,
{
    type Output = Permutation<N>;

    fn mul(self, rhs: &'b Permutation<N>) -> Self::Output {
        // Safety: the composition of two permutations is a permutation.
        unsafe { Permutation::from_iterator_generic(self.len(), self.iter().map(|i| rhs[i])) }
    }
}

impl<'a, N: Dim> MulAssign<&'a Permutation<N>> for Permutation<N>
where
    DefaultAllocator: Allocator<usize, N>,
{
    fn mul_assign(&mut self, rhs: &'a Permutation<N>) {
        // Safety: the composition of two permutations is a permutation.
        for i in unsafe { self.iter_mut() } {
            *i = rhs[*i];
        }
    }
}

impl<N: Dim> GroupItem for Permutation<N>
where
    DefaultAllocator: Allocator<usize, N>,
{
    type Dim = N;
    type FuzzyOrd = Self;

    fn id(dim: Self::Dim) -> Self {
        let n = dim.value();
        unsafe { Self::from_iterator_generic(n, 0..n) }
    }

    fn inv(&self) -> Self {
        let mut uninit = UninitVector::uninit(N::from_usize(self.len()), Const::<1>);

        for (i, j) in self.iter().enumerate() {
            uninit[j] = MaybeUninit::new(i);
        }

        // Safety: we must have filled all entries, since permutations are
        // bijections.
        Self(unsafe { uninit.assume_init() })
    }

    fn mul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn mul_assign(&mut self, rhs: &Self) {
        *self *= rhs;
    }
}

/// An iterator over the permutations associated to a group.
pub struct PermutationIter<T: GroupItem, D: Dim>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// The elements of the group, provided for easy iteration.
    vec: Vec<T>,

    /// Another copy of the elements of the group, provided so that we can map
    /// each element into an index.
    indices: BTreeMap<T::FuzzyOrd, usize>,

    /// The number of permutation we're generating.
    idx: usize,

    /// This type determines the size of the permutations to output.
    dim: PhantomData<D>,
}

/// A [`PermutationIter`] with a statically known size.
pub type SPermutationIter<I, const N: usize> = PermutationIter<I, Const<N>>;

/// A [`PermutationIter`] with a dynamically known size.
pub type DPermutationIter<I> = PermutationIter<I, Dynamic>;

impl<T: GroupItem + Clone, D: Dim> PermutationIter<T, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Initializes a new iterator over permutations from the elements of a
    /// group.
    pub fn new(vec: Vec<T>) -> Self {
        D::from_usize(vec.len());
        let mut indices = BTreeMap::new();

        for (i, el) in vec.iter().enumerate() {
            indices.insert(Wrapper::from_inner(el.clone()), i);
        }

        Self {
            vec,
            indices,
            idx: 0,
            dim: PhantomData,
        }
    }
}

impl<T: GroupItem + Clone, D: Dim> Iterator for PermutationIter<T, D>
where
    DefaultAllocator: Allocator<usize, D> + Allocator<T, D>,
{
    type Item = Permutation<D>;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.vec.get(self.idx)?;
        self.idx += 1;

        unsafe {
            Some(Permutation::from_iterator_generic(
                self.vec.len(),
                self.vec.iter().map(|b| {
                    self.indices
                        .get(&Wrapper::from_inner(a.mul(b)))
                        .copied()
                        .unwrap()
                }),
            ))
        }
    }
}
