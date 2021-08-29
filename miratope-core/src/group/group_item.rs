//! Defines the properties of items in a group, and provides a few
//! implementations.

use std::{cmp::Ordering, mem};

use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, Quaternion};
use ordered_float::OrderedFloat;

use crate::{
    geometry::{Matrix, MatrixOrd, MatrixOrdMxN},
    Float,
};

/// A trait for any type that behaves as a wrapper around another type.
///
/// # Safety
/// The default implementations will transmute and cast pointers. These
/// operations will most likely be invalid unless this type is a transparent
/// wrapper around the inner type.
pub unsafe trait Wrapper<T>: Sized {
    /// Creates a value for the wrapper from an inner value.
    fn from_inner(inner: T) -> Self {
        unsafe { mem::transmute_copy(&inner) }
    }

    /// Gets the inner value from a wrapper.
    fn into_inner(self) -> T {
        unsafe { mem::transmute_copy(&self) }
    }

    /// Interprets a reference to an inner value as a reference to the wrapper.
    fn as_wrapper(inner: &T) -> &Self {
        unsafe { &*(inner as *const _ as *const _) }
    }

    /// Gets the reference to the inner value from the wrapper.
    fn as_inner(&self) -> &T {
        unsafe { &*(self as *const _ as *const _) }
    }
}

// Everything is ultimately a wrapper for itself!
unsafe impl<T> Wrapper<T> for T {
    fn from_inner(inner: T) -> Self {
        inner
    }

    fn into_inner(self) -> T {
        self
    }

    fn as_wrapper(inner: &T) -> &Self {
        inner
    }

    fn as_inner(&self) -> &T {
        self
    }
}

unsafe impl<T: Float> Wrapper<T> for OrderedFloat<T> {
    fn from_inner(inner: T) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> T {
        self.0
    }

    fn as_inner(&self) -> &T {
        &self.0
    }
}

unsafe impl<T: Float, R: Dim, C: Dim> Wrapper<OMatrix<T, R, C>> for MatrixOrdMxN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn from_inner(inner: OMatrix<T, R, C>) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> OMatrix<T, R, C> {
        self.0
    }

    fn as_inner(&self) -> &OMatrix<T, R, C> {
        &self.0
    }
}

unsafe impl<T: Float> Wrapper<Quaternion<T>> for MatrixOrdMxN<T, Const<4>, Const<1>> {
    fn from_inner(inner: Quaternion<T>) -> Self {
        Self(inner.coords)
    }

    fn into_inner(self) -> Quaternion<T> {
        Quaternion::from_vector(self.0)
    }
}

/// A trait for a type that can be used as the elements of a group.
pub trait GroupItem: Sized {
    /// The type of any parameters, like the dimension of a matrix or the length
    /// of a permutation, on which the identity of the type may depend.
    ///
    /// In case there are no such parameters, and any two values of your type
    /// may be operated on, you should set this to `()`.
    type Dim: Copy;

    /// A wrapper type for an ordered version of the type. This should be
    /// resistant to small perturbations. It's strongly encouraged this
    /// conversion is zero-cost.
    type FuzzyOrd: Ord + Clone + Wrapper<Self>;

    /// Returns the multiplicative identity of the type.
    fn id(dim: Self::Dim) -> Self;

    /// Returns the multiplicative inverse of the value.
    fn inv(&self) -> Self;

    /// Multiplies two elements of the type.
    fn mul(&self, rhs: &Self) -> Self;

    /// Multiplies and assigns two elements of the type.
    fn mul_assign(&mut self, rhs: &Self);

    /// Determines whether two values are equal, using the wrapper specified by
    /// the trait.
    fn eq(&self, other: &Self) -> bool {
        Self::FuzzyOrd::as_wrapper(self) == Self::FuzzyOrd::as_wrapper(other)
    }

    /// Compares two values, using the wrapper specified by the trait.
    fn cmp(&self, other: &Self) -> Ordering {
        Self::FuzzyOrd::as_wrapper(self).cmp(Self::FuzzyOrd::as_wrapper(other))
    }
}

impl GroupItem for () {
    type Dim = ();
    type FuzzyOrd = ();

    fn id(_: ()) {}

    fn inv(&self) {}

    fn mul(&self, _: &()) {}

    fn mul_assign(&mut self, _: &()) {}
}

impl<T: Float> GroupItem for T {
    type Dim = ();
    type FuzzyOrd = OrderedFloat<T>;

    fn id(_: ()) -> T {
        T::ONE
    }

    fn inv(&self) -> T {
        T::ONE / *self
    }

    fn mul(&self, rhs: &T) -> T {
        *self * *rhs
    }

    fn mul_assign(&mut self, rhs: &T) {
        *self *= *rhs;
    }
}

impl<T: Float> GroupItem for Matrix<T> {
    type Dim = usize;
    type FuzzyOrd = MatrixOrd<T>;

    fn id(dim: usize) -> Self {
        Self::identity(dim, dim)
    }

    fn inv(&self) -> Self {
        let mut mat = self.clone();
        mat.try_inverse_mut();
        mat
    }

    fn mul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn mul_assign(&mut self, rhs: &Self) {
        *self *= rhs;
    }
}

impl<T: Float> GroupItem for Quaternion<T> {
    type Dim = ();
    type FuzzyOrd = MatrixOrdMxN<T, Const<4>, Const<1>>;

    fn id(_: ()) -> Self {
        Self::identity()
    }

    fn inv(&self) -> Self {
        self.try_inverse().unwrap()
    }

    fn mul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn mul_assign(&mut self, rhs: &Self) {
        *self *= rhs;
    }
}
