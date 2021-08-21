//! Declares a general [`VecLike`] trait for all structs that wrap around
//! `Vec<T>`s, and an [`impl_veclike`] macro that automatically implements this
//! trait for a type.

/// A trait for anything that should work as an index in a vector. Any type that
/// implements this trait should be quickly convertible into a `usize`.
///
/// An implementation of this trait might look like this:
/// ```
/// struct Number(u8);
///
/// impl vec_like::VecIndex for Number {
///     fn index(self) -> usize {
///         self.0 as usize
///     }   
/// }
/// ```
pub trait VecIndex {
    fn index(self) -> usize;
}

impl VecIndex for usize {
    fn index(self) -> usize {
        self
    }
}

/// A trait for any type that acts as a wrapper around a `Vec<T>`. Will
/// automatically implement all corresponding methods.
pub trait VecLike:
    Default
    + std::ops::Index<Self::VecIndex>
    + std::ops::IndexMut<Self::VecIndex>
    + AsRef<[Self::VecItem]>
    + AsMut<[Self::VecItem]>
    + AsRef<Vec<Self::VecItem>>
    + AsMut<Vec<Self::VecItem>>
    + Into<Vec<Self::VecItem>>
    + From<Vec<Self::VecItem>>
    + Extend<Self::VecItem>
    + IntoIterator
    + std::iter::FromIterator<Self::VecItem>
where
    for<'a> &'a Self: IntoIterator,
    for<'a> &'a mut Self: IntoIterator,
{
    /// The item contained in the wrapped vector.
    type VecItem;

    /// The type used to index over the vector.
    type VecIndex: VecIndex;

    /// Returns a reference to the inner vector. This is equivalent to `as_ref`,
    /// but is more robust in case any types need to be inferred.
    fn as_inner(&self) -> &Vec<Self::VecItem> {
        self.as_ref()
    }

    /// Returns a mutable reference to the inner vector. This is equivalent to
    /// `as_mut`, but is more robust in case any types need to be inferred.
    fn as_inner_mut(&mut self) -> &mut Vec<Self::VecItem> {
        self.as_mut()
    }

    /// Returns the owned inner vector. This is equivalent to `into`, but is
    /// more robust in case any types need to be inferred
    fn into_inner(self) -> Vec<Self::VecItem> {
        self.into()
    }

    /// Wraps around an owned vector. This is equivalent to `into`, but is more
    /// robust in case any types need to be inferred.
    fn from_inner(vec: Vec<Self::VecItem>) -> Self {
        vec.into()
    }

    /// Initializes a new empty `Self` with no elements.
    fn new() -> Self {
        Vec::new().into()
    }

    /// Initializes a new empty `Self` with a given capacity.
    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in `self`.
    fn reserve(&mut self, additional: usize) {
        self.as_inner_mut().reserve(additional)
    }

    /// Returns true if `self` contains an element with the given value.
    fn contains(&self, x: &Self::VecItem) -> bool
    where
        Self::VecItem: PartialEq,
    {
        self.as_inner().contains(x)
    }

    /// Pushes a value onto `self`.
    fn push(&mut self, value: Self::VecItem) {
        self.as_inner_mut().push(value)
    }

    /// Pops a value from `self`.
    fn pop(&mut self) -> Option<Self::VecItem> {
        self.as_inner_mut().pop()
    }

    /// Removes and returns the element at position index within the `self`,
    /// shifting all elements after it to the left.
    fn remove(&mut self, index: usize) -> Self::VecItem {
        self.as_inner_mut().remove(index)
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering, but is O(1).
    fn swap_remove(&mut self, index: usize) -> Self::VecItem {
        self.as_inner_mut().swap_remove(index)
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity of the
    /// vector.
    fn clear(&mut self) {
        self.as_inner_mut().clear()
    }

    /// Returns a reference to an element or `None` if out of bounds.
    fn get(&self, index: Self::VecIndex) -> Option<&Self::VecItem> {
        self.as_inner().get(index.index())
    }

    /// Returns a mutable reference to an element or `None` if out of bounds.
    fn get_mut(&mut self, index: Self::VecIndex) -> Option<&mut Self::VecItem> {
        self.as_inner_mut().get_mut(index.index())
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    fn append(&mut self, other: &mut Self) {
        self.as_inner_mut().append(other.as_mut())
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    fn insert(&mut self, index: Self::VecIndex, element: Self::VecItem) {
        self.as_inner_mut().insert(index.index(), element)
    }

    /// Extracts a slice containing the entire vector.
    fn as_slice(&self) -> &[Self::VecItem] {
        self.as_ref()
    }

    /// Extracts a mutable slice of the entire vector.
    fn as_mut_slice(&mut self) -> &mut [Self::VecItem] {
        self.as_mut()
    }

    /// Returns an iterator over the slice.
    fn iter(&self) -> std::slice::Iter<<Self as VecLike>::VecItem> {
        self.as_inner().iter()
    }

    /// Returns a mutable iterator over the slice.
    fn iter_mut(&mut self) -> std::slice::IterMut<<Self as VecLike>::VecItem> {
        self.as_inner_mut().iter_mut()
    }

    /// Returns `true` if `self` contains no elements.
    fn is_empty(&self) -> bool {
        self.as_inner().is_empty()
    }

    /// Returns the number of elements in `self`.
    fn len(&self) -> usize {
        self.as_inner().len()
    }

    /// Returns a reference to the last element of `Self`, or `None` if it's
    /// empty.
    fn last(&self) -> Option<&Self::VecItem> {
        self.as_inner().last()
    }

    /// Returns a mutable reference to the last element of `Self`, or `None` if
    /// it's empty.
    fn last_mut(&mut self) -> Option<&mut Self::VecItem> {
        self.as_inner_mut().last_mut()
    }

    /// Reverses the order of elements in `self`, in place.
    fn reverse(&mut self) {
        self.as_inner_mut().reverse()
    }

    /// Sorts `self`.
    fn sort(&mut self)
    where
        Self::VecItem: Ord,
    {
        self.as_inner_mut().sort()
    }

    /// Sorts `self`, but may not preserve the order of equal elements.
    fn sort_unstable(&mut self)
    where
        Self::VecItem: Ord,
    {
        self.as_inner_mut().sort_unstable()
    }

    /// Sorts `self` with a key extraction function, but may not preserve the
    /// order of equal elements.
    fn sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        Self::VecItem: Ord,
        F: FnMut(&Self::VecItem) -> K,
        K: Ord,
    {
        self.as_inner_mut().sort_unstable_by_key(f)
    }

    /// Divides `self` into two slices at an index.
    ///
    /// The first will contain all indices from [0, `mid`) (excluding the index
    /// `mid` itself) and the second will contain all indices from [`mid`,
    /// `len`) (excluding the index `len` itself).
    fn split_at_mut(
        &mut self,
        mid: Self::VecIndex,
    ) -> (&mut [Self::VecItem], &mut [Self::VecItem]) {
        self.as_inner_mut().split_at_mut(mid.index())
    }

    /// Swaps two elements in `Self`.
    fn swap(&mut self, a: Self::VecIndex, b: Self::VecIndex) {
        self.as_inner_mut().swap(a.index(), b.index())
    }
}

/// Implements the [`VecLike`] trait for a given type. This macro will assume
/// that the type we implement this trait for is a tuple struct containing
/// either a `Vec` or a type implementing `VecLike` as its first argument.
///
/// The [`VecLike::VecItem`] parameter will be set to the second argument, and
/// the [`VecLike::VecIndex`] parameter will be set to the third argument. This
/// macro will also implement all required subtraits automatically.
///
/// This macro can either be called like this:
///
/// ```
/// # use vec_like::{VecLike, impl_veclike};
/// struct VecItem;
/// struct Wrapper(Vec<VecItem>);
/// impl_veclike!(Wrapper, Item = VecItem, Index = usize);
/// ```
///
/// Or like this:
///
/// ```
/// # use vec_like::{VecLike, impl_veclike};
/// struct Wrapper<T>(Vec<T>);
/// impl_veclike!(@for [T] Wrapper<T>, Item = T, Index = usize);
/// ```
///
/// # Todo
/// It would be nice to turn this into something that can be derived.
#[macro_export]
macro_rules! impl_veclike {
    ($(@for [$($generics: tt)*])? $Type:ty, Item = $VecItem:ty, Index = $VecIndex:ty$(,)?) => {
        vec_like::impl_veclike_field!($(@for [$($generics)*])? $Type, Item = $VecItem, Index = $VecIndex, Field = .0);

        impl$(<$($generics)*>)? From<Vec<$VecItem>> for $Type {
            fn from(vec: Vec<$VecItem>) -> Self {
                Self(vec)
            }
        }
    };
}

/// Implements the [`VecLike`] trait for a given type. This macro allows one to
/// specify which of the fields of the type works as the inner storage. You may
/// also use this with a tuple struct.
///
/// The [`VecLike::VecItem`] parameter will be set to the second argument, and
/// the [`VecLike::VecIndex`] parameter will be set to the third argument. This
/// macro will also implement almost all required subtraits automatically.
/// However, one must manually implement `From<Vec<Item>>` for the type.
///
/// This macro can either be called like this:
///
/// ```
/// # use vec_like::{VecLike, impl_veclike_field};
/// struct VecItem;
/// struct Wrapper {
///    vec: Vec<VecItem>,
///    other: ()
/// }
///
/// impl From<Vec<VecItem>> for Wrapper {
///     // ...
///     # fn from(vec: Vec<VecItem>) -> Self { Self{vec, other: ()} }
/// }
///
/// impl_veclike_field!(Wrapper, Item = VecItem, Index = usize, Field = .vec);
/// ```
///
/// Or like this:
///
/// ```
/// # use vec_like::{VecLike, impl_veclike_field};
/// struct Wrapper<T> {
///    vec: Vec<T>,
///    other: ()
/// }
///
/// impl<T> From<Vec<T>> for Wrapper<T> {
///     // ...
///     # fn from(vec: Vec<T>) -> Self { Self{vec, other: ()} }
/// }
///
/// impl_veclike_field!(@for [T] Wrapper<T>, Item = T, Index = usize, Field = .vec);
/// ```
///
/// # Todo
/// It would be nice to turn this into something that can be derived.
#[macro_export]
macro_rules! impl_veclike_field {
    ($(@for [$($generics: tt)*])? $Type:ty, Item = $VecItem:ty, Index = $VecIndex:ty, Field = .$field:tt$(,)?) => {
        impl<$($($generics)*)?> vec_like::VecLike for $Type {
            type VecItem = $VecItem;
            type VecIndex = $VecIndex;
        }

        impl$(<$($generics)*>)? Default for $Type {
            fn default() -> Self {
                Self::from_inner(Vec::new())
            }
        }

        impl$(<$($generics)*>)? std::ops::Index<$VecIndex> for $Type {
            type Output = $VecItem;

            fn index(&self, index: $VecIndex) -> &Self::Output {
                use vec_like::VecIndex;
                &self.as_inner()[index.index()]
            }
        }

        impl$(<$($generics)*>)? std::ops::IndexMut<$VecIndex> for $Type {
            fn index_mut(&mut self, index: $VecIndex) -> &mut Self::Output {
                use vec_like::VecIndex;
                &mut self.as_inner_mut()[index.index()]
            }
        }

        impl$(<$($generics)*>)? AsRef<Vec<$VecItem>> for $Type {
            fn as_ref(&self) -> &Vec<$VecItem> {
                self.$field.as_ref()
            }
        }

        impl$(<$($generics)*>)? AsMut<Vec<$VecItem>> for $Type {
            fn as_mut(&mut self) -> &mut Vec<$VecItem> {
                self.$field.as_mut()
            }
        }

        impl$(<$($generics)*>)? AsRef<[$VecItem]> for $Type {
            fn as_ref(&self) -> &[$VecItem] {
                self.as_inner().as_slice()
            }
        }

        impl$(<$($generics)*>)? AsMut<[$VecItem]> for $Type {
            fn as_mut(&mut self) -> &mut [$VecItem] {
                self.as_inner_mut().as_mut_slice()
            }
        }

        impl$(<$($generics)*>)? From<$Type> for Vec<$VecItem> {
            fn from(t: $Type) -> Self {
                t.$field.into()
            }
        }

        impl$(<$($generics)*>)? Extend<$VecItem> for $Type {
            fn extend<I:IntoIterator<Item = $VecItem>>(&mut self, iter: I) {
                self.as_inner_mut().extend(iter)
            }
        }

        impl$(<$($generics)*>)? IntoIterator for $Type {
            type Item = $VecItem;
            type IntoIter = std::vec::IntoIter<$VecItem>;

            fn into_iter(self) -> Self::IntoIter {
                self.$field.into_iter()
            }
        }

        impl<'a, $($($generics)*)?> IntoIterator for &'a $Type {
            type Item = &'a $VecItem;
            type IntoIter = std::slice::Iter<'a, $VecItem>;

            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        impl<'a, $($($generics)*)?> IntoIterator for &'a mut $Type {
            type Item = &'a mut $VecItem;
            type IntoIter = std::slice::IterMut<'a, $VecItem>;

            fn into_iter(self) -> Self::IntoIter {
                self.iter_mut()
            }
        }

        impl$(<$($generics)*>)? std::iter::FromIterator<$VecItem> for $Type {
            fn from_iter<__I__: IntoIterator<Item = $VecItem>>(iter: __I__) -> Self {
                Self::from_inner(iter.into_iter().collect())
            }
        }
    };
}
