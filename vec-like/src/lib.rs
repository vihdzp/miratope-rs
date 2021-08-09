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
    Default + std::ops::Index<Self::VecIndex> + std::ops::IndexMut<Self::VecIndex> + IntoIterator
{
    /// The item contained in the wrapped vector.
    type VecItem;

    /// The type used to index over the vector.
    type VecIndex: VecIndex;

    /// Returns a reference to the inner vector.
    fn as_inner(&self) -> &Vec<Self::VecItem>;

    /// Returns a mutable reference to the inner vector.
    fn as_inner_mut(&mut self) -> &mut Vec<Self::VecItem>;

    /// Returns the owned inner vector.
    fn into_inner(self) -> Vec<Self::VecItem>;

    /// Wraps around an owned vector.
    fn from_inner(vec: Vec<Self::VecItem>) -> Self;

    /// Initializes a new empty `Self` with no elements.
    fn new() -> Self {
        Self::from_inner(Vec::new())
    }

    /// Initializes a new empty `Self` with a given capacity.
    fn with_capacity(capacity: usize) -> Self {
        Self::from_inner(Vec::with_capacity(capacity))
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
        self.as_inner_mut().append(other.as_inner_mut())
    }

    fn insert(&mut self, index: Self::VecIndex, element: Self::VecItem) {
        self.as_inner_mut().insert(index.index(), element)
    }

    fn iter(&self) -> std::slice::Iter<<Self as VecLike>::VecItem> {
        self.as_inner().iter()
    }

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

    /// Returns the last element of `Self`, or `None` if it's empty.
    fn last(&self) -> Option<&Self::VecItem> {
        self.as_inner().last()
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

/// Implements the [`VecLike`] trait for the type of the first argument, setting
/// the [`VecLike::VecItem`] parameter to the second argument and the
/// [`VecLike::VecIndex`] parameter to the third argument. Will also implement
/// all required subtraits automatically.
///
/// This macro can either be called like this:
///
/// ```
/// use vec_like::{VecLike, impl_veclike};
///
/// struct VecItem;
/// struct Wrapper(Vec<VecItem>);
/// impl_veclike!(Wrapper, Item = VecItem, Index = usize);
/// ```
///
/// Or like this:
///
/// ```
/// use vec_like::{VecLike, impl_veclike};
///
/// struct Wrapper<T>(Vec<T>);
/// impl_veclike!(@for [T] Wrapper<T>, Item = T, Index = usize);
/// ```
///
/// TODO: probably turn this into something that can be derived.
#[macro_export]
macro_rules! impl_veclike {
    ($(@for [$($generics: tt)*])? $Type: ty, Item = $VecItem: ty, Index = $VecIndex: ty $(,)?) => {
        impl<'a, $($($generics)*)?> vec_like::VecLike for $Type {
            type VecItem = $VecItem;
            type VecIndex = $VecIndex;

            fn as_inner(&self) -> &Vec<$VecItem> {
                &self.0
            }

            fn as_inner_mut(&mut self) -> &mut Vec<$VecItem> {
                &mut self.0
            }

            fn into_inner(self) -> Vec<$VecItem> {
                self.0
            }

            fn from_inner(vec: Vec<$VecItem>) -> Self {
                Self(vec)
            }
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

        impl$(<$($generics)*>)? IntoIterator for $Type {
            type Item = $VecItem;

            type IntoIter = std::vec::IntoIter<$VecItem>;

            fn into_iter(self) -> Self::IntoIter {
                self.0.into_iter()
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
    };
}
