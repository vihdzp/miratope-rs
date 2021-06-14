/// A trait for anything that should work as an index in a vector.
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
pub trait VecLike<'a>:
    Default
    + From<Vec<Self::VecItem>>
    + AsRef<Vec<Self::VecItem>>
    + AsMut<Vec<Self::VecItem>>
    + std::ops::Index<Self::VecIndex>
    + std::ops::IndexMut<Self::VecIndex>
    + IntoIterator
{
    type VecItem;
    type VecIndex: VecIndex;

    fn new() -> Self {
        Vec::new().into()
    }

    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
    }

    fn reserve(&mut self, additional: usize) {
        self.as_mut().reserve(additional)
    }

    fn contains(&self, x: &Self::VecItem) -> bool
    where
        <Self as VecLike<'a>>::VecItem: PartialEq,
    {
        self.as_ref().contains(x)
    }

    fn push(&mut self, value: Self::VecItem) {
        self.as_mut().push(value)
    }

    fn pop(&mut self) -> Option<Self::VecItem> {
        self.as_mut().pop()
    }

    fn remove(&mut self, index: usize) -> Self::VecItem {
        self.as_mut().remove(index)
    }

    fn get(&self, index: Self::VecIndex) -> Option<&Self::VecItem> {
        self.as_ref().get(index.index())
    }

    fn get_mut(&mut self, index: Self::VecIndex) -> Option<&mut Self::VecItem> {
        self.as_mut().get_mut(index.index())
    }

    fn append(&mut self, other: &mut Self) {
        self.as_mut().append(other.as_mut())
    }

    fn insert(&mut self, index: Self::VecIndex, element: Self::VecItem) {
        self.as_mut().insert(index.index(), element)
    }

    fn iter(&'a self) -> std::slice::Iter<'a, <Self as VecLike<'a>>::VecItem> {
        self.as_ref().iter()
    }

    fn iter_mut(&'a mut self) -> std::slice::IterMut<'a, <Self as VecLike<'a>>::VecItem> {
        self.as_mut().iter_mut()
    }

    fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn last(&self) -> Option<&Self::VecItem> {
        self.as_ref().last()
    }

    fn reverse(&mut self) {
        self.as_mut().reverse()
    }

    fn sort(&mut self)
    where
        <Self as VecLike<'a>>::VecItem: Ord,
    {
        self.as_mut().sort()
    }

    fn sort_unstable(&mut self)
    where
        <Self as VecLike<'a>>::VecItem: Ord,
    {
        self.as_mut().sort_unstable()
    }

    fn sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        <Self as VecLike<'a>>::VecItem: Ord,
        F: FnMut(&Self::VecItem) -> K,
        K: Ord,
    {
        self.as_mut().sort_unstable_by_key(f)
    }

    fn swap(&mut self, a: Self::VecIndex, b: Self::VecIndex) {
        self.as_mut().swap(a.index(), b.index())
    }

    fn split_at_mut(
        &mut self,
        mid: Self::VecIndex,
    ) -> (&mut [Self::VecItem], &mut [Self::VecItem]) {
        self.as_mut().split_at_mut(mid.index())
    }
}

#[macro_export]
/// Implements the [`VecLike`] trait for the type of the first argument, setting
/// the [`VecLike::VecItem`] parameter to the second argument and the
/// [`VecLike::VecIndex`] parameter to the third argument. Will also implement
/// all required subtraits automatically.
///
/// This macro can either be called like this:
///
/// ```
/// struct Wrapper(Vec<Item>);
/// impl_veclike!(Wrapper, Item, usize);
/// ```
///
/// Or like this:
///
/// ```
/// struct Wrapper<T>(Vec<T>);
/// impl_veclike!(@for [T] Wrapper<T>, T, usize);
/// ```
macro_rules! impl_veclike {
    ($(@for [$($generics: tt)*])? $Type: ty, $VecItem: ty, $VecIndex: ty $(,)?) => {
        impl<'a, $($($generics)*)?> crate::vec_like::VecLike<'a> for $Type {
            type VecItem = $VecItem;
            type VecIndex = $VecIndex;
        }

        impl$(<$($generics)*>)? From<Vec<$VecItem>> for $Type {
            fn from(list: Vec<$VecItem>) -> Self {
                Self(list)
            }
        }

        impl$(<$($generics)*>)? AsRef<Vec<$VecItem>> for $Type {
            fn as_ref(&self) -> &Vec<$VecItem> {
                &self.0
            }
        }

        impl$(<$($generics)*>)? AsMut<Vec<$VecItem>> for $Type {
            fn as_mut(&mut self) -> &mut Vec<$VecItem> {
                &mut self.0
            }
        }

        impl$(<$($generics)*>)? Default for $Type {
            fn default() -> Self {
                Vec::new().into()
            }
        }

        impl$(<$($generics)*>)? std::ops::Index<$VecIndex> for $Type {
            type Output = $VecItem;

            fn index(&self, index: $VecIndex) -> &Self::Output {
                use crate::vec_like::VecIndex;
                &self.as_ref()[index.index()]
            }
        }

        impl$(<$($generics)*>)? std::ops::IndexMut<$VecIndex> for $Type {
            fn index_mut(&mut self, index: $VecIndex) -> &mut Self::Output {
                use crate::vec_like::VecIndex;
                &mut self.as_mut()[index.index()]
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

        impl<'a, $($($generics: Send)*)?> rayon::iter::IntoParallelIterator for &'a mut $Type {
            type Iter = rayon::slice::IterMut<'a, $VecItem>;

            type Item = &'a mut $VecItem;

            fn into_par_iter(self) -> Self::Iter {
                self.as_mut().into_par_iter()
            }
        }
    };
}
