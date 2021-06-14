/// A trait for any type that acts as a wrapper around a `Vec<T>`. Will
/// automatically implement all corresponding methods.

pub trait VecLike:
    Default
    + From<Vec<Self::VecItem>>
    + AsRef<Vec<Self::VecItem>>
    + AsMut<Vec<Self::VecItem>>
    + std::ops::Index<usize>
    + std::ops::IndexMut<usize>
    + IntoIterator
{
    type VecItem;

    fn new() -> Self {
        Vec::new().into()
    }

    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
    }

    fn contains(&self, x: &Self::VecItem) -> bool
    where
        <Self as VecLike>::VecItem: PartialEq,
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

    fn get(&self, index: usize) -> Option<&Self::VecItem> {
        self.as_ref().get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Self::VecItem> {
        self.as_mut().get_mut(index)
    }

    fn append(&mut self, other: &mut Self) {
        self.as_mut().append(other.as_mut())
    }

    fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn iter(&self) -> std::slice::Iter<Self::VecItem> {
        self.as_ref().iter()
    }

    fn iter_mut(&mut self) -> std::slice::IterMut<Self::VecItem> {
        self.as_mut().iter_mut()
    }

    fn sort(&mut self)
    where
        <Self as VecLike>::VecItem: Ord,
    {
        self.as_mut().sort()
    }

    fn sort_unstable(&mut self)
    where
        <Self as VecLike>::VecItem: Ord,
    {
        self.as_mut().sort_unstable()
    }

    fn sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        <Self as VecLike>::VecItem: Ord,
        F: FnMut(&Self::VecItem) -> K,
        K: Ord,
    {
        self.as_mut().sort_unstable_by_key(f)
    }
}

#[macro_export]
/// Implements the [`VecLike`] trait for the type of the first argument, and
/// sets the [`VecLike::VecItem`] parameter to the second argument. Will also
/// implement all required traits automatically.
///
/// This macro assumes that the first argument's declaration looks like this:
/// ```
/// struct Wrapper(Vec<Item>);
/// ```
macro_rules! impl_veclike {
    ($T: tt, $VecItem: tt) => {
        impl crate::vec_like::VecLike for $T {
            type VecItem = $VecItem;
        }

        impl Default for $T {
            fn default() -> Self {
                Vec::new().into()
            }
        }

        impl From<Vec<$VecItem>> for $T {
            fn from(list: Vec<$VecItem>) -> Self {
                Self(list)
            }
        }

        impl AsRef<Vec<$VecItem>> for $T {
            fn as_ref(&self) -> &Vec<$VecItem> {
                &self.0
            }
        }

        impl AsMut<Vec<$VecItem>> for $T {
            fn as_mut(&mut self) -> &mut Vec<$VecItem> {
                &mut self.0
            }
        }

        impl std::ops::Index<usize> for $T {
            type Output = $VecItem;

            fn index(&self, index: usize) -> &Self::Output {
                &self.as_ref()[index]
            }
        }

        impl std::ops::IndexMut<usize> for $T {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut()[index]
            }
        }

        impl IntoIterator for $T {
            type Item = $VecItem;

            type IntoIter = std::vec::IntoIter<$VecItem>;

            fn into_iter(self) -> Self::IntoIter {
                self.0.into_iter()
            }
        }

        impl<'a> IntoIterator for &'a $T {
            type Item = &'a $VecItem;

            type IntoIter = std::slice::Iter<'a, $VecItem>;

            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        impl<'a> rayon::iter::IntoParallelIterator for &'a mut $T {
            type Iter = rayon::slice::IterMut<'a, $VecItem>;

            type Item = &'a mut $VecItem;

            fn into_par_iter(self) -> Self::Iter {
                self.as_mut().into_par_iter()
            }
        }
    };
}
