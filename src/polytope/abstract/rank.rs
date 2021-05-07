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
    pub fn with_capacity(rank: isize) -> Self {
        RankVec(Vec::with_capacity((rank + 2) as usize))
    }

    /// Returns the greatest rank stored in the array.
    pub fn rank(&self) -> isize {
        self.0.len() as isize - 2
    }

    pub fn push(&mut self, value: T) {
        self.0.push(value)
    }

    /// Returns a reference to the element at a given position or `None` if out
    /// of bounds.
    pub fn get(&self, index: isize) -> Option<&T> {
        if index < -1 {
            None
        } else {
            self.0.get((index + 1) as usize)
        }
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
    pub fn split_at_mut(&mut self, mid: isize) -> (&mut [T], &mut [T]) {
        self.0.split_at_mut((mid + 1) as usize)
    }

    /// Returns a mutable reference to an element or `None` if the index is out
    /// of bounds.
    pub fn get_mut(&mut self, index: isize) -> Option<&mut T> {
        if index < -1 {
            None
        } else {
            self.0.get_mut((index + 1) as usize)
        }
    }

    /// Swaps two elements in the vector.
    pub fn swap(&mut self, a: isize, b: isize) {
        self.0.swap((a + 1) as usize, (b + 1) as usize);
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    pub fn insert(&mut self, index: isize, element: T) {
        self.0.insert((index + 1) as usize, element)
    }
}

impl<T> std::ops::Index<isize> for RankVec<T> {
    type Output = T;

    fn index(&self, index: isize) -> &Self::Output {
        let rank = self.rank();
        self.0.get((index + 1) as usize).unwrap_or_else(|| {
            panic!(
                "index out of bounds: the rank is {} but the index is {}",
                rank, index
            )
        })
    }
}

impl<T> std::ops::IndexMut<isize> for RankVec<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let rank = self.rank();
        self.0.get_mut((index + 1) as usize).unwrap_or_else(|| {
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
        impl FnMut((usize, &'a T)) -> (isize, &'a T),
    > {
        self.0.enumerate().map(|(idx, t)| (idx as isize - 1, t))
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
        impl FnMut((usize, &'a mut T)) -> (isize, &'a mut T),
    > {
        self.0.enumerate().map(|(idx, t)| (idx as isize - 1, t))
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
        impl FnMut((usize, T)) -> (isize, T),
    > {
        self.0.enumerate().map(|(idx, t)| (idx as isize - 1, t))
    }
}
