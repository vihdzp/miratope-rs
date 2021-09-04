//! Contains the iterators over pairs.

/// Represents a type that can be seen as containing various pairs of type
/// `(A, B)`.
pub trait AsPair: Sized {
    /// The type of elements of the first set.
    type A;

    /// The type of elements of the second set.
    type B;

    /// Returns the first set of elements.
    fn first(&self) -> &[Self::A];

    /// Returns the second set of elements.
    fn second(&self) -> &[Self::B];

    /// Interprets `self` as a pair of sets of elements.
    fn into_pairs(self) -> Pair<Self> {
        Pair(self)
    }
}

impl<A, B> AsPair for (Vec<A>, Vec<B>) {
    type A = A;
    type B = B;

    fn first(&self) -> &[A] {
        &self.0
    }

    fn second(&self) -> &[B] {
        &self.1
    }
}

impl<A> AsPair for Vec<A> {
    type A = A;
    type B = A;

    fn first(&self) -> &[A] {
        self
    }

    fn second(&self) -> &[A] {
        self
    }
}

/// Represents a map from pairs of elements into another type.
pub struct PairMap<T: AsPair, F> {
    /// The pairs of elements.
    pairs: T,

    /// The index of the element in the first set.
    pos_a: usize,

    /// The index of the element in the second set.
    pos_b: usize,

    /// The function used to map pairs into the new type.
    f: F,
}

impl<U, T: AsPair, F: for<'a, 'b> FnMut(&'a T::A, &'b T::B) -> U> PairMap<T, F> {
    /// Initializes a new map over pairs.
    pub fn new(pairs: T, f: F) -> Self {
        Self {
            pairs,
            pos_a: 0,
            pos_b: 0,
            f,
        }
    }
}

impl<U, T: AsPair, F: for<'a, 'b> FnMut(&'a T::A, &'b T::B) -> U> Iterator for PairMap<T, F> {
    type Item = U;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(b) = self.pairs.second().get(self.pos_b) {
            let a = &self.pairs.first()[self.pos_a];

            if self.pos_a < self.pairs.first().len() - 1 {
                self.pos_a += 1;
            } else if self.pos_b < self.pairs.second().len() - 1 {
                self.pos_a = 0;
                self.pos_b += 1;
            } else {
                self.pos_b += 1;
            }

            Some((self.f)(a, b))
        } else {
            None
        }
    }
}

/// Represents a filter-map from pairs of elements into another type.
pub struct PairFilterMap<T: AsPair, F>(PairMap<T, F>);

impl<U, T: AsPair, F: for<'a, 'b> FnMut(&'a T::A, &'b T::B) -> Option<U>> PairFilterMap<T, F> {
    /// Initializes a new filter map over pairs.
    pub fn new(pairs: T, f: F) -> Self {
        Self(PairMap::new(pairs, f))
    }
}

impl<U, T: AsPair, F: for<'a, 'b> FnMut(&'a T::A, &'b T::B) -> Option<U>> Iterator
    for PairFilterMap<T, F>
{
    type Item = U;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.0.next() {
                Some(Some(x)) => return Some(x),
                Some(None) => continue,
                None => return None,
            }
        }
    }
}

/// Represents a pair of sets.
///
/// This type doesn't do much by itself, other than to give a simple API to the
/// iterators on pairs.
pub struct Pair<T: AsPair>(T);

/// The type of the function that clones a pair.
type CloneFn<T> = for<'a, 'b> fn(
    &'a <T as AsPair>::A,
    &'b <T as AsPair>::B,
) -> (<T as AsPair>::A, <T as AsPair>::B);

impl<T: AsPair> Pair<T> {
    /// Maps each pair using the specified function.
    pub fn map<U, F: for<'a, 'b> FnMut(&'a T::A, &'b T::B) -> U>(self, f: F) -> PairMap<T, F> {
        PairMap::new(self.0, f)
    }

    /// Filters and maps each pair using the specified function.
    pub fn filter_map<U, F: for<'a, 'b> FnMut(&'a T::A, &'b T::B) -> Option<U>>(
        self,
        f: F,
    ) -> PairFilterMap<T, F> {
        PairFilterMap::new(self.0, f)
    }

    /// Returns the cloned pairs.
    pub fn cloned(self) -> PairMap<T, CloneFn<T>>
    where
        T::A: Clone,
        T::B: Clone,
    {
        self.map(|a, b| (a.clone(), b.clone()))
    }
}
