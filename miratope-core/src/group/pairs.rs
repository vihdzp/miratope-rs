//! Contains the iterators over pairs.

/// An iterator over pairs of elements, which then maps these into another type.
pub struct PairMap<A, B, F> {
    /// The first set of values.
    a: Vec<A>,

    /// The second set of values.
    b: Vec<B>,

    /// The position we're seeking from the first set.
    pos_a: usize,

    /// The position we're seeking from the second set.
    pos_b: usize,

    /// The function we're using to map pairs into another type.
    f: F,
}

impl<T, A, B, F: FnMut(&A, &B) -> T> PairMap<A, B, F> {
    /// Initializes a new iterator over pairs. Uses `f` to map pairs of these
    /// types into another type.
    pub fn new(a: Vec<A>, b: Vec<B>, f: F) -> Self {
        Self {
            a,
            b,
            pos_a: 0,
            pos_b: 0,
            f,
        }
    }

    /// Initializes a new iterator over pairs by collecting two iterators. Uses
    /// `f` to map pairs of these types into another type.
    pub fn new_iter<G: Iterator<Item = A>, H: Iterator<Item = B>>(g: G, h: H, f: F) -> Self {
        Self::new(g.collect::<Vec<_>>(), h.collect::<Vec<_>>(), f)
    }
}

impl<T, A, B, F: FnMut(&A, &B) -> T> Iterator for PairMap<A, B, F> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(b) = self.b.get(self.pos_b) {
            let a = &self.a[self.pos_a];

            if self.pos_a < self.a.len() - 1 {
                self.pos_a += 1;
            } else if self.pos_b < self.b.len() - 1 {
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

/// An iterator over pairs of elements, which then maps and filters these into
/// another type.
pub struct PairFilterMap<A, B, F>(PairMap<A, B, F>);

impl<T, A, B, F: FnMut(&A, &B) -> Option<T>> PairFilterMap<A, B, F> {
    /// Initializes a new iterator over pairs. Uses `f` to filter and map pairs
    /// of these types into another type.
    pub fn new(a: Vec<A>, b: Vec<B>, f: F) -> Self {
        Self(PairMap::new(a, b, f))
    }

    /// Initializes a new iterator over pairs by collecting two iterators. Uses
    /// `f` to filter and map pairs of these types into another type.
    pub fn new_iter<G: Iterator<Item = A>, H: Iterator<Item = B>>(g: G, h: H, f: F) -> Self {
        Self(PairMap::new_iter(g, h, f))
    }
}

impl<T, A, B, F: FnMut(&A, &B) -> Option<T>> Iterator for PairFilterMap<A, B, F> {
    type Item = T;

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
