use std::collections::HashMap;

use crate::{impl_veclike, vec_like::VecLike};

/// Represents a set with at most two values.
#[derive(Clone, Copy)]
pub enum Pair<T> {
    /// No entry.
    None,

    /// One entry.
    One(T),

    /// Two entries.
    Two(T, T),
}

impl<T> Default for Pair<T> {
    fn default() -> Self {
        Self::None
    }
}

impl<T> Pair<T> {
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Returns the number of elements stored in the pair.
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
        }
    }

    /// Pushes a value onto the pair by copy.
    ///
    /// # Panics
    /// The code will panic if you attempt to push a value onto a pair that
    /// already has two elements in it.
    pub fn push(&mut self, value: T)
    where
        T: Copy,
    {
        *self = match self {
            Self::None => Self::One(value),
            Self::One(first) => Self::Two(*first, value),
            Self::Two(_, _) => panic!("Can't push a value onto a pair with two elements!"),
        };
    }
}

/// A helper struct to build a cycle of vertices from a polygonal path.
///
/// Internally, each vertex is mapped to a [`Pair`], which stores the (at most)
/// two other vertices it's connected to. By traversing this map, we're able to
/// recover the vertex cycles.
#[derive(Default)]
pub struct CycleBuilder(HashMap<usize, Pair<usize>>);

impl CycleBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    /// Initializes a cycle builder with a given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    /// Returns the number of vertices in the vertex loop.
    fn len(&self) -> usize {
        self.0.len()
    }

    fn iter(&self) -> std::collections::hash_map::Iter<usize, Pair<usize>> {
        self.0.iter()
    }

    /// Removes the entry associated to a given vertex and returns it, or `None`
    /// if no such entry exists.
    fn remove(&mut self, idx: usize) -> Option<Pair<usize>> {
        self.0.remove(&idx)
    }

    /// Returns a mutable reference to the edge associated to a vertex, adding
    /// it if it doesn't exist.
    fn get_mut(&mut self, idx: usize) -> &mut Pair<usize> {
        use std::collections::hash_map::Entry;

        match self.0.entry(idx) {
            // Returns a reference to the entry.
            Entry::Occupied(entry) => entry.into_mut(),

            // Adds the entry, returns the reference to its value.
            Entry::Vacant(entry) => entry.insert(Pair::None),
        }
    }

    /// Pushes a pair of vertices into the vertex loop.
    pub fn push(&mut self, vertex0: usize, vertex1: usize) {
        self.get_mut(vertex0).push(vertex1);
        self.get_mut(vertex1).push(vertex0);
    }

    /// Returns the indices of the two vertices adjacent to a given one.
    ///
    /// # Panics
    /// This method will panic if there are less than two elements adjacent to
    /// the specified one.
    pub fn get_remove(&mut self, idx: usize) -> (usize, usize) {
        let pair = self.remove(idx).unwrap_or_default();

        if let Pair::Two(v0, v1) = pair {
            (v0, v1)
        } else {
            panic!("Expected 2 elements in pair, found {}.", pair.len())
        }
    }

    /// Cycles through the vertex loop, returns the vector of vertices in cyclic
    /// order.
    pub fn cycles(&mut self) -> Vec<Cycle> {
        let mut cycles = Vec::new();

        // While there's some vertex from which we haven't generated a cycle:
        while let Some((&init, _)) = self.iter().next() {
            let mut cycle = Cycle::with_capacity(self.len());
            let mut prev = init;
            let mut cur = self.get_remove(prev).0;

            cycle.push(cur);

            // We traverse the polygon, finding the next vertex over and over, until
            // we reach the initial vertex again.
            loop {
                // The two candidates for the next vertex.
                let (next0, next1) = self.get_remove(cur);

                let next_is_next1 = next0 == prev;
                prev = cur;

                // We go to whichever adjacent vertex isn't equal to the one we were
                // previously at.
                if next_is_next1 {
                    cycle.push(next1);
                    cur = next1;
                } else {
                    cycle.push(next0);
                    cur = next0;
                };

                // Whenever we reach the initial vertex, we break out of the loop.
                if cur == init {
                    break;
                }
            }

            cycles.push(cycle);
        }

        cycles
    }
}

/// Represents a cyclic list of vertex indices, which may then be turned into a
/// path and tessellated.
pub struct Cycle(Vec<usize>);
impl_veclike!(Cycle, usize, usize);
