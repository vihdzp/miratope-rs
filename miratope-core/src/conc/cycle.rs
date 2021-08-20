//! Contains the code to convert from a polygon as a set of edges into a polygon
//! as a cycle of vertices.

use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    fmt::Display,
    iter::FromIterator,
};

use vec_like::*;

/// Represents a cyclic list of vertex indices, which may then be turned into a
/// path and tessellated.
pub struct Cycle(Vec<usize>);
impl_veclike!(Cycle, Item = usize, Index = usize);

/// A list of [`Cycles`](Cycle).
pub struct CycleList(Vec<Cycle>);
impl_veclike!(CycleList, Item = Cycle, Index = usize);

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

/// An error when converting a pair into a tuple. The inner value stores whether
/// the flag was empty (or whether it had a single element).
#[derive(Clone, Copy, Debug)]
pub struct PairError(bool);

impl PairError {
    /// Initializes a new error from a `Pair` with less than two elements.
    pub fn new<T>(pair: &Pair<T>) -> Self {
        Self(pair.is_empty())
    }
}

impl Display for PairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 {
            f.write_str("pair was empty")
        } else {
            f.write_str("pair had a single member")
        }
    }
}

impl std::error::Error for PairError {}

impl<T> TryFrom<Pair<T>> for (T, T) {
    type Error = PairError;

    fn try_from(pair: Pair<T>) -> Result<Self, PairError> {
        if let Pair::Two(v0, v1) = pair {
            Ok((v0, v1))
        } else {
            Err(PairError::new(&pair))
        }
    }
}

impl<T> Pair<T> {
    /// Returns `true` if the pair is `None`.
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

    /// Pushes a value onto the pair.
    ///
    /// # Panics
    /// The code will panic if you attempt to push a value onto a pair that
    /// already has two elements in it.
    pub fn push(&mut self, value: T)
    where
        T: Clone,
    {
        *self = match self {
            Self::None => Self::One(value),
            Self::One(first) => Self::Two(first.clone(), value),
            _ => panic!("Can't push a value onto a pair with two elements!"),
        };
    }
}

impl<'a, T: 'a + AsRef<[usize]>> FromIterator<T> for CycleBuilder {
    fn from_iter<I: IntoIterator<Item = T>>(edges: I) -> Self {
        let mut cycle = CycleBuilder::new();

        for edge in edges {
            cycle.push_edge(edge.as_ref());
        }

        cycle
    }
}

/// A helper struct to find cycles in a graph where the degree of all nodes
/// equals 2. This is most useful when working with (compound) polygons.
///
/// Internally, each node is mapped to a [`Pair`], which stores the indices of
/// the (at most) two other nodes it's connected to. By traversing this map,
/// we're able to recover the cycles.
#[derive(Default)]
pub struct CycleBuilder(HashMap<usize, Pair<usize>>);

impl CycleBuilder {
    /// Initializes a new empty cycle builder.
    pub fn new() -> Self {
        Default::default()
    }

    /// Initializes an empty cycle builder with a given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    /// Returns `true` if no vertices have been added.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the number of vertices that have been added.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns the first index and pair in the hash map, under some arbitrary
    /// order.
    pub fn first(&self) -> Option<(&usize, &Pair<usize>)> {
        self.0.iter().next()
    }

    /// Removes the entry associated to a given node and returns it, or `None`
    /// if no such entry exists.
    pub fn remove(&mut self, idx: usize) -> Option<Pair<usize>> {
        self.0.remove(&idx)
    }

    /// Returns a mutable reference to the edge associated to a node, adding it
    /// if it doesn't exist.
    pub fn get_mut(&mut self, idx: usize) -> &mut Pair<usize> {
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

    /// Pushes a given edge into the graph. In debug mode, asserts that the edge
    /// has exactly two elements.
    pub fn push_edge(&mut self, edge: &[usize]) {
        debug_assert_eq!(edge.len(), 2);
        self.push(edge[0], edge[1]);
    }

    /// Returns the indices of the two nodes adjacent to a given one.
    ///
    /// # Panics
    /// This method will panic if there are less than two elements adjacent to
    /// the specified one.
    pub fn get_remove(&mut self, idx: usize) -> (usize, usize) {
        self.remove(idx).unwrap_or_default().try_into().unwrap()
    }

    /// Cycles through the graph, returns a vector of node indices in cyclic
    /// order.
    pub fn build(&mut self) -> CycleList {
        let mut cycles = CycleList::new();

        // While there's some vertex from which we haven't generated a cycle:
        while let Some((&init, _)) = self.first() {
            let mut cycle = Cycle::with_capacity(self.len());
            let mut prev = init;
            let mut cur = self.get_remove(prev).0;

            cycle.push(cur);

            // We traverse the graph, finding the next node over and over, until
            // we reach the initial node again.
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

impl CycleList {
    /// Builds a list of cycles from a list of edges.
    pub fn from_edges<'a, T: 'a + AsRef<[usize]>, I: IntoIterator<Item = T>>(edges: I) -> Self {
        edges.into_iter().collect::<CycleBuilder>().build()
    }
}
