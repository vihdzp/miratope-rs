use derive_deref::{Deref, DerefMut};
use petgraph::graph::{DefaultIx, DiGraph, IndexType, NodeIndex};

/// A `Vec` indexed by [rank](https://polytope.miraheze.org/wiki/Rank). Wraps
/// around operations that offset by a constant for our own convenience.
#[derive(Debug, Clone, Deref, DerefMut)]
pub struct RankVec<T>(pub Vec<T>);

impl<T> RankVec<T> {
    /// Constructs a new, empty `RankVec<T>`.
    pub fn new() -> Self {
        RankVec(Vec::new())
    }

    /// Constructs a new, empty `RankVec<T>` with the specified capacity.
    pub fn with_rank(rank: isize) -> Self {
        RankVec(Vec::with_capacity((rank + 2) as usize))
    }

    /// Returns the greatest rank stored in the array.
    pub fn rank(&self) -> isize {
        self.len() as isize - 2
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
        &self.0[(index + 1) as usize]
    }
}

impl<T> std::ops::IndexMut<isize> for RankVec<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        &mut self.0[(index + 1) as usize]
    }
}

#[derive(Clone, Copy)]
struct RankIndex<Ix: IndexType = DefaultIx> {
    rank: isize,
    idx: NodeIndex<Ix>,
}

impl<Ix: IndexType> RankIndex<Ix> {
    /// Returns the index as a `usize`.
    pub fn index(&self) -> usize {
        self.idx.index()
    }
}

/// A partially ranked poset.
struct RankedPoset<N, E, Ix: IndexType = DefaultIx> {
    nodes: RankVec<Vec<NodeIndex<Ix>>>,
    graph: DiGraph<N, E, Ix>,
}

impl<N, E, Ix: IndexType> std::ops::Index<RankIndex<Ix>> for RankedPoset<N, E, Ix> {
    type Output = N;

    fn index(&self, rank_idx: RankIndex<Ix>) -> &Self::Output {
        &self.graph[self.node_idx(rank_idx)]
    }
}

impl<N, E, Ix: IndexType> std::ops::IndexMut<RankIndex<Ix>> for RankedPoset<N, E, Ix> {
    fn index_mut(&mut self, rank_idx: RankIndex<Ix>) -> &mut Self::Output {
        let idx = self.node_idx(rank_idx);
        &mut self.graph[idx]
    }
}

impl<N, E, Ix: IndexType> RankedPoset<N, E, Ix> {
    /// Creates a new empty `RankedPoset`.
    pub fn new() -> Self {
        Self {
            nodes: RankVec::new(),
            graph: DiGraph::default(),
        }
    }

    /// Returns the index into the graph corresponding to the given
    /// [`RankIndex`].
    pub fn node_idx(&self, RankIndex { rank, idx }: RankIndex<Ix>) -> NodeIndex<Ix> {
        self.nodes[rank][idx.index()]
    }

    /// Adds a node with a given rank and weight.
    pub fn add_node(&mut self, rank: isize, weight: N) -> RankIndex<Ix> {
        let idx = NodeIndex::new(self.nodes[rank].len());
        self.nodes[rank].push(self.graph.add_node(weight));

        RankIndex { rank, idx }
    }

    /// Removes a node at a given position.
    pub fn remove_node(&mut self, rank_idx: RankIndex<Ix>) -> Option<N> {
        let idx = rank_idx.index();
        let node_idx = self.node_idx(rank_idx);
        self.nodes[rank_idx.rank].swap_remove(idx);

        self.graph.remove_node(node_idx)
    }

    /// Adds an edge to the poset, from one rank to the one directly lower.
    pub fn add_edge(&mut self, a: RankIndex<Ix>, b: RankIndex<Ix>, weight: E) {
        assert_eq!(
            a.rank,
            b.rank + 1,
            "Edges must all go downward exactly one rank."
        );

        self.graph
            .add_edge(self.node_idx(a), self.node_idx(b), weight);
    }
}
