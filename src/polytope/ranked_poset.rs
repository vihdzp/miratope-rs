use petgraph::graph::{DefaultIx, DiGraph, IndexType, NodeIndex};

use super::RankVec;

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
        &mut self.graph[self.node_idx(rank_idx)]
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
