use crate::EPS;
use petgraph::{
    graph::NodeReferences,
    visit::{Dfs, IntoNodeReferences},
};

use super::{
    geometry::{Hyperplane, Point, Subspace},
    Concrete,
};
use float_ord::FloatOrd;
use petgraph::graph::{NodeIndex, NodeWeightsMut, UnGraph};

/// Represents a facet in a convex hull. Can be thought of as an oriented
/// hyperplane, together with a bunch of points that lie on its outer side.
/// To save space, stores the indices of points instead of the points
/// themselves.
struct Facet<'a> {
    /// The index of a single point that lies on the inner side of the facet.
    /// Used to test where other points are with respect to this half space.
    inner: &'a Point,

    /// The indices of the points that lie strictly on the facet.
    contained: Vec<&'a Point>,

    /// The indices of the points that lie on the outer side of the facet, and
    /// which "are associated" to the facet. Any point must be the outer point
    /// of at most one facet.
    outer: Vec<&'a Point>,
}

#[derive(PartialEq)]
enum Position {
    Inner,
    Contained,
    Outer,
}

impl<'a> Facet<'a> {
    fn new(inner: &'a Point, contained: Vec<&'a Point>) -> Self {
        Self {
            inner,
            contained,
            outer: Vec::new(),
        }
    }

    /// Returns the farthest outer point from the facet.
    fn farthest(&self) -> Option<&'a Point> {
        let s = Hyperplane::new(
            Subspace::from_points(self.contained.iter().copied().cloned().collect()),
            self.inner,
        );

        self.outer
            .iter()
            .copied()
            .max_by_key(|&p| FloatOrd(s.distance(p)))
    }

    /// Returns whether a given point is inside, outside, or contained on the
    /// facet.
    fn position(&self, p: &'a Point) -> Position {
        let c = self.contained[0];
        let i = self.inner;
        let dot = (c - i).dot(&(p - i));

        if dot > EPS {
            Position::Inner
        } else if dot < -EPS {
            Position::Outer
        } else {
            Position::Contained
        }
    }
}

/// Represents a ridge in a convex hull.
struct Ridge<'a> {
    /// The indices of the points that lie strictly on the ridge.
    contained: Vec<&'a Point>,
}

impl<'a> Ridge<'a> {
    fn new(contained: Vec<&'a Point>) -> Self {
        Self { contained }
    }
}

/// Represents a convex hull. Is made out of facets and ridges, and stores them
/// as an incidence graph.
struct Hull<'a> {
    /// An undirected graph representing adjacencies between facets. Nodes in
    /// the graph represent facets, while edges represent ridges.
    ///
    /// # Todo
    /// The reason we use a graph structure specifically is that it allows us to
    /// DFS over the facets. This is very useful when computing the visible
    /// boundary of a point. Our approach is currently much more naive than
    /// this.
    graph: UnGraph<Facet<'a>, Ridge<'a>>,

    /// The vector of facets that we've verified are on the convex hull. Is
    /// initialized empty, but should have precisely the polytope's facets at
    /// the end.
    final_facets: Vec<Facet<'a>>,
}

impl<'a> Hull<'a> {
    /// Builds a new `Hull` from the facet-ridge graph.
    fn new(graph: UnGraph<Facet<'a>, Ridge<'a>>) -> Self {
        Self {
            graph,
            final_facets: Vec::new(),
        }
    }

    /// Builds a simplicial hull from a set of indices of vertices.
    fn simplex(indices: &[&'a Point]) -> Self {
        let n = indices.len();
        let mut graph = UnGraph::with_capacity(n, n * (n - 1) / 2);

        // Adds facets. Each facet has all vertices but one, and is oriented
        // away of the remaining one.
        for i in 0..n {
            graph.add_node(Facet::new(
                indices[i],
                indices[0..i]
                    .iter()
                    .chain(indices[(i + 1)..n].iter())
                    .cloned()
                    .collect(),
            ));
        }

        // Adds ridges. Each ridge has all vertices but two.
        for j in 1..n {
            for i in 0..j {
                graph.add_edge(
                    NodeIndex::new(i),
                    NodeIndex::new(j),
                    Ridge::new(
                        indices[0..i]
                            .iter()
                            .chain(indices[(i + 1)..j].iter())
                            .chain(indices[(j + 1)..n].iter())
                            .cloned()
                            .collect(),
                    ),
                );
            }
        }

        Self::new(graph)
    }

    /// Returns a reference to the facet with a given index.
    fn get_facet(&self, idx: usize) -> Option<&Facet<'a>> {
        self.graph.node_weight(NodeIndex::new(idx))
    }

    /// Attempts to remove a facet with a given index, and returns it if
    /// successful.
    fn remove_facet(&mut self, idx: usize) -> Option<Facet<'a>> {
        self.graph.remove_node(NodeIndex::new(idx))
    }

    /// Returns an iterator over the facets.
    fn facets(&self)  {
        self.graph.raw_nodes().iter().map(|f| f.weight)
    }

    /// Returns a mutable iterator over the facets.
    fn facets_mut(&mut self) -> NodeWeightsMut<Facet<'a>> {
        self.graph.node_weights_mut()
    }

    /// Fills the outer vertex sets of all facets.
    fn fill_outers(&mut self, vertices: &[&'a Point]) {
        for v in vertices {
            for f in self.facets_mut() {
                if f.position(v) == Position::Outer {
                    f.outer.push(v);

                    // At most one facet can have a given vertex.
                    break;
                }
            }
        }
    }

    /// Checks the first facet of the hull. If it has no outer vertices, it adds
    /// it to `final_facets`. Otherwise, it adds the necessary facets to the
    /// polytope.
    fn check_first_facet(&mut self) -> bool {
        // Takes the first facet in the graph.
        if let Some(f) = self.get_facet(0) {
            if let Some(p) = f.farthest() {
                for facet in self.facets() {}

                todo!()
            } else {
                // Deletes facet from graph. (Would it be faster to delete the
                // last one instead?)
                let f = self.remove_facet(0).unwrap();
                self.final_facets.push(f);
            }

            true
        } else {
            false
        }
    }

    /// Finds a set of d + 1 points not in the same hyperplane, builds a simplicial
    /// hull from them. Returs `None` if no such set of points exists.
    fn initial_hull(vertices: &[&'a Point]) -> Option<Hull<'a>> {
        let mut vert_iter = vertices.iter();
        let mut s = Subspace::new((*vert_iter.next()?).clone());
        let mut hull_vertices = Vec::new();

        for &v in vert_iter {
            if let Some(_) = s.add(v) {
                hull_vertices.push(v);

                // If we've already found d + 1 points not in a hyperplane.
                if s.is_full_rank() {
                    let mut hull = Hull::simplex(&hull_vertices);
                    hull.fill_outers(vertices);
                    return Some(hull);
                }
            }
        }

        None
    }

    // Converts the finished hull into a concrete polytope.
    fn into_polytope(&self) -> Concrete {
        debug_assert_eq!(self.graph.node_count(), 0);

        todo!()
    }
}

/// Computes the convex hull of a set of vertices. Returns `None` if the
/// vertices don't span the ambient space.
pub fn convex_hull(vertices: Vec<Point>) -> Option<Concrete> {
    // Turns the vertices into their references.
    let vertices = vertices.iter().collect::<Vec<_>>();

    // Initializes the hull.
    let mut hull = Hull::initial_hull(&vertices)?;

    // Checks the first facet with a non-empty set of outer vertices, until
    // there is none.
    while hull.check_first_facet() {}

    // Builds the polytope from the hull.
    Some(hull.into_polytope())
}
