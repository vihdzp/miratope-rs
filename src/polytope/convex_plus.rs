use super::{
    geometry::{Hyperplane, Point, Subspace},
    Concrete,
};
use float_ord::FloatOrd;
use petgraph::graph::{NodeIndex, NodeWeightsMut, UnGraph};

/// Represents a facet in a convex hull.
struct Facet {
    hyperplane: Hyperplane,
    outer: Vec<Point>,
}

impl Facet {
    fn new(hyperplane: Hyperplane) -> Self {
        Facet {
            hyperplane,
            outer: Vec::new(),
        }
    }

    fn farthest(&self) -> Option<&Point> {
        self.outer
            .iter()
            .max_by_key(|p| FloatOrd(self.hyperplane.distance(p)))
    }
}

type Ridge = ();

/// Represents a convex hull. Is made out of facets and ridges, and stores them
/// as an incidence graph.
struct Hull {
    /// An undirected graph representing adjacencies between facets. Nodes in
    /// the graph represent facets, while edges represent ridges.
    graph: UnGraph<Facet, Ridge>,

    /// The vector of facets that we've verified are on the convex hull. Is
    /// initialized empty, but should have precisely the polytope's facets at
    /// the end.
    final_facets: Vec<Facet>,
}

impl Hull {
    /// Builds a new `Hull` from the facet-ridge graph.
    fn new(graph: UnGraph<Facet, Ridge>) -> Self {
        Self {
            graph,
            final_facets: Vec::new(),
        }
    }

    /// Builds a simplicial hull from a set of vertices.
    fn simplex(vertices: &[Point]) -> Self {
        let n = vertices.len();
        let mut graph = UnGraph::with_capacity(n, n * (n - 1) / 2);

        // Adds facets. Each facet has all vertices but one, and is oriented
        // away of the remaining one.
        for i in 0..n {
            graph.add_node(Facet::new(Hyperplane::new(
                Subspace::from_points(
                    vertices[0..i]
                        .iter()
                        .chain(vertices[(i + 1)..n].iter())
                        .cloned()
                        .collect(),
                ),
                &vertices[i],
            )));
        }

        // Adds ridges.
        for i in 1..n {
            for j in 0..i {
                graph.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
            }
        }

        Self::new(graph)
    }

    fn get_facet(&self, idx: usize) -> Option<&Facet> {
        self.graph.node_weight(NodeIndex::new(idx))
    }

    fn remove_facet(&mut self, idx: usize) -> Option<Facet> {
        self.graph.remove_node(NodeIndex::new(idx))
    }

    /// Returns a mutable iterator over the facets.
    fn facets_mut(&mut self) -> NodeWeightsMut<Facet> {
        self.graph.node_weights_mut()
    }

    /// Fills the outer vertex sets of all facets.
    fn fill(&mut self, vertices: Vec<Point>) {
        for v in vertices {
            for f in self.facets_mut() {
                if f.hyperplane.is_outer(&v) {
                    f.outer.push(v);
                    break;
                }
            }
        }

        // All internal vertices will be deleted here.
    }

    /// Checks the first facet of the hull. If it has no outer vertices, it adds
    /// it to `final_facets`. Otherwise, it adds the necessary facets to the
    /// polytope.
    fn check_first_facet(&mut self) -> bool {
        // Takes the first facet in the graph.
        if let Some(f) = self.get_facet(0) {
            if let Some(p) = f.farthest() {
                // Dfs...
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
}

/// Finds a set of d + 1 points not in the same hyperplane, builds a simplicial
/// hull from them. Returs `None` if no such set of points exists.
fn initial_hull(vertices: &[Point]) -> Option<Hull> {
    let mut vert_iter = vertices.iter();
    let mut s = Subspace::new(vert_iter.next()?.clone());
    let mut hull_vertices = Vec::new();

    for v in vert_iter {
        if let Some(_) = s.add(v) {
            hull_vertices.push(v.clone());

            // If we've already found d + 1 points not in a hyperplane.
            if s.is_full_rank() {
                return Some(Hull::simplex(&hull_vertices));
            }
        }
    }

    None
}

/// Computes the convex hull of a set of vertices. Returns `None` if the
/// vertices don't span the ambient space.
pub fn convex_hull(vertices: Vec<Point>) -> Option<Concrete> {
    let mut hull = initial_hull(&vertices)?;
    hull.fill(vertices);

    todo!()
}
