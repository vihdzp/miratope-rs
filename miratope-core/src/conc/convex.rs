use std::collections::BTreeSet;
use std::f64::NEG_INFINITY;

use super::Concrete;
use petgraph::{graph::NodeIndex, Directed, Direction, Graph};

/// An entry in the priority queue used in Shell.
enum QueueEntry<'a> {
    /// Represents the event where at a certain time, the first facet that
    /// contains a certain point becomes visible. This facet will have the
    /// specified vertices and normal vector.
    Point {
        time: f64,
        normal: Vector,
        point: Point,
        vertices: Vec<Point>,
    },

    /// Represents the event where at a certain time, a facet containing a
    /// horizon peak and the horizon ridges specified by an element's neighbors
    /// becomes visible. This facet will have the specified normal vector.
    Peak {
        time: f64,
        normal: Vector,
        element: ShellElement<'a>,
    },
}

impl<'a> QueueEntry<'a> {
    /// Returns the time associated with an event.
    pub fn time(&self) -> f64 {
        match self {
            QueueEntry::Point { time: t, .. } => *t,
            QueueEntry::Peak { time: t, .. } => *t,
        }
    }
}

impl<'a> PartialEq for QueueEntry<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.time() == other.time()
    }
}

impl<'a> Eq for QueueEntry<'a> {}

impl<'a> PartialOrd for QueueEntry<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.time().partial_cmp(&other.time())
    }
}

impl<'a> Ord for QueueEntry<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// The metadata returned after deleting an element from a [`ShellQueue`]. This
/// data specifies the minimum time in the queue, the union of all
/// [`Point`](ShellQueueEntry::Point) and [`Peak`](ShellQueueEntry::Peak)
/// entries at this time, and their common normal vector.
struct QueueData<'a> {
    time: f64,
    normal: Vector,
    vertices: Vec<Point>,
    elements: Vec<ShellElement<'a>>,
    points: Vec<Point>,
}

#[derive(Deref, DerefMut)]
struct Queue<'a>(BTreeSet<QueueEntry<'a>>);

impl<'a> Queue<'a> {
    pub fn new() -> Self {
        Self(BTreeSet::new())
    }

    pub fn delete_min() -> QueueData<'a> {
        todo!()
    }
}

/// An element produced by the Shell algorithm. Note that the data it contains
/// is quite different from that of a [`Element`](super::Element).
struct ShellElement<'a> {
    normal: Vector,
    neighbors: Vec<NodeIndex>,
    queue: &'a Queue<'a>,
}

struct ShellEdge<'a>(&'a Point);

struct Line(Vector, Vector);

struct ShellPolytope<'a> {
    dim: usize,
    graph: Graph<ShellElement<'a>, ShellEdge<'a>, Directed>,
}

impl<'a> ShellPolytope<'a> {
    fn new(dim: usize) -> Self {
        todo!()
    }

    fn convex_hull(vertices: Vec<Point>) -> Concrete {
        let s = Subspace::from_points(&vertices);

        // A vector that is contained in s, but is in "general position."
        let y = 0.57 * &s.basis[0] + 0.43 * &s.basis[1];
        let a = s.orthogonal_comp();
        let x: Point = vertices.iter().sum::<Point>() / vertices.len() as f64;

        let vertices = vertices.iter().collect::<Vec<_>>();

        let mut poly = Self::new(vertices[0].nrows());

        poly.shell(
            Line(y, x),
            NEG_INFINITY,
            Vec::new(),
            Vec::new(),
            vertices.clone(),
            vertices,
            a,
        );

        poly.into()
    }

    fn shell(
        &mut self,
        line: Line,
        time: f64,
        ff: Vec<NodeIndex>,
        hr: Vec<NodeIndex>,
        u: Vec<&'a Point>,
        t: Vec<&'a Point>,
        n: Vec<Vector>,
    ) -> ShellElement<'a> {
        // Step 1
        let q = self.graph.add_node(ShellElement {
            normal: vec![].into(),
            neighbors: Vec::new(),
            queue: &Queue::new(),
        });

        for f in ff {
            self.graph.add_edge(f, q, weight);
        }

        // Step 2
        if t.len() == 1 {
            let p = t[0];

            self.graph.add_edge(
                q,
                if let Some(&e) = ff.get(0) {
                    e
                } else {
                    self.graph.add_node(ShellElement {
                        normal: Vector::zeros(self.dim),
                        neighbors: Vec::new(),
                    })
                },
                ShellEdge(&p),
            );
        }

        // Step 3
        let mut hp = Vec::new();
        for f in hr {
            for g in self.graph.neighbors_directed(f, Direction::Outgoing) {
                hp.push(g);
                self.graph[g].neighbors.push(f);
            }
        }
        todo!()
    }
}

impl<'a> Into<Concrete> for ShellPolytope<'a> {
    fn into(self) -> Concrete {
        todo!()
    }
}

impl Concrete {
    pub fn convex_hull_plus(&self) -> Concrete {
        convex_hull(self.vertices.clone())
    }
}
