//! Defines the basic types for a Coxeter diagram.

use std::fmt::Display;

use petgraph::graph::{Edge as GraphEdge, Node as GraphNode, NodeIndex, UnGraph};

use crate::{
    float::Float,
    geometry::{Matrix, Point, Vector},
};

use super::{parse::CdBuilder, Cox};

/// The result of an operation involving Coxeter diagram parsing.
pub type CdResult<T> = Result<T, CdError>;

/// Represents an error while parsing a CD.
#[derive(Clone, Copy, Debug)]
pub enum CdError {
    /// A parenthesis was opened but not closed.
    MismatchedParenthesis {
        /// The position at which the reader found the error.
        pos: usize,
    },

    /// The diagram ended unexpectedly.
    UnexpectedEnding {
        /// The position at which the reader found the error.
        pos: usize,
    },

    /// A number couldn't be parsed.
    ParseError {
        /// The position at which the reader found the error.
        pos: usize,
    },

    /// An invalid symbol was found.
    InvalidSymbol {
        /// The position at which the reader found the error.
        pos: usize,
    },

    /// An invalid edge was found.
    InvalidEdge {
        /// The numerator of the invalid edge.
        num: u32,

        /// The denominator of the invalid edge.
        den: u32,

        /// The position at which the reader found the error.
        pos: usize,
    },

    /// An edge was specified twice.
    RepeatEdge {
        /// The first node in the duplicated edge.
        a: usize,

        /// The second node in the duplicated edge.
        b: usize,
    },
}

impl Display for CdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            // A parenthesis was opened but not closed.
            Self::MismatchedParenthesis { pos } => {
                write!(f, "mismatched parenthesis at position {}", pos)
            }

            // The diagram ended unexpectedly.
            Self::UnexpectedEnding { pos } => {
                write!(f, "CD ended unexpectedly at position {}", pos)
            }

            // A number couldn't be parsed.
            Self::ParseError { pos } => {
                write!(f, "parsing failed at position {}", pos)
            }

            // An invalid symbol was found.
            Self::InvalidSymbol { pos } => write!(f, "invalid symbol found at position {}", pos),

            // An invalid edge was found.
            Self::InvalidEdge { num, den, pos } => {
                write!(f, "invalid edge {}/{} at position {}", num, den, pos)
            }

            // An edge was specified twice.
            Self::RepeatEdge { a, b } => {
                write!(f, "repeat edge between {} and {}", a, b)
            }
        }
    }
}

impl std::error::Error for CdError {}

/// A node in a [`Cd`]. Represents a mirror in hyperspace, and specifies both
/// where a generator point should be located with respect to it, and how it
/// should interact with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Node {
    /// An unringed node. Represents a mirror that contains the generator point.
    /// Crucially, reflecting the generator through this mirror doesn't create a
    /// new edge.
    Unringed,

    /// A ringed node. Represents a mirror at (half) a certain distance from the
    /// generator. Reflecting the generator through this mirror creates an edge.
    Ringed(f64),

    /// A snub node. Represents a mirror at (half) a certain distance from the
    /// generator. In contrast to [`Self::Ringed`] nodes, the generator point
    /// and its reflection through this mirror can't simultaneously be in the
    /// polytope.
    Snub(f64),
}

impl Node {
    /// Returns twice the distance from the generator point to the hyperplane
    /// corresponding to this node.
    pub fn value(&self) -> f64 {
        match self {
            Self::Unringed => 0.0,
            Self::Ringed(val) | Self::Snub(val) => *val,
        }
    }

    /// Shorthand for `NodeVal::Ringed(x)`.
    pub fn ringed(x: f64) -> Self {
        Self::Ringed(x)
    }

    /// Shorthand for `NodeVal::Snub(x)`.
    pub fn snub(x: f64) -> Self {
        Self::Snub(x)
    }

    /// Returns whether this node is ringed.
    pub fn is_ringed(&self) -> bool {
        matches!(self, Self::Ringed(_))
    }

    /// Converts the character into a node value, using [Wendy Krieger's
    /// scheme](https://polytope.miraheze.org/wiki/Coxeter_diagram#Different_edge_lengths).
    ///
    /// # Todo
    /// Make this customizable?
    pub fn from_char(c: char) -> Option<Self> {
        Some(Node::ringed(match c {
            'o' => return Some(Node::Unringed),
            's' => return Some(Node::snub(f64::ONE)),
            'v' => (f64::SQRT_5 - f64::ONE) / f64::TWO,
            'x' => f64::ONE,
            'q' => f64::SQRT_2,
            'f' => (f64::SQRT_5 + f64::ONE) / f64::TWO,
            'h' => f64::SQRT_3,
            'k' => (f64::SQRT_2 + f64::TWO).fsqrt(),
            'u' => f64::TWO,
            'w' => f64::SQRT_2 + f64::ONE,
            'F' => (f64::SQRT_5 + f64::THREE) / f64::TWO,
            'e' => f64::SQRT_3 + f64::ONE,
            'Q' => f64::SQRT_2 * f64::TWO,
            'd' => f64::THREE,
            'V' => f64::SQRT_5 + f64::ONE,
            'U' => f64::SQRT_2 + f64::TWO,
            'A' => (f64::SQRT_5 + f64::ONE) / f64::FOUR + f64::ONE,
            'X' => f64::SQRT_2 * f64::TWO + f64::ONE,
            'B' => f64::SQRT_5 + f64::TWO,
            _ => return None,
        }))
    }

    /// Attempts to convert a character into a [`Node`]. Returns a
    /// [`CdError::InvalidSymbol`] if it fails.
    pub fn from_char_or(c: char, pos: usize) -> CdResult<Self> {
        Self::from_char(c).ok_or(CdError::InvalidSymbol { pos })
    }
}

impl Display for Node {
    /// Prints the value that a node contains.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Unringed => writeln!(f, "o"),
            Node::Ringed(x) => writeln!(f, "x({})", x),
            Node::Snub(s) => writeln!(f, "s({})", s),
        }
    }
}

/// Represents the value of an edge in a [`Cd`]. An edge with a value of `x`
/// represents an angle of π / *x* between two hyperplanes.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    /// The numerator of the edge.
    pub num: u32,

    /// The denominator of the edge.
    pub den: u32,
}

impl Edge {
    /// Initializes a new edge from a given numerator and denominator. If these
    /// are invalid, returns a [`CdError::InvalidEdge`].
    pub fn rational(num: u32, den: u32, pos: usize) -> CdResult<Self> {
        if num > 1 && den != 0 && den < num {
            Ok(Self { num, den })
        } else {
            Err(CdError::InvalidEdge { num, den, pos })
        }
    }

    /// Initializes a new edge from a given integral value. If this is invalid,
    /// returns a [`CdError::InvalidEdge`] using the specified position.
    pub fn int(num: u32, pos: usize) -> CdResult<Self> {
        Self::rational(num, 1, pos)
    }

    /// Returns the numerical value of the edge.
    pub fn value(&self) -> f64 {
        f64::u32(self.num) / f64::u32(self.den)
    }

    /// Returns `true` if the edge stores any value equivalent to 2.
    pub fn eq_two(&self) -> bool {
        self.num == self.den * 2
    }
}

impl Display for Edge {
    /// Prints the value contained in an edge.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{} / {}", self.num, self.den)
        }
    }
}

/// Stores the position of a node, which can either be its index in the array or
/// its offset from the end of the array.
///
/// This is necessary since we can't figure out what node a virtual node like
/// `*-a` is referring to unless we've read the entire diagram already.
#[derive(Clone, Copy)]
pub enum NodeRef {
    /// The index of a node.
    Absolute(usize),

    /// The offset of the node from the array's ending.
    Negative(usize),
}

impl NodeRef {
    /// Initializes a new node reference from an index. The `neg` parameter
    /// determines if indexing should be [`Negative`](Self::Negative) or
    /// [`Absolute`](Self::Absolute).
    pub fn new(neg: bool, idx: usize) -> Self {
        if neg {
            Self::Negative(idx)
        } else {
            Self::Absolute(idx)
        }
    }

    /// Returns the index in the graph that the node reference represents.
    /// Requires knowing the number of nodes in the graph.
    pub fn index(&self, len: usize) -> NodeIndex {
        NodeIndex::new(match *self {
            Self::Absolute(idx) => idx,
            Self::Negative(idx) => len - 1 - idx,
        })
    }
}

/// Stores the [`NodeRef`]s of both ends of an edge, along with its value.
#[derive(Clone, Copy)]
pub struct EdgeRef {
    /// The reference to the first node in the edge.
    pub first: NodeRef,

    /// The reference to the other node in the edge.
    pub other: NodeRef,

    /// The edge value.
    pub edge: Edge,
}

impl EdgeRef {
    /// Initializes a new edge reference from its fields.
    pub fn new(first: NodeRef, other: NodeRef, edge: Edge) -> Self {
        Self { first, other, edge }
    }

    /// Returns the index in the graph of both node references. Requires knowing
    /// the number of nodes in the graph.
    pub fn indices(&self, len: usize) -> [NodeIndex; 2] {
        [self.first.index(len), self.other.index(len)]
    }
}

/// Encodes a [Coxeter diagram](https://polytope.miraheze.org/wiki/Coxeter_diagram)
/// or CD as an undirected labeled graph.
///
/// A Coxeter diagram serves two main functions. It serves as a representation
/// for certain polytopes called [Wythoffians](https://polytope.miraheze.org/wiki/Wythoffian),
/// and as a representation for certain symmetry groups called
/// [Coxeter groups](https://polytope.miraheze.org/wiki/Coxeter_group).
///
/// Each [`Node`] a Coxeter diagram represents a mirror (or hyperplane) in
/// *n*-dimensional space. If two nodes are joined by an [`Edge`] with a value
/// of x, it means that the angle between the mirrors they represent is given
/// by π / x. If two nodes aren't joined by any edge, it means that they are
/// perpendicular.
///
/// To actually build a Coxeter diagram, we use a [`CdBuilder`].
#[derive(Default)]
pub struct Cd(UnGraph<Node, Edge>);

impl Cd {
    /// Initializes a new Coxeter diagram with no nodes nor edges.
    pub fn new() -> Self {
        Default::default()
    }

    /// Parses a Coxeter diagram from ASCII inline notation. For more
    /// information, see [`CdBuilder`].
    pub fn parse(input: &str) -> CdResult<Self> {
        CdBuilder::new(input).build()
    }

    /// The dimension of the polytope the Coxeter diagram describes.
    pub fn dim(&self) -> usize {
        self.node_count()
    }

    /// Returns the number of nodes in the Coxeter diagram.
    pub fn node_count(&self) -> usize {
        self.0.node_count()
    }

    /// Returns the number of edges in the Coxeter diagram.
    pub fn edge_count(&self) -> usize {
        self.0.edge_count()
    }

    /// Returns a reference to the raw node array.
    pub fn raw_nodes(&self) -> &[GraphNode<Node>] {
        self.0.raw_nodes()
    }

    /// Returns a reference to the raw edge array.
    pub fn raw_edges(&self) -> &[GraphEdge<Edge>] {
        self.0.raw_edges()
    }

    /// Adds a node into the Coxeter diagram.
    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        self.0.add_node(node)
    }

    /// Adds an edge into the Coxeter diagram.
    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, edge: Edge) -> CdResult<()> {
        if !edge.eq_two() {
            if self.0.contains_edge(a, b) {
                return Err(CdError::RepeatEdge {
                    a: a.index(),
                    b: b.index(),
                });
            }

            self.0.add_edge(a, b, edge);
        }

        Ok(())
    }

    /// Returns an iterator over the nodes in the Coxeter diagram, in the order
    /// in which they were found.
    pub fn node_iter(&self) -> impl Iterator<Item = Node> + '_ {
        self.0.raw_nodes().iter().map(|node| node.weight)
    }

    /// Returns the nodes in the Coxeter diagram, in the order in which they
    /// were found.
    pub fn nodes(&self) -> Vec<Node> {
        self.node_iter().collect()
    }

    /// Returns the vector whose values represent the node values.
    pub fn node_vector(&self) -> Vector<f64> {
        Vector::from_iterator(self.dim(), self.node_iter().map(|node| node.value()))
    }

    /// Returns whether a CD is minimal, i.e. whether every connected component
    /// has at least one ringed node.
    pub fn minimal(&self) -> bool {
        'COMPONENT: for component in petgraph::algo::tarjan_scc(&self.0) {
            for node in component {
                if self.0[node].is_ringed() {
                    continue 'COMPONENT;
                }
            }

            return false;
        }

        true
    }

    /// Creates a [`Cox`] from a Coxeter diagram.
    pub fn cox(&self) -> Cox<f64> {
        let dim = self.dim();
        let graph = &self.0;

        let matrix = Matrix::from_fn(dim, dim, |i, j| {
            // Every entry in the diagonal of a Coxeter matrix is 1.
            if i == j {
                return 1.0;
            }

            // If an edge connects two nodes, it adds its value to the matrix.
            if let Some(idx) = graph.find_edge(NodeIndex::new(i), NodeIndex::new(j)) {
                graph[idx].value()
            }
            // Else, we write a 2.
            else {
                2.0
            }
        });

        Cox::new(matrix)
    }

    /// Returns the circumradius of the polytope specified by the matrix, or
    /// `None` if this doesn't apply. This is just
    /// calling [`Self::generator`] and taking the norm.
    pub fn circumradius(&self) -> Option<f64> {
        self.generator().as_ref().map(Point::norm)
    }

    /// Returns a point in the position specified by the Coxeter diagram,
    /// using the set of mirrors generated by [`Cox::normals`].    
    pub fn generator(&self) -> Option<Point<f64>> {
        let mut vector = self.node_vector();

        self.cox()
            .normals()?
            .solve_upper_triangular_mut(&mut vector)
            .then(|| vector)
    }
}

impl From<Cd> for Cox<f64> {
    fn from(cd: Cd) -> Self {
        cd.cox()
    }
}

impl Display for Cd {
    /// Prints the node and edge count, along with the value each node and edge contains
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Prints node and edge counts.
        writeln!(f, "{} Nodes", self.dim())?;
        writeln!(f, "{} Edges", self.edge_count())?;

        // Prints out nodes.
        for (i, n) in self.raw_nodes().iter().enumerate() {
            write!(f, "Node {}: {}", i, n.weight)?;
        }

        // Prints out edges.
        for (i, e) in self.raw_edges().iter().enumerate() {
            write!(f, "Edge {}: {}", i, e.weight)?;
        }

        Ok(())
    }
}
