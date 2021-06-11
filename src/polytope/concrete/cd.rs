use crate::{
    geometry::{MatrixOrd, Vector},
    Float, FloatOrd,
};
use nalgebra::{dmatrix, Dynamic, VecStorage};
use petgraph::{
    graph::{Graph, Node as GraphNode, NodeIndex},
    Undirected,
};
use std::{f64::consts::PI, fmt::Display, iter::Enumerate, iter::Peekable, mem, str::Chars};

use crate::geometry::Matrix;

pub type CdResult<T> = Result<T, CdError>;

/// Represents an error while parsing a CD.
#[derive(Clone, Copy, Debug)]
pub enum CdError {
    /// A parenthesis was opened but not closed when it should have been.
    MismatchedParenthesis(usize),

    /// The CD ended unexpectedly.
    UnexpectedEnding(usize),

    /// A number couldn't be parsed.
    ParseError(usize),

    /// An invalid symbol was found.
    InvalidSymbol(usize),
}

impl Display for CdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MismatchedParenthesis(idx) => {
                write!(f, "mismatched parenthesis at index {}", idx)
            }
            Self::UnexpectedEnding(idx) => write!(f, "CD ended unexpectedly at index {}", idx),
            Self::ParseError(idx) => write!(f, "parsing failed at index {}", idx),
            Self::InvalidSymbol(idx) => write!(f, "invalid symbol found at index {}", idx),
        }
    }
}

impl std::error::Error for CdError {}

/// Represents a Coxeter diagram as a matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdMatrix(MatrixOrd);

impl CdMatrix {
    /// Initializes a new CD matrix from a vector of nodes and a matrix.
    pub fn new(matrix: Matrix) -> Self {
        Self(MatrixOrd::new(matrix))
    }

    /// Returns a reference to the inner matrix.
    pub fn as_matrix(&self) -> &Matrix {
        self.0.as_matrix()
    }

    /// Returns a mutable reference to the inner matrix.
    pub fn as_matrix_mut(&mut self) -> &mut Matrix {
        self.0.as_matrix_mut()
    }

    /// Returns the dimensions of the matrix.
    pub fn dim(&self) -> usize {
        self.as_matrix().nrows()
    }

    /// Parses a [`Cd`] and turns it into a Coxeter matrix.
    pub fn parse(input: &str) -> CdResult<Self> {
        Cd::new(input).map(|cd| cd.cox())
    }

    /// Returns the Coxeter matrix for the trivial 1D group.
    pub fn trivial() -> Self {
        Self::new(dmatrix![1.0])
    }

    /// Returns the Coxeter matrix for the I2(x) group.
    pub fn i2(x: Float) -> Self {
        Self::from_lin_diagram(vec![x])
    }

    /// Returns the Coxeter matrix for the An group.
    pub fn a(n: usize) -> Self {
        Self::from_lin_diagram(vec![3.0; n - 1])
    }

    /// Returns the Coxeter matrix for the Bn group.
    pub fn b(n: usize) -> Self {
        let mut diagram = vec![3.0; n - 1];
        diagram[0] = 4.0;
        Self::from_lin_diagram(diagram)
    }

    /// Returns a mutable reference to the elements of the matrix.
    pub fn iter_mut(
        &mut self,
    ) -> nalgebra::iter::MatrixIterMut<Float, Dynamic, Dynamic, VecStorage<Float, Dynamic, Dynamic>>
    {
        self.0.iter_mut()
    }

    /// Creates a Coxeter matrix from a linear diagram, whose edges are
    /// described by the vector.
    pub fn from_lin_diagram(diagram: Vec<Float>) -> Self {
        let dim = diagram.len() + 1;

        Self::new(Matrix::from_fn(dim, dim, |mut i, mut j| {
            // Makes i ≤ j.
            if i > j {
                mem::swap(&mut i, &mut j);
            }

            match j - i {
                0 => 1.0,
                1 => diagram[i],
                _ => 2.0,
            }
        }))
    }
}

impl std::ops::Index<(usize, usize)> for CdMatrix {
    type Output = Float;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

/// Stores the indices of the node and edge of the next edge in the graph. This
/// is used in order to handle virtual nodes. A new edge will be added to the
/// graph only when both fields of the `EdgeMem` are full, and we're reading a
/// new node.
#[derive(Default)]
struct EdgeMem {
    node: Option<NodeIndex>,
    edge: Option<Edge>,
}

/// Possible types of CD
pub struct Cd(
    // Single {
    Graph<Node, Edge, Undirected>,
    // },
    /*
    Compound{count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceSimp{lace_len: f64, count: u32, graph: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceTower{lace_len: f64, count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    LaceRing{lace_len: f64, count: u32, graphs: Vec<Graph<NodeVal, EdgeVal, Undirected>>},
    */
);

impl Cd {
    /// Main function for parsing CDs from strings.
    pub fn new(input: &str) -> CdResult<Self> {
        let mut caret = CdBuilder {
            diagram: input.chars().enumerate().peekable(),
            graph: Graph::new_undirected(),
            edge_mem: Default::default(),
            len: input.len(),
        };

        // Reads through the diagram.
        loop {
            caret.create_node()?;

            // We continue until we find that there's no further edges.
            if let Ok(None) = caret.read_edge() {
                return Ok(Cd(caret.graph));
            }
        }
    }

    /// Returns an iterator over the nodes in the Coxeter Diagram, in the order
    /// in which they were found.
    pub fn node_iter<'a>(
        &'a self,
    ) -> std::iter::Map<std::slice::Iter<GraphNode<Node>>, impl Fn(&'a GraphNode<Node>) -> Node>
    {
        let closure = |node: &GraphNode<Node>| node.weight;
        self.0.raw_nodes().iter().map(closure)
    }

    /// Returns the nodes in the Coxeter Diagram, in the order in which they
    /// were found.
    pub fn nodes(&self) -> Vec<Node> {
        self.0.raw_nodes().iter().map(|node| node.weight).collect()
    }

    pub fn node_vector(&self) -> Vector {
        Vector::from_iterator(self.dim(), self.node_iter().map(|node| node.value()))
    }

    /// Creates a [`CdMatrix`] from a Coxeter diagram.
    pub fn cox(&self) -> CdMatrix {
        let dim = self.dim();

        let matrix = Matrix::from_fn(dim, dim, |i, j| {
            if i == j {
                return 1.0;
            }

            let node_i = NodeIndex::new(i);
            let node_j = NodeIndex::new(j);

            if let Some(idx) = self.0.find_edge(node_i, node_j) {
                self.0[idx].value()
            } else {
                2.0
            }
        });

        CdMatrix::new(matrix)
    }

    pub fn circumradius(&self) -> Float {
        let mut schlafli = self.cox();
        let node_vec = self.node_vector();

        // Converts the Coxeter matrix into the Schläfli matrix.
        for v in schlafli.as_matrix_mut().iter_mut() {
            *v = (PI / *v).cos();
        }

        (node_vec.transpose() * schlafli.as_matrix() * node_vec)[(0, 0)].sqrt()
    }

    /// Returns the number of edges in the CD.
    pub fn edge_count(&self) -> usize {
        self.0.edge_count()
    }

    /// The dimension of the polytope the CD describes.
    pub fn dim(&self) -> usize {
        self.0.node_count()
    }
}

impl From<Cd> for CdMatrix {
    fn from(cd: Cd) -> Self {
        cd.cox()
    }
}

impl Display for Cd {
    ///Prints the node and edge count, along with the value each node and edge contains
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Prints node and edge counts.
        writeln!(f, "{} Nodes", self.dim())?;
        writeln!(f, "{} Edges", self.edge_count())?;

        // Prints out nodes.
        for (i, n) in self.0.raw_nodes().iter().enumerate() {
            write!(f, "Node {}: {}", i, n.weight)?;
        }

        // Prints out edges.
        for (i, e) in self.0.raw_edges().iter().enumerate() {
            write!(f, "Edge {}: {}", i, e.weight)?;
        }

        Ok(())
    }
}

/// A node in a [`Cd`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    ///Unringed Nodes (different from Ringed(0))
    Unringed,

    /// Ringed Nodes, can hold any float
    Ringed(FloatOrd),

    ///Snub Nodes, should definitely make this hold a float
    ///TODO: Agree on a way to specify the length in a snub node
    Snub,
}

impl Node {
    pub fn value(&self) -> Float {
        match self {
            Self::Unringed | Self::Snub => 0.0,
            Self::Ringed(val) => val.0,
        }
    }
    /// Shorthand for `NodeVal::Ringed(FloatOrd::from(x))`.
    pub fn ringed(x: Float) -> Self {
        Self::Ringed(FloatOrd::from(x))
    }
}

impl Display for Node {
    /// Prints the value that a node contains.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Unringed => writeln!(f, "Node is unringed"),
            Node::Ringed(x) => writeln!(f, "Node carries {}", x.0),
            Node::Snub => writeln!(f, "Node is snub"),
        }
    }
}

/// An edge in a [`Cd`].
#[derive(Clone, Copy)]
enum Edge {
    /// Any edge that represents a rational number, including integer edges.
    Rational(i64, i64),

    /// Represents an ∞ symbol for prograde infinity.
    Inf,

    /// Represents an ∞' symbol for retrograde infinity.
    RetInf,

    /// No intersection Ø.
    Non,
}

impl Edge {
    pub fn value(&self) -> f64 {
        match *self {
            Self::Rational(n, d) => n as f64 / d as f64,
            Self::Inf => f64::INFINITY,
            Self::RetInf => 1.0,
            Self::Non => f64::NAN,
        }
    }
}

impl Display for Edge {
    ///Prints the value an edge contains
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Edge::Rational(n, d) => writeln!(f, "Edge carries {}/{}", n, d),
            Edge::Inf => writeln!(f, "Edge carries prograde ∞"),
            Edge::RetInf => writeln!(f, "Edge carries retrograde ∞"),
            Edge::Non => writeln!(f, "Edge carries Ø"),
        }
    }
}

/// Packages important information needed to interpret CDs
pub struct CdBuilder<'a> {
    /// A peekable iterator over the characters of the diagram and their indices.
    diagram: Peekable<Enumerate<Chars<'a>>>,

    /// Represents the CD itself.
    graph: Graph<Node, Edge, Undirected>,
    edge_mem: EdgeMem,

    /// The length of the diagram.
    len: usize,
}

/// Operations that are commonly done to parse CDs.
impl<'a> CdBuilder<'a> {
    pub fn next_or(&mut self) -> CdResult<(usize, char)> {
        self.diagram
            .next()
            .ok_or(CdError::UnexpectedEnding(self.len))
    }

    /// Reads the next node in the diagram. Returns `Ok(())` if succesful, and
    /// a [`CdResult`] otherwise.
    pub fn create_node(&mut self) -> CdResult<()> {
        let mut chars = Vec::new();
        let (idx, c) = self.next_or()?;
        chars.push(c);

        // The index of the new node.
        let mut new_node = NodeIndex::new(self.graph.node_count());

        match c {
            // If the node is various characters inside parentheses.
            '(' => {
                let mut mismatch = true;

                // We read through the diagram until we find ')'.
                while let Ok((idx, c)) = self.next_or() {
                    chars.push(c);

                    if c == ')' {
                        // Converts the read characters into a value and adds the node to the graph.
                        self.graph.add_node(node_to_val(chars, idx)?);
                        mismatch = false;
                        break;
                    }
                }

                // If the parenthesis isn't closed.
                if mismatch {
                    return Err(CdError::MismatchedParenthesis(self.len));
                }
            }

            // If the node is a virtual node.
            '*' => {
                // Reads the index the virtual node refers to.
                let (idx, c) = self.next_or()?;
                let v_idx = NodeIndex::new(match u8::from_str_radix(&c.to_string(), 36) {
                    // Invalid syntax.
                    Ok(0..=9) | Err(_) => return Err(CdError::InvalidSymbol(idx)),

                    // A virtual node, from *a to *z.
                    Ok(c) => (c - 10) as usize,
                });

                // Sets the index of the new node to be where the virtual node is refering to.
                new_node = v_idx
            }

            // If the node is a single character.
            _ => {
                // Converts the read characters into a value and adds the node to the graph.
                self.graph.add_node(node_to_val(chars, idx)?);
            }
        }

        // If the EdgeMem is full, we add a new edge to the graph.
        if let EdgeMem {
            node: Some(prev_node),
            edge: Some(edge),
        } = &self.edge_mem
        {
            self.graph.add_edge(*prev_node, new_node, *edge);
        };

        // Resets the EdgeMem so that it only has the node that was just found.
        self.edge_mem = EdgeMem {
            node: Some(new_node),
            edge: None,
        };

        Ok(())
    }

    /// Reads an edge from a CD and stores into edgemem
    pub fn read_edge(&mut self) -> CdResult<Option<()>> {
        let mut chars = Vec::new();

        // We read through the diagram until we encounter something that looks like the start of a node
        while let Some(&(idx, d)) = self.diagram.peek() {
            if d.is_alphabetic() || d == '(' || d == '*' {
                // Adds the edge value to edge_mem
                self.edge_mem.edge = Some(edge_to_val(chars, idx)?);
                return Ok(Some(()));
            }

            // Here, we want to look ahead before adding characters,
            // so we don't add the first character of the next node
            chars.push(self.next_or()?.1);
        }

        //If we unexpectedly hit the end of the iterator, exit and return None
        Ok(None)
    }

    /*
    ///Reads a lace suffix
    fn read_suff(&self) -> Option<Caret> {}
    */
}

/// Attempts to parse a `String`, returns a [`CdError`] if it fails.
fn parse<R: std::str::FromStr>(string: &str, idx: usize) -> CdResult<R> {
    string.parse().map_err(|_| CdError::ParseError(idx))
}

/// Converts a slice of characters into a wrapped edge value.
///
/// `idx` is the index of the last character in `raw`.
fn edge_to_val(raw: Vec<char>, idx: usize) -> CdResult<Edge> {
    let len = raw.len();
    let mut raw_iter = raw
        .into_iter()
        .enumerate()
        .map(|(str_idx, c)| (idx + str_idx + 1 - len, c));

    let mut edge = String::new();

    let mut numerator = None;
    let (_, c) = raw_iter.next().expect("Slice can't be empty!");

    // Starting character
    edge.push(c);

    // If the value is Rational or an Integer
    if c.is_digit(10) {
        for (idx, c) in raw_iter {
            // If the "/" is encountered
            if c == '/' {
                // Parse and save the numerator
                numerator = Some(parse(&edge, idx)?);

                // Reset what's being read.
                edge = String::new();
            };

            // Wasn't a special character, can continue
            edge.push(c);
        }

        // When you're at the end
        // Parse the end value
        let den = parse(&edge, idx)?;

        // If this was a Rational edge, the end value would be the denominator
        // If this wasn't a Rational edge, the end value would be the numerator
        match numerator {
            Some(num) => Ok(Edge::Rational(num, den)),
            None => Ok(Edge::Rational(den, 1i64)),
        }
    } else {
        // For miscellaneous edge symbols,
        // just read the whole thing as a string
        match edge.as_str() {
            "∞" => Ok(Edge::Inf),
            "∞'" | "'∞" => Ok(Edge::RetInf),
            "Ø" => Ok(Edge::Non),
            _ => Err(CdError::InvalidSymbol(idx)),
        }
    }
}

/// Converts Vecs of chars to wrapped NodeVals
///
/// `idx` is the index of the last character in `raw`.
fn node_to_val(raw: Vec<char>, idx: usize) -> CdResult<Node> {
    let len = raw.len();
    let mut raw_iter = raw
        .into_iter()
        .enumerate()
        .map(|(str_idx, c)| (idx + str_idx + 1 - len, c));

    let mut node = Vec::new();
    let mut sign = 1.0;

    let (mut idx, mut c) = raw_iter.next().expect("Node can't be empty!");

    /// Skips a character from the string, returns a mismatched parenthesis
    /// error if there's no subsequent character.
    macro_rules! skip_char {
        () => {
            idx += 1;
            c = raw_iter
                .next()
                .ok_or(CdError::MismatchedParenthesis(idx))?
                .1;
        };
    }

    // Skips any opening parenthesis.
    if c == '(' {
        skip_char!();
    }

    // Skips a minus sign.
    if c == '-' {
        sign = -1.0;
        skip_char!();
    }

    // Starting character
    node.push(c);

    // If the node has a custom value
    if c.is_digit(10) {
        for (idx, c) in raw_iter {
            // When you're at the end
            if c == ')' {
                // Parse the value
                let val: f64 = parse(&node.into_iter().collect::<String>(), idx)?;

                return if val.is_nan() {
                    Err(CdError::InvalidSymbol(idx))
                } else {
                    Ok(Node::ringed(sign * val))
                };
            }

            // This character was normal, can continue
            node.push(c);
        }

        // We never found the matching parenthesis.
        Err(CdError::MismatchedParenthesis(idx))
    } else {
        // Check shortchord values
        Ok(Node::ringed(match c {
            'o' => return Ok(Node::Unringed),
            's' => return Ok(Node::Snub),
            'v' => (5f64.sqrt() - 1f64) / 2f64,
            'x' => 1f64,
            'q' => 2f64.sqrt(),
            'f' => (5f64.sqrt() + 1f64) / 2f64,
            'h' => 3f64.sqrt(),
            'k' => (2f64.sqrt() + 2f64).sqrt(),
            'u' => 2f64,
            'w' => 2f64.sqrt() + 1f64,
            'F' => (5f64.sqrt() + 3f64) / 2f64,
            'e' => 3f64.sqrt() + 1f64,
            'Q' => 2f64.sqrt() * 2f64,
            'd' => 3f64,
            'V' => 5f64.sqrt() + 1f64,
            'U' => 2f64.sqrt() + 2f64,
            'A' => (5f64.sqrt() + 5f64) / 4f64,
            'X' => 2f64.sqrt() * 2f64 + 1f64,
            'B' => 5f64.sqrt() + 2f64,
            _ => return Err(CdError::InvalidSymbol(idx)),
        }))
    }
}

/// Inverts the value held by a EdgeVal
fn num_retro(val: Edge) -> Edge {
    use Edge::*;

    match val {
        Rational(n, d) => Rational(n, n - d),
        Inf => RetInf,
        RetInf => Inf,
        Non => Non,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    fn x() -> Node {
        Node::ringed(1.0)
    }

    fn o() -> Node {
        Node::Unringed
    }

    fn s() -> Node {
        Node::Snub
    }

    fn test(input: &str, nodes: Vec<Node>, matrix: Matrix) {
        let cd = Cd::new(input).unwrap();
        assert_eq!(cd.nodes(), nodes, "Node mismatch!");
        assert_eq!(cd.cox(), CdMatrix::new(matrix), "Coxeter matrix mismatch!");
    }

    #[test]
    fn i2_10() {
        test(
            "x10x",
            vec![x(), x()],
            dmatrix![
                1.0, 10.0;
                10.0, 1.0
            ],
        )
    }

    #[test]
    fn a3() {
        test(
            "x3o3x",
            vec![x(), o(), x()],
            dmatrix![
                1.0, 3.0, 2.0;
                3.0, 1.0, 3.0;
                2.0, 3.0, 1.0
            ],
        )
    }

    #[test]
    fn e6() {
        test(
            "x3o3o3o3o *c3o",
            vec![x(), o(), o(), o(), o(), o()],
            dmatrix![
                1.0, 3.0, 2.0, 2.0, 2.0, 2.0;
                3.0, 1.0, 3.0, 2.0, 2.0, 2.0;
                2.0, 3.0, 1.0, 3.0, 2.0, 3.0;
                2.0, 2.0, 3.0, 1.0, 3.0, 2.0;
                2.0, 2.0, 2.0, 3.0, 1.0, 2.0;
                2.0, 2.0, 3.0, 2.0, 2.0, 1.0
            ],
        )
    }

    #[test]
    fn node_lengths() {
        test(
            "(1.0)4(2.2)3(-3.0)",
            vec![Node::ringed(1.0), Node::ringed(2.2), Node::ringed(-3.0)],
            dmatrix![
                1.0, 4.0, 2.0;
                4.0, 1.0, 3.0;
                2.0, 3.0, 1.0
            ],
        )
    }

    #[test]
    fn snubs() {
        test(
            "s4s3o4o",
            vec![s(), s(), o(), o()],
            dmatrix![
                1.0, 4.0, 2.0, 2.0;
                4.0, 1.0, 3.0, 2.0;
                2.0, 3.0, 1.0, 4.0;
                2.0, 2.0, 4.0, 1.0
            ],
        )
    }

    #[test]
    fn shortchords() {
        test(
            "v4x3F4f",
            vec![
                Node::ringed((5f64.sqrt() - 1f64) / 2f64),
                x(),
                Node::ringed((5f64.sqrt() + 3f64) / 2f64),
                Node::ringed((5f64.sqrt() + 1f64) / 2f64),
            ],
            dmatrix![
                1.0, 4.0, 2.0, 2.0;
                4.0, 1.0, 3.0, 2.0;
                2.0, 3.0, 1.0, 4.0;
                2.0, 2.0, 4.0, 1.0
            ],
        )
    }
}
