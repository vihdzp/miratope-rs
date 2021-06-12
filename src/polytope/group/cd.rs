use std::{collections::VecDeque, f64::consts::PI, fmt::Display, iter, mem, str::FromStr};

use crate::{
    geometry::{Matrix, MatrixOrd, Point, Vector},
    Consts, Float, FloatOrd,
};

use nalgebra::{dmatrix, Dynamic, VecStorage};
use petgraph::{
    graph::{Graph, Node as GraphNode, NodeIndex},
    Undirected,
};

/// The result of an operation involving Coxeter diagram parsing.
pub type CdResult<T> = Result<T, CdError>;

/// Represents an error while parsing a CD.
#[derive(Clone, Copy, Debug)]
pub enum CdError {
    /// A parenthesis was opened but not closed.
    MismatchedParenthesis { pos: usize },

    /// The diagram ended unexpectedly.
    UnexpectedEnding { pos: usize },

    /// A number couldn't be parsed.
    ParseError { pos: usize },

    /// An invalid symbol was found.
    InvalidSymbol { pos: usize },

    /// An invalid edge was found.
    InvalidEdge { num: u32, den: u32, pos: usize },

    /// An edge was specified twice.
    RepeatEdge { a: usize, b: usize },
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

/// Represents a [Coxeter matrix](https://en.wikipedia.org/wiki/Coxeter_matrix),
/// which itself represents a [`Cd`]. This representation makes many
/// calculations with Coxeter diagrams much more convenient.
///
/// The Coxeter matrix for a Coxeter diagram is defined so that the (i, j) entry
/// corresponds to the value of the edge between the ith and jth node, or 2 if
/// there's no such edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoxMatrix(MatrixOrd);

impl CoxMatrix {
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
        Cd::parse(input).map(|cd| cd.cox())
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

    /// Returns an upper triangular matrix whose columns are unit normal vectors
    /// for the hyperplanes described by the Coxeter matrix.
    pub fn normals(&self) -> Option<Matrix> {
        let dim = self.dim();
        let mut mat = Matrix::zeros(dim, dim);

        // Builds each column from the top down, so that each of the succesive
        // dot products we check matcht he values in the Coxeter matrix.
        for i in 0..dim {
            let (prev_gens, mut n_i) = mat.columns_range_pair_mut(0..i, i);

            for (j, n_j) in prev_gens.column_iter().enumerate() {
                // All other entries in the dot product are zero.
                let dot = n_i.rows_range(0..=j).dot(&n_j.rows_range(0..=j));

                n_i[j] = ((Float::PI / self[(i, j)]).cos() - dot) / n_j[j];
            }

            // If the vector doesn't fit in spherical space.
            let norm_sq = n_i.norm_squared();
            if norm_sq >= 1.0 - Float::EPS {
                return None;
            } else {
                n_i[i] = (1.0 - norm_sq).sqrt();
            }
        }

        Some(mat)
    }
}

impl std::ops::Index<(usize, usize)> for CoxMatrix {
    type Output = Float;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

/// A node in a [`Cd`]. Represents a mirror in hyperspace, and specifies both
/// where a generator point should be located with respect to it, and how it
/// should interact with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    /// An unringed node. Represents a mirror that contains the generator point.
    /// Crucially, reflecting the generator through this mirror doesn't create a
    /// new edge.
    Unringed,

    /// A ringed node. Represents a mirror at (half) a certain distance from the
    /// generator. Reflecting the generator through this mirror creates an edge.
    Ringed(FloatOrd),

    /// A snub node. Represents a mirror at (half) a certain distance from the
    /// generator. In contrast to [`Self::Ringed`] nodes, the generator point
    /// and its reflection through this mirror can't simultaneously be in the
    /// polytope.
    Snub(FloatOrd),
}

impl Node {
    /// Returns twice the distance from the generator point to the hyperplane
    /// corresponding to this node.
    pub fn value(&self) -> Float {
        match self {
            Self::Unringed => 0.0,
            Self::Ringed(val) | Self::Snub(val) => val.0,
        }
    }

    /// Shorthand for `NodeVal::Ringed(FloatOrd::from(x))`.
    pub fn ringed(x: Float) -> Self {
        Self::Ringed(FloatOrd::from(x))
    }

    /// Shorthand for `NodeVal::Snub(FloatOrd::from(x))`.
    pub fn snub(x: Float) -> Self {
        Self::Snub(FloatOrd::from(x))
    }

    pub fn from_char(c: char) -> Option<Self> {
        Some(Node::ringed(match c {
            'o' => return Some(Node::Unringed),
            's' => return Some(Node::snub(1.0)),
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
            _ => return None,
        }))
    }

    pub fn from_char_or(c: char, idx: usize) -> CdResult<Self> {
        if let Some(node) = Self::from_char(c) {
            Ok(node)
        } else {
            Err(CdError::InvalidSymbol { pos: idx })
        }
    }
}

impl Display for Node {
    /// Prints the value that a node contains.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Unringed => writeln!(f, "o"),
            Node::Ringed(x) => writeln!(f, "x({})", x.0),
            Node::Snub(s) => writeln!(f, "s({})", s.0),
        }
    }
}

/// Represents the value of an edge in a [`Cd`]. An edge with a value of `x`
/// represents an angle of π / *x* between two hyperplanes.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    /// The numerator of the edge.
    num: u32,

    /// The denominator of the edge.
    den: u32,
}

impl Edge {
    pub fn rational(num: u32, den: u32, pos: usize) -> CdResult<Self> {
        if num > 1 && den != 0 && den < num {
            Ok(Self { num, den })
        } else {
            Err(CdError::InvalidEdge { num, den, pos })
        }
    }

    pub fn int(num: u32, pos: usize) -> CdResult<Self> {
        Self::rational(num, 1, pos)
    }

    /// Returns the numerical value of the edge.
    pub fn value(&self) -> Float {
        self.num as Float / self.den as Float
    }

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
pub struct EdgeRef {
    first: NodeRef,
    last: NodeRef,
    edge: Edge,
}

impl EdgeRef {
    pub fn new(first: NodeRef, last: NodeRef, edge: Edge) -> Self {
        Self { first, last, edge }
    }

    /// Returns the index in the graph of both node references. Requires knowing
    /// the number of nodes in the graph.
    pub fn indices(&self, len: usize) -> [NodeIndex; 2] {
        [self.first.index(len), self.last.index(len)]
    }
}

/// Helper struct that parses a [`Cd`] based on a textual notation, adapted from
/// [Krieger (year)](https://bendwavy.org/klitzing/pdf/Stott_v8.pdf).
///
/// Nodes in the Coxeter diagram are guaranteed to be in the order in which they
/// were added.
///
/// # Formal specification
///
/// A Coxeter diagram in inline ASCII notation consists of a sequence of tokens:
///
/// ```txt
/// [node]  [edge]?  [node]  ...  [node]
/// ```
///
/// The diagram must start and end with a node, to be later specified. Every
/// node may be followed by either an edge or another node. Every edge must be
/// immediately followed by another node. There may be optional whitespace in
/// between tokens.
///
/// Nodes come in three different types:
///
/// * One character nodes, like `x` or `F`.
/// * Parenthesized lengths, líke `(1.0)` or `(-3.5)`.
/// * Virtual nodes, like `*a` or `*c`.
///
/// Edges come in two different types:
///
/// * A single integer, like `3` or `15`.
/// * Two integers separated by a backslash, like `5/2` or `7/3`.
struct CdBuilder<'a> {
    /// Represents the Coxeter diagram itself. However, we don't add any edges
    /// to it until the very last step. These are provisionally stored in
    /// [`edge_queue`] instead.
    cd: Cd,

    /// A provisional queue in which the edge references are stored up and until
    /// [`Self::build`] is called, when they're added to the `Cd`.
    edge_queue: VecDeque<EdgeRef>,

    /// The Coxeter diagram in inline ASCII notation.
    diagram: &'a str,

    /// A peekable iterator over the characters of the diagram and their
    /// indices. Used to keep track of where we're reading.
    iter: iter::Peekable<std::str::CharIndices<'a>>,

    /// The previously found node.
    prev_node: Option<NodeRef>,

    /// The value of the next edge.
    next_edge: Option<Edge>,
}

/// Operations that are commonly done to parse CDs.
impl<'a> CdBuilder<'a> {
    /// Initializes a new CD builder from a string.
    fn new(diagram: &'a str) -> Self {
        Self {
            diagram,
            iter: diagram.char_indices().peekable(),
            cd: Cd::new(),
            edge_queue: VecDeque::new(),

            prev_node: None,
            next_edge: None,
        }
    }

    /// Returns the length of the Coxeter diagram.
    fn len(&self) -> usize {
        self.diagram.len()
    }

    /// Returns a [`CdError::UnexpectedEnding`]. Such an error always occurs at
    /// the end of the diagram.
    fn unexpected_ending(&self) -> CdError {
        CdError::UnexpectedEnding { pos: self.len() }
    }

    /// Gets the next index-character pair, or returns `None` if we've run out
    /// of them.
    fn next(&mut self) -> Option<(usize, char)> {
        self.iter.next()
    }

    /// Either gets the next index-character pair, or returns a
    /// [`CdError::UnexpectedEnding`] error.
    fn next_or(&mut self) -> CdResult<(usize, char)> {
        self.next().ok_or_else(|| self.unexpected_ending())
    }

    /// Peeks at the next index-character pair, or returns `None` if we've run
    /// out of them.
    fn peek(&mut self) -> Option<(usize, char)> {
        self.iter.peek().copied()
    }

    /// Either peeks at the next index-character pair, or returns a
    /// [`CdError::UnexpectedEnding`] error.
    fn peek_or(&mut self) -> CdResult<(usize, char)> {
        self.peek().ok_or_else(|| self.unexpected_ending())
    }

    /// Skips until the next non-whitespace character.
    fn skip_whitespace(&mut self) {
        while let Some((_, c)) = self.peek() {
            if !c.is_whitespace() {
                return;
            }

            self.next();
        }
    }

    /// Adds a node to the diagram.
    fn add_node(&mut self, node: Node) -> NodeIndex {
        self.cd.add_node(node)
    }

    /// Enqueues an edge, so that it's added when the diagram is built.
    fn enqueue_edge(&mut self, edge: EdgeRef) {
        self.edge_queue.push_back(edge);
    }

    /// Attempts to parse a subslice of characters, determined by the range
    /// `init_idx..=end_idx`. Returns a [`CdError::ParseError`] if it fails.
    fn parse_slice<T: FromStr>(&mut self, init_idx: usize, end_idx: usize) -> CdResult<T> {
        self.diagram[init_idx..=end_idx]
            .parse()
            .map_err(|_| CdError::ParseError { pos: end_idx })
    }

    /// Parses a multi-character node. This contains a floating point literal
    /// inside of a set of parentheses.
    ///
    /// By the time this method is called, we've already skipped the opening
    /// parenthesis.
    fn parse_node(&mut self) -> CdResult<Node> {
        let (init_idx, _) = self.peek().expect("Node can't be empty!");
        let mut end_idx = init_idx;

        // We read the number until we find the closing parenthesis.
        while let Some((idx, c)) = self.next() {
            if c == ')' {
                let val: Float = self.parse_slice(init_idx, end_idx)?;

                // In case the user tries to literally write "NaN" (real funny).
                return if val.is_nan() {
                    Err(CdError::InvalidSymbol { pos: end_idx })
                } else {
                    Ok(Node::ringed(val))
                };
            }

            end_idx = idx;
        }

        // We never found the matching parenthesis.
        Err(CdError::MismatchedParenthesis { pos: self.len() })
    }

    /// Reads the next node in the diagram and adds it to the graph. Returns
    /// `Ok(())` if succesful, and a [`CdResult`] otherwise.
    ///
    /// This method positions the iterator so that the next call to
    /// [`Self::next`] will yield the first character of the next edge.
    fn create_node(&mut self) -> CdResult<()> {
        self.skip_whitespace();
        let (idx, c) = self.next_or()?;

        // The index of the new node.
        let mut new_node = NodeRef::Absolute(self.cd.node_count());

        match c {
            // If the node is various characters inside parentheses.
            '(' => {
                let node = self.parse_node()?;
                self.add_node(node);
            }

            // If the node is a virtual node.
            '*' => {
                // Reads the index the virtual node refers to.
                let (mut idx, mut c) = self.next_or()?;

                // If we have a negative virtual node, we advance the iterator
                // and set the neg flag.
                let neg = if c == '-' {
                    let (new_idx, new_c) = self.next_or()?;
                    idx = new_idx;
                    c = new_c;
                    true
                } else {
                    false
                };

                match c {
                    // A virtual node, from *a to *z.
                    'a'..='z' => new_node = NodeRef::new(neg, c as usize - 'a' as usize),

                    // Any other character is invalid.
                    _ => return Err(CdError::InvalidSymbol { pos: idx }),
                }
            }

            // If the node is a single character.
            _ => {
                self.add_node(Node::from_char_or(c, idx)?);
            }
        }

        // If we have both a previous node and a next edge, we add a new edge to
        // the graph.
        if let Some(prev_node) = self.prev_node {
            if let Some(next_edge) = self.next_edge {
                self.enqueue_edge(EdgeRef::new(prev_node, new_node, next_edge));
            }

            self.next_edge = None;
        }

        // Resets the next edge so that it only has the node that was just found.
        self.prev_node = Some(new_node);

        Ok(())
    }

    fn parse_edge(&mut self) -> CdResult<Option<Edge>> {
        let mut numerator = None;
        let (mut init_idx, c) = self.peek().expect("Slice can't be empty!");

        // If the next character is not numeric, this means this isn't an edge
        // at all, and we return None.
        if !matches!(c, '0'..='9') {
            return Ok(None);
        }

        let mut end_idx = init_idx;

        // We read through the diagram until we encounter something that
        // looks like the start of a node.
        loop {
            let (idx, c) = self.peek_or()?;

            match c {
                // If we're dealing with a fraction:
                '/' => {
                    // Parse and save the numerator.
                    numerator = Some(self.parse_slice(init_idx, end_idx)?);

                    // Reset what's being read.
                    init_idx = idx + 1;
                }

                // If we reached the next node.
                '(' | '*' | ' ' | 'A'..='z' => {
                    // Parse the last value (either the denominator in case of a
                    // fraction, or the single number otherwise).
                    let last = self.parse_slice(init_idx, end_idx)?;

                    return Ok(Some(match numerator {
                        Some(num) => Edge::rational(num, last, end_idx)?,
                        None => Edge::int(last, end_idx)?,
                    }));
                }

                // Business as usual.
                '0'..='9' => {}

                // We found an unexpected symbol.
                _ => return Err(CdError::InvalidSymbol { pos: idx }),
            }

            end_idx = idx;
            self.next();
        }
    }

    /// Reads an edge from a CD and stores into the next edge.
    ///
    /// This method positions the iterator so that the next call to
    /// [`Self::next`] will yield the first character of the next edge.
    fn create_edge(&mut self) -> CdResult<()> {
        self.skip_whitespace();
        self.next_edge = self.parse_edge()?;
        Ok(())
    }

    /// Finishes building the CD and returns it.
    fn build(mut self) -> CdResult<Cd> {
        let len = self.cd.node_count();

        for edge_ref in self.edge_queue.into_iter() {
            let [a, b] = edge_ref.indices(len);
            self.cd.add_edge(a, b, edge_ref.edge)?;
        }

        Ok(self.cd)
    }
}

/// Encodes a [Coxeter diagram](https://polytope.miraheze.org/wiki/Coxeter_diagram)
/// or CD as an undirected labeled graph.
///
/// A Coxeter diagram serves two main functions. It serves as a representation
/// for certain polytopes called [Wythoffians](https://polytope.miraheze.org/wiki/Wythoffian),
/// and as a representation for certain symmetry groups called
/// [Coxeter groups](https://polytope.miraheze.org/wiki/Coxeter_group). In code,
/// these correspond to [`Concrete::truncate`] and [`Group::cox_group`],
/// respectively.
///
/// Each [`Node`] a Coxeter diagram represents a mirror (or hyperplane) in
/// *n*-dimensional space. If two nodes are joined by an [`Edge`] with a value
/// of x, it means that the angle between the mirrors they represent is given
/// by π / x. If two nodes aren't joined by any edge, it means that they are
/// perpendicular.
///
/// To actually build a Coxeter diagram, we use a [`CdBuilder`].
#[derive(Default)]
pub struct Cd(Graph<Node, Edge, Undirected>);

impl Cd {
    /// Initializes a new Coxeter diagram with no nodes nor edges.
    pub fn new() -> Self {
        Default::default()
    }

    /// Parses a Coxeter diagram from ASCII inline notation. For more
    /// information, see [`CdBuilder`].
    pub fn parse(input: &str) -> CdResult<Self> {
        let mut builder = CdBuilder::new(input);

        // Reads through the diagram.
        loop {
            builder.create_node()?;

            // We continue until we find that there's no further edges.
            if builder.peek().is_none() {
                return builder.build();
            }

            builder.create_edge()?;
        }
    }

    /// Returns the number of nodes in the Coxeter diagram.
    pub fn node_count(&self) -> usize {
        self.0.node_count()
    }

    /// Returns the number of edges in the Coxeter diagram.
    pub fn edge_count(&self) -> usize {
        self.0.edge_count()
    }

    /// The dimension of the polytope the Coxeter diagram describes.
    pub fn dim(&self) -> usize {
        self.node_count()
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
    pub fn node_iter<'a>(
        &'a self,
    ) -> std::iter::Map<std::slice::Iter<GraphNode<Node>>, impl Fn(&'a GraphNode<Node>) -> Node>
    {
        let closure = |node: &GraphNode<Node>| node.weight;
        self.0.raw_nodes().iter().map(closure)
    }

    /// Returns the nodes in the Coxeter diagram, in the order in which they
    /// were found.
    pub fn nodes(&self) -> Vec<Node> {
        self.0.raw_nodes().iter().map(|node| node.weight).collect()
    }

    /// Returns the vector whose values represent the node values.
    pub fn node_vector(&self) -> Vector {
        Vector::from_iterator(self.dim(), self.node_iter().map(|node| node.value()))
    }

    /// Creates a [`CoxMatrix`] from a Coxeter diagram.
    pub fn cox(&self) -> CoxMatrix {
        let dim = self.dim();
        let graph = &self.0;

        let matrix = Matrix::from_fn(dim, dim, |i, j| {
            if i == j {
                return 1.0;
            }

            if let Some(idx) = graph.find_edge(NodeIndex::new(i), NodeIndex::new(j)) {
                graph[idx].value()
            } else {
                2.0
            }
        });

        CoxMatrix::new(matrix)
    }

    /// Returns the circumradius of the polytope specified by the matrix, or
    /// `None` if this doesn't apply. This may or may not be faster than just
    /// calling [`Self::generator`] and taking the norm.
    pub fn circumradius(&self) -> Option<Float> {
        let mut schlafli = self.cox();
        let schlafli_ref = schlafli.as_matrix_mut();
        let node_vec = self.node_vector();

        // Converts the Coxeter matrix into the Schläfli matrix.
        for v in schlafli_ref.iter_mut() {
            *v = (PI / *v).cos();
        }

        if !schlafli_ref.try_inverse_mut() {
            return None;
        }

        let sq_radius = (node_vec.transpose() * schlafli.as_matrix() * node_vec)[(0, 0)] / -4.0;
        if sq_radius < -Float::EPS {
            None
        } else if sq_radius > Float::EPS {
            Some(sq_radius.sqrt())
        } else {
            Some(0.0)
        }
    }

    /// Returns a point in the position specified by the Coxeter diagram,
    /// using the set of mirrors generated by [`CoxMatrix::normals`].    
    pub fn generator(&self) -> Option<Point> {
        let normals = self.cox().normals()?;
        let mut vector = self.node_vector();

        if normals.solve_upper_triangular_mut(&mut vector) {
            Some(vector)
        } else {
            None
        }
    }
}

impl From<Cd> for CoxMatrix {
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    /// Returns a ringed node at half-unit distance.
    fn x() -> Node {
        Node::ringed(1.0)
    }

    /// Returns an unringed node.
    fn o() -> Node {
        Node::Unringed
    }

    /// Returns a snub node at half-unit distance.
    fn s() -> Node {
        Node::snub(1.0)
    }

    /// Tests that a parsed diagram's nodes and Coxeter matrix match expected
    /// values.
    fn test(diagram: &str, nodes: Vec<Node>, matrix: Matrix) {
        let cd = Cd::parse(diagram).unwrap();
        assert_eq!(cd.nodes(), nodes, "Node mismatch!");
        assert_eq!(cd.cox(), CoxMatrix::new(matrix), "Coxeter matrix mismatch!");
    }

    #[test]
    /// Tests some of the I2 symmetry groups.
    fn i2() {
        for n in 2..10 {
            let nf = n as Float;

            test(
                &format!("x{}x", n),
                vec![x(), x()],
                dmatrix![
                    1.0, nf;
                    nf, 1.0
                ],
            )
        }
    }

    #[test]
    /// Tests the A3 symmetry group.
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
    /// Tests the E6 symmetry group.
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
    /// Tests a very funny looking diagram.
    fn star() {
        test(
            "x3o3o3o3o3*a *a3*c3*e3*b3*d3*a",
            vec![x(), o(), o(), o(), o()],
            dmatrix![
                1.0, 3.0, 3.0, 3.0, 3.0;
                3.0, 1.0, 3.0, 3.0, 3.0;
                3.0, 3.0, 1.0, 3.0, 3.0;
                3.0, 3.0, 3.0, 1.0, 3.0;
                3.0, 3.0, 3.0, 3.0, 1.0
            ],
        )
    }

    #[test]
    /// Tests snub nodes.
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
    /// Tests some shortchords.
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

    #[test]
    /// Tests some virtual node shenanigans.
    fn virtual_nodes() {
        test(
            "*a4*b3*c3*-aooxx",
            vec![o(), o(), x(), x()],
            dmatrix![
                1.0, 4.0, 2.0, 2.0;
                4.0, 1.0, 3.0, 2.0;
                2.0, 3.0, 1.0, 3.0;
                2.0, 2.0, 3.0, 1.0
            ],
        )
    }

    #[test]
    /// Tests that CDs with spaces parse properly.
    fn spaces() {
        test(
            "   x   3   o   x",
            vec![x(), o(), x()],
            dmatrix![
                1.0, 3.0, 2.0;
                3.0, 1.0, 2.0;
                2.0, 2.0, 1.0
            ],
        )
    }

    #[test]
    /// Tests custom node lengths.
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
    #[should_panic(expected = "MismatchedParenthesis { pos: 6 }")]
    fn mismatched_parenthesis() {
        Cd::parse("x(1.0x").unwrap();
    }

    #[test]
    #[should_panic(expected = "UnexpectedEnding { pos: 6 }")]
    fn unexpected_ending() {
        Cd::parse("x4x3x3").unwrap();
    }

    #[test]
    #[should_panic(expected = "InvalidSymbol { pos: 2 }")]
    fn invalid_symbol() {
        Cd::parse("x3∞5o").unwrap();
    }

    #[test]
    #[should_panic(expected = "ParseError { pos: 5 }")]
    fn parse_error() {
        Cd::parse("(1.1.1)3(2.0)").unwrap();
    }

    #[test]
    #[should_panic(expected = "InvalidEdge { num: 1, den: 0, pos: 3 }")]
    fn invalid_edge() {
        Cd::parse("s1/0s").unwrap();
    }

    #[test]
    #[should_panic(expected = "RepeatEdge { a: 0, b: 1 }")]
    fn repeat_edge() {
        Cd::parse("x3x xx *c3*d *a3*b").unwrap();
    }
}
