use std::{
    f64::consts::PI, fmt::Display, iter, mem, ops::RangeBounds, slice::SliceIndex, str::FromStr,
};

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

/// Represents a [`Cd`] as a matrix, so that the (i, j) entry corresponds to the
/// value of the edge between the ith and jth node.
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

/// A node in a [`Cd`]. Represents a mirror in hyperspace, and specifies where
/// a generator point should be located with respect to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    /// An unringed node.
    Unringed,

    /// A ringed node, at half the specified distance from a corresponding
    /// hyperplane.
    Ringed(FloatOrd),

    /// A snub node, at half the specified distance from a corresponding
    /// hyperplane.
    Snub(FloatOrd),
}

impl Node {
    /// Returns twice the distance from the generator point to the corresponding
    /// hyperplane to this node.
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
            Err(CdError::InvalidSymbol(idx))
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

/// Represents the value of an edge in a [`Cd`].
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    /// The numerator of the edge.
    num: i32,

    /// The denominator of the edge.
    den: i32,
}

impl Edge {
    pub fn rational(num: i32, den: i32) -> Self {
        Self { num, den }
    }

    pub fn int(num: i32) -> Self {
        Self::rational(num, 1)
    }

    /// Returns the numerical value of the edge.
    pub fn value(&self) -> Float {
        self.num as Float / self.den as Float
    }
}

impl Display for Edge {
    /// Prints the value contained in an edge.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} / {}", self.num, self.den)
    }
}

/// Packages important information needed to interpret CDs
pub struct CdBuilder<'a> {
    /// Represents the CD itself.
    cd: Graph<Node, Edge, Undirected>,

    diagram: &'a str,

    /// A peekable iterator over the characters of the diagram and their
    /// indices.
    diagram_iter: iter::Peekable<std::str::CharIndices<'a>>,

    /// The previously found node.
    prev_node: Option<NodeIndex>,

    /// The value of the next edge.
    next_edge: Option<Edge>,

    /// The length of the diagram.
    len: usize,
}

/// Operations that are commonly done to parse CDs.
impl<'a> CdBuilder<'a> {
    /// Initializes a new CD builder from a string.
    pub fn new(diagram: &'a str) -> Self {
        Self {
            diagram,
            diagram_iter: diagram.char_indices().peekable(),
            cd: Graph::new_undirected(),
            prev_node: None,
            next_edge: None,
            len: diagram.len(),
        }
    }

    /// Gets the next index-character pair, or returns `None` if we've run out.
    pub fn next(&mut self) -> Option<(usize, char)> {
        self.diagram_iter.next()
    }

    /// Either gets the next index-character pair, or returns a
    /// [`CdError::UnexpectedEnding`] error.
    pub fn next_or(&mut self) -> CdResult<(usize, char)> {
        self.next().ok_or(CdError::UnexpectedEnding(self.len))
    }

    /// Gets the next index-character pair, or returns `None` if we've run out.
    pub fn peek(&mut self) -> Option<(usize, char)> {
        self.diagram_iter.peek().copied()
    }

    /// Either gets the next index-character pair, or returns a
    /// [`CdError::UnexpectedEnding`] error.
    pub fn peek_or(&mut self) -> CdResult<(usize, char)> {
        self.peek().ok_or(CdError::UnexpectedEnding(self.len))
    }

    /// Attempts to parse a subslice of characters, determined by the specified
    /// range. Returns a [`CdError::ParseError`] if it fails.
    pub fn parse<T: FromStr, U: RangeBounds<usize> + SliceIndex<str, Output = str>>(
        &self,
        range: U,
    ) -> CdResult<T> {
        use std::ops::Bound::*;

        let end = match range.end_bound() {
            Included(&e) => e,
            Excluded(&e) => e - 1,
            Unbounded => self.len - 1,
        };

        self.diagram[range]
            .parse()
            .map_err(|_| CdError::ParseError(end))
    }

    /// Parses a multi-character node. This contains a floating point literal
    /// inside of a set of parentheses.
    ///
    /// By the time this method is called, we've already skipped the opening
    /// parenthesis.
    pub fn parse_node(&mut self) -> CdResult<Node> {
        let (init_idx, _) = self.next().expect("Node can't be empty!");

        // We read the number until we find the closing parenthesis.
        while let Some((idx, c)) = self.next() {
            if c == ')' {
                let val: Float = self.parse(init_idx..idx)?;

                // In case the user tries to literally write "NaN" (real funny).
                return if val.is_nan() {
                    Err(CdError::InvalidSymbol(idx))
                } else {
                    Ok(Node::ringed(val))
                };
            }
        }

        // We never found the matching parenthesis.
        Err(CdError::MismatchedParenthesis(self.len))
    }

    /// Reads the next node in the diagram and adds it to the graph. Returns
    /// `Ok(())` if succesful, and a [`CdResult`] otherwise.
    pub fn create_node(&mut self) -> CdResult<()> {
        let (idx, c) = self.next_or()?;

        // The index of the new node.
        let mut new_node = NodeIndex::new(self.cd.node_count());

        match c {
            // If the node is various characters inside parentheses.
            '(' => {
                let node = self.parse_node()?;
                self.cd.add_node(node);
            }

            // If the node is a virtual node.
            '*' => {
                // Reads the index the virtual node refers to.
                let (idx, c) = self.next_or()?;
                let c = c as usize;

                const A: usize = 'a' as usize;
                const Z: usize = 'z' as usize;

                match c {
                    // A virtual node, from *a to *z.
                    A..=Z => new_node = NodeIndex::new(c - A),

                    // Any other character is invalid.
                    _ => return Err(CdError::InvalidSymbol(idx)),
                }
            }

            // If the node is a single character.
            _ => {
                self.cd.add_node(Node::from_char_or(c, idx)?);
            }
        }

        // If the next edge has been completely built, we add a new edge to the graph.
        if let Some(prev_node) = self.prev_node {
            if let Some(next_edge) = self.next_edge {
                self.cd.add_edge(prev_node, new_node, next_edge);
            }
        }

        // Resets the next edge so that it only has the node that was just found.
        self.prev_node = Some(new_node);
        self.next_edge = None;

        Ok(())
    }

    pub fn parse_edge(&mut self) -> CdResult<Edge> {
        let mut numerator = None;
        let (mut init_idx, _) = self.next().expect("Slice can't be empty!");

        // We read through the diagram until we encounter something that
        // looks like the start of a node.
        loop {
            let (idx, c) = self.peek_or()?;

            // If we're dealing with a fraction:
            if c == '/' {
                // Parse and save the numerator.
                numerator = Some(self.parse(init_idx..idx)?);

                // Reset what's being read.
                init_idx = idx + 1;
            }
            // We reached the next node.
            else if c == '(' || c == '*' || c.is_alphabetic() {
                // Parse the last value (either the denominator in case of a
                // fraction, or the single number otherwise).
                let last = self.parse(init_idx..idx)?;

                return Ok(match numerator {
                    Some(num) => Edge::rational(num, last),
                    None => Edge::int(last),
                });
            }

            self.next();
        }
    }

    /// Reads an edge from a CD and stores into the next edge.
    pub fn create_edge(&mut self) -> CdResult<Option<()>> {
        if self.peek() == None {
            return Ok(None);
        }

        self.next_edge = Some(self.parse_edge()?);
        Ok(self.next_edge.map(|_| ()))
    }
}

/// Encodes a
/// [Coxeter diagram](https://polytope.miraheze.org/wiki/Coxeter_diagram) or CD,
/// which is an undirected labeled graph that doubles as a representation for
/// certain polytopes called
/// [Wythoffians](https://polytope.miraheze.org/wiki/Wythoffian), and certain
/// symmetry groups called
/// [Coxeter groups](https://polytope.miraheze.org/wiki/Coxeter_group).
///
/// Each [`Node`] in the graph represents a mirror (or hyperplane) in
/// *n*-dimensional space. If two nodes are joined by an [`Edge`] with a value
/// of *x*, it means that the angle between the mirrors they represent is given
/// by π / *x*. If two nodes aren't joined by any edge, it means that they are
/// perpendicular.
///
/// Stores the value of the next edge in the graph, along with the index of its
/// first node. This is used in order to handle virtual nodes. A new edge will
/// be added to the graph only when both conditions are met:
///
/// * Both fields of the `EdgeMem` are full.
/// * We're reading a new node.
///
/// The node added will have the `EdgeMem`'s node as a first node, the currently
/// read node as the last node, and an edge value given by the `EdgeMem`.
pub struct Cd(Graph<Node, Edge, Undirected>);

impl Cd {
    /// Main function for parsing CDs from strings.
    pub fn new(input: &str) -> CdResult<Self> {
        let mut builder = CdBuilder::new(input);

        // Reads through the diagram.
        loop {
            builder.create_node()?;

            // We continue until we find that there's no further edges.
            if let Ok(None) = builder.create_edge() {
                return Ok(Cd(builder.cd));
            }
        }
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

    /// Returns the number of edges in the CD.
    pub fn edge_count(&self) -> usize {
        self.0.edge_count()
    }

    /// The dimension of the polytope the CD describes.
    pub fn dim(&self) -> usize {
        self.0.node_count()
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

    fn test(input: &str, nodes: Vec<Node>, matrix: Matrix) {
        let cd = Cd::new(input).unwrap();
        assert_eq!(cd.nodes(), nodes, "Node mismatch!");
        assert_eq!(cd.cox(), CoxMatrix::new(matrix), "Coxeter matrix mismatch!");
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
