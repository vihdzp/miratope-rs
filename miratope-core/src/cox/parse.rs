//! Contains the methods to parse a linear diagram.

use std::{collections::VecDeque, iter, str::FromStr};

use petgraph::graph::NodeIndex;

use crate::Float;

use super::cd::{Cd, CdError, CdResult, Edge, EdgeRef, Node, NodeRef};

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
/// * Virtual nodes, like `*a` or `*-c`.
///
/// Edges come in two different types:
///
/// * A single integer, like `3` or `15`.
/// * Two integers separated by a backslash, like `5/2` or `7/3`.
pub struct CdBuilder<'a, T: Float> {
    /// The Coxeter diagram in inline ASCII notation.
    diagram: &'a str,

    /// A peekable iterator over the characters of the diagram and their
    /// indices. Used to keep track of where we're reading.
    iter: iter::Peekable<std::str::CharIndices<'a>>,

    /// Represents the Coxeter diagram itself. However, we don't add any edges
    /// to it until the very last step. These are provisionally stored in
    /// [`Self::edge_queue`] instead.
    cd: Cd<T>,

    /// A provisional queue in which the [`EdgeRef`]s are stored up and until
    /// [`Self::build`] is called, when they're added to the `Cd`.
    edge_queue: VecDeque<EdgeRef>,

    /// The previously found node.
    prev_node: Option<NodeRef>,

    /// The value of the next edge.
    next_edge: Option<Edge>,
}

/// Operations that are commonly done to parse CDs.
impl<'a, T: Float> CdBuilder<'a, T> {
    /// Initializes a new CD builder from a string.
    pub fn new(diagram: &'a str) -> Self {
        Self {
            // The diagram and the iterator over the diagram.
            diagram,
            iter: diagram.char_indices().peekable(),

            // The final CD and its edges.
            cd: Cd::new(),
            edge_queue: VecDeque::new(),

            // The previous and next node to be built.
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
    fn add_node(&mut self, node: Node<T>) -> NodeIndex {
        self.cd.add_node(node)
    }

    /// Enqueues an edge, so that it's added when the diagram is built.
    fn enqueue_edge(&mut self, edge: EdgeRef) {
        self.edge_queue.push_back(edge);
    }

    /// Attempts to parse a subslice of characters, determined by the range
    /// `init_idx..=end_idx`. Returns a [`CdError::ParseError`] if it fails.
    fn parse_slice<U: FromStr>(&mut self, init_idx: usize, end_idx: usize) -> CdResult<U> {
        self.diagram[init_idx..=end_idx]
            .parse()
            .map_err(|_| CdError::ParseError { pos: end_idx })
    }

    /// Parses a multi-character node. This contains a floating point literal
    /// inside of a set of parentheses.
    ///
    /// By the time this method is called, we've already skipped the opening
    /// parenthesis.
    fn parse_node(&mut self) -> CdResult<Node<T>> {
        let (init_idx, _) = self.peek().expect("Node can't be empty!");
        let mut end_idx = init_idx;

        // We read the number until we find the closing parenthesis.
        while let Some((idx, c)) = self.next() {
            if c == ')' {
                let val: T = self.parse_slice(init_idx, end_idx)?;

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
                let neg = c == '-';
                if neg {
                    let (new_idx, new_c) = self.next_or()?;
                    idx = new_idx;
                    c = new_c;
                }

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

    /// Parses the next edge in the Coxeter diagram. May return `None` if
    /// there's currently no edge to be read.
    ///
    /// # Errors
    /// This method will return a [`CdError::InvalidSymbol`] if it ever
    /// encounters any unexpected symbol. Likewise, it will return a
    /// [`CdError::InvalidEdge`] if the edge is something invalid like `1/0`.
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

    fn read(&mut self) -> CdResult<()> {
        loop {
            self.create_node()?;

            // We continue until we find that there's no further edges.
            if self.peek().is_none() {
                return Ok(());
            }

            self.create_edge()?;
        }
    }

    /// Finishes building the CD and returns it.
    pub fn build(mut self) -> CdResult<Cd<T>> {
        // Reads through the diagram.
        self.read()?;
        let len = self.cd.node_count();

        for edge_ref in self.edge_queue.into_iter() {
            let [a, b] = edge_ref.indices(len);
            self.cd.add_edge(a, b, edge_ref.edge)?;
        }

        Ok(self.cd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cox::CoxMatrix;
    use crate::geometry::Matrix;
    use nalgebra::dmatrix;

    /// Returns a ringed node at half-unit distance.
    fn x() -> Node<f32> {
        Node::ringed(1.0)
    }

    /// Returns an unringed node.
    fn o() -> Node<f32> {
        Node::Unringed
    }

    /// Returns a snub node at half-unit distance.
    fn s() -> Node<f32> {
        Node::snub(1.0)
    }

    /// Tests that a parsed diagram's nodes and Coxeter matrix match expected
    /// values.
    fn test(diagram: &str, nodes: Vec<Node<f32>>, matrix: Matrix<f32>) {
        let cd = Cd::parse(diagram).unwrap();
        assert_eq!(cd.nodes(), nodes, "Node mismatch!");
        assert_eq!(cd.cox(), CoxMatrix::new(matrix), "Coxeter matrix mismatch!");
    }

    #[test]
    /// Tests some of the I2 symmetry groups.
    fn i2() {
        for n in 2..10 {
            let nf = n as f32;

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
    /// Tests a nice looking diagram.
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
                Node::from_char('v').unwrap(),
                x(),
                Node::from_char('F').unwrap(),
                Node::from_char('f').unwrap(),
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
            vec![x(), Node::ringed(2.2), Node::ringed(-3.0)],
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
        Cd::<f32>::parse("x(1.0x").unwrap();
    }

    #[test]
    #[should_panic(expected = "UnexpectedEnding { pos: 6 }")]
    fn unexpected_ending() {
        Cd::<f32>::parse("x4x3x3").unwrap();
    }

    #[test]
    #[should_panic(expected = "InvalidSymbol { pos: 2 }")]
    fn invalid_symbol() {
        Cd::<f32>::parse("x3⊕5o").unwrap();
    }

    #[test]
    #[should_panic(expected = "ParseError { pos: 5 }")]
    fn parse_error() {
        Cd::<f32>::parse("(1.1.1)3(2.0)").unwrap();
    }

    #[test]
    #[should_panic(expected = "InvalidEdge { num: 1, den: 0, pos: 3 }")]
    fn invalid_edge() {
        Cd::<f32>::parse("s1/0s").unwrap();
    }

    #[test]
    #[should_panic(expected = "RepeatEdge { a: 0, b: 1 }")]
    fn repeat_edge() {
        Cd::<f32>::parse("x3x xx *c3*d *a3*b").unwrap();
    }
}
