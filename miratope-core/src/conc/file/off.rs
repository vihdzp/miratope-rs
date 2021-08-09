//! Contains the code that opens an OFF file and parses it into a polytope.

use std::{collections::HashMap, io::Result as IoResult, path::Path, str::FromStr};

use crate::{
    abs::{AbstractBuilder, Ranked, SubelementList, Subelements},
    conc::{Concrete, ElementList, Point, Polytope},
    COMPONENTS, ELEMENT_NAMES,
};

use petgraph::{graph::NodeIndex, visit::Dfs, Graph};
use vec_like::VecLike;

/// A position in a file.
#[derive(Clone, Copy, Default, Debug)]
pub struct Position {
    /// The row index.
    row: u32,

    /// The column index.
    column: u32,
}

impl Position {
    /// Increments the column number by 1.
    pub fn next(&mut self) {
        self.column += 1;
    }

    /// Increments the row number by 1, resets the column number.
    pub fn next_line(&mut self) {
        self.row += 1;
        self.column = 0;
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "row {}, column {}", self.row + 1, self.column + 1)
    }
}

/// Any error encountered while parsing an OFF file.
#[derive(Clone, Copy, Debug)]
pub enum OffError {
    /// Empty file.
    Empty,

    /// The OFF file ended unexpectedly.
    UnexpectedEnding(Position),

    /// Could not parse a number.
    Parsing(Position),

    /// Could not parse rank.
    Rank(Position),

    /// Didn't find the OFF magic word.
    MagicWord(Position),
}

impl std::fmt::Display for OffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "file is empty."),
            Self::UnexpectedEnding(pos) => write!(f, "file ended unexpectedly at {}", pos),
            Self::Parsing(pos) => write!(f, "could not parse number at {}", pos),
            Self::Rank(pos) => write!(f, "could not read rank at {}", pos),
            Self::MagicWord(pos) => write!(f, "no \"OFF\" detected at {}", pos),
        }
    }
}

impl std::error::Error for OffError {}

/// The result of parsing an OFF file.
pub type OffResult<T> = Result<T, OffError>;

/// Gets the name for an element with a given rank.
fn element_name(rank: usize) -> String {
    match ELEMENT_NAMES.get(rank) {
        Some(&name) => String::from(name),
        None => rank.to_string() + "-elements",
    }
}

/// The result of trying to read the next token from an OFF file.
enum OffNext<'a> {
    /// We've read a token from the OFF file. We don't directly store a
    /// [`Token`], since at this point in the code we don't know the starting
    /// position.
    Token(&'a str),

    /// We either read a single character from a comment, or a single whitespace
    /// character. Either way, we don't want it.
    Garbage,
}

/// Represents a token, i.e. any value of importance, in an OFF file.
struct Token<'a> {
    /// The string slice containing our value of importance.
    slice: &'a str,

    /// The starting position of the token.
    pos: Position,
}

/// An iterator over the tokens in an OFF file. It excludes whitespace and
/// comments. It also keeps track of position.
struct TokenIter<'a> {
    /// A reference to the source OFF file.
    src: &'a str,

    /// The inner iterator over the characters.
    iter: std::str::CharIndices<'a>,

    /// Whether we're currently reading a comment.
    comment: bool,

    /// The row and column in the file.
    position: Position,
}

/// Any dummy iterator would've done here.
impl<'a> TokenIter<'a> {
    /// Returns an iterator over the OFF file, with all whitespace and comments
    /// removed.
    fn new(src: &'a str) -> Self {
        Self {
            src,
            iter: src.char_indices(),
            comment: false,
            position: Default::default(),
        }
    }

    /// Attempts to get the next token from the file. Returns `None` if the
    /// inner iterator has been exhausted.
    fn try_next(&mut self) -> Option<OffNext<'a>> {
        let (mut idx, mut c) = self.iter.next()?;
        let init_idx = idx;
        let mut end_idx = init_idx;

        loop {
            match c {
                // The start of a comment.
                '#' => {
                    self.comment = true;
                    self.position.next();
                }

                // End lines also end comments.
                '\n' => {
                    self.comment = false;
                    self.position.next_line();
                }

                // We just advance the position otherwise.
                _ => self.position.next(),
            }

            // If we're in the middle of a comment, or we found a whitespace,
            // then whatever token we were reading has ended.
            if self.comment || c.is_whitespace() {
                break;
            }

            // Advances the iterator.
            end_idx = idx;
            if let Some((new_idx, new_c)) = self.iter.next() {
                idx = new_idx;
                c = new_c;
            } else {
                // We do this so that it seems ilke we read something at the end.
                idx += 1;
                break;
            }
        }

        // If we immediately broke out of the loop, this means we just read a
        // single character in a comment or a whitespace. That is, garbage.
        Some(if init_idx == idx {
            OffNext::Garbage
        } else {
            OffNext::Token(&self.src[init_idx..=end_idx])
        })
    }

    /// Reads and parses the next token from the OFF file.
    pub fn parse_next<U: FromStr>(&mut self) -> OffResult<U>
    where
        <U as FromStr>::Err: std::fmt::Debug,
    {
        let Token { slice, pos } = self
            .next()
            .ok_or(OffError::UnexpectedEnding(self.position))?;

        slice.parse().map_err(|_| OffError::Parsing(pos))
    }
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let pos = self.position;
            if let OffNext::Token(slice) = self.try_next()? {
                return Some(Token { slice, pos });
            }
        }
    }
}

/// An auxiliary struct that reads through an OFF file and builds a concrete
/// polytope out of it.
pub struct OffReader<'a> {
    /// An iterator over the tokens of the OFF file.
    iter: TokenIter<'a>,

    /// The underlying abstract polytope.
    abs: AbstractBuilder,
}

impl<'a> OffReader<'a> {
    /// Initializes a new reader from a source OFF file.
    pub fn new(src: &'a str) -> Self {
        Self {
            iter: TokenIter::new(src),
            abs: AbstractBuilder::new(),
        }
    }

    /// Returns a reference to the underlying OFF file.
    pub fn src(&self) -> &'a str {
        self.iter.src
    }

    /// Advances the underlying iterator.
    fn next(&mut self) -> Option<Token<'a>> {
        self.iter.next()
    }

    /// Reads the rank from the OFF file.
    fn rank(&mut self) -> OffResult<usize> {
        let Token { slice: first, pos } = self.next().ok_or(OffError::Empty)?;
        let rank = first.strip_suffix("OFF").ok_or(OffError::MagicWord(pos))?;

        Ok(if rank.is_empty() {
            4
        } else {
            rank.parse().map_err(|_| OffError::Rank(pos))?
        })
    }

    /// Gets the number of elements from the OFF file. This includes components
    /// iff dim ≤ 2, as this makes things easier down the line.
    fn el_nums(&mut self, rank: usize) -> OffResult<Vec<usize>> {
        let mut el_nums = Vec::with_capacity(rank - 1);

        // Reads entries one by one.
        for _ in 1..rank {
            el_nums.push(self.iter.parse_next()?);
        }

        match rank {
            // A point has a single component (itself)
            1 => el_nums.push(1),

            // A dyad has twice as many vertices as components.
            2 => {
                let comps = el_nums[0] / 2;
                el_nums.push(comps);
            }

            _ => {
                // A polygon always has as many vertices as edges.
                if rank == 2 {
                    el_nums.push(el_nums[0]);
                }

                // 2-elements go before 1-elements, we're undoing that.
                el_nums.swap(1, 2);
            }
        }

        Ok(el_nums)
    }

    /// Parses all vertex coordinates from the OFF file.
    fn parse_vertices(&mut self, num: usize, dim: usize) -> OffResult<Vec<Point>> {
        // Reads all vertices.
        let mut vertices = Vec::with_capacity(num);

        // Add each vertex to the vector.
        for _ in 0..num {
            let mut vert = Vec::with_capacity(dim);

            for _ in 0..dim {
                vert.push(self.iter.parse_next()?);
            }

            vertices.push(vert.into());
        }

        Ok(vertices)
    }

    /// Reads the faces from the OFF file and gets the edges and faces from
    /// them. Since the OFF file doesn't store edges explicitly, this is harder
    /// than reading general elements.
    fn parse_edges_and_faces(
        &mut self,
        rank: usize,
        num_edges: usize,
        num_faces: usize,
    ) -> OffResult<(SubelementList, SubelementList)> {
        let mut edges = SubelementList::with_capacity(num_edges);
        let mut faces = SubelementList::with_capacity(num_faces);

        let mut hash_edges = HashMap::new();

        // Add each face to the element list.
        for _ in 0..num_faces {
            let face_sub_num = self.iter.parse_next()?;

            let mut face = Subelements::new();
            let mut face_verts = Vec::with_capacity(face_sub_num);

            // Reads all vertices of the face.
            for _ in 0..face_sub_num {
                face_verts.push(self.iter.parse_next()?);
            }

            // Gets all edges of the face.
            for i in 0..face_sub_num {
                let mut edge = Subelements(vec![face_verts[i], face_verts[(i + 1) % face_sub_num]]);
                edge.sort();

                if let Some(idx) = hash_edges.get(&edge) {
                    face.push(*idx);
                } else {
                    hash_edges.insert(edge.clone(), edges.len());
                    face.push(edges.len());
                    edges.push(edge);
                }
            }

            // If these are truly faces and not just components, we add them.
            if rank != 3 {
                faces.push(face);
            }
        }

        // If this is a polygon, we add a single maximal element as a face.
        if rank == 3 {
            faces = SubelementList::max(edges.len());
        }

        // The number of edges in the file should match the number of read edges, though this isn't obligatory.
        if edges.len() != num_edges {
            println!("WARNING: Edge count doesn't match expected edge count!");
        }

        Ok((edges, faces))
    }

    /// Parses the next set of d-elements from the OFF file.
    fn parse_els(&mut self, num_el: usize) -> OffResult<SubelementList> {
        let mut els_subs = SubelementList::with_capacity(num_el);

        // Adds every d-element to the element list.
        for _ in 0..num_el {
            let el_sub_num = self.iter.parse_next()?;
            let mut subs = Subelements::with_capacity(el_sub_num);

            // Reads all sub-elements of the d-element.
            for _ in 0..el_sub_num {
                subs.push(self.iter.parse_next()?);
            }

            els_subs.push(subs);
        }

        Ok(els_subs)
    }

    /*
    /// Returns the [`Name`] stored in the OFF file, if any.
    fn name(&self) -> Option<Name<Con>> {
        self.src()
            .lines()
            .next()
            .map(Concrete::name_from_src)
            .flatten()
    }*/

    /// Builds a concrete polytope from the OFF reader.
    pub fn build(mut self) -> OffResult<Concrete> {
        // Reads the rank of the polytope.
        let rank = self.rank()?;

        // Deals with dumb degenerate cases.
        match rank {
            0 => return Ok(Concrete::nullitope()),
            1 => return Ok(Concrete::point()),
            2 => return Ok(Concrete::dyad()),
            _ => {}
        }

        // Reads the element numbers and vertices.
        let num_elems = self.el_nums(rank)?;
        let vertices = self.parse_vertices(num_elems[0], rank - 1)?;

        // Adds nullitope and vertices.
        self.abs.reserve(rank);
        self.abs.push_min();
        self.abs.push_vertices(vertices.len());

        // Reads edges and faces.
        if rank >= 3 {
            let (edges, faces) = self.parse_edges_and_faces(rank, num_elems[1], num_elems[2])?;
            self.abs.push(edges);
            self.abs.push(faces);
        }

        // Adds all higher elements.
        for &num_el in num_elems.iter().take(rank - 1).skip(3) {
            let subelements = self.parse_els(num_el)?;
            self.abs.push(subelements);
        }

        // Caps the abstract polytope.
        if rank != 3 {
            self.abs.push_max();
        }

        // Builds the concrete polytope.
        Ok(Concrete::new(vertices, self.abs.build()))
    }
}

/*
impl Concrete {
    /// Gets the name from the first line of an OFF file.
    fn name_from_src(first_line: &str) -> Option<Name<Con>> {
        let mut fl_iter = first_line.char_indices();

        if let Some((_, '#')) = fl_iter.next() {
            let (idx, _) = fl_iter.next()?;
            if let Ok(new_name) = ron::from_str(&first_line[idx..]) {
                return Some(new_name);
            }
        }

        None
    }

    /// Gets the name from an OFF file, assuming it's stored in RON in the first
    /// line of the file.
    pub fn name_from_off<T: AsRef<Path>>(path: T) -> Option<Name<Con>> {
        use std::io::{BufRead, BufReader};

        let file = BufReader::new(fs::File::open(path).ok()?);
        let first_line = file.lines().next()?.ok()?;

        Self::name_from_src(&first_line)
    }
}*/

/// A set of options to be used when saving the OFF file.
#[derive(Clone, Copy)]
pub struct OffOptions {
    /// Whether the OFF file should have comments specifying each face type.
    pub comments: bool,
}

impl Default for OffOptions {
    fn default() -> Self {
        OffOptions { comments: true }
    }
}

/// An auxiliary struct to write a polytope to an OFF file.
pub struct OffWriter<'a> {
    /// The output OFF file, as a string. (Maybe we should use a file writer
    /// or something similar instead?)
    off: String,

    /// The polytope that we're converting into an OFF file.
    polytope: &'a Concrete,

    /// Options for the text output.
    options: OffOptions,
}

impl<'a> OffWriter<'a> {
    /// Initializes a new OFF writer from a polytope, with a given set of
    /// options.
    pub fn new(polytope: &'a Concrete, options: OffOptions) -> Self {
        Self {
            off: String::new(),
            polytope,
            options,
        }
    }

    /// Writes the polytope's element counts into an OFF file.
    fn write_el_counts(&mut self, mut el_counts: Vec<usize>) {
        let rank = el_counts.len() - 1;

        // # Vertices, Faces, Edges, ...
        if self.options.comments {
            let mut element_names = Vec::with_capacity(rank - 1);

            for r in 1..rank {
                element_names.push(element_name(r));
            }

            if element_names.len() >= 2 {
                element_names.swap(0, 1);
            }

            self.off.push_str("\n# Vertices");
            for element_name in element_names {
                self.off.push_str(", ");
                self.off.push_str(&element_name);
            }
            self.off.push('\n');
        }

        // Swaps edges and faces, because OFF format bad.
        if rank >= 4 {
            el_counts.swap(2, 3);
        }

        for el_count in el_counts.into_iter().skip(1).take(rank - 1) {
            self.off.push_str(&el_count.to_string());
            self.off.push(' ');
        }

        self.off.push('\n');
    }

    /// Writes the vertices of a polytope into an OFF file.
    fn write_vertices(&mut self, vertices: &[Point]) {
        // # Vertices
        if self.options.comments {
            self.off.push_str("\n# ");
            self.off.push_str(&element_name(1));
            self.off.push('\n');
        }

        // Adds the coordinates.
        for v in vertices {
            for c in v.into_iter() {
                self.off.push_str(&c.to_string());
                self.off.push(' ');
            }
            self.off.push('\n');
        }
    }

    /// Gets and writes the faces of a polytope into an OFF file.
    fn write_faces(&mut self, rank: usize, edges: &ElementList, faces: &ElementList) {
        // # Faces
        if self.options.comments {
            let name;
            let el_name = if rank > 2 {
                name = element_name(3);
                &name
            } else {
                COMPONENTS
            };

            self.off.push_str("\n# ");
            self.off.push_str(el_name);
            self.off.push('\n');
        }

        // TODO: write components instead of faces in 2D case.
        // ALSO TODO: reuse code from mesh builder.
        for face in faces {
            self.off.push_str(&face.subs.len().to_string());

            // Maps an OFF index into a graph index.
            let mut hash_edges = HashMap::new();
            let mut graph = Graph::new_undirected();

            // Maps the vertex indices to consecutive integers from 0.
            for &edge_idx in &face.subs {
                let edge = &edges[edge_idx];
                let mut hash_edge = Vec::with_capacity(2);

                for &vertex_idx in &edge.subs.0 {
                    match hash_edges.get(&vertex_idx) {
                        Some(&idx) => hash_edge.push(idx),
                        None => {
                            let idx = hash_edges.len();
                            hash_edges.insert(vertex_idx, idx);
                            hash_edge.push(idx);

                            graph.add_node(vertex_idx);
                        }
                    }
                }
            }

            // There should be as many graph indices as edges on the face.
            // Otherwise, something went wrong.
            debug_assert_eq!(
                hash_edges.len(),
                face.subs.len(),
                "Faces don't have the same number of edges as there are in the polytope!"
            );

            // Adds the edges to the graph.
            for &edge_idx in &face.subs.0 {
                let edge = &edges[edge_idx];
                graph.add_edge(
                    NodeIndex::new(*hash_edges.get(&edge.subs[0]).unwrap()),
                    NodeIndex::new(*hash_edges.get(&edge.subs[1]).unwrap()),
                    (),
                );
            }

            // Retrieves the cycle of vertices.
            let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
            while let Some(nx) = dfs.next(&graph) {
                self.off.push(' ');
                self.off.push_str(&graph[nx].to_string());
            }
            self.off.push('\n');
        }
    }

    /// Writes the n-elements of a polytope into an OFF file.
    fn write_els(&mut self, rank: usize, els: &ElementList) {
        // # n-elements
        if self.options.comments {
            self.off.push_str("\n# ");
            self.off.push_str(&element_name(rank));
            self.off.push('\n');
        }

        // Adds the elements' indices.
        for el in els {
            self.off.push_str(&el.subs.len().to_string());

            for &sub in &el.subs.0 {
                self.off.push(' ');
                self.off.push_str(&sub.to_string());
            }

            self.off.push('\n');
        }
    }

    /// Consumes the OFF writer, returns the actual OFF file as a `String`.
    pub fn build(mut self) -> String {
        let rank = self.polytope.rank();
        let vertices = &self.polytope.vertices;
        let abs = &self.polytope.abs;

        // Serialized name.
        /* self.off.push_str("# ");
        self.off
            .push_str(&ron::to_string(&self.polytope.name).unwrap_or_default());
        self.off.push('\n'); */

        // Blatant advertising.
        if self.options.comments {
            self.off += &format!(
                "# Generated using Miratope v{} (https://github.com/OfficialURL/miratope-rs)\n\n",
                env!("CARGO_PKG_VERSION")
            );
        }

        // Writes header.
        if rank != 4 {
            self.off += &(rank - 1).to_string();
        }
        self.off += "OFF\n";

        // If we have a nullitope or point on our hands, that is all.
        if rank < 2 {
            return self.off;
        }

        // Adds the element counts.
        self.write_el_counts(self.polytope.el_count_iter().collect());

        // Adds vertex coordinates.
        self.write_vertices(vertices);

        // Adds faces.
        if rank >= 3 {
            self.write_faces(rank, &abs[2], &abs[3]);
        }

        // Adds the rest of the elements.
        for r in 4..rank {
            self.write_els(r, &abs[r]);
        }

        self.off
    }
}

impl Concrete {
    /// Converts a polytope into an OFF file.
    pub fn to_off(&self, options: OffOptions) -> String {
        OffWriter::new(self, options).build()
    }

    /// Writes a polytope's OFF file in a specified file path.
    pub fn to_path(&self, fp: &impl AsRef<Path>, opt: OffOptions) -> IoResult<()> {
        std::fs::write(fp, self.to_off(opt))
    }
}

#[cfg(test)]
mod tests {
    // TODO: move all OFF files into a folder.

    use crate::conc::file::FromFile;

    use super::*;

    /// Used to test a particular polytope.
    // TODO: take a `&str` as an argument instead.
    fn test_shape(off: &str, el_nums: &[usize]) {
        // Checks that element counts match up.
        let p = Concrete::from_off(off).expect("OFF file could not be loaded.");
        let p_counts: Vec<_> = p.el_count_iter().collect();
        assert_eq!(p_counts, el_nums);

        // Checks that the polytope can be reloaded correctly.
        let p = Concrete::from_off(&p.to_off(Default::default()))
            .expect("OFF file could not be reloaded.");
        let p_counts: Vec<_> = p.el_count_iter().collect();
        assert_eq!(p_counts, el_nums);
    }

    #[test]
    /// Checks that a point has the correct amount of elements.
    /* fn point_nums() {
        let point = Concrete::from_off("0OFF").unwrap();
        test_shape(point, vec![1, 1])
    } */

    #[test]
    /// Checks that a dyad has the correct amount of elements.
    /* fn dyad_nums() {
        let dyad = Concrete::from_off("1OFF 2 -1 1 0 1").unwrap();
        test_shape(dyad, vec![1, 2, 1])
    } */

    /*
    #[test]
    /// Checks that a hexagon has the correct amount of elements.
    fn hig_nums() {
        let hig =from_src(
            "2OFF 6 1 1 0 0.5 0.8660254037844386 -0.5 0.8660254037844386 -1 0 -0.5 -0.8660254037844386 0.5 -0.8660254037844386 6 0 1 2 3 4 5"
        );

        test_shape(hig, vec![1, 6, 6, 1])
    }

    #[test]
    /// Checks that a hexagram has the correct amount of elements.
    fn shig_nums() {
        let shig: Concrete = from_src(
            "2OFF 6 2 1 0 0.5 0.8660254037844386 -0.5 0.8660254037844386 -1 0 -0.5 -0.8660254037844386 0.5 -0.8660254037844386 3 0 2 4 3 1 3 5"
        ).into();

        test_shape(shig, vec![1, 6, 6, 1])
    }
    */

    #[test]
    /// Checks that a tetrahedron has the correct amount of elements.
    fn tet_nums() {
        test_shape(
            "OFF 4 4 6 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 3 0 1 2 3 3 0 2 3 0 1 3 3 3 1 2",
            &[1, 4, 6, 4, 1],
        )
    }

    #[test]
    /// Checks that a 2-tetrahedron compund has the correct amount of elements.
    fn so_nums() {
        test_shape(include_str!("so.off"), &[1, 8, 12, 8, 1])
    }

    #[test]
    /// Checks that a pentachoron has the correct amount of elements.
    /* fn pen_nums() {
        let pen = Concrete::from_off(include_str!("pen.off")).unwrap();
        test_shape(pen, vec![1, 5, 10, 10, 5, 1])
    } */

    #[test]
    /// Checks that comments are correctly parsed.
    fn comments() {
        test_shape(
            "# So
        OFF # this
        4 4 6 # is
        # a # test # of
        1 1 1 # the 1234 5678
        1 -1 -1 # comment 987
        -1 1 -1 # removal 654
        -1 -1 1 # system 321
        3 0 1 2 #let #us #see
        3 3 0 2# if
        3 0 1 3#it
        3 3 1 2#works!#",
            &[1, 4, 6, 4, 1],
        )
    }

    #[test]
    #[should_panic(expected = "Empty")]
    fn empty() {
        Concrete::from_off("").unwrap();
    }

    #[test]
    #[should_panic(expected = "Rank(Position { row: 0, column: 3 })")]
    fn rank() {
        Concrete::from_off("   fooOFF").unwrap();
    }

    #[test]
    #[should_panic(expected = "MagicWord(Position { row: 1, column: 3 })")]
    fn magic_num() {
        Concrete::from_off("# comment\n   foo bar").unwrap();
    }

    #[test]
    #[should_panic(expected = "Parsing(Position { row: 1, column: 3 })")]
    fn parse() {
        Concrete::from_off("OFF\n10 foo bar").unwrap();
    }
}
