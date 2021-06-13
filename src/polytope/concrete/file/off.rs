use std::{
    collections::HashMap,
    fs,
    path::Path,
    str::FromStr,
    sync::{Arc, Mutex},
};

use super::IoResult;
use crate::{
    lang::name::{Con, Name},
    polytope::{
        concrete::{Concrete, ElementList, Point, Polytope, RankVec, Subelements},
        r#abstract::{
            elements::{AbstractBuilder, SubelementList, Subsupelements},
            rank::Rank,
        },
        COMPONENTS, ELEMENT_NAMES,
    },
};

use petgraph::{graph::NodeIndex, visit::Dfs, Graph};

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

    /// The file couldn't be parsed as UTF-8.
    InvalidFile,
}

impl std::fmt::Display for OffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "file is empty."),
            Self::UnexpectedEnding(pos) => write!(f, "file ended unexpectedly at {}", pos),
            Self::Parsing(pos) => write!(f, "could not parse number at {}", pos),
            Self::Rank(pos) => write!(f, "could not read rank at {}", pos),
            Self::MagicWord(pos) => write!(f, "no \"OFF\" detected at {}", pos),
            Self::InvalidFile => write!(f, "could not parse file as UTF-8"),
        }
    }
}

impl std::error::Error for OffError {}

/// The result of parsing an OFF file.
pub type OffResult<T> = Result<T, OffError>;

/// Gets the name for an element with a given rank.
fn element_name(rank: Rank) -> String {
    match ELEMENT_NAMES.get(rank.usize()) {
        Some(&name) => String::from(name),
        None => rank.to_string() + "-elements",
    }
}

/// An iterator over the tokens in an OFF file that also keeps track of position.
pub struct TokenIter<'a, T: Iterator<Item = &'a str>> {
    /// The inner iterator over the tokens.
    iter: T,

    /// The row and column in the file.
    position: Arc<Mutex<Position>>,
}

/// Any dummy iterator would've done here.
impl<'a> TokenIter<'a, std::vec::IntoIter<&'a str>> {
    /// Returns an iterator over the OFF file, with all whitespace and comments
    /// removed.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(src: &'a str) -> TokenIter<'a, impl Iterator<Item = &'a str>> {
        let mut comment = false;
        let position = Arc::new(Mutex::new(Position::default()));
        let position_clone = Arc::clone(&position);

        TokenIter {
            iter: str::split(src, move |c: char| {
                let mut position = position_clone.lock().unwrap();

                if c == '#' {
                    comment = true;
                    position.next();
                } else if c == '\n' {
                    comment = false;
                    position.next_line();
                } else {
                    position.next();
                }

                comment || c.is_whitespace()
            }),
            position,
        }
    }
}

impl<'a, T: Iterator<Item = &'a str>> TokenIter<'a, T> {
    /// Returns the current position of the iterator.
    pub fn position(&self) -> Position {
        *self.position.lock().unwrap()
    }

    /// Returns the next token from the file, and the position at which it
    /// starts. Manual implementation of a filter over non-empty strings.
    pub fn next(&mut self) -> Option<(&str, Position)> {
        loop {
            let pos = self.position();
            let str = self.iter.next()?;

            if !str.is_empty() {
                return Some((str, pos));
            }
        }
    }

    /// Reads the next integer or float from the OFF file.
    pub fn parse_next<U: FromStr>(&mut self) -> OffResult<U>
    where
        <U as FromStr>::Err: std::fmt::Debug,
    {
        loop {
            let pos = self.position();
            let str = self.iter.next().ok_or(OffError::UnexpectedEnding(pos))?;

            if !str.is_empty() {
                return str.parse().map_err(|_| OffError::Parsing(pos));
            }
        }
    }
}

/// Gets the number of elements from the OFF file.
/// This includes components iff dim ≤ 2, as this makes things easier down the
/// line.
fn el_nums<'a, T: Iterator<Item = &'a str>>(
    rank: Rank,
    toks: &mut TokenIter<'a, T>,
) -> OffResult<Vec<usize>> {
    let rank = rank.usize();
    let mut el_nums = Vec::with_capacity(rank);

    // Reads entries one by one.
    for _ in 0..rank {
        el_nums.push(toks.parse_next()?);
    }

    // A point has a single component (itself)
    if rank == 0 {
        el_nums.push(1);
    }
    // A dyad has twice as many vertices as components.
    else if rank == 1 {
        let comps = el_nums[0] / 2;
        el_nums.push(comps);
    } else {
        // A polygon always has as many vertices as edges.
        if rank == 2 {
            el_nums.push(el_nums[0]);
        }

        // 2-elements go before 1-elements, we're undoing that.
        el_nums.swap(1, 2);
    }

    Ok(el_nums)
}

/// Parses all vertex coordinates from the OFF file.
fn parse_vertices<'a, T: Iterator<Item = &'a str>>(
    num: usize,
    dim: usize,
    toks: &mut TokenIter<'a, T>,
) -> OffResult<Vec<Point>> {
    // Reads all vertices.
    let mut vertices = Vec::with_capacity(num);

    // Add each vertex to the vector.
    for _ in 0..num {
        let mut vert = Vec::with_capacity(dim);

        for _ in 0..dim {
            vert.push(toks.parse_next()?);
        }

        vertices.push(vert.into());
    }

    Ok(vertices)
}

/// Reads the faces from the OFF file and gets the edges and faces from them.
/// Since the OFF file doesn't store edges explicitly, this is harder than
/// reading general elements.
fn parse_edges_and_faces<'a, T: Iterator<Item = &'a str>>(
    rank: Rank,
    num_edges: usize,
    num_faces: usize,
    toks: &mut TokenIter<'a, T>,
) -> OffResult<(SubelementList, SubelementList)> {
    let mut edges = SubelementList::with_capacity(num_edges);
    let mut faces = SubelementList::with_capacity(num_faces);

    let mut hash_edges = HashMap::new();

    // Add each face to the element list.
    for _ in 0..num_faces {
        let face_sub_num = toks.parse_next()?;

        let mut face = Subelements::new();
        let mut face_verts = Vec::with_capacity(face_sub_num);

        // Reads all vertices of the face.
        for _ in 0..face_sub_num {
            face_verts.push(toks.parse_next()?);
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
        if rank != Rank::new(2) {
            faces.push(face);
        }
    }

    // If this is a polygon, we add a single maximal element as a face.
    if rank == Rank::new(2) {
        faces = SubelementList::max(edges.len());
    }

    // The number of edges in the file should match the number of read edges, though this isn't obligatory.
    if edges.len() != num_edges {
        println!("WARNING: Edge count doesn't match expected edge count!");
    }

    Ok((edges, faces))
}

fn parse_els<'a, T: Iterator<Item = &'a str>>(
    num_el: usize,
    toks: &mut TokenIter<'a, T>,
) -> OffResult<SubelementList> {
    let mut els_subs = SubelementList::with_capacity(num_el);

    // Adds every d-element to the element list.
    for _ in 0..num_el {
        let el_sub_num = toks.parse_next()?;
        let mut subs = Subelements::with_capacity(el_sub_num);

        // Reads all sub-elements of the d-element.
        for _ in 0..el_sub_num {
            subs.push(toks.parse_next()?);
        }

        els_subs.push(subs);
    }

    Ok(els_subs)
}

impl Concrete {
    /// Gets the name from the first line of an OFF file.
    fn name_from_src(first_line: &str) -> Option<Name<Con>> {
        let mut first_line = first_line.chars();

        if first_line.next() == Some('#') {
            if let Ok(new_name) = ron::from_str(&first_line.collect::<String>()) {
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

    /// Builds a polytope from the string representation of an OFF file.
    pub fn from_off(src: &str) -> OffResult<Self> {
        // Reads name.
        let name = src
            .lines()
            .next()
            .map(|first_line| Self::name_from_src(first_line))
            .flatten();

        let mut toks = TokenIter::new(&src);
        let rank = {
            let (first, pos) = toks.next().ok_or(OffError::Empty)?;
            let rank = first.strip_suffix("OFF").ok_or(OffError::MagicWord(pos))?;

            if rank.is_empty() {
                Rank::new(3)
            } else {
                Rank::new(rank.parse().map_err(|_| OffError::Rank(pos))?)
            }
        };

        // Deals with dumb degenerate cases.
        if rank == Rank::new(-1) {
            return Ok(Concrete::nullitope());
        } else if rank == Rank::new(0) {
            return Ok(Concrete::point());
        } else if rank == Rank::new(1) {
            return Ok(Concrete::dyad());
        }

        let num_elems = el_nums(rank, &mut toks)?;
        let vertices = parse_vertices(num_elems[0], rank.usize(), &mut toks)?;
        let mut abs = AbstractBuilder::with_capacity(rank);

        // Adds nullitope and vertices.
        abs.push_min();
        abs.push_vertices(vertices.len());

        // Reads edges and faces.
        if rank >= Rank::new(2) {
            let (edges, faces) =
                parse_edges_and_faces(rank, num_elems[1], num_elems[2], &mut toks)?;
            abs.push(edges);
            abs.push(faces);
        }

        // Adds all higher elements.
        for &num_el in num_elems.iter().take(rank.usize()).skip(3) {
            abs.push(parse_els(num_el, &mut toks)?);
        }

        // Caps the abstract polytope, returns the concrete one.
        if rank != Rank::new(2) {
            abs.push_max();
        }

        let poly = Self::new(vertices, abs.build());

        Ok(if let Some(name) = name {
            poly.with_name(name)
        } else {
            poly
        })
    }
}

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

/// Writes the polytope's element counts into an OFF file.
fn write_el_counts(off: &mut String, opt: &OffOptions, mut el_counts: RankVec<usize>) {
    let rank = el_counts.rank();

    // # Vertices, Faces, Edges, ...
    if opt.comments {
        off.push_str("\n# Vertices");

        let mut element_names = Vec::with_capacity(rank.usize() - 1);

        for r in Rank::range_iter(Rank::new(1), rank) {
            element_names.push(element_name(r));
        }

        if element_names.len() >= 2 {
            element_names.swap(0, 1);
        }

        for element_name in element_names {
            off.push_str(", ");
            off.push_str(&element_name);
        }

        off.push('\n');
    }

    // Swaps edges and faces, because OFF format bad.
    if rank >= Rank::new(3) {
        el_counts.swap(Rank::new(1), Rank::new(2));
    }

    for r in Rank::range_iter(Rank::new(0), rank) {
        off.push_str(&el_counts[r].to_string());
        off.push(' ');
    }

    off.push('\n');
}

/// Writes the vertices of a polytope into an OFF file.
fn write_vertices(off: &mut String, opt: &OffOptions, vertices: &[Point]) {
    // # Vertices
    if opt.comments {
        off.push_str("\n# ");
        off.push_str(&element_name(Rank::new(0)));
        off.push('\n');
    }

    // Adds the coordinates.
    for v in vertices {
        for c in v.into_iter() {
            off.push_str(&c.to_string());
            off.push(' ');
        }
        off.push('\n');
    }
}

/// Gets and writes the faces of a polytope into an OFF file.
fn write_faces(
    off: &mut String,
    opt: &OffOptions,
    rank: usize,
    edges: &ElementList,
    faces: &ElementList,
) {
    // # Faces
    if opt.comments {
        let el_name = if rank > 2 {
            element_name(Rank::new(2))
        } else {
            COMPONENTS.to_string()
        };

        off.push_str("\n# ");
        off.push_str(&el_name);
        off.push('\n');
    }

    // TODO: write components instead of faces in 2D case.
    // ALSO TODO: reuse code from mesh builder.
    for face in faces.iter() {
        off.push_str(&face.subs.len().to_string());

        // Maps an OFF index into a graph index.
        let mut hash_edges = HashMap::new();
        let mut graph = Graph::new_undirected();

        // Maps the vertex indices to consecutive integers from 0.
        for &edge_idx in &face.subs.0 {
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
            off.push(' ');
            off.push_str(&graph[nx].to_string());
        }
        off.push('\n');
    }
}

/// Writes the n-elements of a polytope into an OFF file.
fn write_els(off: &mut String, opt: &OffOptions, rank: Rank, els: &ElementList) {
    // # n-elements
    if opt.comments {
        off.push_str("\n# ");
        off.push_str(&element_name(rank));
        off.push('\n');
    }

    // Adds the elements' indices.
    for el in els.iter() {
        off.push_str(&el.subs.len().to_string());

        for &sub in &el.subs.0 {
            off.push(' ');
            off.push_str(&sub.to_string());
        }

        off.push('\n');
    }
}

impl Concrete {
    /// Converts a polytope into an OFF file.
    pub fn to_off(&self, opt: OffOptions) -> String {
        let rank = self.rank();
        let vertices = &self.vertices;
        let abs = &self.abs;
        let mut off = String::new();

        // Serialized name.
        off.push_str("# ");
        off.push_str(&ron::to_string(self.name()).unwrap());
        off.push('\n');

        // Blatant advertising.
        if opt.comments {
            off += &format!(
                "# Generated using Miratope v{} (https://github.com/OfficialURL/miratope-rs)\n\n",
                env!("CARGO_PKG_VERSION")
            );
        }

        // Writes header.
        if rank != Rank::new(3) {
            off += &rank.to_string();
        }
        off += "OFF\n";

        // If we have a nullitope or point on our hands, that is all.
        if rank < Rank::new(1) {
            return off;
        }

        // Adds the element counts.
        write_el_counts(&mut off, &opt, self.el_counts());

        // Adds vertex coordinates.
        write_vertices(&mut off, &opt, vertices);

        // Adds faces.
        if rank >= Rank::new(2) {
            write_faces(
                &mut off,
                &opt,
                rank.usize(),
                &abs[Rank::new(1)],
                &abs[Rank::new(2)],
            );
        }

        // Adds the rest of the elements.
        for r in Rank::range_iter(Rank::new(3), rank) {
            write_els(&mut off, &opt, r, &abs[r]);
        }

        off
    }

    /// Writes a polytope's OFF file in a specified file path.
    pub fn to_path(&self, fp: &impl AsRef<Path>, opt: OffOptions) -> IoResult<()> {
        std::fs::write(fp, self.to_off(opt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Used to test a particular polytope.
    fn test_shape(p: Concrete, el_nums: Vec<usize>) {
        // Checks that element counts match up.
        assert_eq!(p.el_counts().0, el_nums);

        // Checks that the polytope can be reloaded correctly.
        assert_eq!(
            Concrete::from_off(&p.to_off(Default::default()))
                .unwrap()
                .el_counts()
                .0,
            el_nums
        );
    }

    #[test]
    /// Checks that a point has the correct amount of elements.
    fn point_nums() {
        let point = Concrete::from_off("0OFF").unwrap();

        test_shape(point, vec![1, 1])
    }

    #[test]
    /// Checks that a dyad has the correct amount of elements.
    fn dyad_nums() {
        let dyad = Concrete::from_off("1OFF 2 -1 1 0 1").unwrap();

        test_shape(dyad, vec![1, 2, 1])
    }

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
        let tet = Concrete::from_off(
            "OFF 4 4 6 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 3 0 1 2 3 3 0 2 3 0 1 3 3 3 1 2",
        )
        .unwrap();

        test_shape(tet, vec![1, 4, 6, 4, 1])
    }

    #[test]
    /// Checks that a 2-tetrahedron compund has the correct amount of elements.
    fn so_nums() {
        let so = Concrete::from_off(
            "OFF 8 8 12 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 -1 1 1 1 -1 3 0 1 2 3 3 0 2 3 0 1 3 3 3 1 2 3 4 5 6 3 7 4 6 3 4 5 7 3 7 5 6 ",
        ).unwrap();

        test_shape(so, vec![1, 8, 12, 8, 1])
    }

    #[test]
    /// Checks that a pentachoron has the correct amount of elements.
    fn pen_nums() {
        let pen =   Concrete::   from_off(
            "4OFF 5 10 10 5 0.158113883008419 0.204124145231932 0.288675134594813 0.5 0.158113883008419 0.204124145231932 0.288675134594813 -0.5 0.158113883008419 0.204124145231932 -0.577350269189626 0 0.158113883008419 -0.612372435695794 0 0 -0.632455532033676 0 0 0 3 0 3 4 3 0 2 4 3 2 3 4 3 0 2 3 3 0 1 4 3 1 3 4 3 0 1 3 3 1 2 4 3 0 1 2 3 1 2 3 4 0 1 2 3 4 0 4 5 6 4 1 4 7 8 4 2 5 7 9 4 3 6 8 9"
                ,
        ).unwrap();

        test_shape(pen, vec![1, 5, 10, 10, 5, 1])
    }

    #[test]
    /// Checks that comments are correctly parsed.
    fn comments() {
        let tet = Concrete::from_off(
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
        )
        .unwrap();

        test_shape(tet, vec![1, 4, 6, 4, 1])
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
