//! Contains the code that opens an OFF file and parses it into a polytope.

use std::{collections::HashMap, fmt::Display, io::Error as IoError, path::Path, str::FromStr};

use crate::{
    abs::{AbstractBuilder, Ranked, SubelementList, Subelements},
    conc::{cycle::CycleList, Concrete},
    geometry::Point,
    Polytope, COMPONENTS, ELEMENT_NAMES,
};

use vec_like::VecLike;

/// The header for OFF files created with Miratope.
const HEADER: &str = concat!(
    "Generated using Miratope v",
    env!("CARGO_PKG_VERSION"),
    " (https://github.com/galoomba1/miratope-rs)"
);

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

impl Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "row {}, column {}", self.row + 1, self.column + 1)
    }
}

/// Any error encountered while parsing an OFF file.
#[derive(Clone, Copy, Debug)]
pub enum OffParseError {
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

impl Display for OffParseError {
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

impl std::error::Error for OffParseError {}

/// The result of parsing an OFF file.
pub type OffParseResult<T> = Result<T, OffParseError>;

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

impl<'a> Token<'a> {
    /// Attempts to parse the token.
    fn parse<T: FromStr>(&self) -> OffParseResult<T> {
        self.slice
            .parse()
            .map_err(|_| OffParseError::Parsing(self.pos))
    }

    /// Reads the rank from a token of the form `(-?\d+)?OFF`. If the rank is
    /// omitted, we use a default value of 4.
    fn rank(&self) -> OffParseResult<usize> {
        let rank = self
            .slice
            .strip_suffix("OFF")
            .ok_or(OffParseError::MagicWord(self.pos))?;

        if rank.is_empty() {
            Ok(4)
        } else {
            match rank.parse::<isize>() {
                Ok(r) => Ok((r + 1) as usize),
                Err(_) => Err(OffParseError::Rank(self.pos)),
            }
        }
    }
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
    pub fn parse_next<U: FromStr>(&mut self) -> OffParseResult<U> {
        self.next()
            .ok_or(OffParseError::UnexpectedEnding(self.position))?
            .parse()
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

    /// Reads the first token from the OFF file, returns the polytope's rank.
    fn rank(&mut self) -> OffParseResult<usize> {
        self.next().ok_or(OffParseError::Empty)?.rank()
    }

    /// Gets the number of elements from the OFF file from rank 1 up to rank
    /// max(3, dim - 1). For the purposes of the OFF format, the 2-elements of
    /// a polygon are taken to be its components.
    ///
    /// This function ought to only be called when the rank is at least 2.
    fn el_nums(&mut self, rank: usize) -> OffParseResult<Vec<usize>> {
        debug_assert!(rank >= 2);
        let mut el_nums = Vec::with_capacity(rank - 1);

        // Reads entries one by one.
        for _ in 1..rank {
            el_nums.push(self.iter.parse_next()?);
        }

        // A polygon always has as many vertices as edges.
        if rank == 3 {
            el_nums.push(el_nums[0]);
        }

        // 2-elements go before 1-elements, we're undoing that.
        el_nums.swap(1, 2);

        Ok(el_nums)
    }

    /// Parses all vertex coordinates from the OFF file.
    fn parse_vertices(
        &mut self,
        count: usize,
        dim: usize,
    ) -> OffParseResult<Vec<Point<f64>>> {
        // Reads all vertices.
        let mut vertices = Vec::with_capacity(count);

        // Add each vertex to the vector.
        for _ in 0..count {
            let mut v = Vec::with_capacity(dim);

            for _ in 0..dim {
                v.push(self.iter.parse_next()?);
            }

            vertices.push(v.into());
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
    ) -> OffParseResult<(SubelementList, SubelementList)> {
        let mut edges = SubelementList::with_capacity(num_edges);
        let mut faces = SubelementList::with_capacity(num_faces);
        let mut hash_edges = HashMap::new();

        // Add each face to the element list.
        for _ in 0..num_faces {
            let face_sub_num = self.iter.parse_next()?;
            let mut face = Subelements::new();
            let mut face_verts = Vec::with_capacity(face_sub_num + 1);

            // Reads all vertices of the face.
            for _ in 0..face_sub_num {
                face_verts.push(self.iter.parse_next()?);
            }

            // We add the first vertex to the end for simplicity.
            face_verts.push(face_verts[0]);

            // Gets all edges of the face.
            for i in 0..face_sub_num {
                let mut v0 = face_verts[i];
                let mut v1 = face_verts[i + 1];

                if v0 > v1 {
                    std::mem::swap(&mut v0, &mut v1);
                }

                let edge: Subelements = vec![v0, v1].into();

                if let Some(idx) = hash_edges.get(&edge) {
                    face.push(*idx);
                } else {
                    hash_edges.insert(edge.clone(), edges.len());
                    face.push(edges.len());
                    edges.push(edge);
                }
            }

            // If these are truly faces and not just components, we add them.
            // Hopefully the compiler can optimize this better, I'm lazy.
            if rank != 3 {
                faces.push(face);
            }
        }

        // If this is a polygon, we add a single maximal element as a face.
        if rank == 3 {
            faces = SubelementList::max(edges.len());
        }

        // The number of edges in the file should match the number of read
        // edges, though this isn't obligatory.
        if edges.len() != num_edges {
            println!("WARNING: Edge count doesn't match expected edge count!");
        }

        Ok((edges, faces))
    }

    /// Parses the next set of d-elements from the OFF file.
    fn parse_els(&mut self, num_el: usize) -> OffParseResult<SubelementList> {
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
    pub fn build(mut self) -> OffParseResult<Concrete> {
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
        self.abs.reserve(rank + 2);
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

        // Safety: TODO this isn't actually safe. We need to do some checking.
        Ok(Concrete::new(vertices, unsafe { self.abs.build() }))
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

/// An error while writing an OFF file.
#[derive(Clone, Copy, Debug)]
pub enum OffWriteError {
    /// The polytope has a compound 2-element, which can't be represented in
    /// the OFF format.
    CompoundFace {
        /// The index of the compound element.
        idx: usize,
    },

    /// The polytope has two edges with the same vertices, which can't be
    /// represented in the OFF format.
    CoincidentEdges {
        /// The index of the first edge.
        idx0: usize,

        /// The index of the second edge.
        idx1: usize,
    },
}

impl OffWriteError {
    /// Returns the error in which two edges coincide. These must be ordered so
    /// that we can run consistent tests on these.
    fn coincident_edges(mut idx0: usize, mut idx1: usize) -> Self {
        if idx0 > idx1 {
            std::mem::swap(&mut idx0, &mut idx1);
        }

        Self::CoincidentEdges { idx0, idx1 }
    }
}

impl Display for OffWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::CompoundFace { idx } => {
                write!(f, "cannot write compound face with index {}", idx)
            }
            Self::CoincidentEdges { idx0, idx1 } => write!(
                f,
                "cannot write coincident edges with indices {} and {}",
                idx0, idx1
            ),
        }
    }
}

impl std::error::Error for OffWriteError {}

type OffWriteResult<T> = Result<T, OffWriteError>;

/// An auxiliary struct to write a polytope to an OFF file.
pub struct OffWriter<'a> {
    /// The output OFF file, as a string. (Maybe we should use a file writer
    /// or something similar instead?)
    off: String,

    /// The polytope that we're converting into an OFF file.
    poly: &'a Concrete,

    /// Options for the text output.
    options: OffOptions,
}

impl<'a> OffWriter<'a> {
    /// Initializes a new OFF writer from a polytope, with a given set of
    /// options.
    pub fn new(poly: &'a Concrete, options: OffOptions) -> Self {
        Self {
            off: String::new(),
            poly,
            options,
        }
    }

    /// Returns the rank of the polytope.
    fn rank(&self) -> usize {
        self.poly.rank()
    }

    /// Returns the number of elements of a given rank in the polytope.
    fn el_count(&self, rank: usize) -> usize {
        self.poly.el_count(rank)
    }

    /// Whether the OFF file should have comments specifying each face type.
    fn comments(&self) -> bool {
        self.options.comments
    }

    /// Appends a given character to the OFF file.
    fn push(&mut self, ch: char) {
        self.off.push(ch)
    }

    /// Appends a given string slice to the OFF file.
    fn push_str<U: AsRef<str>>(&mut self, string: U) {
        self.off.push_str(string.as_ref())
    }

    /// Appends some data to the OFF file.
    fn push_to_str<U: ToString>(&mut self, data: U) {
        self.push_str(data.to_string())
    }

    /// Writes the OFF format header.
    fn write_rank(&mut self) {
        let rank = self.rank();
        if rank != 4 {
            self.push_to_str(rank as isize - 1);
        }
        self.push_str("OFF\n");
    }

    /// Checks that no two edges coincide.
    ///
    /// This should only be called for polytopes with rank at least 3.
    fn check_edges(&self) -> OffWriteResult<()> {
        use std::collections::hash_map::Entry::*;
        let mut hash_edges = HashMap::new();

        for (idx0, edge) in self.poly[2].iter().enumerate() {
            let subs = &edge.subs;
            let mut edge = (subs[0], subs[1]);

            // We sort each edge.
            if edge.0 > edge.1 {
                std::mem::swap(&mut edge.0, &mut edge.1);
            }

            // We verify no other edge has the same vertices.
            match hash_edges.entry(edge) {
                Occupied(entry) => return Err(OffWriteError::coincident_edges(idx0, *entry.get())),
                Vacant(entry) => {
                    entry.insert(idx0);
                }
            }
        }

        Ok(())
    }

    /// Writes the polytope's element counts into an OFF file.
    ///
    /// This method should only be called on polytopes of rank at least 2.
    fn write_el_counts(&mut self) {
        let rank = self.rank();
        debug_assert!(rank >= 2);

        // # Vertices, Faces, Edges, ...
        if self.comments() {
            self.push_str("\n# Vertices");

            if rank == 3 {
                self.push_str(", Components");
            } else {
                self.push_str(", Faces, Edges");

                for r in 4..rank {
                    self.push_str(", ");
                    self.push_str(element_name(r));
                }
            }

            self.push('\n');
        }

        self.push_to_str(self.el_count(1));

        match rank {
            2 => {}
            3 => {
                self.push(' ');
                self.push_to_str(
                    CycleList::from_edges(self.poly[2].iter().map(|edge| &edge.subs)).len(),
                );
            }
            _ => {
                // Swaps edges and faces because OFF format bad.
                self.push(' ');
                self.push_to_str(self.el_count(3));
                self.push(' ');
                self.push_to_str(self.el_count(2));

                for r in 4..rank {
                    self.push(' ');
                    self.push_to_str(self.el_count(r));
                }
            }
        }

        self.push('\n');
    }

    /// Writes the vertices of a polytope into an OFF file.
    fn write_vertices(&mut self) {
        // # Vertices
        if self.comments() {
            self.push_str("\n# ");
            self.push_str(element_name(1));
            self.push('\n');
        }

        // Adds the coordinates.
        for v in &self.poly.vertices {
            for c in v {
                self.push_to_str(c);
                self.push(' ');
            }
            self.push('\n');
        }
    }

    /// Gets and writes the faces of a polytope into an OFF file.
    ///
    /// This method should only be called when rank >= 3.
    fn write_faces(&mut self) -> OffWriteResult<()> {
        let rank = self.rank();
        debug_assert!(rank >= 3);

        // # Faces
        if self.comments() {
            let el_name = if rank > 3 { "Faces" } else { COMPONENTS };
            self.push_str("\n# ");
            self.push_str(el_name);
            self.push('\n');
        }

        // Writes the components in the polygonal case.
        if rank == 3 {
            for component in CycleList::from_edges(self.poly[1].iter().map(|vert| &vert.sups)) {
                self.push_to_str(component.len());
                for edge in component {
                    self.push(' ');
                    self.push_to_str(edge);
                }
                self.push('\n');
            }
        } else {
            for (idx, face) in self.poly[3].iter().enumerate() {
                self.push_to_str(face.subs.len());
                let mut cycles =
                    CycleList::from_edges(face.subs.iter().map(|&i| &self.poly[(2, i)].subs));

                if cycles.len() > 1 {
                    return Err(OffWriteError::CompoundFace { idx });
                }

                for v in cycles.swap_remove(0) {
                    self.push(' ');
                    self.push_to_str(v);
                }
                self.push('\n');
            }
        }

        Ok(())
    }

    /// Writes the n-elements of a polytope into an OFF file.
    fn write_els(&mut self, rank: usize) {
        // # n-elements
        if self.comments() {
            self.push_str("\n# ");
            self.push_str(element_name(rank));
            self.push('\n');
        }

        // Adds the elements' indices.
        for el in &self.poly[rank] {
            let subs = &el.subs;
            self.push_to_str(subs.len());

            for &sub in subs {
                self.push(' ');
                self.push_to_str(sub);
            }

            self.push('\n');
        }
    }

    /// Consumes the OFF writer, returns the actual OFF file as a `String`.
    pub fn build(mut self) -> OffWriteResult<String> {
        let rank = self.poly.rank();

        // Serialized name.
        /* self.off.push_str("# ");
        self.off
            .push_str(&ron::to_string(&self.polytope.name).unwrap_or_default());
        self.off.push('\n'); */

        // Blatant advertising.
        if self.comments() {
            self.push_str("# ");
            self.push_str(HEADER);
            self.push('\n');
        }

        // Writes header.
        self.write_rank();

        // If we have a nullitope or point on our hands, that is all.
        if rank < 2 {
            return Ok(self.off);
        }

        // Checks that no two edges coincide.
        self.check_edges()?;

        // Adds the element counts.
        self.write_el_counts();

        // Adds vertex coordinates.
        self.write_vertices();

        // Adds faces.
        if rank >= 3 {
            self.write_faces()?;
        }

        // Adds the rest of the elements.
        for r in 4..rank {
            self.write_els(r);
        }

        Ok(self.off)
    }
}

/// An error when saving an OFF file.
#[derive(Debug)]
pub enum OffSaveError {
    /// The OFF file couldn't be created.
    OffWriteError(OffWriteError),

    /// There was a problem saving the file.
    IoError(IoError),
}

impl From<OffWriteError> for OffSaveError {
    fn from(err: OffWriteError) -> Self {
        Self::OffWriteError(err)
    }
}

impl From<IoError> for OffSaveError {
    fn from(err: IoError) -> Self {
        Self::IoError(err)
    }
}

impl Display for OffSaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OffWriteError(err) => err.fmt(f),
            Self::IoError(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for OffSaveError {}

/// The result of trying to save an OFF file.
type OffSaveResult<T> = Result<T, OffSaveError>;

//todo: put this in its own trait
impl Concrete {
    /// Converts a polytope into an OFF file.
    pub fn to_off(&self, options: OffOptions) -> OffWriteResult<String> {
        OffWriter::new(self, options).build()
    }

    /// Writes a polytope's OFF file in a specified file path.
    pub fn to_path<P: AsRef<Path>>(&self, fp: P, opt: OffOptions) -> OffSaveResult<()> {
        std::fs::write(fp, self.to_off(opt)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::FromFile;
    use crate::test;

    /// Tests a particular OFF file.
    fn test_off_file<I: IntoIterator<Item = usize> + Clone>(src: &str, element_counts: I) {
        // Checks that element counts match up.
        let poly = Concrete::from_off(src).expect("OFF file could not be loaded.");
        test(&poly, element_counts.clone());

        // Checks that the polytope can be reloaded correctly.
        const ERR: &str = "OFF file could not be reloaded.";
        test(
            &Concrete::from_off(&poly.to_off(Default::default()).expect(ERR)).expect(ERR),
            element_counts,
        );
    }

    /// Tests a particular OFF file in the folder.
    macro_rules! test_off {
        ($path:literal, $element_counts:expr) => {
            test_off_file(include_str!(concat!($path, ".off")), $element_counts)
        };
    }

    /// Checks that a point has the correct amount of elements.
    #[test]
    fn point_nums() {
        test_off!("point", [1, 1])
    }

    /// Checks that a dyad has the correct amount of elements.
    #[test]
    fn dyad_nums() {
        test_off!("dyad", [1, 2, 1])
    }

    /// Checks that a hexagon has the correct amount of elements.
    #[test]
    fn hig_nums() {
        test_off!("hig", [1, 6, 6, 1])
    }

    /// Checks that a hexagram has the correct amount of elements.
    #[test]
    fn shig_nums() {
        test_off!("shig", [1, 6, 6, 1])
    }

    /// Checks that a tetrahedron has the correct amount of elements.
    #[test]
    fn tet_nums() {
        test_off!("tet", [1, 4, 6, 4, 1])
    }

    /// Checks that a 2-tetrahedron compund has the correct amount of elements.
    #[test]
    fn so_nums() {
        test_off!("so", [1, 8, 12, 8, 1])
    }

    /// Checks that a pentachoron has the correct amount of elements.
    #[test]
    fn pen_nums() {
        test_off!("pen", [1, 5, 10, 10, 5, 1])
    }

    /// Checks that comments are correctly parsed.
    #[test]
    fn comments() {
        test_off!("comments", [1, 4, 6, 4, 1])
    }

    /// Attempts to parse an OFF file, unwraps it.
    fn unwrap_off(src: &str) {
        Concrete::from_off(src).unwrap();
    }

    /// An empty file should fail.
    #[test]
    #[should_panic(expected = "Empty")]
    fn empty() {
        unwrap_off("")
    }

    /// A file without a valid rank should fail.
    #[test]
    #[should_panic(expected = "Rank(Position { row: 0, column: 3 })")]
    fn rank() {
        unwrap_off("   fooOFF")
    }

    /// A file without the magic word should fail.
    #[test]
    #[should_panic(expected = "MagicWord(Position { row: 1, column: 3 })")]
    fn magic_word() {
        unwrap_off("# comment\n   foo bar")
    }

    /// A file with some invalid token should fail.
    #[test]
    #[should_panic(expected = "Parsing(Position { row: 1, column: 3 })")]
    fn parse() {
        unwrap_off("OFF\n10 foo bar")
    }
}
