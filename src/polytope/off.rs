//! Contains methods to load and save OFF files.
//! ## What is an OFF file?
//! An OFF file stores the geometric data of a polytope.
//! [insert format specification here]

use std::collections::HashMap;
use std::io::Result as IoResult;
use std::path::Path;

use petgraph::{graph::NodeIndex, visit::Dfs, Graph};

use super::super::*;

const ELEMENT_NAMES: [&str; 11] = [
    "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta", "Xenna", "Daka",
];
const COMPONENTS: &str = "Components";

/// Gets the name for an element with `dim` dimensions.
fn element_name(dim: usize) -> String {
    match ELEMENT_NAMES.get(dim) {
        Some(&name) => String::from(name),
        None => dim.to_string() + "-elements",
    }
}

/// Removes all whitespace and comments from the OFF file.
fn data_tokens(src: &String) -> impl Iterator<Item = &str> {
    let mut comment = false;
    str::split(&src, move |c: char| {
        if c == '#' {
            comment = true;
        } else if c == '\n' {
            comment = false;
        }
        comment || c.is_whitespace()
    })
    .filter(|s| !s.is_empty())
}

/// Gets the number of elements from the OFF file.
/// This includes components iff dim ≤ 2, as this makes things easier down the line.
fn get_el_counts<'a>(dim: usize, toks: &mut impl Iterator<Item = &'a str>) -> Vec<usize> {
    let mut el_counts = Vec::with_capacity(dim);

    // Reads entries one by one.
    for _ in 0..dim {
        let num_elem = toks.next().expect("OFF file ended unexpectedly.");
        el_counts.push(num_elem.parse().expect("could not parse as integer"));
    }

    // A point has a single component (itself)
    if dim == 0 {
        el_counts.push(1);
    }
    // A dyad has twice as many vertices as components.
    else if dim == 1 {
        let comps = el_counts[0] / 2;
        el_counts.push(comps);
    }
    // A polygon always has as many vertices as edges.
    else if dim == 2 {
        el_counts.push(el_counts[0]);
    }

    // 2-elements go before 1-elements, we're undoing that.
    if dim >= 2 {
        el_counts.swap(1, 2);
    }

    el_counts
}

/// Parses all vertex coordinates from the OFF file.
fn parse_vertices<'a>(
    num: usize,
    dim: usize,
    toks: &mut impl Iterator<Item = &'a str>,
) -> Vec<nalgebra::DVector<f64>> {
    // Reads all vertices.
    let mut vertices = Vec::with_capacity(num);

    // Add each vertex to the vector.
    for _ in 0..num {
        let mut vert = Vec::with_capacity(dim);

        for _ in 0..dim {
            let coord = toks.next().expect("OFF file ended unexpectedly.");
            vert.push(coord.parse().expect("Float parsing failed!"));
        }

        vertices.push(vert.into());
    }

    vertices
}

/// Reads the faces from the OFF file and gets the edges and faces from them.
/// Since the OFF file doesn't store edges explicitly, this is harder than reading
/// general elements.
fn parse_edges_and_faces<'a>(
    num_edges: usize,
    num_faces: usize,
    toks: &mut impl Iterator<Item = &'a str>,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let (mut edges, mut faces) = (Vec::with_capacity(num_edges), Vec::with_capacity(num_faces));
    let mut hash_edges = HashMap::new();

    // Add each face to the element list.
    for _ in 0..num_faces {
        let face_sub_count: usize = toks
            .next()
            .expect("OFF file ended unexpectedly.")
            .parse()
            .expect("Integer parsing failed!");

        let mut face = Vec::with_capacity(face_sub_count);
        let mut verts = Vec::with_capacity(face_sub_count);

        // Reads all vertices of the face.
        for _ in 0..face_sub_count {
            verts.push(
                toks.next()
                    .expect("OFF file ended unexpectedly.")
                    .parse()
                    .expect("Integer parsing failed!"),
            );
        }

        // Gets all edges of the face.
        for i in 0..face_sub_count {
            let mut edge = vec![verts[i], verts[(i + 1) % face_sub_count]];
            edge.sort();

            if let Some(idx) = hash_edges.get(&edge) {
                face.push(*idx);
            } else {
                hash_edges.insert(edge.clone(), edges.len());
                face.push(edges.len());
                edges.push(edge);
            }
        }

        faces.push(face);
    }

    // The number of edges in the file should match the number of read edges, though this isn't obligatory.
    if edges.len() != num_edges {
        println!("Edge count doesn't match expected edge count!");
    }

    (edges, faces)
}

/// Reads the next set of elements from the OFF file, starting from cells.
pub fn parse_els<'a>(num_els: usize, toks: &mut impl Iterator<Item = &'a str>) -> ElementList {
    let mut els = Vec::with_capacity(num_els);

    // Adds every d-element to the element list.
    for _ in 0..num_els {
        let el_sub_count = toks
            .next()
            .expect("OFF file ended unexpectedly.")
            .parse()
            .expect("Integer parsing failed!");
        let mut el = Vec::with_capacity(el_sub_count);

        // Reads all sub-elements of the d-element.
        for _ in 0..el_sub_count {
            let el_sub = toks.next().expect("OFF file ended unexpectedly.");
            el.push(el_sub.parse().expect("Integer parsing failed!"));
        }

        els.push(el);
    }

    els
}

/// Builds a [`Polytope`] from the string representation of an OFF file.
pub fn from_src(src: String) -> Polytope {
    let mut toks = data_tokens(&src);
    let dim = {
        let first = toks.next().expect("OFF file empty");
        let dim = first.strip_suffix("OFF").expect("no \"OFF\" detected");

        if dim.is_empty() {
            3
        } else {
            dim.parse()
                .expect("could not parse dimension as an integer")
        }
    };

    let num_elems = get_el_counts(dim, &mut toks);
    let vertices = parse_vertices(num_elems[0], dim, &mut toks);
    let mut elements = Vec::with_capacity(dim);

    // Reads edges and faces.
    if dim >= 2 {
        let (edges, faces) = parse_edges_and_faces(num_elems[1], num_elems[2], &mut toks);
        elements.push(edges);
        elements.push(faces);
    }

    // Adds all higher elements.
    for d in 3..dim {
        elements.push(parse_els(num_elems[d], &mut toks));
    }

    // Adds components.
    if dim >= 3 {
        return Polytope::new_wo_comps(vertices, elements);
    }
    // Deals with the weird 1D case.
    else if dim == 1 {
        let comp_count = num_elems[1];
        let mut components = Vec::with_capacity(comp_count);

        for _ in 0..comp_count {
            let mut comp = Vec::with_capacity(2);

            for _ in 0..2 {
                comp.push(
                    toks.next()
                        .expect("OFF file ended unexpectedly.")
                        .parse()
                        .expect("Integer parsing failed!"),
                );
            }

            components.push(comp);
        }

        elements.push(components);
    }

    Polytope::new(vertices, elements)
}

/// Loads a polytope from a file path.
pub fn from_path(fp: &impl AsRef<Path>) -> IoResult<Polytope> {
    Ok(from_src(String::from_utf8(std::fs::read(fp)?).unwrap()))
}

/// A set of options to be used when saving the OFF file.
#[derive(Clone, Copy)]
pub struct OFFOptions {
    /// Whether the OFF file should have comments specifying each face type.
    pub comments: bool,
}

impl Default for OFFOptions {
    fn default() -> Self {
        OFFOptions { comments: true }
    }
}

/// Writes the vertices of a polytope into an OFF file.
fn write_vertices(off: &mut String, opt: &OFFOptions, vertices: &Vec<Point>) {
    // # Vertices
    if opt.comments {
        off.push_str("\n# ");
        off.push_str(&element_name(0).to_string());
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
    opt: &OFFOptions,
    dim: usize,
    edges: &ElementList,
    faces: &ElementList,
) {
    // # Faces
    if opt.comments {
        let el_name = if dim > 2 {
            element_name(2)
        } else {
            COMPONENTS.to_string()
        };

        off.push_str("\n# ");
        off.push_str(&el_name);
        off.push('\n');
    }

    // Retrieves the faces, writes them line by line.
    for f in faces {
        off.push_str(&f.len().to_string());

        // Maps an OFF index into a graph index.
        let mut hash_edges = HashMap::new();
        let mut graph = Graph::new_undirected();

        // Maps the vertex indices to consecutive integers from 0.
        for &e in f {
            let e = &edges[e];
            let mut hash_edge = Vec::with_capacity(2);

            for &v in e {
                match hash_edges.get(&v) {
                    Some(&idx) => hash_edge.push(idx),
                    None => {
                        let idx = hash_edges.len();
                        hash_edges.insert(v, idx);
                        hash_edge.push(idx);

                        graph.add_node(v);
                    }
                }
            }
        }

        // There should be as many graph indices as edges on the face.
        // Otherwise, something went wrong.
        debug_assert_eq!(hash_edges.len(), f.len());

        // Adds the edges to the graph.
        for &e in f {
            let e = &edges[e];
            graph.add_edge(
                NodeIndex::new(*hash_edges.get(&e[0]).unwrap()),
                NodeIndex::new(*hash_edges.get(&e[1]).unwrap()),
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
fn write_els(off: &mut String, opt: &OFFOptions, d: usize, els: &ElementList) {
    // # n-elements
    if opt.comments {
        off.push_str("\n# ");
        off.push_str(&element_name(d).to_string());
        off.push('\n');
    }

    // Adds the elements' indices.
    for el in els {
        off.push_str(&el.len().to_string());

        for sub in el {
            off.push(' ');
            off.push_str(&sub.to_string());
        }
        off.push('\n');
    }
}

/// Converts a polytope into an OFF file.
pub fn to_src(p: &Polytope, opt: OFFOptions) -> String {
    let dim = p.rank();
    let vertices = &p.vertices;
    let elements = &p.elements;
    let mut off = String::new();

    // Blatant advertising
    if opt.comments {
        off += &format!(
            "# Generated using Miratope v{} (https://github.com/OfficialURL/miratope-rs)\n",
            env!("CARGO_PKG_VERSION")
        );
    }

    // Writes header.
    if dim != 3 {
        off += &dim.to_string();
    }
    off += "OFF\n";

    // If we have a 0-polytope on our hands, that is all.
    if dim == 0 {
        return off;
    }

    // Comment before element counts (TODO check 2D and lower).
    if opt.comments {
        off += "\n# Vertices";

        let mut element_names = Vec::with_capacity(dim - 1);

        for d in 1..dim {
            element_names.push(element_name(d));
        }

        if element_names.len() >= 2 {
            element_names.swap(0, 1);
        }

        for d in 0..(dim - 1) {
            off += ", ";
            off += &element_names[d];
        }

        off += "\n";
    }

    // Adds element counts.
    let mut el_counts = p.el_counts();

    if el_counts.len() >= 3 {
        el_counts.swap(1, 2);
    }

    for el_count in &el_counts[0..el_counts.len() - 1] {
        off += dbg!(&el_count.to_string());
        off += " ";
    }
    off += "\n";

    // Adds vertex coordinates.
    write_vertices(&mut off, &opt, vertices);

    // Takes care of the weird 1D case.
    if dim == 1 {
        if opt.comments {
            off += "\n# ";
            off += COMPONENTS;
            off += "\n";
        }

        for comp in &elements[0] {
            off += &format!("{} {}\n", comp[0], comp[1]);
        }
    }

    // Adds faces.
    if dim >= 2 {
        let (edges, faces) = (&elements[0], &elements[1]);
        write_faces(&mut off, &opt, dim, edges, faces);
    }

    // Adds the rest of the elements.
    for d in 3..dim {
        let els = &elements[d - 1];
        write_els(&mut off, &opt, d, els);
    }

    off
}

/// Writes a polytope's OFF file in a specified file path.
pub fn to_path(fp: &impl AsRef<Path>, p: &Polytope, opt: OFFOptions) -> IoResult<()> {
    std::fs::write(fp, to_src(p, opt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_counts() {
        let point: Polytope = from_src("0OFF".to_string()).into();

        assert_eq!(point.el_counts(), vec![1])
    }

    #[test]
    fn dyad_counts() {
        let point: Polytope = from_src("1OFF 2 -1 1 0 1".to_string()).into();

        assert_eq!(point.el_counts(), vec![2, 1])
    }

    #[test]
    fn hig_counts() {
        let hig: Polytope = from_src(
            "2OFF 6 1 1 0 0.5 0.8660254037844386 -0.5 0.8660254037844386 -1 0 -0.5 -0.8660254037844386 0.5 -0.8660254037844386 6 0 1 2 3 4 5".to_string()
        ).into();

        assert_eq!(hig.el_counts(), vec![6, 6, 1])
    }

    #[test]
    fn shig_counts() {
        let shig: Polytope = from_src(
            "2OFF 6 2 1 0 0.5 0.8660254037844386 -0.5 0.8660254037844386 -1 0 -0.5 -0.8660254037844386 0.5 -0.8660254037844386 3 0 2 4 3 1 3 5".to_string()
        ).into();

        assert_eq!(shig.el_counts(), vec![6, 6, 2])
    }

    #[test]
    fn tet_counts() {
        let tet: Polytope = from_src(
            "OFF 4 4 6 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 3 0 1 2 3 3 0 2 3 0 1 3 3 3 1 2".to_string(),
        )
        .into();

        assert_eq!(tet.el_counts(), vec![4, 6, 4, 1])
    }

    #[test]
    fn so_counts() {
        let so: Polytope = from_src(
            "OFF 8 8 12 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 -1 1 1 1 -1 3 0 1 2 3 3 0 2 3 0 1 3 3 3 1 2 3 4 5 6 3 7 4 6 3 4 5 7 3 7 5 6 ".to_string(),
        )
        .into();

        assert_eq!(so.el_counts(), vec![8, 12, 8, 2])
    }

    #[test]
    fn pen_counts() {
        let pen: Polytope = from_src(
            "4OFF 5 10 10 5 0.158113883008419 0.204124145231932 0.288675134594813 0.5 0.158113883008419 0.204124145231932 0.288675134594813 -0.5 0.158113883008419 0.204124145231932 -0.577350269189626 0 0.158113883008419 -0.612372435695794 0 0 -0.632455532033676 0 0 0 3 0 3 4 3 0 2 4 3 2 3 4 3 0 2 3 3 0 1 4 3 1 3 4 3 0 1 3 3 1 2 4 3 0 1 2 3 1 2 3 4 0 1 2 3 4 0 4 5 6 4 1 4 7 8 4 2 5 7 9 4 3 6 8 9"
                .to_string(),
        )
        .into();

        assert_eq!(pen.el_counts(), vec![5, 10, 10, 5, 1])
    }

    #[test]
    fn comments() {
        let tet: Polytope = from_src(
            "# So
            OFF # this
            4 4 6 # is
            # a # test # of
            1 1 1 # the 3 1
            1 -1 -1 # comment 4 1
            -1 1 -1 # removal 5 9
            -1 -1 1 # system 2 6
            3 0 1 2 #let #us #see
            3 3 0 2# if
            3 0 1 3#it
            3 3 1 2#works!#"
                .to_string(),
        )
        .into();

        assert_eq!(tet.el_counts(), vec![4, 6, 4, 1])
    }

    #[test]
    fn load_reload_point() {
        let point: Polytope = from_src("0OFF".to_string()).into();
        let point_src = to_src(&point, Default::default());
        let point_reload: Polytope = from_src(point_src).into();

        assert_eq!(point.el_counts(), point_reload.el_counts())
    }

    #[test]
    fn load_reload_dyad() {
        let dyad: Polytope = from_src("1OFF 2 -1 1 0 1".to_string()).into();
        let dyad_src = to_src(&dyad, Default::default());
        let dyad_reload: Polytope = from_src(dyad_src).into();

        assert_eq!(dyad.el_counts(), dyad_reload.el_counts())
    }

    #[test]
    fn load_reload_cube() {
        let cube: Polytope = from_src("OFF 8 6 12 0.5 0.5 0.5 0.5 0.5 -0.5 0.5 -0.5 0.5 0.5 -0.5 -0.5 -0.5 0.5 0.5 -0.5 0.5 -0.5 -0.5 -0.5 0.5 -0.5 -0.5 -0.5 4 4 0 2 6 4 0 1 3 2 4 6 7 3 2 4 5 7 6 4 4 4 0 1 5 4 7 5 1 3".to_string()).into();
        let cube_src = to_src(&cube, Default::default());
        let cube_reload: Polytope = from_src(cube_src).into();

        assert_eq!(cube.el_counts(), cube_reload.el_counts())
    }

    #[test]
    #[should_panic(expected = "OFF file empty")]
    fn empty() {
        Polytope::from(from_src("".to_string()));
    }

    #[test]
    #[should_panic(expected = "no \"OFF\" detected")]
    fn magic_num() {
        Polytope::from(from_src("foo bar".to_string()));
    }
}
