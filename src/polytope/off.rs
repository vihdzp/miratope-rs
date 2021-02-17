use std::collections::HashMap;
use std::fs::read;
use std::io::Result as IoResult;
use std::path::Path;

use petgraph::{graph::Graph, prelude::NodeIndex, Undirected};

use super::*;

/// Removes all whitespace and comments from the OFF file.
fn data_tokens(src: &String) -> impl Iterator<Item = &str> {
    let mut comment = false;
    str::split(&src, move |c: char| {
        if c == '#' {
            comment = true;
        } else if c == '\n' {
            comment = false;
        }
        !comment && c.is_whitespace()
    })
    .filter(|s| match s.chars().next() {
        None => false,
        Some(c) => c != '#',
    })
}

/// Gets the number of elements from the OFF file.
fn get_elem_nums<'a>(dim: usize, toks: &mut impl Iterator<Item = &'a str>) -> Vec<usize> {
    let mut num_elems = Vec::with_capacity(dim);

    for _ in 0..dim {
        let num_elem = toks.next().expect("OFF file ended unexpectedly.");
        num_elems.push(num_elem.parse().expect("could not parse as integer"));
    }

    // 2-elements go before 1-elements, we're undoing that.
    if dim >= 3 {
        num_elems.swap(1, 2);
    }

    num_elems
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
                // Is the clone really necessary?
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

pub fn get_comps(ridges: &ElementList, facets: &ElementList) -> ElementList {
    let num_ridges = ridges.len();
    let num_facets = facets.len();
    let mut g: Graph<(), (), Undirected> = Graph::new_undirected();
    for _ in 0..(num_ridges + num_facets) {
        g.add_node(());
    }

    for (i, f) in facets.iter().enumerate() {
        for r in f.iter() {
            g.add_edge(NodeIndex::new(*r), NodeIndex::new(num_ridges + i), ());
        }
    }

    let g_comps = petgraph::algo::tarjan_scc(&g);
    let mut comps = Vec::with_capacity(g_comps.len());
    for g_comp in g_comps.iter() {
        let mut comp = Vec::new();
        for idx in g_comp.iter() {
            let idx: usize = idx.index();
            if idx < num_ridges {
                comp.push(idx);
            }
        }

        comps.push(comp);
    }

    comps
}

pub fn polytope_from_off_src(src: String) -> PolytopeSerde {
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

    let num_elems = get_elem_nums(dim, &mut toks);
    let vertices = parse_vertices(num_elems[0], dim, &mut toks);

    // Reads edges and faces.
    let mut elements = Vec::with_capacity(dim as usize);
    let (edges, faces) = parse_edges_and_faces(num_elems[1], num_elems[2], &mut toks);
    elements.push(edges);
    elements.push(faces);

    // Adds all higher elements.
    for d in 3..dim {
        elements.push(parse_els(num_elems[d], &mut toks));
    }

    // Adds components.
    elements.push(get_comps(&elements[dim - 3], &elements[dim - 2]));

    PolytopeSerde { vertices, elements }
}

pub fn open_off(fp: &Path) -> IoResult<PolytopeSerde> {
    Ok(polytope_from_off_src(String::from_utf8(read(fp)?).unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tet_counts() {
        let tet: Polytope = polytope_from_off_src(
            "OFF 4 4 6 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 3 0 1 2 3 3 0 2 3 0 1 3 3 3 1 2".to_string(),
        )
        .into();

        assert_eq!(tet.el_counts(), vec![4, 6, 4, 1])
    }

    #[test]
    fn so_counts() {
        let so: Polytope = polytope_from_off_src(
            "OFF 8 8 12 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 -1 1 1 1 -1 3 0 1 2 3 3 0 2 3 0 1 3 3 3 1 2 3 4 5 6 3 7 4 6 3 4 5 7 3 7 5 6 ".to_string(),
        )
        .into();

        assert_eq!(so.el_counts(), vec![8, 12, 8, 2])
    }

    #[test]
    fn comments() {
        let tet: Polytope = polytope_from_off_src(
            "OFF # this
            4 4 6 # is
            # a # test # of
            # Vertices
            1 1 1 # the 3 1
            1 -1 -1 # comment 4 1
            -1 1 -1 # removal 5 9
            -1 -1 1 # system 2 6
            
            # Faces
            3 0 1 2
            3 3 0 2
            3 0 1 3
            3 3 1 2"
                .to_string(),
        )
        .into();

        assert_eq!(tet.el_counts(), vec![4, 6, 4, 1])
    }

    #[test]
    fn pen_counts() {
        let pen: Polytope = polytope_from_off_src(
            "4OFF
            5 10 10 5
            
            # Vertices
            0.158113883008419 0.204124145231932 0.288675134594813 0.5
            0.158113883008419 0.204124145231932 0.288675134594813 -0.5
            0.158113883008419 0.204124145231932 -0.577350269189626 0
            0.158113883008419 -0.612372435695794 0 0
            -0.632455532033676 0 0 0
            
            # Faces
            3 0 3 4
            3 0 2 4
            3 2 3 4
            3 0 2 3
            3 0 1 4
            3 1 3 4
            3 0 1 3
            3 1 2 4
            3 0 1 2
            3 1 2 3
            
            # Cells
            4 0 1 2 3
            4 0 4 5 6
            4 1 4 7 8
            4 2 5 7 9
            4 3 6 8 9"
                .to_string(),
        )
        .into();

        assert_eq!(pen.el_counts(), vec![5, 10, 10, 5, 1])
    }
}
