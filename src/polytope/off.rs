use std::collections::HashMap;
use std::fs::read;
use std::io::Result as IoResult;
use std::path::Path;

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
pub fn read_next_els<'a>(num_els: usize, toks: &mut impl Iterator<Item = &'a str>) -> ElementList {
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
        elements.push(read_next_els(num_elems[d], &mut toks));
    }

    PolytopeSerde { vertices, elements }
}

pub fn open_off(fp: &Path) -> IoResult<PolytopeSerde> {
    Ok(polytope_from_off_src(String::from_utf8(read(fp)?).unwrap()))
}
