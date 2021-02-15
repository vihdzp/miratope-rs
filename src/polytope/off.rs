use std::collections::HashMap;
use std::fs::read;
use std::io::Result as IoResult;
use std::path::Path;

use super::*;

fn strip_comments(src: &mut String) {
    while let Some(com_start) = src.chars().position(|c| c == '#') {
        let line_end = com_start
            + src[com_start..]
                .chars()
                .position(|c| c == '\n')
                .unwrap_or_else(|| src.len() - com_start - 1);
        src.drain(com_start..=line_end);
    }
}

fn non_ws_subs(src: &String) -> impl Iterator<Item = &str> {
    str::split(&src, |c: char| c.is_whitespace()).filter(|s| !s.is_empty())
}

fn get_elem_nums<'a>(subs: &mut impl Iterator<Item = &'a str>, dim: usize) -> Vec<usize> {
    let mut num_elems = Vec::with_capacity(dim);

    for _ in 0..dim {
        let next_subs = subs.next().expect("supposed to be enough numbers here");
        num_elems.push(next_subs.parse().expect("could not parse as integer"));
    }

    // 2-elements go before 1-elements, we're undoing that.
    if dim >= 3 {
        num_elems.swap(1, 2);
    }

    num_elems
}

fn parse_vertices<'a>(
    num: usize,
    dim: usize,
    subs: &mut impl Iterator<Item = &'a str>,
) -> Vec<nalgebra::DVector<f64>> {
    // Reads all vertices.
    let mut vertices = Vec::with_capacity(num);

    // Add each vertex to the vector.
    for _ in 0..num {
        let mut vert = Vec::with_capacity(dim);

        for _ in 0..dim {
            vert.push(subs.next().unwrap().parse().expect("Float parsing failed!"));
        }

        vertices.push(vert.into());
    }

    vertices
}

// no hugs and kisses comes to the person who made this format
fn parse_edges_and_faces<'a>(
    num_edges: usize,
    num_faces: usize,
    subs: &mut impl Iterator<Item = &'a str>,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let (mut edges, mut faces) = (Vec::with_capacity(num_edges), Vec::with_capacity(num_faces));
    let mut hash_edges = HashMap::new();

    // Add each face to the element list.
    for _ in 0..num_faces {
        let face_sub_count: usize = subs
            .next()
            .unwrap()
            .parse()
            .expect("Integer parsing failed!");

        let mut face = Vec::with_capacity(face_sub_count);
        let mut verts = Vec::with_capacity(face_sub_count);

        // Reads all vertices of the face.
        for _ in 0..face_sub_count {
            verts.push(
                subs.next()
                    .unwrap()
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
    assert_eq!(
        edges.len(),
        num_edges,
        "Edge count doesn't match expected edge count!",
    );

    (edges, faces)
}

pub fn polytope_from_off_src(mut src: String) -> PolytopeSerde {
    strip_comments(&mut src);
    let mut subs = non_ws_subs(&src);
    let dim = {
        let first = subs.next().expect("need an OFF magic number and dim");
        let dim = first.strip_suffix("OFF").expect("no \"OFF\" detected");

        if dim.is_empty() {
            3
        } else {
            dim.parse()
                .expect("could not parse dimension as an integer")
        }
    };

    let num_elems = get_elem_nums(&mut subs, dim);
    let vertices = parse_vertices(num_elems[0], dim, &mut subs);

    // Reads edges and faces.
    let mut elements = Vec::with_capacity(dim as usize);
    let (edges, faces) = parse_edges_and_faces(num_elems[1], num_elems[2], &mut subs);
    elements.push(edges);
    elements.push(faces);

    // Adds all higher elements.
    for d in 3..dim {
        let num_els = num_elems[d];
        let mut els = Vec::with_capacity(num_els);

        // Adds every d-element to the element list.
        for _ in 0..num_els {
            let el_sub_count = subs
                .next()
                .unwrap()
                .parse()
                .expect("Integer parsing failed!");
            let mut el = Vec::with_capacity(el_sub_count);

            // Reads all sub-elements of the d-element.
            for _ in 0..el_sub_count {
                el.push(
                    subs.next()
                        .unwrap()
                        .parse()
                        .expect("Integer parsing failed!"),
                );
            }

            els.push(el);
        }

        elements.push(els);
    }

    PolytopeSerde { vertices, elements }
}

pub fn open_off(fp: &Path) -> IoResult<PolytopeSerde> {
    Ok(polytope_from_off_src(String::from_utf8(read(fp)?).unwrap()))
}
