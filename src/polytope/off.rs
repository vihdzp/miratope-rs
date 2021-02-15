use std::io::Result;
use std::path::Path;
use std::{collections::HashMap, fs::read};

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

fn homogenize_whitespace(src: &mut String) {
    *src = src.trim().to_string();
    let mut new_src = String::with_capacity(src.capacity());
    let mut was_ws = false;

    for c in src.chars() {
        if c.is_whitespace() && !was_ws {
            new_src.push(' ');
            was_ws = true;
        } else {
            new_src.push(c);
            was_ws = false;
        }
    }

    *src = new_src;
}

fn read_usize(chars: &mut impl Iterator<Item = char>) -> usize {
    let mut n = 0;

    while let Some(c @ '0'..='9') = chars.next() {
        n *= 10;
        n += (c as usize) - ('0' as usize);
    }

    n
}

fn read_f64(chars: &mut impl Iterator<Item = char>) -> f64 {
    //let mut lookahead = chars.enumerate().find_map(|(i, c)| c);
    todo!()
}

fn get_elem_nums(chars: &mut impl Iterator<Item = char>, dim: usize) -> Vec<usize> {
    let mut num_elems = Vec::with_capacity(dim);

    for _ in 0..dim {
        chars.next();
        num_elems.push(read_usize(chars) as usize);
    }

    // 2-elements go before 1-elements, we're undoing that.
    if dim >= 3 {
        num_elems.swap(1, 2);
    }

    num_elems
}

pub fn polytope_from_off_src(mut src: String) -> Polytope {
    strip_comments(&mut src);
    homogenize_whitespace(&mut src);

    let mut chars = src.chars();
    let mut dim = read_usize(&mut chars);

    // This assumes our OFF file isn't 0D.
    if dim == 0 {
        dim = 3;
    }

    // Checks for our magic word.
    if [chars.next(), chars.next(), chars.next()] != [Some('O'), Some('F'), Some('F')] {
        panic!("ayo this file's not an OFF")
    }

    let num_elems = get_elem_nums(&mut chars, dim);

    // Reads all vertices.
    let num_verts = num_elems[0];
    let mut verts = Vec::with_capacity(num_verts);

    // Add each vertex to the vector.
    for _ in 0..num_verts {
        let mut vert = Vec::with_capacity(dim);

        for _ in 0..dim {
            chars.next();
            vert.push(read_f64(&mut chars));
        }

        verts.push(vert.into());
    }

    // Reads edges and faces.
    let mut elements = Vec::with_capacity(dim as usize);
    let (num_edges, num_faces) = (num_elems[1], num_elems[2]);
    let (mut edges, mut faces) = (Vec::with_capacity(num_edges), Vec::with_capacity(num_faces));
    let mut hash_edges = HashMap::new();

    // Add each face to the element list.
    for _ in 0..num_faces {
        chars.next();
        let face_sub_count = read_usize(&mut chars);

        let mut face = Vec::with_capacity(face_sub_count);
        let mut verts = Vec::with_capacity(face_sub_count);

        // Reads all vertices of the face.
        for _ in 0..face_sub_count {
            chars.next();
            verts.push(read_usize(&mut chars));
        }

        // Gets all edges of the face.
        for i in 0..face_sub_count {
            let edge = vec![verts[i], verts[(i + 1) % face_sub_count]]; // Missing sort!
            let edge_idx = hash_edges.get(&edge);

            match edge_idx {
                None => {
                    // Is the clone really necessary?
                    hash_edges.insert(edge.clone(), edges.len());
                    face.push(edges.len());
                    edges.push(edge);
                }
                Some(idx) => {
                    face.push(*idx);
                }
            }
        }

        faces.push(face);
    }

    // The number of edges in the file should match the number of read edges, though this isn't obligatory.
    if edges.len() != num_edges {
        println!("Edge count doesn't match expected edge count!");
    }

    elements.push(edges);
    elements.push(faces);

    // Adds all higher elements.
    for d in 3..(dim - 1) {
        let num_els = num_elems[d];
        let mut els = Vec::with_capacity(num_els);

        // Adds every d-element to the element list.
        for _ in 0..num_els {
            chars.next();
            let el_sub_count = read_usize(&mut chars);
            let mut el = Vec::with_capacity(el_sub_count);

            // Reads all sub-elements of the d-element.
            for _ in 0..el_sub_count {
                chars.next();
                el.push(read_usize(&mut chars));
            }

            els.push(el);
        }

        elements.push(els);
    }

    Polytope::new(verts, elements)
}

pub fn open_off(fp: &Path) -> Result<Polytope> {
    Ok(polytope_from_off_src(String::from_utf8(read(fp)?).unwrap()))
}
