use std::f64::consts::PI;
use std::{f64, usize};
use ultraviolet::DVec3;

use super::PolytopeC;

/// Creates an [[https://polytope.miraheze.org/wiki/Antiprism | antiprism]] with unit edge length and a given height.
pub fn antiprism_with_height(n: u32, d: u32, mut h: f64) -> PolytopeC {
    let n = n as usize;
    let a = PI / (n as f64) * (d as f64);
    let s = a.sin() * 2.0;

    let mut vertices = Vec::with_capacity(2 * n);
    let mut edges = Vec::with_capacity(4 * n);
    let mut faces = Vec::with_capacity(2 * n + 2);

    for k in 0..(2 * n) {
        // Generates vertices.
        let ka = (k as f64) * a;
        vertices.push(DVec3::new((ka).cos() / s, (ka).sin() / s, h));
        h *= -1.0;

        // Generates edges.
        edges.push((k, (k + 1) % (2 * n)));
        edges.push((k, (k + 2) % (2 * n)));

        // Generates faces.
        faces.push(vec![2 * k, 2 * k + 1, (2 * k + 2) % (4 * n)]);
    }

    let (mut base1, mut base2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for k in 0..n {
        base1.push(4 * k + 1);
        base2.push(4 * k + 3);
    }
    faces.push(base1);
    faces.push(base2);

    PolytopeC::new(vertices, edges, faces)
}

pub fn antiprism(n: u32, d: u32) -> PolytopeC {
    let a = PI / (n as f64);
    let h = (a.cos() - (2.0 * a).cos()).sqrt() / (4.0 * a.sin());

    antiprism_with_height(n, d, h)
}
