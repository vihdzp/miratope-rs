use std::f64::consts::PI;
use std::{f64, usize};
use ultraviolet::DVec3;

use super::PolytopeC;

pub fn tet() -> PolytopeC {
    let x = 2f64.sqrt() / 4.0;

    let vertices = vec![
        DVec3::new(x, x, x),
        DVec3::new(-x, -x, x),
        DVec3::new(x, -x, -x),
        DVec3::new(-x, x, -x),
    ];
    let edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let faces = vec![vec![0, 1, 3], vec![0, 2, 4], vec![1, 2, 5], vec![3, 4, 5]];

    PolytopeC::new(vertices, edges, faces)
}

pub fn cube() -> PolytopeC {
    let x = 0.5;

    let vertices = vec![
        DVec3::new(x, x, x),
        DVec3::new(x, x, -x),
        DVec3::new(x, -x, -x),
        DVec3::new(x, -x, x),
        DVec3::new(-x, x, x),
        DVec3::new(-x, x, -x),
        DVec3::new(-x, -x, -x),
        DVec3::new(-x, -x, x),
    ];
    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 3),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let faces = vec![
        vec![0, 1, 2, 3],
        vec![4, 5, 6, 7],
        vec![0, 4, 8, 9],
        vec![1, 5, 9, 10],
        vec![2, 6, 10, 11],
        vec![3, 7, 11, 8],
    ];

    PolytopeC::new(vertices, edges, faces)
}

pub fn oct() -> PolytopeC {
    let x = 1.0 / 2f64.sqrt();

    let vertices = vec![
        DVec3::new(x, 0.0, 0.0),
        DVec3::new(-x, 0.0, 0.0),
        DVec3::new(0.0, x, 0.0),
        DVec3::new(0.0, 0.0, x),
        DVec3::new(0.0, -x, 0.0),
        DVec3::new(0.0, 0.0, -x),
    ];
    let edges = vec![
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 2),
    ];
    let faces = vec![
        vec![0, 1, 8],
        vec![4, 5, 8],
        vec![1, 2, 9],
        vec![5, 6, 9],
        vec![2, 3, 10],
        vec![6, 7, 10],
        vec![3, 0, 11],
        vec![7, 4, 11],
    ];

    PolytopeC::new(vertices, edges, faces)
}

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
