use gcd::Gcd;
use nalgebra::*;
use std::f64::consts::PI as PI64;

use super::Polytope;

type Matrix = nalgebra::DMatrix<f64>;

fn rotations(angle: f64, num: usize, dim: usize) -> Vec<Matrix> {
    let mut rotations = Vec::with_capacity(num);
    let d = Dynamic::new(dim);
    let mut m = Matrix::identity_generic(d, d);
    let mut r = Matrix::identity_generic(d, d);

    let (s, c) = angle.sin_cos();
    r[(0, 0)] = c;
    r[(1, 0)] = s;
    r[(0, 1)] = -s;
    r[(1, 1)] = c;

    for _ in 0..num {
        rotations.push(m.to_homogeneous());
        m *= &r;
    }

    rotations
}

pub fn compound(p: Polytope, trans: Vec<Matrix>) -> Polytope {
    let comps = trans.len();
    let el_counts = p.el_counts();
    // the vertices, turned into homogeneous points
    let vertices = p
        .vertices
        .into_iter()
        .map(|v| v.push(1.0))
        .collect::<Vec<_>>();
    let vertices = trans
        .into_iter()
        .flat_map(|m| vertices.iter().map(move |v| m.clone() * v))
        .map(|v| {
            // remove the extra homogeneous coordinate
            let row = v.nrows() - 1;
            v.remove_row(row)
        })
        .collect();
    let mut elements = Vec::with_capacity(p.elements.len());

    for (d, els) in p.elements.iter().enumerate() {
        let sub_count = el_counts[d];
        let el_count = el_counts[d + 1];
        let mut new_els = Vec::with_capacity(el_count * comps);

        for comp in 0..comps {
            let offset = comp * sub_count;

            for el in els.iter() {
                new_els.push(el.iter().map(|i| i + offset).collect())
            }
        }

        elements.push(new_els);
    }

    Polytope::new(vertices, elements)
}

pub fn point() -> Polytope {
    let vertices = vec![].into();
    let elements = vec![];

    Polytope::new(vertices, elements)
}

pub fn dyad() -> Polytope {
    let vertices = vec![vec![-0.5].into(), vec![0.5].into()];
    let elements = vec![vec![vec![0, 1]]];

    Polytope::new(vertices, elements)
}

pub fn polygon(n: u32, d: u32) -> Polytope {
    let mut n = n as usize;
    let g = n.gcd(d as usize);
    let a = 2.0 * PI64 / (n as f64) * (d as f64);
    n /= g;
    let s = (a / 2.0).sin() * 2.0;

    let mut vertices = Vec::with_capacity(n);
    let mut edges = Vec::with_capacity(n);
    let mut components = vec![Vec::with_capacity(g)];

    for k in 0..n {
        let ka = (k as f64) * a;
        vertices.push(vec![ka.cos() / s, ka.sin() / s].into());
        edges.push(vec![k, (k + 1) % n]);
        components[0].push(k);
    }

    compound(
        Polytope::new(vertices, vec![edges, components]),
        rotations(a / (g as f64), g, 2),
    )
}

pub fn tet() -> Polytope {
    let x = 2.0_f64.sqrt() / 4.0;

    let vertices = vec![
        vec![x, x, x].into(),
        vec![-x, -x, x].into(),
        vec![-x, x, -x].into(),
        vec![x, -x, -x].into(),
    ];
    let edges = vec![
        vec![0, 1],
        vec![0, 2],
        vec![0, 3],
        vec![1, 2],
        vec![1, 3],
        vec![2, 3],
    ];
    let faces = vec![vec![0, 1, 3], vec![0, 2, 4], vec![1, 2, 5], vec![3, 4, 5]];
    let components = vec![vec![0, 1, 2, 3]];

    Polytope::new(vertices, vec![edges, faces, components])
}

pub fn cube() -> Polytope {
    let x = 0.5;

    let vertices = vec![
        vec![x, x, x].into(),
        vec![x, x, -x].into(),
        vec![x, -x, -x].into(),
        vec![x, -x, x].into(),
        vec![-x, x, x].into(),
        vec![-x, x, -x].into(),
        vec![-x, -x, -x].into(),
        vec![-x, -x, x].into(),
    ];
    let edges = vec![
        vec![0, 1],
        vec![1, 2],
        vec![2, 3],
        vec![3, 0],
        vec![4, 5],
        vec![5, 6],
        vec![6, 7],
        vec![7, 4],
        vec![0, 4],
        vec![1, 5],
        vec![2, 6],
        vec![3, 7],
    ];
    let faces = vec![
        vec![0, 1, 2, 3],
        vec![4, 5, 6, 7],
        vec![0, 4, 8, 9],
        vec![1, 5, 9, 10],
        vec![2, 6, 10, 11],
        vec![3, 7, 11, 8],
    ];
    let components = vec![vec![0, 1, 2, 3, 4, 5]];

    Polytope::new(vertices, vec![edges, faces, components])
}

pub fn oct() -> Polytope {
    let x = 1.0 / 2.0_f64.sqrt();

    let vertices = vec![
        vec![x, 0.0, 0.0].into(),
        vec![-x, 0.0, 0.0].into(),
        vec![0.0, x, 0.0].into(),
        vec![0.0, 0.0, x].into(),
        vec![0.0, -x, 0.0].into(),
        vec![0.0, 0.0, -x].into(),
    ];
    let edges = vec![
        vec![0, 2],
        vec![0, 3],
        vec![0, 4],
        vec![0, 5],
        vec![1, 2],
        vec![1, 3],
        vec![1, 4],
        vec![1, 5],
        vec![2, 3],
        vec![3, 4],
        vec![4, 5],
        vec![5, 2],
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
    let components = vec![vec![0, 1, 2, 3, 4, 5, 6, 7]];

    Polytope::new(vertices, vec![edges, faces, components])
}

/// Creates an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
/// with unit base edge length and a given height.
pub fn antiprism_with_height(n: u32, d: u32, h: f64) -> Polytope {
    let mut n = n as usize;
    let g = n.gcd(d as usize);
    let a = PI64 / (n as f64) * (d as f64);
    n /= g;
    let s = a.sin() * 2.0;
    let mut h = h / 2.0;

    let mut vertices = Vec::with_capacity(2 * n);
    let mut edges = Vec::with_capacity(4 * n);
    let mut faces = Vec::with_capacity(2 * n + 2);
    let mut components = vec![Vec::with_capacity(2 * n + 2)];

    for k in 0..(2 * n) {
        // Generates vertices.
        let ka = (k as f64) * a;
        vertices.push(vec![ka.cos() / s, ka.sin() / s, h].into());
        h *= -1.0;

        // Generates edges.
        edges.push(vec![k, (k + 1) % (2 * n)]);
        edges.push(vec![k, (k + 2) % (2 * n)]);

        // Generates faces.
        faces.push(vec![2 * k, 2 * k + 1, (2 * k + 2) % (4 * n)]);

        // Generates component.
        components[0].push(k);
    }

    let (mut base1, mut base2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for k in 0..n {
        base1.push(4 * k + 1);
        base2.push(4 * k + 3);
    }
    faces.push(base1);
    faces.push(base2);

    components[0].push(2 * n);
    components[0].push(2 * n + 1);

    // Compounds of antiprisms with antiprismatic symmetry must be handled
    // differently than compounds of antiprisms with prismatic symmetry.
    let d = d as usize;
    let a = if d / g % 2 == 0 { a } else { a * 2.0 };
    compound(
        Polytope::new(vertices, vec![edges, faces, components]),
        rotations(a / (g as f64), g, 3),
    )
}

pub fn antiprism(n: u32, d: u32) -> Polytope {
    let a = PI64 / (n as f64) * (d as f64);
    let c = 2.0 * a.cos();
    let h = ((1.0 + c) / (2.0 + c)).sqrt();
    if h.is_nan() {
        panic!("Uniform antiprism could not be built from these parameters.");
    }

    antiprism_with_height(n, d, h)
}

pub fn duoprism(p: &Polytope, q: &Polytope) -> Polytope {
    let (p_rank, q_rank) = (p.rank(), q.rank());
    let rank = p_rank + q_rank;
    let (p_elements, q_elements) = (&p.elements, &q.elements);
    let (p_vertices, q_vertices) = (&p.vertices, &q.vertices);
    let (p_el_counts, q_el_counts) = (p.el_counts(), q.el_counts());

    if p_rank == 0 {
        return q.clone();
    }
    if q_rank == 0 {
        return p.clone();
    }

    let mut vertices = Vec::with_capacity(p_vertices.len() * q_vertices.len());
    let mut elements = Vec::with_capacity(rank);

    // The elements of a given rank are added in order vertex × facet, edge ×
    // ridge, ...
    //
    // el_counts[m][n] will memoize the number of elements of rank m generated
    // by these products up and until those of type n-element × (d - n)-element.
    let mut el_counts = Vec::with_capacity(rank);

    for _ in 0..rank {
        elements.push(Vec::new());
    }

    let mut get_idx = |m: usize, i: usize, n: usize, j: usize| -> usize {
        let offset = i * q_el_counts[n] + j;

        if el_counts.len() <= m {
            el_counts.push(Vec::new());
        }

        if el_counts[m].len() <= n {
            if m == 0 || n == q_rank {
                el_counts[m].push(0);
            } else {
                let idx = el_counts[m - 1][n + 1] + p_el_counts[m - 1] * q_el_counts[n + 1];
                el_counts[m].push(idx);
            }
        }

        el_counts[m][n] + offset
    };

    // Adds vertices.
    for pv in p_vertices {
        for qv in q_vertices {
            let (pv, qv) = (pv.into_iter(), qv.into_iter());
            let mut v = Vec::with_capacity(rank);

            for &c in pv {
                v.push(c);
            }
            for &c in qv {
                v.push(c);
            }

            vertices.push(v.into());
        }
    }

    for m in 0..(p_rank + 1) {
        let p_els = if m > 0 {
            Some(&p_elements[m - 1])
        } else {
            None
        };

        for n in 0..(q_rank + 1) {
            // We already took care of vertices.
            if m == 0 && n == 0 {
                continue;
            }

            let q_els = if n > 0 {
                Some(&q_elements[n - 1])
            } else {
                None
            };

            for i in 0..p_el_counts[m] {
                for j in 0..q_el_counts[n] {
                    let mut els = Vec::new();

                    if m != 0 {
                        let p_els = p_els.unwrap();
                        let p_el = &p_els[i];

                        for &p_sub in p_el {
                            els.push(get_idx(m - 1, p_sub, n, j));
                        }
                    }
                    if n != 0 {
                        let q_els = q_els.unwrap();
                        let q_el = &q_els[j];

                        for &q_sub in q_el {
                            els.push(get_idx(m, i, n - 1, q_sub));
                        }
                    }

                    elements[m + n - 1].push(els);
                }
            }
        }
    }

    Polytope::new(vertices, elements)
}

pub fn multiprism(polytopes: Vec<&Polytope>) -> Polytope {
    let mut r = point();

    for p in polytopes {
        r = duoprism(&p, &r);
    }

    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_counts() {
        assert_eq!(polygon(5, 1).el_counts(), vec![5, 5, 1]);
        assert_eq!(polygon(7, 2).el_counts(), vec![7, 7, 1]);
        assert_eq!(polygon(6, 2).el_counts(), vec![6, 6, 2])
    }

    #[test]
    fn tet_counts() {
        assert_eq!(tet().el_counts(), vec![4, 6, 4, 1])
    }

    #[test]
    fn cube_counts() {
        assert_eq!(cube().el_counts(), vec![8, 12, 6, 1])
    }

    #[test]
    fn oct_counts() {
        assert_eq!(oct().el_counts(), vec![6, 12, 8, 1])
    }

    #[test]
    fn antiprism_counts() {
        assert_eq!(antiprism(5, 1).el_counts(), vec![10, 20, 12, 1]);
        assert_eq!(antiprism(7, 2).el_counts(), vec![14, 28, 16, 1]);

        // We aren't implementing compound antiprisms yet.
        // assert_eq!(antiprism(6, 2).el_counts(), vec![12, 24, 16, 2])
    }

    #[test]
    fn petet_counts() {
        let peg = polygon(5, 1);
        let tet = tet();
        let petet = duoprism(&peg, &tet);

        assert_eq!(petet.el_counts(), vec![20, 50, 54, 31, 9, 1])
    }

    #[test]
    fn trittip_counts() {
        let trig = polygon(3, 1);
        let trittip = multiprism(vec![&trig; 3]);

        assert_eq!(trittip.el_counts(), vec![27, 81, 108, 81, 36, 9, 1])
    }
}
