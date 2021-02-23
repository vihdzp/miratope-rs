use gcd::Gcd;
use nalgebra::Dynamic;
use std::{collections::HashMap, f64::consts::PI as PI64};

use super::{Element, ElementList, Matrix, Point, Polytope};

fn rotations(angle: f64, num: usize, dim: usize) -> Vec<Matrix> {
    let mut rotations = Vec::with_capacity(num);
    let d = Dynamic::new(dim);
    let mut m = nalgebra::Matrix::identity_generic(d, d);
    let mut r = nalgebra::Matrix::identity_generic(d, d);

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

/// Applies a list of transformations to a polytope and creates a compound from
/// all of the copies of the polytope this generates.
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

/// Generates the unique 0D polytope.
pub fn point() -> Polytope {
    let vertices = vec![].into();
    let elements = vec![];

    Polytope::new(vertices, elements)
}

/// Generates a dyad, the unique non-compound 1D polytope.
pub fn dyad() -> Polytope {
    let vertices = vec![vec![-0.5].into(), vec![0.5].into()];
    let elements = vec![vec![vec![0, 1]]];

    Polytope::new(vertices, elements)
}

/// Generates a polygon with Schläfli symbol {n / d}.
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

/// Generates a regular tetrahedron with unit edge length.
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

/// Generates a cube with unit edge length.
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

/// Generates an octahedron with unit edge length.
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

/// Creates a uniform [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
/// with unit edge length.
pub fn antiprism(n: u32, d: u32) -> Polytope {
    let a = PI64 / (n as f64) * (d as f64);
    let c = 2.0 * a.cos();
    let h = ((1.0 + c) / (2.0 + c)).sqrt();

    if h.is_nan() {
        panic!("Uniform antiprism could not be built from these parameters.");
    }

    antiprism_with_height(n, d, h)
}

fn dual_vertices(vertices: &Vec<Point>, elements: &Vec<ElementList>, o: &Point) -> Vec<Point> {
    const EPS: f64 = 1e-9;

    let rank = elements.len();
    let facets = &elements[rank - 2];
    let ridges = &elements[rank - 3];

    let unique_subs = |els: &Vec<&Vec<usize>>| -> Element {
        let mut uniq = HashMap::new();

        for &el in els {
            for &sub in el {
                uniq.insert(sub, ());
            }
        }

        uniq.keys().map(|&u| u).collect()
    };

    facets
        .iter()
        .map(|f| {
            // Retrieves the vertices of each facet.
            let mut els = f.iter().map(|&el| &ridges[el]).collect();
            for d in (0..(rank - 3)).rev() {
                let uniq = unique_subs(&els);
                els = uniq.iter().map(|&el| &elements[d][el]).collect();
            }

            let el = unique_subs(&els);
            let h = el.iter().map(|&v| vertices[v].clone()).collect();
            let v = super::project(o, h);
            let s = v.norm_squared();

            if s < EPS {
                panic!("Facet passes through the dual center.")
            }

            v / s
        })
        .collect()
}

pub fn dual_with_center(p: &Polytope, o: &Point) -> Polytope {
    let el_counts = p.el_counts();

    let vertices = &p.vertices;
    let elements = &p.elements;

    let du_vertices = dual_vertices(vertices, elements, o);
    let mut du_elements = Vec::with_capacity(elements.len());

    // Builds the dual incidence graph.
    let mut elements = elements.iter().enumerate().rev();
    elements.next();

    for (d, els) in elements {
        let c = el_counts[d];
        let mut du_els = Vec::with_capacity(c);

        for _ in 0..c {
            du_els.push(vec![]);
        }

        for (i, el) in els.iter().enumerate() {
            for &sub in el {
                let du_el = &mut du_els[sub];
                du_el.push(i);
            }
        }

        du_elements.push(du_els);
    }

    Polytope::new_wo_comps(du_vertices, du_elements)
}

pub fn dual(p: &Polytope) -> Polytope {
    let dim = p.dimension();
    let mut o = Vec::with_capacity(dim);

    for _ in 0..dim {
        o.push(0.0);
    }

    dual_with_center(p, &o.into())
}

fn duoprism_vertices(p: &Vec<Point>, q: &Vec<Point>) -> Vec<Point> {
    let dimension = p[0].len() + q[0].len();
    let mut vertices = Vec::with_capacity(p.len() * q.len());

    for pv in p {
        for qv in q {
            let (pv, qv) = (pv.into_iter(), qv.into_iter());
            let mut v = Vec::with_capacity(dimension);

            for &c in pv {
                v.push(c);
            }
            for &c in qv {
                v.push(c);
            }

            vertices.push(v.into());
        }
    }

    vertices
}

/// Creates a [duoprism](https://polytope.miraheze.org/wiki/Duoprism)
/// from two given polytopes.
///
/// Duoprisms are usually defined in terms of Cartesian products, but this
/// definition only makes sense in the context of convex polytopes. For general
/// polytopes, a duoprism may be inductively built as a polytope whose facets
/// are the "prism products" of the elements of the first polytope times those
/// of the second, where the prism product of two points is simply the point
/// resulting from concatenating their coordinates.
pub fn duoprism(p: &Polytope, q: &Polytope) -> Polytope {
    let (p_rank, q_rank) = (p.rank(), q.rank());
    let (p_vertices, q_vertices) = (&p.vertices, &q.vertices);
    let (p_elements, q_elements) = (&p.elements, &q.elements);
    let (p_el_counts, q_el_counts) = (p.el_counts(), q.el_counts());

    let rank = p_rank + q_rank;

    if p_rank == 0 {
        return q.clone();
    }
    if q_rank == 0 {
        return p.clone();
    }

    let vertices = duoprism_vertices(&p_vertices, &q_vertices);
    let mut elements = Vec::with_capacity(rank);
    for _ in 0..rank {
        elements.push(Vec::new());
    }

    // The elements of a given rank are added in order vertex × facet, edge ×
    // ridge, ...
    //
    // el_counts[m][n] will memoize the number of elements of rank m generated
    // by these products up to those of type n-element × (m - n)-element.
    let mut el_counts = Vec::with_capacity(rank);
    for m in 0..(p_rank + 1) {
        el_counts.push(Vec::new());

        for n in 0..(q_rank + 1) {
            if m == 0 || n == q_rank {
                el_counts[m].push(0);
            } else {
                let idx = el_counts[m - 1][n + 1] + p_el_counts[m - 1] * q_el_counts[n + 1];
                el_counts[m].push(idx);
            }
        }
    }

    // Gets the index of the prism product of the i-th m-element times the j-th
    // n-element.
    let get_idx = |m: usize, i: usize, n: usize, j: usize| -> usize {
        let offset = i * q_el_counts[n] + j;

        el_counts[m][n] + offset
    };

    // For each of the element lists of p (including vertices):
    for m in 0..(p_rank + 1) {
        // For each of the element lists of q (including vertices):
        for n in 0..(q_rank + 1) {
            // We'll multiply the m-elements times the n-elements inside of this loop.

            // We already took care of vertices.
            if m == 0 && n == 0 {
                continue;
            }

            // For each m-element:
            for i in 0..p_el_counts[m] {
                // For each n-element:
                for j in 0..q_el_counts[n] {
                    let mut els = Vec::new();

                    // The prism product of the i-th m-element A and the j-th n-element B
                    // has the products of A with the facets of B and B with the facets of
                    // A as facets.

                    // Points don't have facets.
                    if m != 0 {
                        let p_els = &p_elements[m - 1];
                        let p_el = &p_els[i];

                        for &p_sub in p_el {
                            els.push(get_idx(m - 1, p_sub, n, j));
                        }
                    }

                    // Points don't have facets.
                    if n != 0 {
                        let q_els = &q_elements[n - 1];
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

pub fn prism_with_height(p: &Polytope, h: f64) -> Polytope {
    let mut dyad = dyad();
    dyad.scale(h);

    duoprism(p, &dyad)
}

pub fn prism(p: &Polytope) -> Polytope {
    prism_with_height(p, 1.0)
}

pub fn multiprism(polytopes: &Vec<&Polytope>) -> Polytope {
    let mut r = point();

    for &p in polytopes {
        r = duoprism(&p, &r);
    }

    r
}

fn pyramid_vertices(p: &Vec<Point>, q: &Vec<Point>, h: f64) -> Vec<Point> {
    let (p_dimension, q_dimension) = (
        match p.get(0) {
            Some(v) => v.len(),
            None => 0,
        },
        match q.get(0) {
            Some(v) => v.len(),
            None => 0,
        },
    );

    let tegum = h == 0.0;
    let dimension = p_dimension + q_dimension + tegum as usize;

    let mut vertices = Vec::with_capacity(p.len() + q.len());

    for vq in q {
        let mut v = Vec::with_capacity(dimension);
        let pad = p_dimension;

        for _ in 0..pad {
            v.push(0.0);
        }
        for &c in vq.iter() {
            v.push(c);
        }
        if !tegum {
            v.push(h / 2.0);
        }

        vertices.push(v.into());
    }
    for vp in p {
        let mut v = Vec::with_capacity(dimension);
        let pad = q_dimension;

        for &c in vp.iter() {
            v.push(c);
        }
        for _ in 0..pad {
            v.push(0.0);
        }
        if !tegum {
            v.push(-h / 2.0);
        }

        vertices.push(v.into());
    }

    vertices
}

pub fn duopyramid_with_height(p: &Polytope, q: &Polytope, h: f64) -> Polytope {
    let (p_rank, q_rank) = (p.rank(), q.rank());
    let (p_vertices, q_vertices) = (&p.vertices, &q.vertices);
    let (p_elements, q_elements) = (&p.elements, &q.elements);
    let (p_el_counts, q_el_counts) = (p.el_counts(), q.el_counts());

    let tegum = h == 0.0;
    let rank = p_rank + q_rank + !tegum as usize;

    let (m_max, n_max) = (p_rank + !tegum as usize + 1, q_rank + !tegum as usize + 1);

    if p_rank == 0 && tegum {
        return q.clone();
    }
    if q_rank == 0 && tegum {
        return p.clone();
    }

    let vertices = pyramid_vertices(&p_vertices, &q_vertices, h);
    let mut elements = Vec::with_capacity(rank);
    for _ in 0..rank {
        elements.push(Vec::new());
    }

    // The elements of a given rank are added in order nullitope × facet, vertex
    // × ridge, ...
    //
    // el_counts[m][n] will memoize the number of elements of rank m - 1
    // generated by these products up to those of type (n - 1)-element ×
    // (m - n)-element.
    let mut el_counts = Vec::with_capacity(rank);
    for m in 0..m_max {
        el_counts.push(Vec::new());

        for n in 0..n_max {
            if m == 0 || n == n_max - 1 {
                el_counts[m].push(0);
            } else {
                let p_el_count = if m == 1 { 1 } else { p_el_counts[m - 2] };
                let idx = el_counts[m - 1][n + 1] + p_el_count * q_el_counts[n];
                el_counts[m].push(idx);
            }
        }
    }

    // Gets the index of the prism product of the i-th m-element times the j-th
    // n-element.
    let get_idx = |m: usize, i: usize, n: usize, j: usize| -> usize {
        let q_el_counts_n = if n == 0 { 1 } else { q_el_counts[n - 1] };
        let offset = i * q_el_counts_n + j;

        el_counts[m][n] + offset
    };

    // For each of the element lists of p (including vertices & the nullitope):
    for m in 0..m_max {
        let p_el_counts_m = if m == 0 { 1 } else { p_el_counts[m - 1] };

        // For each of the element lists of q (including vertices & the nullitope):
        for n in 0..n_max {
            let q_el_counts_n = if n == 0 { 1 } else { q_el_counts[n - 1] };

            // We'll multiply the (m - 1)-elements with the (n - 1)-elements inside of this loop.

            // We already took care of vertices.
            if m + n < 2 {
                continue;
            }

            // For each m-element:
            for i in 0..p_el_counts_m {
                // For each n-element:
                for j in 0..q_el_counts_n {
                    let mut els = Vec::new();

                    // The prism product of the i-th m-element A and the j-th n-element B
                    // has the products of A with the facets of B and B with the facets of
                    // A as facets.

                    // Nullitopes don't have facets.
                    if m != 0 {
                        if m > 1 {
                            let p_els = &p_elements[m - 2];
                            let p_el = &p_els[i];

                            for &p_sub in p_el {
                                els.push(get_idx(m - 1, p_sub, n, j));
                            }
                        }
                        // Dealing with a vertex
                        else {
                            els.push(get_idx(m - 1, 0, n, j));
                        }
                    }

                    // Nullitopes don't have facets.
                    if n != 0 {
                        if n > 1 {
                            let q_els = &q_elements[n - 2];
                            let q_el = &q_els[j];

                            for &q_sub in q_el {
                                els.push(get_idx(m, i, n - 1, q_sub));
                            }
                        }
                        // Dealing with a vertex
                        else {
                            els.push(get_idx(m, i, n - 1, 0));
                        }
                    }

                    elements[m + n - 2].push(els);
                }
            }
        }
    }

    if tegum {
        // We take special care of the components.
        // These are simply the pyramid products of the two polytopes' facets.
        // For each m-element:
        let (m, n) = (p_rank + 1, q_rank + 1);
        let (p_el_counts_m, q_el_counts_n) = (p_el_counts[m - 1], q_el_counts[n - 1]);

        // For each component of p:
        for i in 0..p_el_counts_m {
            // For each component of q:
            for j in 0..q_el_counts_n {
                let mut els = Vec::new();

                // The prism product of the i-th m-element A and the j-th n-element B
                // has the products of A with the facets of B and B with the facets of
                // A as facets.

                let (p_els, q_els) = (&p_elements[m - 2], &q_elements[n - 2]);
                let (p_el, q_el) = (&p_els[i], &q_els[j]);

                for &p_sub in p_el {
                    for &q_sub in q_el {
                        els.push(get_idx(m - 1, p_sub, n - 1, q_sub));
                    }
                }

                elements[m + n - 3].push(els);
            }
        }
    }

    Polytope::new(vertices, elements)
}

pub fn duotegum(p: &Polytope, q: &Polytope) -> Polytope {
    duopyramid_with_height(p, q, 0.0)
}

pub fn tegum_with_height(p: &Polytope, h: f64) -> Polytope {
    let mut dyad = dyad();
    dyad.scale(h);

    duotegum(p, &dyad)
}

pub fn tegum(p: &Polytope) -> Polytope {
    tegum_with_height(p, 1.0)
}

pub fn multitegum(polytopes: &Vec<&Polytope>) -> Polytope {
    let mut r = point();

    for p in polytopes {
        r = duotegum(&p, &r);
    }

    r
}

pub fn duopyramid(p: &Polytope, q: &Polytope) -> Polytope {
    duopyramid_with_height(p, q, 1.0)
}

pub fn pyramid_with_height(p: &Polytope, h: f64) -> Polytope {
    let mut dyad = dyad();
    dyad.scale(h);

    duopyramid(p, &dyad)
}

pub fn pyramid(p: &Polytope) -> Polytope {
    pyramid_with_height(p, 1.0)
}

pub fn multipyramid_with_height(polytopes: &Vec<&Polytope>, h: f64) -> Polytope {
    let mut polytopes = polytopes.iter();
    let mut r = (*polytopes.next().unwrap()).clone();

    for p in polytopes {
        r = duopyramid_with_height(&p, &r, h);
    }

    r
}

pub fn multipyramid(polytopes: &Vec<&Polytope>) -> Polytope {
    multipyramid_with_height(polytopes, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Checks the element counts of a few polygons.
    fn polygon_counts() {
        assert_eq!(polygon(5, 1).el_counts(), vec![5, 5, 1]);
        assert_eq!(polygon(7, 2).el_counts(), vec![7, 7, 1]);
        assert_eq!(polygon(6, 2).el_counts(), vec![6, 6, 2])
    }

    #[test]
    /// Checks the element count of a tetrahedron.
    fn tet_counts() {
        assert_eq!(tet().el_counts(), vec![4, 6, 4, 1])
    }

    #[test]
    /// Checks the element count of a cube.
    fn cube_counts() {
        assert_eq!(cube().el_counts(), vec![8, 12, 6, 1])
    }

    #[test]
    /// Checks the element count of an octahedron.
    fn oct_counts() {
        assert_eq!(oct().el_counts(), vec![6, 12, 8, 1])
    }

    #[test]
    /// Checks the element counts of a few antiprisms.
    fn antiprism_counts() {
        assert_eq!(antiprism(5, 1).el_counts(), vec![10, 20, 12, 1]);
        assert_eq!(antiprism(7, 2).el_counts(), vec![14, 28, 16, 1]);
        assert_eq!(antiprism(6, 2).el_counts(), vec![12, 24, 16, 2])
    }

    #[test]
    /// Checks the element count of a cube dual (octahedron).
    fn cube_dual_counts() {
        let cube_dual = dual(&cube());

        assert_eq!(cube_dual.el_counts(), vec![6, 12, 8, 1])
    }

    #[test]
    /// Checks the element count of a triangular-pentagonal duoprism.
    fn trapedip_counts() {
        let trig = polygon(3, 1);
        let peg = polygon(5, 1);
        let trapedip = duoprism(&trig, &peg);

        assert_eq!(trapedip.el_counts(), vec![15, 30, 23, 8, 1])
    }

    #[test]
    /// Checks the element count of a triangular trioprism.
    fn trittip_counts() {
        let trig = polygon(3, 1);
        let trittip = multiprism(&vec![&trig; 3]);

        assert_eq!(trittip.el_counts(), vec![27, 81, 108, 81, 36, 9, 1])
    }

    #[test]
    /// Checks the element count of a triangular-pentagonal duotegum.
    fn trapedit_counts() {
        let trig = polygon(3, 1);
        let peg = polygon(5, 1);
        let trapedit = duotegum(&trig, &peg);

        assert_eq!(trapedit.el_counts(), vec![8, 23, 30, 15, 1])
    }

    #[test]
    /// Checks the element count of a triangular triotegum.
    fn trittit_counts() {
        let trig = polygon(3, 1);
        let trittit = multitegum(&vec![&trig; 3]);

        assert_eq!(trittit.el_counts(), vec![9, 36, 81, 108, 81, 27, 1])
    }

    #[test]
    /// Checks the element count of a triangular-pentagonal duopyramid.
    fn trapdupy_counts() {
        let trig = polygon(3, 1);
        let peg = polygon(5, 1);
        let trapdupy = duopyramid(&trig, &peg);

        assert_eq!(trapdupy.el_counts(), vec![8, 23, 32, 23, 8, 1])
    }

    #[test]
    /// Checks the element count of a triangular triopyramid.
    fn tritippy_counts() {
        let trig = polygon(3, 1);
        let tritippy = multipyramid(&vec![&trig; 3]);

        assert_eq!(
            tritippy.el_counts(),
            vec![9, 36, 84, 126, 126, 84, 36, 9, 1]
        )
    }
}
