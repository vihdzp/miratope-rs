use std::collections::HashMap;

use nalgebra::Dynamic;

use super::super::{Element, ElementList, Matrix, Point, Polytope};
use super::*;

/// Generates matrices for rotations by the first `num` multiples of `angle`
/// through the xy plane.
pub fn rotations(angle: f64, num: usize, dim: usize) -> Vec<Matrix> {
    let mut rotations = Vec::with_capacity(num);
    let dim = Dynamic::new(dim);
    let mut matrix = nalgebra::Matrix::identity_generic(dim, dim);
    let mut matrices = nalgebra::Matrix::identity_generic(dim, dim);

    // The first rotation matrix.
    let (s, c) = angle.sin_cos();
    matrices[(0, 0)] = c;
    matrices[(1, 0)] = s;
    matrices[(0, 1)] = -s;
    matrices[(1, 1)] = c;

    // Generates the other rotation matrices from r.
    for _ in 0..num {
        rotations.push(matrix.clone());
        matrix *= &matrices;
    }

    rotations
}

pub fn central_inv(dim: usize) -> Vec<Matrix> {
    let dim = Dynamic::new(dim);
    let matrix = nalgebra::Matrix::identity_generic(dim, dim);

    vec![matrix.clone(), -matrix]
}

pub fn compound(polytopes: &[&Polytope]) -> Polytope {
    let mut polytopes = polytopes.iter();
    let p = polytopes.next().unwrap();
    let rank = p.rank();
    let mut vertices = p.vertices.clone();
    let mut comp_elements = p.elements.clone();

    for &p in polytopes {
        let mut el_nums = vec![vertices.len()];
        for comp_els in &comp_elements {
            el_nums.push(comp_els.len());
        }

        vertices.append(&mut p.vertices.clone());

        for i in 0..rank {
            let comp_els = &mut comp_elements[i];
            let els = &p.elements[i];
            let offset = el_nums[i];

            for el in els {
                let mut comp_el = Vec::with_capacity(el.len());

                for &sub in el {
                    comp_el.push(sub + offset);
                }

                comp_els.push(comp_el);
            }
        }
    }

    Polytope::new(vertices, comp_elements)
}

/// Applies a list of transformations to a polytope and creates a compound from
/// all of the copies of the polytope this generates.
pub fn compound_from_trans(p: &Polytope, trans: Vec<Matrix>) -> Polytope {
    let mut polytopes = Vec::with_capacity(trans.len());

    for m in &trans {
        polytopes.push(p.clone().apply(&m));
    }

    compound(&polytopes.iter().collect::<Vec<_>>())
}

pub fn dual_compound(p: &Polytope) -> Polytope {
    compound(&[p, &p.dual()])
}

/// Projects a [`Point`] onto the hyperplane defined by a vector of [`Points`][`Point`].
pub fn project(p: &Point, h: Vec<Point>) -> Point {
    const EPS: f64 = 1e-9;

    let mut h = h.iter();
    let o = h.next().unwrap();
    let mut basis: Vec<Point> = Vec::new();

    for q in h {
        let mut q = q - o;

        for b in &basis {
            q -= b * (q.dot(&b)) / b.norm_squared();
        }

        if q.norm() > EPS {
            basis.push(q);
        }
    }

    let mut p = p - o;

    for b in &basis {
        p -= b * (p.dot(&b)) / b.norm_squared();
    }

    -p
}

/// Builds the vertices of a dual polytope from its facets.
fn dual_vertices(vertices: &[Point], elements: &[ElementList], o: &Point) -> Vec<Point> {
    const EPS: f64 = 1e-9;

    let rank = elements.len();

    // Gets the unique sub-elements from a list of elements.
    let unique_subs = |els: &Vec<&Vec<usize>>| -> Element {
        let mut uniq = HashMap::new();

        for &el in els {
            for &sub in el {
                uniq.insert(sub, ());
            }
        }

        uniq.keys().cloned().collect()
    };

    // We find the indices of the vertices on the facet.
    let projections: Vec<Point>;

    if rank >= 2 {
        let facets = &elements[rank - 2];

        projections = facets
            .iter()
            .map(|f| {
                // We repeatedly retrieve the next subelements of the facets until we get to the vertices.
                let facet_verts;

                if rank >= 3 {
                    let ridges = &elements[rank - 3];
                    let mut els = f.iter().map(|&el| &ridges[el]).collect();

                    for d in (0..(rank - 3)).rev() {
                        let uniq = unique_subs(&els);
                        els = uniq.iter().map(|&el| &elements[d][el]).collect();
                    }

                    facet_verts = unique_subs(&els);
                }
                // If our polytope is 2D, we already know the vertices of the facets.
                else {
                    facet_verts = f.clone();
                }

                // We project the dual center onto the hyperplane defined by the vertices.
                let h = facet_verts.iter().map(|&v| vertices[v].clone()).collect();

                project(o, h)
            })
            .collect()
    }
    // If our polytope is 1D, the vertices themselves are the facets.
    else {
        projections = vertices.into();
    }

    // Reciprocates the projected points.
    projections
        .iter()
        .map(|v| {
            let v = v - o;
            let s = v.norm_squared();

            // We avoid division by 0.
            if s < EPS {
                panic!("Facet passes through the dual center.")
            }

            v / s + o
        })
        .collect()
}

impl Polytope {
    /// Scales a polytope by a given factor.
    pub fn scale(mut self, k: f64) -> Self {
        for v in &mut self.vertices {
            *v *= k;
        }

        self
    }

    pub fn shift(mut self, o: Point) -> Self {
        for v in &mut self.vertices {
            *v -= o.clone();
        }

        self
    }

    pub fn recenter(self) -> Self {
        let gravicenter = self.gravicenter();

        self.shift(gravicenter)
    }

    pub fn apply(mut self, m: &Matrix) -> Self {
        for v in &mut self.vertices {
            *v = m.clone() * v.clone();
        }

        self
    }

    /// Builds the dual polytope of `p`. Uses `o` as the center for reciprocation.
    pub fn dual_with_center(&self, o: &Point) -> Polytope {
        let rank = self.rank();

        // If we're dealing with a point, let's skip all of the bs:
        if rank == 0 {
            return point();
        }

        let el_nums = self.el_nums();

        let vertices = &self.vertices;
        let elements = &self.elements;

        let du_vertices = dual_vertices(vertices, elements, o);
        let mut du_elements = Vec::with_capacity(elements.len());

        // Builds the dual incidence graph.
        let mut elements = elements.iter().enumerate().rev();
        elements.next();

        for (d, els) in elements {
            let c = el_nums[d];
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

        if rank >= 2 {
            Polytope::new_wo_comps(du_vertices, du_elements)
        } else {
            let components = self.elements[0].clone();
            du_elements.push(components);

            Polytope::new(du_vertices, du_elements)
        }
    }

    /// Builds the dual polytope of `p`. Uses the origin as the center for reciprocation.
    pub fn dual(&self) -> Polytope {
        self.dual_with_center(&vec![0.0; self.dimension()].into())
    }
}
