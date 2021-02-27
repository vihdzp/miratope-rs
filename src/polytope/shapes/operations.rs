//! Contains various methods that can be applied to specific polytopes.

use std::collections::VecDeque;

use nalgebra::Dynamic;

use super::super::{Matrix, Point, Polytope};
use super::*;

pub struct Hypersphere {
    center: Point,
    radius: f64,
}

impl Hypersphere {
    /// Represents the unit hypersphere in a certain number of dimensions.
    pub fn unit(dim: usize) -> Hypersphere {
        Hypersphere {
            center: vec![0.0; dim].into(),
            radius: 1.0,
        }
    }
}

pub struct Hyperplane {
    pub points: Vec<Point>,
}

impl Hyperplane {
    /// Projects a [`Point`] onto the hyperplane defined by a slice of [`Points`][`Point`].
    pub fn project(&self, p: &Point) -> Point {
        const EPS: f64 = 1e-9;

        let mut hyperplane = self.points.iter();
        let r = hyperplane.next().unwrap();
        let mut basis: Vec<Point> = Vec::new();

        for q in hyperplane {
            let mut q = q - r;

            for b in &basis {
                q -= b * (q.dot(&b)) / b.norm_squared();
            }

            if q.norm() > EPS {
                basis.push(q);
            }
        }

        let mut p = r - p;

        for b in &basis {
            p -= b * (p.dot(b)) / b.norm_squared();
        }

        p
    }
}

/// Generates matrices for rotations by the first multiples of a given angle
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

/// Generates an array containing only the identity matrix and its negative.
pub fn central_inv(dim: usize) -> [Matrix; 2] {
    let dim = Dynamic::new(dim);
    let matrix = nalgebra::Matrix::identity_generic(dim, dim);

    [matrix.clone(), -matrix]
}

/// Merges various polytopes into a compound.
///
/// # Assumptions
/// * All polytopes are of the same dimension and rank.
/// * The list of polytopes is non-empty.
pub fn compound(polytopes: &[&Polytope]) -> Polytope {
    debug_assert!(!polytopes.is_empty());

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

/// Generates the compound of a polytope and its dual. The dual is rescaled so
/// as to have the same midradius as the original polytope.
pub fn dual_compound(p: &Polytope) -> Polytope {
    let r = p.midradius();

    compound(&[p, &p.dual().scale(r * r)])
}

impl Polytope {
    /// Scales a polytope by a given factor.
    pub fn scale(mut self, k: f64) -> Self {
        for v in &mut self.vertices {
            *v *= k;
        }

        self
    }

    /// Shifts all vertices by a given vector.
    pub fn shift(mut self, o: Point) -> Self {
        for v in &mut self.vertices {
            *v -= &o;
        }

        self
    }

    /// Recenters a polytope so that the gravicenter is at the origin.
    pub fn recenter(self) -> Self {
        let gravicenter = self.gravicenter();

        self.shift(gravicenter)
    }

    /// Applies a matrix to all vertices of a polytope.
    pub fn apply(mut self, m: &Matrix) -> Self {
        for v in &mut self.vertices {
            *v = m * v.clone();
        }

        self
    }

    pub fn get_element_vertices(&self, rank: usize, idx: usize) -> Vec<Point> {
        self.get_element(rank, idx).vertices
    }

    pub fn get_element(&self, rank: usize, idx: usize) -> Self {
        struct Sub {
            rank: usize,
            idx: usize,
        }

        let mut sub_indices: Vec<Vec<Option<usize>>> = Vec::with_capacity(rank);
        let mut index_subs: Vec<Vec<usize>> = Vec::with_capacity(rank);

        let vertices = &self.vertices;
        if rank == 0 {
            return Polytope::new(vec![vertices[idx].clone()], vec![]);
        }

        let elements = &self.elements;
        let el_nums = self.el_nums();

        for el_num in el_nums {
            sub_indices.push(vec![None; el_num]);
            index_subs.push(vec![]);
        }

        let mut sub_deque = VecDeque::new();
        sub_deque.push_back(Sub { rank, idx });

        let mut c = vec![0; rank];
        while let Some(sub) = sub_deque.pop_front() {
            let d = sub.rank - 1;
            let i = sub.idx;

            let els = &elements[d];

            for &j in &els[i] {
                if sub_indices[d][j] == None {
                    sub_indices[d][j] = Some(c[d]);
                    index_subs[d].push(j);
                    c[d] += 1;

                    if d > 0 {
                        sub_deque.push_back(Sub { rank: d, idx: j });
                    }
                }
            }
        }

        let mut new_vertices = Vec::with_capacity(index_subs[0].len());
        for &i in &index_subs[0] {
            new_vertices.push(vertices[i].clone());
        }

        let mut new_elements = Vec::with_capacity(rank);
        for d in 1..rank {
            new_elements.push(Vec::with_capacity(index_subs[d].len()));
            for &i in &index_subs[d] {
                let mut el = elements[d - 1][i].clone();

                for sub in &mut el {
                    *sub = sub_indices[d - 1][*sub].unwrap();
                }

                new_elements[d - 1].push(el);
            }
        }

        let facets = elements[rank - 1][idx].len();
        let mut components = vec![Vec::with_capacity(facets)];
        for i in 0..facets {
            components[0].push(i);
        }
        new_elements.push(components);

        Polytope::new(new_vertices, new_elements)
    }

    /// Gets the [vertex figure](https://polytope.miraheze.org/wiki/Vertex_figure)
    /// of a polytope, corresponding to a given vertex.
    pub fn verf(&self, idx: usize) -> Polytope {
        let dual = self.dual();
        let facet = dual.get_element(self.rank() - 1, idx);

        facet.dual()
    }

    /// Builds the vertices of a dual polytope from its facets.
    fn dual_vertices(&self, sphere: &Hypersphere) -> Vec<Point> {
        const EPS: f64 = 1e-9;

        let vertices = &self.vertices;
        let elements = &self.elements;
        let rank = elements.len();
        let o = &sphere.center;

        // We find the indices of the vertices on the facet.
        let mut projections: Vec<Point>;

        if rank >= 2 {
            let facets = &elements[rank - 2];
            let facets_len = facets.len();

            projections = Vec::with_capacity(facets_len);

            for idx in 0..facets_len {
                let facet_verts = self.get_element_vertices(rank - 1, idx);

                // We project the dual center onto the hyperplane defined by the vertices.
                let h = Hyperplane {
                    points: facet_verts,
                };

                projections.push(h.project(o));
            }
        }
        // If our polytope is 1D, the vertices themselves are the facets.
        else {
            projections = vertices.clone();
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

    /// Builds a dual polytope with a given the center for reciprocation.
    pub fn dual_with_sphere(&self, sphere: &Hypersphere) -> Polytope {
        let rank = self.rank();

        // If we're dealing with a point, let's skip all of the bs:
        if rank == 0 {
            return point();
        }

        let el_nums = self.el_nums();
        let elements = &self.elements;

        let du_vertices = self.dual_vertices(sphere);
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

        // We can only auto-generate the components for 2D and up.
        if rank >= 2 {
            Polytope::new_wo_comps(du_vertices, du_elements)
        }
        // Fortunately, we already know the components in 1D.
        else {
            let components = self.elements[0].clone();
            du_elements.push(components);

            Polytope::new(du_vertices, du_elements)
        }
    }

    /// Builds the dual of a polytope. Defaults to the origin as the center for reciprocation.
    pub fn dual(&self) -> Polytope {
        self.dual_with_sphere(&Hypersphere::unit(self.dimension()))
    }
}
