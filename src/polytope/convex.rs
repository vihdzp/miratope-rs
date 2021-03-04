use std::collections::BTreeSet;

use super::geometry::Point;
use super::Polytope;
use nalgebra::DMatrix;

enum Sign {
    Positive,
    Zero,
    Negative,
}

/// Represents a vertex set in a convex hull.
#[derive(PartialEq, PartialOrd, Eq, Ord, Clone)]
struct VertexSet {
    /// The sorted indices of the vertices on the ridge.
    vertices: Vec<usize>,

    /// The vertices in each ridge are oriented. The fact that we need to order
    /// them makes this inconvenient.
    ///
    /// This variable stores whether two vertices have to be swapped to have the
    /// correct orientation.
    orientation: bool,
}

impl VertexSet {
    fn push(&mut self, v: usize) {
        self.vertices.push(v);
    }
}

fn sign(x: f64) -> Sign {
    const EPS: f64 = 1e-9;

    if x > EPS {
        Sign::Positive
    } else if x < EPS {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

/// Returns the sign of the hypervolume of a simplex specified by a set of
/// n-dimensional points.
fn sign_hypervolume(simplex: &[Point]) -> Sign {
    let dim = simplex.len() - 1;
    let mut m = DMatrix::from_element(dim + 1, dim + 1, 1.0);

    for (j, v) in simplex.iter().enumerate() {
        for (i, &c) in v.iter().enumerate() {
            m[(i, j)] = c;
        }
    }

    sign(m.determinant())
}

/// Finds a single ridge on the convex hull.
fn get_hull_ridge(_vertices: &[Point]) -> VertexSet {
    todo!()
}

/// Finds the index of the leftmost vertex relative to a ridge. "Leftmost" will
/// depend on the orientation of the ridge.
fn leftmost_vertex(_vertices: &[Point], _ridge: &VertexSet) -> usize {
    todo!()
}

fn get_new_ridges(_hull_facet: &VertexSet) -> Vec<VertexSet> {
    todo!()
}

fn get_polytope_from_facets(_facets: Vec<VertexSet>) -> Polytope {
    todo!()
}

pub fn convex_hull(vertices: &[Point]) -> Polytope {
    let mut hull = Vec::new();
    let mut ridges = BTreeSet::new();
    ridges.insert(get_hull_ridge(vertices));

    while let Some(ridge) = ridges.iter().next() {
        let mut hull_facet = ridge.clone();
        hull_facet.push(leftmost_vertex(vertices, &ridge));
        // Sort the vertices, while retaining the orientation.

        // In theory, this facet should always be a new one.
        debug_assert!(!hull.contains(&hull_facet));

        for new_ridge in get_new_ridges(&hull_facet) {
            // Surely there's a better way to do this: by calling `contains`, I
            // should already know where to insert or remove the ridge.
            if ridges.contains(&new_ridge) {
                ridges.remove(&new_ridge);
            } else {
                ridges.insert(new_ridge);
            }
        }

        hull.push(hull_facet);
    }

    get_polytope_from_facets(hull)
}
