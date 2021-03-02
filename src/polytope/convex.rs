use super::geometry::Point;
use super::Polytope;
use nalgebra::DMatrix;

enum Sign {
    Positive,
    Zero,
    Negative,
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

pub fn convex_hull(_vertices: &[Point]) -> Polytope {
    todo!()
}
