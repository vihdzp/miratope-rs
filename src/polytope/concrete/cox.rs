use crate::geometry::Matrix;

use std::mem;

/// Represents a [Coxeter
/// matrix](https://en.wikipedia.org/wiki/Coxeter_group#Coxeter_matrix_and_Schl%C3%A4fli_matrix),
/// which encodes the angles between the mirrors of the generators of a Coxeter
/// group.
pub struct CoxMatrix(pub Matrix);

impl CoxMatrix {
    pub fn from_lin_diagram(diagram: Vec<f64>) -> Self {
        let dim = diagram.len() + 1;

        CoxMatrix(Matrix::from_fn(dim, dim, |mut i, mut j| {
            // Makes i ≤ j.
            if i > j {
                mem::swap(&mut i, &mut j);
            }

            match j - i {
                0 => 1.0,
                1 => diagram[i],
                _ => 2.0,
            }
        }))
    }

    pub fn nrows(&self) -> usize {
        self.0.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.0.ncols()
    }
}

impl std::ops::Index<(usize, usize)> for CoxMatrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

/// Builds a Coxeter group for a given linear diagram.
///
/// # Examples
///
/// ```
/// # #[macro_use]
/// # fn main() {
/// assert_eq!(cox!(4, 3).order(), 48);
/// # }
/// ```
///
/// # Panics
/// Panics if the linear diagram doesn't fit in Euclidean space.
#[macro_export]
macro_rules! cox {
    () => (
        crate::polytope::concrete::group::Group::cox_group(crate::cox_mat!()).unwrap()
    );
    ($($x:expr),+) => (
        crate::polytope::concrete::group::Group::cox_group(crate::cox_mat!($($x),+)).unwrap()
    );
    ($x:expr; $y:expr) => (
        crate::polytope::concrete::group::Group::cox_group(crate::cox_mat!($x; $y)).unwrap()
    )
}

/// Builds a Coxeter matrix for a given linear diagram.
#[macro_export]
macro_rules! cox_mat {
    () => (
        crate::polytope::concrete::cox::CoxMatrix::from_lin_diagram(vec![])
    );
    ($($x:expr),+) => (
        crate::polytope::concrete::cox::CoxMatrix::from_lin_diagram(vec![$($x as f64),+])
    );
    ($x:expr; $y:expr) => (
        crate::polytope::concrete::cox::CoxMatrix::from_lin_diagram(vec![$x as f64; $y])
    )
}
