use nalgebra::DMatrix as Matrix;
use std::{
    mem,
    ops::{Deref, DerefMut},
};

/// Represents a [Coxeter
/// matrix](https://en.wikipedia.org/wiki/Coxeter_group#Coxeter_matrix_and_Schl%C3%A4fli_matrix),
/// which encodes the angles between the mirrors of the generators of a Coxeter
/// group.
pub struct CoxMatrix(pub Matrix<f64>);

impl Deref for CoxMatrix {
    type Target = Matrix<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for CoxMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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
}

/// Builds a Coxeter matrix for a given linear diagram.
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
        crate::polytope::group::Group::cox_group(crate::polytope::cox::CoxMatrix::from_lin_diagram(vec![])).unwrap()
    );
    ($($x:expr),+) => (
       crate::polytope::group::Group::cox_group(crate::polytope::cox::CoxMatrix::from_lin_diagram(vec![$($x as f64),+])).unwrap()
    );
    ($x:expr; $y:expr) => (
        crate::polytope::group::Group::cox_group(crate::polytope::cox::CoxMatrix::from_lin_diagram(vec![$x as f64; $y])).unwrap()
    )
}
