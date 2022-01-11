//! Contains methods to parse and generate Coxeter diagrams and matrices.

pub mod cd;
pub mod parse;

use std::{
    iter,
    ops::{Index, IndexMut},
};

use crate::float::Float;
use crate::group::Group;
use crate::{geometry::Matrix, group::GenIter};

use nalgebra::dmatrix;

use crate::geometry::VectorSlice;

use self::cd::{Cd, CdResult};

/// Represents a [Coxeter matrix](https://en.wikipedia.org/wiki/Coxeter_matrix),
/// which itself represents a [`Cd`]. This representation makes many
/// calculations with Coxeter diagrams much more convenient.
///
/// The Coxeter matrix for a Coxeter diagram is defined so that the (i, j) entry
/// corresponds to the value of the edge between the ith and jth node, or 2 if
/// there's no such edge.
#[derive(Clone, Debug, PartialEq)]
pub struct Cox<T: Float>(Matrix<T>);

impl<T: Float> Index<(usize, usize)> for Cox<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Float> IndexMut<(usize, usize)> for Cox<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Cox<f64> {
    /// Initializes a new CD matrix from a vector of nodes and a matrix.
    pub fn new(matrix: Matrix<f64>) -> Self {
        Self(matrix)
    }

    /// Returns the dimensions of the matrix.
    pub fn dim(&self) -> usize {
        self.0.nrows()
    }

    /// Links together two nodes with a given edge.
    pub fn link(&mut self, i: usize, j: usize, edge: f64) {
        self[(i, j)] = edge;
        self[(j, i)] = edge;
    }

    /// Parses a [`Cd`] and turns it into a Coxeter matrix.
    pub fn parse(input: &str) -> CdResult<Self> {
        Cd::parse(input).map(|cd| cd.cox())
    }

    /// Returns the Coxeter matrix for the trivial 1D group.
    pub fn trivial() -> Self {
        Self::new(dmatrix![1.0])
    }

    /// Returns a mutable reference to the elements of the matrix.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.0.iter_mut()
    }

    /// Creates a Coxeter diagram from an iterator over the entries of its
    /// linear diagram.
    pub fn from_lin_diagram_iter<I: Iterator<Item = f64>>(iter: I, dim: usize) -> Self {
        let mut cox = Self(Matrix::from_fn(dim, dim, |i, j| {
            if i == j {
                1.0
            } else {
                2.0
            }
        }));

        for (i, edge) in iter.enumerate() {
            cox.link(i, i + 1, edge);
        }

        cox
    }

    /// Creates a Coxeter matrix from a linear diagram, whose edges are
    /// described by the vector.
    pub fn from_lin_diagram(diagram: &[f64]) -> Self {
        Self::from_lin_diagram_iter(diagram.iter().copied(), diagram.len())
    }

    /// Returns the Coxeter matrix for the I2(x) group.
    pub fn i2(x: f64) -> Self {
        Self::from_lin_diagram(&[x])
    }

    /// Returns the Coxeter matrix for the An group.
    pub fn a(n: usize) -> Self {
        Self::from_lin_diagram_iter(iter::repeat(3.0).take(n - 1), n)
    }

    /// Returns the Coxeter matrix for the Bn group.
    pub fn b(n: usize) -> Self {
        Self::from_lin_diagram_iter(
            iter::once(4.0).chain(iter::repeat(3.0).take(n - 2)),
            n,
        )
    }

    /// Returns the Coxeter matrix for the Dn group.
    pub fn d(n: usize) -> Self {
        let mut cox = Self::a(n);
        cox.link(0, 1, 2.0);
        cox.link(0, 2, 3.0);
        cox
    }

    /// Returns the Coxeter matrix for the En group.
    pub fn e(n: usize) -> Self {
        let mut cox = Self::a(n);
        cox.link(0, 1, 2.0);
        cox.link(0, 3, 3.0);
        cox
    }

    /// Returns the Coxeter matrix for the Hn group.
    pub fn h(n: usize) -> Self {
        Self::from_lin_diagram_iter(
            iter::once(5.0).chain(iter::repeat(3.0).take(n - 2)),
            n,
        )
    }

    /// Returns an upper triangular matrix whose columns are unit normal vectors
    /// for the hyperplanes described by the Coxeter matrix.
    pub fn normals(&self) -> Option<Matrix<f64>> {
        let dim = self.dim();
        let mut mat = Matrix::zeros(dim, dim);

        // Builds each column from the top down, so that each of the succesive
        // dot products we check match the values in the Coxeter matrix.
        for i in 0..dim {
            let (prev_gens, mut n_i) = mat.columns_range_pair_mut(0..i, i);

            for (j, n_j) in prev_gens.column_iter().enumerate() {
                // All other entries in the dot product between columns are zero.
                let dot = n_i.rows_range(0..=j).dot(&n_j.rows_range(0..=j));
                n_i[j] = ((f64::PI / self[(i, j)]).fcos() - dot) / n_j[j];
            }

            // If the vector doesn't fit in spherical space.
            let norm_sq: f64 = n_i.norm_squared();
            if norm_sq >= 1.0 - f64::EPS {
                return None;
            } else {
                n_i[i] = (1.0 - norm_sq).fsqrt();
            }
        }

        Some(mat)
    }

    /// Returns an iterator over the elements of the Coxeter group.
    pub fn gen_iter(&self) -> Option<GenIter<Matrix<f64>>> {
        let normals = self.normals()?;
        let dim = normals.nrows();

        // Builds a reflection matrix from a vector.
        let refl_mat = |n: VectorSlice<'_, f64>| {
            let nn = n.norm_squared();
            let mut mat = Matrix::identity(dim, dim);

            // Reflects every basis vector, builds a matrix from all of their images.
            for mut e in mat.column_iter_mut() {
                e -= n * (2.0 * e.dot(&n) / nn);
            }

            mat
        };

        Some(GenIter::new(
            dim,
            normals.column_iter().map(refl_mat).collect(),
        ))
    }

    /// Returns the associated Coxeter [`Group`].
    pub fn group(&self) -> Option<Group<GenIter<Matrix<f64>>>> {
        self.gen_iter().map(Into::into)
    }
}
