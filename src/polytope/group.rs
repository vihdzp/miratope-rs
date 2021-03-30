//! Contains methods to generate many symmetry groups.

use dyn_clone::DynClone;
use nalgebra::{DMatrix as Matrix, DVector as Vector};
use std::{
    f64::consts::PI,
    mem,
    ops::{Deref, DerefMut},
};

/// A [group](https://en.wikipedia.org/wiki/Group_(mathematics)) of matrices,
/// acting on a space of a certain dimension.
pub trait Group: Iterator<Item = Matrix<f64>> + DynClone {
    /// Returns the dimension of the space the group acts on.
    fn dimension(&self) -> usize;

    /// Gets all of the elements of the group. Consumes the iterator.
    fn elements(&mut self) -> Vec<Matrix<f64>> {
        self.collect()
    }

    /// Gets the number of elements of the group. Consumes the iterators.
    fn order(&mut self) -> usize {
        self.count()
    }

    /// TODO: figure out a way to have a default implementation for this.
    fn rotations(self) -> RotGroup;

    /// TODO: figure out a way to have a default implementation for this.
    fn quaternions(self) -> RotGroup;
}

dyn_clone::clone_trait_object!(Group);

/// The result of trying to get the next element in a group.
pub enum GroupNext {
    /// We've already found all elements of the group.
    None,

    /// We found an element we had found previously.
    Repeat,

    /// We found a new element.
    New(Matrix<f64>),
}

/// The group of all rotations (matrices with determinant 1) of another group.
#[derive(Clone)]
pub struct RotGroup(Box<dyn Group>);

impl Group for RotGroup {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }

    fn rotations(self) -> RotGroup {
        self
    }

    fn quaternions(self) -> RotGroup {
        todo!()
    }
}

impl Iterator for RotGroup {
    type Item = Matrix<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let el = self.0.next()?;

            // The determinant might not be exactly 1, so we're extra lenient
            // and just test for positive determinants.
            if el.determinant() > 0.0 {
                return Some(el);
            }
        }
    }
}

/// Represents a
/// [Coxeter matrix](https://en.wikipedia.org/wiki/Coxeter_group#Coxeter_matrix_and_Schl%C3%A4fli_matrix),
/// which encodes the angles between the mirrors of the generators of a Coxeter
/// group.
struct CoxMatrix(Matrix<i32>);

impl Deref for CoxMatrix {
    type Target = Matrix<i32>;

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
    fn from_lin_diagram(diagram: Vec<i32>) -> Self {
        let dim = diagram.len() + 1;

        CoxMatrix(Matrix::from_fn(dim, dim, |mut i, mut j| {
            // Makes i ≤ j.
            if i > j {
                mem::swap(&mut i, &mut j);
            }

            match j - i {
                0 => 1,
                1 => diagram[i],
                _ => 2,
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
#[allow(unused_macros)]
#[macro_export]
macro_rules! cox {
    ($($x:expr),+) => (
        CoxMatrix::from_lin_diagram(vec![$($x),+])
    );
    ($x:expr; $y:expr) => (
        CoxMatrix::from_lin_diagram(vec![$x; $y])
    )
}

/// A `Group` [generated](https://en.wikipedia.org/wiki/Generator_(mathematics))
/// by a set of floating point matrices. Its elements are built in a BFS order.
/// It contains a lookup table, used to figure out whether an element has
/// already been found or not.
///
/// # Todo
/// Currently, to figure out whether an element has been found or not, we do a
/// linear search on the entire set of elements that we've found so far. This
/// means that generating a group with *n* elements has O(*n*²) asymptotic
/// complexity, which will be really bad if we ever want to implement big groups
/// like E6, E7, or God forbid E8.
///
/// If all of our matrices had integer entries, which is the case for a lot of
/// Coxeter groups, we could instead use a `HashSet` to reduce the complexity
/// to O(*n* log(*n*)). For floating point entries, where we'll rather want to
/// find the "closest" element to another one (to account for imprecision), a
/// [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) would achieve the same
/// complexity, but it would be much harder to implement.
#[derive(Clone)]
pub struct GenGroup {
    /// The generators for the group.
    generators: Vec<Matrix<f64>>,

    /// The elements that have been generated. Will be put into a more clever
    /// structure that's asymptotically more efficient and doesn't need storing
    /// everything at once eventually.
    elements: Vec<Matrix<f64>>,

    /// Stores the index in (`elements`)[GenGroup.elements] of the element that is currently being
    /// handled. All previous ones will have already had their right neighbors
    /// found. Quirk of the current data structure, subject to change.
    el_idx: usize,

    /// Stores the index in (`generators`)[GenGroup.generators] of the generator
    /// that's being checked. All previous once will have already been
    /// multiplied to the right of the current element. Quirk of the current
    /// data structure, subject to change.
    gen_idx: usize,
}

impl Group for GenGroup {
    fn dimension(&self) -> usize {
        self.generators
            .get(0)
            .expect("GenGroup has no generators.")
            .ncols()
    }

    fn rotations(self) -> RotGroup {
        RotGroup(Box::new(self))
    }

    fn quaternions(self) -> RotGroup {
        todo!()
    }
}

impl Iterator for GenGroup {
    type Item = Matrix<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.try_next() {
                GroupNext::None => return None,
                GroupNext::Repeat => {}
                GroupNext::New(el) => return Some(el),
            };
        }
    }
}

/// Determines whether two matrices are "approximately equal" elementwise.
fn matrix_approx(mat1: &Matrix<f64>, mat2: &Matrix<f64>) -> bool {
    const EPS: f64 = 1e-4;

    let mat1 = mat1.iter();
    let mut mat2 = mat2.iter();

    for x in mat1 {
        let y = mat2.next().unwrap();

        if (x - y).abs() > EPS {
            return false;
        }
    }

    true
}

/// Builds a reflection matrix from a given vector.
pub fn refl_mat(n: Vector<f64>) -> Matrix<f64> {
    let dim = n.nrows();
    let nn = n.norm_squared();

    Matrix::from_columns(
        &Matrix::identity(dim, dim)
            .column_iter()
            .map(|v| v - (2.0 * v.dot(&n) / nn) * &n)
            .collect::<Vec<_>>(),
    )
}

impl GenGroup {
    /// Builds a new group from a set of generators.
    fn new(generators: Vec<Matrix<f64>>) -> Self {
        Self {
            generators,
            elements: Vec::new(),
            el_idx: 0,
            gen_idx: 0,
        }
    }

    /// Determines whether a given element has already been found.
    fn contains(&self, el: &Matrix<f64>) -> bool {
        self.elements.iter().any(|search| matrix_approx(search, el))
    }

    /// Inserts a new element into the group. Assumes that we've already checked
    /// that the element is new.
    fn insert(&mut self, el: Matrix<f64>) {
        self.elements.push(el);
    }

    /// Gets the next element and the next generator to attempt to multiply.
    /// Advances the iterator.
    fn next_el_gen(&mut self) -> Option<[&Matrix<f64>; 2]> {
        let el = self.elements.get(self.el_idx)?;
        let gen = self.generators.get(self.gen_idx).unwrap();

        // Advances the indices.
        self.gen_idx += 1;
        if self.gen_idx == self.generators.len() {
            self.gen_idx = 0;
            self.el_idx += 1;
        }

        Some([el, gen])
    }

    /// Multiplies the current element times the current generator, determines
    /// if it is a new element. Advances the iterator.
    fn try_next(&mut self) -> GroupNext {
        // If there's a next element and generator.
        if let Some([el, gen]) = self.next_el_gen() {
            let new_el = el * gen;

            // If the group element is a repeat.
            if self.contains(&new_el) {
                GroupNext::Repeat
            }
            // If we found something new.
            else {
                self.insert(new_el.clone());
                GroupNext::New(new_el)
            }
        }
        // If this is the first element we generate.
        else if self.elements.is_empty() {
            let dim = self.dimension();
            let i = Matrix::identity(dim, dim);
            self.insert(i.clone());
            GroupNext::New(i)
        }
        // If we already went through the entire group.
        else {
            GroupNext::None
        }
    }

    /// Generates a Coxeter group from its [`CoxMatrix`], or returns `None` if
    /// the group doesn't fit as a matrix group in spherical space.
    fn cox_group(cox: CoxMatrix) -> Option<Self> {
        const EPS: f64 = 1e-6;

        let dim = cox.ncols();
        let mut generators = Vec::with_capacity(dim);

        // Builds each generator from the top down as a triangular matrix, so
        // that the dot products match the values in the Coxeter matrix.
        for i in 0..dim {
            let mut gen_i = Vector::from_element(dim, 0.0);

            for (j, gen_j) in generators.iter().enumerate() {
                let dot = gen_i.dot(gen_j);
                gen_i[j] = ((PI / cox[(i, j)] as f64).cos() - dot) / gen_j[j];
            }

            // The vector doesn't fit in spherical space.
            let norm_sq = gen_i.norm_squared();
            if norm_sq >= 1.0 - EPS {
                return None;
            } else {
                gen_i[i] = (1.0 - norm_sq).sqrt();
            }

            generators.push(gen_i);
        }

        Some(Self::new(
            generators.into_iter().map(|n| refl_mat(n)).collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests a given symmetry group.
    fn test<T: Group + Clone>(group: T, order: usize, rot_order: usize, name: &str) {
        assert_eq!(
            group.clone().order(),
            order,
            "{} does not have the expected order.",
            name
        );

        assert_eq!(
            group.rotations().order(),
            rot_order,
            "The rotational group of {} does not have the expected order.",
            name
        );
    }

    /// Tests the I2(*n*) symmetries, which correspond to the symmetries of a
    /// regular *n*-gon.
    #[test]
    fn i2() {
        for n in 2..=10 {
            test(
                GenGroup::cox_group(cox!(n as i32)).unwrap(),
                2 * n,
                n,
                &format!("I2({})", n),
            );
        }
    }

    /// Tests the A*n* symmetries, which correspond to the symmetries of the
    /// regular simplices.
    #[test]
    fn a() {
        let mut order = 2;

        for n in 2..=5 {
            order *= n + 1;

            test(
                GenGroup::cox_group(cox!(3; n - 1)).unwrap(),
                order,
                order / 2,
                &format!("A{}", n),
            )
        }
    }

    /// Tests the B*n* symmetries, which correspond to the symmetries of the
    /// regular hypercube and orthoplex.
    #[test]
    fn b() {
        let mut order = 2;

        for n in 2..=5 {
            // A better cox! macro would make this unnecessary.
            let mut cox = vec![3; n - 1];
            cox[0] = 4;
            let cox = CoxMatrix::from_lin_diagram(cox);

            order *= n * 2;

            test(
                GenGroup::cox_group(cox).unwrap(),
                order,
                order / 2,
                &format!("B{}", n),
            )
        }
    }

    /// Tests the H*n* symmetries, which correspond to the symmetries of a
    /// regular dodecahedron and a regular hecatonicosachoron.
    #[test]
    fn h() {
        test(GenGroup::cox_group(cox!(5, 3)).unwrap(), 120, 60, &"H3");
        test(
            GenGroup::cox_group(cox!(5, 3, 3)).unwrap(),
            14400,
            7200,
            &"H4",
        );
    }

    /*
     * Here goes future code to test the E*n* symmetries.
    fn e() {

    }
     */
}
