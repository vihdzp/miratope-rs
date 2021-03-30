use std::ops::Mul;

use nalgebra::{DMatrix as Matrix, Scalar};

/// A [group](https://en.wikipedia.org/wiki/Group_(mathematics)) of matrices,
/// acting on a space of a certain dimension.
pub trait Group: Iterator<Item = Matrix<f64>> {
    /// Returns the dimension of the space the group acts on.
    fn dimension(&self) -> usize;

    /// Gets all of the elements of the group. Consumes the iterator.
    fn elements(&mut self) -> Vec<Matrix<f64>> {
        self.collect()
    }
}

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
pub struct RotGroup(dyn Group);

impl Group for RotGroup {
    fn dimension(&self) -> usize {
        self.0.dimension()
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

/// A group [generated](https://en.wikipedia.org/wiki/Generator_(mathematics))
/// by a set of matrices. Its elements are built in a BFS order. It contains
/// some sort of lookup table, implementation dependent, used to figure out
/// whether an element has already been found or not.
pub trait GenGroup: Group
where
    f64: From<Self::T>,
    Matrix<Self::T>: Mul<Output = Matrix<Self::T>>,
{
    /// The type of the entries of the matrix. Should be a numeric type.
    type T: Scalar + Copy;

    /// Builds a new group from a set of generators.
    fn new(generators: Vec<Matrix<Self::T>>) -> Self;

    /// Returns the set of generators of the group.
    fn generators(&self) -> &Vec<Matrix<Self::T>>;

    /// Determines whether a given element has already been found.
    fn contains(&self, el: &Matrix<Self::T>) -> bool;

    /// Inserts a new element into the group.
    fn insert(&mut self, el: Matrix<Self::T>);

    /// Gets the next element and the next generator to attempt to multiply.
    fn next_el_gen(&mut self) -> Option<(&Matrix<Self::T>, &Matrix<Self::T>)>;

    /// Multiplies the current element times the current generator, sees if it's
    /// a new element.
    fn try_next(&mut self) -> GroupNext {
        if let Some((el, gen)) = self.next_el_gen() {
            let new_el = el.clone() * gen.clone();

            if self.contains(&new_el) {
                GroupNext::Repeat
            } else {
                self.insert(new_el.clone());

                // Converts the matrix into a floating point matrix.
                GroupNext::New(Matrix::from_iterator(
                    new_el.nrows(),
                    new_el.ncols(),
                    new_el.into_iter().map(|&x| x.into()),
                ))
            }
        } else {
            GroupNext::None
        }
    }
}

/// Implements [`Group`] for every [`GenGroup`].
impl<T: GenGroup<T = U>, U: Scalar + Copy> Group for T
where
    f64: From<U>,
    Matrix<U>: Mul<Output = Matrix<U>>,
{
    fn dimension(&self) -> usize {
        self.generators()
            .get(0)
            .expect("GenGroup has no generators.")
            .ncols()
    }
}

/// A `Group` with integer matrices, which allows for exact arithmetic.
struct IntGroup {
    /// The generators for the group.
    generators: Vec<Matrix<i32>>,

    /// The elements that have been generated. Will be put into a more clever
    /// structure that's asymptotically more efficient and doesn't need storing
    /// everything at once eventually.
    elements: Vec<Matrix<i32>>,

    /// Stores the index in [`elements`] of the element that is currently being
    /// handled. All previous ones will have already had their right neighbors
    /// found. Quirk of the current data structure, subject to change.
    el_idx: usize,

    /// Stores the index in [`generators`] of the generator that's being
    /// checked. All previous once will have already been multiplied to the
    /// right of the current element. Quirk of the current data structure,
    /// subject to change.
    gen_idx: usize,
}

impl Iterator for IntGroup {
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

/// Every [`IntGroup`] is generated.
impl GenGroup for IntGroup {
    type T = i32;

    /// Initializes a new group with a set of generators.
    fn new(generators: Vec<Matrix<i32>>) -> Self {
        let dim = generators
            .get(0)
            .expect("Vector of generators is empty.")
            .ncols();

        Self {
            generators,
            elements: vec![Matrix::identity(dim, dim)],
            el_idx: 0,
            gen_idx: 0,
        }
    }

    /// Returns the set of generators of the group.
    fn generators(&self) -> &Vec<Matrix<Self::T>> {
        &self.generators
    }

    /// Internal function, used to see if a given element has already been
    /// found. Avoids infinite loops, works precisely due to the fact we're
    /// doing exact arithmetic.
    ///
    /// TODO: use a more clever data structure.
    fn contains(&self, el: &Matrix<i32>) -> bool {
        self.elements.contains(el)
    }

    fn insert(&mut self, el: Matrix<i32>) {
        self.elements.push(el);
    }

    fn next_el_gen(&mut self) -> Option<(&Matrix<i32>, &Matrix<i32>)> {
        let el = self.elements.get(self.el_idx)?;
        let gen = self.generators.get(self.gen_idx).unwrap();

        self.gen_idx += 1;
        if self.gen_idx == self.generators.len() {
            self.gen_idx = 0;
            self.el_idx += 1;
        }

        Some((el, gen))
    }
}
