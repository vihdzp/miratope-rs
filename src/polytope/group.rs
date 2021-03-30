//! Contains

use nalgebra::DMatrix as Matrix;

/// A [group](https://en.wikipedia.org/wiki/Group_(mathematics)) of matrices,
/// acting on a space of a certain dimension.
pub trait Group: Iterator<Item = Matrix<f64>> {
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

    /// Stores the index in [`generators`] of the generator that's being
    /// checked. All previous once will have already been multiplied to the
    /// right of the current element. Quirk of the current data structure,
    /// subject to change.
    gen_idx: usize,
}

impl Group for GenGroup {
    fn dimension(&self) -> usize {
        self.generators
            .get(0)
            .expect("GenGroup has no generators.")
            .ncols()
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
    const EPS: f64 = 1e-6;

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

impl GenGroup {
    /// Builds a new group from a set of generators.
    fn new(generators: Vec<Matrix<f64>>) -> Self {
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

    /// Determines whether a given element has already been found.
    fn contains(&self, el: &Matrix<f64>) -> bool {
        self.elements
            .iter()
            .find(|&search| matrix_approx(search, el))
            .is_some()
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
        // If we already went through the entire group.
        else {
            GroupNext::None
        }
    }
}
