//! Contains methods to generate many symmetry groups.

pub mod cd;

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    iter,
};

use super::concrete::Concrete;
use crate::{
    geometry::{Matrix, MatrixOrd, Point, PointOrd, VectorSlice},
    Consts, Float,
};
use cd::{Cd, CdResult, CoxMatrix};

use approx::{abs_diff_ne, relative_eq};
use nalgebra::{Dynamic, Quaternion, VecStorage};

/// Converts a 3D rotation matrix into a quaternion. Uses the code from
/// [Day (2015)](https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf).
fn mat_to_quat(mat: Matrix) -> Quaternion<Float> {
    debug_assert!(
        relative_eq!(mat.determinant(), 1.0, epsilon = Float::EPS),
        "Only matrices with determinant 1 can be turned into quaternions."
    );

    let t;
    let q;

    if mat[(2, 2)] < 0.0 {
        if mat[(0, 0)] > mat[(1, 1)] {
            t = 1.0 + mat[(0, 0)] - mat[(1, 1)] - mat[(2, 2)];
            q = Quaternion::new(
                t,
                mat[(0, 1)] + mat[(1, 0)],
                mat[(2, 0)] + mat[(0, 2)],
                mat[(1, 2)] - mat[(2, 1)],
            );
        } else {
            t = 1.0 - mat[(0, 0)] + mat[(1, 1)] - mat[(2, 2)];
            q = Quaternion::new(
                mat[(0, 1)] + mat[(1, 0)],
                t,
                mat[(1, 2)] + mat[(2, 1)],
                mat[(2, 0)] - mat[(0, 2)],
            );
        }
    } else if mat[(0, 0)] < -mat[(1, 1)] {
        t = 1.0 - mat[(0, 0)] - mat[(1, 1)] + mat[(2, 2)];
        q = Quaternion::new(
            mat[(2, 0)] + mat[(0, 2)],
            mat[(1, 2)] + mat[(2, 1)],
            t,
            mat[(0, 1)] - mat[(1, 0)],
        );
    } else {
        t = 1.0 + mat[(0, 0)] + mat[(1, 1)] + mat[(2, 2)];
        q = Quaternion::new(
            mat[(1, 2)] - mat[(2, 1)],
            mat[(2, 0)] - mat[(0, 2)],
            mat[(0, 1)] - mat[(1, 0)],
            t,
        );
    }

    q * 0.5 / t.sqrt()
}

/// Converts a quaternion into a matrix, depending on whether it's a left or
/// right quaternion multiplication.
fn quat_to_mat(q: Quaternion<Float>, left: bool) -> Matrix {
    let size = Dynamic::new(4);
    let left = if left { 1.0 } else { -1.0 };

    Matrix::from_data(VecStorage::new(
        size,
        size,
        vec![
            q.w,
            q.i,
            q.j,
            q.k,
            -q.i,
            q.w,
            left * q.k,
            -left * q.j,
            -q.j,
            -left * q.k,
            q.w,
            left * q.i,
            -q.k,
            left * q.j,
            -left * q.i,
            q.w,
        ],
    ))
}

/// Computes the [direct sum](https://en.wikipedia.org/wiki/Block_matrix#Direct_sum)
/// of two matrices.
fn direct_sum(mat1: Matrix, mat2: Matrix) -> Matrix {
    let dim1 = mat1.nrows();
    let dim = dim1 + mat2.nrows();

    Matrix::from_fn(dim, dim, |i, j| {
        if i < dim1 {
            if j < dim1 {
                mat1[(i, j)]
            } else {
                0.0
            }
        } else if j >= dim1 {
            mat2[(i - dim1, j - dim1)]
        } else {
            0.0
        }
    })
}

/// An iterator such that `dyn` objects using it can be cloned. Used to get
/// around orphan rules.
pub trait GroupIter: Iterator<Item = Matrix> + dyn_clone::DynClone {}
impl<T: Iterator<Item = Matrix> + dyn_clone::DynClone> GroupIter for T {}
dyn_clone::clone_trait_object!(GroupIter);

/// A [group](https://en.wikipedia.org/wiki/Group_(mathematics)) of matrices,
/// acting on a space of a certain dimension.
#[derive(Clone)]
pub struct Group {
    /// The dimension of the matrices of the group. Stored separately, since
    /// making the iterator peekable doesn't seem to work.
    dim: usize,

    /// The underlying iterator, which actually outputs the matrices.
    iter: Box<dyn GroupIter>,
}

impl Iterator for Group {
    type Item = Matrix;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl Group {
    pub fn new<T: 'static + GroupIter>(dim: usize, iter: T) -> Self {
        Self {
            dim,
            iter: Box::new(iter),
        }
    }

    /// Gets all of the elements of the group. Consumes the iterator.
    pub fn elements(self) -> Vec<Matrix> {
        self.collect()
    }

    /// Gets the number of elements of the group. Consumes the iterator.
    pub fn order(self) -> usize {
        self.count()
    }

    pub fn from_gens(dim: usize, gens: Vec<Matrix>) -> Self {
        Self::new(dim, Box::new(GenIter::new(dim, gens)))
    }

    /// Buils the rotation subgroup of a group.
    pub fn rotations(self) -> Self {
        // The determinant might not be exactly 1, so we're extra lenient and
        // just test for positive determinants.
        Self::new(self.dim, self.filter(|el| el.determinant() > 0.0))
    }

    /// Builds an iterator over the set of either left or a right quaternions
    /// from a 3D group. **These won't actually generate a group,** as they
    /// don't contain central inversion.
    fn quaternions(self, left: bool) -> Box<dyn GroupIter> {
        if self.dim != 3 {
            panic!("Quaternions can only be generated from 3D matrices.");
        }

        Box::new(
            self.rotations()
                .map(move |el| quat_to_mat(mat_to_quat(el), left)),
        )
    }

    /// Returns the swirl symmetry group of two 3D groups.
    pub fn swirl(g: Self, h: Self) -> Self {
        if g.dim != 3 {
            panic!("g must be a group of 3D matrices.");
        }
        if h.dim != 3 {
            panic!("h must be a group of 3D matrices.");
        }

        Self::new(
            4,
            itertools::iproduct!(g.quaternions(true), h.quaternions(false))
                .map(|(mat1, mat2)| {
                    let mat = mat1 * mat2;
                    iter::once(mat.clone()).chain(iter::once(-mat))
                })
                .flatten(),
        )
    }

    /// Returns a new group whose elements have all been generated already, so
    /// that they can be used multiple times quickly.
    pub fn cache(self) -> Self {
        let elements = self.elements();
        Self::new(elements[0].nrows(), elements.into_iter())
    }

    /// Returns the exact same group, but now asserts that each generated
    /// element has the appropriate dimension. Used for debugging purposes.
    pub fn debug(self) -> Self {
        const MSG: &str = "Size of matrix does not match expected dimension.";
        let dim = self.dim;

        Self::new(
            dim,
            self.map(move |x| {
                assert_eq!(x.nrows(), dim, "{}", MSG);
                assert_eq!(x.ncols(), dim, "{}", MSG);
                x
            }),
        )
    }

    /// Parses a [`Cd`] and turns it into a group.
    pub fn parse(input: &str) -> CdResult<Option<Self>> {
        Cd::parse(input).map(|cd| Self::cox_group(cd.cox()))
    }

    /// Shorthand for `Self::parse(input).unwrap().unwrap()`.
    pub fn parse_unwrap(input: &str) -> Self {
        Self::parse(input).unwrap().unwrap()
    }

    /// Generates the trivial group of a certain dimension.
    pub fn trivial(dim: usize) -> Self {
        Self::new(dim, iter::once(Matrix::identity(dim, dim)))
    }

    /// Generates the group with the identity and a central inversion of a
    /// certain dimension.
    pub fn central_inv(dim: usize) -> Self {
        Self::new(
            dim,
            vec![Matrix::identity(dim, dim), -Matrix::identity(dim, dim)].into_iter(),
        )
    }

    /// Returns the I2(x) symmetry group.
    pub fn i2(x: Float) -> Self {
        Self::cox_group(CoxMatrix::i2(x)).unwrap()
    }

    /// Returns the An symmetry group.
    pub fn a(n: usize) -> Self {
        Self::cox_group(CoxMatrix::a(n)).unwrap()
    }

    /// Returns the Bn symmetry group.
    pub fn b(n: usize) -> Self {
        Self::cox_group(CoxMatrix::b(n)).unwrap()
    }

    /// Generates a step prism group from a base group and a homomorphism into
    /// another group.
    pub fn step(g: Self, f: impl Fn(Matrix) -> Matrix + Clone + 'static) -> Self {
        let dim = g.dim * 2;

        Self {
            dim,
            iter: Box::new(g.map(move |mat| direct_sum(mat.clone(), f(mat)))),
        }
    }

    /// Generates a Coxeter group from its [`CdMatrix`], or returns `None` if
    /// the group doesn't fit as a matrix group in spherical space.
    pub fn cox_group(cox: CoxMatrix) -> Option<Self> {
        Some(Self::new(cox.dim(), GenIter::from_cox(cox)?))
    }

    /// Generates the direct product of two groups. Uses the specified function
    /// to uniquely map the ordered pairs of matrices into other matrices.
    pub fn fn_product(
        g: Self,
        h: Self,
        dim: usize,
        product: (impl Fn((Matrix, Matrix)) -> Matrix + Clone + 'static),
    ) -> Self {
        Self::new(dim, itertools::iproduct!(g, h).map(product))
    }

    /// Returns the group determined by all products between elements of the
    /// first and the second group. **Is meant only for groups that commute with
    /// one another.**
    pub fn matrix_product(g: Self, h: Self) -> Option<Self> {
        // The two matrices must have the same size.
        if g.dim != h.dim {
            return None;
        }

        let dim = g.dim;
        Some(Self::fn_product(g, h, dim, |(mat1, mat2)| mat1 * mat2))
    }

    /// Calculates the direct product of two groups. Pairs of matrices are then
    /// mapped to their direct sum.
    pub fn direct_product(g: Self, h: Self) -> Self {
        let dim = g.dim + h.dim;
        Self::fn_product(g, h, dim, |(mat1, mat2)| direct_sum(mat1, mat2))
    }

    /// Generates the [wreath product](https://en.wikipedia.org/wiki/Wreath_product)
    /// of two symmetry groups.
    pub fn wreath(g: Self, h: Self) -> Self {
        let h = h.elements();
        let g_dim = g.dim;
        let dim = g_dim * h.len();

        // Indexes each element in h.
        let mut h_indices = BTreeMap::new();

        for (i, h_el) in h.iter().enumerate() {
            h_indices.insert(MatrixOrd::new(h_el.clone()), i);
        }

        // Converts h into a permutation group.
        let mut permutations = Vec::with_capacity(h.len());

        for h_el_1 in &h {
            let mut perm = Vec::with_capacity(h.len());

            for h_el_2 in &h {
                perm.push(
                    *h_indices
                        .get(&MatrixOrd::new(h_el_1 * h_el_2))
                        .expect("h is not a valid group!"),
                );
            }

            permutations.push(perm);
        }

        // Computes the direct product of g with itself |h| times.
        let g_prod = vec![&g; h.len() - 1]
            .into_iter()
            .cloned()
            .fold(g.clone(), |acc, g| Group::direct_product(g, acc));

        Self::new(
            dim,
            g_prod
                .map(move |g_el| {
                    let mut matrices = Vec::new();

                    for perm in &permutations {
                        let mut new_el = Matrix::zeros(dim, dim);

                        // Permutes the blocks on the diagonal of g_el.
                        for (i, &j) in perm.iter().enumerate() {
                            for x in 0..g_dim {
                                for y in 0..g_dim {
                                    new_el[(i * g_dim + x, j * g_dim + y)] =
                                        g_el[(i * g_dim + x, i * g_dim + y)];
                                }
                            }
                        }

                        matrices.push(new_el);
                    }

                    matrices.into_iter()
                })
                .flatten(),
        )
    }

    /// Generates the orbit of a point under a given symmetry group.
    pub fn orbit(self, p: Point) -> Vec<Point> {
        let mut points = BTreeSet::new();

        for m in self {
            points.insert(PointOrd::new(m * &p));
        }

        points.into_iter().map(|x| x.0).collect()
    }

    /// Generates a polytope as the convex hull of the orbit of a point under a
    /// given symmetry group.
    pub fn into_polytope(self, _: Point) -> Concrete {
        todo!()
        // convex::convex_hull(self.orbit(p))
    }
}

/// The result of trying to get the next element in a group.
pub enum GroupNext {
    /// We've already found all elements of the group.
    None,

    /// We found an element we had found previously.
    Repeat,

    /// We found a new element.
    New(Matrix),
}

/// An iterator for a `Group` [generated](https://en.wikipedia.org/wiki/Generator_(mathematics))
/// by a set of floating point matrices. Its elements are built in a BFS order.
/// It contains a lookup table, used to figure out whether an element has
/// already been found or not, as well as a queue to store the next elements.
#[derive(Clone)]
pub struct GenIter {
    /// The number of dimensions the group acts on.
    pub dim: usize,

    /// The generators for the group.
    pub gens: Vec<Matrix>,

    /// Stores the elements that have been generated and that can still be
    /// generated again. Is integral for the algorithm to work, as without it,
    /// duplicate group elements will just keep generating forever.
    elements: BTreeMap<MatrixOrd, usize>,

    /// Stores the elements that haven't yet been processed.
    queue: VecDeque<MatrixOrd>,

    /// Stores the index in (`generators`)[GenGroup.generators] of the generator
    /// that's being checked. All previous once will have already been
    /// multiplied to the right of the current element. Quirk of the current
    /// data structure, subject to change.
    gen_idx: usize,
}

impl Iterator for GenIter {
    type Item = Matrix;

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
fn matrix_approx(mat1: &Matrix, mat2: &Matrix) -> bool {
    let mat1 = mat1.iter();
    let mut mat2 = mat2.iter();

    for x in mat1 {
        let y = mat2.next().expect("Matrices don't have the same size!");

        if abs_diff_ne!(x, y, epsilon = Float::EPS) {
            return false;
        }
    }

    true
}

/// Builds a reflection matrix from a given vector.
pub fn refl_mat(n: VectorSlice) -> Matrix {
    let dim = n.nrows();
    let nn = n.norm_squared();
    let identity = Matrix::identity(dim, dim);

    // Reflects every basis vector, builds a matrix from all of their images.
    Matrix::from_columns(
        &identity
            .column_iter()
            .map(|e| e - (2.0 * e.dot(&n) / nn) * n)
            .collect::<Vec<_>>(),
    )
}

impl GenIter {
    /// Builds a new group from a set of generators.
    fn new(dim: usize, gens: Vec<Matrix>) -> Self {
        // Initializes the queue with only the identity matrix.
        let mut queue = VecDeque::new();
        queue.push_back(MatrixOrd::new(Matrix::identity(dim, dim)));

        // We say that the identity has been found zero times. This is a special
        // case that ensures that neither the identity is queued nor found
        // twice.
        let mut elements = BTreeMap::new();
        elements.insert(MatrixOrd::new(Matrix::identity(dim, dim)), 0);

        Self {
            dim,
            gens,
            elements,
            queue,
            gen_idx: 0,
        }
    }

    /// Inserts a new element into the group. Returns whether the element is new.
    fn insert(&mut self, el: Matrix) -> bool {
        let el = MatrixOrd::new(el);

        // If the element has been found before.
        if let Some(value) = self.elements.insert(el.clone(), 1) {
            // Bumps the value by 1, or removes the element if this is the last
            // time we'll find the element.
            if value != self.gens.len() - 1 {
                self.elements.insert(el, value + 1);
            } else {
                self.elements.remove(&el);
            }

            // The element is a repeat, except in the special case of the
            // identity.
            value == 0
        }
        // If the element is new, we add it to the queue as well.
        else {
            self.queue.push_back(el);
            true
        }
    }

    /// Gets the next element and the next generator to attempt to multiply
    /// with. Advances the iterator.
    fn next_el_gen(&mut self) -> Option<[Matrix; 2]> {
        let el = self.queue.front()?.0.clone();
        let gen = self.gens[self.gen_idx].clone();

        // Advances the indices.
        self.gen_idx += 1;
        if self.gen_idx == self.gens.len() {
            self.gen_idx = 0;
            self.queue.pop_front();
        }

        Some([el, gen])
    }

    /// Multiplies the current element times the current generator, determines
    /// if it is a new element. Advances the iterator.
    fn try_next(&mut self) -> GroupNext {
        // If there's a next element and generator.
        if let Some([el, gen]) = self.next_el_gen() {
            let new_el = el * gen;

            // If the group element is new.
            if self.insert(new_el.clone()) {
                GroupNext::New(new_el)
            }
            // If we found a repeat.
            else {
                GroupNext::Repeat
            }
        }
        // If we already went through the entire group.
        else {
            GroupNext::None
        }
    }

    /// Builds a Coxeter group from the matrix of normal vectors that describes
    /// its mirrors.
    pub fn from_normals(normals: Matrix) -> Self {
        Self::new(
            normals.ncols(),
            normals.column_iter().map(refl_mat).collect(),
        )
    }

    /// Builds a Coxeter group from a Coxeter matrix.
    pub fn from_cox(cox: CoxMatrix) -> Option<Self> {
        Some(Self::from_normals(cox.normals()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gcd::Gcd;

    /// Tests a given symmetry group.
    fn test(group: Group, order: usize, rot_order: usize, name: &str) {
        // Makes testing multiple derived groups faster.
        let group = group.cache().debug();

        // Tests the order of the group.
        assert_eq!(
            group.clone().order(),
            order,
            "{} does not have the expected order.",
            name
        );

        // Tests the order of the rotational subgroup.
        assert_eq!(
            group.rotations().order(),
            rot_order,
            "The rotational group of {} does not have the expected order.",
            name
        );
    }

    /// Tests the trivial group in various dimensions.
    #[test]
    fn i() {
        for n in 1..=10 {
            test(Group::trivial(n), 1, 1, &format!("I^{}", n))
        }
    }

    /// Tests the group consisting of the identity and a central inversion in
    /// various dimensions.
    #[test]
    fn pm_i() {
        for n in 1..=10 {
            test(
                Group::central_inv(n),
                2,
                (n + 1) % 2 + 1,
                &format!("±I{}", n),
            )
        }
    }

    /// Tests the I2(*n*) symmetries, which correspond to the symmetries of a
    /// regular *n*-gon.
    #[test]
    fn i2() {
        for n in 2..=10 {
            for d in 1..n {
                if n.gcd(d) != 1 {
                    continue;
                }

                test(
                    Group::i2(n as Float / d as Float),
                    2 * n,
                    n,
                    &format!("I2({})", n),
                );
            }
        }
    }

    /// Tests the A3⁺ @ (I2(*n*) × I) symmetries, the tetrahedron swirl
    /// symmetries.
    #[test]
    fn a3_p_swirl_i2xi_p() {
        for n in 2..10 {
            let order = 24 * n;

            test(
                Group::swirl(
                    Group::a(3),
                    Group::direct_product(Group::i2(n as Float), Group::trivial(1)),
                ),
                order,
                order,
                &format!("A3⁺ @ (I2({}) × I)", n),
            )
        }
    }

    /// Tests the A*n* symmetries, which correspond to the symmetries of the
    /// regular simplices.
    #[test]
    fn a() {
        let mut order = 2;

        for n in 2..=6 {
            order *= n + 1;

            test(Group::a(n), order, order / 2, &format!("A{}", n))
        }
    }
    /// Tests the ±A*n* symmetries, which correspond to the symmetries of the
    /// compound of two simplices.
    #[test]
    fn pm_an() {
        let mut order = 4;

        for n in 2..=6 {
            order *= n + 1;

            test(
                Group::matrix_product(Group::a(n), Group::central_inv(n)).unwrap(),
                order,
                order / 2,
                &format!("±A{}", n),
            )
        }
    }

    /// Tests the BC*n* symmetries, which correspond to the symmetries of the
    /// regular hypercube and orthoplex.
    #[test]
    fn b() {
        let mut order = 2;

        for n in 2..=6 {
            order *= n * 2;

            test(Group::b(n), order, order / 2, &format!("BC{}", n))
        }
    }

    /// Tests the H*n* symmetries, which correspond to the symmetries of a
    /// regular dodecahedron and a regular hecatonicosachoron.
    #[test]
    fn h() {
        test(Group::parse_unwrap("o5o3o"), 120, 60, &"H3");
        test(Group::parse_unwrap("o5o3o3o"), 14400, 7200, &"H4");
    }

    /// Tests the E6 symmetry group.
    #[test]
    fn e6() {
        test(Group::parse_unwrap("o3o3o3o3o *c3o"), 51840, 25920, &"E6");
    }

    /// Tests the E7 symmetry group. This is very expensive, so we enable it
    /// only on release mode.
    #[test]
    #[cfg(not(debug_assertions))]
    fn e7() {
        test(
            Group::parse_unwrap("o3o3o3o3o3o *c3o"),
            2903040,
            1451520,
            &"E7",
        );
    }

    #[test]
    /// Tests the direct product of A3 with itself.
    fn a3xa3() {
        let a3 = Group::parse_unwrap("o3o3o");
        let g = Group::direct_product(a3.clone(), a3.clone());
        test(g, 576, 288, &"A3×A3");
    }

    #[test]
    /// Tests the wreath product of A3 with A1.
    fn a3_wr_a1() {
        test(
            Group::wreath(Group::a(3), Group::a(1)),
            1152,
            576,
            &"A3 ≀ A1",
        );
    }

    #[test]
    /// Tests out some step prisms.
    fn step() {
        for n in 1..10 {
            for d in 1..n {
                test(
                    Group::step(Group::i2(n as Float).rotations(), move |mat| {
                        mat.pow(d).unwrap()
                    }),
                    n,
                    n,
                    "Step prismatic n-d",
                );
            }
        }
    }
}
