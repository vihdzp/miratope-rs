//! Contains structs and methods to faciliate geometry in *n*-dimensional space.

/// A point in *n*-dimensional space.
pub type Point<T> = nalgebra::DVector<T>;

/// A vector in *n*-dimensional space.
pub type Vector<T> = Point<T>;

/// A non-owned form of [`Vector`].
pub type VectorSlice<'a, T> = nalgebra::DVectorSlice<'a, T>;

/// An *n* by *n* matrix.
pub type Matrix<T> = nalgebra::DMatrix<T>;

use std::{
    borrow::Cow,
    ops::{Index, IndexMut},
};

use crate::{
    float::Float,
    ElementMap, conc::Concrete, abs::Ranked, Polytope,
};

use approx::{abs_diff_eq, abs_diff_ne};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dynamic, OMatrix, U1};
use vec_like::VecLike;

/// A hypersphere with a certain center and radius.
///
/// This is mostly used for [duals](crate::conc::ConcretePolytope::try_dual_with),
/// where the hypersphere is used to reciprocate polytopes. For convenience, we
/// allow the hypersphere to have a negative squared radius, which results in
/// the dualized polytope being reflected about its center.
pub struct Hypersphere<T: Float> {
    /// The center of the hypersphere.
    pub center: Point<T>,

    /// The squared radius of the hypersphere. We allow negative numbers as a
    /// convenient way to dual + reflect a polytope.
    pub squared_radius: T,
}

impl<T: Float> Hypersphere<T> {
    /// Returns the radius of the hypersphere, or `NaN` if its squared radius is
    /// negative.
    pub fn radius(&self) -> T {
        self.squared_radius // ????
    }

    /// Constructs a hypersphere with a given dimension and radius,
    /// centered at the origin.
    pub fn with_radius(center: Point<T>, radius: T) -> Self {
        Self::with_squared_radius(center, radius * radius)
    }

    /// Constructs a hypersphere with a given dimension and squared radius,
    /// centered at the origin.
    pub fn with_squared_radius(center: Point<T>, squared_radius: T) -> Self {
        Self {
            center,
            squared_radius,
        }
    }

    /// Represents the unit hypersphere in a certain number of dimensions.
    pub fn unit(dim: usize) -> Self {
        Hypersphere::with_squared_radius(Point::zeros(dim), T::ONE)
    }

    /// Attempts to reciprocate a point in place. If it's too close to the
    /// sphere's center, it returns `false` and leaves it unchanged.
    pub fn reciprocate_mut(&self, p: &mut Point<T>) -> bool {
        let mut q = (p as &Point<T>) - &self.center;
        let s = q.norm_squared();

        // If any face passes through the dual center, the dual does
        // not exist, and we return early.
        if s < T::EPS {
            return false;
        }

        q /= s;
        q *= self.squared_radius;
        *p = q + &self.center;
        true
    }

    /// Attempts to reciprocate a point. If it's too close to the sphere's
    /// center, it returns `None`.
    pub fn reciprocate(&self, mut p: Point<T>) -> Option<Point<T>> {
        self.reciprocate_mut(&mut p).then(|| p)
    }
}

/// Represents an (affine) subspace, passing through a given point and generated
/// by a given basis.
///
/// TODO: Use asserts to guarantee that the basis is an orthogonal basis of unit
/// vectors.
#[derive(Clone,)]
pub struct Subspace<T: Float> {
    /// An orthogonal basis for the subspace, defined by unit vectors.
    pub basis: Vec<Vector<T>>,

    /// An "offset", which represents any point on the subspace.
    pub offset: Point<T>,
}

impl<T: Float> Subspace<T> {
    /// Generates a trivial subspace passing through a given point.
    pub fn new(p: Point<T>) -> Self {
        Self {
            basis: Vec::new(),
            offset: p,
        }
    }

    /// Returns the number of dimensions of the ambient space. For the number of
    /// dimensions spanned by the subspace itself, use [`Self::rank`].
    pub fn dim(&self) -> usize {
        self.offset.nrows()
    }

    /// Returns the rank of the subspace, which corresponds to the number of
    /// vectors in its basis.
    pub fn rank(&self) -> usize {
        self.basis.len()
    }

    /// Returns whether the subspace is actually a hyperplane, i.e. a subspace
    /// whose rank is one less than the ambient dimension.
    pub fn is_hyperplane(&self) -> bool {
        self.dim() == self.rank() + 1
    }

    /// Returns whether the subspace is actually of full rank, i.e. a subspace
    /// whose rank equals the ambient dimension.
    pub fn is_full_rank(&self) -> bool {
        self.dim() == self.rank()
    }

    /// Adds a point to the subspace. If it already lies in the subspace, the
    /// subspace remains unchanged and we return `None`. Otherwise, a new basis
    /// vector is added, and we return a reference to it.
    ///
    /// # Todo:
    /// Implement the [Gram-Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability).
    pub fn add(&mut self, p: &Point<T>) -> Option<&Point<T>> {
        let mut v = p - self.project(p);

        if v.normalize_mut() > T::EPS {
            self.basis.push(v);
            self.basis.last()
        } else {
            None
        }
    }

    /// Creates a subspace from an iterator over points. If we find a subspace
    /// of full rank, we return it early. Otherwise, we traverse through the
    /// entire iterator.
    ///
    /// Consider using [`Self::from_points_with`] if you expect your subspace to
    /// have an exact rank.
    pub fn from_points<'a, U: Iterator<Item = &'a Point<T>>>(mut iter: U) -> Self {
        let mut subspace = Self::new(
            iter.next()
                .expect("A hyperplane can't be created from an empty point array!")
                .clone(),
        );

        for p in iter {
            // If the subspace is of full rank, we don't need to check any
            // more points.
            if subspace.add(p).is_some() && subspace.is_full_rank() {
                return subspace;
            }
        }

        subspace
    }

    /// Attempts to create a subspace from an iterator over points with a
    /// specified rank.
    ///
    /// If the points have a higher rank than expected, the method will return
    /// `None` early. If the points have a lower rank, the method will return
    /// `None` after traversing all of them.
    ///
    /// This method is only faster than the usual one when the specified rank
    /// isn't equal to the dimension of the points.
    pub fn from_points_with<'a, U: Iterator<Item = &'a Point<T>>>(
        mut points: U,
        rank: usize,
    ) -> Option<Self> {
        let mut subspace = Self::new(
            points
                .next()
                .expect("A hyperplane can't be created from an empty point array!")
                .clone(),
        );

        for p in points {
            if subspace.add(p).is_some() && subspace.rank() > rank {
                return None;
            }
        }

        Some(subspace)
    }

    /// Projects a point onto the subspace.
    pub fn project(&self, p: &Point<T>) -> Point<T> {
        let p = p - &self.offset;
        let mut q = self.offset.clone();

        for b in &self.basis {
            q += b * p.dot(b);
        }

        q
    }

    /// Projects a point onto the subspace, but returns lower-dimensional
    /// coordinates in the subspace's basis.
    pub fn flatten(&self, p: &Point<T>) -> Point<T> {
        let p = p - &self.offset;
        Point::from_iterator(self.rank(), self.basis.iter().map(|b| p.dot(b)))
    }

    /// Projects a set of points onto the subspace, but returns
    /// lower-dimensional coordinates in the subspace's basis.
    ///
    /// This optimizes [`Self::flatten`] by just returning the original set in
    /// case that the subspace is of full rank.
    pub fn flatten_vec<'a>(&self, vec: &'a [Point<T>]) -> Cow<'a, [Point<T>]> {
        if self.is_full_rank() {
            Cow::Borrowed(vec)
        } else {
            Cow::Owned(vec.iter().map(|v| self.flatten(v)).collect())
        }
    }

    /// Calculates the distance from a point to the subspace.
    pub fn distance(&self, p: &Point<T>) -> T {
        (p - self.project(p)).norm()
    }

    /// Computes a normal vector to the subspace, so that the specified point is
    /// left out of it. Returns `None` if the point given lies on the subspace.
    pub fn normal(&self, p: &Point<T>) -> Option<Vector<T>> {
        (p - self.project(p)).try_normalize(T::EPS)
    }

    // Computes a set of independent vectors that span the orthogonal
    // complement of the subspace.
    /* pub fn orthogonal_comp(&self) -> Vec<Vector> {
        todo!()
    } */
}

impl Concrete {
    /// Computes the affine hull of an element.
    pub fn affine_hull(&self, rank: usize, idx: usize) -> Subspace<f64> {
        Subspace::from_points(
            &mut self.element(rank, idx).unwrap().vertices.iter(),
        )
    }

    /// Computes the affine hulls of all elements and puts them in an `ElementMap`.
    pub fn element_map_affine_hulls(&self) -> ElementMap<Subspace<f64>> {
        let mut element_map = ElementMap::new();
        for r in 1..self.rank() {
            element_map.push(Vec::new());
            for (idx, _el) in self[r].iter().enumerate() {
                element_map[r-1].push(self.affine_hull(r, idx));
            }
        }
        element_map
    }
} 

/// Represents an (oriented) hyperplane together with a normal vector.
pub struct Hyperplane<T: Float> {
    /// The underlying subspace associated to the hyperplane.
    pub subspace: Subspace<T>,

    /// The normal vector of the hyperplane.
    normal: Vector<T>,
}

impl<T: Float> Hyperplane<T> {
    /// Generates an oriented hyperplane from its normal vector.
    pub fn new(normal: Vector<T>, pos: T) -> Self {
        let rank = normal.len();
        let mut subspace = Subspace::new(&normal * pos);
        let mut e = Vector::zeros(rank);

        for i in 0..rank {
            e[i] = T::ONE;
            e += &normal * (pos - e.dot(&normal));
            subspace.add(&e);
            e[i] = T::ZERO;
        }

        Self { subspace, normal }
    }

    /// Projects a point onto the hyperplane.
    pub fn project(&self, p: &Point<T>) -> Point<T> {
        self.subspace.project(p)
    }

    /// Calculates the signed distance from a point to the hyperplane. Points on
    /// the side of the hyperplane containing the vector have positive distance.
    pub fn distance(&self, p: &Point<T>) -> T {
        (p - self.project(p)).dot(&self.normal)
    }

    /// Applies a map from the hyperplane to a lower dimensional space to the
    /// point.
    pub fn flatten(&self, p: &Point<T>) -> Point<T> {
        self.subspace.flatten(p)
    }

    /// Returns whether a point is contained on the hyperplane.
    pub fn is_outer(&self, p: &Point<T>) -> bool {
        abs_diff_eq!(self.distance(p), T::ZERO, epsilon = T::EPS)
    }

    /// Returns the intersection of itself and a line segment, or `None` if it
    /// doesn't exist.
    pub fn intersect(&self, line: Segment<'_, T>) -> Option<Point<T>> {
        let d0 = self.distance(line.0);
        let d1 = self.distance(line.1);

        // This right here is some really sensitive code. If we screw up
        // handling the edge cases, cross-sections through elements will crash.
        (abs_diff_ne!(d0, d1, epsilon = T::EPS) && (d0 < -T::EPS) != (d1 < -T::EPS))
            .then(|| line.at(d1 / (d1 - d0)))
    }
}

/// Represents a line segment between two points.
pub struct Segment<'a, T: Float>(pub &'a Point<T>, pub &'a Point<T>);

impl<'a, T: Float> Segment<'a, T> {
    /// Returns the point at a certain position along the line. If `t` is
    /// between 0 and 1, the point will be contained on the line segment.
    pub fn at(&self, t: T) -> Point<T> {
        self.0 * t + self.1 * (T::ONE - t)
    }
}

/// A matrix ordered by fuzzy lexicographic ordering. That is, lexicographic
/// ordering where two entries that differ by less than an epsilon are
/// considered equal.
///
/// This struct can be used to build a `BTreeSet` of points or matrices in a way
/// that's resistant to floating point errors.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct MatrixOrdMxN<T: Float, R: Dim, C: Dim>(pub OMatrix<T, R, C>)
where
    DefaultAllocator: Allocator<T, R, C>;

impl<T: Float, R: Dim, C: Dim> MatrixOrdMxN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    /// Initializes a new `MatrixOrd` from a `Matrix`.
    pub fn new(mat: OMatrix<T, R, C>) -> Self {
        Self(mat)
    }

    /// Returns a reference to the inner matrix.
    pub fn matrix(&self) -> &OMatrix<T, R, C> {
        &self.0
    }

    /// Iterates over the entries of the `MatrixOrd`.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// Mutably iterates over the entries of the `MatrixOrd`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }

    /// Returns the shape of the matrix.
    pub fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }
}

impl<T: Float, R: Dim, C: Dim> PartialEq for MatrixOrdMxN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.shape(), other.shape(), "matrix shape mismatch");
        self.iter()
            .zip(other.iter())
            .all(|(x, y)| abs_diff_eq!(x, y, epsilon = T::EPS))
    }
}

/// Equality on `MatrixOrds` should be an equality relation, as long as the
/// distance between the `MatrixOrds` you're comparing is "small".
impl<T: Float, R: Dim, C: Dim> Eq for MatrixOrdMxN<T, R, C> where
    DefaultAllocator: Allocator<T, R, C>
{
}

impl<T: Float, R: Dim, C: Dim> PartialOrd for MatrixOrdMxN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        for (x, y) in self.iter().zip(other.iter()) {
            if abs_diff_ne!(x, y, epsilon = T::EPS) {
                return x.partial_cmp(y);
            }
        }

        Some(std::cmp::Ordering::Equal)
    }
}

impl<T: Float, R: Dim, C: Dim> Ord for MatrixOrdMxN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).expect("Matrix has NaN values")
    }
}

impl<T: Float, R: Dim, C: Dim> Index<(usize, usize)> for MatrixOrdMxN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Float, R: Dim, C: Dim> IndexMut<(usize, usize)> for MatrixOrdMxN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: Float, R: Dim, C: Dim> From<OMatrix<T, R, C>> for MatrixOrdMxN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn from(mat: OMatrix<T, R, C>) -> Self {
        Self(mat)
    }
}

/// A matrix ordered by fuzzy lexicographic ordering. For more info, see
/// [`MatrixOrdMxN`].
pub type MatrixOrd<T> = MatrixOrdMxN<T, Dynamic, Dynamic>;

/// A point ordered by fuzzy lexicographic ordering. For more info, see
/// [`MatrixOrdMxN`].
pub type PointOrd<T> = MatrixOrdMxN<T, Dynamic, U1>;

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use nalgebra::dvector;

    fn assert_eq(p: Point<f32>, q: Point<f32>) {
        assert_abs_diff_eq!((p - q).norm(), 0.0, epsilon = f32::EPS)
    }

    #[test]
    /// Reciprocates points about spheres.
    pub fn reciprocate() {
        assert_eq(
            Hypersphere::unit(2)
                .reciprocate(dvector![3.0, 4.0])
                .unwrap(),
            dvector![0.12, 0.16],
        );

        assert_eq(
            Hypersphere::with_radius(Point::zeros(3), 13.0)
                .reciprocate(dvector![3.0, 4.0, 12.0])
                .unwrap(),
            dvector![3.0, 4.0, 12.0],
        );

        assert_eq(
            Hypersphere {
                squared_radius: -4.0,
                center: dvector![1.0, 1.0, 1.0, 1.0],
            }
            .reciprocate(dvector![-2.0, -2.0, -2.0, -2.0])
            .unwrap(),
            dvector![4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0],
        );
    }
}
