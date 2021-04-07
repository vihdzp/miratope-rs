//! Contains a few structs and methods to faciliate geometry in n-dimensional
//! space.

pub type Point = nalgebra::DVector<f64>;
pub type Matrix = nalgebra::DMatrix<f64>;

use approx::abs_diff_eq;

use crate::EPS;
use std::fmt;

#[derive(Debug)]
/// A hypersphere with a certain center and radius.
pub struct Hypersphere {
    pub center: Point,
    pub radius: f64,
}

impl fmt::Display for Hypersphere {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{center: {}\nradius: {}}}", self.center, self.radius)
    }
}

impl Hypersphere {
    /// Constructs a hypersphere with a given dimension and radius, centered at
    /// the origin.
    pub fn with_radius(dim: usize, radius: f64) -> Hypersphere {
        Hypersphere {
            center: vec![0.0; dim].into(),
            radius,
        }
    }

    /// Represents the unit hypersphere in a certain number of dimensions.
    pub fn unit(dim: usize) -> Hypersphere {
        Hypersphere::with_radius(dim, 1.0)
    }

    /// Returns whether two hyperspheres are "approximately" equal.
    /// Used for testing.
    pub fn approx(&self, sphere: &Hypersphere) -> bool {
        const EPS: f64 = 1e-9;

        (&self.center - &sphere.center).norm() < EPS && self.radius - sphere.radius < EPS
    }
}

/// Represents an (affine) subspace, passing through a given point and generated
/// by a given basis.
pub struct Subspace {
    pub basis: Vec<Point>,
    pub rank: usize,
    pub offset: Point,
}

impl Subspace {
    /// Generates a trivial subspace passing through a given point.
    pub fn new(p: Point) -> Self {
        Self {
            basis: Vec::new(),
            rank: 0,
            offset,
        }
    }

    /// Returns the number of dimensions of the ambient space. For the number of
    /// dimensions spanned by the subspace itself, use `.rank`.
    pub fn dim(&self) -> usize {
        self.offset.nrows()
    }

    /// Returns whether the subspace is actually a hyperplane, i.e. a subspace
    /// whose rank is one less than the ambient dimension.
    pub fn is_hyperplane(&self) -> bool {
        self.dim() == self.rank + 1
    }

    /// Returns whether the subspace is actually of full rank, i.e. a subspace
    /// whose rank equals the ambient dimension.
    pub fn is_full_rank(&self) -> bool {
        self.dim() == self.rank
    }

    /// Adds a point to the subspace. If the rank increases, returns a new
    /// basis vector for the subspace.
    pub fn add(&mut self, p: &Point) -> Option<Point> {
        const EPS: f64 = 1e-9;

        let mut v = p - self.project(p);
        if v.normalize_mut() > EPS {
            self.basis.push(v.clone());
            self.rank += 1;

            Some(v)
        } else {
            None
        }
    }

    /// Creates a subspace from a list of points.
    pub fn from_points(points: Vec<Point>) -> Self {
        let mut points = points.into_iter();
        let mut h = Self::new(
            points
                .next()
                .expect("A hyperplane can't be created from an empty point array!"),
        );

        for p in points {
            h.add(&p);
        }

        h
    }

    /// Projects a point onto the subspace.
    pub fn project(&self, p: &Point) -> Point {
        let p = p - &self.offset;
        let mut q = self.offset.clone();

        for b in &self.basis {
            q += b * (p.dot(b)) / b.norm_squared();
        }

        q
    }

    /// Calculates the distance from a point to the subspace.
    pub fn distance(&self, p: &Point) -> f64 {
        (p - self.project(p)).norm()
    }

    /// Computes a normal vector to the subspace, so that the specified point is
    /// left out of it. Returns `None` if the point given lies on the subspace.
    pub fn normal(&self, p: &Point) -> Option<Point> {
        (p - self.project(&p)).try_normalize(EPS)
    }
}

/// Represents an (oriented) hyperplane together with a normal vector.
pub struct Hyperplane {
    pub subspace: Subspace,
    normal: Point,
}

impl Hyperplane {
    /// Defines a new oriented hyperplane from a hyperplane and a point outside
    /// of it.
    pub fn new(subspace: Subspace, p: &Point) -> Self {
        debug_assert!(
            subspace.is_hyperplane(),
            "An oriented hyperplane needs to be defined on a hyperplane."
        );

        let normal = subspace
            .normal(p)
            .expect("Specified point not outside the hyperplane.");

        Self { subspace, normal }
    }

    /// Projects a point onto the hyperplane.
    pub fn project(&self, p: &Point) -> Point {
        self.subspace.project(p)
    }

    /// Calculates the signed distance from a point to the hyperplane. Points on
    /// the side of the hyperplane containing the vector have positive distance.
    pub fn distance(&self, p: &Point) -> f64 {
        (p - self.project(p)).dot(&self.normal)
    }

    pub fn is_outer(&self, p: &Point) -> bool {
        abs_diff_eq!(self.distance(p), 0.0, epsilon = EPS)
    }
}
