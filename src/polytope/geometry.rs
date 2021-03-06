//! Contains a few structs and methods to faciliate geometry in n-dimensional
//! space.

pub type Point = nalgebra::DVector<f64>;
pub type Matrix = nalgebra::DMatrix<f64>;

use std::fmt;

#[derive(Debug)]
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

pub struct Hyperplane {
    pub basis: Vec<Point>,
    pub rank: usize,
    pub offset: Point,
}

impl Hyperplane {
    /// Generates a new hyperplane, passing through a given point.
    pub fn new(offset: Point) -> Hyperplane {
        Hyperplane {
            basis: Vec::new(),
            rank: 0,
            offset,
        }
    }

    /// Adds a point to the hyperplane. If the rank increases, returns a new
    /// basis vector for the hyperplane.
    pub fn add(&mut self, p: Point) -> Option<Point> {
        const EPS: f64 = 1e-9;

        let mut v = &p - self.project(&p);
        if v.normalize_mut() > EPS {
            self.basis.push(v.clone());
            self.rank += 1;

            return Some(v);
        }

        None
    }

    /// Creates a hyperplane from a vector of points.
    pub fn from_points(points: Vec<Point>) -> Hyperplane {
        let mut points = points.into_iter();
        let mut h = Hyperplane::new(
            points
                .next()
                .expect("A hyperplane can't be created from an empty point array!"),
        );

        for p in points {
            h.add(p);
        }

        h
    }

    /// Projects a [`Point`] onto the hyperplane.
    pub fn project(&self, p: &Point) -> Point {
        let p = p - &self.offset;
        let mut q = self.offset.clone();

        for b in &self.basis {
            q += b * (p.dot(b)) / b.norm_squared();
        }

        q
    }

    /// Calculates the distance from a [`Point`] to the hyperplane.
    pub fn distance(&self, p: &Point) -> f64 {
        (p - self.project(p)).norm()
    }
}
