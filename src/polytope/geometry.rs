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
    points: Vec<Point>,
}

impl Hyperplane {
    pub fn new(p: Point) -> Hyperplane {
        Hyperplane {
            basis: Vec::new(),
            rank: 0,
            points: vec![p],
        }
    }

    /// Adds a point to the hyperplane. Returns whether the rank increased or not.
    pub fn add(&mut self, p: &Point) -> Option<Point> {
        const EPS: f64 = 1e-9;

        let mut v = p - self.project(p);
        if v.normalize_mut() > EPS {
            self.points.push(p.clone());
            self.basis.push(v.clone());
            self.rank += 1;

            return Some(v);
        }

        None
    }

    pub fn from_points(points: &[Point]) -> Hyperplane {
        let mut points = points.iter();
        let mut h = Hyperplane::new(
            points
                .next()
                .expect("A hyperplane can't be created from an empty point array!")
                .clone(),
        );

        for p in points {
            h.add(p);
        }

        h
    }

    /// Every hyperplane can be represented as a set of linear combinations,
    /// offset by some vector. This function returns any such vector.
    pub fn offset(&self) -> &Point {
        debug_assert!(
            !self.points.is_empty(),
            "A hyperplane can't contain no points!"
        );

        &self.points[0]
    }

    /// Projects a [`Point`] onto the hyperplane.
    pub fn project(&self, p: &Point) -> Point {
        let offset = self.offset();
        let p = p - offset;
        let mut q = offset.clone();

        for b in &self.basis {
            q += b * (p.dot(b)) / b.norm_squared();
        }

        q
    }
}
