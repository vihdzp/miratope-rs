
use std::f64::consts::{PI, TAU};

use gcd::Gcd;

use super::{Concrete, convex, geometry::Point};

impl Concrete {
    /// Generates a [bowtie](https://polytope.miraheze.org/wiki/Bowtie) with
    /// given edge lengths.
    fn bowtie_vertices(mut len_a: f64, mut len_b: f64) -> Vec<Point> {
        // Guarantees len_a ≤ len_b.
        if len_a > len_b {
            std::mem::swap(&mut len_a, &mut len_b);
        }

        // The coordinates of the bowtie.
        let (a, b) = (len_a / 2.0, (len_b * len_b - len_a * len_a).sqrt() / 2.0);

        vec![
            vec![a, b].into(),
            vec![-a, -b].into(),
            vec![a, -b].into(),
            vec![-a, b].into(),
        ]
    }

    fn sreg_vertices(mut n: usize, d: usize, len_a: f64, len_b: f64) -> Vec<Point> {
        // Builds the triangle from three adjacent vertices, and finds its side
        // lengths and angles.
        //
        // This triangle has side lengths len_a, len_b, len_c, and the opposite
        // respective angles are alpha, beta, gamma.
        let sq_a = len_a * len_a;
        let sq_b = len_b * len_b;

        let gamma = PI * (1.0 - (d as f64) / (n as f64));
        let sq_c = sq_a + sq_b - 2.0 * len_a * len_b * gamma.cos();
        let len_c = sq_c.sqrt();

        let mut alpha = ((sq_b + sq_c - sq_a) / (2.0 * len_b * len_c)).acos();
        let mut beta = ((sq_c + sq_a - sq_b) / (2.0 * len_c * len_a)).acos();

        let radius = len_c / (2.0 * gamma.sin());

        // Fixes the angles in case anything goes wrong in the calculation.
        let theta = PI * (d as f64) / (n as f64);
        if alpha.is_nan() {
            alpha = theta;
        }
        if beta.is_nan() {
            beta = theta;
        }

        // We only want to generate a single component.
        n /= n.gcd(d);

        let regular = len_a == 0.0 || len_b == 0.0;
        let vertex_num = if regular { n } else { 2 * n };
        let mut vertices = Vec::with_capacity(vertex_num);

        // Adds vertices.
        let mut angle = 0f64;
        for _ in 0..n {
            if len_a != 0.0 {
                vertices.push(vec![angle.cos() * radius, angle.sin() * radius].into());
                angle += 2.0 * alpha;
            }

            if len_b != 0.0 {
                vertices.push(vec![angle.cos() * radius, angle.sin() * radius].into());
                angle += 2.0 * beta;
            }
        }

        vertices
    }

    /// Generates a semiregular polygon, with order n rotational symmetry and
    /// winding number d.
    /// Bowties correspond to shapes where `d == 0`.
    pub fn sreg_polygon(n: usize, d: usize, len_a: f64, len_b: f64) -> Concrete {
        let vertices;

        let comp_num;
        let mut comp_angle;

        // Bowties are a special case that must be considered separately.
        if d == 0 {
            vertices = Concrete::bowtie_vertices(len_a, len_b);

            comp_num = n / 2;
            comp_angle = PI / comp_num as f64;
        } else {
            vertices = Concrete::sreg_vertices(n, d, len_a, len_b);

            comp_num = n.gcd(d);
            comp_angle = 2.0 * PI / (n as f64 * comp_num as f64);

            if len_a == 0.0 || len_b == 0.0 {
                comp_angle *= 2.0;
            }
        };
        /*
        compound_from_trans(
            &Concrete::polygon(vertices),
            rotations(comp_angle, comp_num, 2),
        )*/

        todo!()
    }

    /// Generates a regular polygon with Schläfli symbol {n / d}.
    pub fn reg_polygon(n: usize, d: usize) -> Concrete {
        if d == 0 {
            panic!("Invalid parameter d = 0.")
        }

        Concrete::sreg_polygon(n, d, 1.0, 0.0)
    }

    pub fn step_prism(n: usize, rotations: &[usize]) -> Concrete {
        let dim = rotations.len() * 2;
        let mut vertices = Vec::with_capacity(n);

        for i in 0..n {
            let mut v = Vec::with_capacity(dim);

            let n = n as f64;
            for &r in rotations {
                let (x, y) = (TAU * (r * i) as f64 / n).sin_cos();

                v.push(x);
                v.push(y);
            }

            vertices.push(v.into());
        }

        convex::convex_hull(vertices)
    }
}
