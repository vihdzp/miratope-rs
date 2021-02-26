use gcd::Gcd;
use std::f64::consts::PI as PI64;

use super::super::{Point, Polytope};
use super::*;

/// Generates the unique 0D polytope.
pub fn point() -> Polytope {
    let vertices = vec![vec![].into()];
    let elements = vec![];

    Polytope::new(vertices, elements)
}

/// Generates a dyad, the unique non-compound 1D polytope.
pub fn dyad() -> Polytope {
    let vertices = vec![vec![-0.5].into(), vec![0.5].into()];
    let elements = vec![vec![vec![0, 1]]];

    Polytope::new(vertices, elements)
}

/// Builds a polygon from the vertices in order.
pub fn polygon(vertices: Vec<Point>) -> Polytope {
    let n = vertices.len();
    let mut edges = Vec::with_capacity(n);
    let mut component = Vec::with_capacity(n);

    for k in 0..n {
        edges.push(vec![k, (k + 1) % n]);
        component.push(k);
    }

    Polytope::new(vertices, vec![edges, vec![component]])
}

/// Generates a semiregular polygon, with order n rotational symmetry and winding number d.
/// Bowties correspond to shapes where `d == 0`.
pub fn sreg_polygon(mut n: usize, d: usize, mut len_a: f64, mut len_b: f64) -> Polytope {
    let comp_num;
    let comp_angle;

    let regular = len_a == 0.0 || len_b == 0.0;
    let vertex_num = if regular { n } else { 2 * n };
    let mut vertices = Vec::with_capacity(vertex_num);

    // Bowties are a special case that must be considered separately.
    if d == 0 {
        if len_a > len_b {
            std::mem::swap(&mut len_a, &mut len_b);
        }

        let (a, b) = (len_a / 2.0, (len_b * len_b - len_a * len_a).sqrt() / 2.0);

        vertices = vec![
            vec![a, b].into(),
            vec![-a, -b].into(),
            vec![a, -b].into(),
            vec![-a, b].into(),
        ];

        comp_num = n / 2;
        comp_angle = PI64 / comp_num as f64;
    } else {
        // Builds the triangle from three adjacent vertices, and finds its side lengths and angles.
        let gamma = PI64 * (1.0 - (2.0 * d as f64) / (n as f64));
        let len_c = (len_a * len_a + len_b * len_b - 2.0 * len_a * len_b * gamma.cos()).sqrt();
        let mut alpha =
            ((len_b * len_b + len_c * len_c - len_a * len_a) / (2.0 * len_b * len_c)).acos();
        let mut beta =
            ((len_c * len_c + len_a * len_a - len_b * len_b) / (2.0 * len_c * len_a)).acos();
        let radius = gamma / (2.0 * gamma.sin());

        // Fixes the angles in case anything goes wrong in the calculation.
        let theta = 2.0 * PI64 * (d as f64) / (n as f64);
        if alpha.is_nan() {
            alpha = theta;
        }
        if beta.is_nan() {
            beta = theta;
        }

        comp_num = n.gcd(d);
        n /= comp_num;

        // Adds vertices.
        let mut angle = 0f64;
        for _ in 0..n {
            if len_a != 0.0 {
                vertices.push(vec![angle.cos() * radius, angle.sin() * radius].into());
                angle += alpha;
            }

            if len_b != 0.0 {
                vertices.push(vec![angle.cos() * radius, angle.sin() * radius].into());
                angle += beta;
            }
        }

        comp_angle = 2.0 * PI64 / (n as f64 * comp_num as f64);
    }

    compound_from_trans(&polygon(vertices), rotations(comp_angle, comp_num, 2))
}

/// Generates a regular polygon with Schläfli symbol {n / d}.
pub fn reg_polygon(n: usize, d: usize) -> Polytope {
    if d == 0 {
        panic!("Invalid parameter d = 0.")
    }

    sreg_polygon(n, d, 1.0, 0.0)
}

/// Creates an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
/// with unit base edge length and a given height.
pub fn antiprism_with_height(mut n: usize, d: usize, height: f64) -> Polytope {
    let component_num = n.gcd(d);
    let theta = PI64 / (n as f64) * (d as f64);
    n /= component_num;
    let r = theta.sin() * 2.0;
    let mut height = height / 2.0;

    let mut vertices = Vec::with_capacity(2 * n);
    let mut edges = Vec::with_capacity(4 * n);
    let mut faces = Vec::with_capacity(2 * n + 2);
    let mut components = vec![Vec::with_capacity(2 * n + 2)];

    for k in 0..(2 * n) {
        // Generates vertices.
        let angle = (k as f64) * theta;
        vertices.push(vec![angle.cos() / r, angle.sin() / r, height].into());
        height *= -1.0;

        // Generates edges.
        edges.push(vec![k, (k + 1) % (2 * n)]);
        edges.push(vec![k, (k + 2) % (2 * n)]);

        // Generates faces.
        faces.push(vec![2 * k, 2 * k + 1, (2 * k + 2) % (4 * n)]);

        // Generates component.
        components[0].push(k);
    }

    let (mut base1, mut base2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for k in 0..n {
        base1.push(4 * k + 1);
        base2.push(4 * k + 3);
    }
    faces.push(base1);
    faces.push(base2);

    components[0].push(2 * n);
    components[0].push(2 * n + 1);

    // Compounds of antiprisms with antiprismatic symmetry must be handled
    // differently than compounds of antiprisms with prismatic symmetry.
    let angle = theta * (d / component_num % 2 + 1) as f64;
    compound_from_trans(
        &Polytope::new(vertices, vec![edges, faces, components]),
        rotations(angle / component_num as f64, component_num, 3),
    )
}

/// Creates a uniform [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
/// with unit edge length.
pub fn antiprism(n: usize, d: usize) -> Polytope {
    let angle = PI64 / (n as f64) * (d as f64);
    let x = 2.0 * angle.cos();
    let height = ((1.0 + x) / (2.0 + x)).sqrt();

    if height.is_nan() {
        panic!("Uniform antiprism could not be built from these parameters.");
    }

    antiprism_with_height(n, d, height)
}

pub fn simplex(d: usize) -> Polytope {
    let mut simplex = point();
    let point = point();

    for n in 1..(d + 1) {
        let n = n as f64;
        let height = ((n + 1.0) / (2.0 * n)).sqrt();

        simplex = duopyramid_with_height(&simplex, &point, height).recenter();
    }

    simplex
}

pub fn hypercube(d: usize) -> Polytope {
    let dyad = dyad();

    multiprism(&vec![&dyad; d])
}

pub fn orthoplex(d: usize) -> Polytope {
    let dyad = dyad();

    multitegum(&vec![&dyad; d])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Used to test a particular polytope.
    /// We assume that the polytope has no hemi facets.
    fn test_shape(mut p: Polytope, mut el_nums: Vec<usize>) {
        // Checks that element counts match up.
        assert_eq!(p.el_nums(), el_nums);

        // Checks that the dual element counts match up as well.
        let len = el_nums.len();
        p = p.dual();
        el_nums[0..len - 1].reverse();
        assert_eq!(p.el_nums(), el_nums);
    }

    #[test]
    /// Checks the element counts of a point.
    fn point_nums() {
        test_shape(point(), vec![1]);
    }

    #[test]
    /// Checks the element counts of a dyad.
    fn dyad_nums() {
        test_shape(dyad(), vec![2, 1]);
    }

    #[test]
    /// Checks the element counts of a few regular polygons.
    fn reg_polygon_nums() {
        test_shape(reg_polygon(5, 1), vec![5, 5, 1]);
        test_shape(reg_polygon(7, 2), vec![7, 7, 1]);
        test_shape(reg_polygon(6, 2), vec![6, 6, 2]);
    }

    #[test]
    /// Checks the element counts of a few regular polygons.
    fn sreg_polygon_nums() {
        test_shape(sreg_polygon(5, 1, 1.0, 1.0), vec![10, 10, 1]);
        test_shape(sreg_polygon(7, 2, 1.0, 1.0), vec![14, 14, 1]);
        test_shape(sreg_polygon(6, 2, 1.0, 1.0), vec![12, 12, 2]);
    }

    #[test]
    /// Checks the element counts of a tetrahedron.
    fn tet_nums() {
        test_shape(simplex(3), vec![4, 6, 4, 1]);
    }

    #[test]
    /// Checks the element counts of a cube.
    fn cube_nums() {
        test_shape(hypercube(3), vec![8, 12, 6, 1]);
    }

    #[test]
    /// Checks the element counts of an octahedron.
    fn oct_nums() {
        test_shape(orthoplex(3), vec![6, 12, 8, 1]);
    }

    #[test]
    /// Checks the element counts of a few antiprisms.
    fn antiprism_nums() {
        test_shape(antiprism(5, 1), vec![10, 20, 12, 1]);
        test_shape(antiprism(7, 2), vec![14, 28, 16, 1]);
        test_shape(antiprism(6, 2), vec![12, 24, 16, 2]);
    }

    #[test]
    #[should_panic(expected = "Facet passes through the dual center.")]
    fn bowtie_dual() {
        let bowtie = sreg_polygon(2, 0, 2.0, 1.0);

        bowtie.dual();
    }

    #[test]
    /// Checks the element counts of a hexagonal prism.
    fn hip_nums() {
        let hig = reg_polygon(6, 1);
        let hip = hig.prism();

        test_shape(hip, vec![12, 18, 8, 1]);
    }

    #[test]
    /// Checks the element counts of a triangular-pentagonal duoprism.
    fn trapedip_nums() {
        let trig = reg_polygon(3, 1);
        let peg = reg_polygon(5, 1);
        let trapedip = duoprism(&trig, &peg);

        test_shape(trapedip, vec![15, 30, 23, 8, 1]);
    }

    #[test]
    /// Checks the element num of a triangular trioprism.
    fn trittip_nums() {
        let trig = reg_polygon(3, 1);
        let trittip = multiprism(&[&trig; 3]);

        test_shape(trittip, vec![27, 81, 108, 81, 36, 9, 1]);
    }

    #[test]
    /// Checks the element counts of a hexagonal bipyramid.
    fn hib_nums() {
        let hig = reg_polygon(6, 1);
        let hib = hig.tegum();

        test_shape(hib, vec![8, 18, 12, 1]);
    }

    #[test]
    /// Checks the element num of a triangular-pentagonal duotegum.
    fn trapedit_nums() {
        let trig = reg_polygon(3, 1);
        let peg = reg_polygon(5, 1);
        let trapedit = duotegum(&trig, &peg);

        test_shape(trapedit, vec![8, 23, 30, 15, 1]);
    }

    #[test]
    /// Checks the element num of a triangular triotegum.
    fn trittit_nums() {
        let trig = reg_polygon(3, 1);
        let trittit = multitegum(&[&trig; 3]);

        test_shape(trittit, vec![9, 36, 81, 108, 81, 27, 1]);
    }

    #[test]
    /// Checks the element num of a triangular-pentagonal duopyramid.
    fn trapdupy_nums() {
        let trig = reg_polygon(3, 1);
        let peg = reg_polygon(5, 1);
        let trapdupy = duopyramid(&trig, &peg);

        test_shape(trapdupy, vec![8, 23, 32, 23, 8, 1]);
    }

    #[test]
    /// Checks the element num of a triangular triopyramid.
    fn tritippy_nums() {
        let trig = reg_polygon(3, 1);
        let tritippy = multipyramid(&[&trig; 3]);

        test_shape(tritippy, vec![9, 36, 84, 126, 126, 84, 36, 9, 1]);
    }

    #[test]
    fn cube_element() {
        let cube = hypercube(3);
        let square = cube.get_element(2, 4);

        assert_eq!(square.el_nums(), vec![4, 4, 1]);
    }
}
