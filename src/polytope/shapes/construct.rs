use gcd::Gcd;
use std::f64::consts::{PI as PI64, SQRT_2, TAU as TAU64};

use super::super::{convex, geometry::Point, Polytope};
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

/// Generates a [bowtie](https://polytope.miraheze.org/wiki/Bowtie) with given
/// edge lengths.
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

    let gamma = PI64 * (1.0 - (d as f64) / (n as f64));
    let sq_c = sq_a + sq_b - 2.0 * len_a * len_b * gamma.cos();
    let len_c = sq_c.sqrt();

    let mut alpha = ((sq_b + sq_c - sq_a) / (2.0 * len_b * len_c)).acos();
    let mut beta = ((sq_c + sq_a - sq_b) / (2.0 * len_c * len_a)).acos();

    let radius = len_c / (2.0 * gamma.sin());

    // Fixes the angles in case anything goes wrong in the calculation.
    let theta = PI64 * (d as f64) / (n as f64);
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

/// Generates a semiregular polygon, with order n rotational symmetry and winding number d.
/// Bowties correspond to shapes where `d == 0`.
pub fn sreg_polygon(n: usize, d: usize, len_a: f64, len_b: f64) -> Polytope {
    let vertices;

    let comp_num;
    let comp_angle;

    // Bowties are a special case that must be considered separately.
    if d == 0 {
        vertices = bowtie_vertices(len_a, len_b);

        comp_num = n / 2;
        comp_angle = PI64 / comp_num as f64;
    } else {
        vertices = sreg_vertices(n, d, len_a, len_b);

        comp_num = n.gcd(d);
        comp_angle = 2.0 * PI64 / (n as f64 * comp_num as f64);
    };

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
    let mut component = Vec::with_capacity(2 * n + 2);

    // Goes through the vertices in the ring of triangles in order.
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
        component.push(k);
    }

    // Adds the bases.
    let (mut base1, mut base2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for k in 0..n {
        base1.push(4 * k + 1);
        base2.push(4 * k + 3);
    }
    faces.push(base1);
    faces.push(base2);

    component.push(2 * n);
    component.push(2 * n + 1);

    // Compounds of antiprisms with antiprismatic symmetry must be handled
    // differently than compounds of antiprisms with prismatic symmetry.
    let angle = theta * (d / component_num % 2 + 1) as f64;

    let components = vec![component];
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

/// Creates a regular [simplex](https://polytope.miraheze.org/wiki/Simplex) with
/// unit edge length.
pub fn simplex(d: usize) -> Polytope {
    let point = point();
    let mut heights = Vec::with_capacity(d - 1);

    for n in 1..(d + 1) {
        let n = n as f64;
        heights.push(((n + 1.0) / (2.0 * n)).sqrt());
    }

    multipyramid_with_heights(&vec![&point; d + 1], &heights)
}

/// Creates a regular [hypercube](https://polytope.miraheze.org/wiki/Hypercube)
/// with unit edge length.
pub fn hypercube(d: usize) -> Polytope {
    let dyad = dyad();

    multiprism(&vec![&dyad; d])
}

/// Creates a regular [orthoplex](https://polytope.miraheze.org/wiki/Orthoplex)
/// with unit edge length.
pub fn orthoplex(d: usize) -> Polytope {
    let dyad = dyad().scale(SQRT_2);

    multitegum(&vec![&dyad; d])
}

pub fn step_prism(n: usize, rotations: &[usize]) -> Polytope {
    let dim = rotations.len() * 2;
    let mut vertices = Vec::with_capacity(n);

    for i in 0..n {
        let mut v = Vec::with_capacity(dim);

        let n = n as f64;
        for &r in rotations {
            let (x, y) = (TAU64 * (r * i) as f64 / n).sin_cos();

            v.push(x);
            v.push(y);
        }

        vertices.push(v.into());
    }

    convex::convex_hull(vertices)
}

#[cfg(test)]
mod tests {
    use super::super::super::geometry::Hypersphere;
    use super::{test_circumsphere, test_el_nums, test_equilateral};

    #[test]
    /// Checks a point.
    fn point() {
        let point = super::point();

        test_el_nums(&point, vec![1]);
        test_equilateral(&point, 0.0);
    }

    #[test]
    /// Checks a dyad.
    fn dyad() {
        let dyad = super::dyad();

        test_el_nums(&dyad, vec![2, 1]);
        test_equilateral(&dyad, 1.0);
        test_circumsphere(&dyad, &Hypersphere::with_radius(1, 0.5))
    }

    #[test]
    /// Checks a few regular polygons.
    fn reg_polygon() {
        let star = super::reg_polygon(5, 1);
        let hag = super::reg_polygon(7, 2);
        let shig = super::reg_polygon(6, 2);

        test_el_nums(&star, vec![5, 5, 1]);
        test_el_nums(&hag, vec![7, 7, 1]);
        test_el_nums(&shig, vec![6, 6, 2]);

        test_equilateral(&star, 1.0);
        test_equilateral(&hag, 1.0);
        test_equilateral(&shig, 1.0);

        test_circumsphere(&star, &Hypersphere::with_radius(2, 0.850650808352040));
        test_circumsphere(&hag, &Hypersphere::with_radius(2, 0.639524003844966));
        test_circumsphere(&shig, &Hypersphere::with_radius(2, 0.577350269189626));
    }

    #[test]
    /// Checks a few semi-regular polygons.
    fn sreg_polygon() {
        let trunc_star = super::sreg_polygon(5, 1, 1.0, 1.0);
        let trunc_hag = super::sreg_polygon(7, 2, 1.0, 1.0);
        let trunc_shig = super::sreg_polygon(6, 2, 1.0, 1.0);

        test_el_nums(&trunc_star, vec![10, 10, 1]);
        test_el_nums(&trunc_hag, vec![14, 14, 1]);
        test_el_nums(&trunc_shig, vec![12, 12, 2]);

        test_equilateral(&trunc_star, 1.0);
        test_equilateral(&trunc_hag, 1.0);
        test_equilateral(&trunc_shig, 1.0);

        test_circumsphere(&trunc_star, &Hypersphere::with_radius(2, 1.61803398874989));
        test_circumsphere(&trunc_hag, &Hypersphere::with_radius(2, 1.15238243548124));
        test_circumsphere(&trunc_shig, &Hypersphere::with_radius(2, 1.0));
    }

    #[test]
    /// Checks a tetrahedron.
    fn tet() {
        let tet = super::simplex(3);

        test_el_nums(&tet, vec![4, 6, 4, 1]);
        test_equilateral(&tet, 1.0);
        test_circumsphere(&tet, &Hypersphere::with_radius(3, 0.612372435695794));
    }

    #[test]
    /// Checks a cube.
    fn cube() {
        let cube = super::hypercube(3);

        test_el_nums(&cube, vec![8, 12, 6, 1]);
        test_equilateral(&cube, 1.0);
        test_circumsphere(&cube, &Hypersphere::with_radius(3, 0.866025403784439));
    }

    #[test]
    /// Checks an octahedron.
    fn oct() {
        let oct = super::orthoplex(3);

        test_el_nums(&oct, vec![6, 12, 8, 1]);
        test_equilateral(&oct, 1.0);
        test_circumsphere(&oct, &Hypersphere::with_radius(3, 0.707106781186548));
    }

    #[test]
    /// Checks the element counts of a few antiprisms.
    fn antiprism() {
        let pap = super::antiprism(5, 1);
        let shap = super::antiprism(7, 2);
        let shigp = super::antiprism(6, 2); //wrong obsa lol

        test_el_nums(&pap, vec![10, 20, 12, 1]);
        test_el_nums(&shap, vec![14, 28, 16, 1]);
        test_el_nums(&shigp, vec![12, 24, 16, 2]);

        test_equilateral(&pap, 1.0);
        test_equilateral(&shap, 1.0);
        test_equilateral(&shigp, 1.0);

        test_circumsphere(&pap, &Hypersphere::with_radius(3, 0.951056516295154));
        test_circumsphere(&shap, &Hypersphere::with_radius(3, 0.762886832630778));
        test_circumsphere(&shigp, &Hypersphere::with_radius(3, 0.707106781186548));
    }

    #[test]
    #[should_panic(expected = "Facet passes through the dual center.")]
    fn bowtie() {
        let bowtie = super::sreg_polygon(2, 0, 2.0, 1.0);

        test_circumsphere(&bowtie, &Hypersphere::with_radius(2, 1.11803398874989));
        test_el_nums(&bowtie, vec![4, 4, 1]);
    }

    #[test]
    fn cube_element() {
        let cube = super::hypercube(3);
        let square = cube.get_element(2, 4);

        assert_eq!(square.el_nums(), vec![4, 4, 1]);
    }
}
