use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use super::{
    geometry::{Hyperplane, Point},
    ElementList, Polytope,
};

use nalgebra::DMatrix;
use scapegoat::SGSet as BTreeSet;
use std::fmt::Debug;

#[derive(PartialEq, Debug)]
enum Sign {
    Positive,
    Zero,
    Negative,
}

/// Represents a facet in a convex hull.
#[derive(Clone, Debug)]
struct Facet {
    /// The indices of the vertices on the facet.
    vertices: Vec<usize>,

    /// The indices of the ridges on the facet.
    ridges: Vec<usize>,

    /// The indices of the vertices "outside" of this facet.
    outer: Vec<usize>,

    /// A normal vector for the facet, used to determine orientation.
    normal: Point,

    /// A unique identifier.
    id: u32,
}

struct Ridge {
    /// The indices of the vertices on the ridge.
    vertices: Vec<usize>,

    /// The indices of the facets on the ridge.
    facets: Vec<usize>,

    /// A unique identifier.
    id: u32,
}

/// Determines whether two permutations of a vector have different parities.
/// Uses cycle sort to sort a vector into the other, and records the parity of
/// the number of swaps.
fn parity(mut p0: Vec<usize>, p1: &[usize]) -> bool {
    let mut parity = false;
    let mut hash = HashMap::new();

    // Maps each value in p0 to its index in the array.
    for (i, &val) in p1.iter().enumerate() {
        hash.insert(val, i);
    }

    // Cycle sort, but we use the hash table to find the correct position of
    // every element in constant time.
    for i in 0..p0.len() {
        loop {
            let j = *hash.get(&p0[i]).unwrap();

            if i == j {
                break;
            }

            p0.swap(i, j);
            parity = !parity;
        }
    }

    parity
}

fn sgn(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl Facet {
    /// Returns the clones of the vertices the facet refers to.
    fn clone_vertices(&self, vertices: &[Point]) -> Vec<Point> {
        self.vertices
            .iter()
            .map(|&idx| vertices[idx].clone())
            .collect()
    }

    /// Returns the signed distance of a point to the facet.
    fn signed_distance(&self, vertices: &[Point], p: &Point) -> f64 {
        let h = Hyperplane::from_points(self.clone_vertices(vertices));
        let v = p - h.project(&p);

        v.norm() * sgn(v.dot(&self.normal))
    }
}

fn connect(f: &mut Facet, r: &mut Ridge) {}

/// Returns the sign of a number, up to some imprecision.
fn sign(x: f64) -> Sign {
    const EPS: f64 = 1e-9;

    if x > EPS {
        Sign::Positive
    } else if x < -EPS {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

/// Returns the sign of the hypervolume of a simplex specified by a set of
/// n-dimensional points.
fn sign_hypervolume(simplex: &[&Point], orientation: bool) -> Sign {
    let dim = simplex.len() - 1;
    let mut m = DMatrix::from_element(dim + 1, dim + 1, 1.0);

    debug_assert_eq!(dim, simplex[0].len());

    for (j, v) in simplex.iter().enumerate() {
        for (i, &c) in v.iter().enumerate() {
            m[(i, j)] = c;
        }
    }

    let flip = if orientation { 1.0 } else { -1.0 };
    sign(flip * m.determinant())
}

/// Finds a single simplex on the convex hull.
fn get_hull_simplex(vertices: &[Point]) -> Facet {
    let mut vertices = vertices.iter().enumerate();
    let (_, v0) = vertices.next().unwrap();
    let dim = v0.len();
    let mut h = Hyperplane::new(v0.clone());

    let mut ridge = vec![0];

    // Takes the first `dim` points in lexicographic order, so that no three
    // are collinear, no four are coplanar... these ought to create a facet.
    while h.rank != dim - 1 {
        let (i, v) = vertices
            .next()
            .expect("Polytope has higher dimension than rank!");

        if h.add(v.clone()).is_some() {
            facet.push(i);
        }
    }

    // Getting the ridges goes here.

    // The starting orientation is irrelevant.
    Facet {
        vertices: ridge,
        orientation: true,
    }
}

/// Checks whether an array is sorted in increasing order. Used only for debugs.
fn is_sorted(el: &[usize]) -> bool {
    for i in 0..(el.len() - 1) {
        if el[i] >= el[i + 1] {
            return false;
        }
    }

    true
}

/// Finds the common elements of two arrays.
/// # Assumptions:
/// * Both arrays must be sorted in increasing order.
fn common(el0: &[usize], el1: &[usize]) -> Vec<usize> {
    // Nightly Rust has el.is_sorted().
    debug_assert!(is_sorted(el0));
    debug_assert!(is_sorted(el1));

    let mut common = Vec::new();
    let (mut i, mut j) = (0, 0);

    // Again, nightly rust would allow us to compress this.
    while let Some(&sub0) = el0.get(i) {
        if let Some(sub1) = el1.get(j) {
            match sub0.cmp(sub1) {
                Ordering::Equal => {
                    common.push(sub0);
                    i += 1;
                }
                Ordering::Greater => j += 1,
                Ordering::Less => i += 1,
            };
        } else {
            break;
        }
    }

    common
}

/// Checks whether a given vertex set actually generates a valid d-polytope.
fn check_subelement(vertices: &[Point], el: &[usize], d: usize) -> bool {
    // A d-element must have at least d + 1 vertices.
    if el.len() < d + 1 {
        return false;
    }

    // It is possible for two d-elements to share more than d
    // elements without them being a common (d - 1)-element, but
    // only when d >= 4.
    if d >= 4 {
        // The hyperplane of the intersection of the elements.
        let h = Hyperplane::from_points(
            el.iter()
                .map(|&sub| vertices[sub].clone())
                .collect::<Vec<Point>>(),
        );

        // If this hyperplane does not have the correct dimension, it
        // can't actually be a subelement.
        if h.rank != d - 1 {
            return false;
        }
    }

    true
}

/// Gift wrapping is only able to find the vertices of the facets of the polytope.
/// This function retrieves all other elements from them.
fn get_polytope_from_facets(vertices: Vec<Point>, facets: ElementList) -> Polytope {
    let dim = vertices[0].len();
    let mut elements = Vec::with_capacity(dim);

    // Adds a single component.
    let len = facets.len();
    let mut component = Vec::with_capacity(len);

    for i in 0..len {
        component.push(i);
    }

    elements.push(vec![component]);

    // Adds everything else.
    let mut els_verts = facets;

    for d in (1..(dim - 1)).rev() {
        let mut subs = HashMap::new();
        let len = els_verts.len();

        // Each element of `els_verts` contains the indices of the vertices,
        // and not the subelements of the element. This vector fixes that.
        let mut els_subs = Vec::with_capacity(len);

        for _ in 0..len {
            els_subs.push(vec![]);
        }

        // Checks every pair of d-elements to see if their intersection forms
        // a (d - 1)-element.
        for i in 0..(len - 1) {
            for j in (i + 1)..len {
                // The intersection of the two elements.
                let el = common(&els_verts[i], &els_verts[j]);

                // Checks that el actually has the correct rank.
                if !check_subelement(&vertices, &el, d) {
                    continue;
                }

                match subs.get(&el) {
                    Some(&idx) => {
                        if !els_subs[i].contains(&idx) {
                            els_subs[i].push(idx);
                        }

                        if !els_subs[j].contains(&idx) {
                            els_subs[j].push(idx);
                        }
                    }
                    None => {
                        let idx = subs.len();

                        els_subs[i].push(idx);
                        els_subs[j].push(idx);

                        subs.insert(el, idx);
                    }
                }
            }
        }

        els_verts = Vec::new();
        els_verts.resize(subs.len(), vec![]);

        for (sub, idx) in subs {
            els_verts[idx] = sub;
        }

        elements.push(els_subs);
    }

    elements.push(els_verts);
    elements.reverse();
    Polytope::new(vertices, elements)
}

/// Finds the furthest away outer vertex of a face.
fn find_eye_point(vertices: &[Point], facet: &Facet) -> usize {
    assert!(!facet.vertices.is_empty());

    let h = Hyperplane::from_points(facet.clone_vertices(vertices));

    // There's probably a one line implementation of this that is ultimately
    // clearer.
    let mut max_idx = 0;
    let mut max_distance = 0.0;
    for v in facet.outer {
        let distance = h.distance(&vertices[v]);

        if distance > max_distance {
            max_idx = v;
            max_distance = distance;
        }
    }

    max_idx
}

fn new_facets(vertices: &[Point], facet: &Facet, eye_point: usize, hull_point: &Point) {}

/// Builds the convex hull of a set of vertices. Uses the gift wrapping algorithm.
pub fn convex_hull(mut vertices: Vec<Point>) -> Polytope {
    let mut facets = HashMap::new();
    let mut ridges = HashMap::new();

    // Gets first facet.
    ridges.insert(get_hull_simplex(&mut vertices));

    // While there's still a ridge we need to check...
    while let Some(old_ridge) = ridges.iter().get(0) {
        let eye_point;
    }

    get_polytope_from_facets(vertices, facets.into_iter().collect())
}

impl Polytope {
    pub fn convex_hull(&self) -> Polytope {
        convex_hull(self.vertices.clone())
    }
}
