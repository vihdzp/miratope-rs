use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use crate::{
    geometry::{Point, Subspace},
    polytope::{
        concrete::Concrete,
        r#abstract::elements::SubelementList,
        r#abstract::{
            elements::{Element, ElementList, Subelements, Subsupelements},
            rank::Rank,
            Abstract,
        },
    },
    Consts, Float,
};

use nalgebra::DMatrix;
use rand::Rng;
use scapegoat::SGSet as BTreeSet;

#[derive(PartialEq, Debug)]
enum Sign {
    Positive,
    Zero,
    Negative,
}

/// Represents a vertex set in a convex hull.
#[derive(PartialEq, PartialOrd, Eq, Ord, Clone, Debug)]
struct VertexSet {
    /// The sorted indices of the vertices on the ridge.
    vertices: Vec<usize>,

    /// The vertices in each ridge are oriented. The fact that we need to order
    /// them makes this inconvenient.
    ///
    /// This variable stores whether two vertices have to be swapped to have the
    /// correct orientation.
    orientation: bool,
}

/// Determines whether two permutations of a vector have different parities.
/// Uses cycle sort to sort a vector into the other, and records the parity of
/// the number of swaps.
///
/// Has O(n) complexity.
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

impl VertexSet {
    /// Sorts the vertices and updates the orientation accordingly.
    fn sort(&mut self) {
        let old_vertices = self.vertices.clone();
        self.vertices.sort_unstable();

        self.orientation ^= parity(old_vertices, &self.vertices);
    }
}

/// Returns the sign of a number, up to some imprecision.
fn sign(x: Float) -> Sign {
    if x > Float::EPS {
        Sign::Positive
    } else if x < -Float::EPS {
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

/// Finds a single ridge on the convex hull. As a side effect, sorts the
/// vertices of the polytope lexicographically.
fn get_hull_ridge(vertices: &mut [Point]) -> VertexSet {
    // Sorts the vertices lexicographically, by turning them into vecs and
    // applying their ordering.
    vertices.sort_unstable_by(|a, b| {
        a.iter()
            .cloned()
            .collect::<Vec<_>>()
            .partial_cmp(&b.iter().cloned().collect::<Vec<_>>())
            .expect("NaN vertex found!")
    });

    let mut vertices = vertices.iter().enumerate();
    let (_, v0) = vertices.next().unwrap();
    let dim = v0.len();
    let mut h = Subspace::new(v0.clone());

    let mut ridge = vec![0];

    // Takes the first `dim - 1` points in lexicographic order, so that no three
    // are collinear, no four are coplanar... these ought to create a ridge.
    while h.rank() != dim - 2 {
        let (i, v) = vertices
            .next()
            .expect("Polytope has higher dimension than rank!");

        if h.add(&v).is_some() {
            ridge.push(i);
        }
    }

    // The starting orientation is irrelevant.
    VertexSet {
        vertices: ridge,
        orientation: true,
    }
}

/// Finds the index of the closest leftmost vertex relative to a ridge.
/// "Leftmost" will depend on the orientation of the ridge.
fn leftmost_vertex(vertices: &[Point], ridge: &VertexSet) -> usize {
    debug_assert!(is_sorted(&ridge.vertices));

    let mut leftmost_vertex;
    let mut facet: Vec<_> = ridge.vertices.iter().map(|&idx| &vertices[idx]).collect();

    let mut vertex_iter = vertices.iter().enumerate();

    // We find a starting vertex not on the ridge, and add it to the facet.
    let mut h = Subspace::from_point_refs(&facet.to_vec());
    loop {
        let (i, v0) = vertex_iter.next().expect("All points coplanar!");

        // The ridge should be sorted, so we can optimize this.
        if h.add(&v0).is_some() {
            facet.push(v0);
            leftmost_vertex = Some(i);

            break;
        }
    }

    // The previous to last vertex on the facet will always be one of the
    // leftmost vertices found so far.

    // We compare with all of the other vertices.
    for (i, v) in vertex_iter {
        facet.push(v);

        // If the new vertex is to the left of the previous leftmost one:
        if sign_hypervolume(&facet, ridge.orientation) == Sign::Positive {
            // Resets leftmost vertex.
            leftmost_vertex = Some(i);

            // Adds new leftmost to the facet.
            let len = facet.len();
            facet.swap(len - 2, len - 1);
        }

        facet.pop();
    }

    let leftmost_vertex = leftmost_vertex.expect("No leftmost vertex!");
    debug_assert!(
        !ridge.vertices.contains(&leftmost_vertex),
        "New vertex in old ridge!"
    );
    leftmost_vertex
}

/// Gets the new ridges that have to be searched, in the correct orientation.
fn get_new_ridges(old_ridge: &VertexSet, new_vertex: usize) -> Vec<VertexSet> {
    let len = old_ridge.vertices.len();
    let mut new_ridges = Vec::with_capacity(len);

    // We simply add the new vertex in each position of the ridge.
    // These ridges should have the correct orientation.
    for k in 0..len {
        let mut new_ridge = old_ridge.clone();
        new_ridge.vertices[k] = new_vertex;
        new_ridge.sort();

        new_ridges.push(new_ridge);
    }

    new_ridges
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
fn common(el0: &[usize], el1: &[usize]) -> Subelements {
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

    Subelements(common)
}

/// Checks whether a given vertex set actually generates a valid d-polytope.
fn check_subelement(vertices: &[Point], el: &Subelements, rank: Rank) -> bool {
    let rank = rank.usize();

    // A d-element must have at least d + 1 vertices.
    if el.len() < rank + 1 {
        return false;
    }

    // It is possible for two d-elements to share more than d
    // elements without them being a common (d - 1)-element, but
    // only when d >= 4.
    if rank >= 4 {
        // The hyperplane of the intersection of the elements.
        let h =
            Subspace::from_point_refs(&el.iter().map(|&sub| &vertices[sub]).collect::<Vec<_>>());

        // If this hyperplane does not have the correct dimension, it
        // can't actually be a subelement.
        if h.rank() != rank - 1 {
            return false;
        }
    }

    true
}

/// Gift wrapping is only able to find the vertices of the facets of the polytope.
/// This function retrieves all other elements from them.
fn get_polytope_from_facets(vertices: Vec<Point>, facets: ElementList) -> Concrete {
    let dim = vertices[0].len();

    // Initializes the abstract polytope.
    let mut abs = Abstract::with_capacity(Rank::new(dim as isize));
    abs.push_empty();
    for _ in 0..(dim as isize) {
        abs.push_subs(SubelementList::new());
    }

    // Adds everything else.
    let mut els_verts = facets;

    for r in Rank::range_iter(Rank::new(2), Rank::new(dim as isize)).rev() {
        let mut subs_map = HashMap::new();
        let len = els_verts.len();

        // Each element of `els_verts` contains the indices of the vertices,
        // and not the subelements of the element. This vector fixes that.
        let mut els_subs = ElementList::with_capacity(len);

        for _ in 0..len {
            els_subs.push(Element::new());
        }

        // Checks every pair of d-elements to see if their intersection forms
        // a (d - 1)-element.
        for i in 0..(len - 1) {
            for j in (i + 1)..len {
                // The intersection of the two elements.
                let el = common(&els_verts[i].subs.0, &els_verts[j].subs.0);

                // Checks that el actually has the correct rank.
                if !check_subelement(&vertices, &el, r.minus_one()) {
                    continue;
                }

                match subs_map.get(&el) {
                    Some(&idx) => {
                        if !els_subs[i].subs.contains(&idx) {
                            els_subs[i].subs.push(idx);
                        }

                        if !els_subs[j].subs.contains(&idx) {
                            els_subs[j].subs.push(idx);
                        }
                    }
                    None => {
                        let idx = subs_map.len();

                        els_subs[i].subs.push(idx);
                        els_subs[j].subs.push(idx);

                        subs_map.insert(el, idx);
                    }
                }
            }
        }

        els_verts = ElementList::new();
        els_verts.resize(subs_map.len(), Element::new());

        for (subs, idx) in subs_map {
            els_verts[idx].subs = subs;
        }

        abs[r] = els_subs;
    }

    // At this point, els_verts contains, for each edge, the indices of its
    // vertices, which are precisely the rank 1 elements of the polytope.
    abs[Rank::new(1)] = els_verts;

    // Adds the vertices and the maximal element. THIS LINE DOES NOT WORK!
    // abs[Rank::new(0)] = ElementList::vertices(vertices.len());
    abs.push_max();

    Concrete::new(vertices, abs)
}

fn perturb(v: &Point) -> Point {
    const PERTURB: Float = 1e-4;

    let dim = v.nrows();
    let mut rng = rand::thread_rng();

    let pert: Point = (0..dim)
        .into_iter()
        .map(|_| rng.gen::<Float>() * PERTURB)
        .collect::<Vec<_>>()
        .into();

    v + pert
}

/// Builds the convex hull of a set of vertices. Uses the gift wrapping algorithm.
pub fn convex_hull(mut vertices: Vec<Point>) -> Concrete {
    let mut facets = HashSet::new();
    let mut ridges = BTreeSet::new();

    // Gets first ridge, reorders elements in the process.
    ridges.insert(get_hull_ridge(&mut vertices));

    // Perturbs each point randomly.
    let vertices_pert = vertices.iter().map(&perturb).collect::<Vec<_>>();

    // While there's still a ridge we need to check...
    while let Some(old_ridge) = ridges.pop_first() {
        let new_vertex = leftmost_vertex(&vertices_pert, &old_ridge);
        let new_ridges = get_new_ridges(&old_ridge, new_vertex);

        let mut facet = Element::from_subs(Subelements(old_ridge.vertices.clone()));
        facet.subs.push(new_vertex);
        facet.subs.sort();

        // We skip the facet if it isn't new.
        if facets.contains(&facet) {
            continue;
        }

        // For each new ridge:
        for new_ridge in new_ridges {
            // If this is the second time we find this ridge, it means we've
            // already added both of the facets corresponding to it.
            if ridges.contains(&new_ridge) {
                ridges.remove(&new_ridge);
            }
            // Else, we still need to find the other facet.
            else {
                ridges.insert(new_ridge);
            }
        }

        facets.insert(facet);
    }

    get_polytope_from_facets(vertices, ElementList(facets.into_iter().collect()))
}

impl Concrete {
    pub fn convex_hull(&self) -> Concrete {
        convex_hull(self.vertices.clone())
    }
}
