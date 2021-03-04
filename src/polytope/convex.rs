use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap},
};

use super::{
    geometry::{Hyperplane, Point},
    ElementList, Polytope,
};
use nalgebra::DMatrix;

enum Sign {
    Positive,
    Zero,
    Negative,
}

/// Represents a vertex set in a convex hull.
#[derive(PartialEq, PartialOrd, Eq, Ord, Clone)]
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

/// Determines whether two permutations of a vector have the same permutation.
/// Uses cycle sort to sort p0 into p1, and records the parity of the number of swaps.
fn parity(mut p0: Vec<usize>, p1: &[usize]) -> bool {
    let mut parity = true;
    let mut hash = HashMap::new();

    // Maps each value in p0 to its index in the array.
    for (i, val) in p1.iter().enumerate() {
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
fn sign(x: f64) -> Sign {
    const EPS: f64 = 1e-9;

    if x > EPS {
        Sign::Positive
    } else if x < EPS {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

/// Returns the sign of the hypervolume of a simplex specified by a set of
/// n-dimensional points.
fn sign_hypervolume(simplex: &[&Point]) -> Sign {
    let dim = simplex.len() - 1;
    let mut m = DMatrix::from_element(dim + 1, dim + 1, 1.0);

    debug_assert_eq!(dim, simplex[0].len());

    for (j, v) in simplex.iter().enumerate() {
        for (i, &c) in v.iter().enumerate() {
            m[(i, j)] = c;
        }
    }

    sign(m.determinant())
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
    let mut h = Hyperplane::new(v0.clone());

    let mut ridge = Vec::new();

    // Takes the first `dim` points in lexicographic order, so that no three are
    // collinear, no four are coplanar... these ought to create a ridge.
    while h.rank != dim - 1 {
        let (i, v) = vertices
            .next()
            .expect("Polytope has higher dimension than rank!");

        if h.add(v).is_some() {
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
    let mut leftmost_vertices = Vec::new();
    let mut facet: Vec<_> = ridge.vertices.iter().map(|&idx| &vertices[idx]).collect();

    let mut vertex_iter = vertices.iter().enumerate();
    let (_, v0) = vertex_iter.next().unwrap();

    // The previous to last vertex on the facet will always be one of the
    // leftmost vertices found so far.
    facet.push(v0);

    for (i, v) in vertex_iter {
        facet.push(v);

        match sign_hypervolume(&facet) {
            // If the new vertex is to the left of the previous leftmost one:
            Sign::Positive => {
                // Resets leftmost vertices.
                leftmost_vertices.clear();
                leftmost_vertices.push(i);

                // Adds new leftmost to the facet.
                let len = facet.len();
                facet.swap(len - 2, len - 1);
            }
            // If the new vertex is as left as the previous leftmost one:
            Sign::Zero => {
                leftmost_vertices.push(i);
            }
            // If the new vertex is to the right of the previous leftmost one:
            Sign::Negative => {
                break;
            }
        }

        facet.pop();
    }

    // The hyperplane of the ridge.
    let h = Hyperplane::from_points(&facet.iter().cloned().cloned().collect::<Vec<_>>());

    // From the leftmost vertices, finds the one closest to the ridge.
    *leftmost_vertices
        .iter()
        .min_by(|&&v1, &&v2| {
            h.distance(&vertices[v1])
                .partial_cmp(&h.distance(&vertices[v2]))
                .unwrap()
        })
        .unwrap()
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

fn common_subs(el0: &[usize], el1: &[usize]) -> Vec<usize> {
    // Nightly Rust!
    // debug_assert!(el0.is_sorted());
    // debug_assert!(el1.is_sorted());

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

fn check_subelement(vertices: &[Point], el: &[usize], d: usize) -> bool {
    // A (d - 1)-element must have at least d vertices.
    if el.len() < d {
        return false;
    }

    // It is possible for two d-elements to share more than d
    // elements without them being a common (d - 1)-element, but
    // only when d >= 4.
    if d >= 4 {
        // The hyperplane of the intersection of the elements.
        let h = Hyperplane::from_points(
            &el.iter()
                .map(|&sub| vertices[sub].clone())
                .collect::<Vec<_>>(),
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

    // Add a single component.
    let len = facets.len();
    let mut component = Vec::with_capacity(len);

    for i in 0..len {
        component.push(i);
    }

    elements.push(vec![component]);

    // Add everything else.
    let mut els = facets;

    for d in (1..dim).rev() {
        let mut new_subs: HashMap<Vec<usize>, usize> = HashMap::new();
        let len = els.len();
        let mut new_els = Vec::with_capacity(len);

        for _ in 0..len {
            new_els.push(vec![]);
        }

        for i in 0..(len - 1) {
            for j in (i + 1)..len {
                // The intersection of the two elements.
                let el = common_subs(&els[i], &els[j]);

                // Checks that el actually has the correct rank.
                if !check_subelement(&vertices, &el, d) {
                    break;
                }

                match new_subs.get(&el) {
                    Some(&idx) => {
                        if !new_els[i].contains(&idx) {
                            new_els[i].push(idx);
                        }

                        if !new_els[j].contains(&idx) {
                            new_els[j].push(idx);
                        }
                    }
                    None => {
                        let idx = new_subs.len();

                        new_els[i].push(idx);
                        new_els[j].push(idx);

                        new_subs.insert(el, idx);
                    }
                }
            }
        }

        els = new_els.to_vec();
        elements.push(new_els);
    }

    elements.reverse();
    Polytope::new(vertices, elements)
}

/// Builds the convex hull of a set of vertices. Uses the gift wrapping algorithm.
pub fn convex_hull(mut vertices: Vec<Point>) -> Polytope {
    let mut facets = Vec::new();
    let mut ridges = BTreeSet::new();

    // Gets first ridge, reorders elements in the process.
    ridges.insert(get_hull_ridge(&mut vertices));

    // In the future, we should replace this by ridges.pop_front().
    while let Some(old_ridge) = ridges.iter().next() {
        let new_vertex = leftmost_vertex(&vertices, &old_ridge);
        let new_ridges = get_new_ridges(old_ridge, new_vertex);

        let mut facet = old_ridge.vertices.clone();
        facet.push(new_vertex);
        facet.sort_unstable();

        // In theory, this facet should always be a new one.
        debug_assert!(!facets.contains(&facet));

        for new_ridge in new_ridges {
            // Surely there's a better way to do this: by calling `contains`, I
            // should already know where to insert or remove the ridge.

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

        facets.push(facet);
    }

    get_polytope_from_facets(vertices, facets)
}
