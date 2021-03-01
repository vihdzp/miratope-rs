use super::super::{Point, Polytope};
use super::*;

/// Generates the vertices for a duoprism with two given vertex sets.
fn duoprism_vertices(p: &[Point], q: &[Point]) -> Vec<Point> {
    let mut vertices = Vec::with_capacity(p.len() * q.len());

    for pv in p {
        for qv in q {
            let (pv, qv) = (pv.into_iter(), qv.into_iter());
            let v = pv.chain(qv).cloned().collect::<Vec<_>>().into();

            vertices.push(v);
        }
    }

    vertices
}

/// Creates a [duoprism](https://polytope.miraheze.org/wiki/Duoprism)
/// from two given polytopes.
///
/// Duoprisms are usually defined in terms of Cartesian products, but this
/// definition only makes sense in the context of convex polytopes. For general
/// polytopes, a duoprism may be inductively built as a polytope whose facets
/// are the "products" of the elements of the first polytope times those of the
/// second, where the prism product of two points is simply the point resulting
/// from concatenating their coordinates.
pub fn duoprism(p: &Polytope, q: &Polytope) -> Polytope {
    let (p_rank, q_rank) = (p.rank(), q.rank());
    let (p_vertices, q_vertices) = (&p.vertices, &q.vertices);
    let (p_elements, q_elements) = (&p.elements, &q.elements);
    let (p_el_nums, q_el_nums) = (p.el_nums(), q.el_nums());

    let rank = p_rank + q_rank;

    if p_rank == 0 {
        return q.clone();
    }
    if q_rank == 0 {
        return p.clone();
    }

    let vertices = duoprism_vertices(&p_vertices, &q_vertices);
    let mut elements = Vec::with_capacity(rank);
    for _ in 0..rank {
        elements.push(Vec::new());
    }

    // The elements of a given rank are added in order vertex × facet, edge ×
    // ridge, ...
    //
    // el_nums[m][n] will memoize the number of elements of rank m generated
    // by these products up to those of type n-element × (m - n)-element.
    let mut el_nums = Vec::with_capacity(rank);
    for m in 0..(p_rank + 1) {
        el_nums.push(Vec::new());

        for n in 0..(q_rank + 1) {
            if m == 0 || n == q_rank {
                el_nums[m].push(0);
            } else {
                let idx = el_nums[m - 1][n + 1] + p_el_nums[m - 1] * q_el_nums[n + 1];
                el_nums[m].push(idx);
            }
        }
    }

    // Gets the index of the prism product of the i-th m-element times the j-th
    // n-element.
    let get_idx = |m: usize, i: usize, n: usize, j: usize| -> usize {
        let offset = i * q_el_nums[n] + j;

        el_nums[m][n] + offset
    };

    // For each of the element lists of p (including vertices):
    for m in 0..(p_rank + 1) {
        // For each of the element lists of q (including vertices):
        for n in 0..(q_rank + 1) {
            // We'll multiply the m-elements times the n-elements inside of this loop.

            // We already took care of vertices.
            if m == 0 && n == 0 {
                continue;
            }

            // For each m-element:
            for i in 0..p_el_nums[m] {
                // For each n-element:
                for j in 0..q_el_nums[n] {
                    let mut els = Vec::new();

                    // The prism product of the i-th m-element A and the j-th n-element B
                    // has the products of A with the facets of B and B with the facets of
                    // A as facets.

                    // Points don't have facets.
                    if m != 0 {
                        let p_els = &p_elements[m - 1];
                        let p_el = &p_els[i];

                        for &p_sub in p_el {
                            els.push(get_idx(m - 1, p_sub, n, j));
                        }
                    }

                    // Points don't have facets.
                    if n != 0 {
                        let q_els = &q_elements[n - 1];
                        let q_el = &q_els[j];

                        for &q_sub in q_el {
                            els.push(get_idx(m, i, n - 1, q_sub));
                        }
                    }

                    elements[m + n - 1].push(els);
                }
            }
        }
    }

    Polytope::new(vertices, elements)
}

/// Builds a multiprism from a set of polytopes. Internally uses [`duoprism`]
/// to do this.
pub fn multiprism(polytopes: &[&Polytope]) -> Polytope {
    let mut r = point();

    for &p in polytopes {
        r = duoprism(&p, &r);
    }

    r
}

/// Generates the vertices for a pyramid product with two given vertex sets and a given height.
fn pyramid_vertices(p: &[Point], q: &[Point], height: f64) -> Vec<Point> {
    let (p_dimension, q_dimension) = (p[0].len(), q[0].len());

    let tegum = height == 0.0;
    let dimension = p_dimension + q_dimension + tegum as usize;

    let mut vertices = Vec::with_capacity(p.len() + q.len());

    for vq in q {
        let mut v = Vec::with_capacity(dimension);
        let pad = p_dimension;

        v.resize(pad, 0.0);

        for &c in vq.iter() {
            v.push(c);
        }

        if !tegum {
            v.push(height / 2.0);
        }

        vertices.push(v.into());
    }
    for vp in p {
        let mut v = Vec::with_capacity(dimension);

        for &c in vp.iter() {
            v.push(c);
        }

        v.resize(p_dimension + q_dimension, 0.0);

        if !tegum {
            v.push(-height / 2.0);
        }

        vertices.push(v.into());
    }

    vertices
}

/// Builds a duopyramid with a given height, or a duotegum if the height is 0.
pub fn duopyramid_with_height(p: &Polytope, q: &Polytope, height: f64) -> Polytope {
    let (p_rank, q_rank) = (p.rank(), q.rank());
    let (p_vertices, q_vertices) = (&p.vertices, &q.vertices);
    let (p_elements, q_elements) = (&p.elements, &q.elements);
    let (p_el_nums, q_el_nums) = (p.el_nums(), q.el_nums());

    let tegum = height == 0.0;
    let rank = p_rank + q_rank + !tegum as usize;

    let (m_max, n_max) = (p_rank + !tegum as usize + 1, q_rank + !tegum as usize + 1);

    // The tegum product of a polytope and a point is just the polytope.
    if tegum {
        if p_rank == 0 {
            return q.clone();
        }

        if q_rank == 0 {
            return p.clone();
        }
    }

    let vertices = pyramid_vertices(&p_vertices, &q_vertices, height);
    let mut elements = Vec::with_capacity(rank);
    for _ in 0..rank {
        elements.push(Vec::new());
    }

    // The elements of a given rank are added in order nullitope × facet, vertex
    // × ridge, ...
    //
    // el_nums[m][n] will memoize the number of elements of rank m - 1
    // generated by these products up to those of type (n - 1)-element ×
    // (m - n)-element.
    let mut el_nums = Vec::with_capacity(rank);

    for m in 0..m_max {
        el_nums.push(Vec::new());

        #[allow(clippy::needless_range_loop)]
        for n in 0..n_max {
            if m == 0 || n == n_max - 1 {
                el_nums[m].push(0);
            } else {
                let p_el_num = if m == 1 { 1 } else { p_el_nums[m - 2] };
                let q_el_num = q_el_nums[n];

                let idx = el_nums[m - 1][n + 1] + p_el_num * q_el_num;
                el_nums[m].push(idx);
            }
        }
    }

    // Gets the index of the prism product of the i-th m-element times the j-th
    // n-element.
    let get_idx = |m: usize, i: usize, n: usize, j: usize| -> usize {
        let q_el_nums_n = if n == 0 { 1 } else { q_el_nums[n - 1] };
        let offset = i * q_el_nums_n + j;

        el_nums[m][n] + offset
    };

    // For each of the element lists of p (including vertices & the nullitope):
    for m in 0..m_max {
        let p_el_nums_m = if m == 0 { 1 } else { p_el_nums[m - 1] };

        // For each of the element lists of q (including vertices & the nullitope):
        for n in 0..n_max {
            let q_el_nums_n = if n == 0 { 1 } else { q_el_nums[n - 1] };

            // We'll multiply the (m - 1)-elements with the (n - 1)-elements inside of this loop.

            // We already took care of vertices.
            if m + n < 2 {
                continue;
            }

            // For each m-element:
            for i in 0..p_el_nums_m {
                // For each n-element:
                for j in 0..q_el_nums_n {
                    let mut els = Vec::new();

                    // The prism product of the i-th m-element A and the j-th n-element B
                    // has the products of A with the facets of B and B with the facets of
                    // A as facets.

                    // Nullitopes don't have facets.
                    if m != 0 {
                        if m > 1 {
                            let p_els = &p_elements[m - 2];
                            let p_el = &p_els[i];

                            for &p_sub in p_el {
                                els.push(get_idx(m - 1, p_sub, n, j));
                            }
                        }
                        // Dealing with a vertex
                        else {
                            els.push(get_idx(m - 1, 0, n, j));
                        }
                    }

                    // Nullitopes don't have facets.
                    if n != 0 {
                        if n > 1 {
                            let q_els = &q_elements[n - 2];
                            let q_el = &q_els[j];

                            for &q_sub in q_el {
                                els.push(get_idx(m, i, n - 1, q_sub));
                            }
                        }
                        // Dealing with a vertex.
                        else {
                            els.push(get_idx(m, i, n - 1, 0));
                        }
                    }

                    elements[m + n - 2].push(els);
                }
            }
        }
    }

    // Components are a special case in tegum products.
    if tegum {
        // We take special care of the components.
        // These are simply the pyramid products of the two polytopes' facets.
        // For each m-element:
        let (m, n) = (p_rank + 1, q_rank + 1);
        let (p_el_nums_m, q_el_nums_n) = (p_el_nums[m - 1], q_el_nums[n - 1]);

        // For each component of p:
        for i in 0..p_el_nums_m {
            // For each component of q:
            for j in 0..q_el_nums_n {
                let mut els = Vec::new();

                // The prism product of the i-th m-element A and the j-th n-element B
                // has the products of A with the facets of B and B with the facets of
                // A as facets.

                let (p_els, q_els) = (&p_elements[m - 2], &q_elements[n - 2]);
                let (p_el, q_el) = (&p_els[i], &q_els[j]);

                for &p_sub in p_el {
                    for &q_sub in q_el {
                        els.push(get_idx(m - 1, p_sub, n - 1, q_sub));
                    }
                }

                elements[m + n - 3].push(els);
            }
        }
    }

    Polytope::new(vertices, elements)
}

/// Creates a [duotegum](https://polytope.miraheze.org/wiki/Duotegum)
/// from two given polytopes.
///
/// Duotegums are usually defined in terms of convex hulls, but this definition
/// only makes sense in the context of convex polytopes. For general polytopes,
/// a duotegum may be inductively built as a polytope whose facets are the
/// "products" of the elements of the first polytope times those of the second,
/// where the prism product of a point and the nullitope is simply the point
/// resulting from padding its coordinates with zeros.
pub fn duotegum(p: &Polytope, q: &Polytope) -> Polytope {
    duopyramid_with_height(p, q, 0.0)
}

pub fn multitegum(polytopes: &[&Polytope]) -> Polytope {
    let mut multitegum = point();

    for &p in polytopes {
        multitegum = duotegum(p, &multitegum);
    }

    multitegum
}

pub fn duopyramid(p: &Polytope, q: &Polytope) -> Polytope {
    duopyramid_with_height(p, q, 1.0)
}

pub fn multipyramid_with_heights(polytopes: &[&Polytope], heights: &[f64]) -> Polytope {
    let mut polytopes = polytopes.iter();
    let mut r = (*polytopes.next().unwrap()).clone();

    let mut heights = heights.iter();
    for &p in polytopes {
        r = duopyramid_with_height(p, &r, *heights.next().unwrap_or(&1.0)).recenter();
    }

    r
}

pub fn multipyramid(polytopes: &[&Polytope]) -> Polytope {
    multipyramid_with_heights(polytopes, &[])
}

impl Polytope {
    /// Builds a prism from a given polytope with a given height. Internally calls
    /// [`duoprism`] with the polytope and a [`dyad`].
    pub fn prism_with_height(&self, height: f64) -> Polytope {
        duoprism(self, &dyad().scale(height))
    }

    /// Builds a prism from a given polytope with unit height. Internally calls
    /// [`prism_with_height`](`Polytope::prism_with_height`).
    pub fn prism(&self) -> Polytope {
        self.prism_with_height(1.0)
    }

    /// Builds a tegum from a given polytope with a given height. Internally calls
    /// [`duotegum`] with the polytope and a [`dyad`].
    pub fn tegum_with_height(&self, height: f64) -> Polytope {
        duotegum(self, &dyad().scale(2.0 * height))
    }

    /// Builds a tegum from a given polytope with unit height. Internally calls
    /// [`tegum_with_height`](`Polytope::tegum_with_height`).
    pub fn tegum(&self) -> Polytope {
        self.tegum_with_height(1.0)
    }

    pub fn pyramid_with_height(&self, h: f64) -> Polytope {
        let point = point();

        duopyramid_with_height(self, &point, h)
    }

    pub fn pyramid(&self) -> Polytope {
        self.pyramid_with_height(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::geometry::Hypersphere;
    use super::{test_circumsphere, test_el_nums, test_equilateral};

    #[test]
    /// Checks a pentagonal prism.
    fn hip() {
        let peg = super::reg_polygon(5, 1);
        let pip = peg.prism();

        test_el_nums(&pip, vec![10, 15, 7, 1]);
        test_equilateral(&pip, 1.0);
        test_circumsphere(&pip, &Hypersphere::with_radius(3, 0.986715155325983));
    }

    #[test]
    /// Checks a triangular-pentagonal duoprism.
    fn trapedip() {
        let trig = super::reg_polygon(3, 1);
        let peg = super::reg_polygon(5, 1);
        let trapedip = super::duoprism(&trig, &peg);

        test_el_nums(&trapedip, vec![15, 30, 23, 8, 1]);
        test_equilateral(&trapedip, 1.0);
        test_circumsphere(&trapedip, &Hypersphere::with_radius(4, 1.02807593643821));
    }

    #[test]
    /// Checks a triangular trioprism.
    fn trittip() {
        let trig = super::reg_polygon(3, 1);
        let trittip = super::multiprism(&[&trig; 3]);

        test_el_nums(&trittip, vec![27, 81, 108, 81, 36, 9, 1]);
        test_equilateral(&trittip, 1.0);
        test_circumsphere(&trittip, &Hypersphere::with_radius(6, 1.0));
    }

    #[test]
    /// Checks a pentagonal bipyramid.
    fn pedpy() {
        let peg = super::reg_polygon(5, 1);
        let height = ((5.0 - 5f64.sqrt()) / 10.0).sqrt();
        let pedpy = peg.tegum_with_height(height);

        test_el_nums(&pedpy, vec![7, 15, 10, 1]);
        test_equilateral(&pedpy, 1.0);
    }

    #[test]
    /// Checks a triangular-pentagonal duotegum.
    fn trapedit() {
        let trig = super::reg_polygon(3, 1);
        let peg = super::reg_polygon(5, 1);
        let trapedit = super::duotegum(&trig, &peg);

        test_el_nums(&trapedit, vec![8, 23, 30, 15, 1]);
    }

    #[test]
    /// Checks a triangular triotegum.
    fn trittit() {
        let trig = super::reg_polygon(3, 1);
        let trittit = super::multitegum(&[&trig; 3]);

        test_el_nums(&trittit, vec![9, 36, 81, 108, 81, 27, 1]);
    }

    #[test]
    /// Checks a pentagonal pyramid.
    fn peppy() {
        let peg = super::reg_polygon(5, 1);
        let height = ((5.0 - 5f64.sqrt()) / 10.0).sqrt();
        let peppy = peg.pyramid_with_height(height);

        test_el_nums(&peppy, vec![6, 10, 6, 1]);
        test_equilateral(&peppy, 1.0);
    }

    #[test]
    /// Checks a triangular-pentagonal duopyramid.
    fn trapdupy() {
        let trig = super::reg_polygon(3, 1);
        let peg = super::reg_polygon(5, 1);
        let trapdupy = super::duopyramid(&trig, &peg);

        test_el_nums(&trapdupy, vec![8, 23, 32, 23, 8, 1]);
    }

    #[test]
    /// Checks a triangular triopyramid.
    fn tritippy() {
        let trig = super::reg_polygon(3, 1);
        let tritippy = super::multipyramid_with_heights(&[&trig; 3], &[1.0 / 3f64.sqrt(), 0.5]);

        test_el_nums(&tritippy, vec![9, 36, 84, 126, 126, 84, 36, 9, 1]);
        test_equilateral(&tritippy, 1.0);
    }
}
