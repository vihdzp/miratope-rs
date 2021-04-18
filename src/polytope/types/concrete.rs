use crate::{
    polytope::{
        geometry::{Hyperplane, Hypersphere, Matrix, Point, Segment, Subspace},
        rank::RankVec,
        Abstract, Element, ElementList, Polytope, Subelements,
    },
    EPS,
};
use approx::{abs_diff_eq, abs_diff_ne};
use std::{collections::HashMap, f64::consts::SQRT_2};

#[derive(Debug, Clone)]
/// Represents a
/// [concrete polytope](https://polytope.miraheze.org/wiki/Polytope), which is
/// an [`Abstract`] together with the corresponding vertices.
pub struct Concrete {
    /// The list of vertices as points in Euclidean space.
    pub vertices: Vec<Point>,

    /// The underlying abstract polytope.
    pub abs: Abstract,
}

impl Concrete {
    pub fn new(vertices: Vec<Point>, abs: Abstract) -> Self {
        // There must be as many abstract vertices as concrete ones.
        debug_assert_eq!(vertices.len(), abs.el_count(0));

        if let Some(vertex0) = vertices.get(0) {
            for vertex1 in &vertices {
                debug_assert_eq!(vertex0.len(), vertex1.len());
            }
        }

        Self { vertices, abs }
    }

    /// Returns the rank of the polytope.
    pub fn rank(&self) -> isize {
        self.abs.rank()
    }

    /// Returns the number of dimensions of the space the polytope lives in,
    /// or `None` in the case of the nullitope.
    pub fn dim(&self) -> Option<usize> {
        Some(self.vertices.get(0)?.len())
    }

    /// Scales a polytope by a given factor.
    pub fn scale(mut self, k: f64) -> Self {
        for v in &mut self.vertices {
            *v *= k;
        }

        self
    }

    /// Shifts all vertices by a given vector.
    pub fn shift(mut self, o: Point) -> Self {
        for v in &mut self.vertices {
            *v -= &o;
        }

        self
    }

    /// Recenters a polytope so that the gravicenter is at the origin.
    pub fn recenter(self) -> Self {
        if let Some(gravicenter) = self.gravicenter() {
            self.shift(gravicenter)
        } else {
            self
        }
    }

    /// Applies a matrix to all vertices of a polytope.
    pub fn apply(mut self, m: &Matrix) -> Self {
        for v in &mut self.vertices {
            *v = m * v.clone();
        }

        self
    }

    /// Calculates the circumsphere of a polytope. Returns it if the polytope
    /// has one, and returns `None` otherwise.
    pub fn circumsphere(&self) -> Option<Hypersphere> {
        let mut vertices = self.vertices.iter();

        let v0 = vertices.next().expect("Polytope has no vertices!").clone();
        let mut o: Point = v0.clone();
        let mut h = Subspace::new(v0.clone());

        for v in vertices {
            // If the new vertex does not lie on the hyperplane of the others:
            if let Some(b) = h.add(&v) {
                // Calculates the new circumcenter.
                let k = ((&o - v).norm_squared() - (&o - &v0).norm_squared())
                    / (2.0 * (v - &v0).dot(&b));

                o += k * b;
            }
            // If the new vertex lies on the others' hyperplane, but is not at
            // the correct distance from the first vertex:
            else if abs_diff_ne!((&o - &v0).norm(), (&o - v).norm(), epsilon = EPS) {
                return None;
            }
        }

        Some(Hypersphere {
            radius: (&o - v0).norm(),
            center: o,
        })
    }

    /// Gets the gravicenter of a polytope, or `None` in the case of the
    /// nullitope.
    pub fn gravicenter(&self) -> Option<Point> {
        let mut g: Point = vec![0.0; self.dim()? as usize].into();

        for v in &self.vertices {
            g += v;
        }

        Some(g / (self.vertices.len() as f64))
    }

    /// Gets the edge lengths of all edges in the polytope, in order.
    pub fn edge_lengths(&self) -> Vec<f64> {
        let mut edge_lengths = Vec::new();

        // If there are no edges, we just return the empty vector.
        if let Some(edges) = self.abs.get(1) {
            edge_lengths.reserve_exact(edges.len());

            for edge in edges.iter() {
                let sub0 = edge.subs[0];
                let sub1 = edge.subs[1];

                edge_lengths.push((&self.vertices[sub0] - &self.vertices[sub1]).norm());
            }
        }

        edge_lengths
    }

    pub fn is_equilateral_with_len(&self, len: f64) -> bool {
        let edge_lengths = self.edge_lengths().into_iter();

        // Checks that every other edge length is equal to the first.
        for edge_len in edge_lengths {
            if abs_diff_eq!(edge_len, len, epsilon = EPS) {
                return false;
            }
        }

        true
    }

    /// Checks whether a polytope is equilateral to a fixed precision.
    pub fn is_equilateral(&self) -> bool {
        if let Some(edges) = self.abs.get(1) {
            if let Some(edge) = edges.get(0) {
                let vertices = edge
                    .subs
                    .iter()
                    .map(|&v| &self.vertices[v])
                    .collect::<Vec<_>>();
                let (v0, v1) = (vertices[0], vertices[1]);

                return self.is_equilateral_with_len((v0 - v1).norm());
            }
        }

        true
    }

    /// I haven't actually implemented this in the general case.
    ///
    /// # Todo
    /// Maybe make this work in the general case?
    pub fn midradius(&self) -> f64 {
        let vertices = &self.vertices;
        let edges = &self[0];
        let edge = &edges[0];

        let sub0 = edge.subs[0];
        let sub1 = edge.subs[1];

        (&vertices[sub0] + &vertices[sub1]).norm() / 2.0
    }

    /// Returns the dual of a polytope, or `None` if any facets pass through the
    /// origin.
    pub fn dual(&self) -> Option<Self> {
        let mut clone = self.clone();

        if clone.dual_mut().is_ok() {
            Some(clone)
        } else {
            None
        }
    }

    /// Builds the dual of a polytope in place, or does nothing in case any
    /// facets go through the origin. Returns the dual if successful, and `None`
    /// otherwise.
    pub fn dual_mut(&mut self) -> Result<(), ()> {
        self.dual_mut_with_sphere(&Hypersphere::unit(self.dim().unwrap_or(1)))
    }

    /// Returns the dual of a polytope with a given reciprocation sphere, or
    /// `None` if any facets pass through the reciprocation center.
    pub fn dual_with_sphere(&self, sphere: &Hypersphere) -> Option<Self> {
        let mut clone = self.clone();

        if clone.dual_mut_with_sphere(sphere).is_ok() {
            Some(clone)
        } else {
            None
        }
    }

    /// Builds the dual of a polytope with a given reciprocation sphere in
    /// place, or does nothing in case any facets go through the reciprocation
    /// center. Returns the dual if successful, and `None` otherwise.
    pub fn dual_mut_with_sphere(&mut self, sphere: &Hypersphere) -> Result<(), ()> {
        // If we're dealing with a nullitope, the dual is itself.
        let rank = self.rank();
        if rank == -1 {
            return Ok(());
        }
        // In the case of points, we reciprocate them.
        else if rank == 0 {
            for v in self.vertices.iter_mut() {
                if sphere.reciprocate(v).is_err() {
                    return Err(());
                }
            }
        }

        // We project the sphere's center onto the polytope's hyperplane to
        // avoid skew weirdness.
        let h = Subspace::from_points(&self.vertices);
        let o = h.project(&sphere.center);

        let mut projections;

        // We project our inversion center onto each of the facets.
        if rank >= 2 {
            let facet_count = self.el_count(rank - 1);
            projections = Vec::with_capacity(facet_count);

            for idx in 0..facet_count {
                projections.push(
                    Subspace::from_point_refs(&self.element_vertices_ref(rank - 1, idx).unwrap())
                        .project(&o),
                );
            }
        }
        // If our polytope is 1D, the vertices themselves are the facets.
        else {
            projections = self.vertices.clone();
        }

        // Reciprocates the projected points.
        for v in projections.iter_mut() {
            if sphere.reciprocate(v).is_err() {
                return Err(());
            }
        }

        self.vertices = projections;

        // Takes the abstract dual.
        self.abs.dual_mut();

        Ok(())
    }

    /// Gets the references to the (geometric) vertices of an element on the
    /// polytope.
    pub fn element_vertices_ref(&self, rank: isize, idx: usize) -> Option<Vec<&Point>> {
        Some(
            self.abs
                .element_vertices(rank, idx)?
                .iter()
                .map(|&v| &self.vertices[v])
                .collect(),
        )
    }

    /// Gets the (geometric) vertices of an element on the polytope.
    pub fn element_vertices(&self, rank: isize, idx: usize) -> Option<Vec<Point>> {
        Some(
            self.element_vertices_ref(rank, idx)?
                .into_iter()
                .cloned()
                .collect(),
        )
    }

    /// Gets an element of a polytope, as its own polytope.
    pub fn element(&self, rank: isize, idx: usize) -> Option<Self> {
        Some(Concrete {
            vertices: self.element_vertices(rank, idx)?,
            abs: self.abs.element(rank, idx)?,
        })
    }

    /// Gets the [vertex figure](https://polytope.miraheze.org/wiki/Vertex_figure)
    /// of a polytope corresponding to a given vertex.
    pub fn verf(&self, idx: usize) -> Option<Self> {
        self.dual()?.element(self.rank() - 1, idx)?.dual()
    }

    /// Generates the vertices for either a tegum or a pyramid product with two
    /// given vertex sets and a given height.
    fn duopyramid_vertices(p: &[Point], q: &[Point], height: f64, tegum: bool) -> Vec<Point> {
        let p_dim = p[0].len();
        let q_dim = q[0].len();

        let dim = p_dim + q_dim + tegum as usize;

        let mut vertices = Vec::with_capacity(p.len() + q.len());

        // The vertices corresponding to products of p's nullitope with q's
        // vertices.
        for q_vertex in q {
            let mut prod_vertex = Vec::with_capacity(dim);
            let pad = p_dim;

            // Pads prod_vertex to the left.
            prod_vertex.resize(pad, 0.0);

            // Copies q_vertex into prod_vertex.
            for &c in q_vertex.iter() {
                prod_vertex.push(c);
            }

            // Adds the height, in case of a pyramid product.
            if !tegum {
                prod_vertex.push(height / 2.0);
            }

            vertices.push(prod_vertex.into());
        }

        // The vertices corresponding to products of q's nullitope with p's
        // vertices.
        for p_vertex in p {
            let mut prod_vertex = Vec::with_capacity(dim);

            // Copies p_vertex into prod_vertex.
            for &c in p_vertex.iter() {
                prod_vertex.push(c);
            }

            // Pads prod_vertex to the right.
            prod_vertex.resize(p_dim + q_dim, 0.0);

            // Adds the height, in case of a pyramid product.
            if !tegum {
                prod_vertex.push(-height / 2.0);
            }

            vertices.push(prod_vertex.into());
        }

        vertices
    }

    /// Generates the vertices for a duoprism with two given vertex sets.
    fn duoprism_vertices(p: &[Point], q: &[Point]) -> Vec<Point> {
        let mut vertices = Vec::with_capacity(p.len() * q.len());

        // Concatenates all pairs of vertices in order.
        for p_vertex in p {
            for q_vertex in q {
                let p_vertex = p_vertex.into_iter();
                let q_vertex = q_vertex.into_iter();

                vertices.push(p_vertex.chain(q_vertex).cloned().collect::<Vec<_>>().into());
            }
        }

        vertices
    }

    /// Generates a duopyramid from two given polytopes with a given height.
    pub fn duopyramid_with_height(p: &Self, q: &Self, height: f64) -> Self {
        Self::new(
            Self::duopyramid_vertices(&p.vertices, &q.vertices, height, false),
            Abstract::duopyramid(&p.abs, &q.abs),
        )
    }

    /// Takes the cross-section of a polytope through a given hyperplane.
    ///
    /// # Todo
    /// We should make this function take a general [`Subspace`] instead.
    pub fn slice(&self, slice: Hyperplane) -> Self {
        let mut vertices = Vec::new();

        let mut abs = Abstract::new();

        // We map all indices of k-elements in the original polytope to the
        // indices of the new (k-1)-elements resulting from taking their
        // intersections with the slicing hyperplane.
        let mut hash_element = HashMap::new();

        // Determines the vertices of the cross-section.
        for (idx, edge) in self[1].iter().enumerate() {
            let segment = Segment(
                self.vertices[edge.subs[0]].clone(),
                self.vertices[edge.subs[1]].clone(),
            );

            // If we got ourselves a new vertex:
            if let Some(p) = slice.intersect(segment) {
                hash_element.insert(idx, vertices.len());
                vertices.push(slice.flatten(&p));
            }
        }

        let vertex_count = vertices.len();

        // The slice does not intersect the polytope.
        if vertex_count == 0 {
            return Self::nullitope();
        }

        abs.push(ElementList::min(vertex_count));
        abs.push(ElementList::vertices(vertex_count));

        // Takes care of building everything else.
        for r in 2..self.rank() {
            let mut new_hash_element = HashMap::new();
            let mut new_els = ElementList::new();

            for (idx, el) in self[r].iter().enumerate() {
                let mut new_subs = Vec::new();
                for sub in el.subs.iter() {
                    if let Some(&v) = hash_element.get(sub) {
                        new_subs.push(v);
                    }
                }

                // If we got ourselves a new edge:
                if !new_subs.is_empty() {
                    new_hash_element.insert(idx, new_els.len());
                    new_els.push(Element::from_subs(Subelements(new_subs)));
                }
            }

            abs.push_subs(new_els);
            hash_element = new_hash_element;
        }

        let facet_count = abs.last().unwrap().len();
        abs.push_subs(ElementList::max(facet_count));

        Self { vertices, abs }
    }
}

impl Polytope for Concrete {
    fn rank(&self) -> isize {
        self.abs.rank()
    }

    fn el_count(&self, rank: isize) -> usize {
        self.abs.el_count(rank)
    }

    fn el_counts(&self) -> RankVec<usize> {
        self.abs.el_counts()
    }

    fn nullitope() -> Self {
        Self {
            abs: Abstract::nullitope(),
            vertices: Vec::new(),
        }
    }

    fn point() -> Self {
        Self::new(vec![vec![].into()], Abstract::point())
    }

    fn dyad() -> Self {
        Self::new(vec![vec![-0.5].into(), vec![0.5].into()], Abstract::dyad())
    }

    fn polygon(n: usize) -> Self {
        Self::reg_polygon(n, 1)
    }

    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::duopyramid_with_height(p, q, 1.0)
    }

    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::new(
            Self::duoprism_vertices(&p.vertices, &q.vertices),
            Abstract::duoprism(&p.abs, &q.abs),
        )
    }

    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::new(
            Self::duopyramid_vertices(&p.vertices, &q.vertices, 0.0, true),
            Abstract::duotegum(&p.abs, &q.abs),
        )
    }

    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::new(
            Self::duoprism_vertices(&p.vertices, &q.vertices),
            Abstract::duocomb(&p.abs, &q.abs),
        )
    }

    fn ditope(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            abs: self.abs.ditope(),
        }
    }

    fn ditope_mut(&mut self) {
        self.abs.ditope_mut();
    }

    fn hosotope(&self) -> Self {
        Self {
            vertices: vec![vec![-0.5].into(), vec![0.5].into()],
            abs: self.abs.hosotope(),
        }
    }

    fn hosotope_mut(&mut self) {
        self.vertices = vec![vec![-0.5].into(), vec![0.5].into()];
        self.abs.hosotope_mut();
    }

    fn antiprism(&self) -> Self {
        todo!()
    }

    fn simplex(rank: isize) -> Self {
        if rank == -1 {
            Self::nullitope()
        } else {
            let dim = rank as usize;
            let mut vertices = Vec::with_capacity(dim + 1);

            // Adds all points with a single entry equal to √2/2, and all others
            // equal to 0.
            for i in 0..dim {
                let mut v = vec![0.0; dim];
                v[i] = SQRT_2 / 2.0;
                vertices.push(v.into());
            }

            // Adds the remaining vertex, all of whose coordinates are equal.
            let a = (1.0 - ((dim + 1) as f64).sqrt()) * SQRT_2 / (2.0 * dim as f64);
            vertices.push(vec![a; dim].into());

            Concrete {
                vertices,
                abs: Abstract::simplex(rank),
            }
            .recenter()
        }
    }

    fn orientable(&self) -> bool {
        self.abs.orientable()
    }
}

impl std::ops::Index<isize> for Concrete {
    type Output = ElementList;

    /// Gets the list of elements with a given rank.
    fn index(&self, rank: isize) -> &Self::Output {
        &self.abs[rank]
    }
}

impl std::ops::IndexMut<isize> for Concrete {
    /// Gets the list of elements with a given rank.
    fn index_mut(&mut self, rank: isize) -> &mut Self::Output {
        &mut self.abs[rank]
    }
}
