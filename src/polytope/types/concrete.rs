use crate::{
    lang::{
        name::{Con, NameType},
        Name,
    },
    polytope::{
        flag::{FlagEvent, FlagIter},
        geometry::{Hyperplane, Hypersphere, Matrix, Point, Segment, Subspace},
        group::OrdPoint,
        rank::RankVec,
        Abstract, Element, ElementList, MeshBuilder, Polytope, Subelements, Subsupelements,
    },
    EPS,
};

use approx::{abs_diff_eq, abs_diff_ne};
use bevy::prelude::Mesh;
use core::f64;
use std::{
    collections::HashMap,
    f64::consts::{SQRT_2, TAU},
};

#[derive(Debug, Clone)]
/// Represents a [concrete polytope](https://polytope.miraheze.org/wiki/Polytope),
/// which is an [`Abstract`] together with its corresponding vertices.
pub struct Concrete {
    /// The list of vertices as points in Euclidean space.
    pub vertices: Vec<Point>,

    /// The underlying abstract polytope.
    pub abs: Abstract,

    /// The concrete name of the polytope.
    pub name: Name<Con>,
}

impl Concrete {
    /// Initializes a new concrete polytope from a set of vertices and an
    /// underlying abstract polytope. Does some debug assertions on the input.
    pub fn new(vertices: Vec<Point>, abs: Abstract) -> Self {
        // There must be as many abstract vertices as concrete ones.
        debug_assert_eq!(vertices.len(), abs.el_count(0));

        // All vertices must have the same dimension.
        if let Some(vertex0) = vertices.get(0) {
            for vertex1 in &vertices {
                debug_assert_eq!(vertex0.len(), vertex1.len());
            }
        }

        // With no further info, we create a generic name for the polytope.
        let n = abs.facet_count();
        let d = abs.rank();
        Self {
            vertices,
            abs,
            name: Name::generic(n, d),
        }
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

    /// Builds the Grünbaumian star polygon `{n / d}`, rotated by an angle.
    fn grunbaum_star_polygon_with_rot(n: usize, d: usize, rot: f64) -> Self {
        assert!(n >= 2);
        assert!(d >= 1);

        // Scaling factor for unit edge length.
        let angle = TAU * d as f64 / n as f64;
        let radius = (2.0 - 2.0 * angle.cos()).sqrt();

        Self::new(
            (0..n)
                .into_iter()
                .map(|k| {
                    let (sin, cos) = (k as f64 * angle + rot).sin_cos();
                    vec![sin / radius, cos / radius].into()
                })
                .collect(),
            Abstract::polygon(n),
        )
    }

    /// Builds the Grünbaumian star polygon `{n / d}`. If `n` and `d` have a
    /// common factor, the result is a multiply-wound polygon.
    pub fn grunbaum_star_polygon(n: usize, d: usize) -> Self {
        Self::grunbaum_star_polygon_with_rot(n, d, 0.0)
    }

    /// Builds the star polygon `{n / d}`. If `n` and `d` have a common factor,
    /// the result is a compound.
    pub fn star_polygon(n: usize, d: usize) -> Self {
        use gcd::Gcd;

        let gcd = n.gcd(d);
        let angle = TAU / n as f64;

        Self::compound_iter(
            (0..gcd)
                .into_iter()
                .map(|k| Self::grunbaum_star_polygon_with_rot(n / gcd, d / gcd, k as f64 * angle)),
        )
        .unwrap()
    }

    /// Scales a polytope by a given factor.
    pub fn scale(&mut self, k: f64) {
        for v in &mut self.vertices {
            *v *= k;
        }
    }

    /// Recenters a polytope so that the gravicenter is at the origin.
    pub fn recenter(&mut self) {
        if let Some(gravicenter) = self.gravicenter() {
            self.recenter_with(&gravicenter);
        }
    }

    pub fn recenter_with(&mut self, p: &Point) {
        for v in &mut self.vertices {
            *v -= p;
        }
    }

    /// Applies a function to all vertices of a polytope.
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

    /// Gets the least `x` coordinate of a vertex of the polytope.
    pub fn x_min(&self) -> Option<f64> {
        self.vertices
            .iter()
            .map(|v| float_ord::FloatOrd(v[0]))
            .min()
            .map(|x| x.0)
    }

    /// Gets the greatest `x` coordinate of a vertex of the polytope.
    pub fn x_max(&self) -> Option<f64> {
        self.vertices
            .iter()
            .map(|v| float_ord::FloatOrd(v[0]))
            .max()
            .map(|x| x.0)
    }

    /// Gets the edge lengths of all edges in the polytope, in order.
    pub fn edge_lengths(&self) -> Vec<f64> {
        let mut edge_lengths = Vec::new();

        // If there are no edges, we just return the empty vector.
        if let Some(edges) = self.abs.ranks.get(1) {
            edge_lengths.reserve_exact(edges.len());

            for edge in edges.iter() {
                let sub0 = edge.subs[0];
                let sub1 = edge.subs[1];

                edge_lengths.push((&self.vertices[sub0] - &self.vertices[sub1]).norm());
            }
        }

        edge_lengths
    }

    /// Checks whether a polytope is equilateral to a fixed precision, and with
    /// a specified edge length.
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
        if let Some(vertices) = self.element_vertices_ref(1, 0) {
            let (v0, v1) = (vertices[0], vertices[1]);

            return self.is_equilateral_with_len((v0 - v1).norm());
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

    pub fn dual(&self) -> Option<Self> {
        self._dual()
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

    pub fn dual_mut(&mut self) -> Result<(), ()> {
        self._dual_mut()
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
                if sphere.reciprocate_mut(v).is_err() {
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
            if sphere.reciprocate_mut(v).is_err() {
                return Err(());
            }
        }

        self.vertices = projections;

        // Takes the abstract dual.
        self.abs.dual_mut();
        *self.name_mut() = self.name().clone().dual();

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

    pub fn volume(&self) -> Option<f64> {
        use factorial::Factorial;

        let rank = self.rank();

        // We leave the nullitope's volume undefined.
        if rank == -1 {
            return None;
        }

        // The vertices, flattened if necessary.
        let flat_vertices = self.flat_vertices();
        let flat_vertices = flat_vertices.as_ref().unwrap_or(&self.vertices);

        if flat_vertices.get(0)?.len() != rank as usize {
            return None;
        }

        // Maps every element of the polytope to one of its vertices.
        let mut vertex_map = Vec::new();

        // Vertices map to themselves.
        let mut vertex_list = Vec::new();
        for v in 0..self[0].len() {
            vertex_list.push(v);
        }
        vertex_map.push(vertex_list);

        // Every other element maps to the vertex of any subelement.
        for r in 1..=self.rank() {
            let mut element_list = Vec::new();

            for el in self[r].iter() {
                element_list.push(vertex_map[r as usize - 1][el.subs[0]]);
            }

            vertex_map.push(element_list);
        }

        let mut volume = 0.0;

        // For each flag, there's a simplex defined by any vertices in its
        // elements and the origin. We add up the volumes of all of these
        // simplices times the sign of the flag that generated them.
        for flag_event in self.flag_events() {
            if let FlagEvent::Flag(flag) = flag_event {
                volume += flag.orientation.sign()
                    * Matrix::from_iterator(
                        rank as usize,
                        rank as usize,
                        flag.elements
                            .into_iter()
                            .enumerate()
                            .map(|(rank, idx)| &flat_vertices[vertex_map[rank][idx]])
                            .flatten()
                            .copied(),
                    )
                    .determinant();
            } else {
                return None;
            }
        }

        Some(volume.abs() / (rank as usize).factorial() as f64)
    }

    pub fn flat_vertices(&self) -> Option<Vec<Point>> {
        let subspace = Subspace::from_points(&self.vertices);

        if subspace.is_full_rank() {
            None
        } else {
            let mut flat_vertices = Vec::new();
            for v in self.vertices.iter() {
                flat_vertices.push(subspace.flatten(v));
            }
            Some(flat_vertices)
        }
    }

    /// Projects the vertices of the polytope into the lowest dimension possible.
    /// If the polytope's subspace is already of full rank, this is a no-op.
    pub fn flatten(&mut self) {
        if !self.vertices.is_empty() {
            self.flatten_into(&Subspace::from_points(&self.vertices));
        }
    }

    pub fn flatten_into(&mut self, subspace: &Subspace) {
        if !subspace.is_full_rank() {
            for v in self.vertices.iter_mut() {
                *v = subspace.flatten(v);
            }
        }
    }

    /// Takes the cross-section of a polytope through a given hyperplane.
    ///
    /// # Todo
    /// We should make this function take a general [`Subspace`] instead.
    pub fn slice(&self, slice: &Hyperplane) -> Self {
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
                vertices.push(p);
            }
        }

        let vertex_count = vertices.len();

        // The slice does not intersect the polytope.
        if vertex_count == 0 {
            return Self::nullitope();
        }

        abs.push(ElementList::single());
        abs.push(ElementList::vertices(vertex_count));

        // Takes care of building everything else.
        for r in 2..self.rank() {
            let mut new_hash_element = HashMap::new();
            let mut new_els = ElementList::new();

            for (idx, el) in self[r].iter().enumerate() {
                let mut new_subs = Subelements::new();
                for sub in el.subs.iter() {
                    if let Some(&v) = hash_element.get(sub) {
                        new_subs.push(v);
                    }
                }

                // If we got ourselves a new edge:
                if !new_subs.is_empty() {
                    new_hash_element.insert(idx, new_els.len());
                    new_els.push(Element::from_subs(new_subs));
                }
            }

            abs.push(new_els);
            hash_element = new_hash_element;
        }

        // Adds a maximal element manually.
        let facet_count = abs.ranks.last().unwrap().len();
        abs.push(ElementList::max(facet_count));

        // Splits compounds of dyads.
        let mut faces = abs[2].clone();

        let mut i = abs[1].len();
        let mut new_edges = Vec::<Element>::new();
        for (idx, edge) in abs[1].0.iter_mut().enumerate() {
            let edge_sub = &mut edge.subs;
            let comps = edge_sub.len() / 2;

            if comps > 1 {
                edge_sub.sort_by(|&i, &j| {
                    OrdPoint::new(vertices[i].clone()).cmp(&OrdPoint::new(vertices[j].clone()))
                });
            }

            for comp in 1..comps {
                let new_subs = Subelements::from_vec(edge_sub[2 * comp..2 * comp + 2].to_vec());
                new_edges.push(Element::from_subs(new_subs));
                for face in faces.0.iter_mut() {
                    if face.subs.contains(&idx) {
                        face.subs.push(i);
                    }
                }
                i += 1;
            }

            let new_subs = Subelements::from_vec(edge_sub[..2].to_vec());
            *edge = Element::from_subs(new_subs);
        }

        abs[1].append(&mut new_edges);
        abs[2] = faces;

        let mut abs2 = Abstract::new();
        for r in abs {
            abs2.push_subs(r);
        }

        Self::new(vertices, abs2)
    }

    /// Loads a polytope from a file path.
    pub fn from_path(fp: &impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        use std::{ffi::OsStr, fs, io};

        let off = OsStr::new("off");
        let ggb = OsStr::new("ggb");
        let ext = fp.as_ref().extension();

        if ext == Some(off) {
            Ok(Self::from_off(String::from_utf8(fs::read(fp)?).unwrap())?)
        } else if ext == Some(ggb) {
            Ok(Self::from_ggb(zip::read::ZipArchive::new(
                &mut fs::File::open(fp)?,
            )?)?)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "File extension not recognized.",
            ))
        }
    }

    pub fn get_mesh(&self) -> Mesh {
        MeshBuilder::new(self).get_mesh()
    }

    pub fn get_wireframe(&self) -> Mesh {
        MeshBuilder::new(self).get_wireframe()
    }
}

impl Polytope<Con> for Concrete {
    /// Returns the rank of the polytope.
    fn rank(&self) -> isize {
        self.abs.rank()
    }

    fn name(&self) -> &Name<Con> {
        &self.name
    }

    fn name_mut(&mut self) -> &mut Name<Con> {
        &mut self.name
    }

    /// Gets the number of elements of a given rank.
    fn el_count(&self, rank: isize) -> usize {
        self.abs.el_count(rank)
    }

    /// Gets the number of elements of all ranks.
    fn el_counts(&self) -> RankVec<usize> {
        self.abs.el_counts()
    }

    /// Builds the unique polytope of rank −1.
    fn nullitope() -> Self {
        Self::new(Vec::new(), Abstract::nullitope()).with_name(Name::Nullitope)
    }

    /// Builds the unique polytope of rank 0.
    fn point() -> Self {
        Self::new(vec![vec![].into()], Abstract::point()).with_name(Name::Point)
    }

    /// Builds a dyad with unit edge length.
    fn dyad() -> Self {
        Self::new(vec![vec![-0.5].into(), vec![0.5].into()], Abstract::dyad()).with_name(Name::Dyad)
    }

    /// Builds a convex regular polygon with `n` sides and unit edge length.
    fn polygon(n: usize) -> Self {
        Self::grunbaum_star_polygon(n, 1).with_name(Name::polygon(Con::regular(true), n))
    }

    /// Returns the dual of a polytope, or `None` if any facets pass through the
    /// origin.
    fn _dual(&self) -> Option<Self> {
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
    fn _dual_mut(&mut self) -> Result<(), ()> {
        self.dual_mut_with_sphere(&Hypersphere::unit(self.dim().unwrap_or(1)))
    }

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks.
    fn append(&mut self, mut p: Self) -> Result<(), ()> {
        if self.abs.append(p.abs).is_err() {
            return Err(());
        }

        self.vertices.append(&mut p.vertices);
        self.name = Name::compound(vec![(1, self.name().clone()), (1, p.name)]);

        Ok(())
    }

    fn element(&self, rank: isize, idx: usize) -> Option<Self> {
        let (vertices, abs) = self.abs.element_and_vertices(rank, idx)?;

        Some(Self::new(
            vertices
                .into_iter()
                .map(|idx| self.vertices[idx].clone())
                .collect(),
            abs,
        ))
    }

    fn flag_events(&self) -> FlagIter {
        self.abs.flag_events()
    }

    /// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// from two polytopes.
    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::duopyramid_with_height(p, q, 1.0)
    }

    /// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
    /// from two polytopes.
    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::new(
            Self::duoprism_vertices(&p.vertices, &q.vertices),
            Abstract::duoprism(&p.abs, &q.abs),
        )
    }

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum(p: &Self, q: &Self) -> Self {
        // Point-polytope duotegums are special cases.
        if p.rank() == 0 {
            q.clone()
        } else if q.rank() == 0 {
            p.clone()
        } else {
            Self::new(
                Self::duopyramid_vertices(&p.vertices, &q.vertices, 0.0, true),
                Abstract::duotegum(&p.abs, &q.abs),
            )
        }
    }

    /// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
    /// from two polytopes.
    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::new(
            Self::duoprism_vertices(&p.vertices, &q.vertices),
            Abstract::duocomb(&p.abs, &q.abs),
        )
    }

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope.
    fn ditope(&self) -> Self {
        Self::new(self.vertices.clone(), self.abs.ditope())
    }

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope in place.
    fn ditope_mut(&mut self) {
        self.abs.ditope_mut();
    }

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope.
    fn hosotope(&self) -> Self {
        Self::new(
            vec![vec![-0.5].into(), vec![0.5].into()],
            self.abs.hosotope(),
        )
    }

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope in place.
    fn hosotope_mut(&mut self) {
        self.vertices = vec![vec![-0.5].into(), vec![0.5].into()];
        self.abs.hosotope_mut();
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope.
    fn antiprism(&self) -> Self {
        todo!()
    }

    /// Determines whether a given polytope is
    /// [orientable](https://polytope.miraheze.org/wiki/Orientability).
    fn orientable(&self) -> bool {
        self.abs.orientable()
    }

    /// Builds a [simplex](https://polytope.miraheze.org/wiki/Simplex) with a
    /// given rank.
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

            let mut simplex = Concrete::new(vertices, Abstract::simplex(rank));
            simplex.recenter();
            simplex
        }
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
