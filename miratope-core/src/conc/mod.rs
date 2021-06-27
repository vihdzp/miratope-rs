//! Declares the [`Concrete`] polytope type and all associated data structures.

pub mod cycle;
pub mod element_types;
pub mod file;

use std::collections::{HashMap, HashSet};

use super::{
    abs::{
        elements::{
            AbstractBuilder, ElementList, ElementRef, SubelementList, Subelements, Superelements,
        },
        flag::{Flag, FlagChanges, FlagEvent, OrientedFlagIter},
        rank::{Rank, RankVec},
        Abstract,
    },
    DualError, DualResult, Polytope,
};
use crate::{
    geometry::{Hyperplane, Hypersphere, Matrix, Point, PointOrd, Segment, Subspace, Vector},
    Consts, Float,
};

use approx::{abs_diff_eq, abs_diff_ne};
use rayon::prelude::*;
use vec_like::*;

/// A trait for concrete polytopes.
///
/// This trait exists so that we can reuse this code for `miratope_lang`. The
/// traits that are not auto-implemented require us to manually set names over
/// there.
pub trait ConcretePolytope: Polytope {
    /// Returns a reference to the underlying [`Concrete`] polytope.
    fn con(&self) -> &Concrete;

    /// Returns a mutable reference to the underlying [`Concrete`] polytope.
    fn con_mut(&mut self) -> &mut Concrete;

    /// Returns a reference to the vertices of the polytope.
    fn vertices(&self) -> &Vec<Point> {
        &self.con().vertices
    }

    /// Returns a mutable reference to the vertices of the polytope.
    fn vertices_mut(&mut self) -> &mut Vec<Point> {
        &mut self.con_mut().vertices
    }

    /// Returns the number of dimensions of the space the polytope lives in,
    /// or `None` in the case of the nullitope.
    fn dim(&self) -> Option<usize> {
        Some(self.con().vertices.get(0)?.len())
    }

    /// Returns the number of dimensions of the space the polytope lives in,
    /// or 0 in the case of the nullitope.
    fn dim_or(&self) -> usize {
        self.dim().unwrap_or(0)
    }

    /// Builds a dyad with a specified height.
    fn dyad_with(height: Float) -> Self;

    fn grunbaum_star_polygon_with_rot(n: usize, d: usize, rot: Float) -> Self;

    /// Builds the Grünbaumian star polygon `{n / d}` with unit circumradius. If
    /// `n` and `d` have a common factor, the result is a multiply-wound
    /// polygon.
    fn grunbaum_star_polygon(n: usize, d: usize) -> Self {
        Self::grunbaum_star_polygon_with_rot(n, d, 0.0)
    }

    /// Builds the star polygon `{n / d}`. with unit circumradius. If `n` and `d`
    /// have a common factor, the result is a compound.
    ///
    /// # Panics
    /// Will panic if either `n < 2` or if `d < 1`, in which case there's
    /// nothing sensible to do.
    fn star_polygon(n: usize, d: usize) -> Self {
        assert!(n >= 2);
        assert!(d >= 1);

        use gcd::Gcd;

        let gcd = n.gcd(d);
        let angle = Float::TAU / n as Float;

        Self::compound_iter(
            (0..gcd).into_iter().map(|k| {
                Self::grunbaum_star_polygon_with_rot(n / gcd, d / gcd, k as Float * angle)
            }),
        )
    }

    /// Scales a polytope by a given factor.
    fn scale(&mut self, k: Float) {
        for v in self.vertices_mut() {
            *v *= k;
        }
    }

    /// Recenters a polytope so that the gravicenter is at the origin.
    fn recenter(&mut self) {
        if let Some(gravicenter) = self.gravicenter() {
            self.recenter_with(&gravicenter);
        }
    }

    /// Recenters a polytope so that a certain point is at the origin.
    fn recenter_with(&mut self, p: &Point) {
        for v in self.vertices_mut() {
            *v -= p;
        }
    }

    /// Applies a linear transformation to all vertices of a polytope.
    fn apply(mut self, m: &Matrix) -> Self {
        for v in self.vertices_mut() {
            let new_v = m * v as &_;
            *v = new_v;
        }

        self
    }

    /// Calculates the circumsphere of a polytope. Returns `None` if the
    /// polytope isn't circumscribable.
    fn circumsphere(&self) -> Option<Hypersphere> {
        let mut vertices = self.vertices().iter();

        let first_vertex = vertices.next()?.clone();
        let mut center: Point = first_vertex.clone();
        let mut subspace = Subspace::new(first_vertex.clone());

        for vertex in vertices {
            // If the new vertex does not lie on the hyperplane of the others:
            if let Some(basis_vector) = subspace.add_basis(&vertex) {
                // Calculates the new circumcenter.
                let distance = ((&center - vertex).norm_squared()
                    - (&center - &first_vertex).norm_squared())
                    / (2.0 * (vertex - &first_vertex).dot(basis_vector));

                center += distance * basis_vector;
            }
            // If the new vertex lies on the others' hyperplane, but is not at
            // the correct distance from the first vertex:
            else if abs_diff_ne!(
                (&center - &first_vertex).norm(),
                (&center - vertex).norm(),
                epsilon = Float::EPS
            ) {
                return None;
            }
        }

        Some(Hypersphere {
            squared_radius: (&center - first_vertex).norm(),
            center,
        })
    }

    /// Calculates the gravicenter of a polytope, or returns `None` in the case
    /// of the nullitope.
    fn gravicenter(&self) -> Option<Point> {
        let mut g = Point::zeros(self.dim()? as usize);
        let vertices = &self.con().vertices;

        // Adds up all vertices.
        for v in vertices {
            g += v;
        }

        // Takes the average.
        Some(g / (vertices.len() as Float))
    }

    /// Gets the least and greatest distance of a vertex of the polytope,
    /// measuring from a specified direction, or returns `None` in the case of
    /// the nullitope.
    fn minmax(&self, direction: &Vector) -> Option<(Float, Float)> {
        use itertools::{Itertools, MinMaxResult};

        let hyperplane = Hyperplane::new(direction.clone(), 0.0);

        match self
            .con()
            .vertices
            .iter()
            .map(|v| ordered_float::OrderedFloat(hyperplane.distance(v)))
            .minmax()
        {
            // The vertex vector is empty.
            MinMaxResult::NoElements => None,

            // The single vertex gives both the minimum and maximum distance.
            MinMaxResult::OneElement(x) => Some((x.0, x.0)),

            // The minimum and maximum distances.
            MinMaxResult::MinMax(x, y) => Some((x.0, y.0)),
        }
    }

    /// Returns the length of a given edge.
    fn edge_len(&self, idx: usize) -> Option<Float> {
        let edge = self.abs().get_element(ElementRef::new(Rank::new(1), idx))?;
        let vertices = self.vertices();
        Some((&vertices[edge.subs[0]] - &vertices[edge.subs[1]]).norm())
    }

    /// Gets the edge lengths of all edges in the polytope, in order.
    fn edge_lengths(&self) -> Vec<Float> {
        let mut edge_lengths = Vec::new();

        // If there are no edges, we just return the empty vector.
        if let Some(edges) = self.ranks().get(Rank::new(1)) {
            edge_lengths.reserve(edges.len());

            let vertices = self.vertices();
            for edge in edges {
                let sub0 = edge.subs[0];
                let sub1 = edge.subs[1];

                edge_lengths.push((&vertices[sub0] - &vertices[sub1]).norm());
            }
        }

        edge_lengths
    }

    /// Checks whether a polytope is equilateral to a fixed precision, and with
    /// a specified edge length.
    fn is_equilateral_with_len(&self, len: Float) -> bool {
        let edge_lengths = self.edge_lengths().into_iter();

        // Checks that every other edge length is equal to the first.
        for edge_len in edge_lengths {
            if abs_diff_eq!(edge_len, len, epsilon = Float::EPS) {
                return false;
            }
        }

        true
    }

    /// Checks whether a polytope is equilateral to a fixed precision.
    fn is_equilateral(&self) -> bool {
        if let Some(vertices) = self
            .con()
            .element_vertices_ref(ElementRef::new(Rank::new(1), 0))
        {
            self.is_equilateral_with_len((vertices[0] - vertices[1]).norm())
        } else {
            true
        }
    }

    /// I haven't actually implemented this in the general case.
    ///
    /// # Todo
    /// Maybe make this work in the general case?
    fn midradius(&self) -> Float {
        let vertices = &self.vertices();
        let edges = &self.ranks()[Rank::new(1)];
        let edge = &edges[0];

        let sub0 = edge.subs[0];
        let sub1 = edge.subs[1];

        (&vertices[sub0] + &vertices[sub1]).norm() / 2.0
    }

    /// Builds the dual of a polytope with a given reciprocation sphere in
    /// place, or does nothing in case any facets go through the reciprocation
    /// center. In case of failure, returns the index of the facet through the
    /// projection center.
    ///
    /// # Panics
    /// This method shouldn't panic. If it does, please file a bug.
    fn try_dual_mut_with(&mut self, sphere: &Hypersphere) -> DualResult<()>;

    /// Returns the dual of a polytope with a given reciprocation sphere, or
    /// `None` if any facets pass through the reciprocation center.
    fn try_dual_with(&self, sphere: &Hypersphere) -> DualResult<Self> {
        let mut clone = self.clone();
        clone.try_dual_mut_with(sphere).map(|_| clone)
    }

    /// Builds a pyramid with a specified apex.
    fn pyramid_with(&self, apex: Point) -> Self;

    /// Builds a prism with a specified height.
    fn prism_with(&self, height: Float) -> Self;

    /// Builds a uniform prism from an {n/d} polygon.
    fn uniform_prism(n: usize, d: usize) -> Self {
        Self::star_polygon(n, d).prism_with(2.0 * (Float::PI * d as Float / n as Float).sin())
    }

    /// Builds a tegum with two specified apices.
    fn tegum_with(&self, apex1: Point, apex2: Point) -> Self;

    fn antiprism_with_vertices<T: Iterator<Item = Point>, U: Iterator<Item = Point>>(
        &self,
        vertices: T,
        dual_vertices: U,
    ) -> Self;

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope.
    fn try_antiprism_with(&self, sphere: &Hypersphere, height: Float) -> DualResult<Self> {
        let half_height = height / 2.0;
        let vertices = self.vertices().iter().map(|v| v.push(-half_height));
        let dual = self.try_dual_with(sphere)?;
        let dual_vertices = dual.vertices().iter().map(|v| v.push(half_height));

        Ok(self.antiprism_with_vertices(vertices, dual_vertices))
    }

    /// Builds an antiprism, using a specified hypersphere to take a dual, and
    /// with a given height.
    ///
    /// # Panics
    /// Panics if any facets pass through the inversion center. If you want to
    /// handle this possibility, use [`Self::try_antiprism_with`] instead.
    fn antiprism_with(&self, sphere: &Hypersphere, height: Float) -> Self {
        self.try_antiprism_with(sphere, height).unwrap()
    }

    /// Builds a uniform antiprism of unit edge length.
    fn uniform_antiprism(n: usize, d: usize) -> Self {
        let polygon = Self::star_polygon(n, d);

        // Appropriately scaled antiprism.
        if n != 2 * d {
            let angle = Float::PI * d as Float / n as Float;
            let cos = angle.cos();
            let height = ((cos - (2.0 * angle).cos()) * 2.0).sqrt();

            polygon.antiprism_with(
                &Hypersphere::with_squared_radius(Point::zeros(2), cos),
                height,
            )
        }
        // Digon compounds are a special case.
        else {
            let half_height = Float::SQRT_2 / 2.0;
            let vertices = polygon.vertices().iter().map(|v| v.push(-half_height));
            let dual_vertices = polygon
                .vertices()
                .iter()
                .map(|v| vec![v[1], -v[0], half_height].into());

            polygon.antiprism_with_vertices(vertices, dual_vertices)
        }
    }

    /// Gets the references to the (geometric) vertices of an element on the
    /// polytope.
    fn element_vertices_ref(&self, el: ElementRef) -> Option<Vec<&Point>> {
        Some(
            self.abs()
                .element_vertices(el)?
                .iter()
                .map(|&v| &self.vertices()[v])
                .collect(),
        )
    }

    /// Generates the vertices for either a tegum or a pyramid product with two
    /// given vertex sets and a given height.
    ///
    /// The vertices are the padded vertices of `q`, followed by the padded
    /// vertices of `p`.
    fn duopyramid_vertices(
        p: &[Point],
        q: &[Point],
        p_pad: &Point,
        q_pad: &Point,
        height: Float,
        tegum: bool,
    ) -> Vec<Point> {
        // Duotegums with points should just return the original polytopes.
        if tegum {
            if p.get(0).map(|vp| vp.len()) == Some(0) {
                return q.to_owned();
            } else if q.get(0).map(|vq| vq.len()) == Some(0) {
                return p.to_owned();
            }
        }

        let half_height = height / 2.0;

        q.iter()
            // To every point in q, we append zeros to the left.
            .map(|vq| {
                let mut v: Vec<_> = p_pad.iter().copied().chain(vq.iter().copied()).collect();
                if !tegum {
                    v.push(-half_height);
                }
                v.into()
            })
            // To every point in p, we append zeros to the right.
            .chain(p.iter().map(|vp| {
                let mut v: Vec<_> = vp.iter().copied().chain(q_pad.iter().copied()).collect();
                if !tegum {
                    v.push(half_height);
                }
                v.into()
            }))
            .collect()
    }

    /// Generates the vertices for a duoprism with two given vertex sets.
    fn duoprism_vertices(p: &[Point], q: &[Point]) -> Vec<Point> {
        // The dimension of the points in p.
        let p_dim = if let Some(vp) = p.get(0) {
            vp.len()
        } else {
            return Vec::new();
        };

        // The dimension of the points in q.
        let q_dim = if let Some(vq) = q.get(0) {
            vq.len()
        } else {
            return Vec::new();
        };

        // The dimension of our new points.
        let dim = p_dim + q_dim;

        // We take all elements in the cartesian product p × q, and chain each
        // pair together.
        itertools::iproduct!(p.iter(), q.iter())
            .map(|(vp, vq)| Point::from_iterator(dim, vp.iter().chain(vq.iter()).copied()))
            .collect::<Vec<_>>()
    }

    /// Generates a duopyramid from two given polytopes with a given height and
    /// a given offset.
    fn duopyramid_with(
        p: &Self,
        q: &Self,
        p_offset: &Point,
        q_offset: &Point,
        height: Float,
    ) -> Self;

    fn duotegum_with(p: &Self, q: &Self, p_offset: &Point, q_offset: &Point) -> Self;

    /// Computes the volume of a polytope by adding up the contributions of all
    /// flags. Returns `None` if the volume is undefined.
    fn volume(&mut self) -> Option<Float> {
        let rank = self.rank();

        // We leave the nullitope's volume undefined.
        if rank == Rank::new(-1) {
            return None;
        }

        // The flattened vertices (may possibly be the original vertices).
        let subspace = Subspace::from_points(self.vertices().iter());
        let flat_vertices = subspace.flatten_vec(&self.vertices());

        match flat_vertices.get(0)?.len().cmp(&rank.into()) {
            // Degenerate polytopes have volume 0.
            std::cmp::Ordering::Less => {
                return Some(0.0);
            }
            // Skew polytopes don't have a defined volume.
            std::cmp::Ordering::Greater => {
                return None;
            }
            _ => {}
        }

        // Maps every element of the polytope to one of its vertices.
        let mut vertex_map = Vec::new();

        // Vertices map to themselves.
        let vertex_count = self.vertex_count();
        let mut vertex_list = Vec::with_capacity(vertex_count);
        for v in 0..vertex_count {
            vertex_list.push(v);
        }
        vertex_map.push(vertex_list);

        // Every other element maps to the vertex of any subelement.
        for r in Rank::range_inclusive_iter(Rank::new(1), self.rank()) {
            let mut element_list = Vec::new();

            for el in &self.ranks()[r] {
                element_list.push(vertex_map[r.into_usize() - 1][el.subs[0]]);
            }

            vertex_map.push(element_list);
        }

        let mut volume = 0.0;
        let rank_usize = rank.into_usize();

        // All of the flags we've found so far.
        let mut all_flags = HashSet::new();

        if !self.abs().sorted {
            panic!("Unsorted!")
        }
        //   self.abs_mut().sort();

        // We iterate over all flags in the polytope.
        for flag in self.flags() {
            // If this flag forms a new component of the polytope, we iterate
            // over the oriented flags in this component.
            if !all_flags.contains(&flag) {
                let mut component_volume = 0.0;

                for flag_event in
                    OrientedFlagIter::with_flags(self.abs(), FlagChanges::all(rank), flag.into())
                {
                    if let FlagEvent::Flag(oriented_flag) = flag_event {
                        let new = all_flags.insert(oriented_flag.flag.clone());
                        debug_assert!(new, "A flag is in two different components.");

                        // For each flag, there's a simplex defined by any vertices in its
                        // elements and the origin. We add up the volumes of all of these
                        // simplices times the sign of the flag that generated them.
                        component_volume += oriented_flag.orientation.sign()
                            * Matrix::from_iterator(
                                rank_usize,
                                rank_usize,
                                oriented_flag
                                    .into_iter()
                                    .enumerate()
                                    .map(|(rank, idx)| &flat_vertices[vertex_map[rank][idx]])
                                    .flatten()
                                    .copied(),
                            )
                            .determinant();
                    }
                    // A non-orientable polytope doesn't have a volume.
                    else {
                        return None;
                    }
                }

                // We add up the volumes of all components.
                volume += component_volume.abs();
            }
        }

        Some(volume / crate::factorial(rank_usize) as Float)
    }

    /// Projects the vertices of the polytope into the lowest dimension possible.
    /// If the polytope's subspace is already of full rank, this is a no-op.
    fn flatten(&mut self);

    /// Flattens the vertices of a polytope into a specified subspace.
    fn flatten_into(&mut self, subspace: &Subspace);

    fn cross_section(&self, slice: &Hyperplane) -> Self;
}

/// Represents a [concrete polytope](https://polytope.miraheze.org/wiki/Polytope),
/// which is an [`Abstract`] together with its corresponding vertices.
#[derive(Debug, Clone)]
pub struct Concrete {
    /// The list of vertices as points in Euclidean space.
    pub vertices: Vec<Point>,

    /// The underlying abstract polytope.
    pub abs: Abstract,
}

impl ConcretePolytope for Concrete {
    fn con(&self) -> &Concrete {
        self
    }

    fn con_mut(&mut self) -> &mut Concrete {
        self
    }

    /// Builds a dyad with a specified height.
    fn dyad_with(height: Float) -> Self {
        let half_height = height / 2.0;

        Self::new(
            vec![vec![-half_height].into(), vec![half_height].into()],
            Abstract::dyad(),
        )
    }

    /// Builds the Grünbaumian star polygon `{n / d}` with unit circumradius,
    /// rotated by an angle.
    fn grunbaum_star_polygon_with_rot(n: usize, d: usize, rot: Float) -> Self {
        debug_assert!(n >= 2);
        debug_assert!(d >= 1);

        let angle = Float::TAU * d as Float / n as Float;

        Self::new(
            (0..n)
                .into_iter()
                .map(|k| {
                    let (sin, cos) = (k as Float * angle + rot).sin_cos();
                    vec![sin, cos].into()
                })
                .collect(),
            Abstract::polygon(n),
        )
    }

    /// Builds the dual of a polytope with a given reciprocation sphere in
    /// place, or does nothing in case any facets go through the reciprocation
    /// center. In case of failure, returns the index of the facet through the
    /// projection center.
    ///
    /// # Panics
    /// This method shouldn't panic. If it does, please file a bug.
    fn try_dual_mut_with(&mut self, sphere: &Hypersphere) -> DualResult<()> {
        // If we're dealing with a nullitope, the dual is itself.
        let rank = self.rank();
        if rank == Rank::new(-1) {
            return Ok(());
        }
        // In the case of points, we reciprocate them.
        else if rank == Rank::new(0) {
            for (idx, v) in self.vertices.iter_mut().enumerate() {
                if !sphere.reciprocate_mut(v) {
                    return Err(DualError(idx));
                }
            }
        }

        // We project the sphere's center onto the polytope's hyperplane to
        // avoid skew weirdness.
        let h = Subspace::from_points(self.vertices.iter());
        let o = h.project(&sphere.center);

        let mut projections;
        let rank_minus_one = rank.minus_one();

        // We project our inversion center onto each of the facets.
        if rank >= Rank::new(2) {
            let facet_count = self.el_count(rank_minus_one);
            let indices: Vec<_> = (0..facet_count).collect();

            projections = indices
                .into_par_iter()
                .map(|idx| {
                    Subspace::from_points(
                        self.element_vertices_ref(ElementRef::new(rank_minus_one, idx))
                            .unwrap()
                            .into_iter(),
                    )
                    .project(&o)
                })
                .collect();
        }
        // If our polytope is 1D, the vertices themselves are the facets.
        else {
            projections = self.vertices.clone();
        }

        // Reciprocates the projected points.
        for (idx, v) in projections.iter_mut().enumerate() {
            if !sphere.reciprocate_mut(v) {
                return Err(DualError(idx));
            }
        }

        self.vertices = projections;

        // Takes the abstract dual.
        self.abs.dual_mut();

        Ok(())
    }

    /// Builds a pyramid with a specified apex.
    fn pyramid_with(&self, apex: Point) -> Self {
        let mut poly = self.pyramid();
        poly.vertices[0] = apex;
        poly
    }

    /// Builds a prism with a specified height.
    fn prism_with(&self, height: Float) -> Self {
        Self::duoprism(self, &Self::dyad_with(height))
    }

    /// Builds a tegum with two specified apices.
    fn tegum_with(&self, apex1: Point, apex2: Point) -> Self {
        let mut poly = self.tegum();
        poly.vertices[0] = apex1;
        poly.vertices[1] = apex2;
        poly
    }

    fn antiprism_with_vertices<T: Iterator<Item = Point>, U: Iterator<Item = Point>>(
        &self,
        vertices: T,
        dual_vertices: U,
    ) -> Self {
        let (abs, vertex_indices, dual_vertex_indices) = self.abs.antiprism_and_vertices();
        let vertex_count = abs.vertex_count();
        let mut new_vertices = Vec::with_capacity(vertex_count);
        new_vertices.resize(vertex_count, vec![].into());

        for (idx, v) in vertices.enumerate() {
            new_vertices[vertex_indices[idx]] = v;
        }
        for (idx, v) in dual_vertices.enumerate() {
            new_vertices[dual_vertex_indices[idx]] = v;
        }

        Self::new(new_vertices, abs)
    }

    /// Generates a duopyramid from two given polytopes with a given height and
    /// a given offset.
    fn duopyramid_with(
        p: &Self,
        q: &Self,
        p_offset: &Point,
        q_offset: &Point,
        height: Float,
    ) -> Self {
        Self::new(
            Self::duopyramid_vertices(&p.vertices, &q.vertices, p_offset, q_offset, height, false),
            Abstract::duopyramid(&p.abs, &q.abs),
        )
    }

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum_with(p: &Self, q: &Self, p_offset: &Point, q_offset: &Point) -> Self {
        Self::new(
            Self::duopyramid_vertices(&p.vertices, &q.vertices, p_offset, q_offset, 0.0, true),
            Abstract::duotegum(&p.abs, &q.abs),
        )
    }

    /// Projects the vertices of the polytope into the lowest dimension possible.
    /// If the polytope's subspace is already of full rank, this is a no-op.
    fn flatten(&mut self) {
        if !self.vertices.is_empty() {
            self.flatten_into(&Subspace::from_points(self.vertices.iter()));
        }
    }

    /// Flattens the vertices of a polytope into a specified subspace.
    fn flatten_into(&mut self, subspace: &Subspace) {
        if !subspace.is_full_rank() {
            for v in self.vertices.iter_mut() {
                *v = subspace.flatten(v);
            }
        }
    }

    /// Takes the cross-section of a polytope through a given hyperplane.
    ///
    /// # Panics
    /// This method shouldn't panic. If it does, please file a bug.
    ///
    /// # Todo
    /// We should make this function take a general [`Subspace`] instead.
    fn cross_section(&self, slice: &Hyperplane) -> Self {
        let mut vertices = Vec::new();
        let mut ranks = RankVec::with_rank_capacity(self.rank().minus_one());

        // We map all indices of k-elements in the original polytope to the
        // indices of the new (k-1)-elements resulting from taking their
        // intersections with the slicing hyperplane.
        let mut hash_element = HashMap::new();

        // Determines the vertices of the cross-section.
        for (idx, edge) in self[Rank::new(1)].iter().enumerate() {
            let segment = Segment(&self.vertices[edge.subs[0]], &self.vertices[edge.subs[1]]);

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

        ranks.push(SubelementList::min());
        ranks.push(SubelementList::vertices(vertex_count));

        // Takes care of building everything else.
        for r in Rank::range_iter(2, self.rank()) {
            let mut new_hash_element = HashMap::new();
            let mut new_els = SubelementList::new();

            for (idx, el) in self[r].iter().enumerate() {
                let mut new_subs = Subelements::new();
                for sub in &el.subs {
                    if let Some(&v) = hash_element.get(sub) {
                        new_subs.push(v);
                    }
                }

                // If we got ourselves a new edge:
                if !new_subs.is_empty() {
                    new_hash_element.insert(idx, new_els.len());
                    new_els.push(new_subs);
                }
            }

            ranks.push(new_els);
            hash_element = new_hash_element;
        }

        // Adds a maximal element manually.
        ranks.push(SubelementList::max(ranks.last().unwrap().len()));

        // Splits compounds of dyads.
        let (first, last) = ranks.split_at_mut(Rank::new(2));

        if let (Some(edges), Some(faces)) = (first.last_mut(), last.first_mut()) {
            // Keeps track of the indices of our new edges.
            let mut edge_num = edges.len();
            let mut new_edges = SubelementList::new();

            // The superelements of all edges.
            let mut edge_sups = Vec::new();
            for _ in 0..edge_num {
                edge_sups.push(Superelements::new());
            }

            for (idx, face) in faces.iter().enumerate() {
                for &sub in face {
                    edge_sups[sub].push(idx);
                }
            }

            for (edge_idx, subs) in edges.iter_mut().enumerate() {
                debug_assert_eq!(
                    subs.len() % 2,
                    0,
                    "A line should always intersect a polygon an even amount of times!"
                );
                let comps = subs.len() / 2;

                if comps > 1 {
                    // Sorts the component's vertices lexicographically.
                    subs.sort_unstable_by_key(|&x| PointOrd::new(vertices[x].clone()));

                    // Splits the edge, adds the new split edges as subelements
                    // to the edge's superelements.
                    for _ in 1..comps {
                        let v0 = subs.pop().unwrap();
                        let v1 = subs.pop().unwrap();
                        new_edges.push(Subelements(vec![v0, v1]));

                        for &sup in &edge_sups[edge_idx] {
                            faces[sup].push(edge_num);
                        }

                        edge_num += 1;
                    }
                }
            }

            // Adds the new edges.
            edges.append(&mut new_edges);
        }

        // Builds the polytope.
        let mut abs = AbstractBuilder::new();
        for subelements in ranks {
            abs.push(subelements);
        }

        Self::new(vertices, abs.build())
    }
}

impl Concrete {
    /// Initializes a new concrete polytope from a set of vertices and an
    /// underlying abstract polytope. Does some debug assertions on the input.
    pub fn new(vertices: Vec<Point>, abs: Abstract) -> Self {
        // There must be as many abstract vertices as concrete ones.
        debug_assert_eq!(
            abs.vertex_count(),
            vertices.len(),
            "Abstract vertex count doesn't match concrete vertex count!"
        );

        // All vertices must have the same dimension.
        if cfg!(debug_assertions) {
            if let Some(vertex0) = vertices.get(0) {
                for vertex1 in &vertices {
                    debug_assert_eq!(vertex0.len(), vertex1.len());
                }
            }
        }

        // With no further info, we create a generic name for the polytope.
        Self { vertices, abs }
    }
}

impl Polytope for Concrete {
    fn abs(&self) -> &Abstract {
        &self.abs
    }

    fn abs_mut(&mut self) -> &mut Abstract {
        &mut self.abs
    }

    /// Builds the unique polytope of rank −1.
    fn nullitope() -> Self {
        Self::new(Vec::new(), Abstract::nullitope())
    }

    /// Builds the unique polytope of rank 0.
    fn point() -> Self {
        Self::new(vec![vec![].into()], Abstract::point())
    }

    /// Builds a dyad with unit edge length.
    fn dyad() -> Self {
        Self::dyad_with(1.0)
    }

    /// Builds a convex regular polygon with `n` sides and unit edge length.
    fn polygon(n: usize) -> Self {
        Self::grunbaum_star_polygon(n, 1)
    }

    /// Returns the dual of a polytope, or `None` if any facets pass through the
    /// origin.
    fn try_dual(&self) -> DualResult<Self> {
        let mut clone = self.clone();
        clone.try_dual_mut().map(|_| clone)
    }

    /// Builds the dual of a polytope in place, or does nothing in case any
    /// facets go through the origin. Returns the dual if successful, and `None`
    /// otherwise.
    fn try_dual_mut(&mut self) -> DualResult<()> {
        self.try_dual_mut_with(&Hypersphere::unit(self.dim().unwrap_or(1)))
    }

    fn petrial_mut(&mut self) -> bool {
        self.abs.petrial_mut()
    }

    /// Builds the Petrie polygon of a polytope from a given flag, or returns
    /// `None` if it's invalid.
    fn petrie_polygon_with(&mut self, flag: Flag) -> Option<Self> {
        let vertices = self.abs.petrie_polygon_vertices(flag)?;
        let n = vertices.len();

        Some(Self::new(
            vertices
                .into_iter()
                .map(|idx| self.vertices[idx].clone())
                .collect(),
            Abstract::polygon(n),
        ))
    }

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks.
    fn comp_append(&mut self, mut p: Self) {
        self.abs.comp_append(p.abs);
        self.vertices.append(&mut p.vertices);
    }

    /// Gets the element with a given rank and index as a polytope, or returns
    /// `None` if such an element doesn't exist.
    fn element(&self, el: ElementRef) -> Option<Self> {
        let (vertices, abs) = self.abs.element_and_vertices(el)?;

        Some(Self::new(
            vertices
                .into_iter()
                .map(|idx| self.vertices[idx].clone())
                .collect(),
            abs,
        ))
    }

    fn omnitruncate(&mut self) -> Self {
        let (abs, flags) = self.abs.omnitruncate_and_flags();
        let dim = self.dim().unwrap();

        // Maps each element to the polytope to some vertex.
        let mut element_vertices = vec![self.vertices.clone()];
        for r in Rank::range_inclusive_iter(Rank::new(1), self.rank()) {
            let mut rank_vertices = Vec::new();

            for el in &self[r] {
                let mut p = Point::zeros(dim);
                let subs = &el.subs;

                for &sub in subs {
                    p += &element_vertices[r.into_usize() - 1][sub];
                }

                rank_vertices.push(p / subs.len() as Float);
            }

            element_vertices.push(rank_vertices);
        }

        let vertices: Vec<_> = flags
            .into_iter()
            .map(|flag| {
                flag.into_iter()
                    .enumerate()
                    .map(|(r, idx)| &element_vertices[r][idx])
                    .sum()
            })
            .collect();

        Self::new(vertices, abs)
    }

    /// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// with unit height from two polytopes.
    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::duopyramid_with(
            p,
            q,
            &Point::zeros(p.dim_or()),
            &Point::zeros(q.dim_or()),
            1.0,
        )
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
        Self::duotegum_with(p, q, &Point::zeros(p.dim_or()), &Point::zeros(q.dim_or()))
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

    fn try_antiprism(&self) -> DualResult<Self> {
        Self::try_antiprism_with(&self, &Hypersphere::unit(self.dim().unwrap_or(1)), 1.0)
    }

    /// Builds a [simplex](https://polytope.miraheze.org/wiki/Simplex) with a
    /// given rank.
    fn simplex(rank: Rank) -> Self {
        if rank == Rank::new(-1) {
            Self::nullitope()
        } else {
            let dim = rank.into_usize();
            let mut vertices = Vec::with_capacity(dim + 1);

            // Adds all points with a single entry equal to √2/2, and all others
            // equal to 0.
            for i in 0..dim {
                let mut v = Point::zeros(dim);
                v[i] = Float::SQRT_2 / 2.0;
                vertices.push(v);
            }

            // Adds the remaining vertex, all of whose coordinates are equal.
            let dim_f = dim as Float;
            let a = (1.0 - (dim_f + 1.0).sqrt()) * Float::SQRT_2 / (2.0 * dim_f);
            vertices.push(vec![a; dim].into());

            let mut simplex = Concrete::new(vertices, Abstract::simplex(rank));
            simplex.recenter();
            simplex
        }
    }
}

impl std::ops::Index<Rank> for Concrete {
    type Output = ElementList;

    /// Gets the list of elements with a given rank.
    fn index(&self, rank: Rank) -> &Self::Output {
        &self.abs[rank]
    }
}

impl std::ops::IndexMut<Rank> for Concrete {
    /// Gets the list of elements with a given rank.
    fn index_mut(&mut self, rank: Rank) -> &mut Self::Output {
        &mut self.abs[rank]
    }
}

#[cfg(test)]
mod tests {
    use super::{Concrete, ConcretePolytope};
    use crate::{abs::rank::Rank, Consts, Float, Polytope};

    use approx::abs_diff_eq;

    /// Tests that a polytope has an expected volume.
    fn test_volume(poly: &mut Concrete, volume: Option<Float>) {
        if let Some(poly_volume) = poly.volume() {
            let volume = volume.expect(&format!(
                "Expected no volume for {}, found volume {}!",
                "TBA: name", poly_volume
            ));

            assert!(
                abs_diff_eq!(poly_volume, volume, epsilon = Float::EPS),
                "Expected volume {} for {}, found volume {}.",
                volume,
                "TBA: name",
                poly_volume
            );
        } else if let Some(volume) = volume {
            panic!(
                "Expected volume {} for {}, found no volume!",
                volume, "TBA: name",
            );
        }
    }

    #[test]
    fn nullitope() {
        test_volume(&mut Concrete::nullitope(), None)
    }

    #[test]
    fn point() {
        test_volume(&mut Concrete::point(), Some(1.0));
    }

    #[test]
    fn dyad() {
        test_volume(&mut Concrete::dyad(), Some(1.0));
    }

    fn polygon_area(n: usize, d: usize) -> Float {
        let n = n as Float;
        let d = d as Float;
        n * (d * Float::TAU / n).sin() / 2.0
    }

    #[test]
    fn polygon() {
        for n in 2..=10 {
            for d in 1..=n / 2 {
                let mut poly = Concrete::star_polygon(n, d);
                test_volume(&mut poly, Some(polygon_area(n, d)));
            }
        }
    }

    #[test]
    fn duopyramid() {
        let mut polygons = Vec::new();
        let mut areas = Vec::new();
        for n in 2..=5 {
            for d in 1..=n / 2 {
                polygons.push(Concrete::star_polygon(n, d));
                areas.push(polygon_area(n, d));
            }
        }

        for m in 0..polygons.len() {
            for n in 0..polygons.len() {
                test_volume(
                    &mut Concrete::duopyramid(&polygons[m], &polygons[n]),
                    Some(areas[m] * areas[n] / 30.0),
                )
            }
        }
    }

    #[test]
    fn duoprism() {
        let mut polygons = Vec::new();
        let mut areas = Vec::new();
        for n in 2..=5 {
            for d in 1..=n / 2 {
                polygons.push(Concrete::star_polygon(n, d));
                areas.push(polygon_area(n, d));
            }
        }

        for m in 0..polygons.len() {
            for n in 0..polygons.len() {
                test_volume(
                    &mut Concrete::duoprism(&polygons[m], &polygons[n]),
                    Some(areas[m] * areas[n]),
                )
            }
        }
    }

    #[test]
    fn duotegum() {
        let mut polygons = Vec::new();
        let mut areas = Vec::new();
        for n in 2..=5 {
            for d in 1..=n / 2 {
                polygons.push(Concrete::star_polygon(n, d));
                areas.push(polygon_area(n, d));
            }
        }

        for m in 0..polygons.len() {
            for n in 0..polygons.len() {
                test_volume(
                    &mut Concrete::duotegum(&polygons[m], &polygons[n]),
                    Some(areas[m] * areas[n] / 6.0),
                )
            }
        }
    }

    #[test]
    fn duocomb() {
        let mut polygons = Vec::new();
        let mut areas = Vec::new();
        for n in 2..=5 {
            for d in 1..=n / 2 {
                polygons.push(Concrete::star_polygon(n, d));
                areas.push(polygon_area(n, d));
            }
        }

        for m in 0..polygons.len() {
            for n in 0..polygons.len() {
                let volume = (m == 0 || n == 0).then(|| 0.0);
                test_volume(&mut Concrete::duocomb(&polygons[m], &polygons[n]), volume)
            }
        }
    }

    #[test]
    fn simplex() {
        for n in 0..=5 {
            test_volume(
                &mut Concrete::simplex(Rank::from(n)),
                Some(
                    ((n + 1) as Float / (1 << n) as Float).sqrt()
                        / crate::factorial(n as usize) as Float,
                ),
            );
        }
    }

    #[test]
    fn hypercube() {
        for n in 0..=5 {
            test_volume(&mut Concrete::hypercube(Rank::new(n)), Some(1.0));
        }
    }

    #[test]
    fn orthoplex() {
        for n in 0..=5 {
            test_volume(
                &mut Concrete::orthoplex(Rank::from(n)),
                Some(1.0 / crate::factorial(n) as Float),
            );
        }
    }
}
