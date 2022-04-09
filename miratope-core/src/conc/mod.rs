//! Declares the [`Concrete`] polytope type and all associated data structures.

pub mod cycle;
pub mod element_types;
pub mod faceting;
pub mod symmetry;

use std::{
    collections::{HashMap, HashSet},
    ops::{Index, IndexMut}, iter,
};

use super::{
    abs::{
        flag::{Flag, FlagChanges, FlagEvent, OrientedFlagIter},
        Abstract, ElementList, Ranked, SubelementList,
    },
    DualError, Polytope,
};
use crate::{
    abs::{AbstractBuilder, Element, ElementMap, Subelements, Superelements},
    float::Float,
    geometry::*,
};

use approx::{abs_diff_eq, abs_diff_ne};
use rayon::prelude::*;
use vec_like::*;

/// Represents a [concrete polytope](https://polytope.miraheze.org/wiki/Polytope),
/// which is an [`Abstract`] together with its corresponding vertices.
#[derive(Debug, Clone)]
pub struct Concrete {
    /// The list of vertices as points in Euclidean space.
    // todo: come up with a more compact representation, making use of the fact
    // all points have the same length?
    pub vertices: Vec<Point<f64>>,

    /// The underlying abstract polytope.
    pub abs: Abstract,
}

impl Index<usize> for Concrete {
    type Output = ElementList;

    /// Gets the list of elements with a given rank.
    fn index(&self, rank: usize) -> &Self::Output {
        &self.abs[rank]
    }
}

impl IndexMut<usize> for Concrete {
    /// Gets the list of elements with a given rank.
    fn index_mut(&mut self, rank: usize) -> &mut Self::Output {
        &mut self.abs[rank]
    }
}

impl Index<(usize, usize)> for Concrete {
    type Output = Element;

    /// Gets the list of elements with a given rank.
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.abs[index]
    }
}

impl IndexMut<(usize, usize)> for Concrete {
    /// Gets the list of elements with a given rank.
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.abs[index]
    }
}

impl Concrete {
    /// Initializes a new concrete polytope from a set of vertices and an
    /// underlying abstract polytope. Does some debug assertions on the input.
    pub fn new(vertices: Vec<Point<f64>>, abs: Abstract) -> Self {
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
    type DualError = DualError;

    fn abs(&self) -> &Abstract {
        &self.abs
    }

    fn abs_mut(&mut self) -> &mut Abstract {
        &mut self.abs
    }

    fn into_abs(self) -> Abstract {
        self.abs
    }

    /// Builds the unique polytope of rank −1.
    fn nullitope() -> Self {
        Self::new(Vec::new(), Abstract::nullitope())
    }

    /// Builds the unique polytope of rank 0.
    fn point() -> Self {
        Self::new(vec![Vec::new().into()], Abstract::point())
    }

    /// Builds a dyad with unit edge length, centered at the origin.
    fn dyad() -> Self {
        Self::dyad_with(1.0)
    }

    /// Builds a convex regular polygon with `n` sides and unit edge length,
    /// centered at the origin.
    fn polygon(n: usize) -> Self {
        Self::grunbaum_star_polygon(n, 1)
    }

    /// Returns the [dual](https://polytope.miraheze.org/wiki/Dual_polytope) of
    /// a polytope using the unit hypersphere, or the index of a facet through
    /// the origin if unsuccessful.
    fn try_dual(&self) -> Result<Self, Self::DualError> {
        let mut clone = self.clone();
        clone.try_dual_mut().map(|_| clone)
    }

    /// Builds the [dual](https://polytope.miraheze.org/wiki/Dual_polytope) of a
    /// polytope in place using the unit hypersphere. If unsuccessful, leaves
    /// the polytope unchanged and returns the index of a facet through the
    /// origin.
    fn try_dual_mut(&mut self) -> Result<(), Self::DualError> {
        self.try_dual_mut_with(&Hypersphere::unit(self.dim().unwrap_or(1)))
    }

    /// Builds the [Petrial](https://polytope.miraheze.org/wiki/Petrial) of a
    /// polytope in place. If unsuccessful, leaves the polytope unchanged and
    /// returns `false`.
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

    /// "Appends" a polytope into another, creating a compound polytope.
    ///
    /// # Panics
    /// This method will panic if the polytopes have different ranks.
    fn comp_append(&mut self, mut p: Self) {
        self.abs.comp_append(p.abs);
        self.vertices.append(&mut p.vertices);
    }

    /// Gets the element with a given rank and index as a polytope, or returns
    /// `None` if such an element doesn't exist.
    fn element(&self, rank: usize, idx: usize) -> Option<Self> {
        let (vertices, abs) = self.abs.element_and_vertices(rank, idx)?;

        Some(Self::new(
            vertices
                .into_iter()
                .map(|idx| self.vertices[idx].clone())
                .collect(),
            abs,
        ))
    }

    /// Gets the element figure with a given rank and index as a polytope.
    fn element_fig(&self, rank: usize, idx: usize) -> Result<Option<Self>, Self::DualError> {
        if rank <= self.rank() {
            // todo: this is quite inefficient for a small element figure since
            // we take the dual of the entire thing.
            if let Some(mut element_fig) = self.try_dual()?.element(self.rank() - rank, idx) {
                let subspace = Subspace::from_points(element_fig.vertices.iter());
                element_fig.flatten();
                element_fig.recenter_with(
                    &subspace.flatten(&subspace.project(&Point::zeros(self.dim().unwrap()))),
                );
                element_fig.try_dual_mut()?;
                return Ok(Some(element_fig));
            }
        }

        Ok(None)
    }

    // TODO: A method that builds an omnitruncate together with a map from flags
    // to vertices? We got some math details to figure out.
    fn omnitruncate(&self) -> Self {
        let (abs, flags) = self.abs.omnitruncate_and_flags();
        let element_vertices = self.avg_vertex_map();

        Self::new(
            flags
                .into_iter()
                .map(|flag| {
                    flag.into_iter()
                        .enumerate()
                        .skip(1)
                        .take(self.rank())
                        .map(|el| &element_vertices[el])
                        .sum()
                })
                .collect(),
            abs,
        )
    }

    /// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// with unit height from two polytopes. Does not offset either polytope.
    fn duopyramid(&self, other: &Self) -> Self {
        self.duopyramid_with(
            other,
            &Point::zeros(self.dim_or()),
            &Point::zeros(other.dim_or()),
            1.0,
        )
    }

    /// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
    /// from two polytopes.
    fn duoprism(&self, p: &Self) -> Self {
        Self::new(
            duoprism_vertices(&self.vertices, &p.vertices),
            self.abs.duoprism(&p.abs),
        )
    }

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum(&self, other: &Self) -> Self {
        Self::duotegum_with(
            self,
            other,
            &Point::zeros(self.dim_or()),
            &Point::zeros(other.dim_or()),
        )
    }

    /// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
    /// from two polytopes.
    fn duocomb(&self, other: &Self) -> Self {
        Self::new(
            duoprism_vertices(&self.vertices, &other.vertices),
            self.abs.duocomb(other.abs()),
        )
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

    /// Attempts to build an antiprism based on a given polytope. Uses the unit
    /// hypersphere to take the dual, and places the bases at a distance of 1.
    /// If it fails, it returns the index of a facet through the inversion
    /// center.
    ///
    /// If you want more control over the arguments, you can use
    /// [`Self::try_antiprism_with`].
    fn try_antiprism(&self) -> Result<Self, Self::DualError> {
        Self::try_antiprism_with(self, &Hypersphere::unit(self.dim().unwrap_or(1)), 1.0)
    }

    /// Builds a [simplex](https://polytope.miraheze.org/wiki/Simplex) with a
    /// given rank.
    fn simplex(rank: usize) -> Self {
        if rank == 0 {
            Self::nullitope()
        } else {
            let dim = rank - 1;
            let mut vertices = Vec::with_capacity(rank);

            // Adds all points with a single entry equal to √2/2, and all others
            // equal to 0.
            for i in 0..dim {
                let mut v = Point::zeros(dim);
                v[i] = f64::HALF_SQRT_2;
                vertices.push(v);
            }

            // Adds the remaining vertex, all of whose coordinates are equal.
            let dim_f = dim as f64;
            let a = (1.0 - (dim_f + 1.0).fsqrt()) * f64::HALF_SQRT_2 / dim_f;
            vertices.push(vec![a; dim].into());

            let mut simplex = Concrete::new(vertices, Abstract::simplex(rank));
            simplex.recenter();
            simplex
        }
    }

    /// Builds an [orthoplex](https://polytope.miraheze.org/wiki/Orthoplex) with
    /// a given rank.
    fn orthoplex(rank: usize) -> Self {
        if rank == 0 {
            Self::nullitope()
        } else {
            Self::multitegum(iter::repeat(&Self::dyad_with(f64::SQRT_2)).take(rank - 1))
        }
    }

    /// Splits compound faces into their components.
    fn untangle_faces(&mut self) {
        self.abs.untangle_faces();
    }
}

/// Generates the vertices for either a tegum or a pyramid product with two
/// given vertex sets and a given height.
///
/// The vertices are the padded vertices of `p`, followed by the padded
/// vertices of `q`.
fn duopyramid_vertices(
    p: &[Point<f64>],
    q: &[Point<f64>],
    p_pad: &Point<f64>,
    q_pad: &Point<f64>,
    height: f64,
    tegum: bool,
) -> Vec<Point<f64>> {
    // Duotegums with points should just return the original polytopes.
    if tegum {
        if p.get(0).map(|vp| vp.len()) == Some(0) {
            return q.to_owned();
        } else if q.get(0).map(|vq| vq.len()) == Some(0) {
            return p.to_owned();
        }
    }

    let half_height = height / 2.0;

    // To every point in p, we append zeros to the right.
    p.iter()
        .map(|vp| {
            let mut v: Vec<_> = vp.iter().copied().chain(q_pad.iter().copied()).collect();
            if !tegum {
                v.push(half_height);
            }
            v.into()
        })
        // To every point in q, we append zeros to the left.
        .chain(q.iter().map(|vq| {
            let mut v: Vec<_> = p_pad.iter().copied().chain(vq.iter().copied()).collect();
            if !tegum {
                v.push(-half_height);
            }
            v.into()
        }))
        .collect()
}

/// Generates the vertices for a duoprism with two given vertex sets.
fn duoprism_vertices(p: &[Point<f64>], q: &[Point<f64>]) -> Vec<Point<f64>> {
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

    /// Returns a reference to the concrete vertices of the polytope.
    fn vertices(&self) -> &Vec<Point<f64>> {
        &self.con().vertices
    }

    /// Returns a mutable reference to the concrete vertices of the polytope.
    fn vertices_mut(&mut self) -> &mut Vec<Point<f64>> {
        &mut self.con_mut().vertices
    }

    /// Returns the number of dimensions of the space the polytope lives in,
    /// or `None` in the case of the nullitope.
    fn dim(&self) -> Option<usize> {
        (!self.is_nullitope()).then(|| self.vertices()[0].len())
    }

    /// Returns the number of dimensions of the space the polytope lives in,
    /// or 0 in the case of the nullitope.
    fn dim_or(&self) -> usize {
        self.dim().unwrap_or(0)
    }

    /// Builds a dyad with a specified height.
    fn dyad_with(height: f64) -> Self;

    /// Builds the Grünbaumian star polygon `{n / d}` with unit circumradius,
    /// rotated by an angle.
    fn grunbaum_star_polygon_with_rot(n: usize, d: usize, rot: f64) -> Self;

    /// Builds the Grünbaumian star polygon `{n / d}` with unit circumradius. If
    /// `n` and `d` have a common factor, the result is a multiply-wound
    /// polygon.
    fn grunbaum_star_polygon(n: usize, d: usize) -> Self {
        Self::grunbaum_star_polygon_with_rot(n, d, 0.0)
    }

    /// Builds the star polygon `{n / d}` with unit circumradius. If `n` and `d`
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
        let angle = f64::TAU / f64::usize(n);

        Self::compound(
            (0..gcd).into_iter().map(|k| {
                Self::grunbaum_star_polygon_with_rot(n / gcd, d / gcd, f64::usize(k) * angle)
            }),
        )
    }

    /// Builds the star polygon `{n / d}` with edge length `a`. If `n` and `d`
    /// have a common factor, the result is a compound.
    ///
    /// # Panics
    /// Will panic if either `n < 2` or if `d < 1`, in which case there's
    /// nothing sensible to do.
    fn star_polygon_with_edge(n: usize, d: usize, a: f64) -> Self {
        let mut p = Self::star_polygon(n, d);
        p.scale(a / 2.0 / (f64::PI * d as f64 / n as f64).fsin());
        p
    }

    /// Scales a polytope by a given factor.
    fn scale(&mut self, k: f64) {
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
    fn recenter_with(&mut self, p: &Point<f64>) {
        for v in self.vertices_mut() {
            *v -= p;
        }
    }

    /// Applies a linear transformation to all vertices of a polytope.
    fn apply(mut self, m: &Matrix<f64>) -> Self {
        for v in self.vertices_mut() {
            *v = m * v as &_;
        }

        self
    }

    /// Returns an arbitrary truncate of a polytope.
    fn truncate_with(&self, truncate_type: Vec<usize>, depth: Vec<f64>) -> Self;

    /// Calculates the circumsphere of a polytope. Returns `None` if the
    /// polytope isn't circumscribable.
    fn circumsphere(&self) -> Option<Hypersphere<f64>> {
        let mut vertices = self.vertices().iter();

        let first_vertex = vertices.next()?.clone();
        let mut center = first_vertex.clone();
        let mut subspace = Subspace::new(first_vertex.clone());

        for vertex in vertices {
            // If the new vertex does not lie on the hyperplane of the others:
            if let Some(basis_vector) = subspace.add(vertex) {
                // Calculates the new circumcenter.
                let distance: f64 = ((&center - vertex).norm_squared()
                    - (&center - &first_vertex).norm_squared())
                    / (2.0 * (vertex - &first_vertex).dot(basis_vector));

                center += basis_vector * distance;
            }
            // If the new vertex lies on the others' hyperplane, but is not at
            // the correct distance from the first vertex:
            else if abs_diff_ne!(
                (&center - &first_vertex).norm(),
                (&center - vertex).norm(),
                epsilon = f64::EPS
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
    fn gravicenter(&self) -> Option<Point<f64>> {
        (!self.is_nullitope())
            .then(|| self.vertices().iter().sum::<Point<f64>>() / (self.vertex_count()) as f64)
    }

    /// Gets the least and greatest distance of a vertex of the polytope,
    /// measuring from a specified direction, or returns `None` in the case of
    /// the nullitope.
    fn minmax(&self, direction: Vector<f64>) -> Option<(f64, f64)> {
        use itertools::{Itertools, MinMaxResult::*};

        let hyperplane = Hyperplane::new(direction, 0.0);

        match self
            .vertices()
            .iter()
            .map(|v| ordered_float::OrderedFloat(hyperplane.distance(v)))
            .minmax()
        {
            // The vertex vector is empty.
            NoElements => None,

            // The single vertex gives both the minimum and maximum distance.
            OneElement(x) => Some((x.0, x.0)),

            // The minimum and maximum distances.
            MinMax(x, y) => Some((x.0, y.0)),
        }
    }

    /// Returns a map from the elements in a polytope to a crude average of
    /// their vertices. Specifically, every vertex is mapped to itself, and
    /// every other element is mapped to the average of the images of its
    /// subelements.
    fn avg_vertex_map(&self) -> ElementMap<Point<f64>> {
        // Maps every element of the polytope to one of its vertices.
        let mut map = ElementMap::new();
        map.push(Vec::new());

        // Vertices map to themselves.
        if self.rank() != 0 {
            map.push(self.vertices().clone());
        }

        // Every other element maps to the average of the locations of their
        // subelements.
        for (r, elements) in self.ranks().iter().enumerate().skip(2) {
            map.push(
                elements
                    .iter()
                    .map(|el| {
                        el.subs
                            .iter()
                            .map(|&idx| &map[(r - 1, idx)])
                            .sum::<Point<f64>>()
                            / f64::usize(el.subs.len())
                    })
                    .collect(),
            );
        }

        map
    }

    /// Returns the length of a given edge.
    fn edge_len(&self, idx: usize) -> Option<f64> {
        let edge = self.get_element(2, idx)?;
        Some((&self.vertices()[edge.subs[0]] - &self.vertices()[edge.subs[1]]).norm())
    }

    /// Checks whether a polytope is equilateral to a fixed precision, and with
    /// a specified edge length.
    fn is_equilateral_with(&self, len: f64) -> bool {
        (0..self.edge_count())
            .all(|idx| abs_diff_eq!(self.edge_len(idx).unwrap(), len, epsilon = f64::EPS))
    }

    /// Checks whether a polytope is equilateral to a fixed precision.
    fn is_equilateral(&self) -> bool {
        self.edge_count() == 0 || self.is_equilateral_with(self.edge_len(0).unwrap())
    }

    /// I haven't actually implemented this in the general case.
    ///
    /// # Todo
    /// Maybe make this work in the general case?
    fn midradius(&self) -> f64 {
        let edge_subs = &self[(2, 0)].subs;
        (&self.vertices()[edge_subs[0]] + &self.vertices()[edge_subs[1]]).norm() / 2.0
    }

    /// Builds the dual of a polytope with a given reciprocation sphere in
    /// place, or does nothing in case any facets go through the reciprocation
    /// center. In case of failure, returns the index of the facet through the
    /// projection center.
    fn try_dual_mut_with(&mut self, sphere: &Hypersphere<f64>) -> Result<(), Self::DualError>;

    /// Returns the dual of a polytope with a given reciprocation sphere, or
    /// `None` if any facets pass through the reciprocation center.
    fn try_dual_with(&self, sphere: &Hypersphere<f64>) -> Result<Self, Self::DualError> {
        let mut clone = self.clone();
        clone.try_dual_mut_with(sphere).map(|_| clone)
    }

    /// Builds a pyramid with a specified apex.
    fn pyramid_with(&self, apex: Point<f64>) -> Self;

    /// Builds a prism with a specified height.
    fn prism_with(&self, height: f64) -> Self;

    /// Builds a uniform prism from an {n/d} polygon.
    fn uniform_prism(n: usize, d: usize) -> Self {
        Self::star_polygon_with_edge(n, d, 1.0).prism()
    }

    /// Builds a tegum with two specified apices.
    fn tegum_with(&self, apex1: Point<f64>, apex2: Point<f64>) -> Self;

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism),
    /// using the specified sets of vertices for the base and the dual base.
    ///
    /// The vertices of the base should be specified in the same order as those
    /// of the original polytope. The vertices of the dual face should be
    /// specified in the same order as the facets of the original polytope.
    fn antiprism_with_vertices<I: Iterator<Item = Point<f64>>, J: Iterator<Item = Point<f64>>>(
        &self,
        vertices: I,
        dual_vertices: J,
    ) -> Self;

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. Uses the specified [`Hypersphere`] to build
    /// the dual base, and separates the bases by the given height.
    fn try_antiprism_with(
        &self,
        sphere: &Hypersphere<f64>,
        height: f64,
    ) -> Result<Self, Self::DualError> {
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
    fn antiprism_with(&self, sphere: &Hypersphere<f64>, height: f64) -> Self {
        self.try_antiprism_with(sphere, height).unwrap()
    }

    /// Builds a uniform antiprism of unit edge length.
    fn uniform_antiprism(n: usize, d: usize) -> Self {
        let polygon = Self::star_polygon(n, d);

        // Appropriately scaled antiprism.
        if n != 2 * d {
            let angle = f64::PI * f64::usize(d) / f64::usize(n);
            let cos = angle.fcos();
            let height = ((cos - (2.0 * angle).fcos()) * 2.0).fsqrt();

            let mut antiprism = polygon.antiprism_with(
                &Hypersphere::with_squared_radius(Point::zeros(2), cos),
                height,
            );
            antiprism.scale(0.5 / (f64::PI * d as f64 / n as f64).fsin());

            antiprism
        }
        // Digon compounds are a special case.
        else {
            let half_height = f64::HALF_SQRT_2;
            let vertices = polygon.vertices().iter().map(|v| v.push(-half_height));
            let dual_vertices = polygon
                .vertices()
                .iter()
                .map(|v| vec![v[1], -v[0], half_height].into());

            let mut antiprism = polygon.antiprism_with_vertices(vertices, dual_vertices);
            antiprism.scale(0.5);

            antiprism
        }
    }

    /// Gets the references to the (geometric) vertices of an element on the
    /// polytope.
    fn element_vertices_ref(&self, rank: usize, idx: usize) -> Option<Vec<&Point<f64>>> {
        Some(
            self.abs()
                .element_vertices(rank, idx)?
                .iter()
                .map(|&v| &self.vertices()[v])
                .collect(),
        )
    }

    /// Generates a duopyramid from two given polytopes with a given height and
    /// a given offset.
    fn duopyramid_with(
        &self,
        other: &Self,
        self_offset: &Point<f64>,
        other_offset: &Point<f64>,
        height: f64,
    ) -> Self;

    /// Generates a duopyramid from two given polytopes with a given offset.
    fn duotegum_with(p: &Self, q: &Self, p_offset: &Point<f64>, q_offset: &Point<f64>) -> Self;

    /// Computes the volume of a polytope by adding up the contributions of all
    /// flags. Returns `None` if the volume is undefined.
    ///
    /// # Panics
    /// You must call [`Polytope::element_sort`] before calling this method.
    fn volume(&self) -> Option<f64> {
        let rank = self.rank();

        // We leave the nullitope's volume undefined.
        if rank == 0 {
            return None;
        }

        // The flattened vertices (may possibly be the original vertices).
        let subspace = Subspace::from_points(self.vertices().iter());
        let flat_vertices = subspace.flatten_vec(self.vertices());

        match flat_vertices.get(0)?.len().cmp(&(rank - 1)) {
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
        let vertex_map = self.vertex_map();
        let mut volume = 0.0;

        // All of the flags we've found so far.
        let mut all_flags = HashSet::new();

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
                                rank - 1,
                                rank - 1,
                                oriented_flag
                                    .into_iter()
                                    .enumerate()
                                    .skip(1)
                                    .take(rank - 1)
                                    .map(|(rank, idx)| &flat_vertices[vertex_map[(rank, idx)]])
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
                volume += component_volume.fabs();
            }
        }

        Some(volume / f64::u32(crate::factorial(rank - 1)))
    }

    /// Computes the volume of a polytope by adding up the contributions of all
    /// flags. Returns `None` if the volume is undefined.
    fn volume_mut(&mut self) -> Option<f64> {
        self.element_sort();
        self.volume()
    }

    /// Projects the vertices of the polytope into the lowest dimension possible.
    /// If the polytope's subspace is already of full rank, this is a no-op.
    fn flatten(&mut self);

    /// Flattens the vertices of a polytope into a specified subspace.
    fn flatten_into(&mut self, subspace: &Subspace<f64>);

    /// Slices the polytope through a given plane.
    fn cross_section(&self, slice: &Hyperplane<f64>) -> Self;
	
	/// Checks if is fissary.
    fn is_fissary(&mut self) -> bool;
	
}

impl ConcretePolytope for Concrete {
    fn con(&self) -> &Concrete {
        self
    }

    fn con_mut(&mut self) -> &mut Concrete {
        self
    }

    /// Builds a dyad with a specified height.
    fn dyad_with(height: f64) -> Self {
        let half_height = height / 2.0;

        Self::new(
            vec![vec![-half_height].into(), vec![half_height].into()],
            Abstract::dyad(),
        )
    }

    /// Builds the Grünbaumian star polygon `{n / d}` with unit circumradius,
    /// rotated by an angle.
    fn grunbaum_star_polygon_with_rot(n: usize, d: usize, rot: f64) -> Self {
        assert!(n >= 2);
        assert!(d >= 1);

        let angle = f64::TAU * f64::usize(d) / f64::usize(n);

        Self::new(
            (0..n)
                .into_iter()
                .map(|k| {
                    let (sin, cos) = (f64::usize(k) * angle + rot).fsin_cos();
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
    fn try_dual_mut_with(&mut self, sphere: &Hypersphere<f64>) -> Result<(), Self::DualError> {
        // If we're dealing with a nullitope, the dual is itself.
        let rank = self.rank();
        if rank == 0 {
            return Ok(());
        }

        // We project the sphere's center onto the polytope's hyperplane to
        // avoid skew weirdness.
        let h = Subspace::from_points(self.vertices.iter());
        let o = h.project(&sphere.center);

        let mut projections;

        // We project our inversion center onto each of the facets.
        if rank >= 2 {
            let facet_count = self.facet_count();
            projections = Vec::with_capacity(facet_count);

            (0..facet_count)
                .into_par_iter()
                .map(|idx| {
                    Subspace::from_points(
                        self.element_vertices_ref(rank - 1, idx)
                            .unwrap()
                            .into_iter(),
                    )
                    .project(&o)
                })
                .collect_into_vec(&mut projections);
        } else {
            projections = self.vertices.clone();
        }

        // Reciprocates the projected points.
        for (idx, v) in projections.iter_mut().enumerate() {
            if !sphere.reciprocate_mut(v) && rank != 1 {
                return Err(DualError(idx));
            }
        }

        self.vertices = projections;
        self.abs.dual_mut();
        Ok(())
    }

    /// Builds a pyramid with a specified apex.
    fn pyramid_with(&self, apex: Point<f64>) -> Self {
        let mut poly = self.pyramid();
        *poly.vertices.last_mut().unwrap() = apex;
        poly
    }

    /// Builds a prism with a specified height.
    fn prism_with(&self, height: f64) -> Self {
        self.duoprism(&Self::dyad_with(height))
    }

    /// Builds a tegum with two specified apices.
    fn tegum_with(&self, apex1: Point<f64>, apex2: Point<f64>) -> Self {
        let mut poly = self.tegum();
        let v = poly.vertices.len();
        poly.vertices[v - 1] = apex1;
        poly.vertices[v - 2] = apex2;
        poly
    }

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism),
    /// using the specified sets of vertices for the base and the dual base.
    ///
    /// The vertices of the base should be specified in the same order as those
    /// of the original polytope. The vertices of the dual face should be
    /// specified in the same order as the facets of the original polytope.
    fn antiprism_with_vertices<I: Iterator<Item = Point<f64>>, J: Iterator<Item = Point<f64>>>(
        &self,
        vertices: I,
        dual_vertices: J,
    ) -> Self {
        let (abs, vertex_indices, dual_vertex_indices) = self.abs.antiprism_and_vertices();
        let vertex_count = abs.vertex_count();

        // TODO: is it worth getting into the dark arts of uninitialized buffers?
        let mut new_vertices = vec![Vec::new().into(); vertex_count];

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
        &self,
        other: &Self,
        self_offset: &Point<f64>,
        other_offset: &Point<f64>,
        height: f64,
    ) -> Self {
        Self::new(
            duopyramid_vertices(
                &self.vertices,
                &other.vertices,
                self_offset,
                other_offset,
                height,
                false,
            ),
            Abstract::duopyramid(&self.abs, &other.abs),
        )
    }

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum_with(p: &Self, q: &Self, p_offset: &Point<f64>, q_offset: &Point<f64>) -> Self {
        Self::new(
            duopyramid_vertices(&p.vertices, &q.vertices, p_offset, q_offset, 0.0, true),
            p.abs.duotegum(&q.abs),
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
    fn flatten_into(&mut self, subspace: &Subspace<f64>) {
        if !subspace.is_full_rank() {
            for v in &mut self.vertices {
                *v = subspace.flatten(v);
            }
        }
    }

    /// Takes the cross-section of a polytope through a given hyperplane.
    ///
    /// # Todo
    /// We should make this function take a general [`Subspace`] instead.
    fn cross_section(&self, slice: &Hyperplane<f64>) -> Self {
        if self.rank() < 4 {
            unreachable!();
        }

        let mut vertices = Vec::new();
        let mut ranks = Vec::with_capacity(self.rank());

        // We map all indices of k-elements in the original polytope to the
        // indices of the new (k-1)-elements resulting from taking their
        // intersections with the slicing hyperplane.
        let mut hash_element = HashMap::new();

        // Determines the vertices of the cross-section.
        for (idx, edge) in self[2].iter().enumerate() {
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
        for r in 3..self.rank() {
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
        let (first, last) = ranks.split_at_mut(3);

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
                        new_edges.push(vec![v0, v1].into());

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
        // TODO: no need for ranks, just use the builder directly.
        let mut builder = AbstractBuilder::new();
        for subelements in ranks {
            builder.push(subelements);
        }

        // Safety: TODO shit, this one's complicated... I'll come back to it.
        unsafe {
            let mut abs = builder.build();
            abs.untangle_faces();
            Self::new(vertices, abs)
        }
    }

    fn truncate_with(&self, truncate_type: Vec<usize>, depth: Vec<f64>) -> Self {
        let (abs, subflags) = self.abs().truncate_and_flags(truncate_type.clone());
        let element_vertices = self.avg_vertex_map();

        let mut vertex_coords = Vec::<Point<f64>>::new();
        for subflag in subflags {
            let mut vector = Point::<f64>::from_vec(vec![0.0; self.rank() - 1]);
            for (r, i) in subflag.iter().enumerate() {
                vector += element_vertices[truncate_type[r] + 1][*i].clone() * depth[truncate_type[r]];
            }
            vertex_coords.push(vector);
        }
        //dbg!(abs.clone());

        Self::new(vertex_coords, abs)
    }
	
	
	/// Checks if is fissary.
    fn is_fissary(&mut self) -> bool {
        let types = self.element_types();
		
		self.abs.element_sort();
		
        let mut i = 1;
		while i < types.len() {
			if i == self.rank() {
				break;
			}
			let mut j = 0;
			while j < types[i].len() {
				let example = types[i][j].example;
				
				let mut element = self.abs.element(i, example).unwrap();
				
				element.element_sort();
				if self.element(i, example).unwrap().is_fissary() && !element.is_compound() {
					return true;
				}
				
				let mut figure = self.abs.element_fig(i, example).unwrap().unwrap();
				figure.element_sort();
				if figure.is_compound() {
					return true;
				}
				j = j+1;
			}
			i = i+1;
		}
		return false;
    }
}

#[cfg(test)]
mod tests {
    use super::{Concrete, ConcretePolytope};
    use crate::{float::Float, Polytope};

    use approx::abs_diff_eq;

    /// Tests that a polytope has an expected volume.
    fn test_volume(mut poly: Concrete, volume: Option<f64>) {
        poly.element_sort();

        if let Some(poly_volume) = poly.volume() {
            let volume = volume.expect(&format!(
                "Expected no volume for {}, found volume {}!",
                "TBA: name", poly_volume
            ));

            assert!(
                abs_diff_eq!(poly_volume, volume, epsilon = f64::EPS),
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
        test_volume(Concrete::nullitope(), None)
    }

    #[test]
    fn point() {
        test_volume(Concrete::point(), Some(1.0));
    }

    #[test]
    fn dyad() {
        test_volume(Concrete::dyad(), Some(1.0));
    }

    fn polygon_area(n: usize, d: usize) -> f64 {
        let n = n as f64;
        let d = d as f64;
        n * (d * f64::TAU / n).sin() / 2.0
    }

    fn test_compound(mut p: Concrete, volume: Option<f64>) {
        p.comp_append(p.clone());
        test_volume(p, volume)
    }

    #[test]
    fn compounds() {
        test_compound(Concrete::nullitope(), None);
        test_compound(Concrete::point(), Some(1.0));
        test_compound(Concrete::polygon(3), Some(2.0 * polygon_area(3, 1)));
        test_compound(Concrete::hypercube(4), Some(2.0));
    }

    #[test]
    fn polygon() {
        for n in 2..=10 {
            for d in 1..=n / 2 {
                test_volume(Concrete::star_polygon(n, d), Some(polygon_area(n, d)));
            }
        }
    }

    fn polygons_areas() -> (Vec<Concrete>, Vec<f64>) {
        let mut polygons = Vec::new();
        let mut areas = Vec::new();
        for n in 2..=5 {
            for d in 1..=n / 2 {
                polygons.push(Concrete::star_polygon(n, d));
                areas.push(polygon_area(n, d));
            }
        }

        (polygons, areas)
    }

    #[test]
    fn duopyramid() {
        let (polygons, areas) = polygons_areas();

        for m in 0..polygons.len() {
            for n in 0..polygons.len() {
                test_volume(
                    Concrete::duopyramid(&polygons[m], &polygons[n]),
                    Some(areas[m] * areas[n] / 30.0),
                )
            }
        }
    }

    #[test]
    fn duoprism() {
        let (polygons, areas) = polygons_areas();

        for m in 0..polygons.len() {
            for n in 0..polygons.len() {
                test_volume(
                    polygons[m].duoprism(&polygons[n]),
                    Some(areas[m] * areas[n]),
                )
            }
        }
    }

    #[test]
    fn duotegum() {
        let (polygons, areas) = polygons_areas();

        for m in 0..polygons.len() {
            for n in 0..polygons.len() {
                test_volume(
                    Concrete::duotegum(&polygons[m], &polygons[n]),
                    Some(areas[m] * areas[n] / 6.0),
                )
            }
        }
    }

    #[test]
    fn duocomb() {
        let (polygons, _) = polygons_areas();

        for m in 0..polygons.len() {
            for n in 0..polygons.len() {
                test_volume(
                    Concrete::duocomb(&polygons[m], &polygons[n]),
                    (m == 0 || n == 0).then(|| 0.0),
                )
            }
        }
    }

    #[test]
    fn simplex() {
        for n in 1..=6 {
            test_volume(
                Concrete::simplex(n),
                Some((n as f64 / (1 << (n - 1)) as f64).sqrt() / crate::factorial(n - 1) as f64),
            );
        }
    }

    #[test]
    fn hypercube() {
        for n in 1..=6 {
            test_volume(Concrete::hypercube(n), Some(1.0));
        }
    }

    #[test]
    fn orthoplex() {
        for n in 1..=6 {
            test_volume(
                Concrete::orthoplex(n),
                Some(1.0 / crate::factorial(n - 1) as f64),
            );
        }
    }
}
