//! Contains the methods that take a polytope and turn it into a mesh.

use std::collections::HashMap;

use super::Concrete;
use crate::{
    geometry::{Point, Subspace, Vector},
    polytope::{
        r#abstract::{
            elements::{ElementList, Subsupelements},
            rank::Rank,
        },
        Polytope,
    },
    ui::camera::ProjectionType,
    Consts, Float,
};

use bevy::{
    prelude::Mesh,
    render::{mesh::Indices, pipeline::PrimitiveTopology},
};
use lyon::{math::point, path::Path, tessellation::*};

/// Represents a set with at most two values.
#[derive(Clone, Copy)]
pub enum Pair<T> {
    /// No entry.
    None,

    /// One entry.
    One(T),

    /// Two entries.
    Two(T, T),
}

impl<T> Default for Pair<T> {
    fn default() -> Self {
        Self::None
    }
}

impl<T> Pair<T> {
    /// Returns the number of elements stored in the pair.
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
        }
    }

    /// Pushes a value onto the pair by copy.
    ///
    /// # Panics
    /// The code will panic if you attempt to push a value onto a pair that
    /// already has two elements in it.
    pub fn push(&mut self, value: T)
    where
        T: Copy,
    {
        *self = match self {
            Self::None => Self::One(value),
            Self::One(first) => Self::Two(*first, value),
            Self::Two(_, _) => panic!("Can't push a value onto a pair with two elements!"),
        };
    }
}

/// A helper struct to build a cycle of vertices from a polygonal path.
///
/// Internally, each vertex is mapped to a [`Pair`], which stores the (at most)
/// two other vertices it's connected to. By traversing this map, we're able to
/// recover the vertex cycles.
pub struct CycleBuilder(HashMap<usize, Pair<usize>>);

impl CycleBuilder {
    /// Initializes a cycle builder with a given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    /// Returns the number of vertices in the vertex loop.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns a reference to the edge associated to a vertex, or `None` if it
    /// doesn't exist.
    fn get(&self, idx: usize) -> Option<&Pair<usize>> {
        self.0.get(&idx)
    }

    /// Removes the entry associated to a given vertex and returns it, or `None`
    /// if no such entry exists.
    fn remove(&mut self, idx: usize) -> Option<Pair<usize>> {
        self.0.remove(&idx)
    }

    /// Returns a mutable reference to the edge associated to a vertex, adding
    /// it if it doesn't exist.
    fn get_mut(&mut self, idx: usize) -> &mut Pair<usize> {
        use std::collections::hash_map::Entry;

        match self.0.entry(idx) {
            // Returns a reference to the entry.
            Entry::Occupied(entry) => entry.into_mut(),

            // Adds the entry, returns the reference to its value.
            Entry::Vacant(entry) => entry.insert(Pair::None),
        }
    }

    /// Pushes a pair of vertices into the vertex loop.
    pub fn push(&mut self, vertex0: usize, vertex1: usize) {
        self.get_mut(vertex0).push(vertex1);
        self.get_mut(vertex1).push(vertex0);
    }

    /// Returns the indices of the two vertices adjacent to a given one.
    ///
    /// # Panics
    /// This method will panic if there are less than two elements adjacent to
    /// the specified one.
    pub fn get_remove(&mut self, idx: usize) -> (usize, usize) {
        let pair = self.remove(idx).unwrap_or_default();

        if let Pair::Two(v0, v1) = pair {
            (v0, v1)
        } else {
            panic!("Expected 2 elements in pair, found {}.", pair.len())
        }
    }

    /// Cycles through the vertex loop, returns the vector of vertices in cyclic
    /// order.
    pub fn cycles(&mut self) -> Vec<Cycle> {
        let mut cycles = Vec::new();

        // While there's some vertex from which we haven't generated a cycle:
        while let Some((&init, _)) = self.0.iter().next() {
            let mut cycle = Cycle::with_capacity(self.len());
            let mut prev = init;
            let mut cur = self.get_remove(prev).0;

            cycle.push(cur);

            // We traverse the polygon, finding the next vertex over and over, until
            // we reach the initial vertex again.
            loop {
                // The two candidates for the next vertex.
                let (next0, next1) = self.get_remove(cur);

                let next_is_next1 = next0 == prev;
                prev = cur;

                // We go to whichever adjacent vertex isn't equal to the one we were
                // previously at.
                if next_is_next1 {
                    cycle.push(next1);
                    cur = next1;
                } else {
                    cycle.push(next0);
                    cur = next0;
                };

                // Whenever we reach the initial vertex, we break out of the loop.
                if cur == init {
                    break;
                }
            }

            cycles.push(cycle);
        }

        cycles
    }
}

/// Represents a cyclic list of vertex indices, which may then be turned into a
/// path and tessellated.
pub struct Cycle(Vec<usize>);

impl std::ops::Index<usize> for Cycle {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Cycle {
    /// Initializes a new cycle with a given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Pushes a vertex onto the cycle.
    pub fn push(&mut self, value: usize) {
        self.0.push(value);
    }

    /// Returns the length of the cycle.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Attempts to turn the cycle into a 2D path, which can then be given to
    /// the tessellator. Uses the specified vertex list to grab the coordinates
    /// of the vertices on the path.
    ///
    /// If the cycle isn't 2D, we return `None`.
    pub fn path(&self, vertices: &[Point]) -> Option<Path> {
        let dim = vertices[0].len();
        let mut cycle_iter = self.0.iter().map(|&v| &vertices[v]);

        // We don't bother with any polygons that aren't in 2D space.
        let s = Subspace::from_points_with(cycle_iter.clone(), 2)?;

        // We find the two axis directions most convenient for projecting down.
        // Convenience is measured as the length of an axis vector projected
        // down onto the plane our cycle lies in.

        // The index of the axis vector that gives the largest length when
        // projected, and that such length.
        let mut idx0 = 0;
        let mut len0 = 0.0;

        // The index of the axis vector that gives the second largest length
        // when projected, and that such length.
        let mut idx1 = 0;
        let mut len1 = 0.0;

        let mut e = Point::zeros(dim);
        for i in 0..dim {
            e[i] = 1.0;

            let len = s.project(&e).norm();
            // This is the largest length we've found so far.
            if len > len0 {
                len1 = len0;
                idx1 = idx0;
                len0 = len;
                idx0 = i;
            }
            // This is the second largest length we've found so far.
            else if len > len1 {
                len1 = len;
                idx1 = i;
            }

            e[i] = 0.0;
        }

        // Converts a point in the polytope to a point in the path via
        // orthogonal projection at our convenient axes.
        let path_point = |v: &Point| point(v[idx0] as f32, v[idx1] as f32);

        // We build a path from the polygon.
        let mut builder = Path::builder();
        let v = cycle_iter.next().unwrap();
        builder.begin(path_point(v));

        for v in cycle_iter {
            builder.line_to(path_point(v));
        }

        builder.close();
        Some(builder.build())
    }
}

/// Represents a triangulation of the faces of a [`Concrete`]. It stores the
/// vertex indices that make up the triangulation of the polytope, as well as
/// the extra vertices that may be needed to represent it.
struct Triangulation {
    /// Extra vertices that might be needed for the triangulation.
    extra_vertices: Vec<Point>,

    /// Indices of the vertices that make up the triangles.
    triangles: Vec<u16>,
}

impl Triangulation {
    /// Creates a new triangulation from a polytope.
    fn new(polytope: &Concrete) -> Triangulation {
        let mut extra_vertices = Vec::new();
        let mut triangles = Vec::new();

        let empty_els = ElementList::new();

        // Either returns a reference to the element list of a given rank, or
        // returns a reference to an empty element list.
        let elements_or = |r| polytope.abs.ranks.get(r).unwrap_or(&empty_els);

        let edges = elements_or(Rank::new(1));
        let faces = elements_or(Rank::new(2));

        let concrete_vertex_len = polytope.vertices.len() as u16;

        // We render each face separately.
        for face in faces.iter() {
            let mut vertex_loop = CycleBuilder::with_capacity(face.subs.len());

            // We first figure out the vertices in order.
            for [v0, v1] in face.subs.iter().map(|&i| {
                let subs = &edges[i].subs;
                let len = subs.len();

                debug_assert_eq!(len, 2, "Edge has {} subelements, expected 2.", len);
                [subs[0], subs[1]]
            }) {
                vertex_loop.push(v0, v1);
            }

            // We tesselate this path.
            for cycle in vertex_loop.cycles() {
                if let Some(path) = cycle.path(&polytope.vertices) {
                    let mut geometry: VertexBuffers<_, u16> = VertexBuffers::new();

                    // Configures all of the options of the tessellator.
                    FillTessellator::new()
                        .tessellate_with_ids(
                            path.id_iter(),
                            &path,
                            None,
                            &FillOptions::with_fill_rule(Default::default(), FillRule::EvenOdd)
                                .with_tolerance(f32::EPS),
                            &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex| {
                                vertex.sources().next().unwrap()
                            }),
                        )
                        .unwrap();

                    // We map the output vertices to the original ones, and add any
                    // extra vertices that may be needed.
                    let mut vertex_hash = HashMap::new();

                    for (new_id, vertex_source) in geometry.vertices.into_iter().enumerate() {
                        let new_id = new_id as u16;

                        match vertex_source {
                            // This is one of the concrete vertices of the polytope.
                            VertexSource::Endpoint { id } => {
                                vertex_hash.insert(new_id, cycle[id.to_usize()] as u16);
                            }

                            // This is a new vertex that has been added to the tesselation.
                            VertexSource::Edge { from, to, t } => {
                                let from = &polytope.vertices[cycle[from.to_usize()]];
                                let to = &polytope.vertices[cycle[to.to_usize()]];

                                let t = t as Float;
                                let p = from * (1.0 - t) + to * t;

                                vertex_hash.insert(
                                    new_id,
                                    concrete_vertex_len + extra_vertices.len() as u16,
                                );

                                extra_vertices.push(p);
                            }
                        }
                    }

                    // Add all of the new indices we've found onto the triangle vector.
                    for new_idx in geometry
                        .indices
                        .iter()
                        .map(|idx| *vertex_hash.get(idx).unwrap())
                    {
                        triangles.push(new_idx);
                    }
                }
            }
        }

        Self {
            extra_vertices,
            triangles,
        }
    }
}

/// Generates normals from a set of vertices by just projecting radially from
/// the origin.
fn normals(vertices: &[[f32; 3]]) -> Vec<[f32; 3]> {
    vertices
        .iter()
        .map(|n| {
            let sq_norm = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
            if sq_norm < f32::EPS {
                [0.0, 0.0, 0.0]
            } else {
                let norm = sq_norm.sqrt();
                let mut n = *n;
                n[0] /= norm;
                n[1] /= norm;
                n[2] /= norm;
                n
            }
        })
        .collect()
}

/// Returns an empty mesh.
fn empty_mesh() -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::LineList);
    mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0; 3]]);
    mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3]]);
    mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0; 2]]);
    mesh.set_indices(Some(Indices::U16(Vec::new())));

    mesh
}

impl Concrete {
    /// Gets the coordinates of the vertices, after projecting down into 3D.
    fn get_vertex_coords<'a, T: Iterator<Item = &'a Point>>(
        &self,
        vertices: T,
        projection_type: ProjectionType,
    ) -> Vec<[f32; 3]> {
        let dim = self.dim_or();

        // If the polytope is at most 3D, we just embed it into 3D space.
        if projection_type.is_orthogonal() || dim <= 3 {
            vertices
                .map(|point| {
                    let mut iter = point.iter().take(3).map(|&c| c as f32);
                    let x = iter.next().unwrap_or(0.0);
                    let y = iter.next().unwrap_or(0.0);
                    let z = iter.next().unwrap_or(0.0);
                    [x, y, z]
                })
                .collect()
        }
        // Else, we project it down.
        else {
            // Distance from the projection planes.
            let mut direction = Vector::zeros(dim);
            direction[3] = 1.0;

            let (min, max) = self.minmax(&direction).unwrap();
            let dist = (min as f32 - 1.0).abs().max(max as f32 + 1.0).abs();

            vertices
                .map(|point| {
                    let factor: f32 = point.iter().skip(3).map(|&x| x as f32 + dist).product();

                    // We scale the first three coordinates accordingly.
                    let mut iter = point.iter().copied().take(3).map(|c| c as f32 / factor);
                    let x = iter.next().unwrap();
                    let y = iter.next().unwrap();
                    let z = iter.next().unwrap();
                    [x, y, z]
                })
                .collect()
        }
    }

    /// Builds the mesh of a polytope.
    pub fn get_mesh(&self, projection_type: ProjectionType) -> Mesh {
        // If there's no vertices, returns an empty mesh.
        if self.vertex_count() == 0 {
            return empty_mesh();
        }

        // Triangulates the polytope's faces, projects the vertices of both the
        // polytope and the triangulation.
        let triangulation = Triangulation::new(self);
        let vertices = self.get_vertex_coords(
            self.vertices
                .iter()
                .chain(triangulation.extra_vertices.iter()),
            projection_type,
        );

        // Builds the actual mesh.
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 1.0]; vertices.len()]);
        mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals(&vertices));
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.set_indices(Some(Indices::U16(triangulation.triangles)));

        mesh
    }

    /// Builds the wireframe of a polytope.
    pub fn get_wireframe(&self, projection_type: ProjectionType) -> Mesh {
        let vertex_count = self.vertex_count();

        // If there's no vertices, returns an empty mesh.
        if vertex_count == 0 {
            return empty_mesh();
        }

        let edges = self.abs.ranks.get(Rank::new(1));
        let edge_count = self.el_count(Rank::new(1));

        // We add a single vertex so that Miratope doesn't crash.
        let vertices = self.get_vertex_coords(self.vertices.iter(), projection_type);
        let mut indices = Vec::with_capacity(edge_count * 2);

        // Adds the edges to the wireframe.
        if let Some(edges) = edges {
            for edge in edges.iter() {
                debug_assert_eq!(
                    edge.subs.len(),
                    2,
                    "Edge must have exactly 2 elements, found {}.",
                    edge.subs.len()
                );

                indices.push(edge.subs[0] as u16);
                indices.push(edge.subs[1] as u16);
            }
        }

        // Sets the mesh attributes.
        let mut mesh = Mesh::new(PrimitiveTopology::LineList);
        mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals(&vertices));
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0; 2]; vertex_count]);
        mesh.set_indices(Some(Indices::U16(indices)));

        mesh
    }
}
