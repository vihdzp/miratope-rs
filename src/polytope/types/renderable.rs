use bevy::prelude::Mesh;
use bevy::render::{mesh::Indices, pipeline::PrimitiveTopology};

use crate::polytope::{geometry::Point, Concrete, ElementList};

/// A [`Concrete`], together with a triangulation used to render it.
///
/// This struct doesn't actually implement [`Polytope`](crate::Polytope), though
/// it still acts as a type of polytope by virtue of storing one directly.
#[derive(Debug, Clone)]
pub struct Renderable {
    /// The underlying concrete polytope.
    pub concrete: Concrete,

    /// Extra vertices that might be needed for the triangulation.
    extra_vertices: Vec<Point>,

    /// Indices of the vertices that make up the triangles.
    triangles: Vec<[usize; 3]>,
}

impl Renderable {
    /// Generates the triangulation of a `Concrete`.
    pub fn new(concrete: Concrete) -> Self {
        // let vertices = &concrete.vertices;
        let empty_els = ElementList::new();
        let edges = concrete.abs.ranks.get(1).unwrap_or(&empty_els);
        let faces = concrete.abs.ranks.get(2).unwrap_or(&empty_els);

        let extra_vertices = Vec::new();
        let mut triangles = Vec::new();

        for face in faces.iter() {
            let edge_i = *face.subs.first().expect("No indices in face.");
            let vert_i = edges
                .get(edge_i)
                .expect("Index out of bounds: you probably screwed up the polytope's indices.")
                .subs[0];

            for verts in face.subs[1..].iter().map(|&i| {
                let edge = &edges[i];
                assert_eq!(edge.subs.len(), 2, "Edge has more than two subelements.");
                [edge.subs[0], edge.subs[1]]
            }) {
                let [vert_j, vert_k]: [usize; 2] = verts;
                if vert_i != vert_j && vert_i != vert_k {
                    triangles.push([vert_i, vert_j, vert_k]);
                }
            }
        }

        Renderable {
            concrete,
            extra_vertices,
            triangles,
        }
    }

    /// Gets the coordinates of the vertices, after projecting down into 3D.
    fn get_vertex_coords(&self) -> Vec<[f32; 3]> {
        // Enables orthogonal projection.
        const ORTHOGONAL: bool = false;

        let vert_iter = self
            .concrete
            .vertices
            .iter()
            .chain(self.extra_vertices.iter());

        // If the polytope is at most 3D, we just embed it into 3D space.
        if ORTHOGONAL || self.concrete.dim().unwrap_or(0) <= 3 {
            vert_iter
                .map(|point| {
                    let mut iter = point.iter().copied().take(3);
                    let x = iter.next().unwrap_or(0.0);
                    let y = iter.next().unwrap_or(0.0);
                    let z = iter.next().unwrap_or(0.0);
                    [x as f32, y as f32, z as f32]
                })
                .collect()
        }
        // Else, we project it down.
        else {
            // Distance from the projection planes.
            const DIST: f64 = 2.0;

            vert_iter
                .map(|point| {
                    let factor: f64 = point.iter().skip(3).map(|x| x + DIST).product();

                    // We scale the first three coordinates accordingly.
                    let mut iter = point.iter().copied().take(3);
                    let x: f64 = iter.next().unwrap() / factor;
                    let y: f64 = iter.next().unwrap() / factor;
                    let z: f64 = iter.next().unwrap() / factor;
                    [x as f32, y as f32, z as f32]
                })
                .collect()
        }
    }

    /// Generates a mesh from the polytope.
    pub fn get_mesh(&self) -> Mesh {
        let vertices = self.get_vertex_coords();
        let mut indices = Vec::with_capacity(self.triangles.len() * 3);
        for &[i, j, k] in &self.triangles {
            indices.push(i as u16);
            indices.push(j as u16);
            indices.push(k as u16);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.set_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vec![[0.0, 1.0, 0.0]; vertices.len()],
        );
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertices.len()]);
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.set_indices(Some(Indices::U16(indices)));

        mesh
    }

    /// Generates the wireframe for a polytope.
    pub fn get_wireframe(&self) -> Mesh {
        let empty_els = ElementList::new();
        let edges = self.concrete.abs.ranks.get(1).unwrap_or(&empty_els);
        let vertices = self.get_vertex_coords();
        let mut indices = Vec::with_capacity(edges.len() * 2);

        for edge in edges.iter() {
            indices.push(edge.subs[0] as u16);
            indices.push(edge.subs[1] as u16);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::LineList);
        mesh.set_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vec![[0.0, 1.0, 0.0]; vertices.len()],
        );
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertices.len()]);
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.set_indices(Some(Indices::U16(indices)));

        mesh
    }
}
