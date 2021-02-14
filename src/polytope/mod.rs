use bevy::prelude::Mesh;
use bevy::render::mesh::Indices;
use bevy::render::pipeline::PrimitiveTopology;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;

pub mod convex;
pub mod off;
pub mod shapes;

pub type Element = Vec<usize>;
pub type ElementList = Vec<Element>;
pub type Point = nalgebra::DVector<f64>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolytopeSerde {
    pub vertices: Vec<Point>,
    pub elements: Vec<ElementList>,
}

impl From<Polytope> for PolytopeSerde {
    fn from(p: Polytope) -> Self {
        PolytopeSerde {
            vertices: p.vertices,
            elements: p.elements,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polytope {
    /// The concrete vertices of the polytope.
    ///
    /// Here, "dimension" is the number being encoded by the [`Dim`][nalgebra::Dim]
    /// used in [`Point`]s. This is not to be confused by the rank, which is defined
    /// in terms of the length of the element list.
    ///
    /// # Assumptions
    ///
    /// * All points have the same dimension.
    ///
    ///
    pub vertices: Vec<Point>,
    pub elements: Vec<ElementList>,

    triangles: Vec<[usize; 3]>,
}

impl Polytope {
    pub fn new(vertices: Vec<Point>, elements: Vec<ElementList>) -> Self {
        let triangles;

        if elements.len() >= 2 {
            triangles = Self::triangulate(&elements[0], &elements[1]);
        } else {
            triangles = vec![];
        }

        Polytope {
            vertices,
            elements,
            triangles,
        }
    }

    fn triangulate(edges: &Vec<Element>, faces: &Vec<Element>) -> Vec<[usize; 3]> {
        let mut triangles = Vec::with_capacity(4 * (edges.len() - faces.len()));

        for face in faces {
            let edge_i = face.first().expect("no indices in face").clone();
            let vert_i = edges[edge_i][0];

            for verts in face[1..].iter().map(|&i| {
                edges[i]
                    .clone()
                    .try_into()
                    .expect("edges has more than two elements")
            }) {
                let [vert_j, vert_k]: [usize; 2] = verts;
                if vert_i != vert_j && vert_i != vert_k {
                    triangles.push([vert_i, vert_j, vert_k]);
                }
            }
        }

        triangles
    }

    fn rank(&self) -> usize {
        self.elements.len()
    }

    fn dimension(&self) -> usize {
        if self.vertices.len() == 0 {
            0
        } else {
            self.vertices[0].len()
        }
    }

    fn scale(&mut self, k: f64) {
        for v in &mut self.vertices {
            *v *= k;
        }
    }
}

impl From<PolytopeSerde> for Polytope {
    fn from(ps: PolytopeSerde) -> Self {
        Polytope::new(ps.vertices, ps.elements)
    }
}

impl From<Polytope> for Mesh {
    fn from(p: Polytope) -> Self {
        let vertices: Vec<_> = p
            .vertices
            .into_iter()
            .map(|point| {
                let mut iter = point.iter().copied().take(3);
                let x = iter.next().unwrap_or(0.0);
                let y = iter.next().unwrap_or(0.0);
                let z = iter.next().unwrap_or(0.0);
                [x as f32, y as f32, z as f32]
            })
            .collect();
        let mut indices = Vec::new();
        for [i, j, k] in p.triangles {
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
}
