use std::collections::BTreeSet;
use bevy::prelude::Mesh;
use bevy::render::mesh::Indices;
use bevy::render::pipeline::PrimitiveTopology;
use ultraviolet::DVec3;

pub struct PolytopeC {
    pub vertices: Vec<DVec3>,
    pub edges: Vec<(usize, usize)>,
    pub faces: Vec<Vec<usize>>,
    triangles: Vec<[usize; 3]>,
}

impl PolytopeC {
    fn triangulate(
        edges: &Vec<(usize, usize)>,
        faces: &Vec<Vec<usize>>,
    ) -> Vec<[usize; 3]> {
        let mut triangles = Vec::with_capacity(4 * (edges.len() - faces.len()));

        for face in faces {
            let edge_i = face.first().expect("no indices in face").clone();
            let vert_i = edges[edge_i].0;

            for &(vert_j, vert_k) in face[1..].iter().map(|&i| &edges[i]) {
                if vert_i != vert_j && vert_i != vert_k {
                    triangles.push([vert_i, vert_j, vert_k]);
                    triangles.push([vert_i, vert_k, vert_j]);
                }
            }
        }

        triangles
    }

    pub fn new(vertices: Vec<DVec3>, edges: Vec<(usize, usize)>, faces: Vec<Vec<usize>>) -> Self {
        let triangles = Self::triangulate(&edges, &faces);

        PolytopeC {
            vertices,
            edges,
            faces,
            triangles,
        }
    }
}

impl From<PolytopeC> for Mesh {
    fn from(p: PolytopeC) -> Self {
        let vertices: Vec<_> = p
            .vertices
            .into_iter()
            .map(|DVec3 { x, y, z }| [x as f32, y as f32, z as f32])
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
        mesh.set_indices(Some(Indices::U16(dbg!(indices))));

        mesh
    }
}
