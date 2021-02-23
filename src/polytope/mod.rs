use bevy::prelude::Mesh;
use bevy::render::mesh::Indices;
use bevy::render::pipeline::PrimitiveTopology;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;

use petgraph::{graph::Graph, prelude::NodeIndex, Undirected};

pub mod convex;
pub mod off;
pub mod shapes;

pub type Element = Vec<usize>;
pub type ElementList = Vec<Element>;
pub type Point = nalgebra::DVector<f64>;
pub type Matrix = nalgebra::DMatrix<f64>;
pub type Hyperplane = Vec<Point>;

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

pub fn project(p: &Point, h: Hyperplane) -> Point {
    const EPS: f64 = 1e-9;

    let mut h = h.iter();
    let o = h.next().unwrap();
    let mut basis: Vec<Point> = Vec::new();

    for q in h {
        let mut q = q - o;

        for b in &basis {
            q -= b * (q.dot(&b)) / b.norm_squared();
        }

        if q.norm() > EPS {
            basis.push(q);
        }
    }

    let mut p = p - o;

    for b in &basis {
        p -= b * (p.dot(&b)) / b.norm_squared();
    }

    p
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
    pub vertices: Vec<Point>,
    pub elements: Vec<ElementList>,

    triangles: Vec<[usize; 3]>,
}

impl Polytope {
    pub fn new(vertices: Vec<Point>, elements: Vec<ElementList>) -> Self {
        let triangles = if elements.len() >= 2 {
            Self::triangulate(&elements[0], &elements[1])
        } else {
            vec![]
        };

        Polytope {
            vertices,
            elements,
            triangles,
        }
    }

    /// Builds a polytope and auto-generates its connected components.
    pub fn new_wo_comps(vertices: Vec<Point>, elements: Vec<ElementList>) -> Self {
        let rank = elements.len() + 1;
        let num_ridges = elements[rank - 3].len();
        let facets = &elements[rank - 2];
        let num_facets = facets.len();

        // g is the incidence graph of ridges and facets.
        // The ith ridge is stored at position i.
        // The ith facet is stored at position num_ridges + i.
        let mut graph: Graph<(), (), Undirected> = Graph::new_undirected();
        for _ in 0..(num_ridges + num_facets) {
            graph.add_node(());
        }

        for (i, f) in facets.iter().enumerate() {
            for r in f.iter() {
                graph.add_edge(NodeIndex::new(*r), NodeIndex::new(num_ridges + i), ());
            }
        }

        // Converts the connected components of our facet + ridge graph
        // into just the lists of facets in each component.
        let g_comps = petgraph::algo::kosaraju_scc(&graph);
        let mut comps = Vec::with_capacity(g_comps.len());

        for g_comp in g_comps.iter() {
            let mut comp = Vec::new();

            for idx in g_comp.iter() {
                let idx: usize = idx.index();

                if idx < num_ridges {
                    comp.push(idx);
                }
            }

            comps.push(comp);
        }

        let mut elements = elements.clone();
        elements.push(comps);

        Polytope::new(vertices, elements)
    }

    fn triangulate(edges: &Vec<Element>, faces: &Vec<Element>) -> Vec<[usize; 3]> {
        let mut triangles = Vec::new();

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

    fn el_counts(&self) -> Vec<usize> {
        let mut counts = Vec::with_capacity(self.elements.len() + 1);
        counts.push(self.vertices.len());

        for e in self.elements.iter() {
            counts.push(e.len());
        }

        counts
    }

    fn scale(&mut self, k: f64) {
        for v in &mut self.vertices {
            *v *= k;
        }
    }

    fn get_vertex_coords(&self) -> Vec<[f32; 3]> {
        self.vertices
            .iter()
            .map(|point| {
                let mut iter = point.iter().copied().take(3);
                let x = iter.next().unwrap_or(0.0);
                let y = iter.next().unwrap_or(0.0);
                let z = iter.next().unwrap_or(0.0);
                [x as f32, y as f32, z as f32]
            })
            .collect()
    }

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

    pub fn get_wireframe(&self) -> Mesh {
        let edges = &self.elements[0];
        let vertices: Vec<_> = self.get_vertex_coords();
        let mut indices = Vec::with_capacity(edges.len() * 2);
        for edge in edges {
            indices.push(edge[0] as u16);
            indices.push(edge[1] as u16);
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

impl From<PolytopeSerde> for Polytope {
    fn from(ps: PolytopeSerde) -> Self {
        Polytope::new(ps.vertices, ps.elements)
    }
}
