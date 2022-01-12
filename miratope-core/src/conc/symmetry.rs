//! The code used to get the symmetry a polytope and do operations based on that.

use std::{collections::BTreeMap, vec, iter::FromIterator};

use crate::{
    abs::{Ranked, flag::{FlagIter, Flag}},
    conc::Concrete,
    group::Group, geometry::{Matrix, PointOrd}, Polytope,
};

use vec_like::*;

impl Flag {
    /// Outputs a sequence of vertices obtained from applying a fixed sequence of flag changes to a flag.
    /// Used for computing the elements of a symmetry group. 
    fn vertex_sequence(&mut self, p: &Concrete) -> Matrix<f64> {
        let rank = p.rank();
        let mut basis = Matrix::<f64>::zeros(rank-1,rank-1);
        let mut columns = basis.column_iter_mut();
        let vertex = &p.vertices[self[1]];

        columns.next().unwrap().copy_from(&vertex);
        for mut col in columns {
            for r in 1..rank {
                self.change_mut(&p.abs, r);
            }
            let vertex = &p.vertices[self[1]];
            col.copy_from(&vertex);
        }

        basis
    }
}

impl Concrete {
    /// Computes the symmetry group of a polytope, along with a list of vertex mappings.
    pub fn get_symmetry_group(&mut self) -> (Group<vec::IntoIter<Matrix<f64>>>, Vec<Vec<usize>>) {
        self.element_sort();
        let flag_iter = FlagIter::new(&self.abs);
        let types = &self.types_of_elements();

        let mut vertices_pointord = Vec::<PointOrd<f64>>::new();
        for v in &self.vertices {
            vertices_pointord.push(PointOrd::new(v.clone()));
        }
        let vertices = BTreeMap::from_iter((vertices_pointord).into_iter().zip(0..));
        let mut vertex_map: Vec<Vec<usize>> = Vec::new();

        let base_flag = self.first_flag();
        let base_basis = base_flag.clone().vertex_sequence(&self);
        let base_basis_inverse = base_basis.clone().try_inverse().unwrap();

        let mut group = Vec::<Matrix<f64>>::new();

        'a: for flag in flag_iter {
            if flag
                .iter()
                .enumerate()
                .map(|(r, x)| (types[r][*x] != types[r][base_flag[r]]) as usize)
                .sum::<usize>() == 0 // this checks if all the elements in the flag have the same types as the ones in the base flag, else it skips it
                {
                // calculate isometry
                let basis = flag.clone().vertex_sequence(&self);
                let isometry = basis * &base_basis_inverse;

                // check if vertices match up
                // Really, you should check if all the elements match up, but this should be enough in most cases.
                let mut vertex_map_row = Vec::<usize>::new();
                for vertex in &vertices {
                    let new_vertex = PointOrd::new(isometry.clone() * vertex.0.matrix());
                    match vertices.get(&new_vertex) {
                        Some(idx) => {
                            vertex_map_row.push(*idx);
                        }
                        None => {
                            continue 'a;
                        }
                    }
                }

                // add to group if so
                group.push(isometry);
                vertex_map.push(vertex_map_row);
            }
        }
        unsafe {
            (Group::new(&self.rank()-1, group.into_iter()), vertex_map)
        }
    }
}