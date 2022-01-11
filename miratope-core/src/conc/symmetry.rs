//! The code used to get the symmetry a polytope and do operations based on that.

use std::{collections::HashMap, vec};

use crate::{
    abs::{ElementMap, Ranked, flag::{FlagIter, Flag}},
    conc::Concrete,
    float::Float,
    group::Group, geometry::Matrix, Polytope,
};

use vec_like::*;

impl Flag {
    fn vertex_sequence (&mut self, p: &Concrete) -> Matrix<f64> {
        let rank = p.rank();
        let mut basis = Matrix::<f64>::zeros(rank-1,rank-1);
        let mut columns = basis.column_iter_mut();
        let vertex = &p.vertices[self[1]];

        columns.next().unwrap().copy_from(&vertex);
        self.change_mut(&p.abs, 1);
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
    ///
    pub fn get_symmetry_group(&mut self) -> Group<vec::IntoIter<Matrix<f64>>> {
        self.element_sort();
        let mut flag_iter = FlagIter::new(&self.abs);
        let types = &self.types_of_elements();
        // dbg!(types);
        // let affine_hulls = &self.element_map_affine_hulls();

        let mut base_flag = flag_iter.next().unwrap();
        let base_basis = base_flag.vertex_sequence(&self);

        let mut group = Vec::<Matrix<f64>>::new();
        group.push(Matrix::identity(&self.rank()-1,&self.rank()-1));
        for mut flag in flag_iter {
            if flag.iter().enumerate().map(|(r, x)| types[r][*x]).sum::<usize>() == 0 {
                // calculate isometry
                let basis = flag.vertex_sequence(&self);
                let isometry = basis*base_basis.clone().pseudo_inverse(f64::EPS).unwrap();

                // check if elements match up
                // TBD

                // add to group if so
                dbg!(&flag);
                group.push(isometry);
            }
        }
        unsafe {
            Group::new(&self.rank()-1, group.into_iter())
        }
    }
}