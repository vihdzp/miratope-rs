use std::mem;

use super::NamedPolytope;
use crate::name::{Con, ConData, Name, NameData, Regular};

use miratope_core::conc::file::off::{OffError, OffResult};
use miratope_core::conc::file::FromFile;
use miratope_core::conc::ConcretePolytope;
use miratope_core::{abs::Abstract, conc::Concrete, geometry::Point, Polytope};

/// A [`Concrete`] polytope bundled together with a [`Name`] of [`Con`] type.
#[derive(Clone)]
pub struct NamedConcrete {
    /// The underlying concrete polytope.
    pub con: Concrete,

    /// The name of the concrete polytope.
    pub name: Name<Con>,
}

impl FromFile for NamedConcrete {
    fn from_off(src: &str) -> OffResult<Self> {
        let con = Concrete::from_off(src)?;

        if let Some(first_line) = src.lines().next() {
            Ok(if let Some(name) = Name::from_src(first_line) {
                Self::new(con, name)
            } else {
                Self::new_generic(con)
            })
        } else {
            Err(OffError::Empty)
        }
    }

    fn from_ggb(file: std::fs::File) -> miratope_core::conc::file::ggb::GgbResult<Self> {
        Ok(Self::new_generic(Concrete::from_ggb(file)?))
    }
}

impl NamedConcrete {
    pub fn new(con: Concrete, name: Name<Con>) -> Self {
        Self { con, name }
    }

    pub fn new_generic(con: Concrete) -> Self {
        let name = Name::generic(con.facet_count(), con.rank());
        Self::new(con, name)
    }

    pub fn set_generic(&mut self) {
        self.name = Name::generic(self.facet_count(), self.rank())
    }
}

impl NamedPolytope<Con> for NamedConcrete {
    fn name(&self) -> &Name<Con> {
        &self.name
    }

    fn name_mut(&mut self) -> &mut Name<Con> {
        &mut self.name
    }
}

impl Polytope for NamedConcrete {
    fn abs(&self) -> &Abstract {
        &self.con.abs
    }

    fn abs_mut(&mut self) -> &mut Abstract {
        &mut self.con.abs
    }

    fn nullitope() -> Self {
        Self::new(Concrete::nullitope(), Name::Nullitope)
    }

    fn point() -> Self {
        Self::new(Concrete::point(), Name::Point)
    }

    fn dyad() -> Self {
        Self::new(Concrete::dyad(), Name::Dyad)
    }

    fn polygon(n: usize) -> Self {
        Self::new(Concrete::polygon(n), Name::polygon(Default::default(), n))
    }

    fn try_dual(&self) -> miratope_core::DualResult<Self> {
        let con = self.con.try_dual()?;
        let name = Name::dual(
            self.name.clone(),
            ConData::new(Point::zeros(con.dim_or())),
            con.facet_count(),
            con.rank(),
        );
        Ok(Self::new(con, name))
    }

    fn try_dual_mut(&mut self) -> miratope_core::DualResult<()> {
        self.con.try_dual_mut()?;
        self.name = Name::dual(
            mem::take(&mut self.name),
            ConData::new(Point::zeros(self.con.dim_or())),
            self.con.facet_count(),
            self.con.rank(),
        );
        Ok(())
    }

    fn comp_append(&mut self, _p: Self) {
        todo!()
    }

    fn element(&self, el: miratope_core::abs::elements::ElementRef) -> Option<Self> {
        Some(Self::new_generic(self.con.element(el)?))
    }

    fn petrial_mut(&mut self) -> bool {
        let res = self.con.petrial_mut();
        if res {
            self.name = Name::petrial(mem::take(&mut self.name));
        }
        res
    }

    fn petrie_polygon_with(&mut self, flag: miratope_core::abs::flag::Flag) -> Option<Self> {
        Some(Self::new_generic(self.con.petrie_polygon_with(flag)?))
    }

    fn omnitruncate(&self) -> Self {
        Self::new_generic(self.con.omnitruncate())
    }

    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::new(
            Concrete::duopyramid(&p.con, &q.con),
            Name::multipyramid(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::new(
            Concrete::duoprism(&p.con, &q.con),
            Name::multiprism(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::new(
            Concrete::duotegum(&p.con, &q.con),
            Name::multitegum(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::new(
            Concrete::duocomb(&p.con, &q.con),
            Name::multicomb(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn ditope(&self) -> Self {
        todo!()
    }

    fn ditope_mut(&mut self) {
        todo!()
    }

    fn hosotope(&self) -> Self {
        todo!()
    }

    fn hosotope_mut(&mut self) {
        todo!()
    }

    fn try_antiprism(&self) -> miratope_core::DualResult<Self> {
        Ok(Self::new(
            self.con().try_antiprism()?,
            Name::antiprism(self.name.clone()),
        ))
    }
}

impl ConcretePolytope for NamedConcrete {
    fn con(&self) -> &Concrete {
        &self.con
    }

    fn con_mut(&mut self) -> &mut Concrete {
        &mut self.con
    }

    fn dyad_with(height: miratope_core::Float) -> Self {
        Self::new(Concrete::dyad_with(height), Name::Dyad)
    }

    fn grunbaum_star_polygon_with_rot(n: usize, d: usize, rot: miratope_core::Float) -> Self {
        Self::new(
            Concrete::grunbaum_star_polygon_with_rot(n, d, rot),
            Name::polygon(
                ConData::new(Regular::Yes {
                    center: Point::zeros(2),
                }),
                n,
            ),
        )
    }

    fn try_dual_mut_with(
        &mut self,
        sphere: &miratope_core::geometry::Hypersphere,
    ) -> miratope_core::DualResult<()> {
        let res = self.con.try_dual_mut_with(sphere);
        if res.is_ok() {
            self.name = Name::dual(
                mem::take(&mut self.name),
                ConData::new(sphere.center.clone()),
                self.facet_count(),
                self.rank(),
            );
        }
        res
    }

    fn pyramid_with(&self, apex: Point) -> Self {
        Self::new(
            self.con().pyramid_with(apex),
            Name::pyramid(self.name.clone()),
        )
    }

    fn prism_with(&self, height: miratope_core::Float) -> Self {
        Self::new(
            self.con().prism_with(height),
            Name::prism(self.name.clone()),
        )
    }

    fn tegum_with(&self, apex1: Point, apex2: Point) -> Self {
        Self::new(
            self.con().tegum_with(apex1, apex2),
            Name::tegum(self.name.clone()),
        )
    }

    fn antiprism_with_vertices<T: Iterator<Item = Point>, U: Iterator<Item = Point>>(
        &self,
        vertices: T,
        dual_vertices: U,
    ) -> Self {
        Self::new(
            self.con().antiprism_with_vertices(vertices, dual_vertices),
            Name::antiprism(self.name.clone()),
        )
    }

    fn duopyramid_with(
        p: &Self,
        q: &Self,
        p_offset: &Point,
        q_offset: &Point,
        height: miratope_core::Float,
    ) -> Self {
        Self::new(
            Concrete::duopyramid_with(p.con(), q.con(), p_offset, q_offset, height),
            Name::multipyramid(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duotegum_with(p: &Self, q: &Self, p_offset: &Point, q_offset: &Point) -> Self {
        Self::new(
            Concrete::duotegum_with(p.con(), q.con(), p_offset, q_offset),
            Name::multitegum(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn flatten(&mut self) {
        self.con_mut().flatten();
        self.set_generic();
    }

    fn flatten_into(&mut self, subspace: &miratope_core::geometry::Subspace) {
        self.con_mut().flatten_into(subspace);
        self.set_generic();
    }

    fn cross_section(&self, slice: &miratope_core::geometry::Hyperplane) -> Self {
        Self::new_generic(self.con().cross_section(slice))
    }
}
