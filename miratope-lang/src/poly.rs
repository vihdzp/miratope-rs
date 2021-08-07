//! Contains the named abstract and concrete polytope types.

use std::mem;

use miratope_core::{
    abs::{flag::Flag, Abstract, Ranked},
    conc::{
        file::{
            off::{OffError, OffResult},
            FromFile,
        },
        Concrete, ConcretePolytope,
    },
    geometry::Point,
    Polytope,
};

use crate::name::{Abs, Con, ConData, Name, NameData, NameType};

#[derive(Clone, Debug)]
pub struct Named<T: NameType> {
    pub poly: T::Polytope,
    pub name: Name<T>,
}

impl<T: NameType> Named<T> {
    fn new(poly: T::Polytope, name: Name<T>) -> Self {
        Self { poly, name }
    }

    fn new_generic(poly: T::Polytope) -> Self {
        let name = Name::generic(poly.facet_count(), poly.rank());
        Self::new(poly, name)
    }

    fn set_generic(&mut self) {
        self.name = Name::generic(self.poly.facet_count(), self.poly.rank())
    }
}

pub type NamedAbstract = Named<Abs>;
pub type NamedConcrete = Named<Con>;

impl<T: NameType> Polytope for Named<T> {
    type DualError = <T::Polytope as Polytope>::DualError;

    fn abs(&self) -> &Abstract {
        self.poly.abs()
    }

    fn abs_mut(&mut self) -> &mut Abstract {
        self.poly.abs_mut()
    }

    fn into_abs(self) -> Abstract {
        self.poly.into_abs()
    }

    fn nullitope() -> Self {
        Self::new(T::Polytope::nullitope(), Name::Nullitope)
    }

    fn point() -> Self {
        Self::new(T::Polytope::point(), Name::Point)
    }

    fn dyad() -> Self {
        Self::new(T::Polytope::dyad(), Name::Dyad)
    }

    fn polygon(n: usize) -> Self {
        Self::new(
            T::Polytope::polygon(n),
            Name::polygon(T::DataRegular::new_lazy(|| Point::zeros(2).into()), n),
        )
    }

    fn try_dual(&self) -> Result<Self, Self::DualError> {
        let poly = self.poly.try_dual()?;
        let rank = poly.rank();

        let name = self.name.clone().dual(
            T::DataPoint::new_lazy(|| Point::zeros(rank - 1)),
            poly.facet_count(),
            rank,
        );

        Ok(Self::new(poly, name))
    }

    fn try_dual_mut(&mut self) -> Result<(), Self::DualError> {
        self.poly.try_dual_mut()?;
        let rank = self.poly.rank();

        self.name = mem::take(&mut self.name).dual(
            T::DataPoint::new_lazy(|| Point::zeros(rank - 1)),
            self.poly.facet_count(),
            rank,
        );

        Ok(())
    }

    fn comp_append(&mut self, _p: Self) {
        // Compound names are TBA
        todo!()
    }

    fn element(&self, rank: usize, idx: usize) -> Option<Self> {
        Some(Self::new_generic(self.poly.element(rank, idx)?))
    }

    fn petrial_mut(&mut self) -> bool {
        let res = self.poly.petrial_mut();
        if res {
            self.name = Name::petrial(mem::take(&mut self.name));
        }
        res
    }

    fn petrie_polygon_with(&mut self, flag: Flag) -> Option<Self> {
        Some(Self::new_generic(self.poly.petrie_polygon_with(flag)?))
    }

    fn omnitruncate(&self) -> Self {
        Self::new_generic(self.poly.omnitruncate())
    }

    fn prism(&self) -> Self {
        Self::new(self.poly.prism(), self.name.clone().prism())
    }

    fn tegum(&self) -> Self {
        Self::new(self.poly.tegum(), self.name.clone().tegum())
    }

    fn pyramid(&self) -> Self {
        Self::new(self.poly.pyramid(), self.name.clone().pyramid())
    }

    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duopyramid(&p.poly, &q.poly),
            Name::multipyramid(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duoprism(&p.poly, &q.poly),
            Name::multiprism(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duotegum(&p.poly, &q.poly),
            Name::multitegum(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duocomb(&p.poly, &q.poly),
            Name::multicomb(vec![p.name.clone(), q.name.clone()]),
        )
    }

    // TODO: manually implement multiproducts.

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

    fn try_antiprism(&self) -> Result<Self, Self::DualError> {
        Ok(Self::new(
            self.poly.try_antiprism()?,
            Name::antiprism(self.name.clone()),
        ))
    }

    fn simplex(rank: usize) -> Self {
        Self::new(
            T::Polytope::simplex(rank),
            Name::simplex(Default::default(), rank),
        )
    }

    fn hypercube(rank: usize) -> Self {
        Self::new(
            T::Polytope::hypercube(rank),
            Name::hyperblock(Default::default(), rank),
        )
    }

    fn orthoplex(rank: usize) -> Self {
        Self::new(
            T::Polytope::orthoplex(rank),
            Name::orthoplex(Default::default(), rank),
        )
    }
}

impl ConcretePolytope for NamedConcrete {
    fn con(&self) -> &Concrete {
        &self.poly
    }

    fn con_mut(&mut self) -> &mut Concrete {
        &mut self.poly
    }

    fn dyad_with(height: miratope_core::Float) -> Self {
        Self::new(Concrete::dyad_with(height), Name::Dyad)
    }

    fn grunbaum_star_polygon_with_rot(n: usize, d: usize, rot: miratope_core::Float) -> Self {
        Self::new(
            Concrete::grunbaum_star_polygon_with_rot(n, d, rot),
            Name::polygon(ConData::new(Point::zeros(2).into()), n),
        )
    }

    fn try_dual_mut_with(
        &mut self,
        sphere: &miratope_core::geometry::Hypersphere,
    ) -> Result<(), Self::DualError> {
        let res = self.poly.try_dual_mut_with(sphere);
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
