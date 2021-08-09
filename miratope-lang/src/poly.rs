//! Contains the named abstract and concrete polytope types.

use std::array;

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
    Float, Polytope,
};

use crate::name::{Abs, Con, ConData, Name, NameData, NameType};

/// Represents a named polytope, either [`Abstract`] or [`Concrete`].
#[derive(Clone, Debug)]
pub struct Named<T: NameType> {
    /// The inner polytope.
    pub polytope: T::Polytope,

    /// The stored name.
    pub name: Name<T>,
}

impl<T: NameType> Named<T> {
    /// Initializes a new named polytope.
    fn new(polytope: T::Polytope, name: Name<T>) -> Self {
        Self { polytope, name }
    }

    /// Initializes a new named polytope with a generic name.
    fn new_generic(polytope: T::Polytope) -> Self {
        let name = Name::generic(polytope.facet_count(), polytope.rank());
        Self::new(polytope, name)
    }

    /// Sets the polytope's name to the corresponding generic name.
    fn set_generic(&mut self) {
        self.name = Name::generic(self.polytope.facet_count(), self.polytope.rank())
    }
}

/// An [`Abstract`] polytope with a [`Name`].
pub type NamedAbstract = Named<Abs>;

/// A [`Concrete`] polytope with a [`Name`].
pub type NamedConcrete = Named<Con>;

impl<T: NameType> Polytope for Named<T> {
    type DualError = <T::Polytope as Polytope>::DualError;

    fn abs(&self) -> &Abstract {
        self.polytope.abs()
    }

    fn abs_mut(&mut self) -> &mut Abstract {
        self.polytope.abs_mut()
    }

    fn into_abs(self) -> Abstract {
        self.polytope.into_abs()
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
        let poly = self.polytope.try_dual()?;
        let rank = poly.rank();

        let name = self.name.clone().dual(
            T::DataPoint::new_lazy(|| Point::zeros(rank - 1)),
            poly.facet_count(),
            rank,
        );

        Ok(Self::new(poly, name))
    }

    fn try_dual_mut(&mut self) -> Result<(), Self::DualError> {
        self.polytope.try_dual_mut()?;
        let rank = self.polytope.rank();
        let facet_count = self.polytope.facet_count();

        self.name.into_mut(|name| {
            name.dual(
                T::DataPoint::new_lazy(|| Point::zeros(rank - 1)),
                facet_count,
                rank,
            )
        });

        Ok(())
    }

    fn comp_append(&mut self, p: Self) {
        self.polytope.comp_append(p.polytope);
        println!("Compound names are TBA!")
    }

    fn element(&self, rank: usize, idx: usize) -> Option<Self> {
        Some(Self::new_generic(self.polytope.element(rank, idx)?))
    }

    fn petrial_mut(&mut self) -> bool {
        let res = self.polytope.petrial_mut();
        if res {
            self.name.into_mut(Name::petrial);
        }
        res
    }

    fn petrie_polygon_with(&mut self, flag: Flag) -> Option<Self> {
        Some(Self::new_generic(self.polytope.petrie_polygon_with(flag)?))
    }

    fn omnitruncate(&self) -> Self {
        Self::new_generic(self.polytope.omnitruncate())
    }

    fn prism(&self) -> Self {
        Self::new(self.polytope.prism(), self.name.clone().prism())
    }

    fn prism_mut(&mut self) {
        self.polytope.prism_mut();
        self.name.into_mut(Name::prism);
    }

    fn tegum(&self) -> Self {
        Self::new(self.polytope.tegum(), self.name.clone().tegum())
    }

    fn tegum_mut(&mut self) {
        self.polytope.tegum_mut();
        self.name.into_mut(Name::tegum);
    }

    fn pyramid(&self) -> Self {
        Self::new(self.polytope.pyramid(), self.name.clone().pyramid())
    }

    fn pyramid_mut(&mut self) {
        self.polytope.pyramid_mut();
        self.name.into_mut(Name::pyramid);
    }

    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duopyramid(&p.polytope, &q.polytope),
            Name::multipyramid(array::IntoIter::new([p.name.clone(), q.name.clone()])),
        )
    }

    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duoprism(&p.polytope, &q.polytope),
            Name::multiprism(array::IntoIter::new([p.name.clone(), q.name.clone()])),
        )
    }

    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duotegum(&p.polytope, &q.polytope),
            Name::multitegum(array::IntoIter::new([p.name.clone(), q.name.clone()])),
        )
    }

    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duocomb(&p.polytope, &q.polytope),
            Name::multicomb(array::IntoIter::new([p.name.clone(), q.name.clone()])),
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
            self.polytope.try_antiprism()?,
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
        &self.polytope
    }

    fn con_mut(&mut self) -> &mut Concrete {
        &mut self.polytope
    }

    fn dyad_with(height: Float) -> Self {
        Self::new(Concrete::dyad_with(height), Name::Dyad)
    }

    fn grunbaum_star_polygon_with_rot(n: usize, d: usize, rot: Float) -> Self {
        Self::new(
            Concrete::grunbaum_star_polygon_with_rot(n, d, rot),
            Name::polygon(ConData::new_lazy(|| Point::zeros(2).into()), n),
        )
    }

    fn try_dual_mut_with(
        &mut self,
        sphere: &miratope_core::geometry::Hypersphere,
    ) -> Result<(), Self::DualError> {
        let res = self.polytope.try_dual_mut_with(sphere);
        let facet_count = self.facet_count();
        let rank = self.rank();

        if res.is_ok() {
            self.name.into_mut(|name| {
                name.dual(
                    ConData::new_lazy(|| sphere.center.clone()),
                    facet_count,
                    rank,
                )
            });
        }
        res
    }

    fn pyramid_with(&self, apex: Point) -> Self {
        Self::new(
            self.con().pyramid_with(apex),
            Name::pyramid(self.name.clone()),
        )
    }

    fn prism_with(&self, height: Float) -> Self {
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
        height: Float,
    ) -> Self {
        Self::new(
            Concrete::duopyramid_with(p.con(), q.con(), p_offset, q_offset, height),
            Name::multipyramid(array::IntoIter::new([p.name.clone(), q.name.clone()])),
        )
    }

    fn duotegum_with(p: &Self, q: &Self, p_offset: &Point, q_offset: &Point) -> Self {
        Self::new(
            Concrete::duotegum_with(p.con(), q.con(), p_offset, q_offset),
            Name::multitegum(array::IntoIter::new([p.name.clone(), q.name.clone()])),
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
