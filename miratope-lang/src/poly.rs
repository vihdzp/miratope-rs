//! Contains the named abstract and concrete polytope types.

use std::{
    array,
    ops::{Index, IndexMut},
};

use miratope_core::{
    abs::{flag::Flag, Abstract, Element, ElementList, Ranked},
    conc::{Concrete, ConcretePolytope},
    file::{
        off::{OffParseError, OffParseResult},
        FromFile,
    },
    geometry::Point,
    Float, Polytope,
};
use serde::de::DeserializeOwned;

use crate::name::{Abs, Con, ConData, Name, NameData, NameType};

/// Represents a named polytope, either [`Abstract`] or [`Concrete`].
#[derive(Clone, Debug)]
pub struct Named<T: NameType> {
    /// The inner polytope.
    pub poly: T::Polytope,

    /// The stored name.
    pub name: Name<T>,
}

impl<T: NameType> Named<T> {
    /// Initializes a new named polytope.
    fn new(poly: T::Polytope, name: Name<T>) -> Self {
        Self { poly, name }
    }

    /// Initializes a new named polytope with a generic name.
    fn new_generic(polytope: T::Polytope) -> Self {
        let name = Name::generic(polytope.facet_count(), polytope.rank());
        Self::new(polytope, name)
    }

    /// Sets the polytope's name to the corresponding generic name.
    fn set_generic(&mut self) {
        self.name = Name::generic(self.poly.facet_count(), self.poly.rank())
    }
}

impl<T: NameType> Index<usize> for Named<T> {
    type Output = ElementList;

    fn index(&self, index: usize) -> &Self::Output {
        &self.poly[index]
    }
}

impl<T: NameType> IndexMut<usize> for Named<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.poly[index]
    }
}

impl<T: NameType> Index<(usize, usize)> for Named<T> {
    type Output = Element;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.poly[index]
    }
}

impl<T: NameType> IndexMut<(usize, usize)> for Named<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.poly[index]
    }
}

/// An [`Abstract`] polytope with a [`Name`].
pub type NamedAbstract = Named<Abs>;

/// A [`Concrete`] polytope with a [`Name`].
pub type NamedConcrete<T> = Named<Con<T>>;

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
        let facet_count = self.poly.facet_count();

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
        self.poly.comp_append(p.poly);
        println!("Compound names are TBA!")
    }

    fn element(&self, rank: usize, idx: usize) -> Option<Self> {
        Some(Self::new_generic(self.poly.element(rank, idx)?))
    }

    fn petrial_mut(&mut self) -> bool {
        let res = self.poly.petrial_mut();
        if res {
            self.name.into_mut(Name::petrial);
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

    fn prism_mut(&mut self) {
        self.poly.prism_mut();
        self.name.into_mut(Name::prism);
    }

    fn tegum(&self) -> Self {
        Self::new(self.poly.tegum(), self.name.clone().tegum())
    }

    fn tegum_mut(&mut self) {
        self.poly.tegum_mut();
        self.name.into_mut(Name::tegum);
    }

    fn pyramid(&self) -> Self {
        Self::new(self.poly.pyramid(), self.name.clone().pyramid())
    }

    fn pyramid_mut(&mut self) {
        self.poly.pyramid_mut();
        self.name.into_mut(Name::pyramid);
    }

    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duopyramid(&p.poly, &q.poly),
            Name::multipyramid(array::IntoIter::new([p.name.clone(), q.name.clone()])),
        )
    }

    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duoprism(&p.poly, &q.poly),
            Name::multiprism(array::IntoIter::new([p.name.clone(), q.name.clone()])),
        )
    }

    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duotegum(&p.poly, &q.poly),
            Name::multitegum(array::IntoIter::new([p.name.clone(), q.name.clone()])),
        )
    }

    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::new(
            T::Polytope::duocomb(&p.poly, &q.poly),
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

impl<T: Float + DeserializeOwned> ConcretePolytope<T> for NamedConcrete<T> {
    fn con(&self) -> &Concrete<T> {
        &self.poly
    }

    fn con_mut(&mut self) -> &mut Concrete<T> {
        &mut self.poly
    }

    fn dyad_with(height: T) -> Self {
        Self::new(Concrete::dyad_with(height), Name::Dyad)
    }

    fn grunbaum_star_polygon_with_rot(n: usize, d: usize, rot: T) -> Self {
        Self::new(
            Concrete::grunbaum_star_polygon_with_rot(n, d, rot),
            Name::polygon(ConData::new_lazy(|| Point::zeros(2).into()), n),
        )
    }

    fn try_dual_mut_with(
        &mut self,
        sphere: &miratope_core::geometry::Hypersphere<T>,
    ) -> Result<(), Self::DualError> {
        let res = self.poly.try_dual_mut_with(sphere);
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

    fn pyramid_with(&self, apex: Point<T>) -> Self {
        Self::new(
            self.con().pyramid_with(apex),
            Name::pyramid(self.name.clone()),
        )
    }

    fn prism_with(&self, height: T) -> Self {
        Self::new(
            self.con().prism_with(height),
            Name::prism(self.name.clone()),
        )
    }

    fn tegum_with(&self, apex1: Point<T>, apex2: Point<T>) -> Self {
        Self::new(
            self.con().tegum_with(apex1, apex2),
            Name::tegum(self.name.clone()),
        )
    }

    fn antiprism_with_vertices<I: Iterator<Item = Point<T>>, J: Iterator<Item = Point<T>>>(
        &self,
        vertices: I,
        dual_vertices: J,
    ) -> Self {
        Self::new(
            self.con().antiprism_with_vertices(vertices, dual_vertices),
            Name::antiprism(self.name.clone()),
        )
    }

    fn duopyramid_with(
        p: &Self,
        q: &Self,
        p_offset: &Point<T>,
        q_offset: &Point<T>,
        height: T,
    ) -> Self {
        Self::new(
            Concrete::duopyramid_with(p.con(), q.con(), p_offset, q_offset, height),
            Name::multipyramid(array::IntoIter::new([p.name.clone(), q.name.clone()])),
        )
    }

    fn duotegum_with(p: &Self, q: &Self, p_offset: &Point<T>, q_offset: &Point<T>) -> Self {
        Self::new(
            Concrete::duotegum_with(p.con(), q.con(), p_offset, q_offset),
            Name::multitegum(array::IntoIter::new([p.name.clone(), q.name.clone()])),
        )
    }

    fn flatten(&mut self) {
        self.con_mut().flatten();
        self.set_generic();
    }

    fn flatten_into(&mut self, subspace: &miratope_core::geometry::Subspace<T>) {
        self.con_mut().flatten_into(subspace);
        self.set_generic();
    }

    fn cross_section(&self, slice: &miratope_core::geometry::Hyperplane<T>) -> Self {
        Self::new_generic(self.con().cross_section(slice))
    }

    fn truncate(&self, a: Vec<usize>, b: Vec<T>) -> Self {
        Self::new_generic(self.con().truncate(a,b).clone())
    }
}

impl<T: Float + DeserializeOwned> FromFile for NamedConcrete<T> {
    fn from_off(src: &str) -> OffParseResult<Self> {
        let con = Concrete::from_off(src)?;

        if let Some(first_line) = src.lines().next() {
            Ok(if let Some(name) = Name::from_src(first_line) {
                Self::new(con, name)
            } else {
                Self::new_generic(con)
            })
        } else {
            Err(OffParseError::Empty)
        }
    }

    fn from_ggb(file: std::fs::File) -> miratope_core::file::ggb::GgbResult<Self> {
        Concrete::from_ggb(file).map(Self::new_generic)
    }
}
