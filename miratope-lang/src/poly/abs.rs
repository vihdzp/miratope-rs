use std::borrow::{Borrow, BorrowMut};
use std::convert::Infallible;
use std::mem;

use super::NamedPolytope;
use crate::name::{Abs, Name};

use miratope_core::{abs::Abstract, Polytope};

/// An [`Abstract`] polytope bundled together with a [`Name`] of [`Abs`] type.
#[derive(Clone)]
pub struct NamedAbstract {
    /// The underlying abstract polytope.
    pub abs: Abstract,

    /// The name of the abstract polytope.
    pub name: Name<Abs>,
}

impl AsRef<Abstract> for NamedAbstract {
    fn as_ref(&self) -> &Abstract {
        &self.abs
    }
}

impl AsMut<Abstract> for NamedAbstract {
    fn as_mut(&mut self) -> &mut Abstract {
        &mut self.abs
    }
}

impl AsRef<Name<Abs>> for NamedAbstract {
    fn as_ref(&self) -> &Name<Abs> {
        &self.name
    }
}

impl AsMut<Name<Abs>> for NamedAbstract {
    fn as_mut(&mut self) -> &mut Name<Abs> {
        &mut self.name
    }
}

impl NamedAbstract {
    pub fn new(abs: Abstract, name: Name<Abs>) -> Self {
        Self { abs, name }
    }

    pub fn new_generic(abs: Abstract) -> Self {
        let name = Name::generic(abs.facet_count(), abs.rank());
        Self::new(abs, name)
    }
}

impl Polytope for NamedAbstract {
    type DualError = Infallible;

    fn abs(&self) -> &Abstract {
        &self.abs
    }

    fn abs_mut(&mut self) -> &mut Abstract {
        &mut self.abs
    }

    fn nullitope() -> Self {
        Self::new(Abstract::nullitope(), Name::Nullitope)
    }

    fn point() -> Self {
        Self::new(Abstract::point(), Name::Point)
    }

    fn dyad() -> Self {
        Self::new(Abstract::dyad(), Name::Dyad)
    }

    fn polygon(n: usize) -> Self {
        Self::new(Abstract::polygon(n), Name::polygon(Default::default(), n))
    }

    fn try_dual(&self) -> Result<Self, Self::DualError> {
        let abs = self.abs.dual();
        let name = Name::dual(
            self.name.clone(),
            Default::default(),
            abs.facet_count(),
            abs.rank(),
        );

        Ok(Self::new(abs, name))
    }

    fn try_dual_mut(&mut self) -> Result<(), Self::DualError> {
        self.abs.dual_mut();
        self.name = Name::dual(
            mem::take(&mut self.name),
            Default::default(),
            self.abs.facet_count(),
            self.abs.rank(),
        );

        Ok(())
    }

    fn comp_append(&mut self, _p: Self) {
        // Compound names are TBA
        todo!()
    }

    fn element(&self, rank: usize, idx: usize) -> Option<Self> {
        Some(Self::new_generic(self.abs.element(rank, idx)?))
    }

    fn petrial_mut(&mut self) -> bool {
        let res = self.abs.petrial_mut();
        if res {
            self.name = Name::petrial(mem::take(&mut self.name));
        }
        res
    }

    fn petrie_polygon_with(&mut self, flag: miratope_core::abs::flag::Flag) -> Option<Self> {
        Some(Self::new_generic(self.abs.petrie_polygon_with(flag)?))
    }

    fn omnitruncate(&self) -> Self {
        Self::new_generic(self.abs.omnitruncate())
    }

    fn prism(&self) -> Self {
        Self::new(self.abs().prism(), self.name.clone().prism())
    }

    fn tegum(&self) -> Self {
        Self::new(self.abs().tegum(), self.name.clone().tegum())
    }

    fn pyramid(&self) -> Self {
        Self::new(self.abs().pyramid(), self.name.clone().pyramid())
    }

    fn duopyramid(p: &Self, q: &Self) -> Self {
        Self::new(
            Abstract::duopyramid(&p.abs, &q.abs),
            Name::multipyramid(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duoprism(p: &Self, q: &Self) -> Self {
        Self::new(
            Abstract::duoprism(&p.abs, &q.abs),
            Name::multiprism(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duotegum(p: &Self, q: &Self) -> Self {
        Self::new(
            Abstract::duotegum(&p.abs, &q.abs),
            Name::multitegum(vec![p.name.clone(), q.name.clone()]),
        )
    }

    fn duocomb(p: &Self, q: &Self) -> Self {
        Self::new(
            Abstract::duocomb(&p.abs, &q.abs),
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

    fn try_antiprism(&self) -> Result<Self, Self::DualError> {
        Ok(Self::new(
            self.abs().try_antiprism()?,
            Name::antiprism(self.name.clone()),
        ))
    }

    fn simplex(rank: usize) -> Self {
        Self::new(
            Abstract::simplex(rank),
            Name::simplex(Default::default(), rank),
        )
    }

    fn hypercube(rank: usize) -> Self {
        Self::new(
            Abstract::hypercube(rank),
            Name::hyperblock(Default::default(), rank),
        )
    }

    fn orthoplex(rank: usize) -> Self {
        Self::new(
            Abstract::orthoplex(rank),
            Name::orthoplex(Default::default(), rank),
        )
    }
}

impl Borrow<Name<Abs>> for NamedAbstract {
    fn borrow(&self) -> &Name<Abs> {
        &self.name
    }
}

impl BorrowMut<Name<Abs>> for NamedAbstract {
    fn borrow_mut(&mut self) -> &mut Name<Abs> {
        &mut self.name
    }
}

impl NamedPolytope<Abs> for NamedAbstract {}
