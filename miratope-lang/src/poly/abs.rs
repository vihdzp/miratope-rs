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

    fn try_dual(&self) -> miratope_core::DualResult<Self> {
        let abs = self.abs.try_dual()?;
        let name = Name::dual(
            self.name.clone(),
            Default::default(),
            abs.facet_count(),
            abs.rank(),
        );
        Ok(Self::new(abs, name))
    }

    fn try_dual_mut(&mut self) -> miratope_core::DualResult<()> {
        self.abs.try_dual_mut()?;
        self.name = Name::dual(
            mem::take(&mut self.name),
            Default::default(),
            self.abs.facet_count(),
            self.abs.rank(),
        );
        Ok(())
    }

    fn comp_append(&mut self, _p: Self) {
        todo!()
    }

    fn element(&self, el: miratope_core::abs::elements::ElementRef) -> Option<Self> {
        Some(Self::new_generic(self.abs.element(el)?))
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

    fn omnitruncate(&mut self) -> Self {
        Self::new_generic(self.abs.omnitruncate())
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

    fn try_antiprism(&self) -> miratope_core::DualResult<Self> {
        todo!()
    }
}

impl NamedPolytope<Abs> for NamedAbstract {
    fn name(&self) -> &Name<Abs> {
        &self.name
    }

    fn name_mut(&mut self) -> &mut Name<Abs> {
        &mut self.name
    }
}
