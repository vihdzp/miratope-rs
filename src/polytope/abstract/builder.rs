use super::{
    elements::{SubelementList, SuperelementList},
    rank::Rank,
    Abstract,
};
use crate::polytope::r#abstract::Subelements;
use crate::polytope::Polytope;

/// Builds a polytope from the bottom up.
pub struct AbstractBuilder(Abstract);

impl AbstractBuilder {
    pub fn new() -> Self {
        Self(Abstract::new())
    }

    pub fn with_capacity(rank: Rank) -> Self {
        Self(Abstract::with_capacity(rank))
    }

    pub fn push(&mut self, subelements: SubelementList) {
        self.0.push_subs(subelements)
    }

    pub fn push_single(&mut self) {
        self.0.push_empty();
    }

    pub fn push_vertices(&mut self, vertex_count: usize) {
        self.0.push_vertices(vertex_count);
    }

    pub fn push_max(&mut self) {
        self.0.push_max();
    }

    pub fn build(self) -> Abstract {
        self.0
    }
}

/// Builds a polytope from the top down. This API is very bad and I am very
/// tired. I need to think this a bit more.
pub struct AbstractBuilderRev(Abstract);

impl AbstractBuilderRev {
    pub fn new() -> Self {
        Self(Abstract::new())
    }

    pub fn with_capacity(rank: Rank) -> Self {
        Self(Abstract::with_capacity(rank))
    }

    pub fn push(&mut self, superelements: SuperelementList) {
        let mut subelements = SubelementList::with_capacity(superelements.len());

        for sup_el in superelements {
            subelements.push(Subelements(sup_el.0));
        }

        self.0.push_subs(subelements)
    }

    pub fn push_max(&mut self) {
        self.0.push_empty();
    }

    pub fn build(mut self) -> Abstract {
        self.0.dual_mut();
        self.0
    }
}
