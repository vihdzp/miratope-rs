//! Contains the named abstract and concrete polytope types.

use miratope_core::Polytope;

use crate::{
    lang::En,
    name::{Name, NameType},
    Language,
};

pub mod abs;
pub mod conc;

pub trait NamedPolytope<T: NameType>: Polytope {
    fn name(&self) -> &Name<T>;

    fn name_mut(&mut self) -> &mut Name<T>;

    fn wiki_link(&self) -> String {
        crate::WIKI_LINK.to_owned() + &En::parse(self.name()).replace(" ", "_")
    }
}
