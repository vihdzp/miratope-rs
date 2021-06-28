//! Contains the named abstract and concrete polytope types.

use miratope_core::Polytope;
use std::borrow::BorrowMut;

use crate::{
    lang::En,
    name::{Name, NameType},
    Language,
};

pub mod abs;
pub mod conc;

pub trait NamedPolytope<T: NameType>: Polytope + BorrowMut<Name<T>> {
    fn wiki_link(&self) -> String {
        crate::WIKI_LINK.to_owned() + &En::parse(self.borrow()).replace(" ", "_")
    }
}
