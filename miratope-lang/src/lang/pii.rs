//! Reconstructed Proto-Indo Iranian. Credits to Gap.

use super::{Language, Options, Prefix};

/// Reconstructed Proto-Indo Iranian.
pub struct Pii;

impl Prefix for Pii {}

impl Language for Pii {
    type Count = super::Plural; // Probably not true.
    type Gender = super::Agender;

    fn nullitope(_options: Options<Self::Count, Self::Gender>) -> String {
        todo!()
    }
}
