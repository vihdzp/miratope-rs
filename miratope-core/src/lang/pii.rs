//! Reconstructed Proto-Indo Iranian. Credits to Gap.

use super::{Agender, Language, Options, Prefix};

/// Reconstructed Proto-Indo Iranian.
pub struct Pii;

impl Prefix for Pii {}

impl Language for Pii {
    type Gender = Agender;

    fn nullitope(_options: Options<Self::Gender>) -> String {
        todo!()
    }
}
