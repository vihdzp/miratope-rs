//! The folder containing all of the different languages Miratope has been
//! translated into. Each file contains one specific language.

pub mod en;
pub mod es;
pub use en::En;
pub use es::Es;

/// Returns `true` if `c` matches any of `a`, `e`, `i`, `o`, or `u`.
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}
