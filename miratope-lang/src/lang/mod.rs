//! The folder containing all of the different languages Miratope has been
//! translated into. Each file contains one specific language.

pub mod de;
pub mod en;
pub mod es;
pub use de::De;
pub use en::En;
pub use es::Es;

pub(crate) fn uppercase_mut(str: &mut String) {
    // The first character of the result.
    let c = str.chars().next();

    if let Some(c) = c {
        if c.is_ascii() {
            // Safety: c and c.to_ascii_uppercase() are a single byte.
            // Therefore, we can just replace one by the other.
            unsafe {
                str.as_bytes_mut()[0] = c.to_ascii_uppercase() as u8;
            }
        }
    }
}

/// Returns `true` if `c` matches any of `a`, `e`, `i`, `o`, or `u`.
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}
