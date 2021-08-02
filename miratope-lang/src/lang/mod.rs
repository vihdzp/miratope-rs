//! The folder containing all of the different languages Miratope has been
//! translated into. Each file contains one specific language.

//pub mod de;
pub mod en;
//pub mod es;
//pub use de::De;
pub use en::En;
//pub use es::Es;

pub(crate) fn uppercase_mut(str: &mut String) {
    let mut indices = str.char_indices();

    // The first character of the result.
    if let Some((idx_first, c_first)) = indices.next() {
        let range = if let Some((idx_next, _)) = indices.next() {
            idx_first..idx_next
        } else {
            idx_first..idx_first + c_first.len_utf8()
        };
        str.replace_range(range, &c_first.to_uppercase().to_string());
    }
}

/// Returns `true` if `c` matches any of `a`, `e`, `i`, `o`, or `u`.
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}
