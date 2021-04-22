pub struct Es;

use super::super::{GreekPrefix, Language, Options, Prefix};

impl GreekPrefix for Es {}

/// In Spanish, polygon names have the last vowel in their prefix accented.
/// This function places such accent.
fn last_vowel_tilde(prefix: String) -> String {
    let mut chars = prefix.chars().collect::<Vec<_>>();
    for c in chars.iter_mut().rev() {
        match c {
            'a' => {
                *c = 'á';
                break;
            }
            'e' => {
                *c = 'é';
                break;
            }
            'i' => {
                *c = 'í';
                break;
            }
            'o' => {
                *c = 'ó';
                break;
            }
            'u' => {
                *c = 'ú';
                break;
            }
            _ => {}
        }
    }
    chars.into_iter().collect()
}

impl Language for Es {
    fn basic(n: usize, d: usize, options: Options) -> String {
        let mut prefix = Self::prefix(n);

        if d == 2 && !options.adjective {
            prefix = last_vowel_tilde(prefix);
        }

        format!("{}{}", prefix, Self::suffix(d, options))
    }

    fn unknown() -> String {
        String::from("desconocido")
    }
}
