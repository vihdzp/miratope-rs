pub struct Es;

use super::{GreekPrefix, Language, Options};

impl GreekPrefix for Es {
    fn zero(alone: bool) -> &'static str {
        if alone {
            "nuli"
        } else {
            ""
        }
    }

    fn thousand() -> &'static str {
        "quilia"
    }

    fn two_thousand() -> &'static str {
        "disquilia"
    }

    fn three_thousand() -> &'static str {
        "trisquilia"
    }

    fn ten_thousand() -> &'static str {
        "miria"
    }

    fn twenty_thousand() -> &'static str {
        "dismiria"
    }

    fn thirty_thousand() -> &'static str {
        "trismiria"
    }

    fn units(n: usize) -> &'static str {
        const UNITS: [&str; 10] = [
            "", "hena", "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "ennea",
        ];

        UNITS[n]
    }

    fn prefix(n: usize, _options: super::Options) -> String {
        Self::prefix_trailing(n, _options, true)
    }

    fn prefix_trailing(n: usize, _options: super::Options, alone: bool) -> String {
        match n {
            0 => Self::zero(alone).to_string(),
            1 => Self::one(alone).to_string(),
            2..=9 => Self::units(n).to_string(),
            11 => Self::eleven().to_string(),
            12 => Self::twelve().to_string(),
            10 | 13..=19 => format!("{}{}", Self::units(n % 10), Self::plus_ten()),
            20..=29 => format!("{}{}", Self::twenty(), Self::units(n % 10)),
            30..=39 => format!("{}{}", Self::thirty(), Self::units(n % 10)),
            40..=99 => format!(
                "{}{}{}",
                Self::units(n / 10),
                Self::tenfold(),
                Self::units(n % 10)
            ),
            100 => Self::hundred().to_string(),
            101..=199 => format!(
                "{}{}",
                Self::plus_hundred(),
                Self::prefix_trailing(n % 100, _options, false)
            ),
            200..=299 => format!(
                "{}{}",
                Self::two_hundred(),
                Self::prefix_trailing(n % 100, _options, false)
            ),
            300..=399 => format!(
                "{}{}",
                Self::three_hundred(),
                Self::prefix_trailing(n % 100, _options, false)
            ),
            400..=999 => format!(
                "{}{}{}",
                Self::units(n / 100),
                Self::hundredfold(),
                Self::prefix_trailing(n % 100, _options, false)
            ),
            1000..=1999 => format!(
                "{}{}",
                Self::thousand(),
                Self::prefix_trailing(n % 1000, _options, false)
            ),
            2000..=2999 => format!(
                "{}{}",
                Self::two_thousand(),
                Self::prefix_trailing(n % 1000, _options, false)
            ),
            3000..=3999 => format!(
                "{}{}",
                Self::three_thousand(),
                Self::prefix_trailing(n % 1000, _options, false)
            ),
            4000..=9999 => format!(
                "{}{}{}",
                Self::units(n / 1000),
                Self::thousand(),
                Self::prefix_trailing(n % 1000, _options, false)
            ),
            10000..=19999 => format!(
                "{}{}",
                Self::ten_thousand(),
                Self::prefix_trailing(n % 10000, _options, false)
            ),
            20000..=29999 => format!(
                "{}{}",
                Self::twenty_thousand(),
                Self::prefix_trailing(n % 10000, _options, false)
            ),
            30000..=39999 => format!(
                "{}{}",
                Self::thirty_thousand(),
                Self::prefix_trailing(n % 10000, _options, false)
            ),
            40000..=99999 => format!(
                "{}{}{}",
                Self::units(n / 10000),
                Self::ten_thousand(),
                Self::prefix_trailing(n % 10000, _options, false)
            ),
            _ => format!("{}-", n),
        }
    }
}

impl Language for Es {
    fn simple(n: usize, d: usize, options: Options) -> String {
        let mut prefix = Self::prefix(n, options);

        if d == 2 && !options.adjective {
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
            prefix = chars.into_iter().collect();
        }

        format!("{}{}", prefix, Self::suffix(d, options))
    }

    fn unknown() -> String {
        String::from("desconocido")
    }
}
