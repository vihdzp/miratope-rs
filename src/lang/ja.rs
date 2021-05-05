//! Credits to Nayuta Ito

use super::{Language, Options, Prefix};

pub struct Ja;

const UNITS: [&str; 10] = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
const TEN: &str = "十";
const HUNDRED: &str = "百";
const THOUSAND: &str = "千";
const TEN_THOUSAND: &str = "万";

impl Prefix for Ja {
    fn prefix(n: usize) -> String {
        match n {
            0..=9 => String::from(UNITS[n]),
            10..=19 => format!("{}{}", TEN, UNITS[n - 10]),
            20..=99 => format!("{}{}{}", UNITS[n / 10], TEN, UNITS[n % 10]),
            100..=199 => format!("{}{}", HUNDRED, Self::prefix(n - 100)),
            200..=999 => format!("{}{}{}", n / 100, HUNDRED, Self::prefix(n - 100)),
            1000..=1999 => format!("{}{}", THOUSAND, Self::prefix(n - 1000)),
            2000..=9999 => format!("{}{}{}", n / 1000, THOUSAND, Self::prefix(n % 1000)),
            10000..=99999999 => format!("{}{}{}", n / 10000, TEN_THOUSAND, Self::prefix(n % 10000)),
            _ => format!("{}-", n),
        }
    }
}

impl Language for Ja {
    fn point(_options: Options) -> String {
        String::from("点")
    }

    /// The name for a dyad.
    fn dyad(_options: Options) -> String {
        String::from("線分")
    }

    fn generic(n: usize, rank: usize, _options: Options) -> String {
        match rank {
            2 => format!("{}形", Self::prefix(n)),
            3 => format!("{}面体", Self::prefix(n)),
            4 => format!("{}胞体", Self::prefix(n)),
            _ => format!("{}-tope", n), // placeholder
        }
    }

    fn triangle(options: Options) -> String {
        Self::generic(3, 2, options)
    }

    fn rectangle(_options: Options) -> String {
        String::from("長方形")
    }

    fn square(_options: Options) -> String {
        String::from("正方形")
    }

    fn pyramid(_options: Options) -> String {
        String::from("錐")
    }
}
