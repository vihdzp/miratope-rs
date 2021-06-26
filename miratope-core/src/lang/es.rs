use crate::{
    abs::rank::Rank,
    lang::name::{Name, NameType},
};

use super::{Bigender, GreekPrefix, Language, Options, Prefix};

/// The Spanish language.
pub struct Es;

impl GreekPrefix for Es {
    const UNITS: [&'static str; 10] = [
        "", "hena", "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "enea",
    ];

    const CHILIA: &'static str = "quilia";

    const DISCHILIA: &'static str = "disquilia";

    const TRISCHILIA: &'static str = "trisquilia";

    const MYRIA: &'static str = "miria";

    const DISMYRIA: &'static str = "dismiria";

    const TRISMYRIA: &'static str = "trismiria";
}

/// In Spanish, polygon names have the last vowel in their prefix accented.
/// This function places such accent.
fn last_vowel_tilde(prefix: &str) -> String {
    let mut chars = prefix.chars().collect::<Vec<_>>();
    for c in chars.iter_mut().rev() {
        *c = match c {
            'a' => 'á',
            'e' => 'é',
            'i' => 'í',
            'o' => 'ó',
            'u' => 'ú',
            _ => continue,
        };

        break;
    }

    chars.into_iter().collect()
}

impl Language for Es {
    type Gender = Bigender;

    /// Returns the suffix for a d-polytope. Only needs to work up to d = 20, we
    /// won't offer support any higher than that.
    fn suffix(d: Rank, options: Options<Self::Gender>) -> String {
        const SUFFIXES: [&str; 21] = [
            "mon", "tel", "gon", "edr", "cor", "ter", "pet", "ex", "zet", "yot", "xen", "dac",
            "hendac", "doc", "tradac", "teradac", "petadac", "exdac", "zetadac", "yotadac",
            "xendac",
        ];

        format!(
            "{}{}",
            SUFFIXES[d.into_usize()],
            options.four("o", "os", "al", "ales")
        )
    }

    /// The name of a nullitope.
    fn nullitope(options: Options<Self::Gender>) -> String {
        format!(
            "nul{}",
            options.six(
                "itopo",
                "itopos",
                "itópico",
                "itópicos",
                "itópica",
                "itópicas"
            )
        )
    }

    /// The name of a point.
    fn point(options: Options<Self::Gender>) -> String {
        format!("punt{}", options.four("o", "os", "ual", "uales"))
    }

    /// The name of a dyad.
    fn dyad(options: Options<Self::Gender>) -> String {
        format!(
            "d{}",
            options.six("íada", "íadas", "iádico", "iádicos", "iádica", "iádicas")
        )
    }

    /// The name of a triangle.
    fn triangle(options: Options<Self::Gender>) -> String {
        format!(
            "tri{}",
            options.four("ángulo", "ángulos", "angular", "angulares")
        )
    }

    /// The name of a square.
    fn square(options: Options<Self::Gender>) -> String {
        String::from("cuadrad") + options.six("o", "os", "o", "os", "a", "as")
    }

    /// The name of a rectangle.
    fn rectangle(options: Options<Self::Gender>) -> String {
        String::from("rect") + options.four("ángulo", "ángulos", "angular", "angulares")
    }

    /// The generic name for a polytope with `n` facets in `d` dimensions.
    fn generic(n: usize, d: Rank, options: Options<Self::Gender>) -> String {
        let mut prefix = Self::prefix(n);

        if d == Rank::new(2) && !options.adjective {
            prefix = last_vowel_tilde(&prefix);
        }

        prefix + &Self::suffix(d, options)
    }

    /// The name for a pyramid.
    fn pyramid(options: Options<Self::Gender>) -> String {
        String::from("pir") + options.four("ámide", "ámides", "amidal", "amidales")
    }

    /// The name for a pyramid with a given base.
    fn pyramid_of<T: NameType>(base: &Name<T>, options: Options<Self::Gender>) -> String {
        format!(
            "{} {}",
            Self::pyramid(options),
            Self::to_adj_with(base, options, Bigender::Female)
        )
    }

    /// The name for a prism.
    fn prism(options: Options<Self::Gender>) -> String {
        "prism".to_owned() + options.six("a", "as", "ático", "áticos", "ática", "áticas")
    }

    /// The name for a prism with a given base.
    fn prism_of<T: NameType>(base: &Name<T>, options: Options<Self::Gender>) -> String {
        format!(
            "{} {}",
            Self::prism(options),
            Self::to_adj_with(base, options, Bigender::Male)
        )
    }

    /// The name for a tegum.
    fn tegum(options: Options<Self::Gender>) -> String {
        "teg".to_owned() + options.six("o", "os", "mático", "máticos", "mática", "máticas")
    }

    /// The name for a tegum with a given base.
    fn tegum_of<T: NameType>(base: &Name<T>, options: Options<Self::Gender>) -> String {
        format!(
            "{} {}",
            Self::tegum(options),
            Self::to_adj_with(base, options, Bigender::Male)
        )
    }

    fn multiproduct<T: NameType>(name: &Name<T>, options: Options<Self::Gender>) -> String {
        // Gets the bases and the kind of multiproduct.
        let (bases, kind, gender) = match name {
            Name::Multipyramid(bases) => (bases, Self::pyramid(options), Bigender::Female),
            Name::Multiprism(bases) => (bases, Self::prism(options), Bigender::Male),
            Name::Multitegum(bases) => (bases, Self::tegum(options), Bigender::Male),
            Name::Multicomb(bases) => (
                bases,
                String::from(options.four("panal", "panales", "de panal", "de panales")),
                Bigender::Male,
            ),
            _ => panic!("Not a product!"),
        };

        let n = bases.len();
        let prefix = match n {
            2 => String::from("duo"),
            3 => String::from("trio"),
            _ => Self::prefix(n),
        };
        let kind = format!("{}{}", prefix, kind);

        let mut str_bases = String::new();
        let (last, bases) = bases.split_last().unwrap();
        for base in bases {
            str_bases.push_str(&Self::to_adj_with(base, options, gender));
            str_bases.push('-');
        }
        str_bases.push_str(&Self::to_adj_with(last, options, gender));

        format!("{} {}", kind, str_bases)
    }

    fn hyperblock(rank: Rank, options: Options<Self::Gender>) -> String {
        match rank.into_usize() {
            3 => format!("cuboid{}", options.four("e", "es", "al", "ales")),
            n => {
                format!(
                    "{} {}bloque{}",
                    if options.adjective { "de" } else { "" },
                    Self::prefix(n),
                    options.two("", "s")
                )
            }
        }
    }

    /// The name for a hypercube with a given rank.
    fn hypercube(rank: Rank, options: Options<Self::Gender>) -> String {
        match rank.into_usize() {
            3 => format!(
                "c{}",
                options.six("ubo", "ubos", "úbico", "úbicos", "úbica", "úbicas")
            ),
            4 => format!(
                "teser{}",
                options.six("acto", "actos", "áctico", "ácticoa", "áctica", "ácticas")
            ),
            n => {
                let prefix = Self::prefix(n).chars().collect::<Vec<_>>();

                // Penta -> Pente, or Deca -> Deque
                // Penta -> Pente, or Deca -> Deke
                let (_, str0) = prefix.split_last().unwrap();
                let (c1, str1) = str0.split_last().unwrap();

                let suffix = options.six("acto", "actos", "áctico", "ácticos", "áctica", "ácticas");
                if *c1 == 'c' {
                    format!("{}quer{}", str1.iter().collect::<String>(), suffix)
                } else {
                    format!("{}eract{}", str0.iter().collect::<String>(), suffix)
                }
            }
        }
    }

    /// The name for an orthoplex with a given rank.
    fn orthoplex(rank: Rank, options: Options<Self::Gender>) -> String {
        Self::generic(1 << rank.into_usize(), rank, options)
    }
}
