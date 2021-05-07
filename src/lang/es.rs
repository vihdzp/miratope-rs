use crate::{
    lang::{
        name::{Name, NameType},
        Gender,
    },
    polytope::r#abstract::rank::Rank,
};

use super::{GreekPrefix, Language, Options, Prefix};

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
    /// Returns the suffix for a d-polytope. Only needs to work up to d = 20, we
    /// won't offer support any higher than that.
    fn suffix(d: Rank, options: Options) -> String {
        const SUFFIXES: [&str; 21] = [
            "mon", "tel", "gon", "edr", "cor", "ter", "pet", "ex", "zet", "yot", "xen", "dac",
            "hendac", "doc", "tradac", "teradac", "petadac", "exdac", "zetadac", "yotadac",
            "xendac",
        ];

        format!(
            "{}{}",
            SUFFIXES[d.usize()],
            options.four("o", "os", "al", "ales")
        )
    }

    /// The name of a nullitope.
    fn nullitope(options: Options) -> String {
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
    fn point(options: Options) -> String {
        format!("punt{}", options.four("o", "os", "ual", "uales"))
    }

    /// The name of a dyad.
    fn dyad(options: Options) -> String {
        format!(
            "d{}",
            options.six("íada", "íadas", "iádico", "iádicos", "iádica", "iádicas")
        )
    }

    /// The name of a triangle.
    fn triangle(options: Options) -> String {
        format!(
            "tri{}",
            options.four("ángulo", "ángulos", "angular", "angulares")
        )
    }

    /// The name of a square.
    fn square(options: Options) -> String {
        format!("cuadrad{}", options.six("o", "os", "o", "os", "a", "as"))
    }

    /// The name of a rectangle.
    fn rectangle(options: Options) -> String {
        format!(
            "rect{}",
            options.four("ángulo", "ángulos", "angular", "angulares")
        )
    }

    /// The generic name for a polytope with `n` facets in `d` dimensions.
    fn generic(n: usize, d: Rank, options: Options) -> String {
        let mut prefix = Self::prefix(n);

        if d == Rank::new(2) && !options.adjective {
            prefix = last_vowel_tilde(prefix);
        }

        format!("{}{}", prefix, Self::suffix(d, options))
    }

    /// The name for a pyramid.
    fn pyramid(options: Options) -> String {
        format!(
            "pir{}",
            options.four("ámide", "ámides", "amidal", "amidales")
        )
    }

    /// The name for a pyramid with a given base.
    fn pyramid_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        format!(
            "{} {}",
            Self::pyramid(options),
            Self::base_adj(
                base,
                Options {
                    gender: options.gender | Gender::Female,
                    ..options
                }
            )
        )
    }

    /// The name for a prism.
    fn prism(options: Options) -> String {
        format!(
            "prism{}",
            options.six("a", "as", "ático", "áticos", "ática", "áticas")
        )
    }

    /// The name for a prism with a given base.
    fn prism_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        format!(
            "{} {}",
            Self::prism(options),
            Self::base_adj(
                base,
                Options {
                    gender: options.gender | Gender::Male,
                    ..options
                }
            )
        )
    }

    /// The name for a tegum.
    fn tegum(options: Options) -> String {
        format!(
            "teg{}",
            options.six("o", "os", "mático", "máticos", "mática", "máticas")
        )
    }

    /// The name for a tegum with a given base.
    fn tegum_of<T: NameType>(base: &Name<T>, options: Options) -> String {
        format!(
            "{} {}",
            Self::tegum(options),
            Self::base_adj(
                base,
                Options {
                    gender: options.gender | Gender::Male,
                    ..options
                }
            )
        )
    }

    fn multiproduct<T: NameType>(name: &Name<T>, options: Options) -> String {
        // Gets the bases and the kind of multiproduct.
        let (bases, kind, gender) = match name {
            Name::Multipyramid(bases) => (bases, Self::pyramid(options), Gender::Female),
            Name::Multiprism(bases) => (bases, Self::prism(options), Gender::Male),
            Name::Multitegum(bases) => (bases, Self::tegum(options), Gender::Male),
            Name::Multicomb(bases) => (
                bases,
                String::from(options.four("panal", "panales", "de panal", "de panales")),
                Gender::Male,
            ),
            _ => panic!("Not a product!"),
        };
        let gender = options.gender | gender;

        let n = bases.len();
        let prefix = match n {
            2 => String::from("duo"),
            3 => String::from("trio"),
            _ => Self::prefix(n),
        };
        let kind = format!("{}{}", prefix, kind);

        let mut str_bases = String::new();
        let new_options = Options { gender, ..options };
        let (last, bases) = bases.split_last().unwrap();
        for base in bases {
            str_bases.push_str(&Self::base_adj(base, new_options));
            str_bases.push('-');
        }
        str_bases.push_str(&Self::base_adj(last, new_options));

        format!("{} {}", kind, str_bases)
    }

    fn hyperblock(rank: Rank, options: Options) -> String {
        match rank.usize() {
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
    fn hypercube(rank: Rank, options: Options) -> String {
        match rank.usize() {
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
    fn orthoplex(rank: Rank, options: Options) -> String {
        Self::generic(2u32.pow(rank.u32()) as usize, rank, options)
    }

    fn compound<T: NameType>(components: &[(usize, Name<T>)], options: Options) -> String {
        let ((last_rep, last_component), first_components) = components.split_last().unwrap();
        let mut str = String::from(options.four(
            "compuesto",
            "compuestos",
            "del compuesto",
            "de los compuestos",
        ));
        str.push_str(" de");

        let parse_component = |rep, component| {
            Self::parse(
                component,
                Options {
                    count: rep,
                    ..Options::default()
                },
            )
        };

        let comma = if components.len() == 2 { "" } else { "," };
        for (rep, component) in first_components {
            str.push_str(&format!(
                "{} {}{} ",
                rep,
                parse_component(*rep, component),
                comma
            ));
        }

        str.push_str(&format!(
            "y {} {}",
            last_rep,
            parse_component(*last_rep, last_component)
        ));

        str
    }
}
