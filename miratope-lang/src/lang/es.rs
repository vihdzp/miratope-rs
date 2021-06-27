use crate::{Bigender, GreekPrefix, Language, Options, Position, Prefix};

use miratope_core::abs::rank::Rank;

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
    type Count = crate::Plural;
    type Gender = Bigender;

    /// The default position to place adjectives. This will be used for the
    /// default implementations, but it can be overridden in any specific case.
    fn default_pos() -> Position {
        Position::After
    }

    /// Returns the suffix for a d-polytope. Only needs to work up to d = 20, we
    /// won't offer support any higher than that.
    fn suffix(d: Rank, options: Options<Self::Count, Self::Gender>) -> String {
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
    fn nullitope(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six(
            "nulitopo",
            "nulitopos",
            "nulitópico",
            "nulitópicos",
            "nulitópica",
            "nulitópicas",
        )
    }

    /// The name of a point.
    fn point(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.four("punto", "puntos", "puntual", "puntuales")
    }

    /// The name of a dyad.
    fn dyad(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six(
            "díada",
            "díadas",
            "diádico",
            "diádicos",
            "diádica",
            "diádicas",
        )
    }

    /// The name of a triangle.
    fn triangle(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.four("triángulo", "triángulos", "triangular", "triangulares")
    }

    /// The name of a square.
    fn square(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six(
            "cuadrado",
            "cuadrados",
            "cuadrado",
            "cuadrados",
            "cuadrada",
            "cuadradas",
        )
    }

    /// The name of a rectangle.
    fn rectangle(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.four("rectángulo", "rectángulos", "rectangular", "rectangulares")
    }

    /// The generic name for a polytope with `n` facets in `d` dimensions.
    fn generic(n: usize, d: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        let mut prefix = Self::prefix(n);

        if d == Rank::new(2) && !options.adjective {
            prefix = last_vowel_tilde(&prefix);
        }

        prefix + &Self::suffix(d, options)
    }

    /// The name for a pyramid.
    fn pyramid(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.four("pirámide", "pirámides", "piramidal", "piramidales")
    }

    /// The gender of the "pyramid" noun. We assume this is shared by
    /// "multipyramid".
    fn pyramid_gender() -> Self::Gender {
        Bigender::Female
    }

    /// The name for a prism.
    fn prism(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six(
            "prisma",
            "prismas",
            "prismático",
            "prismáticos",
            "prismática",
            "prismáticas",
        )
    }

    /// The name for a tegum.
    fn tegum(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six(
            "tego",
            "tegos",
            "tegmático",
            "tegmáticos",
            "tegmática",
            "tegmáticas",
        )
    }

    /// The name for a comb.
    fn comb(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.two("panal", "panales")
    }

    fn hyperblock(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
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
    fn hypercube(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
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

    /// The adjective for a "dual" polytope.
    fn dual(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.two("dual", "duales")
    }

    /// The name for an antiprism.
    fn antiprism(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six(
            "antiprisma",
            "antiprismas",
            "antiprismático",
            "antiprismáticos",
            "antiprismática",
            "antiprismáticas",
        )
    }

    /// The name for an antitegum.
    fn antitegum(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six(
            "antitego",
            "antitegos",
            "antitegmático",
            "antitegmáticos",
            "antitegmática",
            "antitegmáticas",
        )
    }

    /// The adjective for a Petrial.
    fn petrial(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "Petrial"
    }

    /// The adjective for a "great" version of a polytope.
    fn great(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "gran"
    }

    /// The adjective for a "small" version of a polytope.
    fn small(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six(
            "pequeño",
            "pequeños",
            "pequeño",
            "pequeños",
            "pequeña",
            "pequeñas",
        )
    }

    /// The adjective for a "stellated" version of a polytope.
    fn stellated(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six(
            "estrellado",
            "estrellados",
            "estrellado",
            "estrellados",
            "estrellada",
            "estrelladas",
        )
    }
}
