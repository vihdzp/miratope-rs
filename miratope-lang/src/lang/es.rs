//! Implements the Spanish language.
use crate::{
    options::{Bigender, Options},
    GreekPrefix, Language, Position, Prefix,
};

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

impl Prefix for Es {
    fn prefix(n: usize) -> String {
        Self::greek_prefix(n)
    }

    fn polygon_prefix(n: usize) -> String {
        let mut chars = Self::prefix(n).chars().collect::<Vec<_>>();
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

    fn multi_prefix(n: usize) -> String {
        match n {
            2 => String::from("duo"),
            3 => String::from("trio"),
            _ => Self::prefix(n),
        }
    }

    /// In the case `n == 4`, we return "tese". Otherwise, this is the same as
    /// the usual Greek prefix, except that we apply the following
    /// transformation rules:
    /// - If the last letter is a vowel, we convert it into an 'e'.
    ///   - If the next to last letter is a 'c', we convert it into a 'qu'.
    /// - Otherwise, we add an 'e'.
    ///
    /// # Safety
    /// If [`Self::prefix`] returns a non-ASCII string, it might cause UB in
    /// this method.
    fn hypercube_prefix(n: usize) -> String {
        if n == 4 {
            return "tese".to_owned();
        }

        let mut prefix = Self::prefix(n);
        let mut chars = prefix.char_indices().rev();

        // The last letter.
        let (idx_last, c) = chars.next().unwrap();
        debug_assert!(c.is_ascii());

        // Converts a vowel into an e.
        if super::is_vowel(c) {
            // The previous to last letter.
            let (idx_prev, c) = chars.next().unwrap();
            debug_assert!(c.is_ascii());

            // Converts a c into a que.
            if c == 'c' {
                // SAFETY: `Self::prefix` consists of ASCII characters only.
                unsafe {
                    prefix.as_bytes_mut()[idx_prev] = b'q';
                    prefix.as_bytes_mut()[idx_last] = b'u';
                }
            } else {
                // SAFETY: `Self::prefix` consists of ASCII characters only.
                unsafe {
                    prefix.as_bytes_mut()[idx_last] = b'e';
                }

                return prefix;
            }
        }

        // Adds an 'e'.
        prefix.push('e');
        prefix
    }
}

impl Language for Es {
    type Count = crate::options::Plural;
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

        SUFFIXES[d.into_usize()].to_owned() + options.four("o", "os", "al", "ales")
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
        (if d == Rank::new(2) && !options.adjective {
            Self::polygon_prefix(n)
        } else {
            Self::prefix(n)
        }) + &Self::suffix(d, options)
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

    /// The name for a cuboid.
    fn cuboid(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.four("cuboide", "cuboides", "cuboidal", "cuboidales")
    }

    /// The name for a cube.
    fn cube(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.six("cubo", "cubos", "cúbico", "cúbicos", "cúbica", "cúbicas")
    }

    /// The name for a hyperblock with a given rank.
    fn hyperblock(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        Self::prefix(rank.into()) + options.two("bloque", "bloques")
    }

    /// The name for a hypercube with a given rank.
    fn hypercube(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        Self::hypercube_prefix(rank.into())
            + options.six(
                "racto",
                "ractos",
                "ráctico",
                "rácticos",
                "ráctica",
                "rácticas",
            )
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

    fn hosotope(rank: Rank, options: Options<Self::Count, Self::Gender>) -> String {
        "hoso".to_owned() + &Self::suffix(rank, options)
    }

    /// The adjective for a Petrial.
    fn petrial(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "Petrial"
    }

    /// The adjective for a "great" version of a polytope.
    fn great(_options: Options<Self::Count, Self::Gender>) -> &'static str {
        "gran"
    }

    /// The position of the "great" adjective.
    fn great_pos() -> Position {
        Position::Before
    }

    /// The adjective for a "small" version of a polytope.
    fn small(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.four_adj("pequeño", "pequeños", "pequeña", "pequeñas")
    }

    /// The position of the "small" adjective.
    fn small_pos() -> Position {
        Position::Before
    }

    /// The adjective for a "stellated" version of a polytope.
    fn stellated(options: Options<Self::Count, Self::Gender>) -> &'static str {
        options.four_adj("estrellado", "estrellados", "estrellada", "estrelladas")
    }
}
