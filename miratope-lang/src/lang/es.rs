//! Implements the Spanish language.
use crate::{gender::Bigender, greek_prefixes, GreekPrefix, Language, Position, Prefix};

impl Bigender {
    /// Adds either an 'o' or an 'a' to the end of a string depending on the gender.
    fn choose_auto(self, str: String) -> String {
        str + self.choose("o", "a")
    }
}

/// The Spanish language.
pub struct Es;

impl GreekPrefix for Es {
    greek_prefixes! {
        UNITS = [
            "", "hena", "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "enea",
        ];
        CHILIA = "quilia";
        DISCHILIA = "disquilia";
        TRISCHILIA = "trisquilia";
        MYRIA = "miria";
        DISMYRIA = "dismiria";
        TRISMYRIA = "trismiria";
    }
}

impl Prefix for Es {
    /// Converts a number into its Greek prefix equivalent.
    ///
    /// # Safety
    /// If this method ever returns a non-ASCII String, it might cause UB in
    /// [`Self::hypercube_prefix`]. Since letters such as á, é, í, ó, ú, ü are
    /// not in the ASCII range, we need to be **really careful** here.
    fn prefix(n: usize) -> String {
        Self::greek_prefix(n)
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

            // Converts a c into a qu.
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

fn polygon_prefix(n: usize) -> String {
    let mut chars = Es::prefix(n).chars().collect::<Vec<_>>();
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

const SUFFIXES: [&str; 21] = [
    "mon", "tel", "gon", "edr", "cor", "ter", "pet", "ex", "zet", "yot", "xen", "dac", "hendac",
    "doc", "tradac", "teradac", "petadac", "exdac", "zetadac", "yotadac", "xendac",
];

impl Language for Es {
    type Gender = Bigender;

    /// The default position to place adjectives. This will be used for the
    /// default implementations, but it can be overridden in any specific case.
    fn default_pos() -> Position {
        Position::After
    }

    fn suffix_noun_str(rank: usize) -> String {
        SUFFIXES[rank].to_owned() + "o"
    }

    fn suffix_gender(_: usize) -> Self::Gender {
        Bigender::Male
    }

    fn suffix_adj(_: Self::Gender, rank: usize) -> String {
        SUFFIXES[rank].to_owned() + "al"
    }

    fn generic_noun_str(facet_count: usize, rank: usize) -> String {
        if rank == 3 {
            polygon_prefix(facet_count) + &Self::suffix_noun_str(rank)
        } else {
            Self::prefix(facet_count) + &Self::suffix_noun_str(rank)
        }
    }

    fn nullitope_noun_str() -> String {
        "nulítopo".to_owned()
    }

    fn nullitope_gender() -> Self::Gender {
        Bigender::Male
    }

    fn nullitope_adj(gender: Self::Gender) -> String {
        gender.choose_auto("nulitópic".to_owned())
    }

    fn point_noun_str() -> String {
        "punto".to_owned()
    }

    fn point_gender() -> Self::Gender {
        Bigender::Male
    }

    fn point_adj(_: Self::Gender) -> String {
        "puntual".to_owned()
    }

    fn dyad_noun_str() -> String {
        "díada".to_owned()
    }

    fn dyad_gender() -> Self::Gender {
        Bigender::Female
    }

    fn dyad_adj(gender: Self::Gender) -> String {
        gender.choose_auto("diádic".to_owned())
    }

    fn triangle_noun_str() -> String {
        "triángulo".to_owned()
    }

    fn triangle_gender() -> Self::Gender {
        Bigender::Male
    }

    fn triangle_adj(_: Self::Gender) -> String {
        "triangular".to_owned()
    }

    fn square_noun_str() -> String {
        "cuadrado".to_owned()
    }

    fn square_gender() -> Self::Gender {
        Bigender::Male
    }

    fn square_adj(gender: Self::Gender) -> String {
        gender.choose_auto("cuadrad".to_owned())
    }

    fn rectangle_noun_str() -> String {
        "rectángulo".to_owned()
    }

    fn rectangle_gender() -> Self::Gender {
        Bigender::Male
    }

    fn rectangle_adj(_: Self::Gender) -> String {
        "rectangular".to_owned()
    }

    fn pyramid_noun_str() -> String {
        "pirámide".to_owned()
    }

    fn pyramid_gender() -> Self::Gender {
        Bigender::Male
    }

    fn pyramid_adj(_: Self::Gender) -> String {
        "piramidal".to_owned()
    }

    fn prism_noun_str() -> String {
        "prisma".to_owned()
    }

    fn prism_gender() -> Self::Gender {
        Bigender::Male
    }

    fn prism_adj(gender: Self::Gender) -> String {
        gender.choose_auto("prismátic".to_owned())
    }

    fn tegum_noun_str() -> String {
        "tego".to_owned()
    }

    fn tegum_gender() -> Self::Gender {
        Bigender::Male
    }

    fn tegum_adj(gender: Self::Gender) -> String {
        gender.choose_auto("tegmátic".to_owned())
    }

    fn comb_noun_str() -> String {
        "panal".to_owned()
    }

    fn comb_gender() -> Self::Gender {
        Bigender::Male
    }

    fn comb_adj(_: Self::Gender) -> String {
        "panal".to_owned()
    }

    fn antiprism_noun_str() -> String {
        "antiprisma".to_owned()
    }

    fn antiprism_gender() -> Self::Gender {
        Bigender::Male
    }

    fn antiprism_adj(gender: Self::Gender) -> String {
        gender.choose_auto("antiprismátic".to_owned())
    }

    fn antitegum_noun_str() -> String {
        "antitego".to_owned()
    }

    fn antitegum_gender() -> Self::Gender {
        Bigender::Male
    }

    fn antitegum_adj(gender: Self::Gender) -> String {
        gender.choose_auto("antitegmátic".to_owned())
    }

    fn hosotope_noun_str(rank: usize) -> String {
        "hoso".to_owned() + &Self::suffix_noun_str(rank)
    }

    fn hosotope_gender(_: usize) -> Self::Gender {
        Bigender::Male
    }

    fn hosotope_adj(gender: Self::Gender, rank: usize) -> String {
        "hoso".to_owned() + &Self::suffix_adj(gender, rank)
    }

    fn ditope_noun_str(rank: usize) -> String {
        "hoso".to_owned() + &Self::suffix_noun_str(rank)
    }

    fn ditope_gender(_: usize) -> Self::Gender {
        Bigender::Male
    }

    fn ditope_adj(gender: Self::Gender, rank: usize) -> String {
        "di".to_owned() + &Self::suffix_adj(gender, rank)
    }

    fn petrial_adj(_: Self::Gender) -> String {
        "Petrial".to_owned()
    }

    fn simplex_noun_str(rank: usize) -> String {
        Self::generic_noun_str(rank, rank)
    }

    fn simplex_adj(gender: Self::Gender, rank: usize) -> String {
        Self::generic_adj(gender, rank, rank)
    }

    fn cuboid_noun_str() -> String {
        "cuboide".to_owned()
    }

    fn cuboid_adj(_: Self::Gender) -> String {
        "cuboidal".to_owned()
    }

    fn cube_noun_str() -> String {
        "cubo".to_owned()
    }

    fn cube_adj(gender: Self::Gender) -> String {
        gender.choose_auto("cúbic".to_owned())
    }

    fn hyperblock_noun_str(rank: usize) -> String {
        Self::greek_prefix(rank) + "bloque"
    }

    fn hyperblock_adj(_: Self::Gender, rank: usize) -> String {
        Self::greek_prefix(rank) + "bloque"
    }

    fn hypercube_noun_str(rank: usize) -> String {
        Self::greek_prefix(rank) + "racto"
    }

    fn hypercube_adj(gender: Self::Gender, rank: usize) -> String {
        gender.choose_auto(Self::greek_prefix(rank) + "ráctic")
    }

    fn orthoplex_noun_str(rank: usize) -> String {
        Self::generic_noun_str(1 << rank, rank)
    }

    fn orthoplex_adj(gender: Self::Gender, rank: usize) -> String {
        Self::generic_adj(gender, 1 << rank, rank)
    }

    fn great_adj(_: Self::Gender) -> String {
        "gran".to_owned()
    }

    fn great_pos() -> Position {
        Position::Before
    }

    fn small_adj(gender: Self::Gender) -> String {
        gender.choose_auto("pequeñ".to_owned())
    }

    fn stellated_adj(gender: Self::Gender) -> String {
        gender.choose_auto("estrellad".to_owned())
    }

    fn dual_adj(_: Self::Gender) -> String {
        "dual".to_owned()
    }
}
