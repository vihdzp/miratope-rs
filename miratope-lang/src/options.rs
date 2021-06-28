/// Represents the grammatical genders in any given language. We assume that
/// these propagate from nouns to adjectives, i.e. an adjective that describes
/// a given noun is declensed with the gender of the noun.
///
/// The most common implementations of this trait are [`Agender`] (for languages
/// without grammatical gender) and [`Bigender`] (for languages with a
/// male/female distinction).
///
/// Genders must have a default value that will be given by adjectives by
/// default. This can then be overridden on a case-by-case basis.
pub trait Gender: Copy + Default {}

/// The gender system for a non-gendered language.
#[derive(Clone, Copy, Default)]
pub struct Agender;

impl Gender for Agender {}

/// The gender system for a language with a male/female distinction.
#[derive(Clone, Copy)]
pub enum Bigender {
    /// Male gender.
    Male,

    /// Female gender.
    Female,
}

impl Default for Bigender {
    fn default() -> Self {
        Self::Male
    }
}

impl Gender for Bigender {}

/// Represents the grammatical numbers in any given language. These propagate
/// from nouns to adjectives, i.e. an adjective that describes a given noun is
/// declensed with the count of the noun.
pub trait Count: Copy {
    /// The grammatical number corresponding to a given number count.
    fn from_count(n: usize) -> Self;
}

/// The number system for a language with a singular/plural distinction.
#[derive(Clone, Copy)]
pub enum Plural {
    /// Exactly one object.
    One,

    /// Two or more objects.
    More,
}

impl Plural {
    /// Returns whether `self` matches `Self::One`.
    pub fn is_one(self) -> bool {
        matches!(self, Self::One)
    }
}

impl Count for Plural {
    fn from_count(n: usize) -> Self {
        if n == 1 {
            Self::One
        } else {
            Self::More
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Represents the different modifiers that can be applied to a term.
///
/// This struct is internal and is modified as any given [`Name`] is parsed.
/// We might want to make a "public" version of this struct. Or not.
pub struct Options<C: Count, G: Gender> {
    /// Determines whether the polytope acts as an adjective.
    pub adjective: bool,

    /// The grammatical number corresponding to the number of polytopes.
    pub count: C,

    /// The grammatical gender of the polytope.
    pub gender: G,
}

impl<C: Count, G: Gender> Default for Options<C, G> {
    /// The options default to a single polytope, as a noun, in the default gender.
    fn default() -> Self {
        Options {
            adjective: false,
            count: C::from_count(1),
            gender: Default::default(),
        }
    }
}

impl<G: Gender> Options<Plural, G> {
    /// Chooses a suffix from two options:
    ///
    /// * Base form.
    /// * A plural.
    ///
    /// The adjectives will take the same form as the nouns.
    pub fn two<'a>(&self, base: &'a str, plural: &'a str) -> &'a str {
        if self.count.is_one() {
            base
        } else {
            plural
        }
    }

    /// Chooses a suffix from three options:
    ///
    /// * Base form.
    /// * A plural.
    /// * An adjective for both the singular and plural.
    pub fn three<'a>(&self, base: &'a str, plural: &'a str, adj: &'a str) -> &'a str {
        if self.adjective {
            adj
        } else if self.count.is_one() {
            base
        } else {
            plural
        }
    }

    /// Chooses a suffix from four options:
    ///
    /// * Base form.
    /// * A plural.
    /// * A singular adjective.
    /// * A plural adjective.
    pub fn four<'a>(
        &self,
        base: &'a str,
        plural: &'a str,
        adj: &'a str,
        plural_adj: &'a str,
    ) -> &'a str {
        if self.adjective {
            if self.count.is_one() {
                adj
            } else {
                plural_adj
            }
        } else if self.count.is_one() {
            base
        } else {
            plural
        }
    }
}

impl Options<Plural, Bigender> {
    /// Chooses a suffix for an adjective from four options:
    ///
    /// * A singular adjective (male).
    /// * A plural adjective (male).
    /// * A singular adjective (female).
    /// * A plural adjective (female).
    ///
    /// Assumes that the word will be used as an adjective, regardless of
    /// `self.adjective`.
    pub fn four_adj<'a>(
        &self,
        adj_m: &'a str,
        plural_adj_m: &'a str,
        adj_f: &'a str,
        plural_adj_f: &'a str,
    ) -> &'a str {
        if self.count.is_one() {
            match self.gender {
                Bigender::Male => adj_m,
                Bigender::Female => adj_f,
            }
        } else {
            match self.gender {
                Bigender::Male => plural_adj_m,
                Bigender::Female => plural_adj_f,
            }
        }
    }

    /// Chooses a suffix from six options:
    ///
    /// * Base form.
    /// * A plural.
    /// * A singular adjective (male).
    /// * A plural adjective (male).
    /// * A singular adjective (female).
    /// * A plural adjective (female).
    pub fn six<'a>(
        &self,
        base: &'a str,
        plural: &'a str,
        adj_m: &'a str,
        plural_adj_m: &'a str,
        adj_f: &'a str,
        plural_adj_f: &'a str,
    ) -> &'a str {
        if self.adjective {
            self.four_adj(adj_m, plural_adj_m, adj_f, plural_adj_f)
        } else if self.count.is_one() {
            base
        } else {
            plural
        }
    }
}
