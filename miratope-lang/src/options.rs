//! Contains all of the code that configures the [`Options`] for most languages.
//! This includes the traits for grammatical gender and count.

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
pub trait Gender: Copy + std::fmt::Debug + Default {}

/// The gender system for a non-gendered language.
#[derive(Clone, Copy, Debug, Default)]
pub struct Agender;

impl Gender for Agender {}

/// The gender system for a language with a male/female distinction.
#[derive(Clone, Copy, Debug)]
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

/// The word class of whatever we're currently parsing.
#[derive(Clone, Copy, Debug)]
pub enum WordClass<G: Gender> {
    /// We're parsing a noun.
    Noun,

    /// We're parsing an adjective, and we should use the specified gender to
    /// declense it.
    Adjective(G),
}

impl<G: Gender> WordClass<G> {
    pub fn is_noun(&self) -> bool {
        matches!(self, Self::Noun)
    }

    pub fn is_adj(&self) -> bool {
        matches!(self, Self::Adjective(_))
    }

    pub fn unwrap_adj(&self) -> G {
        if let &Self::Adjective(gender) = self {
            gender
        } else {
            panic!("Expected adjective, found {:?}", self)
        }
    }
}

impl<G: Gender> Default for WordClass<G> {
    fn default() -> Self {
        Self::Noun
    }
}

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

/// Represents the different modifiers that can be applied to a term.
///
/// This struct is internal and is modified as any given [`Name`] is parsed.
/// We might want to make a "public" version of this struct. Or not.
#[derive(Clone, Copy, Debug)]
pub struct Options<C: Count, G: Gender> {
    /// The word class corresponding to the polytope.
    pub class: WordClass<G>,

    /// The grammatical number corresponding to the number of polytopes.
    pub count: C,
}

impl<C: Count, G: Gender> Default for Options<C, G> {
    fn default() -> Self {
        Self {
            class: Default::default(),
            count: C::from_count(1),
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
        if self.class.is_adj() {
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
        if self.class.is_adj() {
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
    /// # Panics
    /// This method will panic if `self.class` is not an adjective.
    pub fn four_adj<'a>(
        &self,
        adj_m: &'a str,
        plural_adj_m: &'a str,
        adj_f: &'a str,
        plural_adj_f: &'a str,
    ) -> &'a str {
        let gender = self.class.unwrap_adj();

        if self.count.is_one() {
            match gender {
                Bigender::Male => adj_m,
                Bigender::Female => adj_f,
            }
        } else {
            match gender {
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
        if self.class.is_adj() {
            self.four_adj(adj_m, plural_adj_m, adj_f, plural_adj_f)
        } else if self.count.is_one() {
            base
        } else {
            plural
        }
    }
}
