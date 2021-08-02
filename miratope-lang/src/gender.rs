/// Represents the grammatical genders in any given language. We assume that
/// these propagate from nouns to adjectives, i.e. an adjective that describes
/// a given noun is declensed with the gender of the noun.
///
/// The most common implementations of this trait are [`Agender`] (for languages
/// without grammatical gender) and [`Bigender`] (for languages with a
/// male/female distinction).
///
/// Genders must have a default value that will be given by adjectives by
/// default. This helps simplify implementations for non-gendered languages.
/// This can then be overridden on a case-by-case basis.
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

impl Bigender {
    /// Chooses one of two arguments lazily depending on `self`.
    pub fn choose_lazy<T, F1, F2>(self, m: F1, f: F2) -> T
    where
        F1: FnOnce() -> T,
        F2: FnOnce() -> T,
    {
        match self {
            Self::Male => m(),
            Self::Female => f(),
        }
    }

    /// Chooses one of two arguments depending on `self`.
    pub fn choose<T>(self, m: T, f: T) -> T {
        self.choose_lazy(|| m, || f)
    }
}

/// The gender system for a language with a male/female distinction.
#[derive(Clone, Copy)]
pub enum Trigender {
    /// Male gender.
    Male,

    /// Female gender.
    Female,

    /// Neuter gender.
    Neuter,
}

impl Default for Trigender {
    fn default() -> Self {
        Self::Neuter
    }
}

impl Gender for Trigender {}

impl Trigender {
    /// Chooses one of three arguments lazily depending on `self`.
    pub fn choose_lazy<T, F1, F2, F3>(self, m: F1, f: F2, n: F3) -> T
    where
        F1: FnOnce() -> T,
        F2: FnOnce() -> T,
        F3: FnOnce() -> T,
    {
        match self {
            Self::Male => m(),
            Self::Female => f(),
            Self::Neuter => n(),
        }
    }

    /// Chooses one of three arguments depending on `self`.
    pub fn choose<T>(self, m: T, f: T, n: T) -> T {
        self.choose_lazy(|| m, || f, || n)
    }
}
