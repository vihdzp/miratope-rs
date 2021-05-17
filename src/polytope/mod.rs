//! Contains the definitions of the different traits and structs for
//! [polytopes](https://polytope.miraheze.org/wiki/Polytope), as well as some
//! basic methods to operate on them.

pub mod r#abstract;
pub mod concrete;

use self::r#abstract::{
    flag::{Flag, FlagEvent, FlagIter},
    rank::{Rank, RankVec},
};
use crate::lang::{
    self,
    name::{Name, NameData, NameType, Regular},
    Language,
};

/// The names for 0-elements, 1-elements, 2-elements, and so on.
const ELEMENT_NAMES: [&str; 11] = [
    "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta", "Xenna", "Daka",
];

/// The word "Components".
const COMPONENTS: &str = "Components";

/// The result of taking a dual: can either be a success value of `T`, or the
/// index of a facet through the inversion center.
type DualResult<T> = Result<T, usize>;

/// The trait for methods common to all polytopes.
pub trait Polytope<T: NameType>: Sized + Clone {
    /// The [rank](https://polytope.miraheze.org/wiki/Rank) of the polytope.
    fn rank(&self) -> Rank;

    /// The name of the polytope in its language-independent representation.
    fn name(&self) -> &Name<T>;

    /// A mutable reference to the name of the polytope.
    fn name_mut(&mut self) -> &mut Name<T>;

    /// Gets the wiki link to the polytope, based on its English name.
    fn wiki_link(&self) -> String {
        format!(
            "{}{}",
            crate::WIKI_LINK,
            lang::En::parse(self.name(), Default::default()).replace(" ", "_")
        )
    }

    /// Used as a chaining operator to set the name of a polytope.
    fn with_name(mut self, name: Name<T>) -> Self {
        *self.name_mut() = name;
        self
    }

    /// The number of elements of a given rank.
    fn el_count(&self, rank: Rank) -> usize;

    /// The element counts of the polytope.
    fn el_counts(&self) -> RankVec<usize>;

    /// The number of vertices on the polytope.
    fn vertex_count(&self) -> usize {
        self.el_count(Rank::new(0))
    }

    /// The number of facets on the polytope.
    fn facet_count(&self) -> usize {
        self.rank()
            .try_sub(Rank::new(1))
            .map(|r| self.el_count(r))
            .unwrap_or(0)
    }

    /// Returns an instance of the
    /// [nullitope](https://polytope.miraheze.org/wiki/Nullitope), the unique
    /// polytope of rank &minus;1.
    fn nullitope() -> Self;

    /// Returns an instance of the
    /// [point](https://polytope.miraheze.org/wiki/Point), the unique polytope
    /// of rank 0.
    fn point() -> Self;

    /// Returns an instance of the
    /// [dyad](https://polytope.miraheze.org/wiki/Dyad), the unique polytope of
    /// rank 1.
    fn dyad() -> Self;

    /// Returns an instance of a [polygon](https://polytope.miraheze.org/wiki/Polygon)
    /// with a given number of sides.
    fn polygon(n: usize) -> Self;

    /// Returns the dual of a polytope. Never fails for an abstract polytope. In
    /// case of failing on a concrete polytope, returns the index of a facet
    /// through the inversion center.
    fn try_dual(&self) -> DualResult<Self>;

    /// Calls [`try_dual`] and unwraps the result.
    fn dual(&self) -> Self {
        self.try_dual().unwrap()
    }

    /// Builds the dual of a polytope in place. Never fails for an abstract
    /// polytope. In case of failing on a concrete polytope, returns the index
    /// of a facet through the inversion center and does nothing.
    fn try_dual_mut(&mut self) -> DualResult<()>;

    /// Calls [`try_dual_mut`] and unwraps the result.
    fn dual_mut(&mut self) {
        self.try_dual_mut().unwrap();
    }

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks.
    fn append(&mut self, p: Self);

    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, rank: Rank, idx: usize) -> Option<Self>;

    /// Gets the element figure with a given rank and index as a polytope.
    fn element_fig(&self, rank: Rank, idx: usize) -> DualResult<Option<Self>> {
        Ok(
            if let Some(mut element_fig) = self
                .try_dual()?
                .element(self.rank().minus_one() - rank, idx)
            {
                element_fig.try_dual_mut()?;
                Some(element_fig)
            } else {
                None
            },
        )
    }

    /// Gets the section defined by two elements with given ranks and indices as
    /// a polytope, or returns `None` in case no section is defined by these
    /// elements.
    fn section(
        &self,
        rank_lo: Rank,
        idx_lo: usize,
        rank_hi: Rank,
        idx_hi: usize,
    ) -> DualResult<Option<Self>> {
        Ok(if let Some(el) = self.element(rank_hi, idx_hi) {
            el.element_fig(rank_lo, idx_lo)?
        } else {
            None
        })
    }

    /// Gets the facet associated to the element of a given index as a polytope.
    fn facet(&self, idx: usize) -> Option<Self> {
        self.element(self.rank().minus_one(), idx)
    }

    /// Gets the verf associated to the element of a given index as a polytope.
    fn verf(&self, idx: usize) -> Result<Option<Self>, usize> {
        self.element_fig(Rank::new(0), idx)
    }

    /// Builds a compound polytope from a set of components.
    fn compound(components: Vec<Self>) -> Self {
        Self::compound_iter(components.into_iter())
    }

    /// Builds a compound polytope from an iterator over components.
    fn compound_iter<U: Iterator<Item = Self>>(mut components: U) -> Self {
        if let Some(mut p) = components.next() {
            for q in components {
                p.append(q);
            }

            p
        } else {
            Self::nullitope()
        }
    }

    fn petrial_mut(&mut self) -> Result<(), ()>;

    fn petrial(&self) -> Option<Self> {
        let mut clone = self.clone();
        clone.petrial_mut().ok().map(|_| clone)
    }

    fn petrie_polygon(&self) -> Option<Self> {
        self.petrie_polygon_with(self.some_flag()?)
    }

    fn petrie_polygon_with(&self, flag: Flag) -> Option<Self>;

    /// Returns any flag of the polytope.
    fn some_flag(&self) -> Option<Flag>;

    /// Returns an iterator over all "flag events" of a polytope. For more info,
    /// see [`FlagIter`].
    fn flag_events(&self) -> FlagIter;

    /// Returns an iterator over all flags of a polytope.
    fn flags(&self) -> std::iter::FilterMap<FlagIter, &dyn Fn(FlagEvent) -> Option<Flag>> {
        // Debug assert that all elements are sorted!

        self.flag_events().filter_map(&|flag_event| {
            if let FlagEvent::Flag(flag) = flag_event {
                Some(flag)
            } else {
                None
            }
        })
    }

    fn flag_omnitruncate(&self) -> Self;

    /// Builds a [duopyramid](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// from two polytopes.
    fn duopyramid(p: &Self, q: &Self) -> Self;

    /// Builds a [duoprism](https://polytope.miraheze.org/wiki/Prism_product)
    /// from two polytopes.
    fn duoprism(p: &Self, q: &Self) -> Self;

    /// Builds a [duotegum](https://polytope.miraheze.org/wiki/Tegum_product)
    /// from two polytopes.
    fn duotegum(p: &Self, q: &Self) -> Self;

    /// Builds a [duocomb](https://polytope.miraheze.org/wiki/Honeycomb_product)
    /// from two polytopes.
    fn duocomb(p: &Self, q: &Self) -> Self;

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope.
    fn ditope(&self) -> Self;

    /// Builds a [ditope](https://polytope.miraheze.org/wiki/Ditope) of a given
    /// polytope in place.
    fn ditope_mut(&mut self);

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope.
    fn hosotope(&self) -> Self;

    /// Builds a [hosotope](https://polytope.miraheze.org/wiki/hosotope) of a
    /// given polytope in place.
    fn hosotope_mut(&mut self);

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope.
    fn try_antiprism(&self) -> Result<Self, usize>;

    fn antiprism(&self) -> Self {
        self.try_antiprism().unwrap()
    }

    /// Determines whether a given polytope is
    /// [orientable](https://polytope.miraheze.org/wiki/Orientability).
    fn orientable(&self) -> bool;

    /// Builds a [pyramid](https://polytope.miraheze.org/wiki/Pyramid) from a
    /// given base.
    fn pyramid(&self) -> Self {
        Self::duopyramid(self, &Self::point()).with_name(self.name().clone().pyramid())
    }

    /// Builds a [prism](https://polytope.miraheze.org/wiki/Prism) from a
    /// given base.
    fn prism(&self) -> Self {
        Self::duoprism(self, &Self::dyad()).with_name(self.name().clone().prism())
    }

    /// Builds a [tegum](https://polytope.miraheze.org/wiki/Bipyramid) from a
    /// given base.
    fn tegum(&self) -> Self {
        Self::duotegum(self, &Self::dyad()).with_name(self.name().clone().tegum())
    }

    /// Takes the [pyramid product](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// of a set of polytopes.
    fn multipyramid(factors: &[&Self]) -> Self {
        Self::multipyramid_iter(factors.iter().copied())
    }

    /// Takes the [pyramid product](https://polytope.miraheze.org/wiki/Pyramid_product)
    /// of an iterator over polytopes.
    fn multipyramid_iter<'a, U: Iterator<Item = &'a Self>>(factors: U) -> Self
    where
        Self: 'a,
    {
        factors.fold(Self::nullitope(), |p, q| Self::duopyramid(&p, q))
    }

    /// Takes the [prism product](https://polytope.miraheze.org/wiki/Prism_product)
    /// of a set of polytopes.
    fn multiprism(factors: &[&Self]) -> Self {
        Self::multiprism_iter(factors.iter().copied())
    }

    /// Takes the [prism product](https://polytope.miraheze.org/wiki/Prism_product)
    /// of an iterator over polytopes.
    fn multiprism_iter<'a, U: Iterator<Item = &'a Self>>(factors: U) -> Self
    where
        Self: 'a,
    {
        factors.fold(Self::point(), |p, q| Self::duoprism(&p, q))
    }

    /// Takes the [tegum product](https://polytope.miraheze.org/wiki/Tegum_product)
    /// of a set of polytopes.
    fn multitegum(factors: &[&Self]) -> Self {
        Self::multitegum_iter(factors.iter().copied())
    }

    /// Takes the [tegum product](https://polytope.miraheze.org/wiki/Tegum_product)
    /// of an iterator over polytopes.
    fn multitegum_iter<'a, U: Iterator<Item = &'a Self>>(factors: U) -> Self
    where
        Self: 'a,
    {
        factors.fold(Self::point(), |p, q| Self::duotegum(&p, q))
    }

    /// Takes the [comb product](https://polytope.miraheze.org/wiki/Comb_product)
    /// of a set of polytopes.
    fn multicomb(factors: &[&Self]) -> Self {
        Self::multicomb_iter(factors.iter().copied())
    }

    /// Takes the [comb product](https://polytope.miraheze.org/wiki/Comb_product)
    /// of an iterator over polytopes.
    fn multicomb_iter<'a, U: Iterator<Item = &'a Self>>(mut factors: U) -> Self
    where
        Self: 'a,
    {
        if let Some(init) = factors.next().cloned() {
            factors.fold(init, |p, q| Self::duocomb(&p, q))
        }
        // There's no sensible way to take an empty comb product, so we just
        // make it a nullitope for simplicity.
        else {
            Self::nullitope()
        }
    }

    /// Builds a [simplex](https://polytope.miraheze.org/wiki/Simplex) with a
    /// given rank.
    fn simplex(rank: Rank) -> Self {
        if rank == Rank::new(-1) {
            Self::nullitope()
        } else {
            Self::multipyramid(&vec![&Self::point(); rank.0]).with_name(Name::simplex(
                T::DataRegular::new(Regular::Yes {
                    center: vec![0.0; rank.usize()].into(),
                }),
                rank,
            ))
        }
    }

    /// Builds a [hypercube](https://polytope.miraheze.org/wiki/Hypercube) with
    /// a given rank.
    fn hypercube(rank: Rank) -> Self {
        if rank == Rank::new(-1) {
            Self::nullitope()
        } else {
            let rank_u = rank.usize();

            Self::multiprism(&vec![&Self::dyad(); rank_u]).with_name(Name::hyperblock(
                T::DataRegular::new(Regular::Yes {
                    center: vec![0.0; rank_u].into(),
                }),
                rank,
            ))
        }
    }

    /// Builds an [orthoplex](https://polytope.miraheze.org/wiki/Orthoplex) with
    /// a given rank.
    fn orthoplex(rank: Rank) -> Self {
        if rank == Rank::new(-1) {
            Self::nullitope()
        } else {
            let rank_u = rank.usize();

            Self::multitegum(&vec![&Self::dyad(); rank_u]).with_name(Name::orthoplex(
                T::DataRegular::new(Regular::Yes {
                    center: vec![0.0; rank_u].into(),
                }),
                rank,
            ))
        }
    }
}
