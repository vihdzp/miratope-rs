//! Contains the definitions of the different traits and structs for
//! [polytopes](https://polytope.miraheze.org/wiki/Polytope), as well as some
//! basic methods to operate on them.

pub mod r#abstract;
pub mod concrete;

use std::iter;

use self::r#abstract::{
    elements::{Element, ElementRef, Section},
    flag::{Flag, FlagEvent, FlagIter, OrientedFlag, OrientedFlagIter},
    rank::{Rank, RankVec},
    Abstract,
};
use crate::{
    geometry::Point,
    lang::{
        self,
        name::{Name, NameData, NameType, Regular},
        Language,
    },
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
    /// Returns a reference to the underlying abstract polytope.
    fn abs(&self) -> &Abstract;

    /// Returns a mutable reference to the underlying abstract polytope.
    fn abs_mut(&mut self) -> &mut Abstract;

    /// The [rank](https://polytope.miraheze.org/wiki/Rank) of the polytope.
    fn rank(&self) -> Rank {
        self.abs().ranks.rank()
    }

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

    /// Returns the number of elements of a given rank.
    fn el_count(&self, rank: Rank) -> usize {
        self.abs()
            .ranks
            .get(rank)
            .map(|elements| elements.len())
            .unwrap_or(0)
    }

    /// Returns the element counts of the polytope.
    fn el_counts(&self) -> RankVec<usize> {
        let abs = self.abs();
        let mut counts = RankVec::with_capacity(abs.rank());

        for r in Rank::range_inclusive_iter(Rank::new(-1), abs.rank()) {
            counts.push(abs[r].len())
        }

        counts
    }

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

    /// Calls [`Self::try_dual`] and unwraps the result.
    fn dual(&self) -> Self {
        self.try_dual().unwrap()
    }

    /// Builds the dual of a polytope in place. Never fails for an abstract
    /// polytope. In case of failing on a concrete polytope, returns the index
    /// of a facet through the inversion center and does nothing.
    fn try_dual_mut(&mut self) -> DualResult<()>;

    /// Calls [`Self::try_dual_mut`] and unwraps the result.
    fn dual_mut(&mut self) {
        self.try_dual_mut().unwrap();
    }

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks. **Updates neither the name nor
    /// the min/max elements.**
    ///
    /// # Todo
    /// We should make this method take only the `ranks`, so that we can use the
    /// names from the previous polytopes.
    fn _append(&mut self, p: Self);

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks.
    fn append(&mut self, p: Self);

    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, el: &ElementRef) -> Option<Self>;

    /// Gets the element figure with a given rank and index as a polytope.
    fn element_fig(&self, el: &ElementRef) -> DualResult<Option<Self>> {
        if let Some(rank) = (self.rank() - el.rank).try_minus_one() {
            Ok(
                if let Some(mut element_fig) =
                    self.try_dual()?.element(&ElementRef::new(rank, el.idx))
                {
                    element_fig.try_dual_mut()?;
                    Some(element_fig)
                } else {
                    None
                },
            )
        } else {
            // TODO: this isn't one of the usual inversion through a facet
            // errors. Fix this.
            Err(123456789)
        }
    }

    /// Gets the section defined by two elements with given ranks and indices as
    /// a polytope, or returns `None` in case no section is defined by these
    /// elements.
    fn get_section(&self, section: Section) -> DualResult<Option<Self>> {
        Ok(if let Some(el) = self.element(&section.hi) {
            el.element_fig(&section.lo)?
        } else {
            None
        })
    }

    /// Gets the facet associated to the element of a given index as a polytope.
    fn facet(&self, idx: usize) -> Option<Self> {
        self.element(&ElementRef::new(self.rank().try_minus_one()?, idx))
    }

    /// Gets the verf associated to the element of a given index as a polytope.
    fn verf(&self, idx: usize) -> Result<Option<Self>, usize> {
        self.element_fig(&ElementRef::new(Rank::new(0), idx))
    }

    /// Builds a compound polytope from a set of components.
    fn compound(components: Vec<Self>) -> Self {
        Self::compound_iter(components.into_iter())
    }

    /// Builds a compound polytope from an iterator over components.
    fn compound_iter<U: Iterator<Item = Self>>(mut components: U) -> Self {
        if let Some(mut p) = components.next() {
            for q in components {
                p._append(q);
            }

            // Updates the minimal and maximal elements of the compound.
            *p.abs_mut().min_mut() = Element::min(p.vertex_count());
            *p.abs_mut().max_mut() = Element::max(p.facet_count());

            // TODO: UPDATE NAME.
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
        self.petrie_polygon_with(self.first_flag()?)
    }

    fn petrie_polygon_with(&self, flag: Flag) -> Option<Self>;

    /// Returns the first [`Flag`] of a polytope. This is the flag built when we
    /// start at the maximal element and repeatedly take the first subelement.
    fn first_flag(&self) -> Option<Flag> {
        let rank = self.rank();
        let rank_usize = rank.try_usize()?;

        let mut flag = Flag::with_capacity(rank_usize);
        let mut idx = 0;
        flag.push(0);

        for r in Rank::range_iter(Rank::new(1), rank) {
            idx = self
                .abs()
                .get_element(&ElementRef::new(r.minus_one(), idx))
                .unwrap()
                .sups[0];
            flag.push(idx);
        }

        Some(flag)
    }

    /// Returns the first [`OrientedFlag`] of a polytope. This is the flag built
    /// when we start at the maximal element and repeatedly take the first
    /// subelement.
    fn first_oriented_flag(&self) -> Option<OrientedFlag> {
        Some(OrientedFlag::from(self.first_flag()?))
    }

    /// Returns an iterator over all [`Flag`]s of a polytope.
    fn flags(&self) -> FlagIter {
        FlagIter::new(self.abs())
    }

    /// Returns an iterator over all "flag events" of a polytope. For more info,
    /// see [`FlagIter`].
    fn flag_events(&self) -> OrientedFlagIter {
        OrientedFlagIter::new(self.abs())
    }

    /// Returns an iterator over all [`OrientedFlag`]s of a polytope, assuming
    /// flag-connectedness.
    fn oriented_flags(
        &self,
    ) -> std::iter::FilterMap<OrientedFlagIter, &dyn Fn(FlagEvent) -> Option<OrientedFlag>> {
        self.flag_events().filter_map(&|f: FlagEvent| match f {
            FlagEvent::Flag(f) => Some(f),
            FlagEvent::NonOrientable => None,
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

    /// Attempts to build an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope. If it fails, it returns the index of a facet
    /// through the inversion center.
    fn try_antiprism(&self) -> DualResult<Self>;

    /// Builds an [antiprism](https://polytope.miraheze.org/wiki/Antiprism)
    /// based on a given polytope.
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
    /// of an iterator over polytopes.
    fn multipyramid<'a, U: Iterator<Item = &'a Self>>(mut factors: U) -> Self
    where
        Self: 'a,
    {
        if let Some(init) = factors.next().cloned() {
            factors.fold(init, |p, q| Self::duopyramid(&p, q))
        } else {
            Self::nullitope()
        }
    }

    /// Takes the [prism product](https://polytope.miraheze.org/wiki/Prism_product)
    /// of an iterator over polytopes.
    fn multiprism<'a, U: Iterator<Item = &'a Self>>(mut factors: U) -> Self
    where
        Self: 'a,
    {
        if let Some(init) = factors.next().cloned() {
            factors.fold(init, |p, q| Self::duoprism(&p, q))
        } else {
            Self::point()
        }
    }

    /// Takes the [tegum product](https://polytope.miraheze.org/wiki/Tegum_product)
    /// of an iterator over polytopes.
    fn multitegum<'a, U: Iterator<Item = &'a Self>>(mut factors: U) -> Self
    where
        Self: 'a,
    {
        if let Some(init) = factors.next().cloned() {
            factors.fold(init, |p, q| Self::duotegum(&p, q))
        } else {
            Self::point()
        }
    }

    /// Takes the [comb product](https://polytope.miraheze.org/wiki/Comb_product)
    /// of an iterator over polytopes.
    fn multicomb<'a, U: Iterator<Item = &'a Self>>(mut factors: U) -> Self
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
            Self::multipyramid(iter::repeat(&Self::point()).take(rank.plus_one_usize())).with_name(
                Name::simplex(
                    T::DataRegular::new(Regular::Yes {
                        center: Point::zeros(rank.usize()),
                    }),
                    rank,
                ),
            )
        }
    }

    /// Builds a [hypercube](https://polytope.miraheze.org/wiki/Hypercube) with
    /// a given rank.
    fn hypercube(rank: Rank) -> Self {
        if rank == Rank::new(-1) {
            Self::nullitope()
        } else {
            let rank_u = rank.usize();

            Self::multiprism(iter::repeat(&Self::dyad()).take(rank_u)).with_name(Name::hyperblock(
                T::DataRegular::new(Regular::Yes {
                    center: Point::zeros(rank_u),
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

            Self::multitegum(iter::repeat(&Self::dyad()).take(rank_u)).with_name(Name::orthoplex(
                T::DataRegular::new(Regular::Yes {
                    center: Point::zeros(rank_u),
                }),
                rank,
            ))
        }
    }
}
