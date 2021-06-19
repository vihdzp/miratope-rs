#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_panics_doc)]
#![warn(missing_docs)]

//! This is the main dependency of
//! [Miratope](https://github.com/OfficialURL/miratope-rs). It contains all code
//! to build and name [`Abstract`] and [`Concrete`](conc::Concrete) polytopes
//! alike.
//!
//! If you're interested in actually rendering polytopes, you might want to take
//! a look at the [`miratope`](https://crates.io/crates/miratope) crate instead.

pub mod abs;
pub mod conc;
pub mod geometry;
pub mod group;
pub mod lang;

use std::iter;

use abs::{
    elements::{Element, ElementList, ElementRef, SectionRef},
    flag::{Flag, FlagIter, OrientedFlag, OrientedFlagIter},
    rank::{Rank, RankVec},
    Abstract,
};
use geometry::Point;
use lang::{
    name::{Name, NameData, NameType, Regular},
    Language,
};
use vec_like::VecLike;

/// The names for 0-elements, 1-elements, 2-elements, and so on.
const ELEMENT_NAMES: [&str; 11] = [
    "Vertices", "Edges", "Faces", "Cells", "Tera", "Peta", "Exa", "Zetta", "Yotta", "Xenna", "Daka",
];

/// The word "Components".
const COMPONENTS: &str = "Components";

/// The link to the [Polytope Wiki](https://polytope.miraheze.org/wiki/).
pub const WIKI_LINK: &str = "https://polytope.miraheze.org/wiki/";

/// A trait containing the constants associated to each floating point type.
pub trait Consts {
    /// A default epsilon value. Used in general floating point operations that
    /// would return zero given infinite precision.
    const EPS: Self;

    /// Archimedes' constant (π)
    const PI: Self;

    /// The full circle constant (τ)
    ///
    /// Equal to 2π.
    const TAU: Self;

    /// sqrt(2)
    const SQRT_2: Self;
}

/// Constants for `f32`.
impl Consts for f32 {
    const EPS: f32 = 1e-5;
    const PI: f32 = std::f32::consts::PI;
    const TAU: f32 = std::f32::consts::TAU;
    const SQRT_2: f32 = std::f32::consts::SQRT_2;
}

/// Constants for `f64`.
impl Consts for f64 {
    const EPS: f64 = 1e-9;
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const SQRT_2: f64 = std::f64::consts::SQRT_2;
}

/// The floating point type used for all calculations.
pub type Float = f64;

/// A wrapper around [`Float`] to allow for ordering and equality.
pub type FloatOrd = ordered_float::OrderedFloat<Float>;

/// The result of taking a dual: can either be a success value of `T`, or the
/// index of a facet through the inversion center.
pub type DualResult<T> = Result<T, DualError>;

/// Represents an error in a concrete dual, in which a facet with a given index
/// passes through the inversion center.
#[derive(Debug)]
pub struct DualError(usize);

impl std::fmt::Display for DualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "facet {} passes through inversion center", self.0)
    }
}

impl std::error::Error for DualError {}

fn factorial(n: usize) -> u32 {
    const FACTORIALS: [u32; 13] = [
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600,
    ];

    FACTORIALS[n]
}

/// The trait for methods common to all polytopes.
pub trait Polytope<T: NameType>:
    Sized + Clone + AsRef<Abstract> + AsMut<Abstract> + AsRef<Name<T>> + AsMut<Name<T>>
{
    /// The [rank](https://polytope.miraheze.org/wiki/Rank) of the polytope.
    fn rank(&self) -> Rank {
        let abs: &Abstract = self.as_ref();
        abs.ranks.rank()
    }

    /// Gets the wiki link to the polytope, based on its English name.
    fn wiki_link(&self) -> String {
        let name: &Name<_> = self.as_ref();

        format!(
            "{}{}",
            crate::WIKI_LINK,
            lang::En::parse(name, Default::default()).replace(" ", "_")
        )
    }

    /// Used as a chaining operator to set the name of a polytope.
    fn with_name(mut self, name: Name<T>) -> Self {
        *self.as_mut() = name;
        self
    }

    /// Returns the number of elements of a given rank.
    fn el_count(&self, rank: Rank) -> usize {
        let abs: &Abstract = self.as_ref();
        abs.ranks.get(rank).map(ElementList::len).unwrap_or(0)
    }

    /// Returns the element counts of the polytope.
    fn el_counts(&self) -> RankVec<usize> {
        let abs: &Abstract = self.as_ref();
        let mut counts = RankVec::with_rank_capacity(abs.rank());

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
    fn _comp_append(&mut self, p: Self);

    /// "Appends" a polytope into another, creating a compound polytope. Fails
    /// if the polytopes have different ranks.
    fn comp_append(&mut self, p: Self);

    /// Gets the element with a given rank and index as a polytope, if it exists.
    fn element(&self, el: ElementRef) -> Option<Self>;

    /// Gets the element figure with a given rank and index as a polytope.
    fn element_fig(&self, el: ElementRef) -> DualResult<Option<Self>> {
        if let Some(rank) = (self.rank() - el.rank).try_minus_one() {
            if let Some(mut element_fig) = self.try_dual()?.element(ElementRef::new(rank, el.idx)) {
                element_fig.try_dual_mut()?;
                return Ok(Some(element_fig));
            }
        }

        Ok(None)
    }

    /// Gets the section defined by two elements with given ranks and indices as
    /// a polytope, or returns `None` in case no section is defined by these
    /// elements.
    fn section(&self, section: SectionRef) -> DualResult<Option<Self>> {
        Ok(if let Some(el) = self.element(section.hi) {
            el.element_fig(section.lo)?
        } else {
            None
        })
    }

    /// Gets the facet associated to the element of a given index as a polytope.
    fn facet(&self, idx: usize) -> Option<Self> {
        self.element(ElementRef::new(self.rank().try_minus_one()?, idx))
    }

    /// Gets the verf associated to the element of a given index as a polytope.
    fn verf(&self, idx: usize) -> DualResult<Option<Self>> {
        self.element_fig(ElementRef::new(Rank::new(0), idx))
    }

    /// Builds a compound polytope from a set of components.
    fn compound(components: Vec<Self>) -> Self {
        Self::compound_iter(components.into_iter())
    }

    /// Builds a compound polytope from an iterator over components.
    fn compound_iter<U: Iterator<Item = Self>>(mut components: U) -> Self {
        if let Some(mut p) = components.next() {
            for q in components {
                p._comp_append(q);
            }

            // Updates the minimal and maximal elements of the compound.
            let vc = p.vertex_count();
            let fc = p.facet_count();
            let abs: &mut Abstract = p.as_mut();
            *abs.min_mut() = Element::min(vc);
            *abs.max_mut() = Element::max(fc);

            // TODO: UPDATE NAME.
            p
        } else {
            Self::nullitope()
        }
    }

    fn petrial_mut(&mut self) -> bool;

    fn petrial(mut self) -> Option<Self> {
        self.petrial_mut().then(|| self)
    }

    fn petrie_polygon(&mut self) -> Option<Self> {
        self.petrie_polygon_with(self.first_flag()?)
    }

    fn petrie_polygon_with(&mut self, flag: Flag) -> Option<Self>;

    /// Returns the first [`Flag`] of a polytope. This is the flag built when we
    /// start at the maximal element and repeatedly take the first subelement.
    fn first_flag(&self) -> Option<Flag> {
        let rank = self.rank();
        let rank_usize = rank.try_usize()?;

        let mut flag = Flag::with_capacity(rank_usize);
        let mut idx = 0;
        flag.push(0);

        let abs: &Abstract = self.as_ref();
        for r in Rank::range_iter(1, rank) {
            idx = abs
                .get_element(ElementRef::new(r.minus_one(), idx))
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
        Some(self.first_flag()?.into())
    }

    /// Returns an iterator over all [`Flag`]s of a polytope.
    fn flags(&self) -> FlagIter {
        FlagIter::new(self.as_ref())
    }

    /// Returns an iterator over all [`OrientedFlag`]s of a polytope.
    fn flag_events(&self) -> OrientedFlagIter {
        OrientedFlagIter::new(self.as_ref())
    }

    fn omnitruncate(&mut self) -> Self;

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
    fn orientable(&mut self) -> bool;

    /// Builds a [pyramid](https://polytope.miraheze.org/wiki/Pyramid) from a
    /// given base.
    fn pyramid(&self) -> Self {
        let name: &Name<_> = self.as_ref();
        Self::duopyramid(self, &Self::point()).with_name(name.clone().pyramid())
    }

    /// Builds a [prism](https://polytope.miraheze.org/wiki/Prism) from a
    /// given base.
    fn prism(&self) -> Self {
        let name: &Name<_> = self.as_ref();
        Self::duoprism(self, &Self::dyad()).with_name(name.clone().prism())
    }

    /// Builds a [tegum](https://polytope.miraheze.org/wiki/Bipyramid) from a
    /// given base.
    fn tegum(&self) -> Self {
        let name: &Name<_> = self.as_ref();
        Self::duotegum(self, &Self::dyad()).with_name(name.clone().tegum())
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
                        center: Point::zeros(rank.into()),
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
            let rank_u = rank.into();

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
            let rank_u = rank.into();

            Self::multitegum(iter::repeat(&Self::dyad()).take(rank_u)).with_name(Name::orthoplex(
                T::DataRegular::new(Regular::Yes {
                    center: Point::zeros(rank_u),
                }),
                rank,
            ))
        }
    }
}
