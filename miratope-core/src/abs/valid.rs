//! Contains the code that verifies whether a set of [`Ranks`] correspond to a
//! valid [`Abstract`](crate::Abstract) polytope.

// TODO: finish these!

use std::collections::HashMap;

use strum_macros::Display;
use vec_like::VecLike;

use super::{Ranked, Ranks, Section};

/// Represents the way in which two elements with one rank of difference are
/// incident to one another. Used as a field in some [`AbstractError`] variants.

#[derive(Clone, Copy, Debug, Display)]
pub enum IncidenceType {
    /// This element is a subelement of another.
    #[strum(serialize = "subelement")]
    Subelement,

    /// This element is a superelement of another.
    #[strum(serialize = "superelement")]
    Superelement,
}

/// Represents an error in an abstract polytope.
#[derive(Clone, Copy, Debug)]
pub enum AbstractError {
    /// The polytope is not bounded, i.e. it doesn't have a single minimal and
    /// maximal element.
    Bounded {
        /// The number of minimal elements.
        min_count: usize,

        /// The number of maximal elements.
        max_count: usize,
    },

    /// The polytope has some invalid index, i.e. some element points to another
    /// non-existent element.
    Index {
        /// The coordinates of the element at fault.
        el: (usize, usize),

        /// Whether the invalid index is a subelement or a superelement.
        incidence_type: IncidenceType,

        /// The invalid index.
        index: usize,
    },

    /// The polytope has a consistency error, i.e. some element is incident to
    /// another but not viceversa.
    Consistency {
        /// The coordinates of the element at fault.
        el: (usize, usize),

        /// Whether the invalid incidence is a subelement or a superelement.
        incidence_type: IncidenceType,

        /// The invalid index.
        index: usize,
    },

    /// The polytope is not ranked, i.e. some element that's not minimal or not
    /// maximal lacks a subelement or superelement, respectively.
    Ranked {
        /// The coordinates of the element at fault.
        el: (usize, usize),

        /// Whether the missing incidences are at subelements or superelements.
        incidence_type: IncidenceType,
    },

    /// The polytope is not dyadic, i.e. some section of height 1 does not have
    /// exactly 4 elements.
    Dyadic {
        /// The coordinates of the section at fault.
        section: Section,

        /// Whether there were more than 4 elements in the section (or less).
        more: bool,
    },

    /// The polytope is not strictly connected, i.e. some section's flags don't
    /// form a connected graph under flag changes.
    Connected(Section),
}

impl std::fmt::Display for AbstractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // The polytope is not bounded.
            AbstractError::Bounded {
                min_count,
                max_count,
            } => write!(
                f,
                "Polytope is unbounded: found {} minimal elements and {} maximal elements",
                min_count, max_count
            ),

            // The polytope has an invalid index.
            AbstractError::Index {
                el,
                incidence_type,
                index,
            } => write!(
                f,
                "Polytope has an invalid index: {:?} has a {} with index {}, but it doesn't exist",
                el, incidence_type, index
            ),

            AbstractError::Consistency {
                el,
                incidence_type,
                index,
            } => write!(
                f,
                "Polytope has an invalid index: {:?} has a {} with index {}, but not viceversa",
                el, incidence_type, index
            ),

            // The polytope is not ranked.
            AbstractError::Ranked { el, incidence_type } => write!(
                f,
                "Polytope is not ranked: {:?} has no {}s",
                el, incidence_type
            ),

            // The polytope is not dyadic.
            AbstractError::Dyadic { section, more } => write!(
                f,
                "Polytope is not dyadic: there are {} than 2 elements between {}",
                if *more { "more" } else { "less" },
                section
            ),

            // The polytope is not strictly connected.
            AbstractError::Connected(section) => write!(
                f,
                "Polytope is not strictly connected: {} is not connected",
                section
            ),
        }
    }
}

impl std::error::Error for AbstractError {}

/// The return value for [`Ranks::is_valid`].
pub type AbstractResult<T> = Result<T, AbstractError>;

impl Ranks {
    /// Checks whether the ranks form a valid polytope, i.e. whether the poset
    /// is bounded, dyadic, and all of its indices refer to valid elements.
    pub fn is_valid(&self) -> AbstractResult<()> {
        self.bounded()?;
        self.check_incidences()?;
        self.is_dyadic()?;

        Ok(())
        // && self.is_strongly_connected()
    }

    /// Determines whether the polytope is bounded, i.e. whether it has a single
    /// minimal element and a single maximal element. A valid polytope should
    /// always return `true`.
    pub fn bounded(&self) -> AbstractResult<()> {
        let min_count = self.min_count();
        let max_count = self.max_count();

        if min_count == 1 && max_count == 1 {
            Ok(())
        } else {
            Err(AbstractError::Bounded {
                min_count,
                max_count,
            })
        }
    }

    /// Checks whether subelements and superelements match up, and whether they
    /// all refer to valid elements in the polytope. If this returns `false`,
    /// then either the polytope hasn't fully built up, or there's something
    /// seriously wrong.
    pub fn check_incidences(&self) -> AbstractResult<()> {
        for (r, elements) in self.iter().enumerate() {
            for (idx, el) in elements.iter().enumerate() {
                // Only the minimal element can have no subelements.
                if r != 0 && el.subs.is_empty() {
                    return Err(AbstractError::Ranked {
                        el: (r, idx),
                        incidence_type: IncidenceType::Subelement,
                    });
                }

                // Iterates over the element's subelements.
                for &sub in &el.subs {
                    // Attempts to get the subelement's superelements.
                    if r >= 1 {
                        if let Some(sub_el) = self.get_element(r - 1, sub) {
                            if sub_el.sups.contains(&idx) {
                                continue;
                            } else {
                                // The element contains a subelement, but not viceversa.
                                return Err(AbstractError::Consistency {
                                    el: (r, idx),
                                    index: sub,
                                    incidence_type: IncidenceType::Subelement,
                                });
                            }
                        }
                    }

                    // We got ourselves an invalid index.
                    return Err(AbstractError::Index {
                        el: (r, idx),
                        index: sub,
                        incidence_type: IncidenceType::Subelement,
                    });
                }

                // Only the maximal element can have no superelements.
                if r != self.rank() && el.sups.is_empty() {
                    return Err(AbstractError::Ranked {
                        el: (r, idx),
                        incidence_type: IncidenceType::Superelement,
                    });
                }

                // Iterates over the element's superelements.
                for &sup in &el.sups {
                    // Attempts to get the subelement's superelements.
                    if let Some(sub_el) = self.get_element(r + 1, sup) {
                        if sub_el.subs.contains(&idx) {
                            continue;
                        } else {
                            // The element contains a superelement, but not viceversa.
                            return Err(AbstractError::Consistency {
                                el: (r, idx),
                                index: sup,
                                incidence_type: IncidenceType::Superelement,
                            });
                        }
                    }

                    // We got ourselves an invalid index.
                    return Err(AbstractError::Index {
                        el: (r, idx),
                        index: sup,
                        incidence_type: IncidenceType::Superelement,
                    });
                }
            }
        }

        Ok(())
    }

    /// Determines whether the polytope satisfies the diamond property. A valid
    /// non-fissary polytope should always return `true`.
    pub fn is_dyadic(&self) -> AbstractResult<()> {
        /// The number of times we've found an element.
        #[derive(PartialEq)]
        enum Count {
            /// We've found an element once.
            Once,

            /// We've found an element twice.
            Twice,
        }

        // For every element, by looking through the subelements of its
        // subelements, we need to find each exactly twice.
        for r in 2..self.rank() {
            for (idx, el) in self[r].iter().enumerate() {
                let mut hash_sub_subs = HashMap::new();

                for &sub in &el.subs {
                    let sub_el = &self[(r - 1, sub)];

                    for &sub_sub in &sub_el.subs {
                        match hash_sub_subs.get(&sub_sub) {
                            // Found for the first time.
                            None => hash_sub_subs.insert(sub_sub, Count::Once),

                            // Found for the second time.
                            Some(Count::Once) => hash_sub_subs.insert(sub_sub, Count::Twice),

                            // Found for the third time?! Abort!
                            Some(Count::Twice) => {
                                return Err(AbstractError::Dyadic {
                                    section: Section::new(r - 2, sub_sub, r, idx),
                                    more: true,
                                });
                            }
                        };
                    }
                }

                // If any subsubelement was found only once, this also
                // violates the diamond property.
                for (sub_sub, count) in hash_sub_subs.into_iter() {
                    if count == Count::Once {
                        return Err(AbstractError::Dyadic {
                            section: Section::new(r - 2, sub_sub, r, idx),
                            more: false,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Determines whether the polytope is connected. A valid non-compound
    /// polytope should always return `true`.
    pub fn is_connected(&self, _section: Section) -> bool {
        todo!()
        /*
        let section = self.get_section(section).unwrap();
        section.flags().count() == section.oriented_flags().count() */
    }

    /// Determines whether the polytope is strongly connected. A valid
    /// non-compound polytope should always return `true`.
    pub fn is_strongly_connected(&self) -> bool {
        todo!()
    }
}
