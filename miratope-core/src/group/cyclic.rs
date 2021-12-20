//! Defines an iterator over a cyclic group.

use super::group_item::GroupItem;

/// An iterator over a cyclic group, i.e. a group with a single generator. This
/// will be much faster than [`GenIter`](super::GenIter) for large groups.
pub struct Cyclic<T> {
    /// The generator for the group.
    gen: T,

    /// The current item in the iterator.
    cur: Option<T>,
}

impl<T: Clone> Cyclic<T> {
    /// Initializes a new cyclic group.
    pub fn new(gen: T) -> Self {
        Self {
            cur: Some(gen.clone()),
            gen,
        }
    }
}

impl<T: Clone + GroupItem> Iterator for Cyclic<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur.as_mut()?;
        let res = cur.clone();
        cur.mul_assign(&self.gen);

        if cur.eq(&self.gen) {
            self.cur = None;
        }

        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i2() {
        assert_eq!(Cyclic::new(-1.0).collect::<Vec<_>>(), vec![-1.0, 1.0]);
    }
}
