pub mod construct;
pub mod operations;
pub mod product;

pub use construct::*;
pub use operations::*;
pub use product::*;

use super::Polytope;

/// Used to test a particular polytope.
/// We assume that the polytope is equilateral and has no hemi facets.
fn test_el_nums(p: &Polytope, mut el_nums: Vec<usize>) {
    // Checks that element counts match up.
    assert_eq!(p.el_nums(), el_nums, "Element counts don't match up.");

    // Checks that the dual element counts match up as well.
    let len = el_nums.len();
    let p = p.dual();
    el_nums[0..len - 1].reverse();
    assert_eq!(p.el_nums(), el_nums, "Dual element counts don't match up.");
}

fn test_equilateral(p: &Polytope, len: f64) {
    // Checks that the polytope is equilateral.
    assert!(p.is_equilateral_with_len(len), "Polytope not equilateral.");
}
