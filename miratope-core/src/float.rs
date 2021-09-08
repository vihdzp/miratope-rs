//! Defines a [`Float`] trait, which allows Miratope to be generic over `f32` or
//! `f64`.

/// A trait containing the constants associated to each floating point type.
///
/// This trait is only meant to be implemented for `f32` and `f64`.
pub trait Float:
    'static
    + nalgebra::Scalar
    + nalgebra::RealField
    + ordered_float::Float
    + Default
    + std::fmt::Display
    + std::str::FromStr
    + serde::Serialize
    + Copy
{
    /// A default epsilon value for comparing values close to `1.0`. Used in
    /// general floating point operations that would return zero given infinite
    /// precision.
    // todo: just put it in the methods themselves.
    const EPS: Self;

    /// 0
    const ZERO: Self;

    /// 1
    const ONE: Self;

    /// 2
    const TWO: Self;

    /// 3
    const THREE: Self;

    /// 4
    const FOUR: Self;

    /// 5
    const FIVE: Self;

    /// Archimedes' constant (π)
    const PI: Self;

    /// The full circle constant (τ)
    ///
    /// Equal to 2π.
    const TAU: Self;

    /// sqrt(2)
    const SQRT_2: Self;

    /// sqrt(2) / 2
    const HALF_SQRT_2: Self;

    /// sqrt(3)
    const SQRT_3: Self;

    /// sqrt(5)
    const SQRT_5: Self;

    /// Takes the square root of a float.
    fn fsqrt(self) -> Self {
        <Self as ordered_float::Float>::sqrt(self)
    }

    /// Takes the absolute value of a float.
    fn fabs(self) -> Self {
        <Self as ordered_float::Float>::abs(self)
    }

    /// Takes the sine of a float.
    fn fsin(self) -> Self {
        <Self as ordered_float::Float>::sin(self)
    }

    /// Takes the cosine of a float.
    fn fcos(self) -> Self {
        <Self as ordered_float::Float>::cos(self)
    }

    /// Takes the sine and cosine of a float.
    fn fsin_cos(self) -> (Self, Self) {
        <Self as ordered_float::Float>::sin_cos(self)
    }

    /// Makes a float from a `f64`.
    fn f64(f: f64) -> Self;

    /// Makes a float from a `usize`.
    fn usize(u: usize) -> Self;

    /// Makes a float from a `u32`.
    fn u32(u: u32) -> Self;
}

/// Constants for `f32`.
impl Float for f32 {
    const EPS: f32 = 1e-5;
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const TWO: f32 = 2.0;
    const THREE: f32 = 3.0;
    const FOUR: f32 = 4.0;
    const FIVE: f32 = 5.0;
    const PI: f32 = std::f32::consts::PI;
    const TAU: f32 = std::f32::consts::TAU;
    const SQRT_2: f32 = std::f32::consts::SQRT_2;
    const HALF_SQRT_2: f32 = f32::SQRT_2 / 2.0;
    const SQRT_3: f32 = 1.7320508;
    const SQRT_5: f32 = 2.236068;

    fn f64(f: f64) -> Self {
        f as Self
    }

    fn usize(u: usize) -> Self {
        u as Self
    }

    fn u32(u: u32) -> Self {
        u as Self
    }
}

/// Constants for `f64`.
impl Float for f64 {
    const EPS: f64 = 1e-12;
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const THREE: f64 = 3.0;
    const FOUR: f64 = 4.0;
    const FIVE: f64 = 5.0;
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const SQRT_2: f64 = std::f64::consts::SQRT_2;
    const HALF_SQRT_2: f64 = f64::SQRT_2 / 2.0;
    const SQRT_3: f64 = 1.7320508075688772;
    const SQRT_5: f64 = 2.23606797749979;

    fn f64(f: f64) -> Self {
        f
    }

    fn usize(u: usize) -> Self {
        u as Self
    }

    fn u32(u: u32) -> Self {
        u as Self
    }
}
