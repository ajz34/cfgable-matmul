use core::ops::*;
use duplicate::duplicate_item;
use num::traits::{MulAdd, Num, NumAssignOps};

/* #region simple f64x8 */

// TODO: use fearless_simd or pulp, they are more heavy but more complete SIMD
// abstraction crates

/// (dev) GTO internal SIMD type.
///
/// In most cases, we use f64x8 as the SIMD type, which corresponds to AVX-512.
///
/// This type implements basic arithmetic operations and some utility functions,
/// such as arithmetics, fmadd, map, splat, etc.
///
/// Please note this is only a simple implementation. This does not cover any
/// target of protable SIMD propose ([#86656](https://github.com/rust-lang/rust/issues/86656)).
///
/// To fully utilize SIMD capabilities, you need to compile by `RUSTFLAGS="-C
/// target-cpu=native"` or similar flags.
#[repr(align(32))]
#[derive(Clone, Debug, Copy)]
pub struct FpSimd<T: Copy, const LANE: usize>(pub [T; LANE]);

#[allow(non_camel_case_types)]
pub type f64simd = FpSimd<f64, 8>;

impl<T: Copy, const LANE: usize> Index<usize> for FpSimd<T, LANE> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Copy, const LANE: usize> IndexMut<usize> for FpSimd<T, LANE> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: Num + Copy, const LANE: usize> FpSimd<T, LANE> {
    /// Returns a SIMD object with all lanes set to zero.
    #[inline(always)]
    pub fn zero() -> Self {
        FpSimd([T::zero(); LANE])
    }

    /// Returns an uninitialized SIMD object.
    ///
    /// # Safety
    ///
    /// This function returns an uninitialized object. The caller must ensure
    /// that the returned value is properly initialized before use.
    #[inline(always)]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub unsafe fn uninit() -> Self {
        core::mem::MaybeUninit::uninit().assume_init()
    }

    /// Returns a SIMD object with all lanes set to `val`.
    #[inline(always)]
    pub const fn splat(val: T) -> Self {
        FpSimd([val; LANE])
    }

    /// Sets all lanes to `val`.
    #[inline(always)]
    pub fn fill(&mut self, val: T) {
        self.0 = [val; LANE];
    }

    /// Applies function `f` to each lane and returns a new SIMD object.
    #[inline]
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        FpSimd(self.0.map(f))
    }
}

#[duplicate_item(
    Trait trait_fn;
    [Add] [add];
    [Sub] [sub];
    [Mul] [mul];
    [Div] [div];
)]
mod impl_traits {
    use super::*;

    // simd * simd
    impl<T: Num + Copy, const LANE: usize> Trait for FpSimd<T, LANE> {
        type Output = Self;
        #[inline(always)]
        fn trait_fn(mut self, rhs: Self) -> Self::Output {
            for i in 0..LANE {
                self.0[i] = T::trait_fn(self.0[i], rhs.0[i]);
            }
            self
        }
    }

    // simd * scalar
    impl<T: Num + Copy, const LANE: usize> Trait<T> for FpSimd<T, LANE> {
        type Output = Self;
        #[inline(always)]
        fn trait_fn(mut self, rhs: T) -> Self::Output {
            for i in 0..LANE {
                self.0[i] = T::trait_fn(self.0[i], rhs);
            }
            self
        }
    }
}

#[duplicate_item(
    Trait trait_fn;
    [AddAssign] [add_assign];
    [SubAssign] [sub_assign];
    [MulAssign] [mul_assign];
    [DivAssign] [div_assign];
)]
impl<T: NumAssignOps + Copy, const LANE: usize> Trait for FpSimd<T, LANE> {
    #[inline(always)]
    fn trait_fn(&mut self, rhs: Self) {
        for i in 0..LANE {
            T::trait_fn(&mut self.0[i], rhs.0[i]);
        }
    }
}

impl<T: Neg<Output = T> + Copy, const LANE: usize> Neg for FpSimd<T, LANE> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut result = self;
        for i in 0..LANE {
            result.0[i] = -result.0[i];
        }
        result
    }
}

impl<T, const LANE: usize> FpSimd<T, LANE>
where
    T: MulAdd<Output = T> + Copy,
{
    /// Performs fused multiply-add: `self * b + c`.
    ///
    /// This is similar function to [`MulAdd::mul_add`].
    #[inline(always)]
    pub fn mul_add(mut self, b: FpSimd<T, LANE>, c: FpSimd<T, LANE>) -> FpSimd<T, LANE> {
        for i in 0..LANE {
            self.0[i] = self.0[i].mul_add(b.0[i], c.0[i]);
        }
        self
    }

    /// Performs fused multiply-add: `self = self + b * c`.
    ///
    /// Note that the order of multiplication and addition is different from
    /// [`FpSimd::mul_add`].
    #[inline(always)]
    pub fn fma_from(&mut self, b: FpSimd<T, LANE>, c: FpSimd<T, LANE>) {
        for i in 0..LANE {
            self.0[i] = b.0[i].mul_add(c.0[i], self.0[i]);
        }
    }
}

impl<T: Copy, const LANE: usize> FpSimd<T, LANE> {
    #[inline(always)]
    #[allow(clippy::uninit_assumed_init)]
    pub unsafe fn loadu_ptr(src: *const T) -> Self {
        let mut arr: [T; LANE] = core::mem::MaybeUninit::uninit().assume_init();
        for i in 0..LANE {
            arr[i] = *src.add(i);
        }
        Self(arr)
    }

    #[inline(always)]
    pub unsafe fn storeu_ptr(&self, dst: *mut T) {
        for i in 0..LANE {
            *dst.add(i) = *self.0.get_unchecked(i);
        }
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        self.0.get_unchecked(index)
    }

    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.0.get_unchecked_mut(index)
    }
}

/* #endregion */
