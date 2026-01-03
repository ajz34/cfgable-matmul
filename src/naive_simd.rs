use core::ops::*;
use duplicate::duplicate_item;
use num::traits::{MulAdd, Num};

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
#[derive(Clone, Debug, Copy)]
#[repr(C)]
pub struct TySimd<T, const LANE: usize>(pub [T; LANE]);

impl<T, const LANE: usize> Index<usize> for TySimd<T, LANE> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const LANE: usize> IndexMut<usize> for TySimd<T, LANE> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, const LANE: usize> TySimd<T, LANE> {
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
}

impl<T: Num + Copy, const LANE: usize> TySimd<T, LANE> {
    /// Returns a SIMD object with all lanes set to zero.
    #[inline(always)]
    pub fn zero() -> Self {
        TySimd([T::zero(); LANE])
    }

    /// Returns a SIMD object with all lanes set to `val`.
    #[inline(always)]
    pub const fn splat(val: T) -> Self {
        TySimd([val; LANE])
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
        TySimd(self.0.map(f))
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
    impl<T: Trait<Output = T> + Clone, const LANE: usize> Trait for TySimd<T, LANE> {
        type Output = Self;
        #[inline(always)]
        fn trait_fn(mut self, rhs: Self) -> Self::Output {
            for i in 0..LANE {
                self.0[i] = T::trait_fn(self.0[i].clone(), rhs.0[i].clone());
            }
            self
        }
    }

    // simd * scalar
    impl<T: Trait<Output = T> + Clone, const LANE: usize> Trait<T> for TySimd<T, LANE> {
        type Output = Self;
        #[inline(always)]
        fn trait_fn(mut self, rhs: T) -> Self::Output {
            for i in 0..LANE {
                self.0[i] = T::trait_fn(self.0[i].clone(), rhs.clone());
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
impl<T: Trait + Clone, const LANE: usize> Trait for TySimd<T, LANE> {
    #[inline(always)]
    fn trait_fn(&mut self, rhs: Self) {
        for i in 0..LANE {
            T::trait_fn(&mut self.0[i], rhs.0[i].clone());
        }
    }
}

impl<T: Neg<Output = T> + Clone, const LANE: usize> Neg for TySimd<T, LANE> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut result = self;
        for i in 0..LANE {
            result.0[i] = -result.0[i].clone();
        }
        result
    }
}

impl<T, const LANE: usize> TySimd<T, LANE>
where
    T: MulAdd<Output = T> + Clone,
{
    /// Performs fused multiply-add: `self * b + c`.
    ///
    /// This is similar function to [`MulAdd::mul_add`].
    #[inline(always)]
    pub fn mul_add(mut self, b: TySimd<T, LANE>, c: TySimd<T, LANE>) -> TySimd<T, LANE> {
        for i in 0..LANE {
            self.0[i] = self.0[i].clone().mul_add(b.0[i].clone(), c.0[i].clone());
        }
        self
    }

    /// Performs fused multiply-add: `self = self + b * c`.
    ///
    /// Note that the order of multiplication and addition is different from
    /// [`TySimd::mul_add`].
    #[inline(always)]
    pub fn fma_from(&mut self, b: TySimd<T, LANE>, c: TySimd<T, LANE>) {
        for i in 0..LANE {
            self.0[i] = b.0[i].clone().mul_add(c.0[i].clone(), self.0[i].clone());
        }
    }
}

impl<T: Clone, const LANE: usize> TySimd<T, LANE> {
    #[inline(always)]
    #[allow(clippy::uninit_assumed_init)]
    pub unsafe fn loadu_ptr(src: *const T) -> Self {
        let mut arr: [T; LANE] = core::mem::MaybeUninit::uninit().assume_init();
        for i in 0..LANE {
            arr[i] = (*src.add(i)).clone();
        }
        Self(arr)
    }

    #[inline(always)]
    pub unsafe fn storeu_ptr(&self, dst: *mut T) {
        for i in 0..LANE {
            *dst.add(i) = self.0.get_unchecked(i).clone();
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
