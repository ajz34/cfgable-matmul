pub(crate) use libcint::gto::prelude_dev::*;

pub trait F64SimdExtend<T> {
    #[allow(clippy::missing_safety_doc)]
    unsafe fn loadu_ptr(src: *const T) -> Self;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn storeu_ptr(&self, dst: *mut T);

    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked(&self, index: usize) -> &T;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T;
}

impl<T: Copy, const LANE: usize> F64SimdExtend<T> for FpSimd<T, LANE> {
    #[inline(always)]
    #[allow(clippy::uninit_assumed_init)]
    unsafe fn loadu_ptr(src: *const T) -> Self {
        let mut arr: [T; LANE] = core::mem::MaybeUninit::uninit().assume_init();
        for i in 0..LANE {
            arr[i] = *src.add(i);
        }
        Self(arr)
    }

    #[inline(always)]
    unsafe fn storeu_ptr(&self, dst: *mut T) {
        for i in 0..LANE {
            *dst.add(i) = *self.0.get_unchecked(i);
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        self.0.get_unchecked(index)
    }

    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.0.get_unchecked_mut(index)
    }
}

/* #endregion */
