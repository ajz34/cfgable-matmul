pub(crate) use libcint::gto::prelude_dev::*;

pub trait F64SimdExtend {
    #[allow(clippy::missing_safety_doc)]
    unsafe fn loadu_ptr(src: *const f64) -> Self;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn storeu_ptr(&self, dst: *mut f64);
}

impl F64SimdExtend for f64simd {
    #[inline(always)]
    unsafe fn loadu_ptr(src: *const f64) -> Self {
        unsafe { FpSimd([*src.add(0), *src.add(1), *src.add(2), *src.add(3), *src.add(4), *src.add(5), *src.add(6), *src.add(7)]) }
    }

    #[inline(always)]
    unsafe fn storeu_ptr(&self, dst: *mut f64) {
        unsafe {
            *dst.add(0) = self[0];
            *dst.add(1) = self[1];
            *dst.add(2) = self[2];
            *dst.add(3) = self[3];
            *dst.add(4) = self[4];
            *dst.add(5) = self[5];
            *dst.add(6) = self[6];
            *dst.add(7) = self[7];
        }
    }
}
