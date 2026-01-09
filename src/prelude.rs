pub use crate::structs::*;

pub(crate) use crate::naive_simd::*;
pub(crate) use core::mem::zeroed;
pub(crate) use core::ops::*;
pub(crate) use num::traits::MulAdd;
pub(crate) use rayon::prelude::*;
pub(crate) use std::sync::Mutex;
pub(crate) use std::sync::atomic::AtomicUsize;
pub(crate) use std::sync::atomic::Ordering::SeqCst;

#[allow(unused_imports)]
pub(crate) use crate::impl_matmul_non0tab::MatmulMicroKernelMaskKForBAPI;

#[inline]
#[allow(clippy::uninit_vec)]
pub(crate) unsafe fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let mut v: Vec<T> = vec![];
    v.try_reserve_exact(size).unwrap();
    unsafe { v.set_len(size) };
    v
}

#[inline]
#[allow(clippy::mut_from_ref)]
pub(crate) unsafe fn cast_mut_slice<T>(slc: &[T]) -> &mut [T] {
    let len = slc.len();
    let ptr = slc.as_ptr() as *mut T;
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}
