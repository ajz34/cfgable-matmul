pub use crate::structs::*;

pub(crate) use crate::naive_simd::*;
pub(crate) use core::mem::{transmute, zeroed};
pub(crate) use core::ops::*;
pub(crate) use rayon::prelude::*;
pub(crate) use std::sync::Mutex;

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
