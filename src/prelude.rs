pub(crate) use crate::naive_simd::*;

#[allow(clippy::uninit_vec)]
#[inline]
pub unsafe fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let mut v: Vec<T> = vec![];
    v.try_reserve_exact(size).unwrap();
    unsafe { v.set_len(size) };
    v
}
