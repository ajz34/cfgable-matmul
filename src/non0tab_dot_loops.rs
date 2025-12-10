use crate::prelude::*;
use core::mem::transmute;
use core::ops::*;

pub struct MatmulLoops<const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize>;

pub trait MatmulMicroKernelAPI<T, const KC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize>
where
    T: Mul<Output = T> + AddAssign<T> + Copy,
{
    unsafe fn microkernel(
        c: &mut [[[T; LANE]; NR_LANE]], // MR x NR, aligned, register
        a: &[T],                        // MR x kc (lda), packed, cache l2 prefetch l1
        b: &[[[T; LANE]; NR_LANE]],     // kc x NR, packed, aligned, cache l1
        k: usize,
        lda: usize,
    ) {
        for p in 0..k {
            // perform matrix multiply
            for i in 0..MR {
                let a_ip = *a.get_unchecked(i * lda + p);
                for j in 0..NR_LANE {
                    let c_ij = c.get_unchecked_mut(i).get_unchecked_mut(j);
                    for l in 0..LANE {
                        *c_ij.get_unchecked_mut(l) += a_ip * *b.get_unchecked(p).get_unchecked(j).get_unchecked(l);
                    }
                }
            }
        }
    }

    unsafe fn microkernel_mask(
        c: &mut [[[T; LANE]; NR_LANE]], // MR x NR, aligned, register
        a: &[T],                        // MR x kc (lda), packed, cache l2 prefetch l1
        b: &[[[T; LANE]; NR_LANE]],     // kc x NR, packed, aligned, cache l1
        mask: &[bool],                  // kc
        k: usize,
        lda: usize,
    ) {
        // collect mask indices
        let mut mask_indices: [usize; KC] = [0; KC];
        let mut idx = 0;
        for p in 0..k {
            if *mask.get_unchecked(p) {
                *mask_indices.get_unchecked_mut(idx) = p;
                idx += 1;
            }
        }

        for p in mask_indices {
            // perform matrix multiply
            for i in 0..MR {
                let a_ip = *a.get_unchecked(i * lda + p);
                for j in 0..NR_LANE {
                    let c_ij = c.get_unchecked_mut(i).get_unchecked_mut(j);
                    for l in 0..LANE {
                        *c_ij.get_unchecked_mut(l) += a_ip * *b.get_unchecked(p).get_unchecked(j).get_unchecked(l);
                    }
                }
            }
        }
    }
}

impl<const MC: usize, const KC: usize, const NC: usize, const MR: usize> MatmulMicroKernelAPI<f64, KC, MR, 2, 8>
    for MatmulLoops<MC, KC, NC, MR, 2, 8>
{
    #[inline]
    unsafe fn microkernel(
        c: &mut [[[f64; 8]; 2]], // MR x NR, aligned, register
        a: &[f64],               // MR x kc (lda), packed, cache l2 prefetch l1
        b: &[[[f64; 8]; 2]],     // kc x NR, packed, aligned, cache l1
        k: usize,
        lda: usize,
    ) {
        let c: &mut [[f64simd; 2]] = transmute(c);
        let b: &[[f64simd; 2]] = transmute(b);
        for p in 0..k {
            let b_p0 = *b.get_unchecked(p).get_unchecked(0);
            let b_p1 = *b.get_unchecked(p).get_unchecked(1);
            for i in 0..MR {
                let a_ip = f64simd::splat(*a.get_unchecked(i * lda + p));
                c.get_unchecked_mut(i).get_unchecked_mut(0).fma_from(b_p0, a_ip);
                c.get_unchecked_mut(i).get_unchecked_mut(1).fma_from(b_p1, a_ip);
            }
        }
    }

    #[inline]
    unsafe fn microkernel_mask(
        c: &mut [[[f64; 8]; 2]], // MR x NR, aligned, register
        a: &[f64],               // MR x kc (lda), packed, cache l2 prefetch l1
        b: &[[[f64; 8]; 2]],     // kc x NR, packed, aligned, cache l1
        mask: &[bool],           // kc
        k: usize,
        lda: usize,
    ) {
        // collect mask indices
        let mut mask_indices: [usize; KC] = [0; KC];
        let mut idx = 0;
        for p in 0..k {
            if *mask.get_unchecked(p) {
                *mask_indices.get_unchecked_mut(idx) = p;
                idx += 1;
            }
        }

        let c: &mut [[f64simd; 2]] = transmute(c);
        let b: &[[f64simd; 2]] = transmute(b);
        for p in mask_indices {
            let b_p0 = *b.get_unchecked(p).get_unchecked(0);
            let b_p1 = *b.get_unchecked(p).get_unchecked(1);
            for i in 0..MR {
                let a_ip = f64simd::splat(*a.get_unchecked(i * lda + p));
                c.get_unchecked_mut(i).get_unchecked_mut(0).fma_from(b_p0, a_ip);
                c.get_unchecked_mut(i).get_unchecked_mut(1).fma_from(b_p1, a_ip);
            }
        }
    }
}

pub fn microkernel_anyway(
    c: &mut [[[f64; 8]; 2]], // MR x NR, aligned
    a: &[f64],               // kc x MR (lda), packed-transposed
    b: &[[[f64; 8]; 2]],     // kc x NR (ldb), packed, aligned
    k: usize,
    lda: usize,
) {
    unsafe {
        MatmulLoops::<256, 192, 240, 14, 2, 8>::microkernel(c, a, b, k, lda);
    }
}

pub fn microkernel_mask_anyway(
    c: &mut [[[f64; 8]; 2]], // MR x NR, aligned
    a: &[f64],               // kc x MR (lda), packed-transposed
    b: &[[[f64; 8]; 2]],     // kc x NR (ldb), packed, aligned
    mask: &[bool],           // kc
    k: usize,
    lda: usize,
) {
    unsafe {
        MatmulLoops::<256, 192, 240, 11, 2, 8>::microkernel_mask(c, a, b, mask, k, lda);
    }
}
