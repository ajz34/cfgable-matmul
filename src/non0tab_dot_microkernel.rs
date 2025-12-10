#![allow(clippy::missing_safety_doc)]
#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]
#![allow(non_snake_case)]

use crate::prelude::*;

/// Microkernel for matrix multiplication with mask.
///
/// This function follows row-major convention.
///
/// For matrix multiplication $C_{ij} = \sum_p A_{ip} B_{pj}$, the dimensions
/// are
///
/// - $C_{ij}$: `[MR, NR]`.
/// - $A_{ip}$: `[MR, KC]`.
/// - $B_{pj}$: `[KC, NR]`.
///
/// Please note that `NR` in this function is set to be 16, a proper value for
/// AVX-512.
///
/// As reference, BLIS uses the following setting for SKX micro-architecture:
/// - MR = 14, NR = 16, KC = 256, MC = 240, NC = 3752
#[allow(clippy::missing_safety_doc)]
pub unsafe fn microkernel_f64_MRx16_KC_maskKC<const MR: usize, const KC: usize>(
    c: &mut [[f64simd; 2]; MR],
    a: &[f64],
    b: &[f64],
    k: usize,
    lda: usize,
    ldb: usize,
    mask: &[bool],
) {
    // generate mask of index `p`
    let mut p_list = [0; KC];
    let mut nps = 0;
    mask.split_at(KC.min(k)).0.iter().enumerate().for_each(|(p, &m)| {
        if m {
            p_list[nps] = p;
            nps += 1;
        }
    });

    unsafe { core::hint::assert_unchecked(nps <= KC) }

    // microkernel MRx16
    unsafe {
        for ip in 0..nps {
            let p = *p_list.get_unchecked(ip);
            let b0 = f64simd::loadu_ptr(b.as_ptr().add(p * ldb));
            let b1 = f64simd::loadu_ptr(b.as_ptr().add(p * ldb + SIMDD));

            for r in 0..MR {
                let a_r = f64simd::splat(*a.get_unchecked(r * lda + p));
                c.get_unchecked_mut(r).get_unchecked_mut(0).fma_from(a_r, b0);
                c.get_unchecked_mut(r).get_unchecked_mut(1).fma_from(a_r, b1);
            }
        }
    }
}

pub unsafe fn microkernel_f64_8x16_192_maskKC(
    c: &mut [[f64simd; 2]; 8],
    a: &[f64],
    b: &[f64],
    k: usize,
    lda: usize,
    ldb: usize,
    mask: &[bool],
) {
    unsafe { microkernel_f64_MRx16_KC_maskKC::<8, 192>(c, a, b, k, lda, ldb, mask) }
}

pub unsafe fn microkernel_f64_9x16_192_maskKC(
    c: &mut [[f64simd; 2]; 9],
    a: &[f64],
    b: &[f64],
    k: usize,
    lda: usize,
    ldb: usize,
    mask: &[bool],
) {
    unsafe { microkernel_f64_MRx16_KC_maskKC::<9, 192>(c, a, b, k, lda, ldb, mask) }
}

pub unsafe fn microkernel_f64_10x16_192_maskKC(
    c: &mut [[f64simd; 2]; 10],
    a: &[f64],
    b: &[f64],
    k: usize,
    lda: usize,
    ldb: usize,
    mask: &[bool],
) {
    unsafe { microkernel_f64_MRx16_KC_maskKC::<10, 192>(c, a, b, k, lda, ldb, mask) }
}

/// Microkernel for matrix multiplication.
///
/// This function follows row-major convention.
///
/// For matrix multiplication $C_{ij} = \sum_p A_{ip} B_{pj}$, the dimensions
/// are
///
/// - $C_{ij}$: `[MR, NR]`.
/// - $A_{ip}$: `[MR, KC]`.
/// - $B_{pj}$: `[KC, NR]`.
///
/// Please note that `NR` in this function is set to be 16, a proper value for
/// AVX-512.
///
/// As reference, BLIS uses the following setting for SKX micro-architecture:
/// - MR = 14, NR = 16, KC = 256, MC = 240, NC = 3752
#[allow(clippy::missing_safety_doc)]
pub unsafe fn microkernel_f64_MRx16_KC<const MR: usize, const KC: usize>(
    c: &mut [[f64simd; 2]; MR],
    a: &[f64],
    b: &[f64],
    k: usize,
    lda: usize,
    ldb: usize,
) {
    unsafe {
        for p in 0..k {
            let b0 = f64simd::loadu_ptr(b.as_ptr().add(p * ldb));
            let b1 = f64simd::loadu_ptr(b.as_ptr().add(p * ldb + SIMDD));

            for r in 0..MR {
                let a_r = f64simd::splat(*a.get_unchecked(r * lda + p));
                c.get_unchecked_mut(r).get_unchecked_mut(0).fma_from(a_r, b0);
                c.get_unchecked_mut(r).get_unchecked_mut(1).fma_from(a_r, b1);
            }
        }
    }
}

pub unsafe fn microkernel_f64_8x16_192(c: &mut [[f64simd; 2]; 8], a: &[f64], b: &[f64], k: usize, lda: usize, ldb: usize) {
    unsafe { microkernel_f64_MRx16_KC::<8, 192>(c, a, b, k, lda, ldb) }
}

pub unsafe fn microkernel_f64_10x16_192(c: &mut [[f64simd; 2]; 10], a: &[f64], b: &[f64], k: usize, lda: usize, ldb: usize) {
    unsafe { microkernel_f64_MRx16_KC::<10, 192>(c, a, b, k, lda, ldb) }
}

pub unsafe fn microkernel_f64_12x16_192(c: &mut [[f64simd; 2]; 12], a: &[f64], b: &[f64], k: usize, lda: usize, ldb: usize) {
    unsafe { microkernel_f64_MRx16_KC::<12, 192>(c, a, b, k, lda, ldb) }
}

pub unsafe fn microkernel_f64_14x16_192(c: &mut [[f64simd; 2]; 14], a: &[f64], b: &[f64], k: usize, lda: usize, ldb: usize) {
    unsafe { microkernel_f64_MRx16_KC::<14, 192>(c, a, b, k, lda, ldb) }
}
