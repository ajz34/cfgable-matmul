use crate::prelude::*;

pub trait MatmulMicroKernelAPI<T, const KC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize>
where
    T: Mul<Output = T> + AddAssign<T> + Clone,
{
    unsafe fn microkernel(
        c: &mut [[TySimd<T, LANE>; NR_LANE]], // MR x NR, aligned, register
        a: &[[T; MR]],                        // kc x MR (lda), packed-transposed, cache l2 prefetch l1
        b: &[[TySimd<T, LANE>; NR_LANE]],     // kc x NR, packed, aligned, cache l1
        kc: usize,                            // kc, to avoid non-necessary / uninitialized access
    );

    unsafe fn microkernel_with_c_update(a: &[[T; MR]], b: &[[TySimd<T, LANE>; NR_LANE]], kc: usize, c_mem: &mut [T], ldc: usize) {
        // please note that this function only works when the real mr==MR and nr==NR
        // so accessing c_mem will not go out of bound
        let mut c: [[TySimd<T, LANE>; NR_LANE]; MR] = unsafe { zeroed() };
        Self::microkernel(&mut c, a, b, kc);
        for i in 0..MR {
            for j_lane in 0..NR_LANE {
                for jj in 0..LANE {
                    c_mem[i * ldc + j_lane * LANE + jj] += c[i][j_lane][jj].clone();
                }
            }
        }
    }
}

pub struct MatmulLoops<T, const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize> {
    _phantom: core::marker::PhantomData<T>,
}

impl<const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize>
    MatmulMicroKernelAPI<f64, KC, MR, NR_LANE, LANE> for MatmulLoops<f64, MC, KC, NC, MR, NR_LANE, LANE>
{
    #[inline(always)]
    unsafe fn microkernel(
        c: &mut [[TySimd<f64, LANE>; NR_LANE]], // MR x NR, aligned, register
        a: &[[f64; MR]],                        // kc x MR (lda), packed-transposed, cache l2 prefetch l1
        b: &[[TySimd<f64, LANE>; NR_LANE]],     // kc x NR, packed, aligned, cache l1
        kc: usize,
    ) {
        core::hint::assert_unchecked(kc <= KC);
        core::hint::assert_unchecked(a.len() >= kc);
        core::hint::assert_unchecked(b.len() >= kc);
        core::hint::assert_unchecked(c.len() == MR);

        for p in 0..kc {
            for i in 0..MR {
                let a_ip = TySimd::splat(a[p][i]);
                for j_lane in 0..NR_LANE {
                    let b_pj = b[p][j_lane];
                    // c[i][j_lane].fma_from(a_ip, b_pj);
                    // c[i][j_lane] = a_ip * b_pj + c[i][j_lane];
                    c[i][j_lane] += a_ip * b_pj;
                }
                core::hint::black_box(());
            }
        }
    }

    #[inline(always)]
    unsafe fn microkernel_with_c_update(a: &[[f64; MR]], b: &[[TySimd<f64, LANE>; NR_LANE]], kc: usize, c_mem: &mut [f64], ldc: usize) {
        core::hint::assert_unchecked(kc <= KC);
        core::hint::assert_unchecked(a.len() >= kc);
        core::hint::assert_unchecked(b.len() >= kc);

        if kc == 0 {
            return;
        }

        let mut c: [[TySimd<f64, LANE>; NR_LANE]; MR] = unsafe { zeroed() };
        for p in 0..(kc - 1) {
            for i in 0..MR {
                let a_ip = TySimd::splat(a[p][i]);
                for j_lane in 0..NR_LANE {
                    let b_pj = b[p][j_lane];
                    // c[i][j_lane].fma_from(a_ip, b_pj);
                    // c[i][j_lane] = a_ip * b_pj + c[i][j_lane];
                    c[i][j_lane] += a_ip * b_pj;
                }
                core::hint::black_box(());
            }
        }

        for i in 0..MR {
            let a_ip = TySimd::splat(a[kc - 1][i]);
            for j_lane in 0..NR_LANE {
                let mut c_reg = TySimd::loadu_ptr(c_mem.as_ptr().add(i * ldc + j_lane * LANE));
                let b_pj = b[kc - 1][j_lane];
                c_reg += a_ip * b_pj + c[i][j_lane];
                c_reg.storeu_ptr(c_mem.as_mut_ptr().add(i * ldc + j_lane * LANE));
            }
            core::hint::black_box(());
        }
    }
}
