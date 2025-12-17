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
}

pub struct MatmulLoops<
    T,
    const MC: usize,
    const KC: usize,
    const NC: usize,
    const MR: usize,
    const NR_LANE: usize,
    const LANE: usize,
    const MB: usize = 0,
> {
    _phantom: core::marker::PhantomData<T>,
}

impl<const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize, const MB: usize>
    MatmulMicroKernelAPI<f64, KC, MR, NR_LANE, LANE> for MatmulLoops<f64, MC, KC, NC, MR, NR_LANE, LANE, MB>
{
    #[inline]
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

        let c: &mut [[TySimd<f64, LANE>; NR_LANE]] = transmute(c);
        let b: &[[TySimd<f64, LANE>; NR_LANE]] = transmute(b);
        for p in 0..kc {
            for i in 0..MR {
                let a_ip = TySimd::splat(a[p][i]);
                for j_lane in 0..NR_LANE {
                    let b_pj = b[p][j_lane];
                    c[i][j_lane].fma_from(a_ip, b_pj);
                }
                core::hint::black_box(());
            }
        }
    }
}
