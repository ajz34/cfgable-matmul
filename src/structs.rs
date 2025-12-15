use crate::prelude::*;

pub trait MatmulMicroKernelAPI<T, const KC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize>
where
    T: Mul<Output = T> + AddAssign<T> + Clone,
{
    unsafe fn microkernel(
        c: &mut [[FpSimd<T, LANE>; NR_LANE]], // MR x NR, aligned, register
        a: &[[T; MR]],                        // kc x MR (lda), packed-transposed, cache l2 prefetch l1
        b: &[[FpSimd<T, LANE>; NR_LANE]],     // kc x NR, packed, aligned, cache l1
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
    const NB: usize = 0,
> {
    _phantom: core::marker::PhantomData<T>,
}

impl<const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize>
    MatmulMicroKernelAPI<f64, KC, MR, NR_LANE, LANE> for MatmulLoops<f64, MC, KC, NC, MR, NR_LANE, LANE>
{
    #[inline]
    unsafe fn microkernel(
        c: &mut [[FpSimd<f64, LANE>; NR_LANE]], // MR x NR, aligned, register
        a: &[[f64; MR]],                        // kc x MR (lda), packed-transposed, cache l2 prefetch l1
        b: &[[FpSimd<f64, LANE>; NR_LANE]],     // kc x NR, packed, aligned, cache l1
        kc: usize,
    ) {
        core::hint::assert_unchecked(kc <= KC);

        let c: &mut [[FpSimd<f64, LANE>; NR_LANE]] = transmute(c);
        let b: &[[FpSimd<f64, LANE>; NR_LANE]] = transmute(b);
        for p in 0..kc {
            for i in 0..MR {
                let a_ip = FpSimd::splat(*a.get_unchecked(p).get_unchecked(i));
                for j_lane in 0..NR_LANE {
                    let b_pj = *b.get_unchecked(p).get_unchecked(j_lane);

                    c.get_unchecked_mut(i).get_unchecked_mut(j_lane).fma_from(b_pj, a_ip);
                }
            }
        }
    }
}
