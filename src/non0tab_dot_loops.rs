use crate::prelude::*;
use core::mem::{transmute, zeroed};
use core::ops::*;
use rayon::prelude::*;
use std::sync::Mutex;

#[inline]
#[allow(clippy::mut_from_ref)]
unsafe fn cast_mut_slice<T>(slc: &[T]) -> &mut [T] {
    let len = slc.len();
    let ptr = slc.as_ptr() as *mut T;
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

pub struct MatmulLoops<T, const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize> {
    _phantom: core::marker::PhantomData<T>,
}

pub trait MatmulMicroKernelAPI<T, const KC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize>
where
    T: Mul<Output = T> + AddAssign<T> + Copy,
{
    unsafe fn microkernel(
        c: &mut [[FpSimd<T, LANE>; NR_LANE]], // MR x NR, aligned, register
        a: &[[T; MR]],                        // kc x MR (lda), packed-transposed, cache l2 prefetch l1
        b: &[[FpSimd<T, LANE>; NR_LANE]],     // kc x NR, packed, aligned, cache l1
        mr: usize,
        kc: usize,
    );

    unsafe fn microkernel_mask(
        c: &mut [[FpSimd<T, LANE>; NR_LANE]], // MR x NR, aligned, register
        a: &[T],                              // MR x kc (lda), packed, cache l2 prefetch l1
        b: &[[FpSimd<T, LANE>; NR_LANE]],     // kc x NR, packed, aligned, cache l1
        mask: &[bool],                        // kc
        kc: usize,
        lda: usize,
    );
}

impl<const MC: usize, const KC: usize, const NC: usize, const MR: usize> MatmulMicroKernelAPI<f64, KC, MR, 2, 8>
    for MatmulLoops<f64, MC, KC, NC, MR, 2, 8>
{
    #[inline]
    unsafe fn microkernel(
        c: &mut [[f64simd; 2]], // MR x NR, aligned, register
        a: &[[f64; MR]],        // kc x MR (lda), packed-transposed, cache l2 prefetch l1
        b: &[[f64simd; 2]],     // kc x NR, packed, aligned, cache l1
        mr: usize,
        kc: usize,
    ) {
        core::hint::assert_unchecked(kc <= KC);
        core::hint::assert_unchecked(mr <= MR);

        let c: &mut [[f64simd; 2]] = transmute(c);
        let b: &[[f64simd; 2]] = transmute(b);
        if mr == MR {
            for p in 0..kc {
                let b_p0 = *b.get_unchecked(p).get_unchecked(0);
                let b_p1 = *b.get_unchecked(p).get_unchecked(1);
                for i in 0..MR {
                    let a_ip = f64simd::splat(*a.get_unchecked(p).get_unchecked(i));
                    c.get_unchecked_mut(i).get_unchecked_mut(0).fma_from(b_p0, a_ip);
                    c.get_unchecked_mut(i).get_unchecked_mut(1).fma_from(b_p1, a_ip);
                }
            }
        } else {
            for p in 0..kc {
                let b_p0 = *b.get_unchecked(p).get_unchecked(0);
                let b_p1 = *b.get_unchecked(p).get_unchecked(1);
                for i in 0..mr {
                    let a_ip = f64simd::splat(*a.get_unchecked(p).get_unchecked(i));
                    c.get_unchecked_mut(i).get_unchecked_mut(0).fma_from(b_p0, a_ip);
                    c.get_unchecked_mut(i).get_unchecked_mut(1).fma_from(b_p1, a_ip);
                }
            }
        }
    }

    #[inline]
    unsafe fn microkernel_mask(
        c: &mut [[f64simd; 2]], // MR x NR, aligned, register
        a: &[f64],              // MR x kc (lda), packed, cache l2 prefetch l1
        b: &[[f64simd; 2]],     // kc x NR, packed, aligned, cache l1
        mask: &[bool],          // kc
        kc: usize,
        lda: usize,
    ) {
        core::hint::assert_unchecked(kc <= KC);

        // collect mask indices
        let mut mask_indices: [usize; KC] = [0; KC];
        let mut idx = 0;
        for p in 0..kc {
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
                let a_ip = f64simd::splat(*a.as_ptr().add(i * lda + p)); // may overflow, so use as_ptr instead of get_unchecked
                c.get_unchecked_mut(i).get_unchecked_mut(0).fma_from(b_p0, a_ip);
                c.get_unchecked_mut(i).get_unchecked_mut(1).fma_from(b_p1, a_ip);
            }
        }
    }
}

impl<T, const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize>
    MatmulLoops<T, MC, KC, NC, MR, NR_LANE, LANE>
where
    T: Mul<Output = T> + AddAssign<T> + Copy + Default + Sized,
    Self: MatmulMicroKernelAPI<T, KC, MR, NR_LANE, LANE>,
{
    #[inline]
    pub fn matmul_loop_1st(
        c: &mut [T],                      // MC x NR (ldc), DRAM
        a: &[[[T; MR]; KC]],              // KC x MC (lda), packed-transposed, cache l2
        b: &[[FpSimd<T, LANE>; NR_LANE]], // KC x NR, packed, aligned, cache l1
        mc: usize,
        nr: usize,
        kc: usize,
        ldc: usize,
    ) {
        // MR -> MC
        unsafe {
            core::hint::assert_unchecked(mc <= MC);
            core::hint::assert_unchecked(nr <= NR_LANE * LANE);
            core::hint::assert_unchecked(kc <= KC);
        }

        for (i_pack, i) in (0..mc).step_by(MR).enumerate() {
            // initialize C register block
            let mut c_reg: [[FpSimd<T, LANE>; NR_LANE]; MR] = unsafe { zeroed() };

            // call micro-kernel
            let mr = if i + MR <= mc { MR } else { mc - i };
            unsafe { Self::microkernel(&mut c_reg, &a[i_pack], b, mr, kc) }

            // store C register block to C memory block
            for ii in 0..mr {
                for j in 0..nr {
                    let j_loc = j / LANE;
                    let j_lane = j % LANE;
                    c[(i + ii) * ldc + j] += c_reg[ii][j_loc][j_lane];
                }
            }
        }
    }

    #[inline]
    pub fn matmul_loop_2nd(
        c: &mut [T],         // MC x NC (ldc), DRAM
        a: &[[[T; MR]; KC]], // KC x MC (lda), packed-transposed, cache l2
        b: &[T],             // KC x NC, aligned, cache l3 with parallel
        mc: usize,
        nc: usize,
        kc: usize,
        ldb: usize,
        ldc: usize,
    ) {
        // NR -> NC
        unsafe {
            core::hint::assert_unchecked(mc <= MC);
            core::hint::assert_unchecked(nc <= NC);
            core::hint::assert_unchecked(kc <= KC);
        }

        let NR = NR_LANE * LANE;
        let mut buf_b: [[FpSimd<T, LANE>; NR_LANE]; KC] = unsafe { zeroed() }; // KC x NR, packed, aligned, cache l1

        for j in (0..nc).step_by(NR) {
            let nr = if j + NR <= nc { NR } else { nc - j };

            // pack B block
            for p in 0..kc {
                for jj in 0..nr {
                    let j_loc = jj / LANE;
                    let j_lane = jj % LANE;
                    buf_b[p][j_loc][j_lane] = b[p * ldb + j + jj];
                }
            }

            Self::matmul_loop_1st(&mut c[j..], a, &buf_b, mc, nr, kc, ldc);
        }
    }

    pub fn matmul_loop_345_parallel(c: &mut [T], a: &[T], b: &[T], m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize)
    where
        T: Send + Sync,
    {
        // compute number of tasks to split
        let ntask_mc = m.div_ceil(MC);
        let ntask_nc = n.div_ceil(NC);
        let ntask_kc = k.div_ceil(KC);
        let ntask = ntask_mc * ntask_nc * ntask_kc;
        let barrier_c: Vec<Mutex<()>> = (0..(ntask_mc * ntask_nc)).map(|_| Mutex::new(())).collect();

        let nthreads = rayon::current_num_threads();
        let mc_pack = MC.div_ceil(MR);
        let thread_buf_a: Vec<Vec<[[T; MR]; KC]>> = (0..nthreads).map(|_| vec![unsafe { zeroed() }; mc_pack]).collect();

        println!("ntask: {}", ntask);
        (0..ntask).into_par_iter().for_each(|task_id| {
            let task_m = task_id / (ntask_nc * ntask_kc);
            let task_n = (task_id / ntask_kc) % ntask_nc;
            let task_k = task_id % ntask_kc;

            // access buffers
            let thread_id = rayon::current_thread_index().unwrap();
            let buf_a: &mut [[[T; MR]; KC]] = unsafe { cast_mut_slice(&thread_buf_a[thread_id]) };
            let mut buf_c: [[T; NC]; MC] = unsafe { zeroed() };
            let slc_c: &mut [T] = buf_c.as_flattened_mut();

            // get slices
            let a = &a[task_m * MC * lda + task_k * KC..];
            let b = &b[task_k * KC * ldb + task_n * NC..];
            let mc = if (task_m + 1) * MC <= m { MC } else { m - task_m * MC };
            let nc = if (task_n + 1) * NC <= n { NC } else { n - task_n * NC };
            let kc = if (task_k + 1) * KC <= k { KC } else { k - task_k * KC };

            // pack a
            unsafe {
                for p in 0..kc {
                    for i in 0..mc {
                        let i_pack = i / MR;
                        let i_loc = i % MR;
                        *buf_a.get_unchecked_mut(i_pack).get_unchecked_mut(p).get_unchecked_mut(i_loc) = *a.get_unchecked(i * lda + p);
                    }
                }
            }

            unsafe {
                core::hint::assert_unchecked(mc <= MC);
                core::hint::assert_unchecked(nc <= NC);
                core::hint::assert_unchecked(kc <= KC);
            }
            Self::matmul_loop_2nd(slc_c, buf_a, b, mc, nc, kc, ldb, NC);

            // write back to C
            // use barrier to avoid race condition
            let write_lock = barrier_c[task_m * ntask_nc + task_n].lock().unwrap();
            let c = unsafe { cast_mut_slice(&c[task_m * MC * ldc + task_n * NC..]) };
            for i in 0..mc {
                for j in 0..nc {
                    c[i * ldc + j] += buf_c[i][j];
                }
            }
            drop(write_lock)
        });
    }
}

pub fn microkernel_anyway(
    c: &mut [[f64simd; 2]], // MR x NR, aligned
    a: &[[f64; 14]],        // kc x MR (lda), packed-transposed
    b: &[[f64simd; 2]],     // kc x NR (ldb), packed, aligned
    mr: usize,
    kc: usize,
) {
    unsafe {
        MatmulLoops::<f64, 256, 192, 240, 14, 2, 8>::microkernel(c, a, b, mr, kc);
    }
}

pub fn microkernel_mask_anyway(
    c: &mut [[f64simd; 2]], // MR x NR, aligned
    a: &[f64],              // kc x MR (lda), packed-transposed
    b: &[[f64simd; 2]],     // kc x NR (ldb), packed, aligned
    mask: &[bool],          // kc
    k: usize,
    lda: usize,
) {
    unsafe {
        MatmulLoops::<f64, 256, 192, 240, 11, 2, 8>::microkernel_mask(c, a, b, mask, k, lda);
    }
}

pub fn matmul_anyway_full(
    c: &mut [f64], // MC x NC (ldc), DRAM
    a: &[f64],     // MC x KC (lda), packed, cache l2
    b: &[f64],     // KC x NC, aligned, cache l3 with parallel
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    MatmulLoops::<f64, 240, 192, 240, 10, 2, 8>::matmul_loop_345_parallel(c, a, b, m, n, k, lda, ldb, ldc);
}

#[test]
fn test_matmul_anyway_full() {
    let m = 3527;
    let n = 9583;
    let k = 6581;
    let lda = k;
    let ldb = n;
    let ldc = n;
    let a: Vec<f64> = (0..m * lda).map(|x| (x as f64).sin()).collect();
    let b: Vec<f64> = (0..k * ldb).map(|x| (x as f64).cos()).collect();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    let time = std::time::Instant::now();
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc);
    let elapsed = time.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);

    // use rstsr::prelude::*;
    // let device = DeviceOpenBLAS::default();
    // let a_tsr = rt::asarray((&a, [m, k], &device));
    // let b_tsr = rt::asarray((&b, [k, n], &device));
    // let time = std::time::Instant::now();
    // let c_ref = a_tsr % b_tsr;
    // let c_tsr = rt::asarray((&c, [m, n], &device));
    // let elapsed = time.elapsed();
    // let diff = &c_tsr - &c_ref;
    // println!("Elapsed time (ref): {:.3?}", elapsed);
    // println!("Max error: {:.6e}", diff.view().abs().max());

    // println!("c_tsr\n{c_tsr:15.3}");
    // println!("c_ref\n{c_ref:15.3}");
}
