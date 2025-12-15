use crate::prelude::*;

impl<
    T,
    const MC: usize,
    const KC: usize,
    const NC: usize,
    const MR: usize,
    const NR_LANE: usize,
    const LANE: usize,
    const MB: usize,
    const NB: usize,
> MatmulLoops<T, MC, KC, NC, MR, NR_LANE, LANE, MB, NB>
where
    T: Mul<Output = T> + AddAssign<T> + Clone,
    Self: MatmulMicroKernelAPI<T, KC, MR, NR_LANE, LANE>,
{
    #[inline]
    pub fn matmul_loop_1st_mr(
        c: &mut [T],                      // MC x NR (ldc), DRAM
        a: &[[[T; MR]; KC]],              // KC x MC (lda), packed-transposed, cache l2
        b: &[[FpSimd<T, LANE>; NR_LANE]], // KC x NR, packed, aligned, cache l1
        mc: usize,                        // mc, for write to C
        nr: usize,                        // nr, for write to C
        kc: usize,                        // kc, to avoid non-necessary / uninitialized access
        ldc: usize,                       // ldc, for write to C
        barrier: &Mutex<()>,              // barrier for writing to C
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
            unsafe { Self::microkernel(&mut c_reg, &a[i_pack], b, kc) }

            // store C register block to C memory block
            let lock = barrier.lock().unwrap();
            unsafe {
                for ii in 0..mr {
                    let c_assign = &mut c.get_unchecked_mut((i + ii) * ldc..(i + ii) * ldc + nr);
                    let c_regslc = core::slice::from_raw_parts(c_reg[ii].as_ptr() as *const T, nr);
                    for j in 0..nr {
                        *c_assign.get_unchecked_mut(j) += c_regslc.get_unchecked(j).clone();
                    }
                }
            }
            drop(lock);
        }
    }

    #[inline]
    pub fn matmul_loop_2nd_nr_pack_b(
        c: &mut [T],         // MC x NC (ldc), DRAM
        a: &[[[T; MR]; KC]], // KC x MC (lda), packed-transposed, cache l2
        b: &[T],             // KC x NC, aligned, cache l3 with parallel
        mc: usize,           // mc, for write to C
        nc: usize,           // nc, for write to C
        kc: usize,           // kc, to avoid non-necessary / uninitialized access
        ldb: usize,          // ldb, for load and pack B
        ldc: usize,          // ldc, for write to C
        barrier: &Mutex<()>, // barrier for writing to C
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
            if nr == NR_LANE * LANE {
                for p in 0..kc {
                    for j_lane in 0..NR_LANE {
                        buf_b[p][j_lane] = unsafe { FpSimd::loadu_ptr(b[p * ldb + j + j_lane * LANE..].as_ptr()) };
                    }
                }
            } else {
                // avoid out-of-bound access
                for p in 0..kc {
                    for jj in 0..nr {
                        let j_lane = jj / LANE;
                        let j_offset = jj % LANE;
                        buf_b[p][j_lane][j_offset] = b[p * ldb + j + jj].clone();
                    }
                }
            }
            Self::matmul_loop_1st_mr(&mut c[j..], a, &buf_b, mc, nr, kc, ldc, barrier);
        }
    }

    pub fn matmul_loop_parallel_mnk_pack_a(c: &mut [T], a: &[T], b: &[T], m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize)
    where
        T: Send + Sync,
    {
        // compute number of tasks to split
        let ntask_mc = m.div_ceil(MC);
        let ntask_nc = n.div_ceil(NC);
        let ntask_kc = k.div_ceil(KC);
        let ntask_mr = MC.div_ceil(MR);

        // barrier_c: [ntask_mc][ntask_nc]
        let c_barrier: Vec<Mutex<()>> = (0..(ntask_mc * ntask_nc)).map(|_| Mutex::new(())).collect();

        // pack matrix a: [ntask_mc][ntask_kc][ntask_mr][KC][MR]
        let a_pack: Vec<[[T; MR]; KC]> = unsafe { uninitialized_vec(ntask_mc * ntask_kc * ntask_mr) };

        (0..ntask_mc * ntask_kc).into_par_iter().for_each(|task_id| {
            let task_mc = task_id / ntask_kc;
            let task_kc = task_id % ntask_kc;
            let idx_m = task_mc * MC;
            let idx_k = task_kc * KC;
            let mc = if (task_mc + 1) * MC <= m { MC } else { m - idx_m };
            let kc = if (task_kc + 1) * KC <= k { KC } else { k - idx_k };
            let a_pack = unsafe { cast_mut_slice(&a_pack) };
            for task_mr in 0..mc.div_ceil(MR) {
                let idx_mr = task_mr * MR;
                let idx_a_pack = task_mc * (ntask_kc * ntask_mr) + task_kc * ntask_mr + task_mr;
                let mr = if (task_mr + 1) * MR <= mc { MR } else { mc - idx_mr };
                for p in 0..kc {
                    for i in 0..mr {
                        a_pack[idx_a_pack][p][i] = a[(idx_m + idx_mr + i) * lda + (idx_k + p)].clone();
                    }
                }
            }
        });

        let task_count = Mutex::new(0); // schedule tasks exactly by deterministic order [m n k]
        (0..ntask_mc * ntask_nc * ntask_kc).into_par_iter().for_each(|_| {
            let task_id = {
                let mut count = task_count.lock().unwrap();
                let id = *count;
                *count += 1;
                id
            };

            let task_m = task_id % ntask_mc;
            let task_n = (task_id / ntask_mc) % ntask_nc;
            let task_k = task_id / (ntask_mc * ntask_nc);

            // get slices
            let b = &b[task_k * KC * ldb + task_n * NC..];
            let mc = if (task_m + 1) * MC <= m { MC } else { m - task_m * MC };
            let nc = if (task_n + 1) * NC <= n { NC } else { n - task_n * NC };
            let kc = if (task_k + 1) * KC <= k { KC } else { k - task_k * KC };

            let a_pack_mc_kc = &a_pack[task_m * (ntask_kc * ntask_mr) + task_k * ntask_mr..];
            let barrier = &c_barrier[task_m * ntask_nc + task_n];
            let c_mc_nc = unsafe { cast_mut_slice(&c[task_m * MC * ldc + task_n * NC..]) };

            Self::matmul_loop_2nd_nr_pack_b(c_mc_nc, a_pack_mc_kc, b, mc, nc, kc, ldb, ldc, barrier);
        });
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
    // MatmulLoops::<f64, 234, 256, 240, 13, 2,
    // 8>::matmul_loop_parallel_mnk_pack_a(c, a, b, m, n, k, lda, ldb, ldc);

    MatmulLoops::<f64, 234, 256, 240, 13, 2, 8>::matmul_loop_parallel_mnk_pack_a(c, a, b, m, n, k, lda, ldb, ldc);
}

#[test]
fn test_matmul_anyway_full() {
    let m = 3527;
    let n = 9583;
    let k = 6581;
    let lda = k;
    let ldb = n;
    let ldc = n;
    let a: Vec<f64> = (0..m * lda).into_par_iter().map(|x| (x as f64).sin()).collect();
    let b: Vec<f64> = (0..k * ldb).into_par_iter().map(|x| (x as f64).cos()).collect();

    let time = std::time::Instant::now();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc);
    let elapsed = time.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);

    use rstsr::prelude::*;
    let device = DeviceOpenBLAS::default();
    let a_tsr = rt::asarray((&a, [m, k], &device));
    let b_tsr = rt::asarray((&b, [k, n], &device));
    let time = std::time::Instant::now();
    let c_ref = a_tsr % b_tsr;
    let elapsed = time.elapsed();
    println!("Elapsed time (ref): {:.3?}", elapsed);

    let c_tsr = rt::asarray((&c, [m, n], &device));
    let diff = &c_tsr - &c_ref;
    println!("Max error: {:.6e}", diff.view().abs().max());

    // println!("c_tsr\n{c_tsr:15.3}");
    // println!("c_ref\n{c_ref:15.3}");
}

#[test]
fn test_matmul_faer_full() {
    // taxol, def2-TZVP
    // nelec = 452, nbasis = 2228, grids = 968656 (can be batched by 16384)
    // let m = 3527;
    // let n = 9583;
    // let k = 6581;
    let m = 226;
    let n = 16384;
    let k = 2228;
    let lda = k;
    let ldb = n;
    let ldc = n;
    let a: Vec<f64> = (0..m * lda).map(|x| (x as f64).sin()).collect();
    let b: Vec<f64> = (0..k * ldb).map(|x| (x as f64).cos()).collect();

    let time = std::time::Instant::now();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc);
    let elapsed = time.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);

    let time = std::time::Instant::now();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc);
    let elapsed = time.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);

    let time = std::time::Instant::now();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc);
    let elapsed = time.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);

    let a = faer::MatRef::from_column_major_slice(&a, m, k);
    let b = faer::MatRef::from_column_major_slice(&b, k, n);

    let time = std::time::Instant::now();
    let _c = a.mul(&b);
    let elapsed = time.elapsed();
    println!("Elapsed time (faer): {:.3?}", elapsed);

    let time = std::time::Instant::now();
    let _c = a.mul(&b);
    let elapsed = time.elapsed();
    println!("Elapsed time (faer): {:.3?}", elapsed);

    let time = std::time::Instant::now();
    let _c = a.mul(&b);
    let elapsed = time.elapsed();
    println!("Elapsed time (faer): {:.3?}", elapsed);
}
