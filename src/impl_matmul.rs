use crate::prelude::*;

impl<T, const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize, const MB: usize>
    MatmulLoops<T, MC, KC, NC, MR, NR_LANE, LANE, MB>
where
    T: Mul<Output = T> + AddAssign<T> + Clone,
    Self: MatmulMicroKernelAPI<T, KC, MR, NR_LANE, LANE>,
{
    #[inline]
    pub unsafe fn matmul_loop_1st_mr(
        c: &mut [T],                      // MC x NR (ldc), DRAM
        a: &[[[T; MR]; KC]],              // KC x MC (lda), packed-transposed, cache l2
        b: &[[TySimd<T, LANE>; NR_LANE]], // KC x NR, packed, aligned, cache l1
        mc: usize,                        // mc, for write to C
        nr: usize,                        // nr, for write to C
        kc: usize,                        // kc, to avoid non-necessary / uninitialized access
        ldc: usize,                       // ldc, for write to C
        barrier: &[Mutex<()>],            // barrier for writing to C
    ) {
        // MR -> MC
        core::hint::assert_unchecked(mc <= MC);
        core::hint::assert_unchecked(nr <= NR_LANE * LANE);
        core::hint::assert_unchecked(kc <= KC);

        for (task_i, i) in (0..mc).step_by(MR).enumerate() {
            let lock = barrier[task_i].lock().unwrap();
            let mr = if i + MR <= mc { MR } else { mc - i };
            core::hint::assert_unchecked(mr <= MR);

            if nr == NR_LANE * LANE && mr == MR {
                let mut c_reg: [[TySimd<T, LANE>; NR_LANE]; MR] = unsafe { zeroed() };
                // load C memory block to C register block
                for ii in 0..MR {
                    let c_ptr = &c.as_ptr().add((i + ii) * ldc);
                    for j_lane in 0..NR_LANE {
                        c_reg[ii][j_lane] = TySimd::loadu_ptr(c_ptr.add(j_lane * LANE));
                    }
                }
                // call micro-kernel
                Self::microkernel(&mut c_reg, &a[task_i], b, kc);
                // store C register block to C memory block
                for ii in 0..MR {
                    let c_ptr = &mut c.as_mut_ptr().add((i + ii) * ldc);
                    for j_lane in 0..NR_LANE {
                        c_reg[ii][j_lane].storeu_ptr(c_ptr.add(j_lane * LANE));
                    }
                }
            } else {
                // avoid out-of-bound access
                let mut c_reg: [[TySimd<T, LANE>; NR_LANE]; MR] = unsafe { zeroed() };
                // load C memory block to C register block
                for ii in 0..mr {
                    let c_mem = &c[(i + ii) * ldc..(i + ii) * ldc + nr];
                    for jj in 0..nr {
                        let j_lane = jj / LANE;
                        let j_offset = jj % LANE;
                        c_reg[ii][j_lane][j_offset] = c_mem[jj].clone();
                    }
                }
                // call micro-kernel
                Self::microkernel(&mut c_reg, &a[task_i], b, kc);
                // store C register block to C memory block
                for ii in 0..mr {
                    let c_mem = &mut c[(i + ii) * ldc..(i + ii) * ldc + nr];
                    for jj in 0..nr {
                        let j_lane = jj / LANE;
                        let j_offset = jj % LANE;
                        c_mem[jj] = c_reg[ii][j_lane][j_offset].clone();
                    }
                }
            }
            drop(lock);
        }
    }

    #[inline]
    pub fn pack_b_no_trans(dst: &mut [[TySimd<T, LANE>; NR_LANE]], src: &[T], kc: usize, nr: usize, ldb: usize) {
        unsafe { core::hint::assert_unchecked(kc <= KC) };
        unsafe { core::hint::assert_unchecked(nr <= NR_LANE * LANE) };

        if nr == NR_LANE * LANE {
            for p in 0..kc {
                let b_ptr = unsafe { src.as_ptr().add(p * ldb) };
                for j_lane in 0..NR_LANE {
                    dst[p][j_lane] = unsafe { TySimd::loadu_ptr(b_ptr.add(j_lane * LANE)) };
                }
            }
        } else {
            // avoid out-of-bound access
            for p in 0..kc {
                for jj in 0..nr {
                    let j_lane = jj / LANE;
                    let j_offset = jj % LANE;
                    dst[p][j_lane][j_offset] = src[p * ldb + jj].clone();
                }
            }
        }
    }

    #[inline]
    pub fn pack_b_trans(dst: &mut [[TySimd<T, LANE>; NR_LANE]], src: &[T], kc: usize, nr: usize, ldb: usize, buf: &mut [T]) {
        unsafe { core::hint::assert_unchecked(kc <= KC) };
        unsafe { core::hint::assert_unchecked(nr <= NR_LANE * LANE) };

        // first copy to temporary buffer without transpose
        for j in 0..nr {
            buf[j * KC..j * KC + kc].clone_from_slice(&src[j * ldb..j * ldb + kc]);
        }

        // then transpose to dst
        let dst_slc = unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut T, KC * NR_LANE * LANE) };
        for p in 0..KC {
            for j in 0..NR_LANE * LANE {
                dst_slc[p * NR_LANE * LANE + j] = buf[j * KC + p].clone();
            }
        }
    }

    #[inline]
    pub fn matmul_loop_2nd_nr_pack_b(
        c: &mut [T],           // MC x NC (ldc), DRAM
        a: &[[[T; MR]; KC]],   // KC x MC (lda), packed-transposed, cache l2
        b: &[T],               // KC x NC, aligned, cache l3 with parallel
        mc: usize,             // mc, for write to C
        nc: usize,             // nc, for write to C
        kc: usize,             // kc, to avoid non-necessary / uninitialized access
        ldb: usize,            // ldb, for load and pack B
        ldc: usize,            // ldc, for write to C
        transb: bool,          // whether B is transposed
        barrier: &[Mutex<()>], // barrier for writing to C
    ) {
        // NR -> NC
        unsafe {
            core::hint::assert_unchecked(mc <= MC);
            core::hint::assert_unchecked(nc <= NC);
            core::hint::assert_unchecked(kc <= KC);
        }

        let NR = NR_LANE * LANE;
        let ntask_mr = MC.div_ceil(MR);
        let mut buf_b: [[TySimd<T, LANE>; NR_LANE]; KC] = unsafe { zeroed() }; // KC x NR, packed, aligned, cache l1
        let mut buf_trans_b: [[[T; LANE]; NR_LANE]; KC] = unsafe { zeroed() }; // KC x NR, temporary buffer for transpose
        let buf_trans_b = unsafe { core::slice::from_raw_parts_mut(buf_trans_b.as_mut_ptr() as *mut T, KC * NR_LANE * LANE) };

        for (task_j, j) in (0..nc).step_by(NR).enumerate() {
            let nr = if j + NR <= nc { NR } else { nc - j };
            match transb {
                true => Self::pack_b_trans(&mut buf_b, &b[j * ldb..], kc, nr, ldb, buf_trans_b),
                false => Self::pack_b_no_trans(&mut buf_b, &b[j..], kc, nr, ldb),
            }
            unsafe { Self::matmul_loop_1st_mr(&mut c[j..], a, &buf_b, mc, nr, kc, ldc, &barrier[task_j * ntask_mr..]) };
        }
    }

    pub fn matmul_loop_parallel_mnk_pack_a(
        c: &mut [T],
        a: &[T],
        b: &[T],
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        transa: bool,
        transb: bool,
    ) where
        T: Send + Sync,
    {
        // compute number of tasks to split
        let NR = NR_LANE * LANE;
        let ntask_mc = m.div_ceil(MC);
        let ntask_nc = n.div_ceil(NC);
        let ntask_kc = k.div_ceil(KC);
        let ntask_mr = MC.div_ceil(MR);
        let ntask_nr = NC.div_ceil(NR);

        // barrier_c: [ntask_mc][ntask_nc][ntask_nr][ntask_mr]
        let c_barrier: Vec<Mutex<()>> = (0..(ntask_mc * ntask_nc * ntask_nr * ntask_mr)).map(|_| Mutex::new(())).collect();

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
                        match transa {
                            true => a_pack[idx_a_pack][p][i] = a[(idx_k + p) * lda + (idx_m + idx_mr + i)].clone(),
                            false => a_pack[idx_a_pack][p][i] = a[(idx_m + idx_mr + i) * lda + (idx_k + p)].clone(),
                        }
                    }
                }
            }
        });

        // schedule tasks exactly by deterministic order [m n k]
        let task_count = Mutex::new(0);
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
            let b = match transb {
                true => &b[task_n * NC * ldb + task_k * KC..],
                false => &b[task_k * KC * ldb + task_n * NC..],
            };
            let mc = if (task_m + 1) * MC <= m { MC } else { m - task_m * MC };
            let nc = if (task_n + 1) * NC <= n { NC } else { n - task_n * NC };
            let kc = if (task_k + 1) * KC <= k { KC } else { k - task_k * KC };

            let a_pack_mc_kc = &a_pack[task_m * (ntask_kc * ntask_mr) + task_k * ntask_mr..];
            let barrier = &c_barrier[(task_m * ntask_nc + task_n) * (ntask_nr * ntask_mr)..];
            let c_mc_nc = unsafe { cast_mut_slice(&c[task_m * MC * ldc + task_n * NC..]) };

            Self::matmul_loop_2nd_nr_pack_b(c_mc_nc, a_pack_mc_kc, b, mc, nc, kc, ldb, ldc, transb, barrier);
        });
    }

    pub fn matmul_loop_macro_mb(
        c: &mut [T],
        a: &[T],
        b: &[T],
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        transa: bool,
        transb: bool,
    ) where
        T: Send + Sync,
    {
        let mb_size = if MB == 0 { m } else { MB };
        for i in (0..m).step_by(mb_size) {
            let mb = if i + mb_size <= m { mb_size } else { m - i };
            let a_slc = match transa {
                true => &a[i..],
                false => &a[i * lda..],
            };
            let c_slc = &mut c[i * ldc..];
            Self::matmul_loop_parallel_mnk_pack_a(c_slc, a_slc, b, mb, n, k, lda, ldb, ldc, transa, transb);
        }
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
    transa: bool,
    transb: bool,
) {
    MatmulLoops::<f64, 234, 512, 240, 13, 2, 8, 2360>::matmul_loop_macro_mb(c, a, b, m, n, k, lda, ldb, ldc, transa, transb);
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
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc, false, false);
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
fn test_matmul_anyway_transb() {
    let m = 3527;
    let n = 9583;
    let k = 6581;
    let lda = k;
    let ldb = k;
    let ldc = n;
    let a: Vec<f64> = (0..m * lda).into_par_iter().map(|x| (x as f64).sin()).collect();
    let b: Vec<f64> = (0..n * ldb).into_par_iter().map(|x| (x as f64).cos()).collect();

    let time = std::time::Instant::now();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc, false, true);
    let elapsed = time.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);

    use rstsr::prelude::*;
    let device = DeviceOpenBLAS::default();
    let a_tsr = rt::asarray((&a, [m, k], &device));
    let b_tsr = rt::asarray((&b, [n, k], &device));
    let time = std::time::Instant::now();
    let c_ref = a_tsr % b_tsr.t();
    let elapsed = time.elapsed();
    println!("Elapsed time (ref): {:.3?}", elapsed);

    let c_tsr = rt::asarray((&c, [m, n], &device));
    let diff = &c_tsr - &c_ref;
    println!("Max error: {:.6e}", diff.view().abs().max());

    // println!("c_tsr\n{c_tsr:15.3}");
    // println!("c_ref\n{c_ref:15.3}");
}

#[test]
fn test_matmul_anyway_transa() {
    let m = 3527;
    let n = 9583;
    let k = 6581;
    let lda = m;
    let ldb = n;
    let ldc = n;
    let a: Vec<f64> = (0..k * lda).into_par_iter().map(|x| (x as f64).sin()).collect();
    let b: Vec<f64> = (0..k * ldb).into_par_iter().map(|x| (x as f64).cos()).collect();

    let time = std::time::Instant::now();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc, true, false);
    let elapsed = time.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);

    use rstsr::prelude::*;
    let device = DeviceOpenBLAS::default();
    let a_tsr = rt::asarray((&a, [k, m], &device));
    let b_tsr = rt::asarray((&b, [k, n], &device));
    let time = std::time::Instant::now();
    let c_ref = a_tsr.t() % b_tsr;
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
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc, false, false);
    let elapsed = time.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);

    let time = std::time::Instant::now();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc, false, false);
    let elapsed = time.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);

    let time = std::time::Instant::now();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    matmul_anyway_full(&mut c, &a, &b, m, n, k, lda, ldb, ldc, false, false);
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
