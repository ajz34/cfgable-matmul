use crate::prelude::*;

/* #region microkernel-traits */

pub trait MatmulMicroKernelMaskKForBAPI<T, const KC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize>
where
    T: Mul<Output = T> + AddAssign<T> + Clone,
{
    unsafe fn microkernel_mask_k_for_b(
        c: &mut [[TySimd<T, LANE>; NR_LANE]], // MR x NR, aligned, register
        a: &[[T; MR]],                        // kc x MR (lda), packed-transposed, cache l2 prefetch l1
        b: &[[TySimd<T, LANE>; NR_LANE]],     // compressed-kc x NR, compressed-packed, aligned, cache l1
        indices_k_for_b: &[usize],            // compressed-kc indices for B
    );
}

impl<const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize, const MB: usize>
    MatmulMicroKernelMaskKForBAPI<f64, KC, MR, NR_LANE, LANE> for MatmulLoops<f64, MC, KC, NC, MR, NR_LANE, LANE, MB>
{
    #[inline]
    unsafe fn microkernel_mask_k_for_b(
        c: &mut [[TySimd<f64, LANE>; NR_LANE]],
        a: &[[f64; MR]],
        b: &[[TySimd<f64, LANE>; NR_LANE]],
        indices_k_for_b: &[usize],
    ) {
        core::hint::assert_unchecked(indices_k_for_b.len() <= b.len());
        core::hint::assert_unchecked(c.len() == MR);

        let c: &mut [[TySimd<f64, LANE>; NR_LANE]] = transmute(c);
        let b: &[[TySimd<f64, LANE>; NR_LANE]] = transmute(b);
        for (idx_p, &p) in indices_k_for_b.iter().enumerate() {
            core::hint::assert_unchecked(p < a.len());
            for i in 0..MR {
                let a_ip = TySimd::splat(a[p][i]);
                for j_lane in 0..NR_LANE {
                    let b_pj = b[idx_p][j_lane];
                    c[i][j_lane].fma_from(a_ip, b_pj);
                }
                core::hint::black_box(());
            }
        }
    }
}

/* #endregion microkernel-traits */

impl<T, const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize, const MB: usize>
    MatmulLoops<T, MC, KC, NC, MR, NR_LANE, LANE, MB>
where
    T: Mul<Output = T> + AddAssign<T> + Clone,
    Self: MatmulMicroKernelMaskKForBAPI<T, KC, MR, NR_LANE, LANE>,
{
    #[inline]
    pub unsafe fn matmul_loop_1st_mr_mask_k_for_b(
        c: &mut [T],
        a: &[[[T; MR]; KC]],
        b: &[[TySimd<T, LANE>; NR_LANE]],
        mc: usize,
        nr: usize,
        kc: usize,
        ldc: usize,
        indices_k_for_b: &[usize],
        barrier: &[Mutex<()>],
    ) {
        core::hint::assert_unchecked(mc <= MC);
        core::hint::assert_unchecked(nr <= NR_LANE * LANE);
        core::hint::assert_unchecked(kc <= KC);
        core::hint::assert_unchecked(indices_k_for_b.len() <= KC);

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
                Self::microkernel_mask_k_for_b(&mut c_reg, &a[task_i], b, indices_k_for_b);
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
                Self::microkernel_mask_k_for_b(&mut c_reg, &a[task_i], b, indices_k_for_b);
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
    pub fn pack_b_no_trans_non0tab(
        dst: &mut [[TySimd<T, LANE>; NR_LANE]],
        src: &[T],
        kc: usize,
        nr: usize,
        ldb: usize,
        tab: &[bool],
        ldtab: usize,
    ) -> ([usize; KC], usize) {
        unsafe { core::hint::assert_unchecked(kc <= KC) };
        unsafe { core::hint::assert_unchecked(nr <= NR_LANE * LANE) };

        // generate indices_k_for_b
        let mut indices_k_for_b = [0; KC];
        let mut cnt_k_for_b: usize = 0;
        for p in 0..kc {
            if tab[p * ldtab] {
                indices_k_for_b[cnt_k_for_b] = p;
                cnt_k_for_b += 1;
            }
        }

        if cnt_k_for_b == 0 {
            return (indices_k_for_b, cnt_k_for_b);
        }

        if nr == NR_LANE * LANE {
            for (idx_p, &p) in indices_k_for_b[..cnt_k_for_b].iter().enumerate() {
                unsafe { core::hint::assert_unchecked(idx_p < kc) };
                unsafe { core::hint::assert_unchecked(p < kc) };
                let src_ptr = unsafe { src.as_ptr().add(p * ldb) };
                for j_lane in 0..NR_LANE {
                    dst[idx_p][j_lane] = unsafe { TySimd::loadu_ptr(src_ptr.add(j_lane * LANE)) };
                }
            }
        } else {
            // avoid out-of-bound access
            for (idx_p, &p) in indices_k_for_b[..cnt_k_for_b].iter().enumerate() {
                unsafe { core::hint::assert_unchecked(idx_p < kc) };
                unsafe { core::hint::assert_unchecked(p < kc) };
                for jj in 0..nr {
                    let j_lane = jj / LANE;
                    let j_offset = jj % LANE;
                    let src_ptr = unsafe { src.as_ptr().add(p * ldb + jj) };
                    dst[idx_p][j_lane][j_offset] = unsafe { (*src_ptr).clone() };
                }
            }
        }

        (indices_k_for_b, cnt_k_for_b)
    }

    #[inline]
    pub fn matmul_loop_2nd_nr_pack_b_non0tab<const BLKNR: usize>(
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
        tab: &[bool],
        ldtab: usize,
    ) {
        if transb {
            unimplemented!("transb=true is not implemented in matmul_loop_2nd_nr_pack_b_non0tab");
        }

        // NR -> NC
        unsafe {
            core::hint::assert_unchecked(mc <= MC);
            core::hint::assert_unchecked(nc <= NC);
            core::hint::assert_unchecked(kc <= KC);
        }

        let NR = NR_LANE * LANE;
        let ntask_mr = MC.div_ceil(MR);
        let mut buf_b: [[TySimd<T, LANE>; NR_LANE]; KC] = unsafe { zeroed() }; // KC x NR, packed, aligned, cache l1

        for (task_j, j) in (0..nc).step_by(NR).enumerate() {
            let nr = if j + NR <= nc { NR } else { nc - j };
            let task_tab = task_j / BLKNR;
            let (indices_k_for_b, cnt_k_for_b) = Self::pack_b_no_trans_non0tab(&mut buf_b, &b[j..], kc, nr, ldb, &tab[task_tab..], ldtab);
            if cnt_k_for_b == 0 {
                continue;
            }
            unsafe {
                Self::matmul_loop_1st_mr_mask_k_for_b(
                    &mut c[j..],
                    a,
                    &buf_b,
                    mc,
                    nr,
                    kc,
                    ldc,
                    &indices_k_for_b[..cnt_k_for_b],
                    &barrier[task_j * ntask_mr..],
                )
            };
        }
    }

    pub fn matmul_loop_parallel_mnk_pack_a_non0tab<const BLKNR: usize>(
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
        tab: &[bool],
        ldtab: usize,
    ) where
        T: Send + Sync,
    {
        if transb {
            unimplemented!("transb=true is not implemented in matmul_loop_parallel_mnk_pack_a_mask_for_b");
        }
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
            let b = &b[task_k * KC * ldb + task_n * NC..];
            let task_tab = (task_n * NC) / (BLKNR * NR);
            let non0tab = &tab[task_tab..];

            let mc = if (task_m + 1) * MC <= m { MC } else { m - task_m * MC };
            let nc = if (task_n + 1) * NC <= n { NC } else { n - task_n * NC };
            let kc = if (task_k + 1) * KC <= k { KC } else { k - task_k * KC };

            let a_pack_mc_kc = &a_pack[task_m * (ntask_kc * ntask_mr) + task_k * ntask_mr..];
            let barrier = &c_barrier[(task_m * ntask_nc + task_n) * (ntask_nr * ntask_mr)..];
            let c_mc_nc = unsafe { cast_mut_slice(&c[task_m * MC * ldc + task_n * NC..]) };

            Self::matmul_loop_2nd_nr_pack_b_non0tab::<BLKNR>(
                c_mc_nc,
                a_pack_mc_kc,
                b,
                mc,
                nc,
                kc,
                ldb,
                ldc,
                transb,
                barrier,
                non0tab,
                ldtab,
            );
        });
    }

    pub fn matmul_loop_macro_mb_non0tab<const BLKNR: usize>(
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
        tab: &[bool],
        ldtab: usize,
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
            Self::matmul_loop_parallel_mnk_pack_a_non0tab::<BLKNR>(c_slc, a_slc, b, mb, n, k, lda, ldb, ldc, transa, transb, tab, ldtab);
        }
    }
}

pub fn matmul_anyway_full_non0tab(
    c: &mut [f64],
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    transa: bool,
    transb: bool,
    tab: &[bool],
    ldtab: usize,
) {
    MatmulLoops::<f64, 252, 512, 240, 14, 2, 8, 2360>::matmul_loop_macro_mb_non0tab::<3>(
        c, a, b, m, n, k, lda, ldb, ldc, transa, transb, tab, ldtab,
    );
}

#[test]
fn test_matmul_anyway_full_mask_k_for_b() {
    let m: usize = 3527;
    let n: usize = 9583;
    let k: usize = 6581;
    let lda = k;
    let ldb = n;
    let ldc = n;
    let a: Vec<f64> = (0..m * lda).into_par_iter().map(|x| (x as f64).sin()).collect();
    let b: Vec<f64> = (0..k * ldb).into_par_iter().map(|x| (x as f64).cos()).collect();

    // special treatment for zeroing
    // non0tab: (k, n / BLKSIZE)
    const BLKSIZE: usize = 48;
    let nblk = n.div_ceil(BLKSIZE);
    let mod_pattern = [true, false, true, false, false, true, false, false, false, false]; // 30%
    let non0tab: Vec<bool> = (0..k * nblk).into_par_iter().map(|i| mod_pattern[i % 10]).collect();
    // set zero elements in B
    (0..k * nblk).into_par_iter().for_each(|idx| {
        let b = unsafe { cast_mut_slice(&b) };
        let p = idx / nblk;
        let blk = idx % nblk;
        if !non0tab[idx] {
            let j0 = blk * BLKSIZE;
            let j1 = ((blk + 1) * BLKSIZE).min(n);
            for j in j0..j1 {
                b[p * ldb + j] = 0.0;
            }
        }
    });

    let time = std::time::Instant::now();
    let mut c: Vec<f64> = vec![0.0; m * ldc];
    matmul_anyway_full_non0tab(&mut c, &a, &b, m, n, k, lda, ldb, ldc, false, false, &non0tab, nblk);
    let elapsed = time.elapsed();
    println!("Elapsed time (mask_k_for_b): {:.3?}", elapsed);

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
}
