use crate::prelude::*;

impl<T, const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize, const MB: usize>
    MatmulLoops<T, MC, KC, NC, MR, NR_LANE, LANE, MB>
where
    T: Mul<Output = T> + AddAssign<T> + Add<Output = T> + MulAdd<T, Output = T> + Clone,
    Self: MatmulMicroKernelAPI<T, KC, MR, NR_LANE, LANE>,
    Self: MatmulMicroKernelMaskKForBAPI<T, KC, MR, NR_LANE, LANE>,
{
    pub unsafe fn pack_dm_in_aodmao2rho(
        dm_packed: &mut [[[T; MR]; KC]],
        dm: &[T],
        stride_ao: usize,
        indices_m_for_a: &[usize],
        indices_k_for_b: &[usize],
    ) {
        core::hint::assert_unchecked(indices_m_for_a.len() <= MC);
        core::hint::assert_unchecked(indices_k_for_b.len() <= KC);

        for (idx_p, &p) in indices_k_for_b.iter().enumerate() {
            for (task_m, indices_mr) in indices_m_for_a.chunks(MR).enumerate() {
                for (idx_m, m) in indices_mr.iter().enumerate() {
                    let idx_ao = m * stride_ao + p;
                    core::hint::assert_unchecked(task_m < dm_packed.len());
                    dm_packed[task_m][idx_p][idx_m] = (*dm.get_unchecked(idx_ao)).clone();
                }
            }
        }
    }

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    pub unsafe fn aodmao2rho_loop_1st(
        rho: &mut [T],
        dm_packed: &mut [Vec<[[T; MR]; KC]>],
        ao_lhs_packed: &[[TySimd<T, LANE>; NR_LANE]],
        ao_rhs_packed: &[[TySimd<T, LANE>; NR_LANE]],
        gr: usize,
        mc: usize,
        vc: usize,
        strides: [usize; 2],
        nset: usize,
        barrier: &Mutex<()>,
    ) {
        core::hint::assert_unchecked(gr <= NR_LANE * LANE);
        core::hint::assert_unchecked(mc <= MC);
        core::hint::assert_unchecked(vc <= KC);

        if mc == 0 || vc == 0 {
            return;
        }

        let [_, stride_grid] = strides;

        for (task_mr, m_start) in (0..mc).step_by(MR).enumerate() {
            let mr = if m_start + MR <= mc { MR } else { mc - m_start };
            // pack density matrix
            for iset in 0..nset {
                // For this case, rust's release build may fail to put c_reg in registers (to
                // let rho_reg to be also kept in registers). We need explicit
                // black_box to enforce it this code to be kept in registers.
                let c_reg = core::hint::black_box({
                    let mut c_reg: [[TySimd<T, LANE>; NR_LANE]; MR] = zeroed();
                    Self::microkernel(&mut c_reg, &dm_packed[iset][task_mr], ao_rhs_packed, vc);
                    c_reg
                });
                let mut rho_reg: [TySimd<T, LANE>; NR_LANE] = zeroed();
                // perform reduce
                for m in 0..mr {
                    for j_lane in 0..NR_LANE {
                        assert!(m_start + m < ao_lhs_packed.len());
                        let ao_lhs_lane = ao_lhs_packed[m_start + m][j_lane].clone();
                        rho_reg[j_lane].fma_from(c_reg[m][j_lane].clone(), ao_lhs_lane);
                    }
                }
                // load and store rho
                let lock = barrier.lock().unwrap();
                if gr == NR_LANE * LANE {
                    for j_lane in 0..NR_LANE {
                        let idx_rho = iset * stride_grid + j_lane * LANE;
                        let rho_load = TySimd::loadu_ptr(&rho[idx_rho]);
                        (rho_reg[j_lane].clone() + rho_load).storeu_ptr(&mut rho[idx_rho]);
                    }
                } else {
                    for j in 0..gr {
                        let j_lane = j / LANE;
                        let j_offset = j % LANE;
                        let idx_rho = iset * stride_grid + j;
                        rho[idx_rho] = rho_reg[j_lane][j_offset].clone();
                    }
                }
                drop(lock);
            }
        }
    }

    pub fn aodmao2rho_loop_2nd<const BLKSIZE: usize>(
        rhos: &mut [T],
        dms: &[T],
        ao_lhs: &[T],
        ao_rhs: &[T],
        strides: [usize; 2],
        nuvg: [usize; 3],
        nset: usize,
        barrier: &[Mutex<()>],
        tab_lhs: &[bool],
        tab_rhs: &[bool],
        ldtab: usize,
        dm_packed_buffer: &mut [Vec<[[T; MR]; KC]>],
    ) {
        debug_assert!(BLKSIZE.is_multiple_of(NR_LANE * LANE));
        let [uc, vc, gc] = nuvg;
        let [stride_ao, stride_grid] = strides;

        unsafe {
            core::hint::assert_unchecked(uc <= MC);
            core::hint::assert_unchecked(vc <= KC);
            core::hint::assert_unchecked(gc <= NC);
        }

        let NR = NR_LANE * LANE;
        let BLKNR = BLKSIZE / NR;
        let mut ao_lhs_packed: [[TySimd<T, LANE>; NR_LANE]; MC] = unsafe { zeroed() };
        let mut ao_rhs_packed: [[TySimd<T, LANE>; NR_LANE]; KC] = unsafe { zeroed() };
        // iterate the non0tab blocks
        for (task_gblk, g) in (0..gc).step_by(BLKSIZE).enumerate() {
            let gblk = if g + BLKSIZE <= gc { BLKSIZE } else { gc - g };
            let (indices_v, count_v) = Self::get_indices_from_non0tab::<KC>(&tab_rhs[task_gblk..], ldtab, vc);
            let (indices_u, count_u) = Self::get_indices_from_non0tab::<MC>(&tab_lhs[task_gblk..], ldtab, uc);
            if count_v == 0 || count_u == 0 {
                continue;
            }
            // pack dm
            for iset in 0..nset {
                unsafe {
                    Self::pack_dm_in_aodmao2rho(
                        &mut dm_packed_buffer[iset],
                        &dms[iset * stride_ao * stride_ao..],
                        stride_ao,
                        &indices_u[..count_u],
                        &indices_v[..count_v],
                    )
                };
            }
            // iterate the grid points
            for (task_g, g) in (g..g + gblk).step_by(NR).enumerate() {
                let gr = if g + NR <= g + gblk { NR } else { g + gblk - g };
                // pack rhs
                Self::pack_b_no_trans_non0tab(&mut ao_rhs_packed, &ao_rhs[g..], vc, gr, stride_grid, &indices_v[..count_v]);
                // pack lhs
                Self::pack_b_no_trans_non0tab(&mut ao_lhs_packed, &ao_lhs[g..], uc, gr, stride_grid, &indices_u[..count_u]);
                // compute
                unsafe {
                    Self::aodmao2rho_loop_1st(
                        &mut rhos[g..],
                        dm_packed_buffer,
                        &ao_lhs_packed,
                        &ao_rhs_packed,
                        gr,
                        count_u,
                        count_v,
                        [stride_ao, stride_grid],
                        nset,
                        &barrier[task_gblk * BLKNR + task_g],
                    );
                }
            }
        }
    }

    pub fn aodmao2rho_loop_parallel<const BLKSIZE: usize>(
        rhos: &mut [T],
        dms: &[T],
        ao_lhs: &[T],
        ao_rhs: &[T],
        strides: [usize; 2],
        shape: [usize; 2],
        nset: usize,
        tab: &[bool],
        ldtab: usize,
    ) where
        T: Send + Sync,
    {
        let [stride_ao, stride_grid] = strides;
        let [nao, ngrid] = shape;

        let NR = NR_LANE * LANE;
        assert!(NC.is_multiple_of(BLKSIZE));
        let ntask_uc = nao.div_ceil(MC);
        let ntask_vc = nao.div_ceil(KC);
        let ntask_gc = ngrid.div_ceil(NC);
        let ntask_ur = MC.div_ceil(MR);
        let ntask_gr = NC.div_ceil(NR);
        let g_barrier: Vec<Mutex<()>> = (0..ntask_gc * ntask_gr).map(|_| Mutex::new(())).collect();

        let nthreads = rayon::current_num_threads();
        let dm_packed_buffer: Vec<Vec<Vec<[[T; MR]; KC]>>> =
            (0..nthreads).map(|_| (0..nset).map(|_| unsafe { uninitialized_vec(ntask_ur) }).collect()).collect();

        let task_count = Mutex::new(0);
        (0..ntask_uc * ntask_vc * ntask_gc).into_par_iter().for_each(|_| {
            let task_id = {
                let mut count = task_count.lock().unwrap();
                let id = *count;
                *count += 1;
                id
            };
            let idx_thread = rayon::current_thread_index().unwrap_or(0);

            let task_u = task_id % ntask_uc;
            let task_g = (task_id / ntask_uc) % ntask_gc;
            let task_v = task_id / (ntask_uc * ntask_gc);
            let task_tab = (task_g * NC) / BLKSIZE;

            // get slices
            let u_start = task_u * MC;
            let v_start = task_v * KC;
            let g_start = task_g * NC;
            let u_end = ((task_u + 1) * MC).min(nao);
            let v_end = ((task_v + 1) * KC).min(nao);
            let g_end = ((task_g + 1) * NC).min(ngrid);
            let u_len = u_end - u_start;
            let v_len = v_end - v_start;
            let g_len = g_end - g_start;

            let ao_lhs = &ao_lhs[u_start * stride_grid + g_start..];
            let ao_rhs = &ao_rhs[v_start * stride_grid + g_start..];
            let dm = &dms[u_start * stride_ao + v_start..];
            let tab_lhs = &tab[u_start * ldtab + task_tab..];
            let tab_rhs = &tab[v_start * ldtab + task_tab..];
            let rho = unsafe { cast_mut_slice(&rhos[g_start..]) };
            let dm_packed = unsafe { cast_mut_slice(&dm_packed_buffer[idx_thread]) };

            Self::aodmao2rho_loop_2nd::<BLKSIZE>(
                rho,
                dm,
                ao_lhs,
                ao_rhs,
                [stride_ao, stride_grid],
                [u_len, v_len, g_len],
                nset,
                &g_barrier[task_g * ntask_gr..],
                tab_lhs,
                tab_rhs,
                ldtab,
                dm_packed,
            );
        });
    }
}

pub fn aodmao2rho_anyway(
    rhos: &mut [f64],
    dms: &[f64],
    ao_lhs: &[f64],
    ao_rhs: &[f64],
    strides: [usize; 2],
    shape: [usize; 2],
    nset: usize,
    tab: &[bool],
    ldtab: usize,
) {
    MatmulLoops::<f64, 252, 512, 240, 14, 2, 8>::aodmao2rho_loop_parallel::<48>(
        rhos, dms, ao_lhs, ao_rhs, strides, shape, nset, tab, ldtab,
    );
}

#[test]
pub fn test_aodmao2rho() {
    let nao: usize = 2228;
    let ngrid: usize = 16384;
    let nset: usize = 3;

    let ao_lhs: Vec<f64> = (0..nao * ngrid).map(|x| (x as f64).sin()).collect();
    let ao_rhs: Vec<f64> = (0..nao * ngrid).map(|x| (x as f64).cos()).collect();
    let dm: Vec<f64> = (0..nset * nao * nao).map(|x| (x as f64 * 0.38 + 1.2).sin()).collect();

    // special treatment for zeroing
    // non0tab: (k, n / BLKSIZE)
    const BLKSIZE: usize = 48;
    let nblk = ngrid.div_ceil(BLKSIZE);
    let mod_pattern = [true, false, true, false, false, true, false, false, false, false]; // 30%
    let non0tab: Vec<bool> = (0..nao * nblk).into_par_iter().map(|i| mod_pattern[i % 10]).collect();
    // set zero elements in B
    (0..nao * nblk).into_par_iter().for_each(|idx| {
        let ao_lhs = unsafe { cast_mut_slice(&ao_lhs) };
        let ao_rhs = unsafe { cast_mut_slice(&ao_rhs) };
        let p = idx / nblk;
        let blk = idx % nblk;
        if !non0tab[idx] {
            let j0 = blk * BLKSIZE;
            let j1 = ((blk + 1) * BLKSIZE).min(ngrid);
            for j in j0..j1 {
                ao_lhs[p * ngrid + j] = 0.0;
                ao_rhs[p * ngrid + j] = 0.0;
            }
        }
    });

    let time = std::time::Instant::now();
    let mut rho = vec![0.0f64; nset * ngrid];
    aodmao2rho_anyway(&mut rho, &dm, &ao_lhs, &ao_rhs, [nao, ngrid], [nao, ngrid], nset, &non0tab, nblk);
    let elapsed = time.elapsed();
    println!("Elapsed time (aodm2rho): {:.3?}", elapsed);

    use rstsr::prelude::*;
    let device = DeviceOpenBLAS::default();
    let ao_rhs_tsr = rt::asarray((&ao_rhs, [nao, ngrid], &device));
    let dm_tsr = rt::asarray((&dm, [nset, nao, nao], &device));

    let time = std::time::Instant::now();
    let dm_dot_ao = &dm_tsr % &ao_rhs_tsr; // (nset, nao, ngrid)
    println!("Elapsed time (dm_dot_ao reference): {:.3?}", time.elapsed());

    let time = std::time::Instant::now();
    let rho_ref = vec![0.0f64; nset * ngrid];
    (0..ngrid).into_par_iter().chunks(1024).for_each(|chunk| {
        let rho_ref = unsafe { cast_mut_slice(&rho_ref) };
        for iset in 0..nset {
            for u in 0..nao {
                for &j in chunk.iter() {
                    rho_ref[iset * ngrid + j] += ao_lhs[u * ngrid + j] * dm_dot_ao.raw()[iset * nao * ngrid + u * ngrid + j];
                }
            }
        }
    });
    println!("Elapsed time (rho from reduce reference): {:.3?}", time.elapsed());

    let rho = rt::asarray((&rho, [nset, ngrid], &device));
    let rho_ref = rt::asarray((&rho_ref, [nset, ngrid], &device));
    let diff = rho.view() - rho_ref.view();
    let err = diff.view().abs().max();
    println!("Max abs error: {:.6e}", err);
}
