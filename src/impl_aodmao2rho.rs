use std::mem::MaybeUninit;

use crate::prelude::*;

impl<T, const MC: usize, const KC: usize, const NC: usize, const MR: usize, const NR_LANE: usize, const LANE: usize, const MB: usize>
    MatmulLoops<T, MC, KC, NC, MR, NR_LANE, LANE, MB>
where
    T: Mul<Output = T> + AddAssign<T> + Add<Output = T> + MulAdd<T, Output = T> + Clone,
    Self: MatmulMicroKernelAPI<T, KC, MR, NR_LANE, LANE>,
    Self: MatmulMicroKernelMaskKForBAPI<T, KC, MR, NR_LANE, LANE>,
{
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    pub unsafe fn aodmao2rho_loop_1st(
        rho: &mut [T],
        dm: &[T],
        ao_lhs: &[T],
        ao_rhs_packed: &[[TySimd<T, LANE>; NR_LANE]],
        gr: usize,
        indices_m_for_a: &[usize],
        indices_k_for_b: &[usize],
        strides: [usize; 2],
        nset: usize,
        barrier: &Mutex<()>,
    ) {
        core::hint::assert_unchecked(gr <= NR_LANE * LANE);
        core::hint::assert_unchecked(indices_m_for_a.len() <= MC);
        core::hint::assert_unchecked(indices_k_for_b.len() <= KC);

        if indices_m_for_a.is_empty() || indices_k_for_b.is_empty() {
            return;
        }

        let [stride_ao, stride_grid] = strides;

        for indices_m_for_a_chunk in indices_m_for_a.chunks(MR) {
            let mr = indices_m_for_a_chunk.len();
            let kc = indices_k_for_b.len();
            // pack density matrix
            let mut dm_packed: [[T; MR]; KC] = MaybeUninit::uninit().assume_init();
            for (idx_p, &p) in indices_k_for_b.iter().enumerate() {
                for (idx_m, &m) in indices_m_for_a_chunk.iter().enumerate() {
                    dm_packed[idx_p][idx_m] = dm[m * stride_ao + p].clone();
                }
            }
            let mut c_reg: [[TySimd<T, LANE>; NR_LANE]; MR] = zeroed();
            Self::microkernel(&mut c_reg, &dm_packed, ao_rhs_packed, kc);
            for iset in 0..nset {
                let mut rho_reg: [TySimd<T, LANE>; NR_LANE] = zeroed();
                // perform reduce
                if gr == NR_LANE * LANE {
                    for m in 0..mr {
                        for j_lane in 0..NR_LANE {
                            let ao_lhs_lane = TySimd::loadu_ptr(
                                &ao_lhs[iset * stride_ao * stride_grid + indices_m_for_a_chunk[m] * stride_grid + j_lane * LANE],
                            );
                            rho_reg[j_lane].fma_from(c_reg[m][j_lane].clone(), ao_lhs_lane);
                        }
                    }
                } else {
                    for m in 0..mr {
                        for j_lane in 0..gr.div_ceil(LANE) {
                            let ao_lhs_lane = TySimd::loadu_ptr(
                                &ao_lhs[iset * stride_ao * stride_grid + indices_m_for_a_chunk[m] * stride_grid + j_lane * LANE],
                            );
                            rho_reg[j_lane].fma_from(c_reg[m][j_lane].clone(), ao_lhs_lane);
                        }
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
                        rho[idx_rho] += rho_reg[j_lane][j_offset].clone();
                    }
                }
                drop(lock);
            }
        }
    }

    pub fn aodmao2rho_loop_2nd<const BLKNR: usize>(
        rho: &mut [T],
        dm: &[T],
        ao_lhs: &[T],
        ao_rhs: &[T],
        strides: [usize; 2],
        nuvg: [usize; 3],
        nset: usize,
        barrier: &[Mutex<()>],
        tab_lhs: &[bool],
        tab_rhs: &[bool],
        ldtab: usize,
    ) {
        let [uc, vc, gc] = nuvg;
        let [stride_ao, stride_grid] = strides;

        unsafe {
            core::hint::assert_unchecked(uc <= MC);
            core::hint::assert_unchecked(vc <= KC);
            core::hint::assert_unchecked(gc <= NC);
        }

        let NR = NR_LANE * LANE;
        let mut ao_rhs_packed: [[TySimd<T, LANE>; NR_LANE]; KC] = unsafe { zeroed() };
        for (task_g, g) in (0..gc).step_by(NR).enumerate() {
            let gr = if g + NR <= gc { NR } else { gc - g };
            let task_tab = task_g / BLKNR;
            // get indices and quick return
            let (indices_v, count_v) = Self::get_indices_from_non0tab::<KC>(&tab_rhs[task_tab..], ldtab, vc);
            let (indices_u, count_u) = Self::get_indices_from_non0tab::<MC>(&tab_lhs[task_tab..], ldtab, uc);
            if count_v == 0 || count_u == 0 {
                continue;
            }
            // pack rhs
            Self::pack_b_no_trans_non0tab(&mut ao_rhs_packed, &ao_rhs[g..], vc, gr, stride_grid, &indices_v[..count_v]);
            unsafe {
                Self::aodmao2rho_loop_1st(
                    &mut rho[g..],
                    dm,
                    &ao_lhs[g..],
                    &ao_rhs_packed,
                    gr,
                    &indices_u[..count_u],
                    &indices_v[..count_v],
                    [stride_ao, stride_grid],
                    nset,
                    &barrier[task_g],
                );
            }
        }
    }

    pub fn aodmao2rho_loop_parallel<const BLKNR: usize>(
        rho: &mut [T],
        dm: &[T],
        ao_lhs: &[T],
        ao_rhs: &[T],
        strides: [usize; 2],
        shape: [usize; 3],
        tab: &[bool],
        ldtab: usize,
    ) where
        T: Send + Sync,
    {
        let [stride_ao, stride_grid] = strides;
        let [nset, nao, ngrid] = shape;

        let NR = NR_LANE * LANE;
        let BLKSIZE = BLKNR * NR;
        assert!(NC.is_multiple_of(BLKSIZE));
        let ntask_uc = nao.div_ceil(MC);
        let ntask_vc = nao.div_ceil(KC);
        let ntask_gc = ngrid.div_ceil(NC);
        let ntask_gr = NC.div_ceil(NR);
        let g_barrier: Vec<Mutex<()>> = (0..ntask_gc * ntask_gr).map(|_| Mutex::new(())).collect();

        let task_count = Mutex::new(0);

        (0..ntask_uc * ntask_vc * ntask_gc).into_par_iter().for_each(|_| {
            let task_id = {
                let mut count = task_count.lock().unwrap();
                let id = *count;
                *count += 1;
                id
            };

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
            let dm = &dm[u_start * stride_ao + v_start..];
            let tab_lhs = &tab[u_start * ldtab + task_tab..];
            let tab_rhs = &tab[v_start * ldtab + task_tab..];
            let rho = unsafe { cast_mut_slice(&rho[g_start..]) };

            Self::aodmao2rho_loop_2nd::<BLKNR>(
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
            );
        });
    }
}

pub fn aodmao2rho_anyway(
    rho: &mut [f64],
    dm: &[f64],
    ao_lhs: &[f64],
    ao_rhs: &[f64],
    strides: [usize; 2],
    shape: [usize; 3],
    tab: &[bool],
    ldtab: usize,
) {
    MatmulLoops::<f64, 252, 512, 240, 14, 2, 8>::aodmao2rho_loop_parallel::<3>(rho, dm, ao_lhs, ao_rhs, strides, shape, tab, ldtab);
}

#[test]
pub fn test_aodmao2rho() {
    let nao: usize = 2228;
    let ngrid: usize = 16384;

    let ao_lhs: Vec<f64> = (0..nao * ngrid).map(|x| (x as f64).sin()).collect();
    let ao_rhs: Vec<f64> = (0..nao * ngrid).map(|x| (x as f64).cos()).collect();
    let dm: Vec<f64> = (0..nao * nao).map(|x| (x as f64 * 0.38 + 1.2).sin()).collect();

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
    let mut rho = vec![0.0f64; ngrid];
    aodmao2rho_anyway(&mut rho, &dm, &ao_lhs, &ao_rhs, [nao, ngrid], [1, nao, ngrid], &non0tab, nblk);
    let elapsed = time.elapsed();
    println!("Elapsed time (aodm2rho): {:.3?}", elapsed);

    // use rstsr::prelude::*;
    // let device = DeviceOpenBLAS::default();
    // let ao_rhs_tsr = rt::asarray((&ao_rhs, [nao, ngrid], &device));
    // let dm_tsr = rt::asarray((&dm, [nao, nao], &device));

    // let time = std::time::Instant::now();
    // let dm_dot_ao = &dm_tsr % &ao_rhs_tsr; // (nao, ngrid)
    // println!("Elapsed time (dm_dot_ao reference): {:.3?}", time.elapsed());

    // let time = std::time::Instant::now();
    // let rho_ref = (0..ngrid)
    //     .into_par_iter()
    //     .map(|g| {
    //         let mut sum = 0.0;
    //         for p in 0..nao {
    //             sum += ao_lhs[p * ngrid + g] * dm_dot_ao.raw()[p * ngrid +
    // g];         }
    //         sum
    //     })
    //     .collect::<Vec<f64>>();
    // println!("Elapsed time (ao_to_rho reference): {:.3?}", time.elapsed());

    // println!("rho     {:10.3?}", &rho[..10.min(ngrid)]);
    // println!("rho_ref {:10.3?}", &rho_ref[..10.min(ngrid)]);

    // let ao_lhs_tsr = rt::asarray((&ao_lhs, [nao, ngrid], &device));
    // let rho_ref = (dm_dot_ao * ao_lhs_tsr).sum_axes(0);
    // println!("rho_ref (rstsr) {:10.3?}", &rho_ref.raw()[..10.min(ngrid)]);

    // let diff = rt::asarray((&rho, &device)) - rho_ref.view();
    // let err = diff.view().abs().max();
    // println!("Max abs error: {:.6e}", err);
}
