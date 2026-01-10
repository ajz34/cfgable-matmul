//! Naive matrix multiplication (without very delicated SIMD optimizations).
//!
//! Following example is 800x4000 multiplied by 4000x4000 matrix multiplication
//! benchmark on Ryzen 7945HX (Zen4).
//!
//! | Scheme | Time |
//! |--------|-----:|
//! | serial IJP              |  34.95 sec |
//! | serial IPJ              |   4.00 sec |
//! | block    IJ, serial IPJ |   2.89 sec |
//! | parallel IJ, serial IPJ |   1.50 sec |
//! | cfgable_matmul          |  40   msec |
//! | (exceed registers)      |  78   msec |
//! | OpenBLAS                |  31   msec |
//! | Faer                    |  46   msec |
//!
//! Following example is 4000x4000 multiplied by 4000x4000 matrix multiplication
//! benchmark on Ryzen 7945HX (Zen4).
//!
//! | Scheme | Time |
//! |--------|-----:|
//! | serial IJP              | 173.19 sec |
//! | serial IPJ              |  19.61 sec |
//! | block    IJ, serial IPJ |  14.48 sec |
//! | parallel IJ, serial IPJ | ~ 6    sec |
//! | cfgable_matmul          | 168   msec |
//! | (exceed registers)      | 358   msec |
//! | OpenBLAS                | 133   msec |
//! | Faer                    | 152   msec |

use rayon::prelude::*;
use rstest::rstest;

#[inline]
#[allow(clippy::mut_from_ref)]
unsafe fn cast_mut_slice<T>(slc: &[T]) -> &mut [T] {
    let len = slc.len();
    let ptr = slc.as_ptr() as *mut T;
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

fn prepare_matrices(m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..m * k).into_par_iter().map(|x| (x as f64).sin()).collect();
    let b: Vec<f64> = (0..k * n).into_par_iter().map(|x| (x as f64).cos()).collect();
    (a, b)
}

#[rstest]
#[case(800, 4000, 4000)]
#[case(4000, 4000, 4000)]
fn test_naive_matmul_ipj(#[case] m: usize, #[case] n: usize, #[case] k: usize) {
    let (a, b) = prepare_matrices(m, n, k);

    let time = std::time::Instant::now();
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                let b_pj = b[p * n + j];
                c[i * n + j] += a_ip * b_pj;
            }
        }
    }
    println!("{m}x{n} @ {k}x{n}: {:.3?}.", time.elapsed());
}

#[rstest]
#[case(800, 4000, 4000)]
#[case(4000, 4000, 4000)]
fn test_naive_matmul_ijp(#[case] m: usize, #[case] n: usize, #[case] k: usize) {
    let (a, b) = prepare_matrices(m, n, k);

    let time = std::time::Instant::now();
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let c_ij = &mut c[i * n + j];
            for p in 0..k {
                let a_ip = a[i * k + p];
                let b_pj = b[p * n + j];
                *c_ij += a_ip * b_pj;
            }
        }
    }
    println!("{m}x{n} @ {k}x{n}: {:.3?}.", time.elapsed());
}

#[rstest]
#[case(800, 4000, 4000)]
#[case(4000, 4000, 4000)]
fn test_matmul_parblk_ij(#[case] m: usize, #[case] n: usize, #[case] k: usize) {
    const MB: usize = 128;
    const NB: usize = 256;

    let (a, b) = prepare_matrices(m, n, k);

    let time = std::time::Instant::now();
    let c = vec![0.0f64; m * n];
    let mblk = m.div_ceil(MB);
    let nblk = n.div_ceil(NB);

    (0..mblk * nblk).into_par_iter().for_each(|task_id| {
        let c = unsafe { cast_mut_slice(&c) };
        let bi = task_id / nblk;
        let bj = task_id % nblk;
        let i_start = bi * MB;
        let j_start = bj * NB;
        let i_end = ((bi + 1) * MB).min(m);
        let j_end = ((bj + 1) * NB).min(n);

        for i in i_start..i_end {
            for p in 0..k {
                let a_ip = a[i * k + p];
                for j in j_start..j_end {
                    let b_pj = b[p * n + j];
                    c[i * n + j] += a_ip * b_pj;
                }
            }
        }
    });
    println!("{m}x{n} @ {k}x{n}: {:.3?}.", time.elapsed());
}

#[rstest]
#[case(800, 4000, 4000)]
#[case(4000, 4000, 4000)]
fn test_matmul_serblk_ij(#[case] m: usize, #[case] n: usize, #[case] k: usize) {
    const MB: usize = 128;
    const NB: usize = 256;

    let (a, b) = prepare_matrices(m, n, k);

    let time = std::time::Instant::now();
    let mut c = vec![0.0f64; m * n];

    for i_start in (0..m).step_by(MB) {
        for j_start in (0..n).step_by(NB) {
            let i_end = (i_start + MB).min(m);
            let j_end = (j_start + NB).min(n);

            for i in i_start..i_end {
                for p in 0..k {
                    let a_ip = a[i * k + p];
                    for j in j_start..j_end {
                        let b_pj = b[p * n + j];
                        c[i * n + j] += a_ip * b_pj;
                    }
                }
            }
        }
    }
    println!("{m}x{n} @ {k}x{n}: {:.3?}.", time.elapsed());
}

#[rstest]
#[case(800, 4000, 4000)]
#[case(4000, 4000, 4000)]
fn test_cfgable_matmul(#[case] m: usize, #[case] n: usize, #[case] k: usize) {
    let (a, b) = prepare_matrices(m, n, k);

    let time = std::time::Instant::now();
    use cfgable_matmul::structs::MatmulLoops;
    let mut c = vec![0.0f64; m * n];
    MatmulLoops::<f64, 252, 512, 240, 14, 2, 8>::matmul_loop_parallel_mnk_pack_a(&mut c, &a, &b, m, n, k, k, n, n, false, false);
    println!("{m}x{n} @ {k}x{n}: {:.3?}.", time.elapsed());
}

#[rstest]
#[case(800, 4000, 4000)]
#[case(4000, 4000, 4000)]
fn test_cfgable_matmul_exceed_registers(#[case] m: usize, #[case] n: usize, #[case] k: usize) {
    let (a, b) = prepare_matrices(m, n, k);

    let time = std::time::Instant::now();
    use cfgable_matmul::structs::MatmulLoops;
    let mut c = vec![0.0f64; m * n];
    MatmulLoops::<f64, 252, 512, 240, 21, 2, 8>::matmul_loop_parallel_mnk_pack_a(&mut c, &a, &b, m, n, k, k, n, n, false, false);
    println!("{m}x{n} @ {k}x{n}: {:.3?}.", time.elapsed());
}

#[rstest]
#[case(800, 4000, 4000)]
#[case(4000, 4000, 4000)]
fn test_openblas_via_rstsr(#[case] m: usize, #[case] n: usize, #[case] k: usize) {
    let (a, b) = prepare_matrices(m, n, k);

    let time = std::time::Instant::now();
    use rstsr::prelude::*;
    let device = DeviceOpenBLAS::default();
    let a_tsr = rt::asarray((&a, [m, k], &device));
    let b_tsr = rt::asarray((&b, [k, n], &device));
    let _c_tsr = core::hint::black_box(&a_tsr % &b_tsr);
    println!("{m}x{n} @ {k}x{n}: {:.3?}.", time.elapsed());
}

#[rstest]
#[case(800, 4000, 4000)]
#[case(4000, 4000, 4000)]
fn test_faer_via_rstsr(#[case] m: usize, #[case] n: usize, #[case] k: usize) {
    let (a, b) = prepare_matrices(m, n, k);

    let time = std::time::Instant::now();
    use rstsr::prelude::*;
    let device = DeviceFaer::default();
    let a_tsr = rt::asarray((&a, [m, k], &device));
    let b_tsr = rt::asarray((&b, [k, n], &device));
    let _c_tsr = core::hint::black_box(&a_tsr % &b_tsr);
    println!("{m}x{n} @ {k}x{n}: {:.3?}.", time.elapsed());
}
