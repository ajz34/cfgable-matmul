# Configurable Matmul

**A Rust Configurable Architecture-Dependent Matrix Multiplication (with its application to specialized sparse DFT AO-DM contraction)**

This is not a finished project.

This program uses algorithm similar to BLIS's loop  (see also [LAFF on programming for high performance](https://www.cs.utexas.edu/~flame/laff/pfhp/)).

You can call a somehow fast (not extremely fast compared to [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/), but is probably as-same-fast-as [Faer](https://github.com/sarah-quinones/faer-rs/).

For example of a matrix multiplication of size (3527, 6581) x (6581, 9583) on AMD Ryzen 7945HX CPU (16 cores, 1.1 TFLOP/sec):

| Library | Time | CPU efficiency |
|--|--|--|
| cfgable-matmul (this) | 530 ms | 780 GFLOP/sec (71%) |
| Faer | 530 ms | 780 GFLOP/sec (71%) |
| OpenBLAS | 470 ms | 880 GFLOP/sec (80%) |

You can use the following function on Zen4 CPUs (please note this is in row-major order, not the column-major order in the original BLIS algorithm):

```rust
MatmulLoops::<
    f64,  // type
    252,  // parallel   block `MC`, multiple of `MR`
    512,  // L1/2 cache block `KC`
    240,  // L1/2 cache block `NC`, multiple of `NR`
    14,   // register block `MR`, `(MR + 1) * NR_LANE` should fit in registers (less than 32)
    2,    // register block `NR_LANE` (= NR / LANE, 2 for Intel and Zen4/5, 1 for Zen1-4)
    8,    // SIMD lane `LANE` (8 x f64 for AVX-512)
    2360, // L3   cache block `MB`
>::matmul_loop_macro_mb(c, a, b, m, n, k, lda, ldb, ldc, transa, transb);
```

It should work on other CPUs with SIMD support (AVX2, AVX-512, ARM Neon), but
- must use `RUSTFLAGS="-C target-cpu=native"` or similar flags `RUSTFLAGS="-C target-feature=+avx2"` to explicitly enable LLVM auto-vectorization (`--release` or `opt-level=3` is not enough);
- strongly additionally recommend use `RUSTFLAGS="-C llvm-args=-fp-contract=fast"` to enable fused-multiply-add (FMA) optimization. Current microkernel implementation use `c += a * b` pattern which can be optimized to FMA by LLVM with zmm registers; with only xmm registers, this microkernel is not that efficient, but still better than using `f64::mul_add` **if architecture does not support FMA of f64 and requires something like `libm`**.
- the configurable parameters may need to be adjusted for better performance (SIMD lane fit, size of L1/2/3 cache).

This serves as a reference implementation for fast matmulplication, without explicit use of intrinsics/assembly or nightly rust. The implementation code [impl_matmul.rs](src/impl_matmul.rs) only have about 250 lines of code.

---

As a building block, this also provides specialized AO-DM-AO contraction (to generate density grids) for DFT calculations without using sparse tabulation for basis functions (see [impl_aodmao2rho.rs](src/impl_aodmao2rho.rs)):

$$
\rho_{g}^\mathbb{A} = \sum_{\mu \nu} \phi_{\mu g} D_{\mu \nu}^\mathbb{A} \phi_{\nu g}
$$

where the tabulated sparse mask is similar (not exactly the same) to the format of PySCF's non0tab.

As an example, for taxol molecule with def2-TZVP basis set (2228 basis functions) and 16384 grid points and 30% sparsity, given 3 density matrices ($\mathbb{A}$ have three components), the AO-DM-AO contraction can be done in about 120 ms (340 GFLOP/sec) on AMD Ryzen 7945HX CPU (16 cores). This is about 5 times faster than naive matmul 600 msec and then partial reduce 40 msec (710 GFLOP/sec in total).
