# Cheatsheet for Cfgable-Matmul

This section summarizes the key principles and techniques for implementing efficient matrix multiplication using SIMD (Single Instruction, Multiple Data) and loop optimizations in Rust.

We refer BLIS algorithms for detailed reference: [LAFF-On Programming for High Performance](https://www.cs.utexas.edu/~flame/laff/pfhp/LAFF-On-PfHP.html).

## Introduction: Original BLIS 5-loop 

First we will introduce the original 5-loop matrix multiplication BLIS algorithm. We use some of the notations and thoughts from BLIS algorithm.

<!-- following code insert a graph with width 400px and centered -->
<center>
<img src="https://www.cs.utexas.edu/~flame/laff/pfhp/images/Week3/BLISPicturePack.png" width="600px" alt="BLISPicture5Loop.png"/>
</center>

<div class="warning">

**Row-major Order Notes**

This cheatsheet and the code implementations assume that matrices are stored in row-major order. So the notations are different to BLIS original implementation and LAFF book, which assume column-major order.

In the following, we will use row-major order notations consistently. So dimension and matrix multiplication order are different to BLIS original implementation. The parameters `MC, KC, NC, MR, NR` for BLIS is similar to `NC, KC, MC, NR, MR` in our code. The matrix multiplication $C_{ji} = \sum_p B_{jp} A_{pi}$ for BLIS is $C_{ij} = \sum_p A_{ip} B_{pj}$ in our code.

</div>

## Loop 0: Macro-kernel

Micro-kernel is a very small matrix multiplication kernel:

$$
C_{ij} = \sum_{p} A_{pi} B_{pj}
$$

This should be implemented by trait function [`MatmulMicroKernelAPI::microkernel`].

| parameter | dimensionality or constraints | SKX/Zen4 f64 |
|--|--|--|
| `c` | (`MR`, `NR_LANE` * `LANE`) | (13, 16) |
| `a` | (`KC`, `MR`) | (256, 13) |
| `b` | (`KC`, `NR_LANE` * `LANE`) | (256, 16) |
| `kc` | `kc <= KC` | 256 |

Please note that 
