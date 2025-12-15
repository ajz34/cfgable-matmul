# Cheatsheet for matrix multiplication with SIMD and loops

This section summarizes the key principles and techniques for implementing efficient matrix multiplication using SIMD (Single Instruction, Multiple Data) and loop optimizations in Rust.

We refer BLIS algorithms for detailed reference: [LAFF-On Programming for High Performance](https://www.cs.utexas.edu/~flame/laff/pfhp/LAFF-On-PfHP.html).

## Introduction: Original 5-loop 

First we will introduce the original 5-loop matrix multiplication BLIS algorithm. We use some of the notations and thoughts from BLIS algorithm.

<!-- following code insert a graph with width 400px and centered -->
<center>
<img src="https://www.cs.utexas.edu/~flame/laff/pfhp/images/Week3/BLISPicturePack.png" width="600px" alt="BLISPicture5Loop.png"/>
</center>

<div class="warning">

**Row-major Order Notes**

This cheatsheet and the code implementations assume that matrices are stored in row-major order. So the notations are different to BLIS original implementation and LAFF book, which assume column-major order.

In the following, we will use row-major order notations consistently. So dimension and matrix multiplication order are different to BLIS original implementation.

</div>
