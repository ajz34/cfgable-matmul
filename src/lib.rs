#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![doc = include_str!("principle-loops.md")]

pub mod impl_matmul;
pub mod impl_matmul_mask_k_for_b;
pub mod naive_simd;
pub mod prelude;
pub mod structs;

#[allow(unused_imports)]
use crate::prelude::*;

#[allow(unused_imports)]
use impl_matmul_mask_k_for_b::MatmulMicroKernelMaskKForBAPI;
