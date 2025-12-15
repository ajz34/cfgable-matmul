#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!("principle-loops.md")]

pub mod impl_matmul;
pub mod naive_simd;
pub mod prelude;
pub mod structs;
