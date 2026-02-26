//! GPU field arithmetic for finite fields used in bellperson.
//!
//! Each field has its own sub-module containing the WGSL shader constants,
//! the Rust driver (`run_fr_op_gpu`), and correctness tests.
//!
//! # Extending to new fields
//!
//! Add a new `pub mod <field_name>;` here and implement the same
//! `run_*_op_gpu` / `*Op` API pattern in the new module.  The WGSL library
//! shader goes into `../shaders/<field_name>_arith.wgsl`.

pub mod bls12_381_fr;

#[cfg(not(target_arch = "wasm32"))]
pub use bls12_381_fr::{FrOp, run_fr_op_gpu};
