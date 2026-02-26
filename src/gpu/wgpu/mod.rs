//! WebGPU (wgpu) compute backend for bellperson.
//!
//! This module provides GPU acceleration using the wgpu library, which supports
//! both native platforms (Vulkan, Metal, DX12) and web browsers via WebGPU.
//!
//! # Feature Flag
//!
//! Enable this backend with the `wgpu` feature:
//!
//! ```toml
//! bellperson = { version = "...", features = ["groth16", "wgpu"] }
//! ```
//!
//! # Usage
//!
//! The wgpu backend integrates with bellperson's existing GPU abstraction layer.
//! It provides FFT and multiexp kernels similar to the CUDA and OpenCL backends.
//!
//! # Browser Support
//!
//! When compiling for `wasm32-unknown-unknown`, this backend uses WebGPU.
//! The prover can then run directly in web browsers.
//! Supports both Chrome (113+) and Firefox (126+).

mod device;
mod error;
mod fft;
pub(crate) mod field_arith;
mod multiexp;
mod packing;
mod policy;

pub use self::device::{
    WgpuDevice, WgpuEngine, clear_wgpu_backends_override, set_wgpu_adapter_name_filter,
    set_wgpu_backends_from_str,
};

#[cfg(target_arch = "wasm32")]
pub use self::device::{
    wasm_clear_wgpu_backends_override, wasm_set_wgpu_adapter_name_filter, wasm_set_wgpu_backends,
};
pub use self::error::{WgpuError, WgpuResult};
#[cfg(not(target_arch = "wasm32"))]
pub use self::field_arith::{FrOp, run_fr_op_gpu};
pub use self::fft::{
    WgpuFftKernel, fft_bitrev_dispatches, fft_gpu_dispatches, fft_pipeline_completed,
    fft_stage_dispatches, reset_fft_gpu_dispatches,
};
pub use self::multiexp::{
    WgpuMultiexpKernel, msm_bucket_dispatches, msm_decomp_dispatches, msm_gpu_dispatches,
    msm_pipeline_completed, msm_sum_dispatches, msm_window_dispatches, reset_msm_gpu_dispatches,
};
pub use self::policy::{require_gpu, use_gpu_fft, use_gpu_msm, validate_gpu};

use std::marker::PhantomData;

use ff::Field;
use group::prime::PrimeCurveAffine;

use crate::gpu::GpuName;

/// A locked FFT kernel that ensures only one process uses the GPU at a time.
/// This type provides the same API as the CUDA/OpenCL LockedFftKernel.
pub struct LockedFftKernel<F> {
    _phantom: PhantomData<F>,
    priority: bool,
    kernel: Option<WgpuFftKernel>,
}

impl<F> LockedFftKernel<F>
where
    F: Field + GpuName,
{
    pub fn new(priority: bool) -> Self {
        Self {
            _phantom: PhantomData,
            priority,
            kernel: None,
        }
    }

    pub fn with<Fun, R>(&mut self, mut f: Fun) -> WgpuResult<R>
    where
        Fun: FnMut(&mut WgpuFftKernel) -> WgpuResult<R>,
    {
        if self.kernel.is_none() {
            self.kernel = Some(WgpuFftKernel::new(self.priority)?);
        }
        f(self.kernel.as_mut().unwrap())
    }
}

/// A locked multiexp kernel that ensures only one process uses the GPU at a time.
/// This type provides the same API as the CUDA/OpenCL LockedMultiexpKernel.
pub struct LockedMultiexpKernel<G> {
    priority: bool,
    kernel: Option<WgpuMultiexpKernel<G>>,
}

impl<G> LockedMultiexpKernel<G>
where
    G: PrimeCurveAffine + GpuName,
{
    pub fn new(priority: bool) -> Self {
        Self {
            priority,
            kernel: None,
        }
    }

    pub fn with<Fun, R>(&mut self, mut f: Fun) -> WgpuResult<R>
    where
        Fun: FnMut(&mut WgpuMultiexpKernel<G>) -> WgpuResult<R>,
    {
        if self.kernel.is_none() {
            self.kernel = Some(WgpuMultiexpKernel::new(self.priority)?);
        }
        f(self.kernel.as_mut().unwrap())
    }
}
