mod error;

pub use self::error::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod locks;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::locks::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod multiexp;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::multiexp::CpuGpuMultiexpKernel;

#[cfg(feature = "wgpu")]
mod wgpu;

#[cfg(feature = "wgpu")]
pub use self::wgpu::*;

#[cfg(all(test, feature = "wgpu"))]
pub use self::wgpu::{
    fft_gpu_dispatches, msm_gpu_dispatches, reset_fft_gpu_dispatches, reset_msm_gpu_dispatches,
};

// Re-export locked kernel types for wgpu feature (same API as cuda/opencl)
// LockedFftKernel and LockedMultiexpKernel are used throughout the codebase

#[cfg(not(any(feature = "cuda", feature = "opencl", feature = "wgpu")))]
mod nogpu;

#[cfg(not(any(feature = "cuda", feature = "opencl", feature = "wgpu")))]
pub use self::nogpu::*;

// This is a hack, so that the same traits can be used for the GPU and non-GPU code path.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use ec_gpu::GpuName;
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
pub trait GpuName {}
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
impl GpuName for blstrs::G1Affine {}
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
impl GpuName for blstrs::G2Affine {}
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
impl GpuName for blstrs::Scalar {}
