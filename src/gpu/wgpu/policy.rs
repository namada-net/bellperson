//! Runtime policy flags for wgpu proving paths.

#[inline]
pub fn require_gpu() -> bool {
    std::env::var("BELLMAN_WGPU_REQUIRE_GPU").ok().as_deref() == Some("1")
}

#[inline]
pub fn use_gpu_fft() -> bool {
    std::env::var("BELLMAN_WGPU_USE_GPU_FFT").ok().as_deref() == Some("1")
}

#[inline]
pub fn use_gpu_msm() -> bool {
    match std::env::var("BELLMAN_WGPU_USE_GPU_MSM") {
        Ok(v) => v == "1",
        Err(_) => true,
    }
}

#[inline]
pub fn validate_gpu() -> bool {
    std::env::var("BELLMAN_WGPU_VALIDATE").ok().as_deref() == Some("1")
}
