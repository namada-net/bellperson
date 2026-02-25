#![cfg(feature = "wgpu")]

use nam_bellperson::gpu::{require_gpu, use_gpu_fft, use_gpu_msm, validate_gpu};
use temp_env::{with_var, with_vars};

#[test]
fn wgpu_policy_defaults_and_overrides() {
    with_vars(
        vec![
            ("BELLMAN_WGPU_REQUIRE_GPU", None::<&str>),
            ("BELLMAN_WGPU_USE_GPU_FFT", None::<&str>),
            ("BELLMAN_WGPU_USE_GPU_MSM", None::<&str>),
            ("BELLMAN_WGPU_VALIDATE", None::<&str>),
        ],
        || {
            assert!(!require_gpu());
            assert!(!use_gpu_fft());
            assert!(use_gpu_msm());
            assert!(!validate_gpu());
        },
    );

    with_var("BELLMAN_WGPU_REQUIRE_GPU", Some("1"), || {
        assert!(require_gpu());
    });

    with_var("BELLMAN_WGPU_USE_GPU_FFT", Some("1"), || {
        assert!(use_gpu_fft());
    });

    with_var("BELLMAN_WGPU_USE_GPU_MSM", Some("0"), || {
        assert!(!use_gpu_msm());
    });

    with_var("BELLMAN_WGPU_VALIDATE", Some("1"), || {
        assert!(validate_gpu());
    });
}
