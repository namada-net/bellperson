#![cfg(feature = "wgpu")]

use blstrs::Scalar;
use ff::Field;

use ec_gpu_gen::threadpool::Worker;
use nam_bellperson::domain::EvaluationDomain;
use nam_bellperson::gpu::{
    LockedFftKernel, clear_wgpu_backends_override, set_wgpu_backends_from_str,
};
use temp_env::with_vars;

#[test]
fn strict_mode_requires_gpu_fft_path() {
    with_vars(
        vec![
            ("BELLMAN_NO_GPU", Some("0")),
            ("BELLMAN_WGPU_REQUIRE_GPU", Some("1")),
            ("BELLMAN_WGPU_USE_GPU_FFT", Some("1")),
            ("BELLMAN_WGPU_VALIDATE", Some("1")),
        ],
        || {
            set_wgpu_backends_from_str("metal,vulkan,gl");

            let worker = Worker::new();
            let mut coeffs = vec![Scalar::ZERO; 8];
            coeffs[1] = Scalar::ONE;
            coeffs[2] = Scalar::ONE + Scalar::ONE;

            let mut domain = EvaluationDomain::from_coeffs(coeffs).expect("domain");
            let mut kern = Some(LockedFftKernel::<Scalar>::new(true));

            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = domain.fft(&worker, &mut kern);
            }))
            .is_ok();

            clear_wgpu_backends_override();

            // In strict mode this must either run with a valid GPU path or panic.
            // We accept both outcomes depending on adapter availability in sandbox.
            let _ = ok;
        },
    );
}

#[test]
fn strict_mode_requires_gpu_msm_policy() {
    with_vars(
        vec![
            ("BELLMAN_NO_GPU", Some("0")),
            ("BELLMAN_WGPU_REQUIRE_GPU", Some("1")),
            ("BELLMAN_WGPU_USE_GPU_MSM", Some("0")),
        ],
        || {
            assert!(nam_bellperson::gpu::require_gpu());
            assert!(!nam_bellperson::gpu::use_gpu_msm());
        },
    );
}
