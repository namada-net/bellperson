#![cfg(feature = "wgpu")]

use blstrs::{G1Affine, Scalar};
use ff::Field;
use ff::PrimeField;

use ec_gpu_gen::threadpool::Worker;
use nam_bellperson::domain::EvaluationDomain;
use nam_bellperson::gpu::{
    LockedFftKernel, WgpuMultiexpKernel, fft_pipeline_completed, msm_pipeline_completed,
    msm_sum_dispatches, reset_fft_gpu_dispatches, reset_msm_gpu_dispatches,
};
use temp_env::with_vars;

#[test]
fn pipeline_completion_flags_reset_and_progress() {
    reset_fft_gpu_dispatches();
    reset_msm_gpu_dispatches();

    assert!(!fft_pipeline_completed());
    assert!(!msm_pipeline_completed());

    // Only MSM can be touched directly in this unit-level test.
    if let Ok(mut kern) = WgpuMultiexpKernel::<G1Affine>::new(true) {
        let scalars = vec![Scalar::ONE.to_repr(); 8];
        let _ = kern.touch_gpu::<Scalar>(&scalars);
        assert!(msm_pipeline_completed());
        assert!(msm_sum_dispatches() >= 1);
    }
}

#[test]
fn fft_pipeline_completion_progresses() {
    reset_fft_gpu_dispatches();
    assert!(!fft_pipeline_completed());

    with_vars(
        vec![
            ("BELLMAN_NO_GPU", Some("0")),
            ("BELLMAN_WGPU_VALIDATE", Some("1")),
            ("BELLMAN_WGPU_USE_GPU_FFT", Some("1")),
        ],
        || {
            let worker = Worker::new();
            let mut coeffs = vec![Scalar::ZERO; 8];
            coeffs[1] = Scalar::ONE;
            coeffs[2] = Scalar::ONE + Scalar::ONE;

            let mut domain = EvaluationDomain::from_coeffs(coeffs).expect("domain");
            let mut kern = Some(LockedFftKernel::<Scalar>::new(true));
            let _ = domain.fft(&worker, &mut kern);
            assert!(fft_pipeline_completed());
        },
    );
}
