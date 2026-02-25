#![cfg(feature = "wgpu")]

use blstrs::{G1Affine, Scalar};
use ff::{Field, PrimeField};

use ec_gpu_gen::threadpool::Worker;
use nam_bellperson::domain::EvaluationDomain;
use nam_bellperson::gpu::{
    LockedFftKernel, WgpuEngine, WgpuMultiexpKernel, clear_wgpu_backends_override,
    fft_bitrev_dispatches, fft_gpu_dispatches, fft_pipeline_completed, fft_stage_dispatches,
    msm_bucket_dispatches, msm_decomp_dispatches, msm_gpu_dispatches, msm_pipeline_completed,
    msm_sum_dispatches, msm_window_dispatches, reset_fft_gpu_dispatches, reset_msm_gpu_dispatches,
    set_wgpu_adapter_name_filter, set_wgpu_backends_from_str,
};
use temp_env::with_vars;

fn should_run_wgpu_backend_tests() -> bool {
    std::env::var("BELLMAN_RUN_WGPU_TESTS").ok().as_deref() == Some("1")
}

fn require_engine_or_skip(strict: bool) -> bool {
    match WgpuEngine::new() {
        Ok(_) => true,
        Err(e) => {
            if strict {
                panic!("wgpu engine init failed in strict mode: {e}");
            }
            eprintln!("Skipping wgpu backend test; adapter unavailable: {e}");
            false
        }
    }
}

#[test]
fn wgpu_fft_dispatches_on_native_host() {
    let strict = should_run_wgpu_backend_tests();

    with_vars(
        vec![
            ("BELLMAN_NO_GPU", Some("0")),
            ("BELLMAN_WGPU_VALIDATE", Some("1")),
            ("BELLMAN_WGPU_USE_GPU_FFT", Some("1")),
        ],
        || {
            if let Ok(backends) = std::env::var("WGPU_BACKEND") {
                set_wgpu_backends_from_str(&backends);
            }
            set_wgpu_adapter_name_filter(std::env::var("WGPU_ADAPTER_NAME").ok().as_deref());
            if !require_engine_or_skip(strict) {
                clear_wgpu_backends_override();
                set_wgpu_adapter_name_filter(None);
                return;
            }

            // Build a tiny domain and run FFT path with wgpu feature.
            let worker = Worker::new();
            let mut coeffs = vec![Scalar::ZERO; 8];
            coeffs[1] = Scalar::ONE;
            coeffs[2] = Scalar::ONE + Scalar::ONE;

            let mut domain = EvaluationDomain::from_coeffs(coeffs).expect("domain must build");
            let mut kern = Some(LockedFftKernel::<Scalar>::new(true));

            reset_fft_gpu_dispatches();
            let _ = domain.fft(&worker, &mut kern);
            let dispatches = fft_gpu_dispatches();

            assert!(dispatches >= 4, "expected staged wgpu FFT dispatches");
            assert!(fft_stage_dispatches() >= 3, "expected staged FFT passes");
            assert!(
                fft_bitrev_dispatches() >= 1,
                "expected FFT bit-reversal pass"
            );
            assert!(
                fft_pipeline_completed(),
                "expected full FFT pipeline completion"
            );
            clear_wgpu_backends_override();
            set_wgpu_adapter_name_filter(None);
        },
    );
}

#[test]
fn wgpu_msm_dispatches_on_native_host() {
    let strict = should_run_wgpu_backend_tests();

    with_vars(
        vec![
            ("BELLMAN_NO_GPU", Some("0")),
            ("BELLMAN_WGPU_VALIDATE", Some("1")),
            ("BELLMAN_WGPU_USE_GPU_FFT", Some("1")),
        ],
        || {
            if let Ok(backends) = std::env::var("WGPU_BACKEND") {
                set_wgpu_backends_from_str(&backends);
            }
            set_wgpu_adapter_name_filter(std::env::var("WGPU_ADAPTER_NAME").ok().as_deref());
            if !require_engine_or_skip(strict) {
                clear_wgpu_backends_override();
                set_wgpu_adapter_name_filter(None);
                return;
            }

            let mut kern = WgpuMultiexpKernel::<G1Affine>::new(true).expect("wgpu kernel init");
            let scalars = vec![Scalar::ONE.to_repr(); 16];

            reset_msm_gpu_dispatches();
            let _ = kern.touch_gpu::<Scalar>(&scalars);
            let dispatches = msm_gpu_dispatches();

            assert!(dispatches >= 4, "expected staged wgpu MSM dispatches");
            assert!(
                msm_decomp_dispatches() >= 1,
                "expected MSM decomp pass dispatch"
            );
            assert!(
                msm_window_dispatches() >= 1,
                "expected MSM window pass dispatch"
            );
            assert!(
                msm_bucket_dispatches() >= 1,
                "expected MSM bucket pass dispatch"
            );
            assert!(msm_sum_dispatches() >= 1, "expected MSM sum pass dispatch");
            assert!(
                msm_pipeline_completed(),
                "expected full MSM pipeline completion"
            );
            clear_wgpu_backends_override();
            set_wgpu_adapter_name_filter(None);
        },
    );
}
