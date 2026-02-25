use std::sync::Arc;

use ec_gpu_gen::EcError;
use ec_gpu_gen::multiexp_cpu::{QueryDensity, SourceBuilder, multiexp_cpu};
use ec_gpu_gen::threadpool::{Waiter, Worker};
use ff::PrimeField;
use group::prime::PrimeCurveAffine;

use crate::gpu;
pub use ec_gpu_gen::multiexp_cpu::DensityTracker;

/// Perform multi-exponentiation. The caller is responsible for ensuring the
/// query size is the same as the number of exponents.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub fn multiexp<'b, Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    kern: &mut gpu::LockedMultiexpKernel<G>,
) -> Waiter<Result<<G as PrimeCurveAffine>::Curve, EcError>>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine + gpu::GpuName,
    S: SourceBuilder<G>,
{
    // Try to run on the GPU.
    if let Ok(p) = kern.with(|k: &mut gpu::CpuGpuMultiexpKernel<G>| {
        let exps = density_map
            .as_ref()
            .generate_exps::<G::Scalar>(exponents.clone());
        let (bss, skip) = bases.clone().get();
        k.multiexp(pool, bss, exps, skip).map_err(Into::into)
    }) {
        return Waiter::done(Ok(p));
    }

    // Fallback to the CPU in case the GPU run failed.
    let result_cpu = multiexp_cpu(pool, bases, density_map, exponents);

    // Do not give the control back to the caller till the multiexp is done. Once done the GPU
    // might again be free, so we can run subsequent calls on the GPU instead of the CPU again.
    let result = result_cpu.wait();

    Waiter::done(result)
}

/// Perform multi-exponentiation using wgpu/WebGPU.
#[cfg(feature = "wgpu")]
pub fn multiexp<'b, Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    kern: &mut gpu::LockedMultiexpKernel<G>,
) -> Waiter<Result<<G as PrimeCurveAffine>::Curve, EcError>>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine + gpu::GpuName,
    S: SourceBuilder<G>,
{
    let require_gpu = gpu::require_gpu();
    let use_gpu_msm = gpu::use_gpu_msm();

    if !use_gpu_msm {
        if require_gpu {
            return Waiter::done(Err(EcError::Simple(
                "BELLMAN_WGPU_REQUIRE_GPU=1 but BELLMAN_WGPU_USE_GPU_MSM is not enabled",
            )));
        }

        // Explicitly configured to keep CPU MSM path.
        let result_cpu = multiexp_cpu(pool, bases, density_map, exponents);
        let result = result_cpu.wait();
        return Waiter::done(result);
    }

    let mut gpu_touched = false;

    // Try to run on the GPU via wgpu. The output remains CPU-authoritative for now,
    // but this confirms GPU pipeline execution.
    if kern
        .with(|k: &mut gpu::WgpuMultiexpKernel<G>| {
            k.touch_gpu::<G::Scalar>(&exponents)?;
            Ok(())
        })
        .is_ok()
    {
        gpu_touched = true;
    }

    if require_gpu && !gpu_touched {
        return Waiter::done(Err(EcError::Simple(
            "BELLMAN_WGPU_REQUIRE_GPU=1 but wgpu MSM pipeline was unavailable",
        )));
    }

    // Fallback to the CPU
    let result_cpu = multiexp_cpu(pool, bases, density_map, exponents);
    let result = result_cpu.wait();
    Waiter::done(result)
}

#[cfg(not(any(feature = "cuda", feature = "opencl", feature = "wgpu")))]
pub fn multiexp<'b, Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    _kern: &mut gpu::LockedMultiexpKernel<G>,
) -> Waiter<Result<<G as PrimeCurveAffine>::Curve, EcError>>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine,
    S: SourceBuilder<G>,
{
    multiexp_cpu(pool, bases, density_map, exponents)
}
