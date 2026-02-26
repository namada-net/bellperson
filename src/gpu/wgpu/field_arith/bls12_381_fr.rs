//! GPU field arithmetic for the BLS12-381 scalar field Fr.
//!
//! Uploads pairs of standard-form Fr elements to the GPU, dispatches the
//! WGSL field-arithmetic shader, reads back results, and returns them as
//! 8-word little-endian `u32` arrays — the same byte layout as
//! [`blstrs::Scalar::to_repr`].
//!
//! # Shader sources
//!
//! The shader is assembled at compile time from two WGSL files:
//!
//! * `bls12_381_fr_arith.wgsl` — pure WGSL library: Montgomery multiply,
//!   add, sub, conditional-subtract, neg.
//! * `bls12_381_fr_test.wgsl` — `@compute` entry point that dispatches one
//!   of the above operations per GPU thread.
//!
//! Both files reside in `src/gpu/wgpu/shaders/`.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

use crate::gpu::wgpu::device::WgpuEngine;
use crate::gpu::wgpu::error::{WgpuError, WgpuResult};

/// Operation codes understood by the test entry-point shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FrOp {
    Mul = 0,
    Add = 1,
    Sub = 2,
}

/// Combined WGSL source: library functions prepended to the entry-point.
const FR_TEST_SHADER: &str = concat!(
    include_str!("../shaders/bls12_381_fr_arith.wgsl"),
    "\n",
    include_str!("../shaders/bls12_381_fr_test.wgsl"),
);

/// Run a BLS12-381 Fr field operation on the GPU for each `(a, b)` pair.
///
/// Each element is represented as 8 little-endian `u32` words, matching
/// [`blstrs::Scalar::to_repr`]'s byte layout packed into 32-bit words.
///
/// Returns a [`Vec`] of 8-word results in the same encoding.
///
/// # Errors
///
/// Returns [`WgpuError`] if the GPU device cannot be initialised, a buffer
/// operation fails, or the readback mapping is rejected by the driver.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_fr_op_gpu(pairs: &[([u32; 8], [u32; 8])], op: FrOp) -> WgpuResult<Vec<[u32; 8]>> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    let engine = WgpuEngine::new()?;
    let device = engine.device_arc();
    let queue  = engine.queue_arc();

    let n = pairs.len() as u32;
    let buf_bytes = (n as usize) * 8 * std::mem::size_of::<u32>();

    // Flatten input pairs into contiguous word arrays.
    let mut a_words: Vec<u32> = Vec::with_capacity(n as usize * 8);
    let mut b_words: Vec<u32> = Vec::with_capacity(n as usize * 8);
    for (a, b) in pairs {
        a_words.extend_from_slice(a);
        b_words.extend_from_slice(b);
    }

    // ----- GPU buffers -------------------------------------------------------
    let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("bls12-381-fr-a"),
        size:               buf_bytes as u64,
        usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("bls12-381-fr-b"),
        size:               buf_bytes as u64,
        usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("bls12-381-fr-out"),
        size:               buf_bytes as u64,
        usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_data: [u32; 4] = [n, op as u32, 0, 0];
    let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("bls12-381-fr-params"),
        size:               std::mem::size_of::<[u32; 4]>() as u64,
        usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buf_a,      0, bytemuck::cast_slice(&a_words));
    queue.write_buffer(&buf_b,      0, bytemuck::cast_slice(&b_words));
    queue.write_buffer(&buf_params, 0, bytemuck::cast_slice(&params_data));

    // ----- Shader pipeline ---------------------------------------------------
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("bls12-381-fr-shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(FR_TEST_SHADER)),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label:   Some("bls12-381-fr-bgl"),
        entries: &[
            // binding 0 — in_a (storage, read)
            wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            },
            // binding 1 — in_b (storage, read)
            wgpu::BindGroupLayoutEntry {
                binding:    1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            },
            // binding 2 — out (storage, read_write)
            wgpu::BindGroupLayoutEntry {
                binding:    2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            },
            // binding 3 — params (uniform)
            wgpu::BindGroupLayoutEntry {
                binding:    3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("bls12-381-fr-pl"),
        bind_group_layouts:   &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:        Some("bls12-381-fr-pipeline"),
        layout:       Some(&pipeline_layout),
        module:       &shader,
        entry_point:  "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("bls12-381-fr-bg"),
        layout:  &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    // ----- Dispatch ----------------------------------------------------------
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bls12-381-fr-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("bls12-381-fr-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = n.div_ceil(64).max(1);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit([encoder.finish()]);
    }

    // ----- Readback ----------------------------------------------------------
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("bls12-381-fr-readback"),
        size:               buf_bytes as u64,
        usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bls12-381-fr-copy-encoder"),
        });
        encoder.copy_buffer_to_buffer(&buf_out, 0, &readback, 0, buf_bytes as u64);
        queue.submit([encoder.finish()]);
    }

    let (tx, rx) = mpsc::channel();
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);

    match rx.recv() {
        Ok(Ok(())) => {
            let data     = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&data);
            let mut results = Vec::with_capacity(n as usize);
            for chunk in raw.chunks_exact(8) {
                let mut elem = [0u32; 8];
                elem.copy_from_slice(chunk);
                results.push(elem);
            }
            drop(data);
            readback.unmap();
            Ok(results)
        }
        _ => Err(WgpuError::Simple("bls12-381-fr readback map failed")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use blstrs::Scalar;
    use ff::{Field, PrimeField};

    // Skip the whole test body gracefully when no GPU is available,
    // rather than failing CI environments that lack hardware.
    fn gpu_available() -> bool {
        WgpuEngine::new().is_ok()
    }

    // Convert a blstrs::Scalar to the 8-word little-endian u32 representation
    // used by the WGSL shaders (same as to_repr() byte layout packed as u32-LE).
    fn scalar_to_words(s: &Scalar) -> [u32; 8] {
        let repr  = s.to_repr();   // 32 bytes, little-endian
        let bytes = repr.as_ref();
        let mut words = [0u32; 8];
        for (i, chunk) in bytes.chunks(4).enumerate() {
            words[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        words
    }

    // Convert the 8-word result back to a blstrs::Scalar for comparison.
    // `Scalar::Repr` is `[u8; 32]` in nam-blstrs, with little-endian byte order.
    fn words_to_scalar(words: &[u32; 8]) -> Option<Scalar> {
        let mut bytes = [0u8; 32];
        for (i, &w) in words.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        Option::from(Scalar::from_repr(bytes))
    }

    // -----------------------------------------------------------------------
    // fr_add
    // -----------------------------------------------------------------------

    #[test]
    fn fr_add_zero_is_identity() {
        if !gpu_available() { return; }

        let cases: Vec<_> = [
            Scalar::ZERO,
            Scalar::ONE,
            Scalar::from(0xdeadbeef_u64),
        ]
        .iter()
        .map(|a| (scalar_to_words(a), scalar_to_words(&Scalar::ZERO)))
        .collect();

        let results = run_fr_op_gpu(&cases, FrOp::Add).expect("GPU fr_add failed");
        for ((a_w, _), result) in cases.iter().zip(results.iter()) {
            assert_eq!(result, a_w, "0 + a should equal a");
        }
    }

    #[test]
    fn fr_add_small_values() {
        if !gpu_available() { return; }

        let pairs: Vec<(Scalar, Scalar)> = vec![
            (Scalar::from(1u64),     Scalar::from(1u64)),
            (Scalar::from(2u64),     Scalar::from(3u64)),
            (Scalar::from(100u64),   Scalar::from(200u64)),
            (Scalar::from(u64::MAX), Scalar::from(u64::MAX)),
        ];

        let cases: Vec<_> = pairs.iter()
            .map(|(a, b)| (scalar_to_words(a), scalar_to_words(b)))
            .collect();
        let results = run_fr_op_gpu(&cases, FrOp::Add).expect("GPU fr_add failed");

        for ((a, b), result) in pairs.iter().zip(results.iter()) {
            let expected   = a + b;
            let expected_w = scalar_to_words(&expected);
            let result_s   = words_to_scalar(result).expect("result should be valid Fr");
            assert_eq!(
                result_s, expected,
                "fr_add mismatch: {:?} + {:?}; got {:?} expected {:?}",
                a, b, result, &expected_w,
            );
        }
    }

    #[test]
    fn fr_add_near_modulus() {
        if !gpu_available() { return; }

        // r - 1 + 1 should wrap to 0
        let r_minus_1 = Scalar::ZERO - Scalar::ONE;
        let pairs = vec![
            (scalar_to_words(&r_minus_1), scalar_to_words(&Scalar::ONE)),
            (scalar_to_words(&r_minus_1), scalar_to_words(&r_minus_1)),
        ];
        let results = run_fr_op_gpu(&pairs, FrOp::Add).expect("GPU fr_add failed");

        let expected_0 = r_minus_1 + Scalar::ONE;
        let expected_1 = r_minus_1 + r_minus_1;
        assert_eq!(results[0], scalar_to_words(&expected_0), "(r-1)+1 should wrap to 0");
        assert_eq!(results[1], scalar_to_words(&expected_1), "(r-1)+(r-1) mismatch");
    }

    // -----------------------------------------------------------------------
    // fr_sub
    // -----------------------------------------------------------------------

    #[test]
    fn fr_sub_self_is_zero() {
        if !gpu_available() { return; }

        let vals = [
            Scalar::ONE,
            Scalar::from(42u64),
            Scalar::from(0x1234_5678_9abc_def0_u64),
        ];
        let pairs: Vec<_> = vals.iter()
            .map(|v| (scalar_to_words(v), scalar_to_words(v)))
            .collect();
        let results = run_fr_op_gpu(&pairs, FrOp::Sub).expect("GPU fr_sub failed");

        for r in &results {
            assert_eq!(*r, [0u32; 8], "a - a should be zero");
        }
    }

    #[test]
    fn fr_sub_wraps_correctly() {
        if !gpu_available() { return; }

        // 0 - 1 = r - 1
        let pairs = vec![
            (scalar_to_words(&Scalar::ZERO), scalar_to_words(&Scalar::ONE)),
        ];
        let results = run_fr_op_gpu(&pairs, FrOp::Sub).expect("GPU fr_sub failed");

        let expected = Scalar::ZERO - Scalar::ONE;
        assert_eq!(results[0], scalar_to_words(&expected), "0 - 1 should be r-1");
    }

    // -----------------------------------------------------------------------
    // fr_mul
    // -----------------------------------------------------------------------

    #[test]
    fn fr_mul_by_zero_is_zero() {
        if !gpu_available() { return; }

        let vals = [Scalar::ONE, Scalar::from(42u64), -Scalar::ONE];
        let pairs: Vec<_> = vals.iter()
            .map(|v| (scalar_to_words(v), scalar_to_words(&Scalar::ZERO)))
            .collect();
        let results = run_fr_op_gpu(&pairs, FrOp::Mul).expect("GPU fr_mul failed");
        for r in &results {
            assert_eq!(*r, [0u32; 8], "a * 0 should be 0");
        }
    }

    #[test]
    fn fr_mul_by_one_is_identity() {
        if !gpu_available() { return; }

        let vals = [
            Scalar::ONE,
            Scalar::from(7u64),
            Scalar::from(0xdeadbeef_cafebabe_u64),
        ];
        let pairs: Vec<_> = vals.iter()
            .map(|v| (scalar_to_words(v), scalar_to_words(&Scalar::ONE)))
            .collect();
        let results = run_fr_op_gpu(&pairs, FrOp::Mul).expect("GPU fr_mul failed");

        for (v, result) in vals.iter().zip(results.iter()) {
            let result_s = words_to_scalar(result).expect("valid Fr");
            assert_eq!(result_s, *v, "a * 1 should be a, got {:?}", result);
        }
    }

    #[test]
    fn fr_mul_small_values() {
        if !gpu_available() { return; }

        let pairs: Vec<(Scalar, Scalar)> = vec![
            (Scalar::from(2u64),     Scalar::from(3u64)),
            (Scalar::from(10u64),    Scalar::from(10u64)),
            (Scalar::from(100u64),   Scalar::from(200u64)),
            (Scalar::from(u64::MAX), Scalar::from(u64::MAX)),
        ];

        let gpu_pairs: Vec<_> = pairs.iter()
            .map(|(a, b)| (scalar_to_words(a), scalar_to_words(b)))
            .collect();
        let results = run_fr_op_gpu(&gpu_pairs, FrOp::Mul).expect("GPU fr_mul failed");

        for ((a, b), result) in pairs.iter().zip(results.iter()) {
            let expected = a * b;
            let result_s = words_to_scalar(result).expect("valid Fr");
            assert_eq!(
                result_s, expected,
                "{:?} * {:?}: GPU={:?}, CPU={:?}",
                a, b, result, scalar_to_words(&expected),
            );
        }
    }

    #[test]
    fn fr_mul_commutativity() {
        if !gpu_available() { return; }

        let a = Scalar::from(0x1234_5678_9abc_def0_u64);
        let b = Scalar::from(0xfed_cba98_7654_3210_u64);
        let pairs = vec![
            (scalar_to_words(&a), scalar_to_words(&b)),
            (scalar_to_words(&b), scalar_to_words(&a)),
        ];
        let results = run_fr_op_gpu(&pairs, FrOp::Mul).expect("GPU fr_mul failed");
        assert_eq!(results[0], results[1], "fr_mul should be commutative");
    }

    #[test]
    fn fr_mul_associativity() {
        if !gpu_available() { return; }

        let a = Scalar::from(0x1111_1111_u64);
        let b = Scalar::from(0x2222_2222_u64);
        let c = Scalar::from(0x3333_3333_u64);

        // (a * b) * c  vs  a * (b * c) — both computed entirely on CPU
        let cpu_left  = (a * b) * c;
        let cpu_right = a * (b * c);
        assert_eq!(cpu_left, cpu_right, "CPU associativity sanity check");

        // Compute (a * b) on GPU, then feed result back to get ((a*b) * c)
        let step1 = run_fr_op_gpu(&[(scalar_to_words(&a), scalar_to_words(&b))], FrOp::Mul)
            .expect("GPU step1 failed");
        let ab_gpu = step1[0];

        let step2 = run_fr_op_gpu(&[(ab_gpu, scalar_to_words(&c))], FrOp::Mul)
            .expect("GPU step2 failed");
        let result_s = words_to_scalar(&step2[0]).expect("valid Fr");

        assert_eq!(result_s, cpu_left, "(a*b)*c GPU should match CPU");
    }

    #[test]
    fn fr_mul_distributivity() {
        if !gpu_available() { return; }

        // a * (b + c) == a*b + a*c
        let a = Scalar::from(7u64);
        let b = Scalar::from(0xabcd_ef01_u64);
        let c = Scalar::from(0x1234_5678_u64);

        let b_plus_c = b + c;
        let lhs_pairs = vec![(scalar_to_words(&a), scalar_to_words(&b_plus_c))];
        let lhs_gpu = run_fr_op_gpu(&lhs_pairs, FrOp::Mul).expect("lhs GPU failed");

        let mul_pairs = vec![
            (scalar_to_words(&a), scalar_to_words(&b)),
            (scalar_to_words(&a), scalar_to_words(&c)),
        ];
        let muls    = run_fr_op_gpu(&mul_pairs, FrOp::Mul).expect("mul GPU failed");
        let rhs_gpu = run_fr_op_gpu(&[(muls[0], muls[1])], FrOp::Add).expect("add GPU failed");

        assert_eq!(lhs_gpu[0], rhs_gpu[0], "distributivity: a*(b+c) != a*b + a*c");
    }

    /// Comprehensive battery comparing GPU results against blstrs::Scalar
    /// for a range of deterministic pseudo-random field elements.
    #[test]
    fn fr_mul_exhaustive_vs_cpu() {
        if !gpu_available() { return; }

        let seeds: Vec<u64> = vec![
            0, 1, 2, 3, 7, 31, 127, 255,
            0xdead_beef, 0xcafe_babe,
            0x1234_5678_9abc_def0,
            0xffff_ffff_ffff_fffe,
            0xffff_ffff_ffff_ffff,
        ];

        let mut pairs_cpu: Vec<(Scalar, Scalar)> = Vec::new();
        for i in 0..seeds.len() {
            let j = (i + 1) % seeds.len();
            pairs_cpu.push((Scalar::from(seeds[i]), Scalar::from(seeds[j])));
        }
        // Also stress the top limbs.
        let top = -Scalar::ONE;  // r - 1
        pairs_cpu.push((top, top));
        pairs_cpu.push((top, Scalar::from(2u64)));
        pairs_cpu.push((Scalar::from(2u64), top));

        let gpu_pairs: Vec<_> = pairs_cpu.iter()
            .map(|(a, b)| (scalar_to_words(a), scalar_to_words(b)))
            .collect();
        let results = run_fr_op_gpu(&gpu_pairs, FrOp::Mul).expect("GPU fr_mul batch failed");

        for (idx, ((a, b), result)) in pairs_cpu.iter().zip(results.iter()).enumerate() {
            let expected = a * b;
            let result_s = words_to_scalar(result)
                .unwrap_or_else(|| panic!("result[{}] is not a valid Fr element", idx));
            assert_eq!(
                result_s, expected,
                "fr_mul mismatch at index {idx}: {:?} * {:?}\n  GPU  = {:?}\n  CPU  = {:?}",
                a, b, result, scalar_to_words(&expected),
            );
        }
    }
}
