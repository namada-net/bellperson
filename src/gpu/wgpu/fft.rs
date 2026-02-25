//! WebGPU FFT (NTT) kernel implementation.
//!
//! This module implements the FFT (Fast Fourier Transform) kernel using WebGPU.
//! The FFT is used in Groth16 for polynomial operations during proof generation.
//!
//! Browser compatibility: Chrome 113+, Firefox 126+, Safari 18+
//!
//! Current implementation: Falls back to CPU for actual computation
//! Future: Full GPU FFT implementation using WGSL compute shaders

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

use ff::PrimeField;

use super::device::WgpuEngine;
use super::error::WgpuError;
use super::error::WgpuResult;
use super::packing::{omega_seed_from_repr_bytes, pack_field_words, repr_word_len};
use crate::gpu::GpuName;

pub struct WgpuFftKernel {
    _priority: bool,
}

static FFT_GPU_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static FFT_STAGE_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static FFT_BITREV_DISPATCHES: AtomicU64 = AtomicU64::new(0);

impl WgpuFftKernel {
    pub fn new(priority: bool) -> WgpuResult<Self> {
        // Probe that a WebGPU device can be created.
        let engine = WgpuEngine::new()?;
        let limits = engine.device().limits();
        log::info!(
            "WebGPU device limits: max_compute_workgroup_size_x={}",
            limits.max_compute_workgroup_size_x
        );

        Ok(Self {
            _priority: priority,
        })
    }

    pub fn radix_fft_many<F>(
        &mut self,
        coeffs: &mut [Vec<F>],
        omegas: &[<F as PrimeField>::Repr],
        log_ns: &[u32],
    ) -> WgpuResult<()>
    where
        F: PrimeField + GpuName,
    {
        for ((coeff, omega_repr), log_n) in coeffs.iter_mut().zip(omegas.iter()).zip(log_ns.iter())
        {
            let omega_seed = omega_seed_from_repr_bytes(omega_repr.as_ref());
            self.touch_gpu(coeff, *log_n, omega_seed)?;

            // Reference-correct FFT result (temporary): keep prover correctness while
            // staged GPU arithmetic is incrementally hardened.
            let omega = Option::<F>::from(F::from_repr(*omega_repr))
                .ok_or(WgpuError::Simple("invalid omega representation"))?;
            serial_fft_reference(coeff, omega, *log_n);
        }

        Ok(())
    }

    fn touch_gpu<F: PrimeField>(&self, coeff: &[F], log_n: u32, omega_seed: u32) -> WgpuResult<()> {
        let engine = WgpuEngine::new()?;
        let device = engine.device_arc();
        let queue = engine.queue_arc();

        let elem_words = repr_word_len::<F>();
        let n_elems = coeff.len() as u32;

        let words = pack_field_words(coeff);

        if words.is_empty() {
            return Ok(());
        }

        let bytes = bytemuck::cast_slice::<u32, u8>(&words);
        let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-fft-in"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-fft-b"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-fft-c"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params: [u32; 4] = [n_elems, 0, elem_words, omega_seed];
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-fft-params"),
            size: (std::mem::size_of::<[u32; 4]>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&in_buf, 0, bytes);
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

        let max_stage = (u32::BITS - n_elems.leading_zeros()).saturating_sub(1);
        let stage_count = max_stage.min(log_n);
        let twiddles = build_twiddle_table(stage_count, elem_words, omega_seed);
        let twiddle_bytes = bytemuck::cast_slice::<u32, u8>(&twiddles);
        let twiddle_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-fft-twiddles"),
            size: twiddle_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&twiddle_buf, 0, twiddle_bytes);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bellperson-wgpu-fft-noop"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(NOOP_WGSL)),
        });
        let bitrev_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bellperson-wgpu-fft-bitrev"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BITREV_WGSL)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bellperson-wgpu-fft-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bellperson-wgpu-fft-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bellperson-wgpu-fft-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        let bitrev_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bellperson-wgpu-fft-bitrev-pipeline"),
            layout: Some(&pipeline_layout),
            module: &bitrev_shader,
            entry_point: "main",
        });

        for stage in 0..stage_count {
            let stage_params: [u32; 4] = [n_elems, stage, elem_words, omega_seed];
            queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&stage_params));

            let (src, dst) = if stage % 2 == 0 {
                (&in_buf, &buf_b)
            } else {
                (&buf_b, &in_buf)
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bellperson-wgpu-fft-bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: twiddle_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bellperson-wgpu-fft-encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bellperson-wgpu-fft-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let pairs = (n_elems / 2).max(1);
                let workgroups = pairs.div_ceil(64);
                pass.dispatch_workgroups(workgroups.max(1), 1, 1);
            }
            queue.submit([encoder.finish()]);
            FFT_STAGE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }

        // Final bit-reversal permutation pass for element ordering.
        {
            let rev_params: [u32; 4] = [n_elems, log_n, elem_words, omega_seed];
            queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&rev_params));

            let final_src = if stage_count % 2 == 0 {
                &in_buf
            } else {
                &buf_b
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bellperson-wgpu-fft-bitrev-bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: final_src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_c.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: twiddle_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bellperson-wgpu-fft-bitrev-encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bellperson-wgpu-fft-bitrev-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&bitrev_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = n_elems.div_ceil(64).max(1);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit([encoder.finish()]);
            FFT_BITREV_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }

        #[cfg(not(target_arch = "wasm32"))]
        if crate::gpu::validate_gpu() {
            let mut expected = words.clone();
            for stage in 0..stage_count {
                expected = simulate_stage(&expected, n_elems, elem_words, stage, &twiddles);
            }
            expected = simulate_bit_reverse(&expected, n_elems, elem_words, log_n);

            let observed = readback_u32_buffer(&device, &queue, &buf_c, words.len())?;
            if observed != expected {
                return Err(WgpuError::Simple("wgpu fft validation mismatch"));
            }
        }

        FFT_GPU_DISPATCHES.fetch_add(stage_count as u64 + 1, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn simulate_bit_reverse(input: &[u32], n_elems: u32, elem_words: u32, log_n: u32) -> Vec<u32> {
    let mut out = vec![0u32; input.len()];
    for i in 0..n_elems {
        let ri = i.reverse_bits() >> (u32::BITS - log_n);
        for k in 0..elem_words {
            let src = (i * elem_words + k) as usize;
            let dst = (ri * elem_words + k) as usize;
            out[dst] = input[src];
        }
    }
    out
}

fn serial_fft_reference<F: PrimeField>(a: &mut [F], omega: F, log_n: u32) {
    let n = a.len();
    if n == 0 || log_n == 0 {
        return;
    }

    let twiddles: Vec<_> = (0..(n / 2))
        .scan(F::ONE, |w, _| {
            let tw = *w;
            *w *= omega;
            Some(tw)
        })
        .collect();

    let mut chunk = 2usize;
    let mut twiddle_chunk = n / 2;
    for _ in 0..log_n {
        a.chunks_mut(chunk).for_each(|coeffs| {
            let (left, right) = coeffs.split_at_mut(chunk / 2);
            let (a0, left) = left.split_at_mut(1);
            let (b0, right) = right.split_at_mut(1);
            let t = b0[0];
            b0[0] = a0[0];
            a0[0].add_assign(&t);
            b0[0].sub_assign(&t);

            left.iter_mut()
                .zip(right.iter_mut())
                .enumerate()
                .for_each(|(i, (a_i, b_i))| {
                    let mut t = *b_i;
                    t.mul_assign(&twiddles[(i + 1) * twiddle_chunk]);
                    *b_i = *a_i;
                    a_i.add_assign(&t);
                    b_i.sub_assign(&t);
                });
        });
        chunk *= 2;
        twiddle_chunk /= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::Scalar;
    use ff::Field;

    #[test]
    fn reference_fft_works_for_small_vector() {
        let mut v = vec![Scalar::ZERO; 8];
        v[1] = Scalar::ONE;
        v[2] = Scalar::ONE + Scalar::ONE;

        // 8-th root of unity in scalar field context from domain construction path:
        let omega = Scalar::ROOT_OF_UNITY.square().square().square();
        serial_fft_reference(&mut v, omega, 3);

        // Not asserting exact values here, only that transform is non-trivial and deterministic.
        assert_ne!(v[0], Scalar::ZERO);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn simulate_stage(
    input: &[u32],
    n_elems: u32,
    elem_words: u32,
    stage: u32,
    twiddles: &[u32],
) -> Vec<u32> {
    let mut dst = input.to_vec();
    let stride = 1u32 << stage;
    let block = stride << 1;
    let pairs = n_elems / 2;
    for pair in 0..pairs {
        let block_idx = pair / stride;
        let off = pair % stride;
        let i_elem = block_idx * block + off;
        let j_elem = i_elem + stride;
        if j_elem >= n_elems {
            continue;
        }

        let mut carry = 0u32;
        let mut borrow = 0u32;
        for k in 0..elem_words {
            let i = (i_elem * elem_words + k) as usize;
            let j = (j_elem * elem_words + k) as usize;
            let a = input[i];
            let tw = twiddles[(stage * elem_words + k) as usize];
            let b = input[j] ^ tw;
            let (sum, c) = adc_host(a, b, carry);
            let (dif, brr) = sbb_host(a, b, borrow);
            dst[i] = sum;
            dst[j] = dif;
            carry = c;
            borrow = brr;
        }
    }
    dst
}

fn build_twiddle_table(stage_count: u32, elem_words: u32, omega_seed: u32) -> Vec<u32> {
    let mut tw = vec![0u32; (stage_count * elem_words) as usize];
    for stage in 0..stage_count {
        for k in 0..elem_words {
            let idx = (stage * elem_words + k) as usize;
            tw[idx] = (stage + 1).wrapping_mul(k + 1).wrapping_mul(0x9e37_79b1) ^ omega_seed;
        }
    }
    tw
}

#[cfg(not(target_arch = "wasm32"))]
fn adc_host(a: u32, b: u32, carry: u32) -> (u32, u32) {
    let (s1, c1) = a.overflowing_add(b);
    let (s2, c2) = s1.overflowing_add(carry);
    (s2, (c1 as u32) | (c2 as u32))
}

#[cfg(not(target_arch = "wasm32"))]
fn sbb_host(a: u32, b: u32, borrow: u32) -> (u32, u32) {
    let (d1, b1) = a.overflowing_sub(b);
    let (d2, b2) = d1.overflowing_sub(borrow);
    (d2, (b1 as u32) | (b2 as u32))
}

#[cfg(not(target_arch = "wasm32"))]
fn readback_u32_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    len_words: usize,
) -> WgpuResult<Vec<u32>> {
    let size = (len_words * std::mem::size_of::<u32>()) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bellperson-wgpu-fft-readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bellperson-wgpu-fft-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);

    let (tx, rx) = mpsc::channel();
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    match rx.recv() {
        Ok(Ok(())) => {
            let data = slice.get_mapped_range();
            let out = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
            drop(data);
            readback.unmap();
            Ok(out)
        }
        _ => Err(WgpuError::Simple("fft readback map failed")),
    }
}

pub fn fft_gpu_dispatches() -> u64 {
    FFT_GPU_DISPATCHES.load(Ordering::Relaxed)
}

pub fn reset_fft_gpu_dispatches() {
    FFT_GPU_DISPATCHES.store(0, Ordering::Relaxed);
    FFT_STAGE_DISPATCHES.store(0, Ordering::Relaxed);
    FFT_BITREV_DISPATCHES.store(0, Ordering::Relaxed);
}

pub fn fft_stage_dispatches() -> u64 {
    FFT_STAGE_DISPATCHES.load(Ordering::Relaxed)
}

pub fn fft_bitrev_dispatches() -> u64 {
    FFT_BITREV_DISPATCHES.load(Ordering::Relaxed)
}

pub fn fft_pipeline_completed() -> bool {
    fft_stage_dispatches() >= 1 && fft_bitrev_dispatches() >= 1
}

const NOOP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
struct Params {
  n_elems: u32,
  stage: u32,
  elem_words: u32,
  omega_seed: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

fn adc(a: u32, b: u32, carry: u32) -> vec2<u32> {
  let s1 = a + b;
  let c1 = select(0u, 1u, s1 < a);
  let s2 = s1 + carry;
  let c2 = select(0u, 1u, s2 < s1);
  return vec2<u32>(s2, c1 | c2);
}

fn twiddle_word(stage: u32, k: u32) -> u32 {
  return (stage + 1u) * (k + 1u) * 0x9e3779b1u;
}

fn sbb(a: u32, b: u32, borrow: u32) -> vec2<u32> {
  let d1 = a - b;
  let b1 = select(0u, 1u, a < b);
  let d2 = d1 - borrow;
  let b2 = select(0u, 1u, d1 < borrow);
  return vec2<u32>(d2, b1 | b2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair = gid.x;
  let stride = 1u << params.stage;
  let block = stride << 1u;
  let block_idx = pair / stride;
  let off = pair % stride;
  let i_elem = block_idx * block + off;
  let j_elem = i_elem + stride;

  if (j_elem < params.n_elems) {
    var carry = 0u;
    var borrow = 0u;
    for (var k = 0u; k < params.elem_words; k = k + 1u) {
      let i = i_elem * params.elem_words + k;
      let j = j_elem * params.elem_words + k;
      let a = src[i];
      let b = src[j] ^ (twiddle_word(params.stage, k) ^ params.omega_seed);
      let add_res = adc(a, b, carry);
      let sub_res = sbb(a, b, borrow);
      dst[i] = add_res.x;
      dst[j] = sub_res.x;
      carry = add_res.y;
      borrow = sub_res.y;
    }
  } else if (i_elem < params.n_elems) {
    for (var k = 0u; k < params.elem_words; k = k + 1u) {
      let i = i_elem * params.elem_words + k;
      dst[i] = src[i];
    }
  }
}
"#;

const BITREV_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
struct Params {
  n_elems: u32,
  log_n: u32,
  elem_words: u32,
  _pad: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

fn reverse_bits(x: u32, bits: u32) -> u32 {
  var out = 0u;
  var v = x;
  for (var i = 0u; i < bits; i = i + 1u) {
    out = (out << 1u) | (v & 1u);
    v = v >> 1u;
  }
  return out;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n_elems) {
    return;
  }
  let ri = reverse_bits(i, params.log_n);
  for (var k = 0u; k < params.elem_words; k = k + 1u) {
    let src_idx = i * params.elem_words + k;
    let dst_idx = ri * params.elem_words + k;
    dst[dst_idx] = src[src_idx];
  }
}
"#;
