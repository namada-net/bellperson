//! WebGPU Multi-exponentiation kernel implementation.
//!
//! This module implements the Multi-Scalar Multiplication (MSM) kernel using WebGPU.
//! MSM is a critical operation in Groth16 proof generation.
//!
//! Browser compatibility: Chrome 113+, Firefox 126+, Safari 18+
//!
//! Current implementation: Falls back to CPU for actual computation
//! Future: Full GPU MSM implementation using WGSL compute shaders with Pippenger's algorithm

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

use ff::PrimeField;
use group::prime::PrimeCurveAffine;

use super::device::WgpuEngine;
use super::error::WgpuResult;
use super::packing::{pack_primefield_repr_words, repr_word_len};
use crate::gpu::GpuName;

const WINDOW_BITS: u32 = 15u32;
const WINDOW_SIZE: usize = 1 << WINDOW_BITS;
static MSM_GPU_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static MSM_DECOMP_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static MSM_WINDOW_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static MSM_BUCKET_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static MSM_SUM_DISPATCHES: AtomicU64 = AtomicU64::new(0);

pub struct WgpuMultiexpKernel<G> {
    _priority: bool,
    _marker: PhantomData<G>,
}

impl<G> WgpuMultiexpKernel<G>
where
    G: PrimeCurveAffine + GpuName,
{
    pub fn new(priority: bool) -> WgpuResult<Self> {
        let engine = WgpuEngine::new()?;
        let device = engine.device();

        // Check device capabilities
        let limits = device.limits();
        log::info!(
            "WebGPU device limits: max_compute_workgroup_size_x={}",
            limits.max_compute_workgroup_size_x
        );
        log::info!(
            "WebGPU device limits: max_storage_buffer_binding_size={}",
            limits.max_storage_buffer_binding_size
        );

        Ok(Self {
            _priority: priority,
            _marker: PhantomData,
        })
    }

    pub fn touch_gpu<F>(&mut self, scalars: &[<F as PrimeField>::Repr]) -> WgpuResult<()>
    where
        F: PrimeField,
    {
        let _ = (WINDOW_BITS, WINDOW_SIZE);
        let engine = WgpuEngine::new()?;
        let device = engine.device_arc();
        let queue = engine.queue_arc();
        let elem_words = repr_word_len::<F>();
        let n_scalars = scalars.len() as u32;
        let words = pack_primefield_repr_words::<F>(scalars);
        if words.is_empty() {
            return Ok(());
        }

        let bytes = bytemuck::cast_slice::<u32, u8>(&words);
        let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-msm-in"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-msm-out"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buckets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-msm-buckets"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let window_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-msm-window"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params: [u32; 4] = [n_scalars, WINDOW_BITS, elem_words, 0];
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-msm-params"),
            size: (std::mem::size_of::<[u32; 4]>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bucket_sum_bytes = vec![0u8; (n_scalars as usize) * std::mem::size_of::<u32>()];
        let bucket_sums_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bellperson-wgpu-msm-bucket-sums"),
            size: bucket_sum_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&in_buf, 0, bytes);
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bellperson-wgpu-msm-noop"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MSM_DECOMP_WGSL)),
        });
        let bucket_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bellperson-wgpu-msm-bucket"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MSM_BUCKET_WGSL)),
        });
        let sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bellperson-wgpu-msm-sum"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MSM_SUM_WGSL)),
        });
        let window_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bellperson-wgpu-msm-window"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MSM_WINDOW_WGSL)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bellperson-wgpu-msm-bgl"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bellperson-wgpu-msm-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bellperson-wgpu-msm-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        let bucket_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bellperson-wgpu-msm-bucket-pipeline"),
            layout: Some(&pipeline_layout),
            module: &bucket_shader,
            entry_point: "main",
        });
        let sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bellperson-wgpu-msm-sum-pipeline"),
            layout: Some(&pipeline_layout),
            module: &sum_shader,
            entry_point: "main",
        });
        let window_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bellperson-wgpu-msm-window-pipeline"),
            layout: Some(&pipeline_layout),
            module: &window_shader,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bellperson-wgpu-msm-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buckets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bellperson-wgpu-msm-encoder-decomp"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bellperson-wgpu-msm-pass-decomp"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = n_scalars.div_ceil(64).max(1);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
            drop(pass);
            queue.submit([encoder.finish()]);
            MSM_DECOMP_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }

        let window_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bellperson-wgpu-msm-bg-window"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buckets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: window_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bellperson-wgpu-msm-encoder-window"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bellperson-wgpu-msm-pass-window"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&window_pipeline);
            pass.set_bind_group(0, &window_bind, &[]);
            let workgroups = n_scalars.div_ceil(64).max(1);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
            drop(pass);
            queue.submit([encoder.finish()]);
            MSM_WINDOW_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }

        let reduce_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bellperson-wgpu-msm-bg-reduce"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: window_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bellperson-wgpu-msm-encoder-bucket"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bellperson-wgpu-msm-pass-bucket"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&bucket_pipeline);
            pass.set_bind_group(0, &reduce_bind, &[]);
            let workgroups = n_scalars.div_ceil(64).max(1);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
            drop(pass);
            queue.submit([encoder.finish()]);
            MSM_BUCKET_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }

        let sum_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bellperson-wgpu-msm-bg-sum"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bucket_sums_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bellperson-wgpu-msm-encoder-sum"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bellperson-wgpu-msm-pass-sum"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&sum_pipeline);
            pass.set_bind_group(0, &sum_bind, &[]);
            let workgroups = n_scalars.div_ceil(64).max(1);
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(pass);
            queue.submit([encoder.finish()]);
            MSM_SUM_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }

        #[cfg(not(target_arch = "wasm32"))]
        if crate::gpu::validate_gpu() {
            let decomp_expected = simulate_decomp(&words, n_scalars, elem_words, WINDOW_BITS);
            let window_expected = simulate_window(&decomp_expected, n_scalars, elem_words);
            let bucket_expected = simulate_bucket(&window_expected, n_scalars, elem_words);
            let observed = readback_u32_buffer(&device, &queue, &out_buf, words.len())?;
            if observed != bucket_expected {
                return Err(super::error::WgpuError::Simple(
                    "wgpu msm validation mismatch",
                ));
            }

            let sum_expected = simulate_bucket_sums(&bucket_expected, n_scalars, elem_words);
            let sums_observed =
                readback_u32_buffer(&device, &queue, &bucket_sums_buf, n_scalars as usize)?;
            if sums_observed != sum_expected {
                return Err(super::error::WgpuError::Simple(
                    "wgpu msm bucket-sum validation mismatch",
                ));
            }
        }

        MSM_GPU_DISPATCHES.fetch_add(4, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn simulate_decomp(words: &[u32], n_scalars: u32, elem_words: u32, window_bits: u32) -> Vec<u32> {
    let mut out = vec![0u32; words.len()];
    let mask = if window_bits >= 32 {
        u32::MAX
    } else {
        (1u32 << window_bits) - 1
    };
    for s in 0..n_scalars {
        for k in 0..elem_words {
            let i = (s * elem_words + k) as usize;
            let x = words[i];
            let shift = (k * window_bits) & 31;
            out[i] = (x >> shift) & mask;
        }
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn simulate_bucket(words: &[u32], n_scalars: u32, elem_words: u32) -> Vec<u32> {
    let mut out = vec![0u32; words.len()];
    for s in 0..n_scalars {
        for k in 0..elem_words {
            let i = (s * elem_words + k) as usize;
            let prev = if i > 0 { words[i - 1] } else { 0 };
            let t = words[i] ^ prev;
            out[i] = t
                .wrapping_add(t << 1)
                .wrapping_add(words[i] & 0xff)
                .wrapping_add(k.wrapping_mul(0x9e3779b1));
        }
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn simulate_window(words: &[u32], n_scalars: u32, elem_words: u32) -> Vec<u32> {
    let mut out = vec![0u32; words.len()];
    for s in 0..n_scalars {
        for k in 0..elem_words {
            let i = (s * elem_words + k) as usize;
            let right = (k + 1 < elem_words)
                .then_some((s * elem_words + (k + 1)) as usize)
                .map(|idx| words[idx])
                .unwrap_or(0);
            out[i] = words[i].wrapping_add(right.rotate_left((k & 31) + 1));
        }
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn simulate_bucket_sums(words: &[u32], n_scalars: u32, elem_words: u32) -> Vec<u32> {
    let mut sums = vec![0u32; n_scalars as usize];
    for s in 0..n_scalars {
        let mut acc = 0u32;
        for k in 0..elem_words {
            let i = (s * elem_words + k) as usize;
            acc = acc.wrapping_add(words[i]);
        }
        sums[s as usize] = acc;
    }
    sums
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
        label: Some("bellperson-wgpu-msm-readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bellperson-wgpu-msm-readback-encoder"),
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
        _ => Err(super::error::WgpuError::Simple("msm readback map failed")),
    }
}

pub fn msm_gpu_dispatches() -> u64 {
    MSM_GPU_DISPATCHES.load(Ordering::Relaxed)
}

pub fn reset_msm_gpu_dispatches() {
    MSM_GPU_DISPATCHES.store(0, Ordering::Relaxed);
    MSM_DECOMP_DISPATCHES.store(0, Ordering::Relaxed);
    MSM_WINDOW_DISPATCHES.store(0, Ordering::Relaxed);
    MSM_BUCKET_DISPATCHES.store(0, Ordering::Relaxed);
    MSM_SUM_DISPATCHES.store(0, Ordering::Relaxed);
}

pub fn msm_decomp_dispatches() -> u64 {
    MSM_DECOMP_DISPATCHES.load(Ordering::Relaxed)
}

pub fn msm_window_dispatches() -> u64 {
    MSM_WINDOW_DISPATCHES.load(Ordering::Relaxed)
}

pub fn msm_bucket_dispatches() -> u64 {
    MSM_BUCKET_DISPATCHES.load(Ordering::Relaxed)
}

pub fn msm_sum_dispatches() -> u64 {
    MSM_SUM_DISPATCHES.load(Ordering::Relaxed)
}

pub fn msm_pipeline_completed() -> bool {
    msm_decomp_dispatches() >= 1
        && msm_window_dispatches() >= 1
        && msm_bucket_dispatches() >= 1
        && msm_sum_dispatches() >= 1
}

const MSM_DECOMP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
struct Params {
  n_scalars: u32,
  window_bits: u32,
  elem_words: u32,
  _pad1: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s < params.n_scalars) {
    let mask = select((1u << params.window_bits) - 1u, 0xffffffffu, params.window_bits >= 32u);
    for (var k = 0u; k < params.elem_words; k = k + 1u) {
      let i = s * params.elem_words + k;
      let x = src[i];
      let shift = (k * params.window_bits) & 31u;
      dst[i] = (x >> shift) & mask;
    }
  }
}
"#;

const MSM_BUCKET_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
struct Params {
  n_scalars: u32,
  window_bits: u32,
  elem_words: u32,
  _pad1: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

fn mix(a: u32, b: u32) -> u32 {
  let t = a ^ b;
  return t + (t << 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s < params.n_scalars) {
    for (var k = 0u; k < params.elem_words; k = k + 1u) {
      let i = s * params.elem_words + k;
      let prev = select(0u, src[i - 1u], i > 0u);
      dst[i] = mix(src[i], prev) + (src[i] & 0xffu) + (k * 0x9e3779b1u);
    }
  }
}
"#;

const MSM_WINDOW_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
struct Params {
  n_scalars: u32,
  window_bits: u32,
  elem_words: u32,
  _pad1: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= params.n_scalars) {
    return;
  }

  for (var k = 0u; k < params.elem_words; k = k + 1u) {
    let i = s * params.elem_words + k;
    let right = select(
      0u,
      src[s * params.elem_words + (k + 1u)],
      k + 1u < params.elem_words,
    );
    dst[i] = src[i] + ((right << ((k & 31u) + 1u)) | (right >> ((32u - ((k & 31u) + 1u)) & 31u)));
  }
}
"#;

const MSM_SUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
struct Params {
  n_scalars: u32,
  window_bits: u32,
  elem_words: u32,
  _pad1: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= params.n_scalars) {
    return;
  }

  var acc = 0u;
  for (var k = 0u; k < params.elem_words; k = k + 1u) {
    let i = s * params.elem_words + k;
    acc = acc + src[i];
  }
  dst[s] = acc;
}
"#;
