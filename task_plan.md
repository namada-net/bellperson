# Task Plan: WebGPU-based Groth16 Prover for Bellperson

## Goal

Implement a GPU-based Groth16 prover for bellperson that runs in web browsers using WebGPU (via wgpu). The prover must have excellent browser performance and primarily support BLS12-381, with generic support if feasible.

## Success Criteria

- [ ] Groth16 prover works in web browsers via WebGPU
- [ ] Performance is excellent (comparable to or better than CPU baseline)
- [ ] Supports BLS12-381 curve
- [ ] Works both in browser (WebGPU) and optionally native (Vulkan/Metal/DX12)
- [ ] Integrates cleanly with existing bellperson architecture
- [ ] Passes existing test suite

## Background Context

### Current Architecture
Bellperson already has:
1. **CUDA backend** (feature = "cuda") - NVIDIA GPUs
2. **OpenCL backend** (feature = "opencl") - Cross-platform GPUs
3. **SupraSeal backend** (feature = "cuda-supraseal") - C++ CUDA implementation

The GPU abstraction is compile-time via feature flags, not runtime polymorphism.

### Key Components to Port
1. **FFT kernels** - Used for polynomial operations (A*B-C quotient computation)
2. **Multi-exponentiation (MSM) kernels** - Used for proof assembly (H, L, A, B queries)

### Existing Infrastructure
- `ec-gpu-gen` crate: Generates CUDA/OpenCL kernels at build time
- `build.rs`: Calls `ec_gpu_gen::generate()` to create GPU source code
- GPU locking system: File-based locks to prevent GPU contention
- Hybrid CPU/GPU split: `BELLMAN_CPU_UTILIZATION` env var

## Phases

### Phase 1: Research & Design [pending]
**Status:** pending
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Research wgpu compute capabilities and WebGPU limits
- [ ] Study WGSL shader language for compute shaders
- [ ] Analyze existing CUDA/OpenCL kernels in `ec-gpu-gen` for porting
- [ ] Design wgpu backend architecture (module structure, trait design)
- [ ] Decide on feature flag strategy (wgpu vs webgpu vs both)
- [ ] Research BLS12-381 field arithmetic in WGSL
- [ ] Check for existing wgpu MSM/FFT implementations

**Deliverables:**
- Architecture design document
- List of GPU kernels to implement
- Performance expectations

**Files to examine:**
- `ec-gpu-gen` source code (external crate, may need to fork)
- CUDA/OpenCL kernel sources (generated at build time)
- `src/groth16/prover/native.rs` - understand proving flow

### Phase 2: wgpu Backend Skeleton [pending]
**Status:** pending
**Estimated Time:** 3-4 hours

**Tasks:**
- [ ] Add wgpu dependencies to Cargo.toml
- [ ] Create new feature flag: `webgpu` or `wgpu`
- [ ] Create `src/gpu/wgpu/` module structure
- [ ] Implement device initialization and management
- [ ] Create wgpu versions of `LockedFftKernel` and `LockedMultiexpKernel`
- [ ] Add conditional compilation to `src/gpu/mod.rs`
- [ ] Set up basic error handling for wgpu-specific errors

**Deliverables:**
- Compiling code with `--features wgpu` (no-op kernels)
- Module structure in place
- Device enumeration working

**Files to create:**
- `src/gpu/wgpu/mod.rs`
- `src/gpu/wgpu/device.rs`
- `src/gpu/wgpu/fft.rs`
- `src/gpu/wgpu/multiexp.rs`
- `src/gpu/wgpu/error.rs`

### Phase 3: BLS12-381 Field Arithmetic in WGSL [pending]
**Status:** pending
**Estimated Time:** 4-6 hours

**Tasks:**
- [ ] Implement modular arithmetic for BLS12-381 scalar field (Fr) in WGSL
- [ ] Implement modular arithmetic for base field (Fp) in WGSL
- [ ] Implement G1 point operations (addition, doubling) in WGSL
- [ ] Write unit tests for field operations
- [ ] Optimize using WGSL intrinsics (if available)

**Deliverables:**
- WGSL shader code for BLS12-381 field arithmetic
- Unit tests passing

**Files to create:**
- `src/gpu/wgpu/shaders/field_arithmetic.wgsl`
- `src/gpu/wgpu/shaders/curve_ops.wgsl`

**Notes:**
- BLS12-381 uses 381-bit numbers, need multi-limb representation
- Scalar field Fr: ~255 bits
- Base field Fp: 381 bits
- May need to use multiple u32 limbs (e.g., 8 x u32 for 256 bits)

### Phase 4: FFT Kernel Implementation [pending]
**Status:** pending
**Estimated Time:** 5-8 hours

**Tasks:**
- [ ] Port FFT algorithm from CUDA/OpenCL to WGSL
- [ ] Implement radix-2 FFT (Cooley-Tukey)
- [ ] Implement twiddle factor computation
- [ ] Handle batched FFT (multiple FFTs in parallel)
- [ ] Optimize memory access patterns (coalescing)
- [ ] Implement bit-reversal permutation
- [ ] Test FFT correctness against CPU implementation
- [ ] Benchmark FFT performance

**Deliverables:**
- Working FFT kernel in WGSL
- Tests passing
- Performance benchmarks

**Files to create/modify:**
- `src/gpu/wgpu/shaders/fft.wgsl`
- `src/gpu/wgpu/fft.rs` (kernel management)
- Integration with `EvaluationDomain::fft_many()`

**Notes:**
- Current code uses `kern.radix_fft_many()` from ec-gpu-gen
- Need to replicate behavior for wgpu
- Input/output: slices of field elements
- Must handle omega (root of unity) correctly

### Phase 5: Multi-Exponentiation (MSM) Kernel [pending]
**Status:** pending
**Estimated Time:** 8-12 hours

**Tasks:**
- [ ] Research MSM algorithms (Pippenger's algorithm)
- [ ] Port MSM from CUDA/OpenCL to WGSL
- [ ] Implement bucket method for MSM
- [ ] Handle variable-base MSM
- [ ] Implement density tracking support
- [ ] Optimize for WebGPU constraints
- [ ] Test MSM correctness
- [ ] Benchmark MSM performance
- [ ] Implement hybrid CPU/GPU split (like existing code)

**Deliverables:**
- Working MSM kernel in WGSL
- Tests passing
- Performance benchmarks

**Files to create/modify:**
- `src/gpu/wgpu/shaders/multiexp.wgsl`
- `src/gpu/wgpu/multiexp.rs`
- Integration with `multiexp()` function

**Notes:**
- MSM is the most computationally intensive operation
- Current code uses Pippenger's algorithm (likely)
- Need to handle both G1 and G2 curves
- Must support density maps (sparse exponent vectors)

### Phase 6: Integration with Prover [pending]
**Status:** pending
**Estimated Time:** 3-5 hours

**Tasks:**
- [ ] Integrate wgpu kernels into `src/groth16/prover/native.rs`
- [ ] Update `src/multiexp.rs` to dispatch to wgpu
- [ ] Update `src/domain.rs` to use wgpu FFT
- [ ] Handle async/await for wgpu operations (if needed)
- [ ] Test with simple circuits
- [ ] Test with complex circuits
- [ ] Verify proof correctness with verifier

**Deliverables:**
- End-to-end proof generation working
- All tests passing

**Files to modify:**
- `src/gpu/mod.rs` - add wgpu to feature flags
- `src/multiexp.rs` - add wgpu dispatch
- `src/domain.rs` - add wgpu FFT path

### Phase 7: Browser Compatibility [pending]
**Status:** pending
**Estimated Time:** 4-6 hours

**Tasks:**
- [ ] Test in browser via wasm-pack
- [ ] Handle wasm-specific compilation issues
- [ ] Add wasm-bindgen annotations if needed
- [ ] Test on different browsers (Chrome, Firefox, Safari)
- [ ] Handle WebGPU adapter/device request in browser
- [ ] Create simple browser demo/benchmark
- [ ] Document browser usage

**Deliverables:**
- Working wasm build
- Browser demo
- Browser compatibility table

**Files to create:**
- `examples/browser_prover.rs` or `www/` directory
- Browser-specific documentation

**Notes:**
- Need `wasm32-unknown-unknown` target
- May need to use async device initialization
- Browser console for debugging
- Test on actual WebGPU-enabled browsers

### Phase 8: Performance Optimization [pending]
**Status:** pending
**Estimated Time:** 5-10 hours

**Tasks:**
- [ ] Profile GPU kernels (workgroup size tuning)
- [ ] Optimize memory transfers (minimize CPU<->GPU copies)
- [ ] Optimize shader code (reduce register pressure)
- [ ] Implement kernel fusion if beneficial
- [ ] Tune CPU/GPU split ratio
- [ ] Compare with CUDA/OpenCL performance
- [ ] Document performance characteristics
- [ ] Create performance comparison table

**Deliverables:**
- Optimized kernels
- Performance comparison report
- Tuning guide

**Notes:**
- WebGPU has different constraints than CUDA/OpenCL
- May need browser-specific tuning
- Document hardware requirements (GPU specs needed for good perf)

### Phase 9: Testing & Documentation [pending]
**Status:** pending
**Estimated Time:** 3-4 hours

**Tasks:**
- [ ] Run full test suite with wgpu backend
- [ ] Add wgpu-specific tests
- [ ] Test edge cases (small/large circuits)
- [ ] Test error handling
- [ ] Write README section for wgpu backend
- [ ] Document feature flags
- [ ] Document browser usage
- [ ] Add code examples
- [ ] Update Cargo.toml metadata

**Deliverables:**
- All tests passing
- Comprehensive documentation
- Usage examples

**Files to modify:**
- `README.md`
- `Cargo.toml`
- Add doc comments to new modules

### Phase 10: Cleanup & Polish [pending]
**Status:** pending
**Estimated Time:** 2-3 hours

**Tasks:**
- [ ] Remove debug code
- [ ] Clean up compiler warnings
- [ ] Format code (cargo fmt)
- [ ] Run clippy and fix issues
- [ ] Review error messages
- [ ] Final testing pass
- [ ] Prepare PR (if contributing upstream)

**Deliverables:**
- Clean, production-ready code
- No warnings
- Ready for review/merge

## Design Decisions

### Feature Flag Strategy
**Decision needed:** How to structure feature flags?

Options:
1. Single `wgpu` feature that works everywhere (browser + native)
2. Separate `webgpu` (browser only) and `wgpu-native` features
3. `wgpu` feature with runtime backend selection

**Recommendation:** Single `wgpu` feature, runtime detection of environment

### Integration Approach
**Decision:** Extend existing feature-flag based architecture

- Add `wgpu` alongside `cuda` and `opencl`
- Use conditional compilation like existing code
- Keep same locking/priority patterns (adapt for wgpu)

### Async Handling
**Decision needed:** wgpu is async-first. How to handle?

Options:
1. Block on GPU operations (use `pollster` or similar)
2. Propagate async all the way up (breaking change)
3. Use sync wrapper internally

**Recommendation:** Block on GPU operations internally to maintain API compatibility

### Shader Organization
**Decision:** Separate WGSL files, embedded at compile time

- Use `include_str!()` to embed shaders
- Keep shaders in `src/gpu/wgpu/shaders/`
- One shader per kernel type (fft, multiexp)

## Dependencies to Add

```toml
[dependencies]
wgpu = { version = "0.19", features = ["webgpu"] }
pollster = "0.3"  # For blocking on async operations
bytemuck = "1.14"  # For safe transmutes

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = ["Gpu", "GpuAdapter", "GpuDevice"] }
```

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| WebGPU not mature enough | High | Fallback to CPU, document limitations |
| Performance worse than CPU | Medium | Optimize, consider hybrid approach |
| Browser compatibility issues | Medium | Test on multiple browsers, document requirements |
| WGSL limitations (no 64-bit int) | High | Use multi-limb arithmetic |
| wgpu async complexity | Medium | Use blocking wrappers internally |
| Large shader compilation time | Low | Cache compiled shaders |

## Environment Variables

Existing (to maintain compatibility):
- `BELLMAN_NO_GPU` - disable GPU entirely
- `BELLMAN_CPU_UTILIZATION` - fraction of work on CPU (0.0-1.0)
- `BELLPERSON_GPUS_PER_LOCK` - number of GPUs to use

New (if needed):
- `BELLMAN_WGPU_BACKEND` - force specific backend (vulkan/metal/dx12/webgpu)
- `BELLMAN_WGPU_POWER_PREFERENCE` - low-power vs high-performance

## Testing Strategy

1. **Unit tests** - Field arithmetic, individual operations
2. **Kernel tests** - FFT correctness, MSM correctness (against CPU)
3. **Integration tests** - Full proof generation
4. **Performance tests** - Benchmarks vs CPU/CUDA/OpenCL
5. **Browser tests** - Manual testing in browsers
6. **Compatibility tests** - Different GPU vendors

## Open Questions

1. Should we fork `ec-gpu-gen` to add wgpu support, or keep it separate?
2. What's the minimum WebGPU version/feature set we need?
3. Should we support compute shader fallbacks for older GPUs?
4. How to handle GPU selection in browser (user choice vs auto)?
5. What's the target circuit size for optimization?

## Progress Log

### Session 1 (Current)
- Explored bellperson repository structure
- Analyzed existing GPU backends (CUDA/OpenCL)
- Studied Groth16 prover implementation
- Understood FFT and multiexp kernel architecture
- Created comprehensive task plan

## Next Steps

1. Complete Phase 1 (Research & Design)
2. Make final architectural decisions
3. Begin Phase 2 (wgpu Backend Skeleton)
