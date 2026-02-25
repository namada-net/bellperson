# Research Findings: Bellperson WebGPU Prover

## Repository Overview

**Package:** nam-bellperson v0.26.6-nam.1 (Namada fork of Filecoin's bellperson)
**Location:** `/Users/sugo/Code/heliax/bellperson`
**Language:** Rust (edition 2024, rust-version 1.85)

## Architecture Analysis

### GPU Backend System (3-Tier Architecture)

#### Tier 1: Feature-Gated Conditional Compilation
Located in `src/gpu/mod.rs` (33 lines)

```rust
#[cfg(any(feature = "cuda", feature = "opencl"))]
mod locks;  // Real GPU support

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
mod nogpu;  // Polyfill stubs
```

**Key insight:** The architecture uses **compile-time** feature flags, not runtime polymorphism. This is critical for our wgpu integration.

#### Tier 2: GPU Kernel Management
Located in `src/gpu/locks.rs` (360 lines)

Implements a sophisticated locking system:
- **GPULock** - File-based locks at `/tmp/bellman.gpu.lock`
- **PriorityLock** - Allows high-priority processes to preempt low-priority ones
- **locked_kernel! macro** - Generates `LockedFftKernel` and `LockedMultiexpKernel`

**Retry loop pattern:**
1. Acquire GPU lock (may wait for priority)
2. Create kernel via `ec_gpu_gen::program!(device)`
3. Execute work
4. If aborted (high priority process), retry
5. If GPU fails, fallback to CPU with warning

**Environment variables:**
- `BELLMAN_NO_GPU` - disable GPU entirely (check for != "0")
- `BELLPERSON_GPUS_PER_LOCK` - control GPU selection
- `BELLMAN_CPU_UTILIZATION` - fraction of work on CPU (0.0-1.0)
- `BELLMAN_CUSTOM_GPU` / `RUST_GPU_TOOLS_CUSTOM_GPU` - custom GPU core counts

#### Tier 3: Hybrid CPU/GPU Execution
Located in `src/gpu/multiexp.rs` (191 lines)

`CpuGpuMultiexpKernel` wraps `ec_gpu_gen::multiexp::MultiexpKernel`:
- Splits work based on `BELLMAN_CPU_UTILIZATION`
- GPU portion: `kernel.parallel_multiexp(...)`
- CPU portion: `multiexp_cpu(...)` from ec_gpu_gen
- Results combined by addition

**Key insight:** We need to replicate this hybrid CPU/GPU split for wgpu.

### Groth16 Prover Flow

Located in `src/groth16/prover/native.rs` (455 lines)

#### Proving Algorithm Steps

1. **Circuit Synthesis** (parallel via rayon)
   - `synthesize_circuits_batch()` - produces `ProvingAssignment` per circuit
   - Captures A, B, C polynomial evaluations
   - Serializes input/aux assignments to `Fr::Repr` (field element representation)

2. **FFT Phase** - `execute_fft()`
   ```
   Input: A, B, C coefficient polynomials
   
   Step 1: IFFT(A), IFFT(B), IFFT(C) - batched
   Step 2: Coset FFT(A), Coset FFT(B), Coset FFT(C) - batched
   Step 3: A * B - C (in evaluation domain)
   Step 4: Divide by vanishing polynomial Z(X) on coset
   Step 5: Inverse coset FFT
   
   Output: Quotient polynomial H(X) coefficients (as Fr::Repr)
   ```
   
   Uses `LockedFftKernel::new(priority)` for GPU acceleration.

3. **Multi-Exponentiation Phase** (uses `LockedMultiexpKernel`)
   
   **H query:**
   ```
   h = multiexp(params.h, fullDensity, a_coeffs)
   ```
   
   **L query:**
   ```
   l = multiexp(params.l, fullDensity, aux_assignment)
   ```
   
   **A query (G1):**
   ```
   a_inputs = multiexp(params.a_inputs, fullDensity, input_assignment)
   a_aux = multiexp(params.a_aux, a_aux_density, aux_assignment)
   ```
   
   **B query (G1, only if ZK):**
   ```
   b_g1_inputs = multiexp(params.b_g1_inputs, b_input_density, input_assignment)
   b_g1_aux = multiexp(params.b_g1_aux, b_aux_density, aux_assignment)
   ```
   
   **B query (G2):**
   ```
   b_g2_inputs = multiexp(params.b_g2_inputs, b_input_density, input_assignment)
   b_g2_aux = multiexp(params.b_g2_aux, b_aux_density, aux_assignment)
   ```
   
   **Critical:** G1 kernel must be dropped before G2 kernel is created (GPU lock constraint).

4. **Proof Assembly**
   ```
   g_a = delta_g1 * r + alpha_g1 + a_answer
   g_b = delta_g2 * s + beta_g2 + b2_answer
   g_c = computation of C component
   
   Proof { a: g_a, b: g_b, c: g_c }
   ```

### FFT Implementation

Located in `src/domain.rs` (469 lines)

**EvaluationDomain<F>** struct:
```rust
{
    coeffs: Vec<F>,        // Polynomial coefficients
    exp: u32,              // log2(domain size)
    omega: F,              // 2^exp primitive root of unity
    omegainv: F,           // omega^-1
    geninv: F,             // multiplicative generator inverse
    minv: F,               // domain size inverse
}
```

**best_fft()** function (line 263):
1. **GPU path** (cuda|opencl): `kern.with(|k| gpu_fft(k, ...))`
   - Calls `kern.radix_fft_many(coeffs, omegas, log_ns)`
2. **CPU fallback:**
   - Bit-reversal permutation
   - Cooley-Tukey radix-2 FFT (serial or parallel via rayon)

**Key operations:**
- `fft_many()` - batch multiple FFTs
- `ifft_many()` - batch multiple inverse FFTs
- `coset_fft_many()` - FFT on coset (multiply by powers of generator first)
- `icoset_fft()` - inverse coset FFT

### Multi-Exponentiation Implementation

Located in `src/multiexp.rs` (64 lines)

**GPU version** (cuda|opencl):
```rust
pub fn multiexp<G>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<G::Scalar::Repr>>,
    kern: &mut LockedMultiexpKernel<G>
) -> Waiter<Result<G::Curve, EcError>>
{
    // Try GPU
    if let Ok(p) = kern.with(|k| k.multiexp(pool, bss, exps, skip)) {
        return Waiter::done(Ok(p));
    }
    
    // Fallback to CPU
    let result_cpu = multiexp_cpu(pool, bases, density_map, exponents);
    let result = result_cpu.wait();  // Block until CPU done
    Waiter::done(result)
}
```

**Non-GPU version:** Direct CPU call to `multiexp_cpu()`.

**Key insight:** The `Waiter` type is from `ec_gpu_gen::threadpool` - it's like a future/promise that can be `.wait()`ed.

### Build-Time Kernel Generation

Located in `build.rs` (30 lines)

```rust
#[cfg(any(feature = "cuda", feature = "opencl"))]
fn gpu_kernel() {
    let source_builder = SourceBuilder::new()
        .add_fft::<Scalar>()                 // BLS12-381 scalar field Fr
        .add_multiexp::<G1Affine, Fp>()      // G1 multiexp
        .add_multiexp::<G2Affine, Fp2>();    // G2 multiexp
    ec_gpu_gen::generate(&source_builder);   // Writes CUDA/OpenCL sources
}
```

**Key insight:** For wgpu, we'll need to either:
1. Fork `ec-gpu-gen` to add wgpu support
2. Create separate build logic for WGSL shaders
3. Embed WGSL shaders directly (simpler for MVP)

### Curve and Field Types (BLS12-381)

From `blstrs` crate (nam-blstrs 0.7.1-nam.0):
- **Scalar** = Fr (scalar field, ~255 bits, order of G1/G2)
- **G1Affine** (base field Fp, 381 bits)
- **G2Affine** (extension field Fp2, 2×381 bits)
- **Fp** (base field, 381-bit prime)
- **Fp2** (quadratic extension, Fp[u]/(u²+1))

**Challenge for WGSL:** No native 64-bit integers, must use multi-limb arithmetic.

## External Dependencies Analysis

### ec-gpu-gen (nam-ec-gpu-gen 0.7.2-nam.0)
**Purpose:** GPU kernel generation and execution for FFT and multiexp

**Key exports:**
- `SourceBuilder` - build-time kernel generation
- `FftKernel<F>` - FFT execution
- `MultiexpKernel<G>` - Multiexp execution
- `multiexp_cpu::*` - CPU fallback implementations
- `threadpool::Worker` - thread pool for parallelism
- `rust_gpu_tools::Device` - GPU device enumeration

**Features:**
- `cuda` - NVIDIA CUDA backend
- `opencl` - OpenCL backend

**For wgpu integration:** We'll likely need to bypass this crate for wgpu-specific code, but reuse the CPU fallback logic.

### ec-gpu (0.2.0)
**Purpose:** GPU trait definitions

**Key exports:**
- `GpuName` trait - marks types with GPU implementations

```rust
pub trait GpuName {
    // Marker trait
}

impl GpuName for blstrs::Scalar {}
impl GpuName for blstrs::G1Affine {}
impl GpuName for blstrs::G2Affine {}
```

### wgpu Integration Strategy

**Plan:** Add wgpu as a peer to cuda/opencl, not through ec-gpu-gen.

**Reasoning:**
1. ec-gpu-gen is tightly coupled to CUDA/OpenCL
2. wgpu has fundamentally different API (async, descriptor-based)
3. WGSL is different from CUDA C / OpenCL C
4. Easier to maintain separate code paths

## GPU Kernel Requirements

### FFT Kernel Requirements

**Input:**
- Multiple coefficient arrays (batch support)
- Omega values (roots of unity) per array
- Log(N) values per array

**Output:**
- Transformed arrays (in-place or separate buffer)

**Operations needed:**
- Bit-reversal permutation
- Butterfly operations (radix-2)
- Twiddle factor multiplication
- Complex field arithmetic

**Batch sizes observed:**
- Typically 3 FFTs at once (A, B, C polynomials)

### Multiexp Kernel Requirements

**Input:**
- Base points (G1Affine or G2Affine) - array of curve points
- Exponents (Scalar::Repr) - array of field elements
- Density map (which exponents are non-zero)
- Skip offset (for source slicing)

**Output:**
- Single curve point (sum of scalar multiplications)

**Algorithm:** Likely Pippenger's algorithm
- Bucket method
- Window size optimization
- Point additions in projective coordinates
- Final conversion to affine

**Challenges:**
- Variable-base MSM (bases different for each proof)
- Density tracking (sparse exponents)
- Large input sizes (10k-1M points typical)

## WGSL / WebGPU Constraints

### WebGPU Limits (to research in Phase 1)
- Max buffer size
- Max workgroup size
- Max storage buffer bindings
- Max compute invocations per workgroup
- Subgroup operations availability

### WGSL Language Features
- No 64-bit integers (use u32 arrays for big integers)
- No recursion
- Limited pointer aliasing
- Structured control flow required
- Compute shaders only (no graphics pipeline needed)

### Browser Compatibility
- Chrome 113+ (stable WebGPU)
- Firefox 126+ (behind flag as of 2024)
- Safari 18+ (WebGPU support)
- Need to test on actual hardware

## Performance Expectations

### CUDA/OpenCL Performance (from logs)
- FFT: GPU significantly faster for large domains (2^20+)
- Multiexp: GPU faster for 10k+ points
- Hybrid CPU/GPU split typical: 10-20% CPU work

### WebGPU Expectations
- **Optimistic:** 80-90% of native GPU performance
- **Realistic:** 50-70% of native GPU performance
- **Pessimistic:** 2-5x faster than CPU (still useful)

**Variables:**
- Browser overhead
- WGSL compiler quality
- Driver maturity
- Hardware support (unified vs discrete GPU)

## Code Size Estimates

Based on CUDA/OpenCL precedent:

| Component | Lines of Code | Complexity |
|-----------|--------------|------------|
| Field arithmetic (WGSL) | 300-500 | High |
| FFT kernel (WGSL) | 400-600 | High |
| Multiexp kernel (WGSL) | 800-1200 | Very High |
| Rust wgpu device mgmt | 200-300 | Medium |
| Rust FFT wrapper | 300-400 | Medium |
| Rust multiexp wrapper | 400-600 | Medium |
| Integration & tests | 500-800 | Medium |
| **Total** | **~3000-4000** | |

## Critical Files Map

### Files to Read (Already Read)
- ✅ `src/groth16/prover/native.rs` - Main proving algorithm
- ✅ `src/groth16/prover/mod.rs` - Prover dispatch
- ✅ `src/groth16/ext.rs` - Public API
- ✅ `src/gpu/mod.rs` - GPU module hub
- ✅ `src/gpu/locks.rs` - Locking system
- ✅ `src/gpu/multiexp.rs` - Hybrid CPU/GPU multiexp
- ✅ `src/gpu/nogpu.rs` - No-GPU polyfill
- ✅ `src/gpu/error.rs` - GPU errors
- ✅ `src/multiexp.rs` - Multiexp dispatch
- ✅ `src/domain.rs` - FFT/evaluation domain
- ✅ `src/build.rs` - Build script
- ✅ `src/lib.rs` - Crate root

### Files to Create (wgpu backend)
- `src/gpu/wgpu/mod.rs`
- `src/gpu/wgpu/device.rs`
- `src/gpu/wgpu/fft.rs`
- `src/gpu/wgpu/multiexp.rs`
- `src/gpu/wgpu/error.rs`
- `src/gpu/wgpu/shaders/field_arithmetic.wgsl`
- `src/gpu/wgpu/shaders/fft.wgsl`
- `src/gpu/wgpu/shaders/multiexp.wgsl`

### Files to Modify (integration)
- `Cargo.toml` - Add wgpu dependencies
- `src/gpu/mod.rs` - Add wgpu feature flag
- `src/multiexp.rs` - Add wgpu dispatch path
- `src/domain.rs` - Add wgpu FFT path
- `build.rs` - Potentially for shader embedding
- `README.md` - Document wgpu feature

### Files to Examine (external)
- `ec-gpu-gen` source (understand kernel structure)
- wgpu examples (device initialization patterns)
- Existing wgpu MSM implementations (if any)

## Open Research Questions

### 1. Multi-limb Arithmetic in WGSL
**Question:** Best representation for 381-bit and 255-bit field elements?

**Options:**
- 8 × u32 (256 bits) + carry handling
- 12 × u32 (384 bits) for Fp
- 4 × u64 emulated as 8 × u32

**Research needed:** Look for existing implementations, benchmark different approaches.

### 2. Pippenger's Algorithm Variants
**Question:** Which Pippenger variant do CUDA/OpenCL implementations use?

**Research needed:**
- Read ec-gpu-gen source
- Check window sizes
- Understand bucket structure

### 3. wgpu Async Handling
**Question:** How to block on wgpu operations without deadlocking?

**Options:**
- `pollster::block_on()` - simple but blocks thread
- `tokio`/`async-std` runtime - heavier dependency
- `wgpu::Device::poll()` in loop - manual polling

**Research needed:** Test different approaches, check wgpu best practices.

### 4. Shader Compilation Caching
**Question:** Can we cache compiled WGSL shaders between runs?

**Research needed:**
- wgpu shader compilation time
- Browser shader caching behavior
- Native pipeline cache support

### 5. WebGPU Feature Detection
**Question:** What WebGPU features/limits do we require?

**Research needed:**
- Minimum buffer sizes needed
- Required compute shader features
- Fallback strategy for limited devices

### 6. G2 Arithmetic Complexity
**Question:** How much slower is G2 multiexp than G1?

**Context:** G2 uses Fp2 (2× the data), but only 2 multiexps vs ~6 for G1.

**Research needed:**
- Profile existing CUDA/OpenCL implementations
- Estimate WGSL performance difference

## References

### Papers & Algorithms
- [Groth16] Jens Groth. "On the Size of Pairing-based Non-interactive Arguments." EUROCRYPT 2016.
- [Pippenger] Nicholas Pippenger. "On the evaluation of powers and monomials." SIAM Journal on Computing, 1980.
- [BLS12-381] Sean Bowe. "BLS12-381: New zk-SNARK Elliptic Curve Construction."

### Code References
- bellperson (this repo)
- ec-gpu-gen: https://github.com/filecoin-project/ec-gpu (Namada fork: nam-ec-gpu-gen)
- wgpu: https://github.com/gfx-rs/wgpu
- blstrs: https://github.com/filecoin-project/blstrs (Namada fork: nam-blstrs)

### WebGPU Resources
- WebGPU Spec: https://www.w3.org/TR/webgpu/
- WGSL Spec: https://www.w3.org/TR/WGSL/
- wgpu Book: https://sotrh.github.io/learn-wgpu/

## Next Research Tasks (Phase 1)

1. Clone and study ec-gpu-gen source code
2. Find and analyze generated CUDA/OpenCL kernel code
3. Research existing wgpu compute shader examples
4. Search for existing wgpu MSM implementations
5. Prototype basic field arithmetic in WGSL
6. Test wgpu device initialization in browser and native
7. Measure WGSL shader compilation time
8. Check WebGPU limits on target hardware
