// Test entry point for BLS12-381 Fr field arithmetic.
//
// This shader must be prepended with bls12_381_fr_arith.wgsl at compile time
// (via concat! + include_str! in Rust) to form a complete WGSL program.
//
// Layout:
//   binding 0 — in_a:  array<u32>  (n × 8 words, little-endian Fr elements)
//   binding 1 — in_b:  array<u32>  (n × 8 words)
//   binding 2 — out:   array<u32>  (n × 8 words, writable result)
//   binding 3 — params: uniform { n: u32, op: u32, _pad: vec2<u32> }
//
// op codes:
//   0 — fr_mul(a, b)
//   1 — fr_add(a, b)
//   2 — fr_sub(a, b)

struct FrTestParams {
    n:    u32,
    op:   u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read>       in_a:   array<u32>;
@group(0) @binding(1) var<storage, read>       in_b:   array<u32>;
@group(0) @binding(2) var<storage, read_write> fr_out: array<u32>;
@group(0) @binding(3) var<uniform>             params: FrTestParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }

    // Load operands (8 words each, little-endian).
    var a: array<u32, 8>;
    var b: array<u32, 8>;
    for (var k = 0u; k < 8u; k = k + 1u) {
        a[k] = in_a[idx * 8u + k];
        b[k] = in_b[idx * 8u + k];
    }

    // Dispatch to the requested operation.
    var c: array<u32, 8>;
    if (params.op == 0u) {
        c = fr_mul(a, b);
    } else if (params.op == 1u) {
        c = fr_add(a, b);
    } else {
        c = fr_sub(a, b);
    }

    // Store result.
    for (var k = 0u; k < 8u; k = k + 1u) {
        fr_out[idx * 8u + k] = c[k];
    }
}
