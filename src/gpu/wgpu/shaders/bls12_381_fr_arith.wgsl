// BLS12-381 scalar field Fr arithmetic in WGSL.
//
// Field modulus:
//   r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
//
// Representation: 8 × u32, little-endian (word[0] is the least-significant word).
//
// Montgomery form uses R = 2^256.
//   mu  = -r^{-1} mod 2^32 = 0xffffffff  (since r[0] = 1)
//   R²  = 2^512 mod r  (used for converting TO Montgomery form)
//   R   = 2^256 mod r  (Montgomery representation of 1)
//
// ── naga 0.19 compatibility note ──────────────────────────────────────────
//
// naga 0.19 (bundled with wgpu 0.19) restricts runtime indexing:
//   • Module-scope `const` arrays may NOT be indexed with a runtime variable.
//   • Value-typed function parameters (array<u32, 8>) may NOT be indexed with
//     a runtime variable — they are "value" expressions in the IR.
//
// Workaround applied throughout this file:
//   1. Each array element of the modulus / R² is a separate scalar `const`.
//   2. At the start of every function that indexes an array at runtime, copy
//      both the scalar consts AND every array-typed parameter into a `var`
//      local:
//          var a_l: array<u32, 8> = a;
//          var m:   array<u32, 8> = array<u32, 8>(FM0, FM1, FM2, FM3,
//                                                  FM4, FM5, FM6, FM7);
//      Local `var` arrays are addressable (pointer-class) and CAN be indexed
//      with a runtime variable.
//
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Module-scope scalar constants
// ---------------------------------------------------------------------------

// r as 8 × u32 little-endian limbs
// r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
const FM0: u32 = 0x00000001u;
const FM1: u32 = 0xffffffffu;
const FM2: u32 = 0xfffe5bfeu;
const FM3: u32 = 0x53bda402u;
const FM4: u32 = 0x09a1d805u;
const FM5: u32 = 0x3339d808u;
const FM6: u32 = 0x299d7d48u;
const FM7: u32 = 0x73eda753u;

// R² mod r  (Montgomery constant for fr_to_mont)
// Verified against nam-blstrs 0.7.1 Scalar::R2 (4×u64 LE split into 8×u32 LE):
//   [0xc999e990_f3f29c6d, 0x2b6cedcb_87925c23,
//    0x05d31496_7254398f, 0x0748d9d9_9f59ff11]
const R2_0: u32 = 0xf3f29c6du;
const R2_1: u32 = 0xc999e990u;
const R2_2: u32 = 0x87925c23u;
const R2_3: u32 = 0x2b6cedcbu;
const R2_4: u32 = 0x7254398fu;
const R2_5: u32 = 0x05d31496u;
const R2_6: u32 = 0x9f59ff11u;
const R2_7: u32 = 0x0748d9d9u;

// mu = -r^{-1} mod 2^32.
// r[0] = 1, so r^{-1} mod 2^32 = 1, hence mu = 2^32 - 1 = 0xffffffff.
const FR_MU: u32 = 0xffffffffu;

// ---------------------------------------------------------------------------
// Low-level 32-bit helpers
// ---------------------------------------------------------------------------

// Compute a × b as a 64-bit result (lo, hi):  a × b = hi·2^32 + lo.
fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;

    let w0 = a0 * b0;
    let w1 = a0 * b1;
    let w2 = a1 * b0;
    let w3 = a1 * b1;

    // Cross terms.  m ≤ 0x1fffd, so the carry into hi is at most 1.
    let m  = (w0 >> 16u) + (w1 & 0xffffu) + (w2 & 0xffffu);
    let lo = (w0 & 0xffffu) | (m << 16u);   // m<<16 wraps in u32
    let hi = w3 + (w1 >> 16u) + (w2 >> 16u) + (m >> 16u);
    return vec2<u32>(lo, hi);
}

// Add with carry:  (a + b + cin, carry_out).  carry_out ∈ {0, 1}.
fn adc(a: u32, b: u32, cin: u32) -> vec2<u32> {
    let s1 = a + b;
    let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + cin;
    let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 | c2);   // c1 and c2 cannot both be 1
}

// Subtract with borrow:  (a − b − bin, borrow_out).  borrow_out ∈ {0, 1}.
fn sbb(a: u32, b: u32, bin: u32) -> vec2<u32> {
    let d1 = a - b;
    let b1 = select(0u, 1u, a < b);
    let d2 = d1 - bin;
    let b2 = select(0u, 1u, d1 < bin);
    return vec2<u32>(d2, b1 | b2);
}

// Multiply-accumulate:  t + a×b + cin → (result_lo, carry_out).
//
// Proof that carry_out ≤ 2^32 − 1 (no second-level overflow):
//   prod.y ≤ (2^32−2)   (max hi from mul_wide when a=b=0xffffffff)
//   r0.y   ≤ 1
//   sum    ≤ 2^32 − 1   ✓
fn mac(t: u32, a: u32, b: u32, cin: u32) -> vec2<u32> {
    let prod = mul_wide(a, b);
    let r0   = adc(t, prod.x, cin);
    return vec2<u32>(r0.x, prod.y + r0.y);
}

// ---------------------------------------------------------------------------
// Fr field operations — all inputs/outputs in standard form ∈ [0, r)
// ---------------------------------------------------------------------------

// Conditional subtraction:  if a ≥ r, return a − r, else return a.
fn fr_cond_sub(a: array<u32, 8>) -> array<u32, 8> {
    // Copy value-typed parameter and const scalars to var locals so naga 0.19
    // allows runtime indexing.
    var a_l: array<u32, 8> = a;
    var m:   array<u32, 8> = array<u32, 8>(FM0, FM1, FM2, FM3, FM4, FM5, FM6, FM7);

    var d: array<u32, 8>;
    var borrow = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let r = sbb(a_l[i], m[i], borrow);
        d[i]   = r.x;
        borrow = r.y;
    }
    // borrow == 0  →  a ≥ r  →  use subtracted result d
    // borrow == 1  →  a < r  →  keep a unchanged
    var out: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        out[i] = select(a_l[i], d[i], borrow == 0u);
    }
    return out;
}

// (a + b) mod r
fn fr_add(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    // Copy value-typed parameters to var locals so naga 0.19 allows runtime indexing.
    var a_l: array<u32, 8> = a;
    var b_l: array<u32, 8> = b;

    var t: array<u32, 8>;
    var carry = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let r = adc(a_l[i], b_l[i], carry);
        t[i]  = r.x;
        carry = r.y;
    }
    // a, b ∈ [0, r) and r < 2^255 → a+b < 2^256, overflow carry is always 0.
    return fr_cond_sub(t);
}

// (a − b) mod r
fn fr_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    // Copy value-typed parameters to var locals so naga 0.19 allows runtime indexing.
    var a_l: array<u32, 8> = a;
    var b_l: array<u32, 8> = b;
    var m:   array<u32, 8> = array<u32, 8>(FM0, FM1, FM2, FM3, FM4, FM5, FM6, FM7);

    var t: array<u32, 8>;
    var borrow = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let r = sbb(a_l[i], b_l[i], borrow);
        t[i]   = r.x;
        borrow = r.y;
    }
    // If borrow = 1 the result was negative; add r back.
    if (borrow != 0u) {
        var carry = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let r = adc(t[i], m[i], carry);
            t[i]  = r.x;
            carry = r.y;
        }
    }
    return t;
}

// CIOS Montgomery multiplication:  a_mont × b_mont × R^{−1} mod r.
//
// Inputs must be in [0, r); output is in [0, r).
// Reference: Koç, Acar, Kaliski — "Analyzing and Comparing Montgomery
//            Multiplication Algorithms" (IEEE Micro, 1996).
fn fr_mont_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    // Copy value-typed parameters to var locals so naga 0.19 allows runtime indexing.
    var a_l: array<u32, 8> = a;
    var b_l: array<u32, 8> = b;
    var m:   array<u32, 8> = array<u32, 8>(FM0, FM1, FM2, FM3, FM4, FM5, FM6, FM7);

    // t[0..8] — 9-word working accumulator (the extra word absorbs one carry bit).
    var t: array<u32, 9>;   // zero-initialized per WGSL spec

    for (var i = 0u; i < 8u; i = i + 1u) {

        // ---- Step A: t += a[i] × b ----------------------------------------
        var c = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let r = mac(t[j], a_l[i], b_l[j], c);
            t[j] = r.x;
            c    = r.y;
        }
        // Final carry into t[8].  By CIOS size bounds, t[8] ≤ 1 always.
        let r8a = adc(t[8], c, 0u);
        t[8] = r8a.x;   // r8a.y == 0

        // ---- Step B: Montgomery reduction ----------------------------------
        // q = t[0] × mu mod 2^32.
        // mu = 0xffffffff = −1 mod 2^32, so q = −t[0] mod 2^32 (wrapping).
        let q = 0u - t[0];  // u32 wrapping subtraction

        c = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let r = mac(t[j], q, m[j], c);
            t[j] = r.x;   // t[0] becomes 0 by Montgomery construction
            c    = r.y;
        }

        // ---- Step C: shift accumulator left by one word --------------------
        // t_new[0..6] = t_old[1..7],  t_new[7] = t_old[8] + c_step_b.
        let r8b = adc(t[8], c, 0u);
        for (var j = 0u; j < 7u; j = j + 1u) {
            t[j] = t[j + 1u];
        }
        t[7] = r8b.x;
        t[8] = r8b.y;   // 0 after the last iteration; handled by fr_cond_sub
    }

    // The result is in [0, 2r); one conditional subtraction yields [0, r).
    var result: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result[i] = t[i];
    }
    return fr_cond_sub(result);
}

// Convert standard form → Montgomery form:  a_mont = a × R mod r.
// = fr_mont_mul(a, R²) = a × R² × R^{−1} mod r = a × R mod r.
fn fr_to_mont(a: array<u32, 8>) -> array<u32, 8> {
    var r2: array<u32, 8> = array<u32, 8>(R2_0, R2_1, R2_2, R2_3, R2_4, R2_5, R2_6, R2_7);
    return fr_mont_mul(a, r2);
}

// Convert Montgomery form → standard form:  a_std = a × R^{−1} mod r.
// = fr_mont_mul(a, 1) = a × 1 × R^{−1} mod r.
fn fr_from_mont(a: array<u32, 8>) -> array<u32, 8> {
    var one: array<u32, 8>;   // zero-initialized; only word[0] = 1
    one[0] = 1u;
    return fr_mont_mul(a, one);
}

// Standard-form multiplication:  (a × b) mod r.
fn fr_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    let a_m = fr_to_mont(a);
    let b_m = fr_to_mont(b);
    let c_m = fr_mont_mul(a_m, b_m);
    return fr_from_mont(c_m);
}

// Standard-form negation:  (−a) mod r  =  r − a,  or 0 if a = 0.
fn fr_neg(a: array<u32, 8>) -> array<u32, 8> {
    // Copy value-typed parameter to var local so naga 0.19 allows runtime indexing.
    var a_l: array<u32, 8> = a;

    var all_zero = true;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (a_l[i] != 0u) { all_zero = false; }
    }
    if (all_zero) { return a_l; }
    var m: array<u32, 8> = array<u32, 8>(FM0, FM1, FM2, FM3, FM4, FM5, FM6, FM7);
    return fr_sub(m, a_l);
}
