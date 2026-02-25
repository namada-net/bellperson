//! Helpers for packing field representations into GPU word buffers.

use ff::PrimeField;

#[inline]
pub fn repr_word_len<F: PrimeField>() -> u32 {
    F::Repr::default().as_ref().len().div_ceil(4) as u32
}

pub fn pack_field_words<F: PrimeField>(values: &[F]) -> Vec<u32> {
    let mut words = Vec::<u32>::new();
    for v in values {
        let repr = v.to_repr();
        words.extend(pack_repr_bytes(repr.as_ref()));
    }
    words
}

pub fn pack_primefield_repr_words<F: PrimeField>(reprs: &[F::Repr]) -> Vec<u32> {
    let mut words = Vec::<u32>::new();
    for r in reprs {
        words.extend(pack_repr_bytes(r.as_ref()));
    }
    words
}

pub fn omega_seed_from_repr_bytes(bytes: &[u8]) -> u32 {
    bytes.chunks(4).fold(0u32, |acc, chunk| {
        let mut w = 0u32;
        for (i, b) in chunk.iter().enumerate() {
            w |= (*b as u32) << (8 * i);
        }
        acc ^ w
    })
}

fn pack_repr_bytes(bytes: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(bytes.len().div_ceil(4));
    for chunk in bytes.chunks(4) {
        let mut w = 0u32;
        for (i, b) in chunk.iter().enumerate() {
            w |= (*b as u32) << (8 * i);
        }
        out.push(w);
    }
    out
}
