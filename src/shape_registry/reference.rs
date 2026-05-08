//! Off-circuit reference for the per-position shape-registry assertion
//! (GH-#2 M.7 / pin §3.2).
//!
//! Per `.claude/rules/cryptography.md`, every primitive ships an in-circuit
//! half AND an off-circuit reference, structurally parallel. The reference
//! is production code (the augmented-circuit caller in
//! `inumbra-spend-circuits` derives the actual `selected` `pp_digest`
//! through it for prover-side witness assembly), not test scaffolding.
//!
//! The reference computes the same function as the in-circuit
//! [`crate::shape_registry::circuit::assert_pp_digest_matches_registry`]:
//! given a `registry: &[F]` of per-position `pp_digest`s and a
//! `chunk_index ∈ [0, registry.len())`, it returns
//! `Ok(registry[chunk_index])` and otherwise rejects with `Err`.
#![cfg(feature = "lookup-fold")]

use ff::PrimeField;

/// Off-circuit shape-registry indexing.
///
/// Returns `Ok(registry[chunk_index])` iff `chunk_index` decodes to an
/// integer in `[0, registry.len())` AND fits in `index_n_bits` bits.
/// Otherwise returns `Err` with a short reason.
///
/// **Soundness contract** (pin §3.2): the in-circuit half asserts
/// `pp_digest_in == registry[chunk_index]` after a range-check on
/// `chunk_index`. The reference's contract is the same equation, plus
/// the same bit-width and upper-bound conditions, so a synthesise-vs-
/// evaluate differential probes both halves on the same domain.
///
/// `index_n_bits` MUST be small enough that `2^index_n_bits` fits in
/// the field; in practice the augmented circuit calls this with
/// `index_n_bits = 5` (production arity, registry size ≤ 30 per
/// ADR-0021). The reference does NOT assume `index_n_bits ≤ 64` —
/// it goes through `PrimeField::to_repr` so it works at any width.
pub fn pp_digest_for_chunk_index<F: PrimeField>(
  registry: &[F],
  chunk_index: F,
  index_n_bits: usize,
) -> Result<F, &'static str> {
  if registry.is_empty() {
    return Err("pp_digest_for_chunk_index: empty registry");
  }

  // Range-check: chunk_index must fit in `index_n_bits` bits.
  super::super::lookup::reference::range_check_native(chunk_index, index_n_bits)
    .map_err(|_| "pp_digest_for_chunk_index: chunk_index out of n_bits range")?;

  // Decode chunk_index to a usize for indexing. We rely on the
  // range-check above ensuring `chunk_index < 2^index_n_bits`. If the
  // decoded value exceeds `registry.len()`, that is the upper-bound
  // failure (the in-circuit half's explicit upper-bound assert; here
  // it is the `usize` indexing rejection).
  let idx = decode_small_index(chunk_index, index_n_bits)?;

  if idx >= registry.len() {
    return Err("pp_digest_for_chunk_index: chunk_index >= registry.len()");
  }

  Ok(registry[idx])
}

/// Decode a field element known to fit in `n_bits` bits to a `usize`.
///
/// Returns `Err` if the field element does not fit in a `usize` (i.e.,
/// `n_bits` exceeds `usize::BITS`); under production usage this never
/// fires because `n_bits ≤ 5`.
fn decode_small_index<F: PrimeField>(
  value: F,
  n_bits: usize,
) -> Result<usize, &'static str> {
  if n_bits > usize::BITS as usize {
    return Err("decode_small_index: n_bits exceeds usize width");
  }
  let repr = value.to_repr();
  let bytes = repr.as_ref();
  // Little-endian byte order (RustCrypto canonical).
  let mut idx: usize = 0;
  let n_full_bytes = n_bits / 8;
  let partial_bits = n_bits % 8;
  for (i, b) in bytes.iter().enumerate() {
    if i < n_full_bytes {
      idx |= (*b as usize) << (i * 8);
      continue;
    }
    if i == n_full_bytes && partial_bits > 0 {
      let mask = (1u8 << partial_bits) - 1;
      idx |= ((*b & mask) as usize) << (i * 8);
      continue;
    }
    // Beyond boundary — must be zero (caller already range-checked).
    if *b != 0 {
      return Err("decode_small_index: value exceeds 2^n_bits");
    }
  }
  Ok(idx)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::Bn256EngineKZG;
  use crate::traits::Engine;
  use ff::Field;
  type Scalar = <Bn256EngineKZG as Engine>::Scalar;

  fn mk_registry(n: usize) -> Vec<Scalar> {
    (0..n).map(|i| Scalar::from((i * 31 + 7) as u64)).collect()
  }

  #[test]
  fn returns_indexed_entry_for_valid_index() {
    let registry = mk_registry(16);
    for i in 0..16 {
      let result = pp_digest_for_chunk_index(&registry, Scalar::from(i as u64), 4)
        .expect("valid index must succeed");
      assert_eq!(result, registry[i]);
    }
  }

  #[test]
  fn rejects_index_at_or_above_len() {
    let registry = mk_registry(16);
    // index = 16 fits in 5 bits but is out-of-range for len=16.
    let result = pp_digest_for_chunk_index(&registry, Scalar::from(16u64), 5);
    assert!(result.is_err());
  }

  #[test]
  fn rejects_index_exceeding_n_bits() {
    let registry = mk_registry(16);
    // index = 16 does NOT fit in 4 bits.
    let result = pp_digest_for_chunk_index(&registry, Scalar::from(16u64), 4);
    assert!(result.is_err());
  }

  #[test]
  fn rejects_empty_registry() {
    let registry: Vec<Scalar> = vec![];
    let result = pp_digest_for_chunk_index(&registry, Scalar::ZERO, 4);
    assert!(result.is_err());
  }

  #[test]
  fn production_arity_5bit_30_entries() {
    let registry = mk_registry(30);
    for i in 0..30 {
      let result = pp_digest_for_chunk_index(&registry, Scalar::from(i as u64), 5)
        .expect("production-arity index must succeed");
      assert_eq!(result, registry[i]);
    }
    // index = 30 fits in 5 bits but registry.len() = 30 → out-of-range.
    assert!(pp_digest_for_chunk_index(&registry, Scalar::from(30u64), 5).is_err());
    // index = 31 fits in 5 bits but registry.len() = 30 → out-of-range.
    assert!(pp_digest_for_chunk_index(&registry, Scalar::from(31u64), 5).is_err());
    // index = 32 does NOT fit in 5 bits.
    assert!(pp_digest_for_chunk_index(&registry, Scalar::from(32u64), 5).is_err());
  }

  #[test]
  fn rejects_arbitrary_field_element_above_n_bits() {
    let registry = mk_registry(16);
    // Random small-ish value that's still > 2^4.
    let result = pp_digest_for_chunk_index(&registry, Scalar::from(100u64), 4);
    assert!(result.is_err());
  }
}
