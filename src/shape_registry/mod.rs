//! Per-position shape-registry assertion (GH-#2 M.7 / pin §3 / spike-
//! conclusion auditor obligation #4).
//!
//! ADR-0021's chunked Strauss-Shamir produces 16 main-loop
//! `Structure<E>` instances (plus ~9 chunk-band shapes; ~25-30 total
//! at production arity). The augmented circuit per IVC step must
//! assert `pp_digest_in == registry[chunk_index_in_z]` to close the
//! cross-position witness substitution attack (pin §3.4 / SuperNova
//! eprint 2022/1758 v3 §3 step-kind discipline).
//!
//! This module ships the in-circuit and off-circuit halves of that
//! assertion as a Gadget pair per `.claude/rules/cryptography.md`:
//!
//! - [`circuit::assert_pp_digest_matches_registry`] — in-circuit
//!   (R1CS-emitting) range-check + constant-time conditional-select
//!   chain + equality assert (pin §3.2). Wired into
//!   [`crate::neutron::circuit::nifs::AllocatedNIFS::verify_with_multi_table_lookup`]
//!   at the surface specified by pin §3.3 (BEFORE
//!   `pp_digest.absorb(ro)`).
//! - [`reference::pp_digest_for_chunk_index`] — off-circuit native
//!   reference, computes the same function via `Vec::get`. Production
//!   code (the augmented-circuit caller in `inumbra-spend-circuits`
//!   uses it for prover-side witness assembly), not test scaffolding.
//!
//! The differential test in this module's `tests` exercises ≥1000
//! reproducibly-seeded `(registry, chunk_index)` pairs and asserts
//! that the in-circuit `is_satisfied()` state matches the
//! reference's `Result::is_ok()` exactly. This satisfies the Gadget
//! contract's "≥1000-iter differential" requirement (cryptography
//! rule, ADR-0013 §4).
#![cfg(feature = "lookup-fold")]

pub mod circuit;
pub mod reference;

pub use circuit::assert_pp_digest_matches_registry;
pub use reference::pp_digest_for_chunk_index;

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{num::AllocatedNum, util_cs::test_cs::TestConstraintSystem, ConstraintSystem},
    provider::Bn256EngineKZG,
    traits::Engine,
  };
  use ff::Field;
  use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
  use rand_core::RngCore;

  type E = Bn256EngineKZG;
  type Scalar = <E as Engine>::Scalar;

  /// Reproducible `(registry, chunk_index_native, index_n_bits,
  /// chunk_index_value, expected_in_range)` sampler driven by
  /// `ChaCha20Rng` per the determinism contract in
  /// `.claude/rules/cryptography.md` ("`Gadget::arbitrary` takes a
  /// `rand_chacha::ChaCha20Rng`. Reseeding with the same seed must
  /// reproduce the same input sequence on any machine.").
  ///
  /// Sample shape: every iteration picks
  /// - `index_n_bits ∈ {4, 5}` (production arities)
  /// - `registry_len ∈ [1, 2^index_n_bits]` (production-realistic;
  ///   includes the `len == 2^n_bits` boundary case)
  /// - `chunk_index_value ∈ [0, 2 * 2^index_n_bits)` to deliberately
  ///   sample some out-of-range cases (~50% in-range / 50% out-of-
  ///   range to cover the full reject domain).
  ///
  /// Returns the registry, the chunk_index as a Scalar, the n_bits,
  /// the raw chunk_index_value (for diagnostic categorisation), and
  /// a boolean indicating whether the reference accepts.
  fn sample_case(rng: &mut ChaCha20Rng) -> (Vec<Scalar>, Scalar, usize, usize, bool) {
    let index_n_bits = if (rng.next_u32() & 1) == 0 { 4usize } else { 5usize };
    let max_len = 1usize << index_n_bits;
    let registry_len = ((rng.next_u32() as usize) % max_len) + 1;
    let registry: Vec<Scalar> = (0..registry_len).map(|_| Scalar::random(&mut *rng)).collect();
    // Sample chunk_index uniformly in [0, 2 * 2^n_bits) to hit:
    //   - in-bit-range AND in-len-range  (accept)
    //   - in-bit-range AND out-len-range (reject via complement)
    //   - out-of-bit-range                 (reject via n-bit decomposition)
    let chunk_index_value = (rng.next_u32() as usize) % (2 * max_len);
    let chunk_index = Scalar::from(chunk_index_value as u64);
    let in_n_bits = chunk_index_value < max_len;
    let in_len_range = chunk_index_value < registry_len;
    let expected_in_range = in_n_bits && in_len_range;
    (registry, chunk_index, index_n_bits, chunk_index_value, expected_in_range)
  }

  /// Differential coverage threshold per `.claude/rules/cryptography.md`
  /// ADR-0013 §4: ≥1000 iterations, reproducibly seeded.
  ///
  /// For every sampled `(registry, chunk_index, index_n_bits)` triple:
  ///
  ///   - Run the off-circuit reference `pp_digest_for_chunk_index`.
  ///     It returns `Ok(registry[chunk_index])` iff the index is
  ///     in-range under both the bit-width and the upper bound.
  ///   - Synthesise the in-circuit `assert_pp_digest_matches_registry`
  ///     with `pp_digest = reference's Ok value` (accept case) or
  ///     `pp_digest = registry[0]` (reject case — emulates the
  ///     "silent select of registry[0]" attack the upper-bound
  ///     complement decomposition closes).
  ///   - Assert `cs.is_satisfied() == reference_result.is_ok()`.
  ///
  /// A satisfy-vs-reference disagreement means either the in-circuit
  /// gadget admits a witness the reference rejects (forgery vector),
  /// or vice versa (reference accepts something the gadget rejects —
  /// usability gap). Per `.claude/rules/cryptography.md`, the right
  /// reflex when these disagree is to investigate the CIRCUIT first
  /// (the reference is the spec).
  ///
  /// Per `.claude/rules/testing.md`, this test runs in `--release`
  /// (debug-mode field arithmetic is 5-10× slower; the differential
  /// would exceed the agent bash timeout in debug).
  #[test]
  fn shape_registry_differential_1000_iters() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_57A6_5BAD_5EE0u64);
    const N: usize = 1000;

    let mut accepts = 0usize;
    let mut rejects_via_n_bits = 0usize;
    let mut rejects_via_upper = 0usize;

    for iter in 0..N {
      let (registry, chunk_index, index_n_bits, chunk_index_value, expected_in_range) =
        sample_case(&mut rng);

      // --- Off-circuit reference ---
      let reference_result =
        pp_digest_for_chunk_index(&registry, chunk_index, index_n_bits);
      assert_eq!(
        reference_result.is_ok(),
        expected_in_range,
        "iter {}: reference accept-vs-expected mismatch (registry.len()={}, n_bits={}, idx_value={}, idx_scalar={:?})",
        iter,
        registry.len(),
        index_n_bits,
        chunk_index_value,
        chunk_index,
      );

      // For accept cases: synthesise with the matching pp_digest from
      // the reference. The constraint system must be satisfied.
      // For reject cases: synthesise with registry[0] as the
      // attacker-supplied pp_digest (the "silent-select-of-zero"
      // attack vector). The constraint system must NOT be satisfied.
      let pp_digest_native = match reference_result {
        Ok(v) => v,
        Err(_) => registry[0],
      };

      let mut cs = TestConstraintSystem::<Scalar>::new();
      let registry_alloc: Vec<_> = registry
        .iter()
        .enumerate()
        .map(|(i, v)| {
          AllocatedNum::alloc(cs.namespace(|| format!("registry[{}]", i)), || Ok(*v)).unwrap()
        })
        .collect();
      let chunk_index_alloc =
        AllocatedNum::alloc(cs.namespace(|| "chunk_index"), || Ok(chunk_index)).unwrap();
      let pp_digest_alloc =
        AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(pp_digest_native)).unwrap();

      assert_pp_digest_matches_registry::<E, _>(
        cs.namespace(|| "assert"),
        &pp_digest_alloc,
        &chunk_index_alloc,
        &registry_alloc,
        index_n_bits,
      )
      .unwrap_or_else(|e| panic!("iter {}: synthesis errored: {:?}", iter, e));

      let satisfied = cs.is_satisfied();
      assert_eq!(
        satisfied,
        expected_in_range,
        "iter {}: in-circuit-vs-reference satisfiability mismatch \
         (registry.len()={}, n_bits={}, idx_value={}, expected_in_range={}, satisfied={})",
        iter,
        registry.len(),
        index_n_bits,
        chunk_index_value,
        expected_in_range,
        satisfied,
      );

      if expected_in_range {
        accepts += 1;
      } else {
        // Categorise rejection cause (informational; the soundness
        // contract is `satisfied == expected_in_range` regardless).
        let max_len = 1usize << index_n_bits;
        if chunk_index_value < max_len {
          rejects_via_upper += 1;
        } else {
          rejects_via_n_bits += 1;
        }
      }
    }

    // Sanity: the sampler is configured to produce a meaningful
    // mixture of accepts and rejects (not all-accept or all-reject —
    // that would mean the differential isn't probing the rejection
    // path). Both sides MUST be substantial (>10%) for the test to
    // be soundness-bearing.
    assert!(
      accepts >= 100,
      "differential sampler produced only {} accept cases out of {}; \
       sampler under-covers the accept path",
      accepts,
      N
    );
    assert!(
      rejects_via_n_bits + rejects_via_upper >= 100,
      "differential sampler produced only {} reject cases out of {}; \
       sampler under-covers the reject path",
      rejects_via_n_bits + rejects_via_upper,
      N
    );
  }

  /// k=1 degenerate case: single-entry registry, chunk_index = 0
  /// must accept. This is the boundary case ensuring the
  /// conditional-select chain's initial `selected = registry[0]`
  /// fires correctly when the loop body is empty. Mirrors the M.6
  /// wire-in fixture at `circuit/lookup.rs` (which uses a single-
  /// entry registry with `pp_digest = Scalar::ZERO`).
  #[test]
  fn single_entry_registry_accepts_index_zero() {
    let registry = vec![Scalar::from(0xDEAD_BEEFu64)];
    let mut cs = TestConstraintSystem::<Scalar>::new();
    let registry_alloc =
      vec![
        AllocatedNum::alloc(cs.namespace(|| "registry[0]"), || Ok(registry[0])).unwrap(),
      ];
    let chunk_index =
      AllocatedNum::alloc(cs.namespace(|| "chunk_index"), || Ok(Scalar::ZERO)).unwrap();
    let pp_digest =
      AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(registry[0])).unwrap();

    assert_pp_digest_matches_registry::<E, _>(
      cs.namespace(|| "assert"),
      &pp_digest,
      &chunk_index,
      &registry_alloc,
      1,
    )
    .expect("synthesis succeeded");

    assert!(cs.is_satisfied());

    // Reference parity at the same input.
    assert_eq!(
      pp_digest_for_chunk_index(&registry, Scalar::ZERO, 1).unwrap(),
      registry[0],
    );
  }

  /// Determinism contract per `.claude/rules/cryptography.md`:
  /// re-seeding produces the same input sequence on any host. This
  /// is the reviewer-reproducibility requirement (US-05).
  #[test]
  fn shape_registry_differential_is_reproducible() {
    let mut rng_a = ChaCha20Rng::seed_from_u64(0xDEAD_BEEF_5EED_50EAu64);
    let mut rng_b = ChaCha20Rng::seed_from_u64(0xDEAD_BEEF_5EED_50EAu64);
    for _ in 0..32 {
      let case_a = sample_case(&mut rng_a);
      let case_b = sample_case(&mut rng_b);
      assert_eq!(case_a.0, case_b.0);
      assert_eq!(case_a.1, case_b.1);
      assert_eq!(case_a.2, case_b.2);
      assert_eq!(case_a.3, case_b.3);
      assert_eq!(case_a.4, case_b.4);
    }
  }
}
