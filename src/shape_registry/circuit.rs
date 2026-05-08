//! In-circuit shape-registry assertion gadget (GH-#2 M.7 / pin §3.2 +
//! §3.3).
//!
//! This is the in-circuit half of the per-position shape-registry
//! binding that closes the cross-position witness substitution attack
//! (pin §3.4). The gadget asserts:
//!
//! ```text
//! pp_digest_in == registry[chunk_index_in_z]
//! ```
//!
//! where:
//!
//! - `pp_digest_in` is the running instance's `pp_digest` allocation,
//!   threaded through the augmented-circuit IO per the existing
//!   Stage I-app.2+3 binding.
//! - `registry` is a precomputed `Vec<AllocatedNum<E::Scalar>>` of
//!   per-`Structure` `pp_digest`s in chunk-position-canonical order
//!   (NOT `table_id` order — `chunk_index ∈ [0, registry.len())` is
//!   the sole index, per pin §3.1).
//! - `chunk_index_in_z` is the running instance's chunk-position index
//!   carried in the public IO (`X`) per ADR-0021's chunked
//!   Strauss-Shamir.
//!
//! Per pin §3.2, the implementation is:
//!
//! 1. Range-check `chunk_index_in_z` to fit in `index_n_bits` bits AND
//!    to be strictly less than `registry.len()`. Without this, an
//!    out-of-range `chunk_index` would silently select `registry[0]`
//!    (the first entry), opening cross-position substitution to chunk
//!    position 0.
//! 2. Constant-time conditional-select chain over the registry: walk
//!    `i = 1..registry.len()` and select `registry[i]` if `chunk_index
//!    == i`, else carry forward the running `selected`. Initial
//!    `selected = registry[0]`.
//! 3. Equality assert: `pp_digest_in == selected`.
//!
//! Cost at production arity (registry.len() ≤ 30, index_n_bits = 5):
//!
//! - Range-check: ~10 cons (pin §3.2: "5-bit decomposition + an
//!   upper-bound assert").
//! - Conditional-select chain: ~`registry.len()` cons for the equality
//!   bit + ~`registry.len()` cons for the select.
//! - Equality assert: 1 cons.
//!
//! At `registry.len() = 16`, index_n_bits = 4, total ≈ 50 cons —
//! well under pin §3.2's ~100-cons-per-step budget.
#![cfg(feature = "lookup-fold")]

use crate::{
  frontend::{
    num::AllocatedNum, AllocatedBit, ConstraintSystem, LinearCombination, SynthesisError,
  },
  gadgets::utils::{alloc_num_equals, conditionally_select},
  traits::Engine,
};
use ff::{PrimeField, PrimeFieldBits};

/// In-circuit shape-registry assertion (pin §3.2 + §3.3).
///
/// Asserts `pp_digest_in == registry[chunk_index_in_z]`. The caller
/// MUST ensure `registry` is the canonical-order list of per-position
/// `pp_digest`s (the augmented circuit constructs this from
/// `inumbra-spend-circuits` public-params per pin's §6.4).
///
/// `index_n_bits` is the bit-width of `chunk_index_in_z`'s
/// representation. At production arity this is 5 (chunk_index ∈
/// [0, 30)); the test suite exercises `index_n_bits = 4` for the
/// 16-entry main-loop case.
///
/// Returns `Ok(())` if synthesis succeeded; the assertion's
/// satisfaction is verified at `cs.is_satisfied()` time. A wrong
/// `pp_digest_in` (cross-position witness substitution) produces an
/// UNSATISFIED constraint system per pin §3.4.
///
/// # Errors
///
/// Returns `SynthesisError::Unsatisfiable` if `registry` is empty
/// (configuration error — caller must supply a non-empty registry)
/// or if `index_n_bits == 0` (degenerate; would only accept
/// `chunk_index == 0` for a single-entry registry, but the same
/// shape is already captured by `index_n_bits == 1`).
pub fn assert_pp_digest_matches_registry<E, CS>(
  mut cs: CS,
  pp_digest_in: &AllocatedNum<E::Scalar>,
  chunk_index_in_z: &AllocatedNum<E::Scalar>,
  registry: &[AllocatedNum<E::Scalar>],
  index_n_bits: usize,
) -> Result<(), SynthesisError>
where
  E: Engine,
  E::Scalar: PrimeFieldBits,
  CS: ConstraintSystem<E::Scalar>,
{
  if registry.is_empty() {
    return Err(SynthesisError::Unsatisfiable(
      "assert_pp_digest_matches_registry: registry must be non-empty".into(),
    ));
  }
  if index_n_bits == 0 {
    return Err(SynthesisError::Unsatisfiable(
      "assert_pp_digest_matches_registry: index_n_bits must be at least 1".into(),
    ));
  }
  // Bit-width sanity: 2^index_n_bits must be representable in usize on
  // the host. In practice index_n_bits ≤ 5 at production arity.
  if index_n_bits >= usize::BITS as usize {
    return Err(SynthesisError::Unsatisfiable(
      "assert_pp_digest_matches_registry: index_n_bits exceeds usize width".into(),
    ));
  }

  // --- Range-check pin §3.2 ---
  //
  // Two bit-decompositions, both at `index_n_bits` bits:
  //
  //   1. Decompose `chunk_index_in_z` into `index_n_bits` bits and
  //      enforce the unpacking constraint. This proves
  //      `chunk_index_in_z ∈ [0, 2^index_n_bits)`.
  //   2. Decompose `(registry.len() - 1) - chunk_index_in_z` into
  //      `index_n_bits` bits. This proves the same bound on the
  //      complement, which (combined with #1) implies
  //      `chunk_index_in_z ∈ [0, registry.len())`.
  //
  // When `registry.len() == 2^index_n_bits`, step #2 is redundant but
  // not incorrect (the complement also fits in `index_n_bits` bits);
  // we keep it unconditionally for shape uniformity. When
  // `registry.len() < 2^index_n_bits` (production case at index_n_bits
  // = 5, registry.len() ≤ 30), step #2 is the load-bearing
  // upper-bound assert.
  let idx_bits = decompose_to_n_bits(
    cs.namespace(|| "chunk_index range bits"),
    chunk_index_in_z,
    index_n_bits,
  )?;
  drop(idx_bits);

  // Allocate `complement = (registry.len() - 1) - chunk_index_in_z`
  // and enforce it as `index_n_bits`-decomposable. The native
  // assignment derives complement from the chunk_index witness; the
  // R1CS constraint binds the algebraic relation.
  let upper = E::Scalar::from((registry.len() as u64).saturating_sub(1));
  let complement = AllocatedNum::alloc(cs.namespace(|| "complement"), || {
    let v = chunk_index_in_z
      .get_value()
      .ok_or(SynthesisError::AssignmentMissing)?;
    Ok(upper - v)
  })?;
  cs.enforce(
    || "complement = (registry.len() - 1) - chunk_index",
    |lc| lc + complement.get_variable() + chunk_index_in_z.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + (upper, CS::one()),
  );
  let complement_bits = decompose_to_n_bits(
    cs.namespace(|| "complement range bits"),
    &complement,
    index_n_bits,
  )?;
  drop(complement_bits);

  // --- Conditional-select chain pin §3.2 ---
  //
  //   selected = registry[0]
  //   for i in 1..registry.len():
  //       eq_bit = (chunk_index == i)
  //       selected = conditionally_select(eq_bit, registry[i], selected)
  //
  // Constant-time: every `i` is visited regardless of the value of
  // `chunk_index`. The equality bit and the select are both
  // R1CS-enforced.
  let mut selected = registry[0].clone();
  for i in 1..registry.len() {
    let i_const = AllocatedNum::alloc(
      cs.namespace(|| format!("registry index {} constant", i)),
      || Ok(E::Scalar::from(i as u64)),
    )?;
    cs.enforce(
      || format!("registry index {} constant binding", i),
      |lc| lc + i_const.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + (E::Scalar::from(i as u64), CS::one()),
    );
    let eq_bit = alloc_num_equals(
      cs.namespace(|| format!("chunk_index == {} ?", i)),
      chunk_index_in_z,
      &i_const,
    )?;
    selected = conditionally_select(
      cs.namespace(|| format!("select registry[{}] if eq", i)),
      &registry[i],
      &selected,
      &eq_bit.into(),
    )?;
  }

  // --- Equality assert pin §3.2 ---
  cs.enforce(
    || "pp_digest_in == registry[chunk_index]",
    |lc| lc + pp_digest_in.get_variable() - selected.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );

  Ok(())
}

/// Decompose `value` into exactly `n_bits` bits (little-endian) and
/// enforce the unpacking constraint. Returns the allocated bits.
///
/// This is the n-bit range-check primitive: a satisfying assignment
/// exists iff `value` represents an integer in `[0, 2^n_bits)`.
fn decompose_to_n_bits<F, CS>(
  mut cs: CS,
  value: &AllocatedNum<F>,
  n_bits: usize,
) -> Result<Vec<AllocatedBit>, SynthesisError>
where
  F: PrimeField + PrimeFieldBits,
  CS: ConstraintSystem<F>,
{
  // Compute the bit pattern of `value` truncated to `n_bits`. The
  // bit-decomposition constraint below will reject any assignment
  // where `value` does NOT fit in `n_bits` bits.
  let value_bits: Option<Vec<bool>> = value.get_value().map(|v| {
    let le = v.to_le_bits();
    le.into_iter().take(n_bits).collect()
  });

  let mut bits = Vec::with_capacity(n_bits);
  for i in 0..n_bits {
    let bit_value = value_bits.as_ref().map(|bs| bs[i]);
    let bit = AllocatedBit::alloc(cs.namespace(|| format!("bit {}", i)), bit_value)?;
    bits.push(bit);
  }

  // Enforce sum_{i} 2^i * bit_i == value.
  let mut lc = LinearCombination::zero();
  let mut coeff = F::ONE;
  for bit in &bits {
    lc = lc + (coeff, bit.get_variable());
    coeff = coeff.double();
  }
  lc = lc - value.get_variable();
  cs.enforce(|| "n-bit unpacking", |lc| lc, |lc| lc, |_| lc);

  Ok(bits)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::util_cs::test_cs::TestConstraintSystem,
    provider::Bn256EngineKZG,
    traits::Engine,
  };
  use ff::Field;
  type E = Bn256EngineKZG;
  type Scalar = <E as Engine>::Scalar;

  fn mk_registry(n: usize) -> Vec<Scalar> {
    (0..n).map(|i| Scalar::from((i * 31 + 7) as u64)).collect()
  }

  /// Allocate a registry of `AllocatedNum`s from a slice of native
  /// scalars in a TestConstraintSystem.
  fn alloc_registry(
    cs: &mut TestConstraintSystem<Scalar>,
    registry: &[Scalar],
  ) -> Vec<AllocatedNum<Scalar>> {
    registry
      .iter()
      .enumerate()
      .map(|(i, v)| {
        AllocatedNum::alloc(cs.namespace(|| format!("registry[{}]", i)), || Ok(*v)).unwrap()
      })
      .collect()
  }

  #[test]
  fn synthesises_satisfied_for_valid_index_and_matching_pp_digest() {
    let registry = mk_registry(16);
    for i in 0..16 {
      let mut cs = TestConstraintSystem::<Scalar>::new();
      let registry_alloc = alloc_registry(&mut cs, &registry);
      let chunk_index =
        AllocatedNum::alloc(cs.namespace(|| "chunk_index"), || Ok(Scalar::from(i as u64)))
          .unwrap();
      let pp_digest =
        AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(registry[i])).unwrap();

      assert_pp_digest_matches_registry::<E, _>(
        cs.namespace(|| "assert"),
        &pp_digest,
        &chunk_index,
        &registry_alloc,
        4,
      )
      .expect("synthesis succeeded");

      assert!(
        cs.is_satisfied(),
        "valid index {} must produce satisfied constraint system; \
         first unsatisfied: {:?}",
        i,
        cs.which_is_unsatisfied()
      );
    }
  }

  #[test]
  fn rejects_wrong_pp_digest_for_valid_index() {
    let registry = mk_registry(16);
    let mut cs = TestConstraintSystem::<Scalar>::new();
    let registry_alloc = alloc_registry(&mut cs, &registry);
    let chunk_index =
      AllocatedNum::alloc(cs.namespace(|| "chunk_index"), || Ok(Scalar::from(3u64))).unwrap();
    // Wrong pp_digest: claim position 3 but supply position 7's digest.
    let pp_digest =
      AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(registry[7])).unwrap();

    assert_pp_digest_matches_registry::<E, _>(
      cs.namespace(|| "assert"),
      &pp_digest,
      &chunk_index,
      &registry_alloc,
      4,
    )
    .expect("synthesis should not error (assertion is at constraint layer)");

    assert!(
      !cs.is_satisfied(),
      "wrong pp_digest must produce UNSATISFIED constraint system"
    );
  }

  #[test]
  fn rejects_index_above_registry_len() {
    // Registry of 16 entries, index = 16 (above bound). With
    // index_n_bits = 5, the n-bit decomposition itself accepts 16
    // (0b10000), but the upper-bound complement decomposition
    // detects: complement = (16 - 1) - 16 = -1 (mod p), which is
    // NOT representable in 5 bits.
    let registry = mk_registry(16);
    let mut cs = TestConstraintSystem::<Scalar>::new();
    let registry_alloc = alloc_registry(&mut cs, &registry);
    let chunk_index =
      AllocatedNum::alloc(cs.namespace(|| "chunk_index"), || Ok(Scalar::from(16u64)))
        .unwrap();
    // Use registry[0]'s digest as a placeholder pp_digest (this is
    // what an attacker would supply, hoping the silent-select-of-
    // registry[0] attack succeeds).
    let pp_digest =
      AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(registry[0])).unwrap();

    assert_pp_digest_matches_registry::<E, _>(
      cs.namespace(|| "assert"),
      &pp_digest,
      &chunk_index,
      &registry_alloc,
      5,
    )
    .expect("synthesis should not error");

    assert!(
      !cs.is_satisfied(),
      "out-of-range chunk_index must produce UNSATISFIED constraint system"
    );
  }

  #[test]
  fn rejects_index_exceeding_n_bits() {
    // Registry of 16 entries, index = 17 (exceeds n_bits = 4 AND
    // exceeds registry.len()). With index_n_bits = 4, the n-bit
    // decomposition itself fails.
    let registry = mk_registry(16);
    let mut cs = TestConstraintSystem::<Scalar>::new();
    let registry_alloc = alloc_registry(&mut cs, &registry);
    let chunk_index =
      AllocatedNum::alloc(cs.namespace(|| "chunk_index"), || Ok(Scalar::from(17u64)))
        .unwrap();
    let pp_digest =
      AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(registry[0])).unwrap();

    assert_pp_digest_matches_registry::<E, _>(
      cs.namespace(|| "assert"),
      &pp_digest,
      &chunk_index,
      &registry_alloc,
      4,
    )
    .expect("synthesis should not error");

    assert!(
      !cs.is_satisfied(),
      "chunk_index exceeding n_bits must produce UNSATISFIED constraint system"
    );
  }

  #[test]
  fn empty_registry_is_synthesis_error() {
    let mut cs = TestConstraintSystem::<Scalar>::new();
    let chunk_index =
      AllocatedNum::alloc(cs.namespace(|| "chunk_index"), || Ok(Scalar::ZERO)).unwrap();
    let pp_digest =
      AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(Scalar::ZERO)).unwrap();

    let result = assert_pp_digest_matches_registry::<E, _>(
      cs.namespace(|| "assert"),
      &pp_digest,
      &chunk_index,
      &[],
      4,
    );
    assert!(matches!(result, Err(SynthesisError::Unsatisfiable(_))));
  }

  #[test]
  fn production_arity_5bit_30_entries() {
    let registry = mk_registry(30);
    for i in 0..30 {
      let mut cs = TestConstraintSystem::<Scalar>::new();
      let registry_alloc = alloc_registry(&mut cs, &registry);
      let chunk_index = AllocatedNum::alloc(cs.namespace(|| "chunk_index"), || {
        Ok(Scalar::from(i as u64))
      })
      .unwrap();
      let pp_digest =
        AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(registry[i])).unwrap();

      assert_pp_digest_matches_registry::<E, _>(
        cs.namespace(|| "assert"),
        &pp_digest,
        &chunk_index,
        &registry_alloc,
        5,
      )
      .expect("synthesis succeeded");

      assert!(
        cs.is_satisfied(),
        "production-arity index {} must produce satisfied constraint system",
        i
      );
    }
  }
}
