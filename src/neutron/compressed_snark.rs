//! NeutronNova-side compressed SNARK envelope.
//!
//! Authored per GH-#7 / Stage K design pin Corrigenda #7+#8+#9. This module
//! lands incrementally across M.GH7.0.1 (this commit; Pedersen MSM-linearity
//! split-E commitment helper) → M.GH7.0.2 (envelope + verifier off-FS
//! binding check) → M.GH7.4 (LogUp identity composition).
//!
//! # M.GH7.0.1 scope (this commit)
//!
//! Provides [`split_E_commitments`], a helper that splits a `FoldedWitness`'s
//! flat `E: Vec<E::Scalar>` (length `left + right`) into two halves
//! `(E1, E2)` of lengths `left` and `right`, and emits commitments
//! `(comm_E1, comm_E2)` against disjoint prefix/suffix slices of the same
//! flat commitment-key generator vector that the existing
//! `commit(ck, &W.E, &W.r_E)` uses. A blinding split `r_E1 + r_E2 == W.r_E`
//! is also emitted so that the Pedersen-additive identity
//! `comm_E1 + comm_E2 == U.comm_E` holds by construction.
//!
//! # Soundness anchor
//!
//! Pedersen MSM-linearity at `vendor/nova/src/provider/pedersen.rs:285-292`:
//!
//! ```ignore
//! Commitment {
//!   comm: E::GE::vartime_multiscalar_mul(v, &ck.ck[..v.len()])
//!     + <E::GE as DlogGroup>::group(&ck.h) * r,
//! }
//! ```
//!
//! For a flat `v = [E1 || E2]` of length `left + right`, the MSM
//! decomposes additively along disjoint generator-vector slices:
//!
//! ```text
//! MSM([E1 || E2], ck.ck[..left+right])
//!   = MSM(E1, ck.ck[..left]) + MSM(E2, ck.ck[left..left+right])
//! ```
//!
//! Combined with `(h * r_E1) + (h * r_E2) = h * (r_E1 + r_E2) = h * W.r_E`,
//! this gives `comm_E1 + comm_E2 = U.comm_E` whenever
//! `r_E1 + r_E2 == W.r_E`.
//!
//! # Generator-vector alignment
//!
//! The helper relies on `CE::commit(ck, &v, &r)` using `ck.ck[..v.len()]` —
//! i.e., the first `v.len()` generators in flat order. For
//! `comm_E1 = CE::commit(ck, E1, r_E1)` this is `ck.ck[..left]` directly.
//! For `comm_E2` the helper passes a length-`(left + right)` vector
//! `[zeros(left) || E2]` so that `CE::commit` selects `ck.ck[..left+right]`,
//! and the zero-prefix contributes zero to the MSM — leaving exactly
//! `MSM(E2, ck.ck[left..left+right]) + h * r_E2`. Both halves therefore
//! commit against the same flat generator vector that `U.comm_E` is
//! committed against; the additive identity is byte-equal at the group
//! level (Corrigendum #8 §1.2(a) "MSM-linear over flat `ck.ck` generator
//! vector").
//!
//! # Blinding-split discipline
//!
//! The helper draws `r_E1` from `OsRng` (matching the
//! `RecursiveSNARK::{new, prove_step}` precedent at
//! `vendor/nova/src/neutron/mod.rs:479, 550`) and emits
//! `r_E2 = W.r_E - r_E1`. The sum identity `r_E1 + r_E2 == W.r_E` holds
//! algebraically; both halves are independently uniformly distributed in
//! the group's exponent space (Pedersen hiding preserved on both halves
//! independently).
//!
//! # Consumer integration (M.GH7.0.0 carry-forward)
//!
//! The Spartan-side `prove_with_split_error` sibling at
//! `vendor/nova/src/spartan/snark.rs` (M.GH7.0.0) operates on
//! **derandomized** commitments (mirroring
//! `vendor/nova/src/spartan/direct.rs:159-175`). This helper itself emits
//! **blinded** commitments preserving `r_E1 + r_E2 == W.r_E`. The
//! M.GH7.0.2 envelope is responsible for the derandomize-before-Spartan-
//! prove dance: it will call this helper, then `CE::derandomize` the
//! results with `(r_E1, r_E2)`, then feed the derandomized commitments
//! into `prove_with_split_error`. Keeping the helper blinded preserves
//! the Pedersen-additive identity at the M.GH7.0.1 acceptance-test
//! callsite (where `U.comm_E` is also blinded), and gives M.GH7.0.2 the
//! information it needs to derandomize correctly.

use crate::{
  neutron::relation::{FoldedInstance, FoldedWitness, Structure},
  provider::traits::DlogGroup,
  traits::{commitment::CommitmentEngineTrait, Engine},
  Commitment, CommitmentKey,
};
use ff::Field;
use rand_core::OsRng;

/// Pedersen MSM-linearity split-E commitment helper (M.GH7.0.1 per
/// Corrigendum #8 §1.2(a)).
///
/// Given a `FoldedWitness<E>` carrying flat `E: Vec<E::Scalar>` of length
/// `structure.left + structure.right` with blinding `W.r_E`, splits `E`
/// into `(E1, E2)` at `structure.left` and emits:
///
/// - `comm_E1 = MSM(E1, ck.ck[..left]) + h * r_E1`,
/// - `comm_E2 = MSM(E2, ck.ck[left..left+right]) + h * r_E2`,
///
/// with blinding-split discipline `r_E1 + r_E2 == W.r_E` so that the
/// Pedersen-additive identity `comm_E1 + comm_E2 == U.comm_E` holds by
/// construction.
///
/// See module-level documentation for the soundness anchor, generator-
/// vector-alignment argument, and M.GH7.0.0/0.2 envelope integration.
///
/// # Panics
///
/// Panics if `W.E.len() != structure.left + structure.right`. This is the
/// `FoldedWitness::default` invariant (`vendor/nova/src/neutron/relation.rs:611`)
/// and is structurally maintained by every fold step.
#[allow(non_snake_case)]
pub fn split_E_commitments<E: Engine>(
  ck: &CommitmentKey<E>,
  W: &FoldedWitness<E>,
  _U: &FoldedInstance<E>,
  structure: &Structure<E>,
) -> (Commitment<E>, Commitment<E>, E::Scalar, E::Scalar)
where
  E::GE: DlogGroup,
{
  assert_eq!(
    W.E.len(),
    structure.left + structure.right,
    "FoldedWitness.E length must equal structure.left + structure.right",
  );

  let (E1, E2) = W.E.split_at(structure.left);

  // Blinding-split discipline: r_E1 fresh from OsRng; r_E2 = W.r_E - r_E1.
  // Preserves Pedersen hiding on both halves independently. The sum
  // identity r_E1 + r_E2 == W.r_E holds algebraically.
  let r_E1 = E::Scalar::random(&mut OsRng);
  let r_E2 = W.r_E - r_E1;

  // comm_E1 = MSM(E1, ck.ck[..left]) + h * r_E1.
  // `CE::commit(ck, v, r)` uses `ck.ck[..v.len()]`, so with `v.len() = left`
  // this commits against the first `left` generators directly.
  let comm_E1 = E::CE::commit(ck, E1, &r_E1);

  // comm_E2 = MSM(E2, ck.ck[left..left+right]) + h * r_E2.
  // We construct a length-(left+right) scalar vector `[zeros(left) || E2]`
  // so that `CE::commit` selects `ck.ck[..left+right]` (matching the flat
  // generator slice that `U.comm_E` is committed against). The zero-prefix
  // contributes zero to the MSM, leaving exactly the desired suffix MSM
  // plus `h * r_E2`.
  let mut e2_padded = vec![E::Scalar::ZERO; structure.left + structure.right];
  e2_padded[structure.left..].copy_from_slice(E2);
  let comm_E2 = E::CE::commit(ck, &e2_padded, &r_E2);

  (comm_E1, comm_E2, r_E1, r_E2)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      Circuit, ConstraintSystem,
    },
    provider::{hyperkzg::EvaluationEngine, Bn256EngineKZG},
    r1cs::R1CSShape,
    spartan::{
      direct::DirectCircuit,
      math::Math,
      polys::eq::EqPolynomial,
      snark::RelaxedR1CSSNARK,
    },
    traits::{circuit::NonTrivialCircuit, snark::RelaxedR1CSSNARKTrait},
  };
  use rand::rngs::OsRng as RandOsRng;
  use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

  /// Build a satisfying `(Structure, FoldedInstance, FoldedWitness)` triple
  /// against a `NonTrivialCircuit` with `num_cons = 16`, matching the
  /// precedent of `test_sat_inner` at
  /// `vendor/nova/src/neutron/relation.rs:942-1017`. Gives
  /// `left = right = 4` (`log2(16) = 4`, `ell1 = ell2 = 2`).
  #[allow(non_snake_case)]
  fn build_satisfying_triple<E, S>() -> (
    CommitmentKey<E>,
    Structure<E>,
    FoldedInstance<E>,
    FoldedWitness<E>,
  )
  where
    E: Engine,
    S: RelaxedR1CSSNARKTrait<E>,
    E::GE: DlogGroup,
  {
    let num_cons: usize = 16;
    let log_num_cons = num_cons.log_2();

    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<E::Scalar>::new(num_cons));

    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();
    let structure = Structure::new(&shape);

    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(2)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (u, w) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    let coords = (0..log_num_cons)
      .map(|_| E::Scalar::random(&mut RandOsRng))
      .collect::<Vec<_>>();
    // EqPolynomial::evals returns a flat `Vec<E::Scalar>` of length
    // `2^log_num_cons`. For the Neutron split-E construction we want the
    // FLAT vector `E` of length `left + right` (NOT `left * right`) — see
    // `FoldedWitness::default` at `relation.rs:611`. We therefore build a
    // length-`(left + right)` vector by concatenating the first `left`
    // and the first `right` entries of the eq-poly evaluation (any flat
    // `E` of the correct length suffices; the helper's contract is
    // shape-only, not algebraic-content-bound). The `is_sat` check at
    // `relation.rs:563-602` expects the outer-product reconstruction
    // `full_E[i*left+j] = E2[i] * E1[j]`, but `is_sat` is not the
    // contract under test here — the Pedersen-additive identity is
    // shape-only.
    let evals = EqPolynomial::new(coords).evals();
    let mut e_flat = Vec::with_capacity(structure.left + structure.right);
    e_flat.extend_from_slice(&evals[..structure.left]);
    e_flat.extend_from_slice(&evals[..structure.right]);

    let mut W_vec = w.W.clone();
    W_vec.resize(structure.S.num_vars, E::Scalar::ZERO);

    let r_E = E::Scalar::random(&mut RandOsRng);

    let W = FoldedWitness {
      W: W_vec,
      r_W: w.r_W,
      E: e_flat.clone(),
      r_E,
    };

    let U = FoldedInstance {
      comm_W: u.comm_W,
      comm_E: E::CE::commit(&ck, &e_flat, &r_E),
      T: E::Scalar::ZERO,
      X: u.X.clone(),
      u: E::Scalar::ONE,
      #[cfg(feature = "lookup-fold")]
      comm_L: None,
      #[cfg(feature = "lookup-fold")]
      comm_ts: None,
      #[cfg(feature = "lookup-fold")]
      comm_inv_w: None,
      #[cfg(feature = "lookup-fold")]
      comm_inv_t: None,
      #[cfg(feature = "lookup-fold")]
      T_lookup: None,
    };

    (ck, structure, U, W)
  }

  /// **Acceptance test** (M.GH7.0.1).
  ///
  /// Asserts the Pedersen-additive binding `comm_E1 + comm_E2 == U.comm_E`
  /// byte-equal at the group level, AND the blinding-split discipline
  /// `r_E1 + r_E2 == W.r_E`, against a satisfying
  /// `(Structure, FoldedInstance, FoldedWitness)` triple.
  ///
  /// This is the M.GH7.0.1 acceptance gate — Corrigendum #8 §1.2(a) MSM-
  /// linearity over the flat `ck.ck` generator vector.
  #[test]
  #[allow(non_snake_case)]
  fn m_gh7_0_1_split_E_commitments_pedersen_additive_binding_byte_equal() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

    let (ck, structure, U, W) = build_satisfying_triple::<E, S>();

    let (comm_E1, comm_E2, r_E1, r_E2) = split_E_commitments(&ck, &W, &U, &structure);

    // Pedersen-additive identity (byte-equal at the group level).
    assert_eq!(
      comm_E1 + comm_E2,
      U.comm_E,
      "Pedersen-additive identity violated: comm_E1 + comm_E2 != U.comm_E",
    );

    // Blinding-split discipline.
    assert_eq!(
      r_E1 + r_E2,
      W.r_E,
      "Blinding-split discipline violated: r_E1 + r_E2 != W.r_E",
    );
  }

  /// **Unit test**: MSM-linearity prefix/suffix decomposition over a
  /// deterministic ChaCha20Rng-seeded input. Verifies the structural
  /// identity `MSM(E1, ck.ck[..left]) + MSM(E2, ck.ck[left..left+right])
  /// == MSM([E1 || E2], ck.ck[..left+right])` against the same flat `ck`
  /// that the helper consumes.
  ///
  /// This is a 1000-iter differential against random `(E1, E2, r_E)`
  /// triples — the structural identity holds for every input.
  #[test]
  #[allow(non_snake_case)]
  fn msm_linearity_prefix_suffix_decomposition_byte_equal_1000_iter() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

    let (ck, structure, _U, W_template) = build_satisfying_triple::<E, S>();

    let mut rng = ChaCha20Rng::from_seed([0u8; 32]);

    for _iter in 0..1000 {
      // Generate random (E1, E2) of the correct shape.
      let e_flat: Vec<<E as Engine>::Scalar> = (0..(structure.left + structure.right))
        .map(|_| <<E as Engine>::Scalar as Field>::random(&mut rng))
        .collect();
      let r_E = <<E as Engine>::Scalar as Field>::random(&mut rng);

      let W = FoldedWitness {
        W: W_template.W.clone(),
        r_W: W_template.r_W,
        E: e_flat.clone(),
        r_E,
      };
      // Reuse the U.comm_E flat-commit as the byte-equal target.
      let U = FoldedInstance {
        comm_W: _U.comm_W,
        comm_E: <E as Engine>::CE::commit(&ck, &e_flat, &r_E),
        T: _U.T,
        X: _U.X.clone(),
        u: _U.u,
        #[cfg(feature = "lookup-fold")]
        comm_L: None,
        #[cfg(feature = "lookup-fold")]
        comm_ts: None,
        #[cfg(feature = "lookup-fold")]
        comm_inv_w: None,
        #[cfg(feature = "lookup-fold")]
        comm_inv_t: None,
        #[cfg(feature = "lookup-fold")]
        T_lookup: None,
      };

      let (comm_E1, comm_E2, r_E1, r_E2) = split_E_commitments(&ck, &W, &U, &structure);

      assert_eq!(comm_E1 + comm_E2, U.comm_E);
      assert_eq!(r_E1 + r_E2, W.r_E);
    }
  }

  /// **Unit test**: blinding-split sum identity is preserved exactly,
  /// independent of the random draw of `r_E1`. Asserts the algebraic
  /// invariant `r_E1 + r_E2 == W.r_E` across distinct invocations (the
  /// random `r_E1` differs each call; the sum must still equal `W.r_E`).
  #[test]
  #[allow(non_snake_case)]
  fn blinding_split_sum_identity_independent_of_random_draw() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

    let (ck, structure, U, W) = build_satisfying_triple::<E, S>();

    // Multiple draws — each call uses a fresh `r_E1` from OsRng. The sum
    // identity must hold for every draw, and the Pedersen-additive
    // identity must also hold (proving the random `r_E1` does not break
    // the binding).
    let mut prior_r_E1 = None;
    for _ in 0..8 {
      let (comm_E1, comm_E2, r_E1, r_E2) = split_E_commitments(&ck, &W, &U, &structure);

      assert_eq!(r_E1 + r_E2, W.r_E);
      assert_eq!(comm_E1 + comm_E2, U.comm_E);

      // Sanity: across distinct draws, `r_E1` should not be the same
      // value (probability of collision is negligible). This pins the
      // "fresh random" property of the blinding split.
      if let Some(prior) = prior_r_E1 {
        assert_ne!(
          r_E1, prior,
          "r_E1 should be fresh-random per call (negligible collision probability)",
        );
      }
      prior_r_E1 = Some(r_E1);
    }
  }

}
