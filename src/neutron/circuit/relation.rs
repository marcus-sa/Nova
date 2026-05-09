//! This module implements various gadgets necessary for folding R1CS types with NeutronNova folding scheme.
use crate::{
  frontend::{num::AllocatedNum, Boolean, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedNonnativePoint,
    utils::{alloc_zero, conditionally_select},
  },
  neutron::{circuit::r1cs::AllocatedNonnativeR1CSInstance, relation::FoldedInstance},
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};
use ff::Field;

/// An in-circuit representation of NeutronNova's FoldedInstance
/// In our context, public IO of circuits folded will have only one entry, so we have X
#[derive(Clone, Debug)]
pub struct AllocatedFoldedInstance<E: Engine> {
  pub(crate) comm_W: AllocatedNonnativePoint<E>,
  pub(crate) comm_E: AllocatedNonnativePoint<E>,
  pub(crate) T: AllocatedNum<E::Scalar>,

  /// Per-table running lookup scalar VECTOR. `None` only at outer base
  /// when no prior fold step has carried lookup data; for fold-depth >= 1
  /// always `Some` (length pinned by the shape registry's
  /// `multi_column_tables.len()`). At outer base each entry that does
  /// exist is `alloc_zero()`.
  ///
  /// GH-#5 design pin §1.4 / §3.2 (W1) binding-via-hash: this field is
  /// absorbed in `absorb_in_ro` between `T` and `u` in `table_id`-canonical
  /// order. The absorption binds the per-table running scalar VECTOR
  /// through the FS hash that becomes `u.X[0]`, preserving the single-IO
  /// `AllocatedNonnativeR1CSInstance` absorption pattern as sound.
  ///
  /// Mirrors the native-side `FoldedInstance::T_lookup: Option<Vec<E::Scalar>>`
  /// at `vendor/nova/src/neutron/relation.rs:265`.
  #[cfg(feature = "lookup-fold")]
  pub(crate) T_lookup_per_table: Option<Vec<AllocatedNum<E::Scalar>>>,

  pub(crate) u: AllocatedNum<E::Scalar>,
  pub(crate) X: AllocatedNum<E::Scalar>,
}

impl<E: Engine> AllocatedFoldedInstance<E> {
  /// Allocates the given `FoldedInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    inst: Option<&FoldedInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    // We do not need to check that W or E are well-formed (e.g., on the curve) as we do a hash check
    // in the Nova augmented circuit, which ensures that the relaxed instance
    // came from a prior iteration of Nova.
    let comm_W = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate W"),
      inst.map(|inst| inst.comm_W.to_coordinates()),
    )?;

    let comm_E = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate E"),
      inst.map(|inst| inst.comm_E.to_coordinates()),
    )?;

    let T = AllocatedNum::alloc(cs.namespace(|| "allocate T"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |inst| inst.T))
    })?;

    let u = AllocatedNum::alloc(cs.namespace(|| "allocate u"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |inst| inst.u))
    })?;

    // GH-#5 design pin §1.4 corrigendum (Halpert 2026-05-09): X-arity
    // invariant — the (W1) binding-via-hash extension is sound only at
    // `num_io == 1`. Native `FoldedInstance::X: Vec<E::Scalar>` (length
    // `S.S.num_io`) absorbs N scalars; in-circuit `AllocatedFoldedInstance::X`
    // is a single AllocatedNum and absorbs exactly one scalar. Silently
    // taking `inst.X[0]` for `X.len() != 1` would be a soundness-relevant
    // truncation. Fail closed at synthesis time. Mirrors the fail-closed
    // posture of `conditionally_select`'s T_lookup_per_table shape check
    // (lines ~379-384).
    if let Some(inst) = inst {
      if inst.X.len() != 1 {
        return Err(SynthesisError::Unsatisfiable(
          "AllocatedFoldedInstance::alloc: native FoldedInstance::X has len != 1; \
           the (W1) binding-via-hash extension requires num_io == 1 per pin §1.4 corrigendum"
            .to_string(),
        ));
      }
    }

    // Allocate X. If the input instance is None, then allocate default values 0.
    let X = AllocatedNum::alloc(cs.namespace(|| "allocate X"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |inst| inst.X[0]))
    })?;

    // GH-#5 design pin §3.2: allocate `T_lookup_per_table` from `inst.t_lookup()`.
    // Length is implicit (whatever the prior running U1 carried). At outer
    // base (`inst == None` OR `inst.t_lookup() == None`), this is `None`.
    #[cfg(feature = "lookup-fold")]
    let T_lookup_per_table = {
      let t_lookup_slice: Option<Vec<E::Scalar>> = inst
        .and_then(|inst| inst.t_lookup())
        .map(|s| s.to_vec());
      match t_lookup_slice {
        None => None,
        Some(slice) => {
          let allocated = slice
            .into_iter()
            .enumerate()
            .map(|(j, t_j)| {
              AllocatedNum::alloc(
                cs.namespace(|| format!("allocate T_lookup_per_table[{j}]")),
                || Ok(t_j),
              )
            })
            .collect::<Result<Vec<_>, _>>()?;
          Some(allocated)
        }
      }
    };

    Ok(Self {
      comm_W,
      comm_E,
      T,
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table,
      u,
      X,
    })
  }

  /// Allocates a default `RelaxedR1CSInstance` with a `lookup-fold`-active
  /// per-table running scalar vector of length `k`.
  ///
  /// W = E = 0, T = 0, T_lookup_per_table = [0; k], u = 0, X = 0.
  ///
  /// GH-#5 design pin §2.3 / §3.2: pinned by the shape registry's
  /// `multi_column_tables.len()`. Used at the augmented-circuit's
  /// `synthesize_base_case` site (M.GH5.3 wire-up); also exercised by the
  /// M.GH5.0 STAGE 0 spike harness in
  /// `crates/inumbra-spend-harness/tests/gh5_m0_stage0_*` to construct
  /// fold-depth-≥1 fixtures with non-empty `T_lookup_per_table` in a way
  /// that survives the `cfg`-gate without depending on the native
  /// multi-table prover's running U. `dead_code` allowed until M.GH5.3
  /// lands; all consumers are external (inumbra-harness integration
  /// tests) at this milestone.
  #[allow(dead_code)] // until M.GH5.3 wires `synthesize_base_case` through this constructor
  #[cfg(feature = "lookup-fold")]
  pub fn default_with_lookup_k<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    k: usize,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::default(cs.namespace(|| "allocate W"))?;
    let comm_E = comm_W.clone();
    let T = alloc_zero(cs.namespace(|| "allocate T"));
    let u = T.clone();
    let X = T.clone();

    let T_lookup_per_table = Some(
      (0..k)
        .map(|j| {
          AllocatedNum::alloc(
            cs.namespace(|| format!("allocate T_lookup_per_table[{j}] = 0")),
            || Ok(E::Scalar::ZERO),
          )
        })
        .collect::<Result<Vec<_>, _>>()?,
    );

    Ok(Self {
      comm_W,
      comm_E,
      T,
      T_lookup_per_table,
      u,
      X,
    })
  }

  /// Allocates the hardcoded default `RelaxedR1CSInstance` in the circuit.
  /// W = E = 0, T = 0, u = 0, X = 0
  ///
  /// For `lookup-fold` builds, `T_lookup_per_table = None`. This represents
  /// the structurally-empty outer base where no prior fold step has carried
  /// lookup data. The augmented-circuit's `synthesize_base_case` (under
  /// M.GH5.3) calls [`Self::default_with_lookup_k`] instead to allocate a
  /// k-typed running instance bound by the shape registry; this method is
  /// retained for non-`lookup-fold` builds and for fixtures that
  /// intentionally model the structurally-empty outer base.
  pub fn default<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::default(cs.namespace(|| "allocate W"))?;
    let comm_E = comm_W.clone();

    // Allocate T = 0. Similar to X0 and X1, we do not need to check that T is well-formed
    let T = alloc_zero(cs.namespace(|| "allocate T"));

    let u = T.clone();

    // X is allocated and set to zero
    let X = T.clone();

    Ok(Self {
      comm_W,
      comm_E,
      T,
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table: None,
      u,
      X,
    })
  }

  /// Absorb the provided instance in the RO
  ///
  /// GH-#5 design pin §1.4 (W1) binding-via-hash: under `lookup-fold`, each
  /// `T_lookup_per_table[j]` is absorbed in `table_id`-canonical order
  /// BETWEEN the existing `T` absorption and the `u`/`X` absorptions. This
  /// pins the per-table running scalar VECTOR through the FS hash that
  /// becomes `u.X[0]`, closing the (W1) cross-step substitution attack at
  /// fold-depth ≥ 1. At outer base where `T_lookup_per_table == None`,
  /// no scalars are absorbed (the sequence is empty, not skipped).
  ///
  /// Native-side mirror: [`crate::neutron::relation::FoldedInstance`]
  /// `absorb_in_ro2` (impl of [`crate::traits::AbsorbInRO2Trait`]) absorbs
  /// `T_lookup` between `T` and `u` in the same `table_id`-canonical order.
  /// Byte-equivalence is the M.GH5.0 STAGE 0 acceptance criterion.
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    ro: &mut E::RO2Circuit,
  ) -> Result<(), SynthesisError> {
    self
      .comm_W
      .absorb_in_ro(cs.namespace(|| "absorb W in RO"), ro)?;
    self
      .comm_E
      .absorb_in_ro(cs.namespace(|| "absorb E in RO"), ro)?;
    ro.absorb(&self.T);

    // GH-#5 (W1) BINDING: per-table running lookup scalar VECTOR is
    // absorbed in `table_id`-canonical order, BEFORE u/X. Pinned by the
    // shape registry's `multi_column_tables.len()`. At outer base
    // (`T_lookup_per_table == None`) no scalars are absorbed (the
    // sequence is empty, not skipped).
    #[cfg(feature = "lookup-fold")]
    if let Some(t_lookup) = &self.T_lookup_per_table {
      for t_j in t_lookup {
        ro.absorb(t_j);
      }
    }

    ro.absorb(&self.u);
    ro.absorb(&self.X);
    Ok(())
  }

  /// Folds self with an r1cs instance and returns the result
  pub fn fold<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    U2: &AllocatedNonnativeR1CSInstance<E>,
    r_b: &AllocatedNum<E::Scalar>,
    T_out: &AllocatedNum<E::Scalar>,
    comm_W_fold: &AllocatedNonnativePoint<E>,
    comm_E_fold: &AllocatedNonnativePoint<E>,
  ) -> Result<Self, SynthesisError> {
    // u_fold = (1-r_b) * self.u + r_b * U2.u
    // u_fold = self.u - r_b * self.u + r_b * U2.u
    // u_fold = self.u + r_b (U2.u - self.u)
    // In our context U2.u = 1, so u_fold = self.u + r_b (1 - self.u)
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "allocate u_fold"), || {
      let u = self
        .u
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U2_u = E::Scalar::ONE;
      Ok(u + r_b * (U2_u - u))
    })?;

    cs.enforce(
      || "enforce u_fold -self.u  = r_b (U2.u - self.u)",
      |lc| lc + r_b.get_variable(),
      |lc| lc + CS::one() - self.u.get_variable(),
      |lc| lc + u_fold.get_variable() - self.u.get_variable(),
    );

    // Fold the IO:
    // X_fold = self.X + r_b (U2.X - self.X)
    let X_fold = AllocatedNum::alloc(cs.namespace(|| "allocate X_fold"), || {
      let X = self
        .X
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U2_X = U2.X.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(X + r_b * (U2_X - X))
    })?;
    cs.enforce(
      || "enforce X_fold - self.X = r_b (U2.X - self.X)",
      |lc| lc + r_b.get_variable(),
      |lc| lc + U2.X.get_variable() - self.X.get_variable(),
      |lc| lc + X_fold.get_variable() - self.X.get_variable(),
    );

    // GH-#5 design pin §3.2: `fold` passes `T_lookup_per_table` through
    // unchanged from `self`. Post-fold update of the per-table running
    // scalar vector is the responsibility of the lookup verifier path
    // (`verify_with_multi_table_lookup`); the augmented-circuit caller at
    // M.GH5.3 wires the post-fold T_lookup values into the post-fold
    // `AllocatedFoldedInstance` via a sibling helper. Mirrors the native
    // `FoldedInstance::fold` passthrough at
    // `vendor/nova/src/neutron/relation.rs:695`.
    Ok(Self {
      comm_W: comm_W_fold.clone(),
      comm_E: comm_E_fold.clone(),
      T: T_out.clone(),
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table: self.T_lookup_per_table.clone(),
      u: u_fold,
      X: X_fold,
    })
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::conditionally_select(
      cs.namespace(|| "W = cond ? self.W : other.W"),
      &self.comm_W,
      &other.comm_W,
      condition,
    )?;

    let comm_E = AllocatedNonnativePoint::conditionally_select(
      cs.namespace(|| "E = cond ? self.E : other.E"),
      &self.comm_E,
      &other.comm_E,
      condition,
    )?;

    let T = conditionally_select(
      cs.namespace(|| "T = cond ? self.T : other.T"),
      &self.T,
      &other.T,
      condition,
    )?;

    let u = conditionally_select(
      cs.namespace(|| "u = cond ? self.u : other.u"),
      &self.u,
      &other.u,
      condition,
    )?;

    let X = conditionally_select(
      cs.namespace(|| "X[0] = cond ? self.X[0] : other.X[0]"),
      &self.X,
      &other.X,
      condition,
    )?;

    // GH-#5 design pin §2.3 / §3.2: under `lookup-fold`, both branches of
    // `conditionally_select` must carry `T_lookup_per_table` of the same
    // shape (`None`/`None`, or `Some(len=k)` / `Some(len=k)`). The
    // augmented-circuit's `synthesize_base_case` allocates via
    // `default_with_lookup_k(cs, k)` and `synthesize_non_base_case`
    // produces the post-fold instance from `verify_with_multi_table_lookup`'s
    // length-k output, so honest synthesis preserves the invariant. A
    // shape mismatch indicates a wire-up bug at the caller; this method
    // returns `SynthesisError::Unsatisfiable` rather than panicking so
    // the constraint system fails-closed at synthesis time.
    #[cfg(feature = "lookup-fold")]
    let T_lookup_per_table = match (&self.T_lookup_per_table, &other.T_lookup_per_table) {
      (None, None) => None,
      (Some(self_v), Some(other_v)) if self_v.len() == other_v.len() => {
        let selected = self_v
          .iter()
          .zip(other_v.iter())
          .enumerate()
          .map(|(j, (s, o))| {
            conditionally_select(
              cs.namespace(|| {
                format!("T_lookup_per_table[{j}] = cond ? self.T_lookup_per_table[{j}] : other.T_lookup_per_table[{j}]")
              }),
              s,
              o,
              condition,
            )
          })
          .collect::<Result<Vec<_>, _>>()?;
        Some(selected)
      }
      _ => {
        return Err(SynthesisError::Unsatisfiable(
          "AllocatedFoldedInstance::conditionally_select: T_lookup_per_table shape mismatch (None/Some or unequal lengths) — invariant violated by caller".to_string(),
        ));
      }
    };

    Ok(Self {
      comm_W,
      comm_E,
      T,
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table,
      u,
      X,
    })
  }
}

// =========================================================================
// M.GH5.0 STAGE 0: byte-equivalence between native `FoldedInstance::absorb_in_ro2`
// and in-circuit `AllocatedFoldedInstance::absorb_in_ro` at fold-depth ≥ 1
// with non-trivial `T_lookup_per_table`.
//
// Source-of-truth: `docs/research/cryptography/gh-5-augmented-circuit-wireup-design-pin-2026-05-08.md`
//   §1.4 — (W1) binding-via-hash + absorption-order pin.
//   §6.2 — STAGE 0 spike contract.
//   §5.2 corrigendum (Halpert 2026-05-09) — STAGE 0 lands in-vendor here,
//     gated `#[cfg(feature = "lookup-fold")]`, following M.6/M.6.5 precedent.
//
// Pass criterion (pin §6.2): byte-equality of derived squeeze output between
// the native and the in-circuit absorption sequences AT EVERY STEP, plus
// `cs.is_satisfied() == true` AT EVERY STEP.
//
// Failure criterion (STOP-AND-ASK): any squeeze-output disagreement OR
// `is_satisfied() == false`. Per pin §6.2: failure halts ALL further GH-#5
// milestones; the absorption-order pin in §1.4 is wrong.
//
// Q3 (Halpert ratification): squeeze-output framing — assert
// `h_native == h_circuit_witness_value` plus `cs.is_satisfied()`. Matches
// M.6's binding-witness pattern at `vendor/nova/src/neutron/circuit/lookup.rs`
// (~lines 1156-1369). Does NOT capture scalar-by-scalar (which would require
// a `CapturingRO2` wrapper triggering visibility cascades — Q2 ruling A
// forbids vendor public-API expansion for this spike).
//
// Pre-check: assert both `running_u_after_step_1.t_lookup()` and
// `running_u_after_step_2.t_lookup()` are `Some(&[non-zero, non-zero])`.
// If either is `None` or all-zero, the fixture isn't exercising the (W1)
// extension at fold-depth ≥ 1 with non-trivial T_lookup.
// =========================================================================
#[cfg(all(test, feature = "lookup-fold"))]
mod stage0_byte_equivalence_tests {
  use super::*;
  use crate::{
    constants::NUM_HASH_BITS,
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      util_cs::test_cs::TestConstraintSystem,
    },
    gadgets::utils::le_bits_to_num,
    neutron::{
      circuit::{NeutronAugmentedCircuit, NeutronAugmentedCircuitInputs},
      nifs::NIFS,
      relation::{
        FoldedInstance, FoldedWitness, LookupPayload, LookupPayloadPublicMultiTable,
        LookupRunningWitness, LookupShape, LookupTableHandle, MultiColumnLookupTable, Structure,
      },
    },
    provider::{hyperkzg::EvaluationEngine as HyperKZGEE, Bn256EngineKZG},
    r1cs::R1CSShape,
    spartan::{polys::power::PowPolynomial, snark::RelaxedR1CSSNARK},
    traits::{
      circuit::NonTrivialCircuit, commitment::CommitmentEngineTrait,
      snark::RelaxedR1CSSNARKTrait, AbsorbInRO2Trait, RO2Constants, RO2ConstantsCircuit, ROTrait,
    },
    Commitment,
  };
  use ff::Field;
  use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

  type E = Bn256EngineKZG;
  type Scalar = <E as Engine>::Scalar;
  type S = RelaxedR1CSSNARK<E, HyperKZGEE<E>>;

  /// Compute the native squeeze output for a `FoldedInstance`'s absorption
  /// sequence. Mirrors the prover-side hash-input computation pattern (cf.
  /// `vendor/nova/src/neutron/mod.rs:406-418`).
  fn native_absorb_squeeze(
    ro_consts: &RO2Constants<E>,
    folded_U: &FoldedInstance<E>,
  ) -> Scalar {
    let mut ro = <E as Engine>::RO2::new(ro_consts.clone());
    folded_U.absorb_in_ro2(&mut ro);
    ro.squeeze(NUM_HASH_BITS, false)
  }

  /// Compute the in-circuit derived squeeze output for the same
  /// `FoldedInstance`'s allocated mirror. Mirrors the augmented-circuit
  /// hash-output extraction pattern at
  /// `vendor/nova/src/neutron/circuit/mod.rs:413-425`.
  ///
  /// Returns `(h_circuit_witness_value, cs_is_satisfied)`.
  fn circuit_absorb_squeeze(
    ro_consts_circuit: &RO2ConstantsCircuit<E>,
    folded_U: &FoldedInstance<E>,
  ) -> (Scalar, bool) {
    let mut cs = TestConstraintSystem::<Scalar>::new();
    let allocated_u =
      AllocatedFoldedInstance::<E>::alloc(cs.namespace(|| "U"), Some(folded_U)).unwrap();
    let mut ro_circuit = <E as Engine>::RO2Circuit::new(ro_consts_circuit.clone());
    allocated_u
      .absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro_circuit)
      .expect("in-circuit absorb_in_ro must synthesise cleanly");
    let hash_bits = ro_circuit
      .squeeze(cs.namespace(|| "squeeze"), NUM_HASH_BITS, false)
      .expect("squeeze must synthesise");
    let hash = le_bits_to_num(cs.namespace(|| "bits to num"), &hash_bits)
      .expect("le_bits_to_num must synthesise");
    let satisfied = cs.is_satisfied();
    let h_witness = hash
      .get_value()
      .expect("hash witness must be assigned in honest synthesis");
    (h_witness, satisfied)
  }

  /// M.GH5.0 STAGE 0 — byte-equivalence at fold-depth 1 (after step 1)
  /// AND fold-depth 2 (after step 2) with non-trivial T_lookup_per_table.
  ///
  /// Fixture pattern adapted from
  /// `vendor/nova/src/neutron/nifs.rs::tests::m4_per_table_c_binding_hard_rejects_corrupted_t_lookup`
  /// (the closest 2-step k=2 honest-IVC precedent). Each step:
  /// 1. Native prove via `prove_with_multi_table_lookup`.
  /// 2. Native verify via `verify_with_multi_table_lookup` (asserts
  ///    `verified_U == folded_U` per M.4 assertion at nifs.rs:4620).
  /// 3. Native squeeze + in-circuit derived squeeze, equality + is_satisfied
  ///    asserted.
  ///
  /// Per pin §6.2: STAGE 0 fail = halt all GH-#5 milestones. The §1.4
  /// absorption-order pin is paper-grounded (eprint 2021/370 v3 §4) +
  /// structurally elaborated; first-run pass is the expected outcome
  /// (Halpert Q1 ratification 2026-05-09).
  #[test]
  fn m_gh5_0_stage0_absorb_in_ro_byte_equivalence_at_fold_depth_1_and_2() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_5BAD_C0DE_0050);
    let ro_consts = RO2Constants::<E>::default();
    let ro_consts_circuit = RO2ConstantsCircuit::<E>::default();
    let pp_digest = Scalar::ZERO;

    // R1CS shape comes from the production augmented circuit
    // `NeutronAugmentedCircuit<E, NonTrivialCircuit<Scalar>>` per pin §1.4
    // corrigendum (Halpert 2026-05-09): the (W1) binding-via-hash extension
    // is sound only at `num_io == 1`, which is the shape the augmented
    // circuit produces (single `hash.inputize` site at `circuit/mod.rs:428`).
    // The earlier `DirectCircuit` shape was `num_io == 2` (z_i + z_i_plus_one
    // both inputized) and falsified STAGE 0 step-1 byte-equivalence; the
    // root cause was pin-incompleteness on the X-arity invariant, not a
    // soundness-class halt. Path A per corrigendum: replace `DirectCircuit`
    // with `NeutronAugmentedCircuit` so STAGE 0 exercises the actual
    // production-shape invariant.
    let num_cons = 32usize;
    let step_circuit = NonTrivialCircuit::<Scalar>::new(num_cons);
    let augmented_shape_builder: NeutronAugmentedCircuit<'_, E, NonTrivialCircuit<Scalar>> =
      NeutronAugmentedCircuit::new(None, &step_circuit, ro_consts_circuit.clone());
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = augmented_shape_builder.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    assert_eq!(
      shape.num_io(),
      1,
      "STAGE 0 fixture: production-shape invariant — `NeutronAugmentedCircuit` \
       must produce R1CS shape with `num_io == 1` per pin §1.4 corrigendum."
    );
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // k=2 multi-table fixture — production chunked-Strauss-Shamir shape per
    // ADR-0021. Random table contents pinned via ChaCha20Rng.
    let table_size = 64usize;
    let table_log2 = 6usize;
    let t1_col0: Vec<Scalar> = (0..table_size).map(|_| Scalar::random(&mut rng)).collect();
    let t2_col0: Vec<Scalar> = (0..table_size).map(|_| Scalar::random(&mut rng)).collect();
    let identity: Vec<Scalar> = (0..table_size).map(|i| Scalar::from(i as u64)).collect();

    let lookup_shape = LookupShape::<E> {
      tables: vec![
        LookupTableHandle {
          table_id: 0,
          size: table_size,
          commitment: <E as Engine>::CE::commit(&ck, &identity, &Scalar::ZERO),
        },
        LookupTableHandle {
          table_id: 1,
          size: table_size,
          commitment: <E as Engine>::CE::commit(&ck, &identity, &Scalar::ZERO),
        },
      ],
      multi_column_tables: vec![
        MultiColumnLookupTable {
          table_id: 0,
          size: table_size,
          columns: vec![t1_col0.clone()],
          value_commitments: vec![<E as Engine>::CE::commit(&ck, &t1_col0, &Scalar::ZERO)],
        },
        MultiColumnLookupTable {
          table_id: 1,
          size: table_size,
          columns: vec![t2_col0.clone()],
          value_commitments: vec![<E as Engine>::CE::commit(&ck, &t2_col0, &Scalar::ZERO)],
        },
      ],
      num_addr_columns: 1,
      num_witness_columns: 2,
      witness_ell_cached: table_log2,
    };
    let str_local = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str_local.S.clone();

    // Two distinct R1CS pairs for the two fold steps.
    //
    // Per pin §1.4 corrigendum (path A): the R1CS instances satisfy the
    // augmented-circuit shape (`num_io == 1`). We synthesize the augmented
    // circuit at base case (`U = None`, `i = 0`, `zi = None`, `nifs = None`)
    // — `test_recursive_circuit_with` (`circuit/mod.rs:467-489`) confirms
    // this is satisfying. Two distinct fold steps are obtained by varying
    // the `r_next` field (a free public input that does not constrain the
    // base-case witness), giving two distinct R1CS instances of identical
    // `num_io == 1` shape — exactly what STAGE 0 needs to exercise the
    // (W1) extension's byte-equivalence at fold depths 1 and 2.
    let make_r1cs = |seed: u64| {
      let inputs: NeutronAugmentedCircuitInputs<E> = NeutronAugmentedCircuitInputs::new(
        pp_digest,        // pp_digest
        Scalar::ZERO,     // i = 0 (base case)
        vec![Scalar::ZERO], // z0 (arity == 1)
        None,             // zi (base case)
        None,             // U (base case)
        None,             // ri
        Scalar::from(seed), // r_next — distinct per step
        None,             // u (base case)
        None,             // nifs (base case)
        None,             // comm_W_fold
        None,             // comm_E_fold
      );
      let circuit: NeutronAugmentedCircuit<'_, E, NonTrivialCircuit<Scalar>> =
        NeutronAugmentedCircuit::new(
          Some(inputs),
          &step_circuit,
          ro_consts_circuit.clone(),
        );
      let mut cs = SatisfyingAssignment::<E>::new();
      let _ = circuit.synthesize(&mut cs);
      let (u, w) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
      (u, w.pad(&shape))
    };
    let (u_step1, w_step1) = make_r1cs(2);
    let (u_step2, w_step2) = make_r1cs(3);

    // Per-table satisfying-witness builder (mirrors M.4 fixture shape).
    let build_satisfying_witness =
      |table: &[Scalar], query_indices: &[usize]| -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
        let mut witness_addr = vec![Scalar::ZERO; table_size];
        let mut witness_v0 = vec![Scalar::ZERO; table_size];
        let mut multiplicities = vec![Scalar::ZERO; table_size];
        for (i, &idx) in query_indices.iter().enumerate() {
          witness_addr[i] = Scalar::from(idx as u64);
          witness_v0[i] = table[idx];
          multiplicities[idx] += Scalar::ONE;
        }
        for i in query_indices.len()..table_size {
          witness_addr[i] = Scalar::from(0u64);
          witness_v0[i] = table[0];
          multiplicities[0] += Scalar::ONE;
        }
        (witness_addr, witness_v0, multiplicities)
      };

    // Per-table eq dimensions (mirrors M.4 commentary at nifs.rs:4498-4504).
    let per_table_log2 = table_log2;
    let ell1 = per_table_log2.div_ceil(2);
    let ell2 = per_table_log2 / 2;
    let per_table_w_left = 1usize << ell1;
    let per_table_w_right = 1usize << ell2;
    let per_table_t_left = per_table_w_left;
    let per_table_t_right = per_table_w_right;

    let mk_eqs =
      |rng: &mut ChaCha20Rng| -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
        let tau_w = Scalar::random(&mut *rng);
        let pow_w = PowPolynomial::new(&tau_w, per_table_log2);
        let combined_w = pow_w.split_evals(per_table_w_left, per_table_w_right);
        let (eq_w_left, eq_w_right) = combined_w.split_at(per_table_w_left);
        let tau_t = Scalar::random(&mut *rng);
        let pow_t = PowPolynomial::new(&tau_t, per_table_log2);
        let combined_t_eq = pow_t.split_evals(per_table_t_left, per_table_t_right);
        let (eq_t_left, eq_t_right) = combined_t_eq.split_at(per_table_t_left);
        (
          eq_w_left.to_vec(),
          eq_w_right.to_vec(),
          eq_t_left.to_vec(),
          eq_t_right.to_vec(),
        )
      };

    // Per-table running witness sized to n_j = table_size (outer base zeros).
    let mk_running_lw = || LookupRunningWitness::<E> {
      witness: vec![Scalar::ZERO; table_size],
      inv_w: vec![Scalar::ZERO; table_size],
      table: vec![Scalar::ZERO; table_size],
      multiplicities: vec![Scalar::ZERO; table_size],
      inv_t: vec![Scalar::ZERO; table_size],
      eq_w_left: vec![Scalar::ZERO; per_table_w_left],
      eq_w_right: vec![Scalar::ZERO; per_table_w_right],
      eq_t_left: vec![Scalar::ZERO; per_table_t_left],
      eq_t_right: vec![Scalar::ZERO; per_table_t_right],
    };

    let mk_bundle =
      |table_id: u64,
       payload: LookupPayload<E>,
       wa: Vec<Scalar>,
       wv: Vec<Scalar>,
       m: Vec<Scalar>,
       eq_w_left: Vec<Scalar>,
       eq_w_right: Vec<Scalar>,
       eq_t_left: Vec<Scalar>,
       eq_t_right: Vec<Scalar>,
       running_lw: LookupRunningWitness<E>|
       -> crate::neutron::nifs::PerTableBundle<E> {
        crate::neutron::nifs::PerTableBundle::<E> {
          table_id,
          payload,
          fresh_witness_address: wa,
          fresh_witness_value_columns: vec![wv],
          fresh_multiplicities: m,
          fresh_eq_w_left: eq_w_left,
          fresh_eq_w_right: eq_w_right,
          fresh_eq_t_left: eq_t_left,
          fresh_eq_t_right: eq_t_right,
          running_lw,
        }
      };

    // ===== Step 1: honest prove + verify =====
    let queries_t1_step1 = [1usize, 4, 9, 16];
    let queries_t2_step1 = [2usize, 5, 11, 22];
    let (wa_1_s1, wv_1_s1, m_1_s1) = build_satisfying_witness(&t1_col0, &queries_t1_step1);
    let (wa_2_s1, wv_2_s1, m_2_s1) = build_satisfying_witness(&t2_col0, &queries_t2_step1);

    let payload_1_s1 = LookupPayload::<E> {
      comm_L: <E as Engine>::CE::commit(&ck, &wa_1_s1, &Scalar::ZERO),
      comm_ts: <E as Engine>::CE::commit(&ck, &m_1_s1, &Scalar::ZERO),
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![<E as Engine>::CE::commit(&ck, &wv_1_s1, &Scalar::ZERO)],
    };
    let payload_2_s1 = LookupPayload::<E> {
      comm_L: <E as Engine>::CE::commit(&ck, &wa_2_s1, &Scalar::ZERO),
      comm_ts: <E as Engine>::CE::commit(&ck, &m_2_s1, &Scalar::ZERO),
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![<E as Engine>::CE::commit(&ck, &wv_2_s1, &Scalar::ZERO)],
    };

    let (eq_w1l_s1, eq_w1r_s1, eq_t1l_s1, eq_t1r_s1) = mk_eqs(&mut rng);
    let (eq_w2l_s1, eq_w2r_s1, eq_t2l_s1, eq_t2r_s1) = mk_eqs(&mut rng);

    let running_W_outer = FoldedWitness::default(&str_local);
    let running_U_outer = FoldedInstance::default(&str_local);
    let bundle_1_s1 = mk_bundle(
      0,
      payload_1_s1.clone(),
      wa_1_s1,
      wv_1_s1,
      m_1_s1,
      eq_w1l_s1,
      eq_w1r_s1,
      eq_t1l_s1,
      eq_t1r_s1,
      mk_running_lw(),
    );
    let bundle_2_s1 = mk_bundle(
      1,
      payload_2_s1.clone(),
      wa_2_s1,
      wv_2_s1,
      m_2_s1,
      eq_w2l_s1,
      eq_w2r_s1,
      eq_t2l_s1,
      eq_t2r_s1,
      mk_running_lw(),
    );

    let (nifs1, (folded_U_s1, folded_W_s1), folded_lw_s1) =
      NIFS::<E>::prove_with_multi_table_lookup(
        &ck,
        &ro_consts,
        &pp_digest,
        &str_local,
        &running_U_outer,
        &running_W_outer,
        &u_step1,
        &w_step1,
        &[bundle_1_s1, bundle_2_s1],
      )
      .expect("STAGE 0 step-1 prove must succeed");

    let public_bundles_s1 = vec![
      LookupPayloadPublicMultiTable::<E> {
        table_id: 0,
        comm_L: payload_1_s1.comm_L,
        comm_values: payload_1_s1.comm_values.clone(),
        comm_ts: payload_1_s1.comm_ts,
      },
      LookupPayloadPublicMultiTable::<E> {
        table_id: 1,
        comm_L: payload_2_s1.comm_L,
        comm_values: payload_2_s1.comm_values.clone(),
        comm_ts: payload_2_s1.comm_ts,
      },
    ];

    let verified_U_s1 = nifs1
      .verify_with_multi_table_lookup(
        &ro_consts,
        &pp_digest,
        &str_local,
        &running_U_outer,
        &u_step1,
        &public_bundles_s1,
      )
      .expect("STAGE 0 step-1 verify must succeed");
    assert_eq!(
      folded_U_s1, verified_U_s1,
      "STAGE 0: step-1 prover/verifier must agree on folded U"
    );
    let running_u_after_step_1: FoldedInstance<E> = verified_U_s1;

    // STAGE 0 pre-check: T_lookup at fold-depth 1 must be Some(&[non-zero, non-zero]).
    {
      let t = running_u_after_step_1
        .t_lookup()
        .expect("STAGE 0 fixture: step-1 running U must carry T_lookup (Some)");
      assert_eq!(
        t.len(),
        2,
        "STAGE 0 fixture: k=2 — running U after step 1 must carry 2-entry T_lookup"
      );
      assert_ne!(
        t[0],
        Scalar::ZERO,
        "STAGE 0 fixture: T_lookup[0] after step 1 must be non-zero (else the (W1) extension isn't being exercised)"
      );
      assert_ne!(
        t[1],
        Scalar::ZERO,
        "STAGE 0 fixture: T_lookup[1] after step 1 must be non-zero (else the (W1) extension isn't being exercised)"
      );
    }

    // STAGE 0 step-1 byte-equivalence: native squeeze == in-circuit
    // derived squeeze witness value, with cs.is_satisfied().
    let h_native_s1 = native_absorb_squeeze(&ro_consts, &running_u_after_step_1);
    let (h_circuit_s1, satisfied_s1) =
      circuit_absorb_squeeze(&ro_consts_circuit, &running_u_after_step_1);
    assert!(
      satisfied_s1,
      "STAGE 0 step-1: in-circuit absorb_in_ro CS must be satisfied. \
       Per pin §6.2 STOP-AND-ASK: halt all further GH-#5 milestones."
    );
    assert_eq!(
      h_native_s1, h_circuit_s1,
      "STAGE 0 step-1: native FoldedInstance::absorb_in_ro2 squeeze MUST equal \
       in-circuit AllocatedFoldedInstance::absorb_in_ro derived squeeze. \
       (h_native, h_circuit) = ({:?}, {:?}). \
       Per pin §6.2 STOP-AND-ASK: the §1.4 absorption-order pin is wrong; \
       halt all further GH-#5 milestones and re-derive.",
      h_native_s1, h_circuit_s1
    );

    // ===== Step 2: honest prove + verify against running_u_after_step_1 =====
    let queries_t1_step2 = [3usize, 7, 13, 25];
    let queries_t2_step2 = [4usize, 8, 17, 33];
    let (wa_1_s2, wv_1_s2, m_1_s2) = build_satisfying_witness(&t1_col0, &queries_t1_step2);
    let (wa_2_s2, wv_2_s2, m_2_s2) = build_satisfying_witness(&t2_col0, &queries_t2_step2);

    let payload_1_s2 = LookupPayload::<E> {
      comm_L: <E as Engine>::CE::commit(&ck, &wa_1_s2, &Scalar::ZERO),
      comm_ts: <E as Engine>::CE::commit(&ck, &m_1_s2, &Scalar::ZERO),
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![<E as Engine>::CE::commit(&ck, &wv_1_s2, &Scalar::ZERO)],
    };
    let payload_2_s2 = LookupPayload::<E> {
      comm_L: <E as Engine>::CE::commit(&ck, &wa_2_s2, &Scalar::ZERO),
      comm_ts: <E as Engine>::CE::commit(&ck, &m_2_s2, &Scalar::ZERO),
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![<E as Engine>::CE::commit(&ck, &wv_2_s2, &Scalar::ZERO)],
    };

    let (eq_w1l_s2, eq_w1r_s2, eq_t1l_s2, eq_t1r_s2) = mk_eqs(&mut rng);
    let (eq_w2l_s2, eq_w2r_s2, eq_t2l_s2, eq_t2r_s2) = mk_eqs(&mut rng);

    let bundle_1_s2 = mk_bundle(
      0,
      payload_1_s2.clone(),
      wa_1_s2,
      wv_1_s2,
      m_1_s2,
      eq_w1l_s2,
      eq_w1r_s2,
      eq_t1l_s2,
      eq_t1r_s2,
      folded_lw_s1[0].clone(),
    );
    let bundle_2_s2 = mk_bundle(
      1,
      payload_2_s2.clone(),
      wa_2_s2,
      wv_2_s2,
      m_2_s2,
      eq_w2l_s2,
      eq_w2r_s2,
      eq_t2l_s2,
      eq_t2r_s2,
      folded_lw_s1[1].clone(),
    );

    let (nifs2, (folded_U_s2, _folded_W_s2), _folded_lw_s2) =
      NIFS::<E>::prove_with_multi_table_lookup(
        &ck,
        &ro_consts,
        &pp_digest,
        &str_local,
        &folded_U_s1,
        &folded_W_s1,
        &u_step2,
        &w_step2,
        &[bundle_1_s2, bundle_2_s2],
      )
      .expect("STAGE 0 step-2 prove must succeed");

    let public_bundles_s2 = vec![
      LookupPayloadPublicMultiTable::<E> {
        table_id: 0,
        comm_L: payload_1_s2.comm_L,
        comm_values: payload_1_s2.comm_values.clone(),
        comm_ts: payload_1_s2.comm_ts,
      },
      LookupPayloadPublicMultiTable::<E> {
        table_id: 1,
        comm_L: payload_2_s2.comm_L,
        comm_values: payload_2_s2.comm_values.clone(),
        comm_ts: payload_2_s2.comm_ts,
      },
    ];

    let verified_U_s2 = nifs2
      .verify_with_multi_table_lookup(
        &ro_consts,
        &pp_digest,
        &str_local,
        &folded_U_s1,
        &u_step2,
        &public_bundles_s2,
      )
      .expect("STAGE 0 step-2 verify must succeed");
    assert_eq!(
      folded_U_s2, verified_U_s2,
      "STAGE 0: step-2 prover/verifier must agree on folded U"
    );
    let running_u_after_step_2: FoldedInstance<E> = verified_U_s2;

    // STAGE 0 pre-check: T_lookup at fold-depth 2 must be Some(&[non-zero, non-zero]).
    {
      let t = running_u_after_step_2
        .t_lookup()
        .expect("STAGE 0 fixture: step-2 running U must carry T_lookup (Some)");
      assert_eq!(
        t.len(),
        2,
        "STAGE 0 fixture: k=2 — running U after step 2 must carry 2-entry T_lookup"
      );
      assert_ne!(
        t[0],
        Scalar::ZERO,
        "STAGE 0 fixture: T_lookup[0] after step 2 must be non-zero (else the (W1) extension isn't being exercised)"
      );
      assert_ne!(
        t[1],
        Scalar::ZERO,
        "STAGE 0 fixture: T_lookup[1] after step 2 must be non-zero (else the (W1) extension isn't being exercised)"
      );
    }

    // STAGE 0 step-2 byte-equivalence.
    let h_native_s2 = native_absorb_squeeze(&ro_consts, &running_u_after_step_2);
    let (h_circuit_s2, satisfied_s2) =
      circuit_absorb_squeeze(&ro_consts_circuit, &running_u_after_step_2);
    assert!(
      satisfied_s2,
      "STAGE 0 step-2: in-circuit absorb_in_ro CS must be satisfied. \
       Per pin §6.2 STOP-AND-ASK: halt all further GH-#5 milestones."
    );
    assert_eq!(
      h_native_s2, h_circuit_s2,
      "STAGE 0 step-2: native FoldedInstance::absorb_in_ro2 squeeze MUST equal \
       in-circuit AllocatedFoldedInstance::absorb_in_ro derived squeeze. \
       (h_native, h_circuit) = ({:?}, {:?}). \
       Per pin §6.2 STOP-AND-ASK: the §1.4 absorption-order pin is wrong; \
       halt all further GH-#5 milestones and re-derive.",
      h_native_s2, h_circuit_s2
    );
  }

  /// M.GH5.0 negative-arity test — `AllocatedFoldedInstance::alloc` MUST
  /// reject `FoldedInstance::X` of length != 1 with `SynthesisError::Unsatisfiable`.
  ///
  /// Per pin §1.4 corrigendum (Halpert 2026-05-09): the (W1) binding-via-hash
  /// extension is sound only at `num_io == 1`. The in-circuit allocator
  /// previously did `inst.X[0]` silently — a soundness-relevant truncation
  /// at any X-arity > 1. The corrigendum elevates `X.len() == 1` to a stated
  /// invariant; this test documents the fail-closed posture at the impl
  /// boundary so the invariant cannot drift back to silent truncation.
  ///
  /// Mirrors the fail-closed posture of `conditionally_select`'s
  /// T_lookup_per_table shape check (lines ~379-384).
  #[test]
  fn m_gh5_0_alloc_rejects_x_arity_two() {
    // Build a synthetic FoldedInstance with X of length 2 — the exact
    // shape that fell out of the prior (DirectCircuit, num_io == 2)
    // STAGE 0 fixture and falsified byte-equivalence at step 1.
    let inst_x_len_two = FoldedInstance::<E> {
      comm_W: Commitment::<E>::default(),
      comm_E: Commitment::<E>::default(),
      T: Scalar::ZERO,
      u: Scalar::ZERO,
      X: vec![Scalar::ZERO, Scalar::ZERO],
      comm_L: None,
      comm_ts: None,
      comm_inv_w: None,
      comm_inv_t: None,
      T_lookup: None,
    };

    let mut cs = TestConstraintSystem::<Scalar>::new();
    let result = AllocatedFoldedInstance::<E>::alloc(
      cs.namespace(|| "alloc rejects X.len() == 2"),
      Some(&inst_x_len_two),
    );

    match result {
      Err(SynthesisError::Unsatisfiable(msg)) => {
        assert!(
          msg.contains("num_io == 1"),
          "Negative-arity test: error message must mention the `num_io == 1` \
           invariant (pin §1.4 corrigendum). Got: {msg}"
        );
      }
      Ok(_) => panic!(
        "Negative-arity test: AllocatedFoldedInstance::alloc accepted a \
         FoldedInstance with X.len() == 2 — silent truncation regression. \
         Per pin §1.4 corrigendum: must fail-closed with \
         SynthesisError::Unsatisfiable mentioning `num_io == 1`."
      ),
      Err(other) => panic!(
        "Negative-arity test: expected SynthesisError::Unsatisfiable with \
         `num_io == 1` in the message; got {other:?}"
      ),
    }
  }
}
