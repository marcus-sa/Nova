//! In-circuit lookup-side companion to `AllocatedNIFS` (Stage G, C1-beta).
//!
//! Mirrors the native `verify_with_lookup` (`src/neutron/nifs.rs:591-672`)
//! transcript and closed-form verifier-side step
//! (`LookupSumcheckInstance::verify_step`, `src/neutron/lookup_sumcheck.rs`).
//!
//! ## Degree-5 (NOT degree-2) — corrected from outline §G.1
//!
//! The C1-β implementation outline §G.1 (`docs/research/cryptography/
//! c1-beta-spike-implementation-outline-stages-b-f.md`) was written prior to
//! Stage B's empirical falsification of the degree-2 claim. Stage B
//! established `poly_lookup` is degree **5** (6 evaluations), mirroring the
//! R1CS-side `prove_helper`. This module allocates `AllocatedUniPoly` at
//! degree 5 to match. Cost increase vs. degree 2: `evaluate(r_b)` does 5
//! mults instead of 2. Acceptable for spike scope.
//!
//! ## Scope
//!
//! All items here are gated behind `#[cfg(feature = "lookup-fold")]` so the
//! non-lookup augmented-circuit verify path is byte-identical to the legacy
//! version when the feature is off.
#![cfg(feature = "lookup-fold")]
// Stage G lands the in-circuit lookup-side primitives in isolation; the
// augmented circuit's Stage-G test exercises them (`circuit/mod.rs` Stage G
// test) but the synthesize-loop wiring is deferred. Until the wiring lands,
// these public items have no in-crate consumer outside the test module.
#![allow(dead_code)]

use crate::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::ecc::AllocatedNonnativePoint,
  neutron::{circuit::univariate::AllocatedUniPoly, nifs::NIFS},
  traits::{commitment::CommitmentTrait, Engine},
};
use ff::Field;

/// Degree of the in-circuit lookup-side polynomial. Pinned by Stage B's
/// empirical analysis in `lookup_sumcheck.rs:9-27`.
pub const LOOKUP_POLY_DEGREE: usize = 5;

/// In-circuit representation of the lookup-side NIFS message.
///
/// Companion to [`super::nifs::AllocatedNIFS`]: carries `poly_lookup` and
/// the two inverse-witness commitments that the native [`NIFS`] adds under
/// `--features lookup-fold`. See the module-level docs for the degree-5
/// rationale.
pub struct AllocatedLookupNIFS<E: Engine> {
  pub(crate) poly_lookup: AllocatedUniPoly<E>,
  pub(crate) comm_inv_w: AllocatedNonnativePoint<E>,
  pub(crate) comm_inv_t: AllocatedNonnativePoint<E>,
}

impl<E: Engine> AllocatedLookupNIFS<E> {
  /// Allocate from an optional native `NIFS<E>` carrying lookup data.
  ///
  /// Mirrors [`super::nifs::AllocatedNIFS::alloc`]. If `nifs` is `Some`, its
  /// `poly_lookup`, `comm_inv_w`, `comm_inv_t` fields must be `Some`
  /// (otherwise this is a degenerate NIFS that does not carry a lookup
  /// payload — caller error).
  pub fn alloc<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    nifs: Option<&NIFS<E>>,
  ) -> Result<Self, SynthesisError> {
    // Multi-table extension (GH-#2, design pin §5.1): `poly_lookup`,
    // `comm_inv_w`, `comm_inv_t` on `NIFS<E>` are now `Option<Vec<…>>`.
    // The single-table in-circuit allocation projects to entry [0]; the
    // multi-table in-circuit allocation (M.6) is a separate type
    // (`AllocatedLookupNIFSMultiTable`) introduced at that stage.
    let poly_lookup = AllocatedUniPoly::alloc(
      cs.namespace(|| "allocate poly_lookup"),
      LOOKUP_POLY_DEGREE,
      nifs.and_then(|n| n.poly_lookup.as_ref()).and_then(|v| v.first()),
    )?;

    let comm_inv_w = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_inv_w"),
      nifs
        .and_then(|n| n.comm_inv_w.as_ref())
        .and_then(|v| v.first())
        .map(|c| c.to_coordinates()),
    )?;

    let comm_inv_t = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_inv_t"),
      nifs
        .and_then(|n| n.comm_inv_t.as_ref())
        .and_then(|v| v.first())
        .map(|c| c.to_coordinates()),
    )?;

    Ok(Self {
      poly_lookup,
      comm_inv_w,
      comm_inv_t,
    })
  }

  /// In-circuit closed-form verifier step for the lookup-side sumcheck.
  ///
  /// Mirrors [`crate::neutron::lookup_sumcheck::LookupSumcheckInstance::verify_step`]:
  ///
  /// 1. (C)-binding assertion `poly_lookup(0) + poly_lookup(1) = t_lookup_running`
  /// 2. Compute `eq_rho_r_b = (1-rho)(1-r_b) + rho*r_b`
  /// 3. Evaluate `poly_lookup` at `r_b`
  /// 4. Compute `t_lookup_out = eval_r_b / eq_rho_r_b`, enforced by
  ///    `t_lookup_out * eq_rho_r_b = eval_r_b`.
  ///
  /// Caller is responsible for FS-transcript absorptions (see
  /// `AllocatedNIFS::verify_with_lookup` for the canonical orchestration).
  ///
  /// Returns the next-step lookup-side running target `T_lookup_out`.
  pub fn verify_step<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    rho: &AllocatedNum<E::Scalar>,
    r_b: &AllocatedNum<E::Scalar>,
    t_lookup_running: &AllocatedNum<E::Scalar>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
    // (1) (C)-binding: poly_lookup(0) + poly_lookup(1) = t_lookup_running.
    self.poly_lookup.check_poly_zero_poly_one_with(
      cs.namespace(|| "lookup poly(0)+poly(1) = t_lookup_running"),
      t_lookup_running,
    )?;

    // (2) eq_rho_r_b = (1-rho)(1-r_b) + rho*r_b.
    // Compute via two sub-allocations to mirror AllocatedNIFS::verify
    // (`src/neutron/circuit/nifs.rs:103-127`).
    let eq_rho_r_b_one =
      AllocatedNum::alloc(cs.namespace(|| "allocate lookup eq_rho_r_b_one"), || {
        let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let r_b_v = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok((E::Scalar::ONE - rho_v) * (E::Scalar::ONE - r_b_v))
      })?;
    cs.enforce(
      || "lookup eq_rho_r_b_one = (1 - rho)(1 - r_b)",
      |lc| lc + CS::one() - rho.get_variable(),
      |lc| lc + CS::one() - r_b.get_variable(),
      |lc| lc + eq_rho_r_b_one.get_variable(),
    );

    let eq_rho_r_b = AllocatedNum::alloc(cs.namespace(|| "allocate lookup eq_rho_r_b"), || {
      let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let r_b_v = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((E::Scalar::ONE - rho_v) * (E::Scalar::ONE - r_b_v) + rho_v * r_b_v)
    })?;
    cs.enforce(
      || "lookup eq_rho_r_b = (1 - rho)(1 - r_b) + rho*r_b",
      |lc| lc + rho.get_variable(),
      |lc| lc + r_b.get_variable(),
      |lc| lc + eq_rho_r_b.get_variable() - eq_rho_r_b_one.get_variable(),
    );

    // (3) Evaluate poly_lookup at r_b.
    let eval_r_b = self
      .poly_lookup
      .evaluate(cs.namespace(|| "evaluate poly_lookup at r_b"), r_b)?;

    // (4) t_lookup_out = eval_r_b / eq_rho_r_b, enforced by
    //     t_lookup_out * eq_rho_r_b = eval_r_b.
    let t_lookup_out = AllocatedNum::alloc(cs.namespace(|| "allocate t_lookup_out"), || {
      let eval = eval_r_b
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let eq_inv = eq_rho_r_b
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?
        .invert()
        .unwrap();
      Ok(eval * eq_inv)
    })?;
    cs.enforce(
      || "enforce t_lookup_out * eq_rho_r_b = eval_r_b",
      |lc| lc + t_lookup_out.get_variable(),
      |lc| lc + eq_rho_r_b.get_variable(),
      |lc| lc + eval_r_b.get_variable(),
    );

    Ok(t_lookup_out)
  }

  /// Absorb `comm_inv_w` and `comm_inv_t` into the RO. Sequencing is the
  /// caller's responsibility — see `AllocatedNIFS::verify_with_lookup`.
  pub fn absorb_inv_comms_in_ro<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    ro: &mut E::RO2Circuit,
  ) -> Result<(), SynthesisError> {
    self
      .comm_inv_w
      .absorb_in_ro(cs.namespace(|| "absorb comm_inv_w"), ro)?;
    self
      .comm_inv_t
      .absorb_in_ro(cs.namespace(|| "absorb comm_inv_t"), ro)?;
    Ok(())
  }

  /// Absorb `poly_lookup` into the RO. Caller-ordered.
  pub fn absorb_poly_in_ro(&self, ro: &mut E::RO2Circuit) {
    self.poly_lookup.absorb_in_ro(ro);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      util_cs::test_cs::TestConstraintSystem,
      Circuit,
    },
    gadgets::utils::alloc_zero,
    neutron::{
      circuit::{
        nifs::AllocatedNIFS, r1cs::AllocatedNonnativeR1CSInstance,
        relation::AllocatedFoldedInstance,
      },
      lookup_sumcheck::LookupSumcheckInstance,
      nifs::NIFS,
      relation::{
        FoldedInstance, FoldedWitness, LookupPayload, LookupRunningWitness, LookupShape,
        LookupTableHandle, MultiColumnLookupTable, Structure,
      },
    },
    provider::{hyperkzg::EvaluationEngine, Bn256EngineKZG},
    r1cs::R1CSShape,
    spartan::{
      direct::DirectCircuit,
      polys::power::PowPolynomial,
      snark::RelaxedR1CSSNARK,
    },
    traits::{
      circuit::NonTrivialCircuit, commitment::CommitmentEngineTrait,
      snark::RelaxedR1CSSNARKTrait, RO2Constants, RO2ConstantsCircuit,
    },
    Commitment,
  };
  use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

  type E = Bn256EngineKZG;
  type Scalar = <E as Engine>::Scalar;
  type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

  /// Stage G: in-circuit `verify_with_lookup` synthesises without
  /// SynthesisError on a real `NIFS::prove_with_lookup` output, and the
  /// resulting `T_lookup_out` agrees with the native verifier byte-for-byte
  /// (FS-rebinding test §A.5.2(b)).
  #[test]
  fn stage_g_in_circuit_verify_with_lookup_matches_native() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_57A6);
    let ro_consts = RO2Constants::<E>::default();
    let ro_consts_circuit = RO2ConstantsCircuit::<E>::default();
    let pp_digest = Scalar::ZERO;

    // --- Build a small R1CS shape ---
    let num_cons: usize = 32;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut shape_cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut shape_cs);
    let shape = shape_cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // --- Build a 16-entry lookup table ---
    let table_size = 16usize;
    let table_log2 = 4usize;
    let table: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 7 + 3) as u64))
      .collect();
    let table_comm = <E as Engine>::CE::commit(&ck, &table, &Scalar::ZERO);
    let lookup_shape = LookupShape::<E> {
      tables: vec![LookupTableHandle {
        table_id: 0,
        size: table_size,
        commitment: table_comm,
      }],
      multi_column_tables: Vec::new(),
      num_addr_columns: 1,
      num_witness_columns: 1,
      witness_ell_cached: table_log2,
    };
    let str = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str.S.clone();

    // --- Make a satisfying U2/W2 ---
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut sat_cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut sat_cs);
    let (U2, W2) = sat_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W2 = W2.pad(&shape);

    // Default (outer-base) running state.
    let running_W = FoldedWitness::default(&str);
    let running_U = FoldedInstance::default(&str);
    let running_lw = LookupRunningWitness::default(&lookup_shape);

    // --- Build a satisfying lookup payload (4 in-table queries, padded) ---
    let query_indices = [0usize, 3, 7, 15];
    let mut witness_pool = vec![Scalar::ZERO; table_size];
    let mut multiplicities = vec![Scalar::ZERO; table_size];
    for (i, &idx) in query_indices.iter().enumerate() {
      witness_pool[i] = table[idx];
      multiplicities[idx] += Scalar::ONE;
    }
    for i in query_indices.len()..table_size {
      witness_pool[i] = table[0];
      multiplicities[0] += Scalar::ONE;
    }
    let comm_L = <E as Engine>::CE::commit(&ck, &witness_pool, &Scalar::ZERO);
    let comm_ts = <E as Engine>::CE::commit(&ck, &multiplicities, &Scalar::ZERO);
    let payload = LookupPayload::<E> {
      comm_L,
      comm_ts,
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: Vec::new(),
    };
    let tau_w = Scalar::random(&mut rng);
    let pow_w = PowPolynomial::new(&tau_w, table_log2);
    let (w_left, w_right) = lookup_shape.witness_split();
    let combined_w = pow_w.split_evals(w_left, w_right);
    let (eq_w_left, eq_w_right) = combined_w.split_at(w_left);
    let tau_t = Scalar::random(&mut rng);
    let pow_t = PowPolynomial::new(&tau_t, table_log2);
    let (t_left, t_right) = lookup_shape.table_split();
    let combined_t = pow_t.split_evals(t_left, t_right);
    let (eq_t_left, eq_t_right) = combined_t.split_at(t_left);

    // --- Native prove_with_lookup ---
    let (nifs, (_folded_U, _folded_W), _folded_lw) = NIFS::<E>::prove_with_lookup(
      &ck,
      &ro_consts,
      &pp_digest,
      &str,
      &running_U,
      &running_W,
      &U2,
      &W2,
      &payload,
      &running_lw,
      &witness_pool,
      &table,
      &multiplicities,
      eq_w_left.to_vec(),
      eq_w_right.to_vec(),
      eq_t_left.to_vec(),
      eq_t_right.to_vec(),
    )
    .expect("native prove_with_lookup must succeed");

    // The payload's inverse commitments are filled in by prove_with_lookup
    // via the NIFS struct; for the in-circuit verifier we use the NIFS
    // commitments directly (the native verifier does the same).
    //
    // Multi-table extension (GH-#2, design pin §5.1): NIFS lookup
    // commitments are now `Option<Vec<Commitment<E>>>`. Single-table tests
    // project to entry [0].
    let comm_inv_w = nifs
      .comm_inv_w
      .as_ref()
      .expect("prove_with_lookup must populate comm_inv_w")[0];
    let comm_inv_t = nifs
      .comm_inv_t
      .as_ref()
      .expect("prove_with_lookup must populate comm_inv_t")[0];

    // Native verify_with_lookup — the in-circuit must match this.
    let payload_for_verify = LookupPayload::<E> {
      comm_L: payload.comm_L,
      comm_ts: payload.comm_ts,
      comm_inv_w,
      comm_inv_t,
      T2_lookup: Scalar::ZERO,
      comm_values: Vec::new(),
    };
    let _verified_U = nifs
      .verify_with_lookup(&ro_consts, &pp_digest, &running_U, &U2, &payload_for_verify)
      .expect("native verify_with_lookup must succeed");

    // Replay the FS transcript using the SAME single-IO U2 absorption
    // that the in-circuit verifier uses. The in-circuit
    // `AllocatedNonnativeR1CSInstance` absorbs only `u.X[0]` (it has a
    // single `X` field, see `circuit/r1cs.rs`), whereas the native
    // `R1CSInstance::absorb_in_ro2` absorbs ALL IO columns. This is a
    // known IO-layout constraint of the augmented circuit (which is
    // designed for `arity ≤ 1` circuits). We mirror the in-circuit
    // absorption pattern here so the FS-rebinding equality test
    // (§A.5.2(b)) is meaningful at this layer; the full multi-IO
    // FS-rebinding test will be exercised once the augmented circuit
    // wires `verify_with_lookup` end-to-end (Stage I+).
    use crate::{
      constants::NUM_CHALLENGE_BITS,
      neutron::lookup_sumcheck::lookup_running_claims_from,
      spartan::polys::univariate::UniPoly,
      traits::{AbsorbInRO2Trait, ROTrait},
    };

    let mut ro = <E as Engine>::RO2::new(ro_consts.clone());
    ro.absorb(pp_digest);
    // Mirror AllocatedNonnativeR1CSInstance::absorb_in_ro: comm_W then X[0] only.
    U2.comm_W.absorb_in_ro2(&mut ro);
    ro.absorb(U2.X[0]);
    payload.comm_L.absorb_in_ro2(&mut ro);
    payload.comm_ts.absorb_in_ro2(&mut ro);
    let _tau = ro.squeeze(NUM_CHALLENGE_BITS, false);
    nifs.comm_E.absorb_in_ro2(&mut ro);
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);
    let _r_logup = ro.squeeze(NUM_CHALLENGE_BITS, false);
    comm_inv_w.absorb_in_ro2(&mut ro);
    comm_inv_t.absorb_in_ro2(&mut ro);
    <UniPoly<Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&nifs.poly, &mut ro);
    let poly_lookup_ref = &nifs.poly_lookup.as_ref().unwrap()[0];
    <UniPoly<Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(poly_lookup_ref, &mut ro);
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // Multi-table extension (GH-#2): `lookup_running_claims_from` returns
    // `Vec<E::Scalar>`. Single-table tests project to entry [0].
    let t_lookup_running_vec = lookup_running_claims_from::<E>(&running_U);
    let t_lookup_running = t_lookup_running_vec
      .first()
      .copied()
      .unwrap_or(Scalar::ZERO);
    let native_t_lookup_out = LookupSumcheckInstance::<E>::verify_step(
      &rho,
      &r_b,
      poly_lookup_ref,
      &t_lookup_running,
    )
    .expect("native verify_step must succeed on circuit-equivalent transcript");

    // --- In-circuit verify_with_lookup ---
    let mut cs = TestConstraintSystem::<Scalar>::new();

    let pp_digest_alloc =
      AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(pp_digest)).unwrap();

    let U1_alloc =
      AllocatedFoldedInstance::<E>::alloc(cs.namespace(|| "U1"), Some(&running_U)).unwrap();
    let U2_alloc =
      AllocatedNonnativeR1CSInstance::<E>::alloc(cs.namespace(|| "U2"), Some(&U2)).unwrap();

    let allocated_nifs = AllocatedNIFS::<E>::alloc(
      cs.namespace(|| "allocate nifs"),
      Some(&nifs),
      5,
    )
    .unwrap();
    let allocated_lookup =
      AllocatedLookupNIFS::<E>::alloc(cs.namespace(|| "allocate lookup nifs"), Some(&nifs))
        .unwrap();

    let comm_L_pub_alloc = AllocatedNonnativePoint::<E>::alloc(
      cs.namespace(|| "comm_L pub"),
      Some(payload.comm_L.to_coordinates()),
    )
    .unwrap();
    let comm_ts_pub_alloc = AllocatedNonnativePoint::<E>::alloc(
      cs.namespace(|| "comm_ts pub"),
      Some(payload.comm_ts.to_coordinates()),
    )
    .unwrap();

    // outer base => t_lookup_running = 0
    let t_lookup_running_alloc = alloc_zero(cs.namespace(|| "t_lookup_running"));

    // Untrusted folded-commitment hints — for this synthesis check we
    // supply default placeholders since fold output is not asserted here
    // (the native FS-rebinding scalar comparison is the test target).
    let comm_W_fold = AllocatedNonnativePoint::<E>::default(cs.namespace(|| "comm_W_fold")).unwrap();
    let comm_E_fold = AllocatedNonnativePoint::<E>::default(cs.namespace(|| "comm_E_fold")).unwrap();

    let out = allocated_nifs
      .verify_with_lookup(
        cs.namespace(|| "in-circuit verify_with_lookup"),
        &pp_digest_alloc,
        &U1_alloc,
        &U2_alloc,
        &allocated_lookup,
        &comm_L_pub_alloc,
        &comm_ts_pub_alloc,
        &t_lookup_running_alloc,
        &comm_W_fold,
        &comm_E_fold,
        ro_consts_circuit,
      )
      .expect("in-circuit verify_with_lookup synthesised cleanly");

    assert!(
      cs.is_satisfied(),
      "in-circuit verify_with_lookup must produce a satisfied constraint system; \
       first unsatisfied: {:?}",
      cs.which_is_unsatisfied()
    );

    let circuit_t_lookup_out = out
      .T_lookup_out
      .get_value()
      .expect("T_lookup_out witness must be assigned");

    assert_eq!(
      circuit_t_lookup_out, native_t_lookup_out,
      "FS-rebinding: in-circuit T_lookup_out must equal native verify_step output"
    );
  }

  /// Stage I-pri: in-circuit `verify_with_multi_column_lookup` mirrors the
  /// native multi-column transcript byte-for-byte, including the α
  /// squeeze between `rho` and `r_logup`. The resulting in-circuit
  /// `T_lookup_out` must equal the native `verify_step` output computed
  /// from the same transcript replay.
  #[test]
  fn stage_i_pri_in_circuit_verify_multi_column_matches_native() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x1A55_E5A8);
    let ro_consts = RO2Constants::<E>::default();
    let ro_consts_circuit = RO2ConstantsCircuit::<E>::default();
    let pp_digest = Scalar::ZERO;

    // Build a small R1CS shape.
    let num_cons: usize = 32;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut shape_cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut shape_cs);
    let shape = shape_cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // 16-row table with 2 value columns.
    let table_size = 16usize;
    let table_log2 = 4usize;
    let t1: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 11 + 5) as u64))
      .collect();
    let t2: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 17 + 9) as u64))
      .collect();

    let identity: Vec<Scalar> =
      (0..table_size).map(|i| Scalar::from(i as u64)).collect();
    let identity_comm = <E as Engine>::CE::commit(&ck, &identity, &Scalar::ZERO);

    let lookup_shape = LookupShape::<E> {
      tables: vec![LookupTableHandle {
        table_id: 0,
        size: table_size,
        commitment: identity_comm,
      }],
      multi_column_tables: vec![MultiColumnLookupTable {
        table_id: 0,
        size: table_size,
        columns: vec![t1.clone(), t2.clone()],
        value_commitments: vec![
          <E as Engine>::CE::commit(&ck, &t1, &Scalar::ZERO),
          <E as Engine>::CE::commit(&ck, &t2, &Scalar::ZERO),
        ],
      }],
      num_addr_columns: 1,
      num_witness_columns: 3,
      witness_ell_cached: table_log2,
    };
    let str = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str.S.clone();

    // Satisfying U2/W2.
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut sat_cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut sat_cs);
    let (U2, W2) = sat_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W2 = W2.pad(&shape);

    let running_W = FoldedWitness::default(&str);
    let running_U = FoldedInstance::default(&str);
    let running_lw = LookupRunningWitness::default(&lookup_shape);

    // Build a satisfying multi-column witness pool.
    let query_indices = [0usize, 3, 7, 15];
    let mut witness_addr = vec![Scalar::ZERO; table_size];
    let mut witness_v1 = vec![Scalar::ZERO; table_size];
    let mut witness_v2 = vec![Scalar::ZERO; table_size];
    let mut multiplicities = vec![Scalar::ZERO; table_size];
    for (i, &idx) in query_indices.iter().enumerate() {
      witness_addr[i] = Scalar::from(idx as u64);
      witness_v1[i] = t1[idx];
      witness_v2[i] = t2[idx];
      multiplicities[idx] += Scalar::ONE;
    }
    for i in query_indices.len()..table_size {
      witness_addr[i] = Scalar::from(0u64);
      witness_v1[i] = t1[0];
      witness_v2[i] = t2[0];
      multiplicities[0] += Scalar::ONE;
    }

    let comm_addr = <E as Engine>::CE::commit(&ck, &witness_addr, &Scalar::ZERO);
    let comm_v1 = <E as Engine>::CE::commit(&ck, &witness_v1, &Scalar::ZERO);
    let comm_v2 = <E as Engine>::CE::commit(&ck, &witness_v2, &Scalar::ZERO);
    let comm_ts = <E as Engine>::CE::commit(&ck, &multiplicities, &Scalar::ZERO);

    let payload = LookupPayload::<E> {
      comm_L: comm_addr,
      comm_ts,
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![comm_v1, comm_v2],
    };

    let tau_w = Scalar::random(&mut rng);
    let pow_w = PowPolynomial::new(&tau_w, table_log2);
    let (w_left, w_right) = lookup_shape.witness_split();
    let combined_w = pow_w.split_evals(w_left, w_right);
    let (eq_w_left, eq_w_right) = combined_w.split_at(w_left);
    let tau_t = Scalar::random(&mut rng);
    let pow_t = PowPolynomial::new(&tau_t, table_log2);
    let (t_left, t_right) = lookup_shape.table_split();
    let combined_t_eq = pow_t.split_evals(t_left, t_right);
    let (eq_t_left, eq_t_right) = combined_t_eq.split_at(t_left);

    // Native prove with multi-column lookup.
    let (nifs, _, _) = NIFS::<E>::prove_with_multi_column_lookup(
      &ck,
      &ro_consts,
      &pp_digest,
      &str,
      &running_U,
      &running_W,
      &U2,
      &W2,
      &payload,
      &running_lw,
      0u64,
      &witness_addr,
      &[witness_v1.clone(), witness_v2.clone()],
      &multiplicities,
      eq_w_left.to_vec(),
      eq_w_right.to_vec(),
      eq_t_left.to_vec(),
      eq_t_right.to_vec(),
    )
    .expect("prove_with_multi_column_lookup must succeed");

    // Multi-table extension (GH-#2): NIFS lookup commitments are now Vec.
    let comm_inv_w = nifs.comm_inv_w.as_ref().unwrap()[0];
    let comm_inv_t = nifs.comm_inv_t.as_ref().unwrap()[0];

    // Native verify_with_multi_column_lookup.
    let payload_for_verify = LookupPayload::<E> {
      comm_L: payload.comm_L,
      comm_ts: payload.comm_ts,
      comm_inv_w,
      comm_inv_t,
      T2_lookup: Scalar::ZERO,
      comm_values: payload.comm_values.clone(),
    };
    let _verified = nifs
      .verify_with_multi_column_lookup(
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &U2,
        &payload_for_verify,
        0u64,
      )
      .expect("native verify must succeed");

    // Replay the FS transcript using the in-circuit's single-IO U2
    // absorption pattern (same caveat as the Stage G test).
    use crate::{
      constants::NUM_CHALLENGE_BITS,
      neutron::lookup_sumcheck::lookup_running_claims_from,
      spartan::polys::univariate::UniPoly,
      traits::{AbsorbInRO2Trait, ROTrait},
    };

    let mut ro = <E as Engine>::RO2::new(ro_consts.clone());
    ro.absorb(pp_digest);
    U2.comm_W.absorb_in_ro2(&mut ro);
    ro.absorb(U2.X[0]);
    payload.comm_L.absorb_in_ro2(&mut ro);
    for cv in &payload.comm_values {
      cv.absorb_in_ro2(&mut ro);
    }
    payload.comm_ts.absorb_in_ro2(&mut ro);
    let _tau = ro.squeeze(NUM_CHALLENGE_BITS, false);
    nifs.comm_E.absorb_in_ro2(&mut ro);
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);
    let _alpha = ro.squeeze(NUM_CHALLENGE_BITS, false);
    let _r_logup = ro.squeeze(NUM_CHALLENGE_BITS, false);
    comm_inv_w.absorb_in_ro2(&mut ro);
    comm_inv_t.absorb_in_ro2(&mut ro);
    <UniPoly<Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&nifs.poly, &mut ro);
    let poly_lookup_ref = &nifs.poly_lookup.as_ref().unwrap()[0];
    <UniPoly<Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(poly_lookup_ref, &mut ro);
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // Multi-table extension (GH-#2): `lookup_running_claims_from` returns Vec.
    let t_lookup_running_vec = lookup_running_claims_from::<E>(&running_U);
    let t_lookup_running = t_lookup_running_vec
      .first()
      .copied()
      .unwrap_or(Scalar::ZERO);
    let native_t_lookup_out = LookupSumcheckInstance::<E>::verify_step(
      &rho,
      &r_b,
      poly_lookup_ref,
      &t_lookup_running,
    )
    .expect("native verify_step must succeed");

    // In-circuit verify_with_multi_column_lookup.
    let mut cs = TestConstraintSystem::<Scalar>::new();
    let pp_digest_alloc =
      AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(pp_digest)).unwrap();
    let U1_alloc =
      AllocatedFoldedInstance::<E>::alloc(cs.namespace(|| "U1"), Some(&running_U)).unwrap();
    let U2_alloc =
      AllocatedNonnativeR1CSInstance::<E>::alloc(cs.namespace(|| "U2"), Some(&U2)).unwrap();
    let allocated_nifs =
      AllocatedNIFS::<E>::alloc(cs.namespace(|| "allocate nifs"), Some(&nifs), 5).unwrap();
    let allocated_lookup = AllocatedLookupNIFS::<E>::alloc(
      cs.namespace(|| "allocate lookup nifs"),
      Some(&nifs),
    )
    .unwrap();

    let comm_L_pub_alloc = AllocatedNonnativePoint::<E>::alloc(
      cs.namespace(|| "comm_L pub"),
      Some(payload.comm_L.to_coordinates()),
    )
    .unwrap();
    let comm_ts_pub_alloc = AllocatedNonnativePoint::<E>::alloc(
      cs.namespace(|| "comm_ts pub"),
      Some(payload.comm_ts.to_coordinates()),
    )
    .unwrap();
    let comm_values_pub_alloc: Vec<_> = payload
      .comm_values
      .iter()
      .enumerate()
      .map(|(i, cv)| {
        AllocatedNonnativePoint::<E>::alloc(
          cs.namespace(|| format!("comm_values[{}] pub", i)),
          Some(cv.to_coordinates()),
        )
        .unwrap()
      })
      .collect();

    let t_lookup_running_alloc = alloc_zero(cs.namespace(|| "t_lookup_running"));
    let comm_W_fold = AllocatedNonnativePoint::<E>::default(cs.namespace(|| "comm_W_fold")).unwrap();
    let comm_E_fold = AllocatedNonnativePoint::<E>::default(cs.namespace(|| "comm_E_fold")).unwrap();

    let out = allocated_nifs
      .verify_with_multi_column_lookup(
        cs.namespace(|| "in-circuit verify_with_multi_column_lookup"),
        &pp_digest_alloc,
        &U1_alloc,
        &U2_alloc,
        &allocated_lookup,
        &comm_L_pub_alloc,
        &comm_values_pub_alloc,
        &comm_ts_pub_alloc,
        &t_lookup_running_alloc,
        &comm_W_fold,
        &comm_E_fold,
        ro_consts_circuit,
      )
      .expect("in-circuit verify_with_multi_column_lookup must synthesise cleanly");

    assert!(
      cs.is_satisfied(),
      "in-circuit multi-column verify must be satisfied; first unsatisfied: {:?}",
      cs.which_is_unsatisfied()
    );

    let circuit_t_lookup_out = out
      .T_lookup_out
      .get_value()
      .expect("T_lookup_out witness must be assigned");
    assert_eq!(
      circuit_t_lookup_out, native_t_lookup_out,
      "Stage I-pri FS-rebinding: in-circuit multi-column T_lookup_out \
       must equal native verify_step output"
    );
  }
}
