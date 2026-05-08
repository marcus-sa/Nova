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

/// In-circuit representation of the multi-table lookup-side NIFS message
/// (GH-#2 M.6, design pin §5.1).
///
/// Companion to [`AllocatedLookupNIFS`] for the multi-table verifier
/// surface. Carries `k` per-table entries — the `k` `poly_lookup_j`
/// polynomials and the `2k` per-table inverse-witness commitments — that
/// the native [`NIFS`] populates under `--features lookup-fold` when
/// `prove_with_multi_table_lookup_inner` is invoked.
///
/// The single-table multi-column projection (the existing
/// [`AllocatedLookupNIFS`]) reads entry `[0]` of these vectors; the
/// multi-table allocation here owns the full per-table vector for the
/// per-table loop in [`super::AllocatedNIFS::verify_with_multi_table_lookup`].
pub struct AllocatedLookupNIFSMultiTable<E: Engine> {
  /// Per-table lookup-side polynomial (one entry per registered table,
  /// in `table_id`-ascending order per pin §2.2).
  pub(crate) poly_lookup: Vec<AllocatedUniPoly<E>>,
  /// Per-table inverse-witness commitment for `1/(w_j + r)`.
  pub(crate) comm_inv_w: Vec<AllocatedNonnativePoint<E>>,
  /// Per-table inverse-table commitment for `1/(T_j + r)`.
  pub(crate) comm_inv_t: Vec<AllocatedNonnativePoint<E>>,
}

impl<E: Engine> AllocatedLookupNIFSMultiTable<E> {
  /// Allocate from an optional native `NIFS<E>` carrying multi-table
  /// lookup data.
  ///
  /// `k` is the structurally-pinned table count
  /// (`LookupShape::multi_column_tables.len()`). The constructor
  /// allocates `k` `poly_lookup` entries at degree
  /// [`LOOKUP_POLY_DEGREE`] and `k` per-table inverse-witness/
  /// inverse-table commitments. If `nifs` is `Some`, its
  /// `poly_lookup` / `comm_inv_w` / `comm_inv_t` fields MUST be `Some`
  /// AND have length `k` (otherwise this is a degenerate NIFS that does
  /// not carry a multi-table payload — caller error). At synthesis time
  /// without a hint (`nifs = None`), the entries default to zero.
  pub fn alloc<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    nifs: Option<&NIFS<E>>,
    k: usize,
  ) -> Result<Self, SynthesisError> {
    let poly_lookup = (0..k)
      .map(|j| {
        AllocatedUniPoly::alloc(
          cs.namespace(|| format!("allocate poly_lookup[{}]", j)),
          LOOKUP_POLY_DEGREE,
          nifs
            .and_then(|n| n.poly_lookup.as_ref())
            .and_then(|v| v.get(j)),
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    let comm_inv_w = (0..k)
      .map(|j| {
        AllocatedNonnativePoint::alloc(
          cs.namespace(|| format!("allocate comm_inv_w[{}]", j)),
          nifs
            .and_then(|n| n.comm_inv_w.as_ref())
            .and_then(|v| v.get(j))
            .map(|c| c.to_coordinates()),
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    let comm_inv_t = (0..k)
      .map(|j| {
        AllocatedNonnativePoint::alloc(
          cs.namespace(|| format!("allocate comm_inv_t[{}]", j)),
          nifs
            .and_then(|n| n.comm_inv_t.as_ref())
            .and_then(|v| v.get(j))
            .map(|c| c.to_coordinates()),
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok(Self {
      poly_lookup,
      comm_inv_w,
      comm_inv_t,
    })
  }

  /// Number of registered tables (k).
  pub fn k(&self) -> usize {
    self.poly_lookup.len()
  }
}

/// In-circuit projection of [`crate::neutron::relation::LookupPayloadPublicMultiTable`]
/// — the per-step public lookup-witness commitments for one table
/// (GH-#2 M.6).
///
/// The verifier consumes a `&[AllocatedLookupPayloadPublicMultiTable<E>]`
/// of length `k` (the structurally-pinned table count), in
/// `table_id`-ascending order per pin §2.2.
pub struct AllocatedLookupPayloadPublicMultiTable<E: Engine> {
  /// Per-step lookup-witness commitment for the table's address column.
  pub comm_L: AllocatedNonnativePoint<E>,
  /// Per-step value-column commitments (Lasso §6.2). Empty for the
  /// single-column degenerate path (pin §2.2 step 5a empty-skip rule).
  pub comm_values: Vec<AllocatedNonnativePoint<E>>,
  /// Per-step multiplicity-vector commitment.
  pub comm_ts: AllocatedNonnativePoint<E>,
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

  /// GH-#2 M.6: in-circuit `verify_with_multi_table_lookup` mirrors the
  /// native multi-table transcript at the scalar-sequence layer (per
  /// pin §2.4) for `k = 2` registered tables. The resulting per-table
  /// in-circuit `T_lookup_out_per_table` must equal the native
  /// `verify_with_multi_table_lookup` per-table outputs computed from
  /// the same transcript replay.
  ///
  /// This is the gating regression test for M.6 per dispatch §3.
  /// It mirrors `stage_i_pri_in_circuit_verify_multi_column_matches_native`,
  /// extended to two structurally-pinned tables (per pin §1.2 list-of-
  /// instances composition). Both tables are queried at this fold step
  /// (no absent shape) — the absent-table differential is M.8 territory
  /// per dispatch.
  #[test]
  fn m6_in_circuit_verify_multi_table_matches_native_k2() {
    use super::AllocatedLookupNIFSMultiTable;
    use crate::neutron::circuit::lookup::AllocatedLookupPayloadPublicMultiTable;
    use crate::neutron::nifs::PerTableBundle;
    use crate::neutron::relation::{LookupPayload, LookupPayloadPublicMultiTable};

    let mut rng = ChaCha20Rng::seed_from_u64(0x1A55_E5A6_C1B2_F006);
    let ro_consts = RO2Constants::<E>::default();
    let ro_consts_circuit = RO2ConstantsCircuit::<E>::default();
    let pp_digest = Scalar::ZERO;

    // Build a small R1CS shape (mirrors the M.4 k=2 fixture and the
    // single-table multi-column in-circuit test).
    let num_cons: usize = 32;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut shape_cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut shape_cs);
    let shape = shape_cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // Two homogeneous-size random tables: n_1 = n_2 = 16, each with 1
    // value column. Random table contents pinned via ChaCha20Rng.
    let table_size = 16usize;
    let table_log2 = 4usize;
    let t1_col0: Vec<Scalar> = (0..table_size).map(|_| Scalar::random(&mut rng)).collect();
    let t2_col0: Vec<Scalar> = (0..table_size).map(|_| Scalar::random(&mut rng)).collect();

    let identity: Vec<Scalar> = (0..table_size).map(|i| Scalar::from(i as u64)).collect();
    let identity_comm_t1 = <E as Engine>::CE::commit(&ck, &identity, &Scalar::ZERO);
    let identity_comm_t2 = <E as Engine>::CE::commit(&ck, &identity, &Scalar::ZERO);

    let lookup_shape = LookupShape::<E> {
      tables: vec![
        LookupTableHandle {
          table_id: 0,
          size: table_size,
          commitment: identity_comm_t1,
        },
        LookupTableHandle {
          table_id: 1,
          size: table_size,
          commitment: identity_comm_t2,
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

    // Satisfying R1CS instance.
    let circuit2: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut sat_cs = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut sat_cs);
    let (U2, W2) = sat_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W2 = W2.pad(&shape);

    // Outer-base running state (per-table T_lookup_running = 0).
    let running_W = FoldedWitness::default(&str_local);
    let running_U = FoldedInstance::default(&str_local);

    // Build a satisfying multi-column witness pool for one table, with
    // distinct query indices per table to make the per-table threading
    // observable (a buggy implementation that aliased per-table state
    // would surface as scalar-sequence divergence).
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

    let queries_t1 = [0usize, 3, 7, 15];
    let queries_t2 = [1usize, 5, 9, 14];
    let (wa_1, wv_1, m_1) = build_satisfying_witness(&t1_col0, &queries_t1);
    let (wa_2, wv_2, m_2) = build_satisfying_witness(&t2_col0, &queries_t2);

    let comm_addr_1 = <E as Engine>::CE::commit(&ck, &wa_1, &Scalar::ZERO);
    let comm_v0_1 = <E as Engine>::CE::commit(&ck, &wv_1, &Scalar::ZERO);
    let comm_ts_1 = <E as Engine>::CE::commit(&ck, &m_1, &Scalar::ZERO);
    let comm_addr_2 = <E as Engine>::CE::commit(&ck, &wa_2, &Scalar::ZERO);
    let comm_v0_2 = <E as Engine>::CE::commit(&ck, &wv_2, &Scalar::ZERO);
    let comm_ts_2 = <E as Engine>::CE::commit(&ck, &m_2, &Scalar::ZERO);

    let payload_1 = LookupPayload::<E> {
      comm_L: comm_addr_1,
      comm_ts: comm_ts_1,
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![comm_v0_1],
    };
    let payload_2 = LookupPayload::<E> {
      comm_L: comm_addr_2,
      comm_ts: comm_ts_2,
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![comm_v0_2],
    };

    // Per-table eq polynomials (dimensions match the per-table
    // hypercube of size n_j, per pin §1.2 list-of-instances).
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
    let (eq_w1l, eq_w1r, eq_t1l, eq_t1r) = mk_eqs(&mut rng);
    let (eq_w2l, eq_w2r, eq_t2l, eq_t2r) = mk_eqs(&mut rng);

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

    let bundle_1 = PerTableBundle::<E> {
      table_id: 0,
      payload: payload_1.clone(),
      fresh_witness_address: wa_1,
      fresh_witness_value_columns: vec![wv_1],
      fresh_multiplicities: m_1,
      fresh_eq_w_left: eq_w1l,
      fresh_eq_w_right: eq_w1r,
      fresh_eq_t_left: eq_t1l,
      fresh_eq_t_right: eq_t1r,
      running_lw: mk_running_lw(),
    };
    let bundle_2 = PerTableBundle::<E> {
      table_id: 1,
      payload: payload_2.clone(),
      fresh_witness_address: wa_2,
      fresh_witness_value_columns: vec![wv_2],
      fresh_multiplicities: m_2,
      fresh_eq_w_left: eq_w2l,
      fresh_eq_w_right: eq_w2r,
      fresh_eq_t_left: eq_t2l,
      fresh_eq_t_right: eq_t2r,
      running_lw: mk_running_lw(),
    };

    // --- Native prove + verify (M.3 + M.4) ---
    let (nifs, _, _) = NIFS::<E>::prove_with_multi_table_lookup(
      &ck,
      &ro_consts,
      &pp_digest,
      &str_local,
      &running_U,
      &running_W,
      &U2,
      &W2,
      &[bundle_1.clone(), bundle_2.clone()],
    )
    .expect("k=2 multi-table prove must succeed");

    // M.4 verifier — produces the per-table T_lookup_out values that
    // the in-circuit M.6 verifier MUST match scalar-for-scalar.
    let public_bundles_native = vec![
      LookupPayloadPublicMultiTable::<E> {
        table_id: 0,
        comm_L: bundle_1.payload.comm_L,
        comm_values: bundle_1.payload.comm_values.clone(),
        comm_ts: bundle_1.payload.comm_ts,
      },
      LookupPayloadPublicMultiTable::<E> {
        table_id: 1,
        comm_L: bundle_2.payload.comm_L,
        comm_values: bundle_2.payload.comm_values.clone(),
        comm_ts: bundle_2.payload.comm_ts,
      },
    ];
    let _verified_U_native = nifs
      .verify_with_multi_table_lookup(
        &ro_consts,
        &pp_digest,
        &str_local,
        &running_U,
        &U2,
        &public_bundles_native,
      )
      .expect("k=2 native multi-table verify must succeed");

    // --- Replay the FS transcript using the in-circuit's single-IO U2
    // absorption pattern (same caveat as the Stage G / I-pri tests:
    // augmented circuit absorbs only U2.X[0], whereas native R1CSInstance
    // absorbs ALL X). The in-circuit verifier's transcript thus matches
    // a "single-IO native replay" rather than the production
    // verify_with_multi_table_lookup's transcript directly. This is the
    // pin §2.4 byte-equivalence layer: scalar-sequence equality across
    // the in-circuit-aligned native replay. ---
    use crate::{
      constants::NUM_CHALLENGE_BITS,
      neutron::lookup_sumcheck::lookup_running_claims_from,
      spartan::polys::univariate::UniPoly,
      traits::{AbsorbInRO2Trait, ROTrait},
    };

    let comm_inv_w_vec = nifs.comm_inv_w.as_ref().unwrap();
    let comm_inv_t_vec = nifs.comm_inv_t.as_ref().unwrap();
    let poly_lookup_vec = nifs.poly_lookup.as_ref().unwrap();
    assert_eq!(comm_inv_w_vec.len(), 2);
    assert_eq!(comm_inv_t_vec.len(), 2);
    assert_eq!(poly_lookup_vec.len(), 2);

    let mut ro = <E as Engine>::RO2::new(ro_consts.clone());
    ro.absorb(pp_digest);
    U2.comm_W.absorb_in_ro2(&mut ro);
    ro.absorb(U2.X[0]);
    // Step 2: per-table commitments BEFORE tau, in `table_id`-ascending
    // order per pin §2.2.
    for b in &public_bundles_native {
      b.comm_L.absorb_in_ro2(&mut ro);
      for cv in &b.comm_values {
        cv.absorb_in_ro2(&mut ro);
      }
      b.comm_ts.absorb_in_ro2(&mut ro);
    }
    // Step 3: tau.
    let _tau = ro.squeeze(NUM_CHALLENGE_BITS, false);
    nifs.comm_E.absorb_in_ro2(&mut ro);
    // Step 4: rho.
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);
    // Step 5a: per-table alpha (gated on c_j > 0). Both tables have one
    // value column, so both squeezes fire.
    for b in &public_bundles_native {
      if !b.comm_values.is_empty() {
        let _alpha_j = ro.squeeze(NUM_CHALLENGE_BITS, false);
      }
    }
    // Step 5b: per-table r_logup.
    for _ in 0..2 {
      let _r_logup_j = ro.squeeze(NUM_CHALLENGE_BITS, false);
    }
    // Step 6: per-table inverse-witness commitments (batched).
    for c in comm_inv_w_vec {
      c.absorb_in_ro2(&mut ro);
    }
    for c in comm_inv_t_vec {
      c.absorb_in_ro2(&mut ro);
    }
    // Step 7: R1CS-side poly absorb.
    <UniPoly<Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&nifs.poly, &mut ro);
    // Step 8: per-table poly_lookup absorb.
    for poly_lookup_j in poly_lookup_vec {
      <UniPoly<Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(poly_lookup_j, &mut ro);
    }
    // Step 9: r_b.
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // Compute the per-table T_lookup_out_j natively from the same
    // transcript replay (per pin §1.3 VECTOR composition). Outer base
    // → all per-table running scalars are zero.
    let t_lookup_running_vec = lookup_running_claims_from::<E>(&running_U);
    assert!(
      t_lookup_running_vec.is_empty(),
      "outer base running U must have empty per-table vector"
    );
    let native_t_lookup_out_per_table: Vec<Scalar> = (0..2)
      .map(|j| {
        let t_lookup_running_j = t_lookup_running_vec
          .get(j)
          .copied()
          .unwrap_or(Scalar::ZERO);
        LookupSumcheckInstance::<E>::verify_step(
          &rho,
          &r_b,
          &poly_lookup_vec[j],
          &t_lookup_running_j,
        )
        .expect("native verify_step per table must succeed")
      })
      .collect();

    // --- In-circuit verify_with_multi_table_lookup ---
    let mut cs = TestConstraintSystem::<Scalar>::new();
    let pp_digest_alloc =
      AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(pp_digest)).unwrap();
    let U1_alloc =
      AllocatedFoldedInstance::<E>::alloc(cs.namespace(|| "U1"), Some(&running_U)).unwrap();
    let U2_alloc =
      AllocatedNonnativeR1CSInstance::<E>::alloc(cs.namespace(|| "U2"), Some(&U2)).unwrap();
    let allocated_nifs =
      AllocatedNIFS::<E>::alloc(cs.namespace(|| "allocate nifs"), Some(&nifs), 5).unwrap();
    let allocated_lookups = AllocatedLookupNIFSMultiTable::<E>::alloc(
      cs.namespace(|| "allocate multi-table lookup nifs"),
      Some(&nifs),
      2,
    )
    .unwrap();

    // Build per-table public bundles (allocated form).
    let mk_public_bundle = |cs: &mut TestConstraintSystem<Scalar>,
                            tag: &str,
                            payload: &LookupPayload<E>|
     -> AllocatedLookupPayloadPublicMultiTable<E> {
      let comm_L = AllocatedNonnativePoint::<E>::alloc(
        cs.namespace(|| format!("{tag} comm_L pub")),
        Some(payload.comm_L.to_coordinates()),
      )
      .unwrap();
      let comm_values: Vec<_> = payload
        .comm_values
        .iter()
        .enumerate()
        .map(|(i, cv)| {
          AllocatedNonnativePoint::<E>::alloc(
            cs.namespace(|| format!("{tag} comm_values[{}] pub", i)),
            Some(cv.to_coordinates()),
          )
          .unwrap()
        })
        .collect();
      let comm_ts = AllocatedNonnativePoint::<E>::alloc(
        cs.namespace(|| format!("{tag} comm_ts pub")),
        Some(payload.comm_ts.to_coordinates()),
      )
      .unwrap();
      AllocatedLookupPayloadPublicMultiTable {
        comm_L,
        comm_values,
        comm_ts,
      }
    };
    let public_bundle_1 = mk_public_bundle(&mut cs, "table[0]", &payload_1);
    let public_bundle_2 = mk_public_bundle(&mut cs, "table[1]", &payload_2);
    let public_bundles_alloc = vec![public_bundle_1, public_bundle_2];

    // Per-table running-target allocations (outer base → zero per
    // pin §1.3).
    let t_lookup_running_alloc: Vec<_> = (0..2)
      .map(|j| alloc_zero(cs.namespace(|| format!("t_lookup_running[{}]", j))))
      .collect();

    let comm_W_fold =
      AllocatedNonnativePoint::<E>::default(cs.namespace(|| "comm_W_fold")).unwrap();
    let comm_E_fold =
      AllocatedNonnativePoint::<E>::default(cs.namespace(|| "comm_E_fold")).unwrap();

    let out = allocated_nifs
      .verify_with_multi_table_lookup(
        cs.namespace(|| "in-circuit verify_with_multi_table_lookup"),
        &pp_digest_alloc,
        &U1_alloc,
        &U2_alloc,
        &allocated_lookups,
        &public_bundles_alloc,
        &t_lookup_running_alloc,
        &comm_W_fold,
        &comm_E_fold,
        ro_consts_circuit,
      )
      .expect("in-circuit multi-table verify must synthesise cleanly");

    assert!(
      cs.is_satisfied(),
      "in-circuit multi-table verify must produce a satisfied constraint \
       system; first unsatisfied: {:?}",
      cs.which_is_unsatisfied()
    );

    assert_eq!(
      out.T_lookup_out_per_table.len(),
      2,
      "M.6 in-circuit output must carry one T_lookup_out per registered table"
    );

    // Pin §2.4 / dispatch §3 gating assertion: in-circuit per-table
    // T_lookup_out_j MUST equal the native per-table values computed
    // from the equivalent (in-circuit-aligned) FS transcript replay.
    for (j, native_v) in native_t_lookup_out_per_table.iter().enumerate() {
      let circuit_v = out.T_lookup_out_per_table[j]
        .get_value()
        .unwrap_or_else(|| panic!("T_lookup_out[{}] witness must be assigned", j));
      assert_eq!(
        circuit_v, *native_v,
        "M.6 FS-rebinding (k=2): in-circuit T_lookup_out[{}] must equal \
         native verify_step output",
        j
      );
    }

    // Pin §1.3 VECTOR composition: per-table independent threading.
    // With distinct query indices per table (queries_t1 ≠ queries_t2)
    // and random table contents, the two per-table T_lookup_out values
    // are overwhelmingly distinct. A buggy implementation that aliased
    // per-table state would fail this.
    assert_ne!(
      native_t_lookup_out_per_table[0], native_t_lookup_out_per_table[1],
      "per-table T_lookup_out values must thread independently"
    );
  }
}
