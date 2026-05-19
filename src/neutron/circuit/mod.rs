//! There are two augmented circuits: the primary and the secondary.
//! Each of them is over a curve in a 2-cycle of elliptic curves.
//! We have two running instances. Each circuit takes as input 2 hashes: one for each
//! of the running instances. Each of these hashes is H(params = H(shape, ck), i, z0, zi, U).
//! Each circuit folds the last invocation of the other into the running instance

use crate::{
  constants::NUM_HASH_BITS,
  frontend::{
    num::AllocatedNum, AllocatedBit, Assignment, Boolean, ConstraintSystem, SynthesisError,
  },
  gadgets::{
    ecc::AllocatedNonnativePoint,
    utils::{alloc_num_equals, alloc_zero, conditionally_select_vec, le_bits_to_num},
  },
  neutron::{nifs::NIFS, relation::FoldedInstance},
  r1cs::R1CSInstance,
  traits::{
    circuit::StepCircuit, commitment::CommitmentTrait, Engine, RO2ConstantsCircuit, ROCircuitTrait,
  },
  Commitment,
};
#[cfg(feature = "lookup-fold")]
use crate::neutron::{
  circuit::lookup::{AllocatedLookupNIFSMultiTable, AllocatedLookupPayloadPublicMultiTable},
  relation::{LookupPayloadPublic, LookupPayloadPublicMultiTable},
};
use ff::Field;
use serde::{Deserialize, Serialize};

#[cfg(feature = "lookup-fold")]
pub mod lookup;
pub mod nifs;
pub mod r1cs;
pub mod relation;
pub mod univariate;

use nifs::AllocatedNIFS;
use r1cs::AllocatedNonnativeR1CSInstance;
use relation::AllocatedFoldedInstance;

/// A type that holds the non-deterministic inputs for the augmented circuit
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronAugmentedCircuitInputs<E: Engine> {
  pp_digest: E::Scalar,
  i: E::Scalar,
  z0: Vec<E::Scalar>,
  zi: Option<Vec<E::Scalar>>,
  U: Option<FoldedInstance<E>>,
  ri: Option<E::Scalar>,
  r_next: E::Scalar,
  u: Option<R1CSInstance<E>>,
  nifs: Option<NIFS<E>>,
  comm_W_fold: Option<Commitment<E>>,
  comm_E_fold: Option<Commitment<E>>,

  /// Lookup-side public payload (`comm_L`, `comm_ts`) — Stage G §G.4.
  /// `None` for non-lookup steps and at outer base.
  #[cfg(feature = "lookup-fold")]
  pub(crate) lookup_payload_pub: Option<LookupPayloadPublic<E>>,
  /// Folded `comm_L` hint (untrusted, like `comm_W_fold`). The augmented
  /// circuit consumes this to absorb into the next-step Nova hash.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_L_fold: Option<Commitment<E>>,
  /// Folded `comm_ts` hint.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_ts_fold: Option<Commitment<E>>,
  /// Folded `comm_inv_w` hint.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_w_fold: Option<Commitment<E>>,
  /// Folded `comm_inv_t` hint.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_t_fold: Option<Commitment<E>>,

  /// GH-#5 M.GH5.3 / pin §3.1: per-table public bundles for the multi-
  /// table lookup-fold verifier. Length pinned by the
  /// `LookupShape::multi_column_tables.len()` (= k = 2 production per
  /// ADR-0021), in `table_id`-ascending order. `None` for non-lookup
  /// steps and at outer base. Populated by inumbra-side `prove_step` when
  /// the multi-table lookup-fold path is active (M.GH5.4+); `None` for
  /// vendor-internal tests using `TrivialCircuit` / `CubicCircuit` and
  /// for the non-lookup-fold-active build path.
  #[cfg(feature = "lookup-fold")]
  pub(crate) public_bundles_multi_table: Option<Vec<LookupPayloadPublicMultiTable<E>>>,

  /// GH-#5 M.GH5.7 / Pin Corrigendum #10 Q2 ruling (Halpert 2026-05-10):
  /// test-only corruption knob that perturbs the **U.T_lookup_per_table**
  /// allocation at index `idx` by additive offset, AFTER the honest
  /// witness-closure value is read from `inputs.U.t_lookup()` and BEFORE
  /// `U.absorb_in_ro` consumes the allocation in the Phase-1 hash check
  /// (`circuit/mod.rs:462`).
  ///
  /// Wired in `alloc_witness` (the U allocation site under
  /// `lookup_fold_k > 0`); perturbing U.T_lookup_per_table directly causes
  /// the Phase-1 hash check to recompute a hash that diverges from the
  /// honest `u.X[0]`, firing the constraint
  /// `"check consistency of u.X[0] with H(params, U, i, z0, zi)"`
  /// (`circuit/mod.rs:467-471`). Per `TestConstraintSystem::which_is_unsatisfied`
  /// declaration-order semantics (`frontend/util_cs/test_cs.rs:99-113`,
  /// Corrigendum #10 Q3 ratification), Phase-1's `alloc_num_equals`
  /// constraint fires BEFORE Phase-2's `verify_with_multi_table_lookup`-
  /// internal (C)-binding constraints — so the FIRST failing constraint
  /// path twin-substrings on Phase-1, not on the (C)-binding.
  ///
  /// `None` (default) preserves honest synthesis; `Some((idx, offset))`
  /// is the test (i) (W1)-direct corruption pattern of pin §3.5
  /// alternative-corruption + §5.3 #3.
  ///
  /// Gated behind `cfg(any(test, feature = "test-debug-knobs"))`:
  /// `cfg(test)` alone is INSUFFICIENT — vendor's `cfg(test)` does NOT
  /// activate when consumed as a dependency from inumbra-spend-harness
  /// (out-of-crate); the `test-debug-knobs` feature flag (declared in
  /// `vendor/nova/Cargo.toml`) is the harness-from-vendor exposure
  /// mechanism. Audit-firm packet documents the feature flag's
  /// test-only contract per Corrigendum #10 Q2 ruling.
  #[cfg(all(feature = "lookup-fold", any(test, feature = "test-debug-knobs")))]
  pub corrupt_t_lookup_at_index: Option<(usize, E::Scalar)>,
}

impl<E: Engine> NeutronAugmentedCircuitInputs<E> {
  /// Create new inputs/witness for the verification circuit
  pub fn new(
    pp_digest: E::Scalar,
    i: E::Scalar,
    z0: Vec<E::Scalar>,
    zi: Option<Vec<E::Scalar>>,
    U: Option<FoldedInstance<E>>,
    ri: Option<E::Scalar>,
    r_next: E::Scalar,
    u: Option<R1CSInstance<E>>,
    nifs: Option<NIFS<E>>,
    comm_W_fold: Option<Commitment<E>>,
    comm_E_fold: Option<Commitment<E>>,
  ) -> Self {
    Self {
      pp_digest,
      i,
      z0,
      zi,
      U,
      ri,
      r_next,
      u,
      nifs,
      comm_W_fold,
      comm_E_fold,
      #[cfg(feature = "lookup-fold")]
      lookup_payload_pub: None,
      #[cfg(feature = "lookup-fold")]
      comm_L_fold: None,
      #[cfg(feature = "lookup-fold")]
      comm_ts_fold: None,
      #[cfg(feature = "lookup-fold")]
      comm_inv_w_fold: None,
      #[cfg(feature = "lookup-fold")]
      comm_inv_t_fold: None,
      #[cfg(feature = "lookup-fold")]
      public_bundles_multi_table: None,
      // GH-#5 M.GH5.7 / Pin Corrigendum #10 Q2: default-`None` preserves
      // honest synthesis. The harness-side `with_corrupt_t_lookup_at_index`
      // builder (under cfg(any(test, feature = "test-debug-knobs"))) is
      // the test-only opt-in for the (W1)-direct Phase-1 corruption test.
      #[cfg(all(feature = "lookup-fold", any(test, feature = "test-debug-knobs")))]
      corrupt_t_lookup_at_index: None,
    }
  }

  /// GH-#5 M.GH5.7 / Pin Corrigendum #10 Q2 ruling (Halpert 2026-05-10):
  /// test-only builder for the (W1)-direct corruption pattern of pin
  /// §3.5 alternative-corruption + §5.3 #3. Sets the
  /// `corrupt_t_lookup_at_index` field on this `NeutronAugmentedCircuitInputs`
  /// so `alloc_witness` perturbs the in-circuit `U.T_lookup_per_table[idx]`
  /// allocation by `offset` after reading the honest witness-closure
  /// value, causing the Phase-1 hash check at `circuit/mod.rs:467-471` to
  /// reject (the recomputed hash diverges from `u.X[0]`).
  ///
  /// Returns `Self` (builder pattern, signature-stable with `with_lookup`
  /// / `with_multi_table_bundles`).
  ///
  /// Gated behind `cfg(any(test, feature = "test-debug-knobs"))` per Q2
  /// ruling: vendor `cfg(test)` does NOT activate from harness-crate
  /// consumers, but the feature flag does. The audit-firm packet records
  /// this method's test-only role.
  #[cfg(all(feature = "lookup-fold", any(test, feature = "test-debug-knobs")))]
  #[allow(dead_code)] // consumed only by inumbra-spend-harness M.GH5.7 negative tests
  pub fn with_corrupt_t_lookup_at_index(mut self, idx: usize, offset: E::Scalar) -> Self {
    self.corrupt_t_lookup_at_index = Some((idx, offset));
    self
  }

  /// GH-#5 M.GH5.3: attach multi-table lookup-fold public bundles to an
  /// existing `NeutronAugmentedCircuitInputs`. Builder-style so the legacy
  /// `new` constructor stays signature-stable.
  ///
  /// `bundles` must have length equal to the structurally-pinned table
  /// count (k=2 production per ADR-0021), in `table_id`-ascending order.
  /// At outer base / non-lookup steps, leave the field as `None` (the
  /// default from `new`).
  #[cfg(feature = "lookup-fold")]
  #[allow(dead_code)] // until GH-#7 wires inumbra-side prove_step through this builder; Pin Corrigendum #9 routing-carry-forward
  pub fn with_multi_table_bundles(
    mut self,
    bundles: Option<Vec<LookupPayloadPublicMultiTable<E>>>,
  ) -> Self {
    self.public_bundles_multi_table = bundles;
    self
  }

  /// Attach lookup-side public payload + folded commitment hints to an
  /// existing `NeutronAugmentedCircuitInputs` (Stage G §G.4).
  ///
  /// Builder-style so the legacy `new` constructor stays signature-stable.
  #[cfg(feature = "lookup-fold")]
  #[allow(clippy::too_many_arguments)]
  #[allow(dead_code)]
  pub fn with_lookup(
    mut self,
    lookup_payload_pub: Option<LookupPayloadPublic<E>>,
    comm_L_fold: Option<Commitment<E>>,
    comm_ts_fold: Option<Commitment<E>>,
    comm_inv_w_fold: Option<Commitment<E>>,
    comm_inv_t_fold: Option<Commitment<E>>,
  ) -> Self {
    self.lookup_payload_pub = lookup_payload_pub;
    self.comm_L_fold = comm_L_fold;
    self.comm_ts_fold = comm_ts_fold;
    self.comm_inv_w_fold = comm_inv_w_fold;
    self.comm_inv_t_fold = comm_inv_t_fold;
    self
  }
}

/// The augmented circuit F' in Neutron that includes a step circuit F
/// and the circuit for the verifier in Neutron's non-interactive folding scheme
///
/// GH-#5 M.GH5.3: under `lookup-fold`, the augmented circuit carries a
/// shape-registry attachment (`shape_registry`, `lookup_fold_k`,
/// `index_n_bits`) that wires the multi-table lookup-fold verifier path
/// at fold-depth ≥ 1. Set via [`Self::with_lookup_fold`]. Default state
/// (`lookup_fold_k == 0`) preserves the upstream-tracking shape — the
/// non-base-case path uses `nifs.verify` and the base case uses
/// `default(cs, num_io)`. The vendor-internal tests using `TrivialCircuit`
/// / `CubicCircuit` (`test_pp_digest`, `test_ivc_*`) operate in this
/// default state and produce R1CS shapes byte-equivalent to pre-M.GH5.3
/// (i.e. `pp_digest` baselines for those tests do NOT refresh).
///
/// At inumbra-side construction (M.GH5.4+), `with_lookup_fold(k,
/// shape_registry, index_n_bits)` is called with the per-position
/// `pp_digest` slice (length 16 production per ADR-0021) and `k=2`
/// (uniform multi-column-table count per Auditor Obligation 10).
pub struct NeutronAugmentedCircuit<'a, E: Engine, SC: StepCircuit<E::Scalar>> {
  ro_consts: RO2ConstantsCircuit<E>,
  inputs: Option<NeutronAugmentedCircuitInputs<E>>,
  step_circuit: &'a SC, // The function that is applied for each step

  /// GH-#5 M.GH5.3 / pin §3.1: per-position shape registry of `pp_digest`
  /// values (one per chunk position), in chunk-position-canonical order
  /// (NOT `table_id` order). Empty slice ⇒ no lookup-fold path; the
  /// augmented circuit falls back to `nifs.verify` in `synthesize_non_base_case`
  /// and `default(cs, num_io)` in `synthesize_base_case`. Populated by
  /// inumbra-side public-params at M.GH5.4 from GH-#3's per-position
  /// `Structure<E>` extraction.
  #[cfg(feature = "lookup-fold")]
  shape_registry: &'a [E::Scalar],

  /// GH-#5 M.GH5.3 / pin §3.1: structurally-pinned per-table count
  /// (`LookupShape::multi_column_tables.len()`). Production: k=2 per
  /// ADR-0021, uniform across all 16 chunked-Strauss-Shamir positions
  /// (Auditor Obligation 10). Zero ⇒ no lookup-fold path.
  #[cfg(feature = "lookup-fold")]
  lookup_fold_k: usize,

  /// GH-#5 M.GH5.3 / pin §3.1: bit-width for the `chunk_index_in_z` range-
  /// check inside the M.7 shape-registry assertion. Must be
  /// ≥ ceil(log2(shape_registry.len())). Production: 5 (≤ 30 positions);
  /// M.GH5.0 STAGE 0 fixtures use 4 (16 positions).
  #[cfg(feature = "lookup-fold")]
  index_n_bits: usize,
}

impl<'a, E: Engine, SC: StepCircuit<E::Scalar>> NeutronAugmentedCircuit<'a, E, SC> {
  /// Create a new verification circuit for the input relaxed r1cs instances.
  ///
  /// This constructor produces a circuit in the non-lookup-fold default
  /// state (under `lookup-fold` build, `lookup_fold_k == 0` and
  /// `shape_registry == &[]`). Use [`Self::with_lookup_fold`] to attach
  /// the per-position shape registry for the multi-table lookup-fold
  /// verifier path (M.GH5.3+).
  pub const fn new(
    inputs: Option<NeutronAugmentedCircuitInputs<E>>,
    step_circuit: &'a SC,
    ro_consts: RO2ConstantsCircuit<E>,
  ) -> Self {
    Self {
      inputs,
      step_circuit,
      ro_consts,
      #[cfg(feature = "lookup-fold")]
      shape_registry: &[],
      #[cfg(feature = "lookup-fold")]
      lookup_fold_k: 0,
      #[cfg(feature = "lookup-fold")]
      index_n_bits: 0,
    }
  }

  /// GH-#5 M.GH5.3 / pin §3.1: attach the multi-table lookup-fold
  /// configuration to a `NeutronAugmentedCircuit`. Builder-style so the
  /// legacy `new` constructor stays signature-stable.
  ///
  /// Arguments:
  /// - `lookup_fold_k`: structurally-pinned per-table count
  ///   (`LookupShape::multi_column_tables.len()`). Production: k=2 per
  ///   ADR-0021. `0` is treated as "no lookup-fold path" and the augmented
  ///   circuit falls back to `nifs.verify` (vendor upstream-tracking path).
  /// - `shape_registry`: per-position `pp_digest` slice. Length pinned by
  ///   the inumbra-side public-params (length 16 production; ≤ 30 with
  ///   chunk-band shapes).
  /// - `index_n_bits`: bit-width for the M.7 shape-registry assertion's
  ///   range-check on `chunk_index_in_z`. Must be ≥ ceil(log2(shape_registry.len()))
  ///   and ≤ 31.
  ///
  /// Sanity checks (synthesis-time): when `lookup_fold_k > 0`,
  /// `shape_registry` must be non-empty and `index_n_bits` must be ≥ 1.
  /// These are enforced in `synthesize_non_base_case` via the
  /// `verify_with_multi_table_lookup` invocation; the constructor itself
  /// does not validate (no `Result` shape).
  #[cfg(feature = "lookup-fold")]
  pub fn with_lookup_fold(
    mut self,
    lookup_fold_k: usize,
    shape_registry: &'a [E::Scalar],
    index_n_bits: usize,
  ) -> Self {
    self.lookup_fold_k = lookup_fold_k;
    self.shape_registry = shape_registry;
    self.index_n_bits = index_n_bits;
    self
  }


  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<E::Scalar>,
      AllocatedNum<E::Scalar>,
      Vec<AllocatedNum<E::Scalar>>,
      Vec<AllocatedNum<E::Scalar>>,
      AllocatedFoldedInstance<E>,
      AllocatedNum<E::Scalar>,
      AllocatedNum<E::Scalar>,
      AllocatedNonnativeR1CSInstance<E>,
      AllocatedNIFS<E>,
      AllocatedNonnativePoint<E>,
      AllocatedNonnativePoint<E>,
    ),
    SynthesisError,
  > {
    // Allocate the pp_digest
    let pp_digest = AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || {
      Ok(self.inputs.get()?.pp_digest)
    })?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate z0
    let z_0 = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("z0_{i}")), || {
          Ok(self.inputs.get()?.z0[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero = vec![E::Scalar::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.inputs.get()?.zi.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?;

    // Allocate the running instance.
    //
    // GH-#5 M.GH5.3: under `lookup-fold`, pass `self.lookup_fold_k` as the
    // shape-derivation hint so the `inst == None` path (shape building
    // via `inputs == None`) allocates `T_lookup_per_table = Some(vec![alloc; k])`
    // matching the non-base-case shape produced by `verify_with_multi_table_lookup`.
    // At `lookup_fold_k == 0` the hint is no-op (legacy `T_lookup_per_table = None`).
    // GH-#5 M.GH5.7 / Pin Corrigendum #10 Q2: the `mut` binding on `U`
    // is required ONLY when the `test-debug-knobs` (or `test`) cfg path
    // is active — that path mutates `U.T_lookup_per_table[idx]` in place
    // via the corruption knob block below. Under the production build
    // (`lookup-fold` alone, no `test-debug-knobs`), `U` is read-only
    // and `let mut` would trip `#[deny(unused_mut)]` at the workspace
    // lint level (`vendor/nova/src/lib.rs:4`). Splitting the cfg gates
    // keeps the production binding immutable.
    #[cfg(all(feature = "lookup-fold", any(test, feature = "test-debug-knobs")))]
    let mut U: AllocatedFoldedInstance<E> = AllocatedFoldedInstance::alloc_with_k_hint(
      cs.namespace(|| "Allocate U"),
      self.inputs.as_ref().and_then(|inputs| inputs.U.as_ref()),
      self.lookup_fold_k,
    )?;
    #[cfg(all(feature = "lookup-fold", not(any(test, feature = "test-debug-knobs"))))]
    let U: AllocatedFoldedInstance<E> = AllocatedFoldedInstance::alloc_with_k_hint(
      cs.namespace(|| "Allocate U"),
      self.inputs.as_ref().and_then(|inputs| inputs.U.as_ref()),
      self.lookup_fold_k,
    )?;
    #[cfg(not(feature = "lookup-fold"))]
    let U: AllocatedFoldedInstance<E> = AllocatedFoldedInstance::alloc(
      cs.namespace(|| "Allocate U"),
      self.inputs.as_ref().and_then(|inputs| inputs.U.as_ref()),
    )?;

    // GH-#5 M.GH5.7 / Pin Corrigendum #10 Q2 ruling (Halpert 2026-05-10):
    // (W1)-direct corruption knob. When the knob is set, replace
    // `U.T_lookup_per_table[idx]` with a fresh `AllocatedNum` whose witness
    // value is the honest scalar plus `offset`. The fresh aux variable has
    // no constraint binding it, so the witness assignment is honored at
    // synthesis. Downstream:
    //   - Phase-1 hash check (`circuit/mod.rs:462`): `U.absorb_in_ro`
    //     absorbs the corrupt allocation. The recomputed hash diverges
    //     from `u.X[0]` (which was honestly produced at the prior step
    //     with the un-corrupted `Unew_step_prev.T_lookup`). The
    //     `alloc_num_equals` constraint at line 467 (path
    //     `"check consistency of u.X[0] with H(params, U, i, z0, zi)"`)
    //     fails.
    //   - Phase-2 (`verify_with_multi_table_lookup`): reads
    //     `U.T_lookup_per_table` as `t_lookup_running_per_table` per
    //     `synthesize_non_base_case_lookup_fold` (line 630-641); the
    //     (C)-binding may also fail downstream, BUT
    //     `which_is_unsatisfied` (per Corrigendum #10 Q3 ratification +
    //     `frontend/util_cs/test_cs.rs:99-113`) returns the FIRST failing
    //     constraint in declaration order — Phase-1 fires before Phase-2.
    //
    // Per pin §3.5 alternative-corruption + Corrigendum #10 Q3, this is
    // the test (i) `gh5_augmented_circuit_corrupt_running_T_lookup_breaks_phase1_hash_rejects`
    // wire-up (§5.3 #3 / (W1)-direct discharge).
    #[cfg(all(feature = "lookup-fold", any(test, feature = "test-debug-knobs")))]
    if let Some((idx, offset)) = self.inputs.as_ref().and_then(|i| i.corrupt_t_lookup_at_index) {
      let t_vec = U.T_lookup_per_table.as_mut().ok_or_else(|| {
        SynthesisError::Unsatisfiable(
          "alloc_witness: corrupt_t_lookup_at_index is Some but \
           U.T_lookup_per_table is None — knob requires lookup_fold_k > 0 \
           and U with non-empty T_lookup (Pin Corrigendum #10 Q2)"
            .to_string(),
        )
      })?;
      if idx >= t_vec.len() {
        return Err(SynthesisError::Unsatisfiable(format!(
          "alloc_witness: corrupt_t_lookup_at_index idx={idx} out of bounds \
           for U.T_lookup_per_table.len()={} (Pin Corrigendum #10 Q2)",
          t_vec.len(),
        )));
      }
      let honest_value = t_vec[idx]
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let corrupt_value = honest_value + offset;
      let corrupt_alloc = AllocatedNum::alloc(
        cs.namespace(|| format!("corrupt T_lookup_per_table[{idx}] (M.GH5.7 knob)")),
        || Ok(corrupt_value),
      )?;
      t_vec[idx] = corrupt_alloc;
    }

    // Allocate ri
    let r_i = AllocatedNum::alloc(cs.namespace(|| "ri"), || {
      Ok(self.inputs.get()?.ri.unwrap_or(E::Scalar::ZERO))
    })?;

    // Allocate r_i+1
    let r_next = AllocatedNum::alloc(cs.namespace(|| "r_i+1"), || Ok(self.inputs.get()?.r_next))?;

    // Allocate the instance to be folded in
    let u = AllocatedNonnativeR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self.inputs.as_ref().and_then(|inputs| inputs.u.as_ref()),
    )?;

    // Allocate nifs
    let nifs = AllocatedNIFS::alloc(
      cs.namespace(|| "allocate nifs"),
      self.inputs.as_ref().and_then(|inputs| inputs.nifs.as_ref()),
      5, // TODO: take this as input
    )?;

    // Allocated comm_W_fold
    let comm_W_fold = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_W"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.comm_W_fold.as_ref().map(|c| c.to_coordinates())),
    )?;

    // Allocate comm_E_fold
    let comm_E_fold = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_E_fold"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.comm_E_fold.as_ref().map(|c| c.to_coordinates())),
    )?;

    Ok((
      pp_digest,
      i,
      z_0,
      z_i,
      U,
      r_i,
      r_next,
      u,
      nifs,
      comm_W_fold,
      comm_E_fold,
    ))
  }

  fn synthesize_base_case<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedFoldedInstance<E>, SynthesisError> {
    // In the base case, we simply return the default running instance.
    // Pin §1.4 Corrigendum #3: the augmented-circuit's R1CS shape has
    // `num_io == 1` (single `hash.inputize` site below), so the default
    // running U1 carries a length-1 X Vec.
    //
    // GH-#5 M.GH5.3 / pin §3.1 + §5.1: under `lookup-fold` with an
    // attached shape registry (`self.lookup_fold_k > 0`), allocate via
    // `default_with_lookup_k(cs, num_io, k)` so the base-case instance
    // carries a length-k `T_lookup_per_table` of `T_lookup_zero`-shared
    // zero-bound variables (Corrigendum #6 Path Y extension). This is
    // the shape-match invariant for `conditionally_select` (the base
    // case's `Some(vec; k)` must match the non-base-case's
    // `verify_with_multi_table_lookup` output's `Some(vec; k)`).
    //
    // At `lookup_fold_k == 0` (default state, vendor-internal tests with
    // TrivialCircuit / CubicCircuit), the legacy `default(cs, 1)` path is
    // preserved so the R1CS shape (and `pp_digest` baseline) is unchanged
    // from pre-M.GH5.3.
    #[cfg(feature = "lookup-fold")]
    {
      if self.lookup_fold_k > 0 {
        return AllocatedFoldedInstance::default_with_lookup_k(
          cs.namespace(|| "Allocate U_default (lookup-fold)"),
          1,
          self.lookup_fold_k,
        );
      }
    }
    AllocatedFoldedInstance::default(cs.namespace(|| "Allocate U_default"), 1)
  }

  /// Synthesizes non base case and returns the new relaxed `FoldedInstance`
  /// And a boolean indicating if all checks pass
  fn synthesize_non_base_case<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    z_0: &[AllocatedNum<E::Scalar>],
    z_i: &[AllocatedNum<E::Scalar>],
    U: &AllocatedFoldedInstance<E>,
    r_i: &AllocatedNum<E::Scalar>,
    u: &AllocatedNonnativeR1CSInstance<E>,
    nifs: &AllocatedNIFS<E>,
    comm_W_fold: &AllocatedNonnativePoint<E>,
    comm_E_fold: &AllocatedNonnativePoint<E>,
  ) -> Result<(AllocatedFoldedInstance<E>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, U, i, z0, zi)
    let mut ro = E::RO2Circuit::new(self.ro_consts.clone());
    ro.absorb(pp_digest);
    ro.absorb(i);
    for e in z_0 {
      ro.absorb(e);
    }
    for e in z_i {
      ro.absorb(e);
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;
    ro.absorb(r_i);

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash"), NUM_HASH_BITS, false)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), &hash_bits)?;
    let check_pass = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      &u.X,
      &hash,
    )?;

    // GH-#5 M.GH5.3 / pin §5.1: under `lookup-fold` with an attached
    // shape registry, route through `verify_with_multi_table_lookup`.
    // Otherwise, fall back to the legacy `nifs.verify` (vendor upstream-
    // tracking shape per ADR-0023). The cfg-gate is `lookup-fold`; the
    // runtime-gate is `self.lookup_fold_k > 0` so that vendor-internal
    // tests using `TrivialCircuit` / `CubicCircuit` (no lookup data)
    // continue to typecheck and run with the upstream `nifs.verify` path
    // even on a `--features lookup-fold` build.
    #[cfg(feature = "lookup-fold")]
    if self.lookup_fold_k > 0 {
      return self.synthesize_non_base_case_lookup_fold(
        cs.namespace(|| "synthesize non base case lookup-fold"),
        pp_digest,
        z_i,
        U,
        u,
        nifs,
        comm_W_fold,
        comm_E_fold,
        check_pass,
      );
    }

    // Run NIFS Verifier (legacy non-lookup-fold-active path)
    let U_fold = nifs.verify(
      cs.namespace(|| "compute fold of U and u"),
      pp_digest,
      U,
      u,
      comm_W_fold,
      comm_E_fold,
      self.ro_consts.clone(),
    )?;

    Ok((U_fold, check_pass))
  }

  /// GH-#5 M.GH5.3 / pin §3.1 + §5.1: lookup-fold-active branch of
  /// `synthesize_non_base_case`.
  ///
  /// Allocates the per-position `chunk_index_in_z` from `U.X[0]` (per
  /// ADR-0021), the per-position shape registry as a slice of
  /// `AllocatedNum`s, the multi-table NIFS message, and the per-table
  /// public bundles. Then invokes `verify_with_multi_table_lookup`,
  /// which internally fires the M.7 shape-registry assertion BEFORE
  /// `pp_digest.absorb(ro)` (per pin §3.4 / vendor `nifs.rs:689-695`).
  /// Finally re-binds the post-fold `T_lookup_per_table` from the
  /// verifier's output via [`AllocatedFoldedInstance::from_lookup_fold_output`].
  ///
  /// Soundness anchors:
  /// - `chunk_index_in_z = z_i[F_arity - 1]`: GH-#7 M.GH7.2 routing per
  ///   pin §3.2 + Corrigendum #4. The augmented-circuit's running-instance
  ///   chunk-position index is carried in the IVC step input slot `z` at
  ///   the last position (`z_i[F_arity - 1]`), threaded per the *extension*
  ///   of ADR-0021's NIVC dispatcher z-carry pattern (the specific
  ///   reservation `z[F_arity-1] = chunk_index` is a NEW authoring decision
  ///   at GH-#7 covered by pin ratification, not directly ratified by
  ///   ADR-0021 — see pin Corrigendum #4 lines 1198-1210). Prior interim
  ///   (Corrigendum #9 era) sourced `chunk_index_in_z` from `U.X[0]`; the
  ///   M.GH7.2 reroute moves the read-site to the `z` slot to align with
  ///   the NIVC dispatcher model. The single-IO invariant on `U.X` (pin
  ///   §1.4 Corrigendum #3, `num_io == 1`) is no longer load-bearing for
  ///   the read-site; the defensive `U.X.is_empty()` check below is
  ///   retained as a structural assertion of the single-IO invariant.
  /// - `t_lookup_running_per_table = U.T_lookup_per_table.as_deref()`:
  ///   per pin §3.3, the augmented circuit consumes the running U's
  ///   per-table running scalars directly. The Phase-1 hash check above
  ///   (via `U.absorb_in_ro`) binds these scalars to `u.X[0]`, closing
  ///   the (W1) cross-step substitution attack.
  /// - M.7 shape-registry assertion fires INTERNAL to
  ///   `verify_with_multi_table_lookup` BEFORE `pp_digest.absorb` per pin
  ///   §2.5 / §3.4 (vendor `nifs.rs:689-695` + line 698). The augmented-
  ///   circuit caller does NOT fire it separately. The M.GH7.2 read-site
  ///   reroute preserves this architectural FS-transcript point (the M.7
  ///   annotation still emits at the same internal site, before the
  ///   `pp_digest.absorb`-then-`squeeze` sequence; the only change is the
  ///   *source* of the `chunk_index_in_z` AllocatedNum — now `z_i[F_arity-1]`
  ///   instead of `U.X[0]`).
  #[cfg(feature = "lookup-fold")]
  #[allow(clippy::too_many_arguments)]
  fn synthesize_non_base_case_lookup_fold<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    z_i: &[AllocatedNum<E::Scalar>],
    U: &AllocatedFoldedInstance<E>,
    u: &AllocatedNonnativeR1CSInstance<E>,
    nifs: &AllocatedNIFS<E>,
    comm_W_fold: &AllocatedNonnativePoint<E>,
    comm_E_fold: &AllocatedNonnativePoint<E>,
    check_pass: AllocatedBit,
  ) -> Result<(AllocatedFoldedInstance<E>, AllocatedBit), SynthesisError> {
    // Sanity: lookup-fold-active path requires non-empty shape registry
    // and index_n_bits ≥ 1.
    if self.shape_registry.is_empty() {
      return Err(SynthesisError::Unsatisfiable(
        "synthesize_non_base_case_lookup_fold: shape_registry is empty under \
         lookup_fold_k > 0; attach via NeutronAugmentedCircuit::with_lookup_fold(...)"
          .to_string(),
      ));
    }
    if self.index_n_bits == 0 {
      return Err(SynthesisError::Unsatisfiable(
        "synthesize_non_base_case_lookup_fold: index_n_bits == 0 under \
         lookup_fold_k > 0; attach via NeutronAugmentedCircuit::with_lookup_fold(...)"
          .to_string(),
      ));
    }

    // GH-#7 M.GH7.2 / pin §3.2 + Corrigendum #4: route `chunk_index_in_z`
    // via `z_i[F_arity - 1]` per the *extension* of ADR-0021's NIVC
    // dispatcher z-carry pattern. The augmented-circuit's structural
    // `num_io == 1` invariant (pin §1.4 Corrigendum #3) is no longer
    // load-bearing for this read-site, but the defensive check below
    // is retained as a structural assertion of the single-IO invariant
    // (an `U.X` of length 0 would indicate a wire-up bug in `alloc` or
    // `default*`; the `synthesize_non_base_case` Phase-1 hash check
    // also assumes `u.X` is single-IO).
    if U.X.is_empty() {
      return Err(SynthesisError::Unsatisfiable(
        "synthesize_non_base_case_lookup_fold: U.X is empty (num_io == 0); \
         augmented-circuit invariant requires num_io == 1 per pin §1.4 \
         Corrigendum #3"
          .to_string(),
      ));
    }
    if z_i.is_empty() {
      return Err(SynthesisError::Unsatisfiable(
        "synthesize_non_base_case_lookup_fold: z_i is empty (F_arity == 0); \
         GH-#7 M.GH7.2 routing requires F_arity ≥ 1 with z_i[F_arity - 1] \
         reserved for chunk_index per pin §3.2 + Corrigendum #4"
          .to_string(),
      ));
    }
    let chunk_index_in_z = &z_i[z_i.len() - 1];

    // Allocate the per-position shape registry as in-circuit AllocatedNums.
    // The `assert_pp_digest_matches_registry` consumer at `nifs.rs:689-695`
    // expects `&[AllocatedNum]`. Each entry is a witness-allocated scalar
    // tied to the lifetime-static slice `self.shape_registry`.
    let shape_registry_alloc: Vec<AllocatedNum<E::Scalar>> = self
      .shape_registry
      .iter()
      .enumerate()
      .map(|(idx, scalar)| {
        AllocatedNum::alloc(
          cs.namespace(|| format!("shape_registry[{idx}]")),
          || Ok(*scalar),
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Allocate the multi-table NIFS message (lookup-side per-table fields).
    let lookups = AllocatedLookupNIFSMultiTable::<E>::alloc(
      cs.namespace(|| "allocate AllocatedLookupNIFSMultiTable"),
      self.inputs.as_ref().and_then(|inputs| inputs.nifs.as_ref()),
      self.lookup_fold_k,
    )?;

    // Allocate the per-table public bundles.
    //
    // For M.GH5.3 the per-table value-column count `num_value_columns` is
    // sourced from the supplied bundle when present; `unwrap_or(0)` — zero-column
    // default per Corrigendum #27 (absent bundle ⇒ no value commitments to allocate,
    // not a synthetic 1-column shape that poisons R1CS-sat at the verify step).
    // M.GH5.4 will supersede with `LookupShape`-sourced per-table column counts.
    let public_bundles_native: Option<&Vec<LookupPayloadPublicMultiTable<E>>> = self
      .inputs
      .as_ref()
      .and_then(|inputs| inputs.public_bundles_multi_table.as_ref());
    let public_bundles_alloc: Vec<AllocatedLookupPayloadPublicMultiTable<E>> = (0..self
      .lookup_fold_k)
      .map(|j| {
        let bundle_j = public_bundles_native.and_then(|bundles| bundles.get(j));
        let num_value_columns = bundle_j.map(|b| b.comm_values.len()).unwrap_or(0);
        AllocatedLookupPayloadPublicMultiTable::alloc(
          cs.namespace(|| format!("allocate public_bundle[{j}]")),
          bundle_j,
          num_value_columns,
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    // t_lookup_running_per_table := U.T_lookup_per_table per pin §3.3.
    // Under `lookup_fold_k > 0`, the U allocation in `alloc_witness`
    // produces `Some(vec; k)` (either from `inst.t_lookup()` at fold-depth
    // ≥ 1, or from the k-hint at shape-derivation). An empty slice
    // indicates a wire-up bug.
    let t_lookup_running_per_table_owned: Vec<AllocatedNum<E::Scalar>> = U
      .T_lookup_per_table
      .as_ref()
      .ok_or_else(|| {
        SynthesisError::Unsatisfiable(
          "synthesize_non_base_case_lookup_fold: U.T_lookup_per_table is None \
           under lookup_fold_k > 0; alloc_witness should populate via k-hint \
           per M.GH5.3 / pin §3.2"
            .to_string(),
        )
      })?
      .clone();

    // Invoke the multi-table verifier. This internally:
    //   - Asserts t_lookup_running_per_table.len() == k (line 666-680).
    //   - Fires the M.7 shape-registry assertion BEFORE pp_digest.absorb
    //     (line 689-695, with pp_digest.absorb at line 698).
    //   - Computes the FS transcript, the (C)-bindings, and the per-table
    //     post-fold T_lookup_out_per_table.
    let lookup_output = nifs.verify_with_multi_table_lookup(
      cs.namespace(|| "verify_with_multi_table_lookup"),
      pp_digest,
      chunk_index_in_z,
      &shape_registry_alloc,
      self.index_n_bits,
      U,
      u,
      &lookups,
      &public_bundles_alloc,
      &t_lookup_running_per_table_owned,
      comm_W_fold,
      comm_E_fold,
      self.ro_consts.clone(),
    )?;

    // Re-bind the post-fold T_lookup_per_table from the verifier's output
    // per pin §3.2: the verifier's `U_fold` came from `U1.fold(...)` which
    // propagates `T_lookup_per_table` from U1 unchanged; we override with
    // `T_lookup_out_per_table` so the new running U carries the correct
    // post-fold per-table running scalars.
    //
    // GH-#7 design pin Corrigendum #19 (M.GH7.5.0b path α.2): ALSO override
    // the M.GH7.5.0a per-table commitment passthrough with the post-fold
    // `comm_L_fold_per_table` / `comm_ts_fold_per_table` hint Vecs sourced
    // off-circuit from the prover's per-step NIFS message (the
    // `NIFS::prove_with_multi_table_lookup_inner`'s `U.comm_L` / `U.comm_ts`
    // post-fold extraction; off-circuit fold body at
    // `vendor/nova/src/neutron/relation.rs:862-884`). This closes the IVC↔
    // envelope binding by making the in-circuit `Unew.comm_L_per_table[j]`
    // (absorbed at the final-step hash via M.GH7.5.0a's `absorb_in_ro`
    // extension) byte-equal to the off-circuit `r_U.comm_L[j]` (absorbed
    // at off-circuit `RecursiveSNARK::verify` hash reconstruction via
    // M.GH7.5.0a's `absorb_in_ro2` extension). Soundness by parallel
    // reasoning to the existing `comm_W_fold` / `comm_E_fold`
    // untrusted-hint discipline at `circuit/nifs.rs:46-55` / `:649-664`.
    let U_fold = AllocatedFoldedInstance::from_lookup_fold_output(
      lookup_output.U_fold,
      lookup_output.T_lookup_out_per_table,
      lookup_output.comm_L_fold_per_table,
      lookup_output.comm_ts_fold_per_table,
    );

    Ok((U_fold, check_pass))
  }
}

impl<E: Engine, SC: StepCircuit<E::Scalar>> NeutronAugmentedCircuit<'_, E, SC> {
  /// synthesize circuit giving constraint system
  pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    let arity = self.step_circuit.arity();

    // Allocate all witnesses
    let (pp_digest, i, z_0, z_i, U, r_i, r_next, u, nifs, comm_W_fold, comm_E_fold) =
      self.alloc_witness(cs.namespace(|| "allocate the circuit witness"), arity)?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // synthesize base case
    let Unew_base = self.synthesize_base_case(cs.namespace(|| "synthesize base case"))?;

    // Synthesize the circuit for the non-base case and get the new running
    // instance along with a boolean indicating if all checks have passed
    let (Unew_non_base, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      &pp_digest,
      &i,
      &z_0,
      &z_i,
      &U,
      &r_i,
      &u,
      &nifs,
      &comm_W_fold,
      &comm_E_fold,
    )?;

    // Either check_non_base_pass=true or we are in the base case
    let should_be_false = AllocatedBit::nor(
      cs.namespace(|| "check_non_base_pass nor base_case"),
      &check_non_base_pass,
      &is_base_case,
    )?;
    cs.enforce(
      || "check_non_base_pass nor base_case = false",
      |lc| lc + should_be_false.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

    // we pick between the base case output and the non-base case output
    let Unew = Unew_base.conditionally_select(
      cs.namespace(|| "compute U_new"),
      &Unew_non_base,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute i + 1
    let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + E::Scalar::ONE)
    })?;
    cs.enforce(
      || "check i + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_new.get_variable() - CS::one() - i.get_variable(),
    );

    // Compute z_{i+1}
    let z_input = conditionally_select_vec(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case),
    )?;

    let z_next = self
      .step_circuit
      .synthesize(&mut cs.namespace(|| "F"), &z_input)?;

    if z_next.len() != arity {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_next".to_string(),
      ));
    }

    // Compute the new hash H(pp_digest, Unew, i+1, z0, z_{i+1})
    let mut ro = E::RO2Circuit::new(self.ro_consts);
    ro.absorb(&pp_digest);
    ro.absorb(&i_new);
    for e in &z_0 {
      ro.absorb(e);
    }
    for e in &z_next {
      ro.absorb(e);
    }
    Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;
    ro.absorb(&r_next);
    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS, false)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), &hash_bits)?;

    // Outputs the computed hash
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok(z_next)
  }
}

/// C1-β BIP-340 witness/ck threading sub-corrigendum §4.2 (Halpert,
/// 2026-05-18): sibling impl block keyed on
/// [`crate::traits::circuit::StepCircuitWithAux<E::Scalar, E>`] carrying
/// the lookup-aware augmented-circuit synthesize entry point. Sibling
/// of [`NeutronAugmentedCircuit::synthesize`] (the upstream-tracking
/// non-lookup-aware path); chosen at the call site by
/// [`crate::neutron::RecursiveSNARK::prove_step_with_lookup_fold_aux`]
/// and [`crate::neutron::PublicParams::setup_with_ptau_dir_aux`].
///
/// The body is byte-identical to `synthesize` except for the inner
/// step-circuit invocation at the `cs.namespace(|| "F")` site: the
/// CS is wrapped in [`crate::lookup::CSWithLookups`] before invoking
/// [`crate::traits::circuit::StepCircuitWithAux::synthesize_with_aux`].
/// The wrapper's local `QueryCollector` is intentionally NOT flushed
/// per sub-corrigendum §4.2 + §3.1 sound-by-presumption analysis (the
/// chunk-lookup binding is carried off-circuit by
/// [`crate::neutron::LookupStepCircuit::per_table_bundles_at_step`]).
#[cfg(feature = "lookup-fold")]
impl<E: Engine, SC> NeutronAugmentedCircuit<'_, E, SC>
where
  SC: crate::traits::circuit::StepCircuitWithAux<E::Scalar, E>,
{
  /// Lookup-aware synthesize sibling — see impl-block doc-comment.
  pub fn synthesize_aux<CS: ConstraintSystem<E::Scalar>>(
    self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    let arity = self.step_circuit.arity();

    // Allocate all witnesses
    let (pp_digest, i, z_0, z_i, U, r_i, r_next, u, nifs, comm_W_fold, comm_E_fold) =
      self.alloc_witness(cs.namespace(|| "allocate the circuit witness"), arity)?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // synthesize base case
    let Unew_base = self.synthesize_base_case(cs.namespace(|| "synthesize base case"))?;

    // Synthesize the circuit for the non-base case and get the new running
    // instance along with a boolean indicating if all checks have passed
    let (Unew_non_base, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      &pp_digest,
      &i,
      &z_0,
      &z_i,
      &U,
      &r_i,
      &u,
      &nifs,
      &comm_W_fold,
      &comm_E_fold,
    )?;

    // Either check_non_base_pass=true or we are in the base case
    let should_be_false = AllocatedBit::nor(
      cs.namespace(|| "check_non_base_pass nor base_case"),
      &check_non_base_pass,
      &is_base_case,
    )?;
    cs.enforce(
      || "check_non_base_pass nor base_case = false",
      |lc| lc + should_be_false.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

    // we pick between the base case output and the non-base case output
    let Unew = Unew_base.conditionally_select(
      cs.namespace(|| "compute U_new"),
      &Unew_non_base,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute i + 1
    let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + E::Scalar::ONE)
    })?;
    cs.enforce(
      || "check i + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_new.get_variable() - CS::one() - i.get_variable(),
    );

    // Compute z_{i+1}
    let z_input = conditionally_select_vec(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case),
    )?;

    // Sub-corrigendum §4.2: wrap the step-circuit namespace in
    // CSWithLookups so `register_lookup_query` / `register_chunk_lookup_table`
    // calls from inside `synthesize_with_aux` find a `LookupConstraintSystem<F>`
    // implementor. The wrapper's local `QueryCollector` is intentionally
    // not flushed (per §4.2 + §3.1 sound-by-presumption).
    //
    // Shape-pass semantic model ratification (Halpert, 2026-05-19) §5
    // note 4 U.host.c: thread `&i` (the fold-step counter allocated at
    // `:370` / unpacked at `:992`) into the step-circuit body so the
    // inumbra-side `synthesize_with_aux` can derive the host PC from
    // `i.get_value()` at prove time (and fall back to a shape-time
    // default when `get_value() == None` under `ShapeCS::alloc`).
    let z_next = {
      let mut ns = cs.namespace(|| "F");
      let mut cs_aux = crate::lookup::CSWithLookups::new(&mut ns);
      self.step_circuit.synthesize_with_aux(&mut cs_aux, &i, &z_input)?
    };

    if z_next.len() != arity {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_next".to_string(),
      ));
    }

    // Compute the new hash H(pp_digest, Unew, i+1, z0, z_{i+1})
    let mut ro = E::RO2Circuit::new(self.ro_consts);
    ro.absorb(&pp_digest);
    ro.absorb(&i_new);
    for e in &z_0 {
      ro.absorb(e);
    }
    for e in &z_next {
      ro.absorb(e);
    }
    Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;
    ro.absorb(&r_next);
    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS, false)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), &hash_bits)?;

    // Outputs the computed hash
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok(z_next)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      solver::SatisfyingAssignment,
      test_shape_cs::TestShapeCS,
    },
    provider::{
      Bn256EngineKZG, GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
    },
    r1cs::R1CSShape,
    traits::{circuit::TrivialCircuit, snark::default_ck_hint, RO2ConstantsCircuit},
  };
  use expect_test::{expect, Expect};

  // In the following we use 1 to refer to the primary, and 2 to refer to the secondary circuit
  fn test_recursive_circuit_with<E1, E2>(num_constraints: &Expect)
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let ro_consts = RO2ConstantsCircuit::<E1>::default();
    let tc = TrivialCircuit::default();

    let circuit: NeutronAugmentedCircuit<'_, E1, TrivialCircuit<E1::Scalar>> =
      NeutronAugmentedCircuit::new(None, &tc, ro_consts.clone());
    let mut cs: TestShapeCS<E1> = TestShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();
    num_constraints.assert_eq(cs.num_constraints().to_string().as_str());

    // Execute the base case for the primary
    let zero = <E1::Scalar as Field>::ZERO;
    let mut cs = SatisfyingAssignment::<E1>::new();
    let inputs: NeutronAugmentedCircuitInputs<E1> = NeutronAugmentedCircuitInputs::new(
      zero, // pass zero for testing
      zero,
      vec![zero],
      None,
      None,
      None,
      zero,
      None,
      None,
      None,
      None,
    );
    let circuit: NeutronAugmentedCircuit<'_, E1, TrivialCircuit<E1::Scalar>> =
      NeutronAugmentedCircuit::new(Some(inputs), &tc, ro_consts);
    let _ = circuit.synthesize(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    // Make sure that this is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  #[test]
  fn test_neutron_recursive_circuit_pasta() {
    test_recursive_circuit_with::<PallasEngine, VestaEngine>(&expect!["5493"]);
    test_recursive_circuit_with::<Bn256EngineKZG, GrumpkinEngine>(&expect!["5773"]);
    test_recursive_circuit_with::<Secp256k1Engine, Secq256k1Engine>(&expect!["6238"]);
  }
}
