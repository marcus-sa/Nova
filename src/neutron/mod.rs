//! This module implements an IVC scheme based on the NeutronNova folding scheme.
//! This code currently lacks certain checks, so do not use this until the experimental feature is removed.
use crate::{
  constants::NUM_HASH_BITS,
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  frontend::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
    ConstraintSystem, SynthesisError,
  },
  r1cs::{CommitmentKeyHint, R1CSInstance, R1CSShape, R1CSWitness},
  traits::{
    circuit::StepCircuit, AbsorbInRO2Trait, Engine, RO2Constants, RO2ConstantsCircuit, ROTrait,
  },
  CommitmentKey,
};
use core::marker::PhantomData;
use ff::Field;
use once_cell::sync::OnceCell;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

mod circuit;
// GH-#5 M.GH5.7 / Pin Corrigendum #10 Q2 ruling (Halpert 2026-05-10):
// targeted re-exports of the circuit-side symbols required by the
// M.GH5.7 negative-test triple's Option A.2 path (harness-side direct
// invocation of `AllocatedNIFS::verify_with_multi_table_lookup` for
// tests (ii) and (iii)). This composes with M.GH5.1's vendor visibility
// bumps on `NIFS::{poly, poly_lookup, comm_E, comm_inv_w, comm_inv_t}`
// (zero-algebra-zero-FS-zero-digest); the audit envelope already covers
// harness-from-vendor reachability for the lookup-fold surface.
//
// Re-exported symbols (test-only consumers in
// `crates/inumbra-spend-harness/tests/m_gh5_7_negative_test_triple.rs`):
//
//   - `NeutronAugmentedCircuit` / `NeutronAugmentedCircuitInputs` (test (i)
//     full augmented-circuit synthesis under the `corrupt_t_lookup_at_index`
//     `test-debug-knobs` knob)
//   - `AllocatedNIFS` / `AllocatedFoldedInstance` /
//     `AllocatedLookupNIFSMultiTable` / `AllocatedLookupPayloadPublicMultiTable`
//     / `AllocatedNonnativeR1CSInstance` (tests (ii) and (iii) direct
//     invocation of the in-circuit verifier under Option A.2).
//
// Production callsites for the augmented circuit live inside vendor
// `PublicParams::setup` (this same file, lines 165-225) and DO NOT
// expose the in-circuit types to inumbra public-params consumers; those
// route through `PublicParams::setup` / `RecursiveSNARK::prove_step`,
// which are already pub.
pub use circuit::{NeutronAugmentedCircuit, NeutronAugmentedCircuitInputs};
#[cfg(feature = "lookup-fold")]
pub use circuit::{
    lookup::{AllocatedLookupNIFSMultiTable, AllocatedLookupPayloadPublicMultiTable},
    nifs::AllocatedNIFS,
    r1cs::AllocatedNonnativeR1CSInstance,
    relation::AllocatedFoldedInstance,
};
pub mod nifs;
pub mod relation;

/// GH-#7 / Stage K design pin Corrigenda #7+#8+#9: NeutronNova-side
/// compressed SNARK envelope, landing incrementally across M.GH7.0.1
/// (Pedersen MSM-linearity split-E commitment helper) → M.GH7.0.2
/// (envelope + verifier off-FS binding check) → M.GH7.4 (LogUp identity
/// composition).
pub mod compressed_snark;

// GH-#7 M.GH7.1 module export hygiene: surface the envelope's public
// items at the `neutron::` path so downstream consumers (e.g. the
// `inumbra-spend-harness` integration tests in
// `crates/inumbra-spend-harness/tests/gh7_stage_k_compressor.rs`) can
// reach them without typing the `compressed_snark::` submodule prefix.
// The fully-qualified path `neutron::compressed_snark::CompressedSNARK`
// remains valid — these re-exports are additive, mirroring the upstream
// `nova::CompressedSNARK` precedent at `nova/mod.rs` where the envelope
// items sit at the `nova::` path directly.
#[doc(inline)]
pub use compressed_snark::{
  BridgedNeutronInstance, CompressedSNARK, ProverKey as CompressedSNARKProverKey,
  SigmaE2EqualityProof, SplitECommitments, VerifierKey as CompressedSNARKVerifierKey,
};

/// Closed-form, fold-step-collapsed sumcheck instance for the C1-β
/// lookup-fold extension. Pinned by §B of the spike implementation
/// outline (amendment 2026-05-07).
#[cfg(feature = "lookup-fold")]
pub mod lookup_sumcheck;

// Note: `NeutronAugmentedCircuit` / `NeutronAugmentedCircuitInputs` are
// `pub use`-re-exported above (M.GH5.7 / Corrigendum #10 Q2). The in-file
// callsites (`PublicParams::setup`, `RecursiveSNARK::*`) reach them via
// the re-export; no separate `use` needed.
use nifs::NIFS;
#[cfg(feature = "lookup-fold")]
use nifs::PerTableBundle;
#[cfg(feature = "lookup-fold")]
use relation::{LookupPayloadPublicMultiTable, LookupRunningWitness, LookupShape};
use relation::{FoldedInstance, FoldedWitness, Structure};

/// A type that holds public parameters of Nova
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  // GH-#7 M.GH7.0.2 visibility bumps (Corrigendum #10 + #11): `F_arity`,
  // `ro_consts`, `ck`, `structure` bumped from fully-private to `pub(crate)`
  // so the sibling `neutron::compressed_snark` module can construct
  // `CompressedSNARK::setup` and `CompressedSNARK::prove` against the
  // underlying R1CS shape, commitment key, and IVC F_arity / ro_consts. The
  // fields remain crate-private from the inumbra side; only the in-crate
  // envelope reads them. Co-classified with the M.GH7.0.1 `r_W/r_E` bump
  // bracket at `relation.rs:233/245` and the M.GH5.1 / M.GH5.7 lookup-fold-
  // surface bracket noted at `neutron/mod.rs:30-32`.
  pub(crate) F_arity: usize,

  pub(crate) ro_consts: RO2Constants<E1>,
  ro_consts_circuit: RO2ConstantsCircuit<E1>,
  pub(crate) ck: CommitmentKey<E1>,
  pub(crate) structure: Structure<E1>,

  /// GH-#5 M.GH5.4 / pin §3.1: per-position `pp_digest` registry, in
  /// chunk-position-canonical order (NOT `table_id` order). Length pinned
  /// by the inumbra-side public-params (length 16 production per
  /// ADR-0021). Empty Vec ⇒ no lookup-fold path; the augmented circuit
  /// falls back to `nifs.verify` and `default(cs, num_io)` (vendor
  /// upstream-tracking shape; what `TrivialCircuit` / `CubicCircuit`
  /// vendor-internal tests exercise via `setup` with default args).
  ///
  /// `#[serde(skip)]` so adding the field does not change the bincode
  /// encoding of `PublicParams` and therefore preserves `pp_digest`
  /// byte-equivalence at the default state. The shape registry is the
  /// inumbra-side artifact the M.7 shape-registry assertion fires
  /// against (`circuit/nifs.rs:689-695` in `verify_with_multi_table_lookup`),
  /// orthogonal to `pp_digest` itself.
  // GH-#7 M.GH7.1 visibility bump: `shape_registry` and `lookup_fold_k`
  // bumped from fully-private to `pub(crate)` so the sibling
  // `neutron::compressed_snark` module can read them at
  // `CompressedSNARK::setup` to populate `VerifierKey::lookup_fold_k`
  // and (later, M.GH7.2 / M.GH7.4) bind `shape_registry_digest` into
  // the verifier key per pin §3.3. Co-classified with the M.GH7.0.2
  // `F_arity` / `ro_consts` / `ck` / `structure` visibility bumps in
  // the bracket above.
  #[cfg(feature = "lookup-fold")]
  #[serde(skip, default)]
  pub shape_registry: Vec<E1::Scalar>,

  /// GH-#5 M.GH5.4 / pin §3.1: structurally-pinned per-table count
  /// (`LookupShape::multi_column_tables.len()`). Production: k=2 per
  /// ADR-0021, uniform across all 16 chunked-Strauss-Shamir positions
  /// (Auditor Obligation 10). `0` ⇒ no lookup-fold path.
  /// `#[serde(skip)]` for the same reason as `shape_registry`.
  #[cfg(feature = "lookup-fold")]
  #[serde(skip, default)]
  pub(crate) lookup_fold_k: usize,

  /// GH-#5 M.GH5.4 / pin §3.1: bit-width for the `chunk_index_in_z`
  /// range-check inside the M.7 shape-registry assertion. Must be
  /// `>= ceil(log2(shape_registry.len()))`. Production: 4 (16 positions
  /// per ADR-0021); 5 if extended to ≤ 30 chunk-band positions (Phase-5+).
  #[cfg(feature = "lookup-fold")]
  #[serde(skip, default)]
  index_n_bits: usize,

  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E1::Scalar>,
  _p: PhantomData<(C, E2)>,
}

impl<E1, E2, C> SimpleDigestible for PublicParams<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
}

impl<E1, E2, C> PublicParams<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  /// Creates a new `PublicParams` for a circuit `C`.
  ///
  /// # Note
  ///
  /// Public parameters set up a number of bases for the homomorphic commitment scheme of Nova.
  ///
  /// Some final compressing SNARKs, like variants of Spartan, use computation commitments that require
  /// larger sizes for these parameters. These SNARKs provide a hint for these values by
  /// implementing `RelaxedR1CSSNARKTrait::ck_floor()`, which can be passed to this function.
  ///
  /// If you're not using such a SNARK, pass `nova_snark::traits::snark::default_ck_hint()` instead.
  ///
  /// # Arguments
  ///
  /// * `c`: The primary circuit of type `C`.
  /// * `ck_hint1`: A `CommitmentKeyHint` for `S1`, which is a function that provides a hint
  ///   for the number of generators required in the commitment scheme for the primary circuit.
  /// * `ck_hint2`: A `CommitmentKeyHint` for `S2`, similar to `ck_hint1`, but for the secondary circuit.
  ///
  /// # Example
  ///
  /// ```rust
  /// # use nova_snark::spartan::ppsnark::RelaxedR1CSSNARK;
  /// # use nova_snark::provider::ipa_pc::EvaluationEngine;
  /// # use nova_snark::provider::{PallasEngine, VestaEngine};
  /// # use nova_snark::traits::{circuit::TrivialCircuit, Engine, snark::RelaxedR1CSSNARKTrait};
  /// # use nova_snark::nova::PublicParams;
  ///
  /// type E1 = PallasEngine;
  /// type E2 = VestaEngine;
  /// type EE<E> = EvaluationEngine<E>;
  /// type SPrime<E> = RelaxedR1CSSNARK<E, EE<E>>;
  ///
  /// let circuit = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
  /// // Only relevant for a SNARK using computational commitments, pass &(|_| 0)
  /// // or &*nova_snark::traits::snark::default_ck_hint() otherwise.
  /// let ck_hint1 = &*SPrime::<E1>::ck_floor();
  /// let ck_hint2 = &*SPrime::<E2>::ck_floor();
  ///
  /// let pp = PublicParams::setup(&circuit, ck_hint1, ck_hint2)?;
  /// Ok::<(), nova_snark::errors::NovaError>(())
  /// ```
  ///
  /// # GH-#5 M.GH5.4: lookup-fold path
  ///
  /// Under `feature = "lookup-fold"`, `setup` accepts a non-empty
  /// `shape_registry` + non-zero `lookup_fold_k` + non-zero `index_n_bits`
  /// to wire `NeutronAugmentedCircuit::with_lookup_fold` per pin §3.1.
  /// At the default `(vec![], 0, 0)` state, the shape derivation reduces
  /// to the upstream-tracking pre-M.GH5.3 path (no `T_lookup_per_table`
  /// allocation, `nifs.verify` instead of `verify_with_multi_table_lookup`),
  /// which is what the vendor-internal `TrivialCircuit` / `CubicCircuit`
  /// tests exercise. The non-`lookup-fold` build ignores all three
  /// `_lookup_*` parameters (the field gate excludes them from the
  /// `PublicParams` struct).
  #[cfg(feature = "lookup-fold")]
  pub fn setup(
    c: &C,
    ck_hint1: &CommitmentKeyHint<E1>,
    _ck_hint2: &CommitmentKeyHint<E2>,
    shape_registry: Vec<E1::Scalar>,
    lookup_fold_k: usize,
    index_n_bits: usize,
    // GH-#7 M.GH7.3a (Corrigendum #14 sub-ratification #2 2026-05-12):
    // optional `LookupShape<E1>` routed into the `Structure<E1>`. `None`
    // preserves the pre-sub-ratification `Structure::new(&r1cs_shape)`
    // path byte-identically (vendor-internal default-state tests with
    // `TrivialCircuit` / `CubicCircuit` and `lookup_fold_k = 0` rely on
    // this; their `pp_digest` baselines at `mod.rs:830-841` MUST pass
    // unchanged). `Some(shape)` invokes `Structure::new_with_lookups(...)`
    // so `prove_step_with_lookup_fold` (added in the same milestone)
    // does not fail at `nifs.rs:1368` with `NovaError::InvalidStructure`.
    // Inumbra-side production wrap pipeline (M.GH7.5a) supplies
    // `Some(build_canonical_lookup_shape::<Bn256EngineKZG>(...))`;
    // vendor-internal callsites supply `None`.
    lookup_shape: Option<LookupShape<E1>>,
  ) -> Result<Self, NovaError> {
    let F_arity = c.arity();

    let ro_consts: RO2Constants<E1> = RO2Constants::<E1>::default();
    let ro_consts_circuit: RO2ConstantsCircuit<E1> = RO2ConstantsCircuit::<E1>::default();

    // Initialize shape for the primary
    //
    // GH-#5 M.GH5.4: thread the shape registry to the augmented-circuit
    // constructor via `with_lookup_fold(...)`. At `lookup_fold_k == 0`
    // the call is a no-op (defaults are already 0/&[]/0 in `new`), so
    // the R1CS shape and `pp_digest` are byte-equivalent to pre-M.GH5.4
    // for the vendor-internal default-state tests.
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(None, c, ro_consts_circuit.clone())
        .with_lookup_fold(lookup_fold_k, &shape_registry, index_n_bits);
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let r1cs_shape = cs.r1cs_shape()?;

    if r1cs_shape.num_io != 1 {
      return Err(NovaError::InvalidStepCircuitIO);
    }

    // Generate the commitment key
    let ck = R1CSShape::commitment_key(&[&r1cs_shape], &[ck_hint1])?;

    // GH-#7 M.GH7.3a (Corrigendum #14 sub-ratification #2): route the
    // `Structure<E1>` through `new_with_lookups` when a `LookupShape`
    // is supplied; otherwise preserve the pre-sub-ratification
    // `Structure::new` path byte-identically (`pp_digest` invariant at
    // `lookup_fold_k = 0` + `lookup_shape = None`).
    let structure = match lookup_shape {
      Some(shape) => Structure::new_with_lookups(&r1cs_shape, shape),
      None => Structure::new(&r1cs_shape),
    };

    let pp = PublicParams {
      F_arity,

      ro_consts,
      ro_consts_circuit,
      ck,
      structure,

      shape_registry,
      lookup_fold_k,
      index_n_bits,

      digest: OnceCell::new(),
      _p: Default::default(),
    };

    // call pp.digest() so the digest is computed here rather than in RecursiveSNARK methods
    let _ = pp.digest();

    Ok(pp)
  }

  /// `setup` for non-`lookup-fold` builds — preserves the pre-M.GH5.4
  /// signature so upstream-tracking consumers (vendor examples / benches
  /// not built under `lookup-fold`) compile unchanged.
  #[cfg(not(feature = "lookup-fold"))]
  pub fn setup(
    c: &C,
    ck_hint1: &CommitmentKeyHint<E1>,
    _ck_hint2: &CommitmentKeyHint<E2>,
  ) -> Result<Self, NovaError> {
    let F_arity = c.arity();

    let ro_consts: RO2Constants<E1> = RO2Constants::<E1>::default();
    let ro_consts_circuit: RO2ConstantsCircuit<E1> = RO2ConstantsCircuit::<E1>::default();

    // Initialize shape for the primary
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(None, c, ro_consts_circuit.clone());
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let r1cs_shape = cs.r1cs_shape()?;

    if r1cs_shape.num_io != 1 {
      return Err(NovaError::InvalidStepCircuitIO);
    }

    // Generate the commitment key
    let ck = R1CSShape::commitment_key(&[&r1cs_shape], &[ck_hint1])?;

    let structure = Structure::new(&r1cs_shape);

    let pp = PublicParams {
      F_arity,

      ro_consts,
      ro_consts_circuit,
      ck,
      structure,

      digest: OnceCell::new(),
      _p: Default::default(),
    };

    // call pp.digest() so the digest is computed here rather than in RecursiveSNARK methods
    let _ = pp.digest();

    Ok(pp)
  }

  /// Creates a new `PublicParams` for a circuit `C` using commitment keys loaded from a ptau directory.
  ///
  /// This is designed for use with HyperKZG or Mercury on the primary curve (e.g., BN256).
  /// The commitment key is loaded from a Powers of Tau ceremony file.
  ///
  /// **Note:** This method requires `E1::GE` to implement `PairingGroup`. It is only available
  /// for pairing-friendly curves (BN256, BLS12-381, etc.).
  ///
  /// # Arguments
  ///
  /// * `c`: The primary circuit of type `C`.
  /// * `ck_hint1`: A `CommitmentKeyHint` for the primary circuit.
  /// * `ck_hint2`: A `CommitmentKeyHint` for the secondary circuit (unused but kept for API consistency).
  /// * `ptau_dir`: Path to the directory containing pruned ptau files.
  #[cfg(all(feature = "io", feature = "lookup-fold"))]
  pub fn setup_with_ptau_dir(
    c: &C,
    ck_hint1: &CommitmentKeyHint<E1>,
    _ck_hint2: &CommitmentKeyHint<E2>,
    ptau_dir: &std::path::Path,
    shape_registry: Vec<E1::Scalar>,
    lookup_fold_k: usize,
    index_n_bits: usize,
    // GH-#7 M.GH7.3a (Corrigendum #14 sub-ratification #2 2026-05-12):
    // optional `LookupShape<E1>` (mirror of `setup` above). See the
    // narration at `setup`'s `lookup_shape` parameter for rationale.
    lookup_shape: Option<LookupShape<E1>>,
  ) -> Result<Self, NovaError>
  where
    E1::GE: crate::provider::traits::PairingGroup,
  {
    let F_arity = c.arity();

    let ro_consts: RO2Constants<E1> = RO2Constants::<E1>::default();
    let ro_consts_circuit: RO2ConstantsCircuit<E1> = RO2ConstantsCircuit::<E1>::default();

    // Initialize shape for the primary
    //
    // GH-#5 M.GH5.4: thread the shape registry to the augmented-circuit
    // constructor via `with_lookup_fold(...)` (mirror of `setup` above).
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(None, c, ro_consts_circuit.clone())
        .with_lookup_fold(lookup_fold_k, &shape_registry, index_n_bits);
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let r1cs_shape = cs.r1cs_shape()?;

    if r1cs_shape.num_io != 1 {
      return Err(NovaError::InvalidStepCircuitIO);
    }

    // Load the commitment key from ptau directory
    let ck = R1CSShape::commitment_key_from_ptau_dir(&[&r1cs_shape], &[ck_hint1], ptau_dir)?;

    // GH-#7 M.GH7.3a (Corrigendum #14 sub-ratification #2): mirror of
    // `setup` above — match-branch routing on `lookup_shape`.
    let structure = match lookup_shape {
      Some(shape) => Structure::new_with_lookups(&r1cs_shape, shape),
      None => Structure::new(&r1cs_shape),
    };

    let pp = PublicParams {
      F_arity,

      ro_consts,
      ro_consts_circuit,
      ck,
      structure,

      shape_registry,
      lookup_fold_k,
      index_n_bits,

      digest: OnceCell::new(),
      _p: Default::default(),
    };

    // call pp.digest() so the digest is computed here rather than in RecursiveSNARK methods
    let _ = pp.digest();

    Ok(pp)
  }

  /// C1-β BIP-340 witness/ck threading sub-corrigendum §6 (Halpert,
  /// 2026-05-18): shape-pass sibling of [`Self::setup_with_ptau_dir`]
  /// that exercises the lookup-aware
  /// [`crate::neutron::circuit::NeutronAugmentedCircuit::synthesize_aux`]
  /// entry point so that `pp.structure` (the R1CS shape consumed at
  /// `cs.r1cs_instance_and_witness(&pp.structure.S, &pp.ck)` at the
  /// prove site of `prove_step_with_lookup_fold_aux`) includes the
  /// per-chunk constraints emitted by the chunked BIP-340 helpers
  /// inside the step-circuit's `synthesize_with_aux` body.
  ///
  /// Mirrors `setup_with_ptau_dir` byte-for-byte except for the
  /// shape-pass synthesize call: `circuit.synthesize_aux(&mut cs)`
  /// instead of `circuit.synthesize(&mut cs)`. The shape pass runs
  /// against `ShapeCS<E1>` — `ShapeCS` implements `ConstraintSystem`
  /// only (no witness assignment), so the `CSWithLookups` wrapper
  /// inside `synthesize_aux` records constraint structure into the
  /// inner `ShapeCS` and queries into the local `QueryCollector` that
  /// drops on `synthesize_aux` return (the off-circuit
  /// `LookupStepCircuit::per_table_bundles_at_step` is NOT exercised
  /// at shape time; the chunk-lookup binding is bound only at prove
  /// time via the off-circuit bundles assembled at
  /// `prove_step_with_lookup_fold_aux:983` and the in-circuit
  /// `register_chunk_lookup_table` content-addressed commitment
  /// re-derivation).
  ///
  /// HG-A1.4-6 close criterion (parent corrigendum §4 CK-α.1): the
  /// `register_chunk_lookup_table::<E>(default_shape_pass_ck(), ID)`
  /// commitment at shape time equals
  /// `register_chunk_lookup_table::<E>(pp.ck(), ID)` commitment at
  /// prove time. Preserved by construction under the inumbra-side
  /// `Default::default()` shape-pass-ck wiring (parent §4 + §5).
  #[cfg(all(feature = "io", feature = "lookup-fold"))]
  pub fn setup_with_ptau_dir_aux(
    c: &C,
    ck_hint1: &CommitmentKeyHint<E1>,
    _ck_hint2: &CommitmentKeyHint<E2>,
    ptau_dir: &std::path::Path,
    shape_registry: Vec<E1::Scalar>,
    lookup_fold_k: usize,
    index_n_bits: usize,
    lookup_shape: Option<LookupShape<E1>>,
  ) -> Result<Self, NovaError>
  where
    E1::GE: crate::provider::traits::PairingGroup,
    C: crate::traits::circuit::StepCircuitWithAux<E1::Scalar, E1>,
  {
    let F_arity = c.arity();

    let ro_consts: RO2Constants<E1> = RO2Constants::<E1>::default();
    let ro_consts_circuit: RO2ConstantsCircuit<E1> = RO2ConstantsCircuit::<E1>::default();

    // Initialize shape for the primary — mirror of `setup_with_ptau_dir`
    // with the synthesize-aux sibling call.
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(None, c, ro_consts_circuit.clone())
        .with_lookup_fold(lookup_fold_k, &shape_registry, index_n_bits);
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit.synthesize_aux(&mut cs);
    let r1cs_shape = cs.r1cs_shape()?;

    if r1cs_shape.num_io != 1 {
      return Err(NovaError::InvalidStepCircuitIO);
    }

    // Load the commitment key from ptau directory
    let ck = R1CSShape::commitment_key_from_ptau_dir(&[&r1cs_shape], &[ck_hint1], ptau_dir)?;

    let structure = match lookup_shape {
      Some(shape) => Structure::new_with_lookups(&r1cs_shape, shape),
      None => Structure::new(&r1cs_shape),
    };

    let pp = PublicParams {
      F_arity,

      ro_consts,
      ro_consts_circuit,
      ck,
      structure,

      shape_registry,
      lookup_fold_k,
      index_n_bits,

      digest: OnceCell::new(),
      _p: Default::default(),
    };

    // call pp.digest() so the digest is computed here rather than in RecursiveSNARK methods
    let _ = pp.digest();

    Ok(pp)
  }

  /// `setup_with_ptau_dir` for non-`lookup-fold` builds — preserves the
  /// pre-M.GH5.4 signature.
  #[cfg(all(feature = "io", not(feature = "lookup-fold")))]
  pub fn setup_with_ptau_dir(
    c: &C,
    ck_hint1: &CommitmentKeyHint<E1>,
    _ck_hint2: &CommitmentKeyHint<E2>,
    ptau_dir: &std::path::Path,
  ) -> Result<Self, NovaError>
  where
    E1::GE: crate::provider::traits::PairingGroup,
  {
    let F_arity = c.arity();

    let ro_consts: RO2Constants<E1> = RO2Constants::<E1>::default();
    let ro_consts_circuit: RO2ConstantsCircuit<E1> = RO2ConstantsCircuit::<E1>::default();

    // Initialize shape for the primary
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(None, c, ro_consts_circuit.clone());
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let r1cs_shape = cs.r1cs_shape()?;

    if r1cs_shape.num_io != 1 {
      return Err(NovaError::InvalidStepCircuitIO);
    }

    // Load the commitment key from ptau directory
    let ck = R1CSShape::commitment_key_from_ptau_dir(&[&r1cs_shape], &[ck_hint1], ptau_dir)?;

    let structure = Structure::new(&r1cs_shape);

    let pp = PublicParams {
      F_arity,

      ro_consts,
      ro_consts_circuit,
      ck,
      structure,

      digest: OnceCell::new(),
      _p: Default::default(),
    };

    // call pp.digest() so the digest is computed here rather than in RecursiveSNARK methods
    let _ = pp.digest();

    Ok(pp)
  }

  /// Retrieve the digest of the public parameters.
  pub fn digest(&self) -> E1::Scalar {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .expect("Failure in retrieving digest")
  }

  /// C1-β BIP-340 witness/ck threading parent corrigendum §3 + §7
  /// (Halpert, 2026-05-18): public accessor for the prove-time
  /// commitment key. The inumbra-side `wrap_pipeline` snapshots this
  /// into `InumbraSpendStepCircuitBn254::new_with_bip340_and_ck` so
  /// the chunked-lookup table commitment registered inside
  /// `synthesize_with_aux` is byte-equal to the shape-pass commitment
  /// (CK-α.1 shape-pass-ck-equals-prove-time-ck discipline).
  pub fn ck(&self) -> &CommitmentKey<E1> {
    &self.ck
  }
}

/// A SNARK that proves the correct execution of an incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  // GH-#7 M.GH7.5 visibility bump (Corrigendum #20 (VS-2) (S2) disposition):
  // `z0` bumped to `pub(crate)` so the sibling `neutron::compressed_snark`
  // module's `prove_with_lookup_fold<E2, C>` wrapper can snapshot it into
  // the envelope `z0_snapshot` field. Parallel to the existing M.GH7.0.2
  // bracket immediately below. See pin §M.GH7.5 work-item 4 for the
  // wrapper's per-field consumption.
  pub(crate) z0: Vec<E1::Scalar>,

  // GH-#7 M.GH7.0.2 visibility bumps (Corrigendum #10 + #11): `r_W`, `r_U`,
  // `zi` bumped from fully-private to `pub(crate)` so the sibling
  // `neutron::compressed_snark` module can construct
  // `CompressedSNARK::prove` against the IVC-final folded state. Co-classified
  // with the M.GH7.0.1 `FoldedWitness::{r_W, r_E}` visibility bump bracket
  // (relation.rs:233/245) and the M.GH5.1 / M.GH5.7 lookup-fold-surface
  // bracket noted at `neutron/mod.rs:30-32`. Fields remain crate-private
  // from the inumbra side; only the in-crate envelope reads them.
  pub(crate) r_W: FoldedWitness<E1>,
  pub(crate) r_U: FoldedInstance<E1>,
  // M.GH7.5 (Corrigendum #20) extension: `ri` bumped to `pub(crate)` so the
  // envelope `ri_snapshot` field can be populated by `prove_with_lookup_fold`.
  pub(crate) ri: E1::Scalar,

  l_w: R1CSWitness<E1>,
  // M.GH7.5 (Corrigendum #20) extension: `l_u` bumped to `pub(crate)` so the
  // envelope `l_u_X0_snapshot` field can be populated (`l_u.X[0]` is the IVC
  // public-input hash bound by the augmented-circuit final-step inputize at
  // `circuit/mod.rs:955`). Per Corrigendum #20 second-order issue #3, only
  // `X[0]` is snapshotted (not the full `R1CSInstance`); `comm_W` is bound
  // redundantly by the Spartan-close envelope.
  pub(crate) l_u: R1CSInstance<E1>,

  // M.GH7.5 (Corrigendum #20) extension: `i` bumped to `pub(crate)` so the
  // envelope `i_snapshot` field can be populated.
  pub(crate) i: usize,

  pub(crate) zi: Vec<E1::Scalar>,

  /// GH-#7 M.GH7.3a (Corrigendum #14 re-typed; sub-ratification #2
  /// 2026-05-12): per-table running lookup witnesses carried across
  /// `prove_step_with_lookup_fold` invocations. Length is pinned by
  /// `pp.structure.lookups.as_ref().unwrap().multi_column_tables.len()`
  /// at `RecursiveSNARK::new`; empty `Vec` at `pp.lookup_fold_k == 0`
  /// (the non-lookup path uses `prove_step` and never reads this field).
  /// Threaded through `LookupStepCircuit::per_table_bundles_at_step` as
  /// `prior_running_lws` on each subsequent sibling-method invocation,
  /// and overwritten with `NIFS::prove_with_multi_table_lookup`'s
  /// `next_running_lws` output.
  ///
  /// Pre-Corrigendum-#14 prescription's `step_bundles` /
  /// `step_lookup_nifs` / `t_lookup_running_per_table` trio is DROPPED
  /// per Corrigendum #14 + sub-ratification #2 (wrong element types or
  /// redundant with `r_U.T_lookup`). Field is `#[cfg(feature =
  /// "lookup-fold")]`-gated so the non-lookup-fold build's
  /// `RecursiveSNARK` struct shape is unchanged. `LookupRunningWitness`
  /// is `Serialize, Deserialize` per `relation.rs:409`, so the
  /// existing `#[derive(Serialize, Deserialize)]` on `RecursiveSNARK`
  /// at line 482-483 continues to typecheck.
  // M.GH7.5 (Corrigendum #20) extension: `running_lws` bumped to `pub(crate)`
  // so `compressed_snark::prove_with_lookup_fold<E2, C>` can read
  // `running_lws[j].witness` (the `v_j` of Corrigendum #17 Claim 2; verified
  // at vendor HEAD `39aaec4` per F2 finding) to project into the
  // length-`num_cons` synthetic-data flat embedding consumed by
  // `prove_from_parts_with_logup`. Crate-private from the inumbra side.
  #[cfg(feature = "lookup-fold")]
  pub(crate) running_lws: Vec<LookupRunningWitness<E1>>,

  _p: PhantomData<(C, E2)>,
}

impl<E1, E2, C> RecursiveSNARK<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  /// Create new instance of recursive SNARK
  pub fn new(pp: &PublicParams<E1, E2, C>, c: &C, z0: &[E1::Scalar]) -> Result<Self, NovaError> {
    if z0.len() != pp.F_arity {
      return Err(NovaError::InvalidInitialInputLength);
    }

    let ri = E1::Scalar::random(&mut OsRng);

    // base case for the primary
    let mut cs = SatisfyingAssignment::<E1>::new();
    let inputs: NeutronAugmentedCircuitInputs<E1> = NeutronAugmentedCircuitInputs::new(
      pp.digest(),
      E1::Scalar::ZERO,
      z0.to_vec(),
      None,
      None,
      None,
      ri, // "r next"
      None,
      None,
      None,
      None,
    );

    // GH-#5 M.GH5.4: thread the shape registry stored in `pp` to the
    // augmented-circuit constructor (mirror of `setup` above so the
    // shape-derivation site and the prove-step site emit the same R1CS).
    #[cfg(feature = "lookup-fold")]
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(Some(inputs), c, pp.ro_consts_circuit.clone())
        .with_lookup_fold(pp.lookup_fold_k, &pp.shape_registry, pp.index_n_bits);
    #[cfg(not(feature = "lookup-fold"))]
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(Some(inputs), c, pp.ro_consts_circuit.clone());
    let zi = circuit.synthesize(&mut cs)?;
    let (l_u, l_w) = cs.r1cs_instance_and_witness(&pp.structure.S, &pp.ck)?;

    assert!((zi.len() == pp.F_arity), "Invalid step length");

    let zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;

    // GH-#7 M.GH7.3a (Corrigendum #14 sub-ratification #2 2026-05-12):
    // bootstrap `running_lws` from `LookupRunningWitness::default(&shape)`
    // per registered multi-column table at `pp.structure.lookups`. At
    // `pp.lookup_fold_k == 0` (or `pp.structure.lookups == None`) the
    // result is `vec![]` and `prove_step_with_lookup_fold` cannot be
    // invoked productively (the inner rejects at `nifs.rs:1368` with
    // `NovaError::InvalidStructure`); the non-lookup `prove_step` path
    // does not read this field and continues to compile + run
    // byte-identical against the pre-Corrigendum-#14 vendor.
    #[cfg(feature = "lookup-fold")]
    let running_lws: Vec<LookupRunningWitness<E1>> = match pp.structure.lookups.as_ref() {
      Some(shape) if pp.lookup_fold_k > 0 => shape
        .multi_column_tables
        .iter()
        .map(|_| LookupRunningWitness::default(shape))
        .collect(),
      _ => vec![],
    };

    Ok(Self {
      z0: z0.to_vec(),
      r_W: FoldedWitness::default(&pp.structure),
      // GH-#7 design pin Corrigendum #25 (Fix-α native-side mirror):
      // initialize `r_U` with constant-shape `Some(vec![..; k])` for the
      // three IVC-hash-chain-absorbed lookup-side Options (`T_lookup`,
      // `comm_L`, `comm_ts`) when `pp.lookup_fold_k > 0`, mirroring the
      // in-circuit `AllocatedFoldedInstance::default_with_lookup_k` at
      // `circuit/relation.rs:363-434` so the verifier's `absorb_in_ro2`
      // hash-chain re-derive at `mod.rs:785` byte-equals the prover's
      // `absorb_in_ro` at `circuit/mod.rs:947`. Closes Obstruction 2 at
      // base case under `lookup_fold_k > 0`. Single production call site
      // narrowing per Corrigendum #25 finding 4 (the 26 remaining
      // call sites are all test-internal and exercise `None`-shape paths
      // — they continue to invoke the legacy `default(S)`). At
      // `pp.lookup_fold_k == 0`, the new constructor returns
      // byte-identical to `default(S)`. Under
      // `#[cfg(not(feature = "lookup-fold"))]`, the legacy `default(S)`
      // path is preserved verbatim.
      #[cfg(feature = "lookup-fold")]
      r_U: FoldedInstance::default_with_lookup_k(&pp.structure, pp.lookup_fold_k),
      #[cfg(not(feature = "lookup-fold"))]
      r_U: FoldedInstance::default(&pp.structure),
      ri,
      l_w,
      l_u,
      i: 0,
      zi,
      #[cfg(feature = "lookup-fold")]
      running_lws,
      _p: Default::default(),
    })
  }

  /// Updates the provided `RecursiveSNARK` by executing a step of the incremental computation
  pub fn prove_step(&mut self, pp: &PublicParams<E1, E2, C>, c: &C) -> Result<(), NovaError> {
    // first step was already done in the constructor
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    // fold the last instance with the running instance
    let (nifs, (r_U, r_W)) = NIFS::prove(
      &pp.ck,
      &pp.ro_consts,
      &pp.digest(),
      &pp.structure,
      &self.r_U,
      &self.r_W,
      &self.l_u,
      &self.l_w,
    )?;

    let r_next = E1::Scalar::random(&mut OsRng);

    let mut cs = SatisfyingAssignment::<E1>::new();
    let inputs: NeutronAugmentedCircuitInputs<E1> = NeutronAugmentedCircuitInputs::new(
      pp.digest(),
      E1::Scalar::from(self.i as u64),
      self.z0.to_vec(),
      Some(self.zi.clone()),
      Some(self.r_U.clone()),
      Some(self.ri),
      r_next,
      Some(self.l_u.clone()),
      Some(nifs),
      Some(r_U.comm_W),
      Some(r_U.comm_E),
    );

    // GH-#5 M.GH5.4: same threading as `RecursiveSNARK::new` above.
    #[cfg(feature = "lookup-fold")]
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(Some(inputs), c, pp.ro_consts_circuit.clone())
        .with_lookup_fold(pp.lookup_fold_k, &pp.shape_registry, pp.index_n_bits);
    #[cfg(not(feature = "lookup-fold"))]
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(Some(inputs), c, pp.ro_consts_circuit.clone());
    let zi = circuit.synthesize(&mut cs)?;

    let (l_u, l_w) = cs.r1cs_instance_and_witness(&pp.structure.S, &pp.ck)?;

    // update the running instances and witnesses
    self.zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;

    self.r_U = r_U;
    self.r_W = r_W;

    self.i += 1;

    self.ri = r_next;

    self.l_u = l_u;
    self.l_w = l_w;

    Ok(())
  }

  /// Verify the correctness of the `RecursiveSNARK`
  pub fn verify(
    &self,
    pp: &PublicParams<E1, E2, C>,
    num_steps: usize,
    z0: &[E1::Scalar],
  ) -> Result<Vec<E1::Scalar>, NovaError> {
    // number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;

    // check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;

    // check if the initial inputs match
    let is_inputs_not_match = self.z0 != z0;

    // check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outputs = self.l_u.X.len() != 1 || self.r_U.X.len() != 1;

    if is_num_steps_zero
      || is_num_steps_not_match
      || is_inputs_not_match
      || is_instance_has_two_outputs
    {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid number of steps or inputs".to_string(),
      });
    }

    // check if the output hashes in R1CS instances point to the right running instance
    let hash = {
      let mut hasher = E1::RO2::new(pp.ro_consts.clone());
      hasher.absorb(pp.digest());
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0 {
        hasher.absorb(*e);
      }
      for e in &self.zi {
        hasher.absorb(*e);
      }
      self.r_U.absorb_in_ro2(&mut hasher);
      hasher.absorb(self.ri);

      hasher.squeeze(NUM_HASH_BITS, false)
    };

    if hash != self.l_u.X[0] {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid output hash in R1CS instance".to_string(),
      });
    }

    // check the satisfiability of the provided instances
    let (res_r, res_l) = rayon::join(
      || pp.structure.is_sat(&pp.ck, &self.r_U, &self.r_W),
      || pp.structure.S.is_sat(&pp.ck, &self.l_u, &self.l_w),
    );

    // check the returned res objects
    res_r?;
    res_l?;

    Ok(self.zi.clone())
  }

  /// Get the outputs after the last step of computation.
  pub fn outputs(&self) -> &[E1::Scalar] {
    &self.zi
  }

  /// The number of steps which have been executed thus far.
  pub fn num_steps(&self) -> usize {
    self.i
  }
}

/// GH-#7 M.GH7.3a (Corrigendum #14 re-typed; sub-ratification #1
/// 2026-05-12 — `r_logup_per_table` argument DROPPED, sub-ratification
/// #2 2026-05-12 — `PublicParams::setup` `lookup_shape` parameter
/// prerequisite): trait extending `StepCircuit<E::Scalar>` for step
/// circuits that participate in the multi-table lookup-fold extension.
///
/// `LookupStepCircuit` is the type-level marker that
/// `RecursiveSNARK::prove_step_with_lookup_fold` consumes — only step
/// circuits implementing this trait reach the lookup-aware sibling
/// method. Non-lookup step circuits (e.g. `TrivialCircuit`,
/// `CubicCircuit`) implement only `StepCircuit` and continue to use
/// the unchanged `prove_step` non-lookup path, preserving vendor
/// upstream-tracking + classical-Nova consumers + STAGE 0 per
/// Corrigendum #13.
///
/// Two methods:
///
/// - `per_table_bundles_at_step` constructs fully-committed prover-side
///   per-table bundles for step `i` in `table_id`-canonical order
///   matching `Structure::lookups.multi_column_tables`. Bundles are
///   r_logup-INDEPENDENT — the inner orchestrates `r_logup_per_table`
///   internally at `nifs.rs:1446-1495` Step 5b (squeezed AFTER absorbing
///   the bundle commitments) and consumes it inside
///   `LookupSumcheckInstance::new` at `lookup_sumcheck.rs:146-221` to
///   compute fresh inverses internally. Production step circuits do
///   NOT consume `r_logup` in their bundle construction — gate #1 of
///   Corrigendum #14 dissolved by the 2026-05-12 sub-ratification
///   after the algebra-level r-independence audit at `nifs.rs:202-227`
///   (`PerTableBundle` fields) was empirically reviewed.
///
/// - `public_bundles` projects the prover-side bundles to the
///   verifier-public hints (`comm_L`, `comm_values`, `comm_ts` per
///   table) carried into the augmented circuit via
///   `NeutronAugmentedCircuitInputs::with_multi_table_bundles`.
///
/// At M.GH7.5a, inumbra-side production circuits
/// (`NoOpComplianceCircuit<Bn254Scalar>`,
/// `WrappedBtcAssetGateComplianceCircuit<Bn254Scalar>` per
/// `wrap_pipeline.rs:55`) implement `LookupStepCircuit<Bn256EngineKZG>`
/// against the canonical 2-table shape per ADR-0021 / Corrigendum #5.
#[cfg(feature = "lookup-fold")]
pub trait LookupStepCircuit<E: Engine>: StepCircuit<E::Scalar> {
  /// Construct fully-committed prover-side per-table bundles for step
  /// `i`.
  ///
  /// `prior_running_lws` is `&self.running_lws` from the calling
  /// `RecursiveSNARK`; at outer base (i=1, first non-bootstrap fold)
  /// it is the `vec![LookupRunningWitness::default(&shape); k]`
  /// bootstrap state; at i > 1 it is the prior step's
  /// `next_running_lws` output from `NIFS::prove_with_multi_table_lookup`.
  /// The implementation reads it as input to the `bundles[j].running_lw`
  /// field (carries the per-table fold-state into the next prove).
  ///
  /// Returns a `Vec<PerTableBundle<E>>` of length
  /// `Structure::lookups.multi_column_tables.len()` in
  /// `table_id`-canonical order. Per-bundle ordering and per-bundle
  /// column-count invariants are debug-asserted inside
  /// `NIFS::prove_with_multi_table_lookup_inner` at `nifs.rs:1378-1438`.
  fn per_table_bundles_at_step(
    &self,
    ck: &CommitmentKey<E>,
    i: usize,
    prior_running_lws: &[LookupRunningWitness<E>],
  ) -> Result<Vec<PerTableBundle<E>>, NovaError>;

  /// Project the prover-side bundles to the verifier-public hints
  /// (`LookupPayloadPublicMultiTable<E>`) delivered into the augmented
  /// circuit via `NeutronAugmentedCircuitInputs::with_multi_table_bundles`.
  ///
  /// Free-function-shaped (no `&self`) per Corrigendum #14 + the pin's
  /// STOP-AND-ASK on ergonomic trait-method dispatch — Rust trait
  /// dispatch accepts associated functions without `&self` parameter,
  /// so this stays a trait method (fallback to free-function
  /// `public_bundles_of<E: Engine>(...)` is not needed at M.GH7.3a).
  fn public_bundles(bundles: &[PerTableBundle<E>]) -> Vec<LookupPayloadPublicMultiTable<E>>;
}

/// GH-#7 M.GH7.3a (Corrigendum #14 re-typed): sibling-method impl block
/// for `RecursiveSNARK::prove_step_with_lookup_fold`.
///
/// This block is structurally parallel to the existing
/// `impl<E1, E2, C> RecursiveSNARK<E1, E2, C> where C: StepCircuit<...>`
/// block at line 514 — same generic parameters, additional
/// `LookupStepCircuit<E1>` trait bound on `C`. The existing `prove_step`
/// (line 578-ish) is UNCHANGED and continues to serve vendor
/// upstream-tracking + classical-Nova consumers + STAGE 0 non-lookup
/// path per Corrigendum #13. Step circuits that implement ONLY
/// `StepCircuit<E1::Scalar>` (e.g. `TrivialCircuit`, `CubicCircuit`)
/// continue to compile + run identically.
#[cfg(feature = "lookup-fold")]
impl<E1, E2, C> RecursiveSNARK<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar> + LookupStepCircuit<E1>,
{
  /// Updates the provided `RecursiveSNARK` by executing a step of the
  /// incremental computation under the multi-table lookup-fold path.
  ///
  /// Sibling of `prove_step` per Corrigendum #14 disposition (option β
  /// — separate method, separate impl block, NO modification to
  /// `prove_step`). Routes through `NIFS::prove_with_multi_table_lookup`
  /// instead of `NIFS::prove`, threads per-table running witnesses via
  /// `self.running_lws`, and synthesizes the augmented circuit with
  /// `NeutronAugmentedCircuitInputs::with_multi_table_bundles(Some(public_bundles))`.
  ///
  /// Body, per 2026-05-12 sub-ratification #1 (dropped
  /// `r_logup_per_table` orchestration helper) + sub-ratification #2
  /// (`PublicParams::setup` `lookup_shape` prerequisite):
  ///
  /// 1. Bootstrap at i=0: increment `self.i = 1` and return (mirrors
  ///    `prove_step:580-583` byte-identically).
  /// 2. Construct bundles via `c.per_table_bundles_at_step(&pp.ck,
  ///    self.i, &self.running_lws)`.
  /// 3. Invoke `NIFS::prove_with_multi_table_lookup` against
  ///    `pp.structure` (which carries `Some(LookupShape)` per
  ///    sub-ratification #2; if `None`, fails immediately at
  ///    `nifs.rs:1368` with `NovaError::InvalidStructure`).
  /// 4. Project verifier-public hints via `C::public_bundles(&bundles)`.
  /// 5. Build `NeutronAugmentedCircuitInputs` byte-identical to
  ///    `prove_step:599-624`, then attach
  ///    `.with_multi_table_bundles(Some(public_bundles))`.
  /// 6. Synthesise the augmented circuit byte-identical to
  ///    `prove_step:615-624` (same `with_lookup_fold` threading).
  /// 7. Update `self.{r_U, r_W, l_u, l_w, zi, ri, running_lws, i}`.
  ///
  /// Acceptance test: `m_gh7_3a_prove_step_with_lookup_fold_sibling_*`
  /// in this module's `#[cfg(test)] mod tests`.
  pub fn prove_step_with_lookup_fold(
    &mut self,
    pp: &PublicParams<E1, E2, C>,
    c: &C,
  ) -> Result<(), NovaError> {
    // (1) Bootstrap at i=0 — mirrors `prove_step:580-583` byte-identically.
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    // (2) Construct prover-side per-table bundles via the trait method.
    //     Per sub-ratification #1, NO `r_logup_per_table` argument —
    //     bundles are r_logup-INDEPENDENT (see trait doc).
    let bundles: Vec<PerTableBundle<E1>> =
      c.per_table_bundles_at_step(&pp.ck, self.i, &self.running_lws)?;

    // (3) Invoke `NIFS::prove_with_multi_table_lookup`. The inner
    //     resolves `S.lookups.multi_column_tables` and debug-asserts
    //     `bundles.len() == k` + per-table `table_id` ordering; returns
    //     the new running `(FoldedInstance, FoldedWitness)` AND the
    //     per-table `next_running_lws` Vec for threading into the next
    //     sibling-method invocation. The public wrapper at
    //     `nifs.rs:1301-1316` takes `bundles: &[PerTableBundle<E>]` as
    //     the trailing argument and samples `r_E` from `OsRng`
    //     internally (the deterministic-`r_E` `_inner` variant is
    //     reserved for byte-equivalence tests under the M.GH5.7 audit
    //     bracket; production callers — including this sibling — use
    //     the random-`r_E` public wrapper).
    let (nifs, (r_U, r_W), next_running_lws) = NIFS::prove_with_multi_table_lookup(
      &pp.ck,
      &pp.ro_consts,
      &pp.digest(),
      &pp.structure,
      &self.r_U,
      &self.r_W,
      &self.l_u,
      &self.l_w,
      &bundles,
    )?;

    // (4) Project bundles to verifier-public hints.
    let public_bundles = C::public_bundles(&bundles);

    let r_next = E1::Scalar::random(&mut OsRng);

    // (5) Build `NeutronAugmentedCircuitInputs` byte-identical to
    //     `prove_step:599-624`, then attach multi-table bundles.
    let mut cs = SatisfyingAssignment::<E1>::new();
    let inputs: NeutronAugmentedCircuitInputs<E1> = NeutronAugmentedCircuitInputs::new(
      pp.digest(),
      E1::Scalar::from(self.i as u64),
      self.z0.to_vec(),
      Some(self.zi.clone()),
      Some(self.r_U.clone()),
      Some(self.ri),
      r_next,
      Some(self.l_u.clone()),
      Some(nifs),
      Some(r_U.comm_W),
      Some(r_U.comm_E),
    )
    .with_multi_table_bundles(Some(public_bundles));

    // (6) Synthesise the augmented circuit byte-identical to
    //     `prove_step:615-624`, same `with_lookup_fold` threading.
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(Some(inputs), c, pp.ro_consts_circuit.clone())
        .with_lookup_fold(pp.lookup_fold_k, &pp.shape_registry, pp.index_n_bits);
    let zi = circuit.synthesize(&mut cs)?;

    let (l_u, l_w) = cs.r1cs_instance_and_witness(&pp.structure.S, &pp.ck)?;

    // (7) Update running state. `running_lws` is overwritten from the
    //     `prove_with_multi_table_lookup` `next_running_lws` output —
    //     this is the byte-equal threading contract the acceptance
    //     test verifies.
    self.zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;

    self.r_U = r_U;
    self.r_W = r_W;

    self.i += 1;

    self.ri = r_next;

    self.l_u = l_u;
    self.l_w = l_w;

    self.running_lws = next_running_lws;

    Ok(())
  }

  /// C1-β BIP-340 witness/ck threading sub-corrigendum §4.3 (Halpert,
  /// 2026-05-18): prove-side sibling of
  /// [`Self::prove_step_with_lookup_fold`] that routes the augmented-
  /// circuit synthesize through
  /// [`crate::neutron::circuit::NeutronAugmentedCircuit::synthesize_aux`]
  /// (the lookup-aware sibling per §4.2). Mirrors
  /// `prove_step_with_lookup_fold` byte-for-byte except:
  ///
  ///   1. Where-clause adds `C: StepCircuitWithAux<E1::Scalar, E1>` on
  ///      top of the impl-block's `LookupStepCircuit<E1>` bound.
  ///   2. The augmented-circuit synthesize call at the inumbra-side
  ///      analog of `prove_step_with_lookup_fold:1037` invokes
  ///      `circuit.synthesize_aux(&mut cs)` instead of
  ///      `circuit.synthesize(&mut cs)`, so the inner step-circuit
  ///      body sees a `CSWithLookups<F, SatisfyingAssignment<E1>>`
  ///      wrapper satisfying its `LookupConstraintSystem<F>` bound.
  ///
  /// HG-A1.4-7 (trait-bound mismatch) DISCHARGED on this sibling
  /// landing in conjunction with `synthesize_aux` (§4.2) and
  /// `StepCircuitWithAux` (§4.1).
  pub fn prove_step_with_lookup_fold_aux(
    &mut self,
    pp: &PublicParams<E1, E2, C>,
    c: &C,
  ) -> Result<(), NovaError>
  where
    C: crate::traits::circuit::StepCircuitWithAux<E1::Scalar, E1>,
  {
    // (1) Bootstrap at i=0 — mirrors `prove_step_with_lookup_fold:973-977`.
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    // (2) Construct prover-side per-table bundles via the trait method.
    let bundles: Vec<PerTableBundle<E1>> =
      c.per_table_bundles_at_step(&pp.ck, self.i, &self.running_lws)?;

    // (3) Invoke `NIFS::prove_with_multi_table_lookup`.
    let (nifs, (r_U, r_W), next_running_lws) = NIFS::prove_with_multi_table_lookup(
      &pp.ck,
      &pp.ro_consts,
      &pp.digest(),
      &pp.structure,
      &self.r_U,
      &self.r_W,
      &self.l_u,
      &self.l_w,
      &bundles,
    )?;

    // (4) Project bundles to verifier-public hints.
    let public_bundles = C::public_bundles(&bundles);

    let r_next = E1::Scalar::random(&mut OsRng);

    // (5) Build `NeutronAugmentedCircuitInputs` byte-identical to
    //     `prove_step_with_lookup_fold:1016-1030`.
    let mut cs = SatisfyingAssignment::<E1>::new();
    let inputs: NeutronAugmentedCircuitInputs<E1> = NeutronAugmentedCircuitInputs::new(
      pp.digest(),
      E1::Scalar::from(self.i as u64),
      self.z0.to_vec(),
      Some(self.zi.clone()),
      Some(self.r_U.clone()),
      Some(self.ri),
      r_next,
      Some(self.l_u.clone()),
      Some(nifs),
      Some(r_U.comm_W),
      Some(r_U.comm_E),
    )
    .with_multi_table_bundles(Some(public_bundles));

    // (6) Synthesise the augmented circuit via the lookup-aware
    //     `synthesize_aux` sibling per §4.2 — the only divergence from
    //     `prove_step_with_lookup_fold`.
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(Some(inputs), c, pp.ro_consts_circuit.clone())
        .with_lookup_fold(pp.lookup_fold_k, &pp.shape_registry, pp.index_n_bits);
    let zi = circuit.synthesize_aux(&mut cs)?;

    let (l_u, l_w) = cs.r1cs_instance_and_witness(&pp.structure.S, &pp.ck)?;

    // (7) Update running state — byte-identical to
    //     `prove_step_with_lookup_fold:1045-1062`.
    self.zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;

    self.r_U = r_U;
    self.r_W = r_W;

    self.i += 1;

    self.ri = r_next;

    self.l_u = l_u;
    self.l_w = l_w;

    self.running_lws = next_running_lws;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    provider::{
      pedersen::CommitmentKeyExtTrait, traits::DlogGroup, Bn256EngineIPA, Bn256EngineKZG,
      GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
    },
    traits::{
      circuit::TrivialCircuit,
      snark::{default_ck_hint, RelaxedR1CSSNARKTrait},
    },
    CommitmentEngineTrait,
  };
  use core::{fmt::Write, marker::PhantomData};
  use expect_test::{expect, Expect};
  use ff::PrimeField;

  type EE<E> = crate::provider::ipa_pc::EvaluationEngine<E>;
  type SPrime<E, EE> = crate::spartan::ppsnark::RelaxedR1CSSNARK<E, EE>;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> StepCircuit<F> for CubicCircuit<F> {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = &z[0];
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
      })?;

      cs.enforce(
        || "y = x^3 + x + 5",
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );

      Ok(vec![y])
    }
  }

  impl<F: PrimeField> CubicCircuit<F> {
    fn output(&self, z: &[F]) -> Vec<F> {
      vec![z[0] * z[0] * z[0] + z[0] + F::from(5u64)]
    }
  }

  fn test_pp_digest_with<E1, E2, C>(circuit: &C, expected: &Expect)
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    E1::GE: DlogGroup,
    E2::GE: DlogGroup,
    C: StepCircuit<E1::Scalar>,
    // required to use the IPA in the initialization of the commitment key hints below
    <E1::CE as CommitmentEngineTrait<E1>>::CommitmentKey: CommitmentKeyExtTrait<E1>,
    <E2::CE as CommitmentEngineTrait<E2>>::CommitmentKey: CommitmentKeyExtTrait<E2>,
  {
    // this tests public parameters with a size specifically intended for a spark-compressed SNARK
    let ck_hint1 = &*SPrime::<E1, EE<E1>>::ck_floor();
    let ck_hint2 = &*SPrime::<E2, EE<E2>>::ck_floor();
    // GH-#5 M.GH5.4: vendor-internal `TrivialCircuit` test exercises the
    // default lookup-fold-OFF state (k=0, empty registry); shape and
    // pp_digest are byte-equivalent to pre-M.GH5.4.
    #[cfg(feature = "lookup-fold")]
    let pp =
      PublicParams::<E1, E2, C>::setup(circuit, ck_hint1, ck_hint2, vec![], 0, 0, None).unwrap();
    #[cfg(not(feature = "lookup-fold"))]
    let pp = PublicParams::<E1, E2, C>::setup(circuit, ck_hint1, ck_hint2).unwrap();

    let digest_str = pp
      .digest()
      .to_repr()
      .as_ref()
      .iter()
      .fold(String::new(), |mut output, b| {
        let _ = write!(output, "{b:02x}");
        output
      });
    expected.assert_eq(&digest_str);
  }

  #[test]
  fn test_pp_digest() {
    test_pp_digest_with::<PallasEngine, VestaEngine, _>(
      &TrivialCircuit::<_>::default(),
      &expect!["9901d360368d8e879b071fd6fe5d25134ab60d969c49d1ea94f80175caebbb03"],
    );

    test_pp_digest_with::<Bn256EngineIPA, GrumpkinEngine, _>(
      &TrivialCircuit::<_>::default(),
      &expect!["4519057d1edb4248371b590b38fa369b79cd0c7b3e72822ab9b1036812f10901"],
    );

    test_pp_digest_with::<Secp256k1Engine, Secq256k1Engine, _>(
      &TrivialCircuit::<_>::default(),
      &expect!["823aa5eb33dd9b54da385c52f3de50252c17097367d08f3e266707e38db58b01"],
    );
  }

  fn test_ivc_trivial_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let test_circuit1 = TrivialCircuit::<<E1 as Engine>::Scalar>::default();

    // produce public parameters
    #[cfg(feature = "lookup-fold")]
    let pp = PublicParams::<E1, E2, TrivialCircuit<<E1 as Engine>::Scalar>>::setup(
      &test_circuit1,
      &*default_ck_hint(),
      &*default_ck_hint(),
      vec![],
      0,
      0,
      None,
    )
    .unwrap();
    #[cfg(not(feature = "lookup-fold"))]
    let pp = PublicParams::<E1, E2, TrivialCircuit<<E1 as Engine>::Scalar>>::setup(
      &test_circuit1,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark =
      RecursiveSNARK::new(&pp, &test_circuit1, &[<E1 as Engine>::Scalar::ZERO]).unwrap();

    let res = recursive_snark.prove_step(&pp, &test_circuit1);

    assert!(res.is_ok(), "prove_step failed: {:?}", res.err());

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ZERO]);
    assert!(res.is_ok(), "verify failed: {:?}", res.err());
  }

  #[test]
  fn test_ivc_trivial() {
    test_ivc_trivial_with::<PallasEngine, VestaEngine>();
    test_ivc_trivial_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ivc_trivial_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_ivc_nontrivial_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let circuit = CubicCircuit::default();

    // produce public parameters
    #[cfg(feature = "lookup-fold")]
    let pp = PublicParams::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
      vec![],
      0,
      0,
      None,
    )
    .unwrap();
    #[cfg(not(feature = "lookup-fold"))]
    let pp = PublicParams::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::new(
      &pp,
      &circuit,
      &[<E1 as Engine>::Scalar::ONE],
    )
    .unwrap();

    for i in 0..num_steps {
      let res = recursive_snark.prove_step(&pp, &circuit);
      assert!(res.is_ok());

      // verify the recursive snark at each step of recursion
      let res = recursive_snark.verify(&pp, i + 1, &[<E1 as Engine>::Scalar::ONE]);
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ONE]);
    assert!(res.is_ok());

    let zn = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    let mut zn_direct = vec![<E1 as Engine>::Scalar::ONE];
    for _i in 0..num_steps {
      zn_direct = circuit.clone().output(&zn_direct);
    }
    assert_eq!(zn, zn_direct);
    assert_eq!(zn, vec![<E1 as Engine>::Scalar::from(0x2aaaaa3u64)]);
  }

  #[test]
  fn test_ivc_nontrivial_neutron() {
    test_ivc_nontrivial_with::<PallasEngine, VestaEngine>();
    test_ivc_nontrivial_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ivc_nontrivial_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_ivc_base_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let test_circuit1 = CubicCircuit::<<E1 as Engine>::Scalar>::default();

    // produce public parameters
    #[cfg(feature = "lookup-fold")]
    let pp = PublicParams::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::setup(
      &test_circuit1,
      &*default_ck_hint(),
      &*default_ck_hint(),
      vec![],
      0,
      0,
      None,
    )
    .unwrap();
    #[cfg(not(feature = "lookup-fold"))]
    let pp = PublicParams::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::setup(
      &test_circuit1,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::new(
      &pp,
      &test_circuit1,
      &[<E1 as Engine>::Scalar::ONE],
    )
    .unwrap();

    // produce a recursive SNARK
    let res = recursive_snark.prove_step(&pp, &test_circuit1);

    assert!(res.is_ok());

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ONE]);
    assert!(res.is_ok());

    let zn = res.unwrap();

    assert_eq!(zn, vec![<E1 as Engine>::Scalar::from(7u64)]);
  }

  #[test]
  fn test_ivc_base() {
    test_ivc_base_with::<PallasEngine, VestaEngine>();
    test_ivc_base_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ivc_base_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_setup_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    #[derive(Clone, Debug, Default)]
    struct CircuitWithInputize<F: PrimeField> {
      _p: PhantomData<F>,
    }

    impl<F: PrimeField> StepCircuit<F> for CircuitWithInputize<F> {
      fn arity(&self) -> usize {
        1
      }

      fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
      ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        let x = &z[0];
        let y = x.square(cs.namespace(|| "x_sq"))?;
        y.inputize(cs.namespace(|| "y"))?; // inputize y
        Ok(vec![y])
      }
    }

    // produce public parameters with trivial secondary
    let circuit = CircuitWithInputize::<<E1 as Engine>::Scalar>::default();
    #[cfg(feature = "lookup-fold")]
    let pp = PublicParams::<E1, E2, CircuitWithInputize<E1::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
      vec![],
      0,
      0,
      None,
    );
    #[cfg(not(feature = "lookup-fold"))]
    let pp = PublicParams::<E1, E2, CircuitWithInputize<E1::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );
    assert!(pp.is_err());
    assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));

    // produce public parameters with the trivial primary
    let circuit = CircuitWithInputize::<E1::Scalar>::default();
    #[cfg(feature = "lookup-fold")]
    let pp = PublicParams::<E1, E2, CircuitWithInputize<E1::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
      vec![],
      0,
      0,
      None,
    );
    #[cfg(not(feature = "lookup-fold"))]
    let pp = PublicParams::<E1, E2, CircuitWithInputize<E1::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );
    assert!(pp.is_err());
    assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));
  }

  #[test]
  fn test_setup() {
    test_setup_with::<Bn256EngineKZG, GrumpkinEngine>();
  }

  /// GH-#7 M.GH7.3a positive-path acceptance test (Corrigendum #14
  /// re-typed; sub-ratifications #1 + #2 2026-05-12; 2026-05-12 Option α
  /// split: negative-test triple deferred to M.GH7.3b).
  ///
  /// **Scope**: round-trip `RecursiveSNARK::prove_step_with_lookup_fold`
  /// across N ≥ 2 honest sibling-method invocations against a trivial
  /// `LookupStepCircuit` fixture mirroring the M.11 §D.1 absent-table
  /// pattern at vendor-test-fixture scope (per
  /// `crates/inumbra-spend-harness/tests/m11_fold_of_four_absent_table_differential.rs:466-525`,
  /// reproduced inline here so the vendor-side test does NOT depend on
  /// inumbra-side `build_canonical_lookup_shape` per the pin's
  /// STOP-AND-ASK on inumbra-side dependency).
  ///
  /// **Assertions**:
  ///
  /// 1. `prove_step_with_lookup_fold` returns `Ok` for each of N=3
  ///    honest invocations (bootstrap at i=0; first fold at i=1→2;
  ///    second fold at i=2→3). N=3 is the minimum for `running_lws`
  ///    threading observability: invocation #2 produces
  ///    `next_running_lws_after_fold_1`; invocation #3 consumes
  ///    `&self.running_lws == next_running_lws_after_fold_1` as
  ///    `prior_running_lws` input (the byte-equal threading assertion).
  /// 2. `r_U.T_lookup` accumulates per §1.2 algebra and Corrigendum #25
  ///    Fix-α: `Some(vec![ZERO; k])` at outer base (after invocation #1,
  ///    bootstrap-only; the Fix-α native-side mirror initializes the
  ///    Some-block at `RecursiveSNARK::new` for IVC-hash-chain
  ///    byte-equivalence with `AllocatedFoldedInstance::default_with_lookup_k`),
  ///    `Some(vec)` of length k with non-zero entries after invocation #2
  ///    (first real fold accumulates the per-table running scalar from
  ///    the inner `prove_with_multi_table_lookup`), still `Some(vec)`
  ///    after invocation #3 (second fold advances each per-table scalar).
  /// 3. `running_lws[j]` next-step input is byte-equal to the prior
  ///    step's `next_running_lws[j]` output — captured via the trait
  ///    method's `prior_running_lws` parameter and asserted against a
  ///    snapshot taken at the prior invocation's exit.
  /// 4. `prove_step` (non-lookup) continues to compile + run
  ///    byte-identical — captured by the existing vendor tests in
  ///    this module passing unchanged (`test_pp_digest`,
  ///    `test_ivc_trivial`, `test_ivc_nontrivial_neutron`,
  ///    `test_ivc_base`, `test_setup`), which exercise the
  ///    `cfg(feature = "lookup-fold")` path at `(vec![], 0, 0, None)`
  ///    (the non-lookup-fold-active default state) without invoking
  ///    `prove_step_with_lookup_fold`.
  ///
  /// **Out of scope** (deferred to M.GH7.3b per 2026-05-12 Option α
  /// split): negative-test triple discharging `.claude/rules/cryptography.md`
  /// §70-78 Constraint hygiene obligation for the new public prover
  /// API. Halpert authoring is the gate.
  ///
  /// **Note on M.7 shape-registry assertion satisfiability**: the
  /// augmented-circuit synthesis under `lookup_fold_k > 0` allocates
  /// constraints from `assert_pp_digest_matches_registry`
  /// (`shape_registry/circuit.rs:83`), which enforces `pp_digest ==
  /// shape_registry[chunk_index_in_z]`. This fixture supplies
  /// `shape_registry = vec![Scalar::ZERO]` and the step circuit
  /// outputs `z_next[F_arity-1] = ZERO`, so the in-circuit lookup
  /// resolves to `shape_registry[0] = ZERO` which does NOT equal
  /// `pp.digest()` (a non-zero hash output). The synthesis itself
  /// succeeds — `r1cs_instance_and_witness` extracts the witness
  /// without checking constraint satisfiability — and
  /// `prove_step_with_lookup_fold` returns `Ok` per the pin contract.
  /// The IVC final verify is NOT invoked in this test (M.7 assertion
  /// unsat would fail `pp.structure.is_sat`, which is a soundness
  /// check, not a sibling-method liveness check). The pin's
  /// observable assertions (T_lookup accumulation + running_lws byte-
  /// equality) are about prover-side state mutations, observable
  /// independently of in-circuit constraint satisfiability.
  /// Verify-side soundness for the sibling-method is M.GH7.3b's scope
  /// (negative-test triple) and M.GH7.5's scope
  /// (compress-and-verify-rejects).
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn m_gh7_3a_prove_step_with_lookup_fold_sibling_threads_running_lws_witness_alloc_preserved_positive_path()
  {
    use crate::neutron::nifs::PerTableBundle;
    use crate::neutron::relation::{
      LookupPayload, LookupPayloadPublicMultiTable, LookupRunningWitness, LookupShape,
      LookupTableHandle, MultiColumnLookupTable,
    };
    use crate::traits::commitment::CommitmentEngineTrait;
    use crate::Commitment;
    use std::sync::{Arc, Mutex};

    type E1Eng = Bn256EngineKZG;
    type E2Eng = GrumpkinEngine;
    type Scalar = <E1Eng as Engine>::Scalar;

    // ── (A) Trivial `LookupStepCircuit` fixture ────────────────────
    //
    // `step_circuit` shape: arity 1, `synthesize` is the upstream
    // `TrivialCircuit` body (no constraints, `z_next = z`). Per pin
    // §3.2 + Corrigendum #4 the augmented-circuit reads
    // `chunk_index_in_z = z_i[F_arity-1] = z_i[0]`. With initial
    // `z0 = vec![Scalar::ZERO]` the trivial circuit propagates
    // `Scalar::ZERO` forward, so chunk_index_in_z is always 0; we
    // configure `shape_registry = vec![Scalar::ZERO]` (length 1,
    // index_n_bits = 1).
    //
    // The `LookupStepCircuit` impl produces absent-table bundles
    // (zero address, zero multiplicities, zero value columns — the
    // single-column-degenerate path per `nifs.rs:57-70` /
    // `m11_fold_of_four_absent_table_differential.rs:458-525`). The
    // (C)-binding holds tautologically (Σ inv_w − Σ inv_t = 0) and
    // the running threading is observable byte-equal per the
    // assertion.
    //
    // **`prior_running_lws` snapshot**: captured via `RefCell` so the
    // test can assert the byte-equal threading without coupling to
    // sibling-method internals.
    const TABLE_SIZE: usize = 4; // log2 = 2; w_left = 2, w_right = 2.
    const TABLE_LOG2: usize = 2;
    const K: usize = 1; // single-table lookup-fold-active path.

    type SnapshotCell = Arc<Mutex<Option<Vec<LookupRunningWitness<E1Eng>>>>>;

    #[derive(Clone)]
    struct AbsentTableStepCircuit {
      // Captured at the start of each `per_table_bundles_at_step` call,
      // so the test can assert byte-equality across step boundaries.
      // `Arc<Mutex<>>` for `Send + Sync` per the `StepCircuit` trait
      // bound at `traits/circuit.rs:7`.
      observed_prior_running_lws: SnapshotCell,
    }

    impl AbsentTableStepCircuit {
      fn new() -> Self {
        Self {
          observed_prior_running_lws: Arc::new(Mutex::new(None)),
        }
      }
    }

    impl StepCircuit<Scalar> for AbsentTableStepCircuit {
      fn arity(&self) -> usize {
        1
      }

      fn synthesize<CS: ConstraintSystem<Scalar>>(
        &self,
        _cs: &mut CS,
        z: &[AllocatedNum<Scalar>],
      ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
        // TrivialCircuit body: z_next = z. The `z[0]` allocation
        // propagates forward as `z_next[0]`, which the augmented
        // circuit reads as `chunk_index_in_z`. At `z0[0] = Scalar::ZERO`
        // (test setup below) this is always zero across all steps,
        // matching `shape_registry[0]`.
        Ok(z.to_vec())
      }
    }

    impl LookupStepCircuit<E1Eng> for AbsentTableStepCircuit {
      fn per_table_bundles_at_step(
        &self,
        ck: &CommitmentKey<E1Eng>,
        _i: usize,
        prior_running_lws: &[LookupRunningWitness<E1Eng>],
      ) -> Result<Vec<PerTableBundle<E1Eng>>, NovaError> {
        // Snapshot `prior_running_lws` so the test can assert byte-
        // equality with the prior invocation's `next_running_lws`
        // output across step boundaries.
        *self
          .observed_prior_running_lws
          .lock()
          .expect("observed_prior_running_lws mutex must not be poisoned") =
          Some(prior_running_lws.to_vec());

        // Build a single absent-table bundle per the M.11 §D.1 pattern.
        let zero_addr = vec![Scalar::ZERO; TABLE_SIZE];
        let zero_mult = vec![Scalar::ZERO; TABLE_SIZE];
        let comm_addr = <E1Eng as Engine>::CE::commit(ck, &zero_addr, &Scalar::ZERO);
        let comm_ts = <E1Eng as Engine>::CE::commit(ck, &zero_mult, &Scalar::ZERO);

        // Single-column-degenerate path: `comm_values` empty.
        let payload = LookupPayload::<E1Eng> {
          comm_L: comm_addr,
          comm_ts,
          comm_inv_w: Commitment::<E1Eng>::default(),
          comm_inv_t: Commitment::<E1Eng>::default(),
          T2_lookup: Scalar::ZERO,
          comm_values: vec![],
        };

        // Per-table eq vector half-lengths: w_left = w_right = 2 (TABLE_LOG2 = 2).
        let ell1 = TABLE_LOG2.div_ceil(2);
        let ell2 = TABLE_LOG2 / 2;
        let w_left = 1usize << ell1;
        let w_right = 1usize << ell2;

        let running_lw = prior_running_lws
          .first()
          .cloned()
          .expect("prior_running_lws must carry K=1 entry per RecursiveSNARK::new bootstrap");

        Ok(vec![PerTableBundle::<E1Eng> {
          table_id: 0,
          payload,
          fresh_witness_address: zero_addr,
          fresh_witness_value_columns: vec![],
          fresh_multiplicities: zero_mult,
          fresh_eq_w_left: vec![Scalar::ZERO; w_left],
          fresh_eq_w_right: vec![Scalar::ZERO; w_right],
          fresh_eq_t_left: vec![Scalar::ZERO; w_left],
          fresh_eq_t_right: vec![Scalar::ZERO; w_right],
          running_lw,
        }])
      }

      fn public_bundles(bundles: &[PerTableBundle<E1Eng>]) -> Vec<LookupPayloadPublicMultiTable<E1Eng>>
      {
        bundles
          .iter()
          .map(|b| LookupPayloadPublicMultiTable::<E1Eng> {
            table_id: b.table_id,
            comm_L: b.payload.comm_L,
            comm_values: b.payload.comm_values.clone(),
            comm_ts: b.payload.comm_ts,
          })
          .collect()
      }
    }

    // ── (B) Build LookupShape + PublicParams ────────────────────────
    //
    // Single-table shape; the table has zero data columns
    // (single-column-degenerate path). `value_commitments` is empty so
    // `payload.comm_values.len() == multi_table.columns.len() == 0` per
    // the bundle-validation invariant at `nifs.rs:1404-1411`.
    //
    // The `tables[0]` `LookupTableHandle` carries an arbitrary
    // identity-vector commitment (the bundle-validation does not check
    // `tables[*]` content; it only checks `multi_column_tables[*]`).
    let circuit = AbsentTableStepCircuit::new();

    // Build a placeholder commitment key (will be reused via `pp.ck`).
    // The fixture uses an identity-vector commitment for the
    // `LookupTableHandle.commitment` field — pinned to a deterministic
    // value to keep `pp_digest` stable across test invocations (no RNG
    // here per `.claude/rules/cryptography.md` determinism contract).
    let identity: Vec<Scalar> =
      (0..TABLE_SIZE).map(|i| Scalar::from(i as u64)).collect();
    // We need a commitment key with at least `TABLE_SIZE` generators.
    // The vendor's `PublicParams::setup` derives `ck` from the
    // augmented-circuit R1CS shape; we pre-build a small key here ONLY
    // to commit to the table identity for the `LookupTableHandle`. The
    // real `ck` used by the prover comes from `pp.ck` after `setup`.
    //
    // For the M.11-style absent-table fixture, the identity commitment
    // is bound into `pp_digest` via the `Structure → LookupShape →
    // tables[0]` serde path. We compute it against a fresh small key
    // sized to `TABLE_SIZE`.
    let identity_ck: CommitmentKey<E1Eng> = <E1Eng as Engine>::CE::setup(
      b"M.GH7.3a/test/identity-ck",
      TABLE_SIZE,
    )
    .expect("CE::setup must produce an identity-vector commitment key at TABLE_SIZE");
    let identity_comm = <E1Eng as Engine>::CE::commit(&identity_ck, &identity, &Scalar::ZERO);

    let lookup_shape = LookupShape::<E1Eng> {
      tables: vec![LookupTableHandle {
        table_id: 0,
        size: TABLE_SIZE,
        commitment: identity_comm,
      }],
      multi_column_tables: vec![MultiColumnLookupTable {
        table_id: 0,
        size: TABLE_SIZE,
        columns: vec![],
        value_commitments: vec![],
      }],
      num_addr_columns: 1,
      num_witness_columns: 1,
      witness_ell_cached: TABLE_LOG2,
    };

    let shape_registry = vec![Scalar::ZERO];

    let pp = PublicParams::<E1Eng, E2Eng, AbsentTableStepCircuit>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
      shape_registry,
      K,
      1, // index_n_bits = 1 (single-entry registry).
      Some(lookup_shape.clone()),
    )
    .expect("PublicParams::setup with Some(LookupShape) must succeed");

    // Sanity: structure carries lookups; lookup_fold_k threaded through.
    assert!(
      pp.structure.lookups.is_some(),
      "sub-ratification #2: Some(lookup_shape) routes through Structure::new_with_lookups"
    );
    assert_eq!(
      pp.structure
        .lookups
        .as_ref()
        .unwrap()
        .multi_column_tables
        .len(),
      K,
      "structure.lookups.multi_column_tables.len() == K"
    );

    // ── (C) RecursiveSNARK::new + bootstrap ─────────────────────────
    let z0 = vec![Scalar::ZERO];
    let mut recursive_snark = RecursiveSNARK::<E1Eng, E2Eng, AbsentTableStepCircuit>::new(
      &pp, &circuit, &z0,
    )
    .expect("RecursiveSNARK::new succeeds on AbsentTableStepCircuit");

    // Sanity on `running_lws` bootstrap (sub-ratification #2):
    // K=1 default running_lw at outer base.
    assert_eq!(
      recursive_snark.running_lws.len(),
      K,
      "RecursiveSNARK::new bootstraps running_lws to length K under pp.lookup_fold_k > 0"
    );

    // Capture the bootstrap `running_lws` snapshot for comparison
    // after invocation #1 (which is the bootstrap-only `if self.i == 0`
    // branch and should NOT mutate running_lws).
    let bootstrap_running_lws = recursive_snark.running_lws.clone();

    // ── (D) Invocation #1: bootstrap (i=0) ──────────────────────────
    //
    // The bootstrap branch increments `self.i` to 1 and returns Ok
    // without invoking the trait method. `running_lws` is unmutated.
    recursive_snark
      .prove_step_with_lookup_fold(&pp, &circuit)
      .expect("invocation #1 (bootstrap) returns Ok");

    assert_eq!(recursive_snark.i, 1, "bootstrap branch increments i to 1");
    assert_eq!(
      recursive_snark.running_lws, bootstrap_running_lws,
      "bootstrap branch does NOT mutate running_lws"
    );
    assert!(
      circuit
        .observed_prior_running_lws
        .lock()
        .expect("mutex must not be poisoned")
        .is_none(),
      "bootstrap branch does NOT invoke per_table_bundles_at_step"
    );
    // Corrigendum #25 (Fix-α native-side mirror): at outer base under
    // `pp.lookup_fold_k > 0`, `r_U.T_lookup` is initialized by
    // `FoldedInstance::default_with_lookup_k` to `Some(vec![ZERO; k])` so
    // the native-side `absorb_in_ro2` walks the same Some-block the
    // circuit-side `absorb_in_ro` emits at base case (closing Obstruction 2).
    // The bootstrap branch of `prove_step_with_lookup_fold` does NOT mutate
    // `r_U`, so the post-bootstrap state matches the post-`new` state.
    let t_lookup_at_bootstrap = recursive_snark
      .r_U
      .T_lookup
      .as_ref()
      .expect("Corrigendum #25 Fix-α: r_U.T_lookup is Some(vec![ZERO; k]) at outer base under lookup_fold_k > 0");
    assert_eq!(
      t_lookup_at_bootstrap.len(),
      K,
      "Fix-α: r_U.T_lookup length is K at outer base, mirroring `running_lws` length-K invariant"
    );
    assert!(
      t_lookup_at_bootstrap.iter().all(|s| s.is_zero().into()),
      "Fix-α: r_U.T_lookup entries are all-zero at outer base (no fold yet)"
    );

    // ── (E) Invocation #2: first real fold (i=1 → 2) ────────────────
    //
    // The non-bootstrap branch invokes per_table_bundles_at_step,
    // observes `prior_running_lws == bootstrap_running_lws`, then folds.
    // After this, `running_lws` is overwritten with
    // `next_running_lws_after_fold_1`, and `r_U.T_lookup` becomes
    // `Some(Vec)` of length K.
    recursive_snark
      .prove_step_with_lookup_fold(&pp, &circuit)
      .expect("invocation #2 (first real fold) returns Ok");

    assert_eq!(
      recursive_snark.i, 2,
      "invocation #2 advances i from 1 to 2"
    );

    // Byte-equality assertion #1: the trait method observed
    // `prior_running_lws == bootstrap_running_lws` (because invocation
    // #1 did not mutate).
    let observed_at_step_2 = circuit
      .observed_prior_running_lws
      .lock()
      .expect("mutex must not be poisoned")
      .clone()
      .expect("per_table_bundles_at_step was invoked at step 2");
    assert_eq!(
      observed_at_step_2, bootstrap_running_lws,
      "step 2's `prior_running_lws` input is byte-equal to the bootstrap state \
       (no mutation across the bootstrap branch)"
    );

    // §1.2 algebra: T_lookup accumulates from None to Some(Vec) of length K.
    let t_lookup_after_fold_1 = recursive_snark
      .r_U
      .T_lookup
      .as_ref()
      .expect("first real fold populates r_U.T_lookup");
    assert_eq!(
      t_lookup_after_fold_1.len(),
      K,
      "T_lookup is per-table of length K=multi_column_tables.len()"
    );

    // Snapshot `running_lws` after fold #1 for the byte-equal threading
    // assertion at fold #2.
    let next_running_lws_after_fold_1 = recursive_snark.running_lws.clone();
    assert_eq!(
      next_running_lws_after_fold_1.len(),
      K,
      "running_lws length stays at K after fold #1"
    );

    // The sibling-method MUST have mutated running_lws (the fold
    // produced a non-trivial `next_running_lws` per inner's §1.2 algebra).
    // For the absent-table fixture this is non-trivial because
    // `LookupSumcheckInstance::new` computes `inv_w`/`inv_t` internally
    // from `r_logup_j` squeezed at Step 5b, then folds against the
    // running state at weight `r_b`.
    assert_ne!(
      next_running_lws_after_fold_1, bootstrap_running_lws,
      "fold #1 mutates running_lws away from the all-zero bootstrap state \
       (inv_w / inv_t computed by LookupSumcheckInstance::new for non-zero r_logup_j)"
    );

    // ── (F) Invocation #3: second real fold (i=2 → 3) ───────────────
    //
    // The trait method is invoked again. It observes
    // `prior_running_lws == next_running_lws_after_fold_1` — this is
    // the load-bearing byte-equal threading assertion: the field
    // mutation at the end of invocation #2 (line `self.running_lws =
    // next_running_lws;` in `prove_step_with_lookup_fold`) is
    // observable as the trait method's input at the next invocation.
    recursive_snark
      .prove_step_with_lookup_fold(&pp, &circuit)
      .expect("invocation #3 (second real fold) returns Ok");

    assert_eq!(
      recursive_snark.i, 3,
      "invocation #3 advances i from 2 to 3"
    );

    // **Load-bearing assertion**: byte-equal threading.
    let observed_at_step_3 = circuit
      .observed_prior_running_lws
      .lock()
      .expect("mutex must not be poisoned")
      .clone()
      .expect("per_table_bundles_at_step was invoked at step 3");
    assert_eq!(
      observed_at_step_3, next_running_lws_after_fold_1,
      "step 3's `prior_running_lws` input is byte-equal to step 2's \
       `next_running_lws` output — the sibling-method threads the field \
       through `self.running_lws = next_running_lws;`"
    );

    // §1.2 algebra: T_lookup accumulated past the first fold; the
    // second fold produces a fresh per-table running scalar (under
    // random r_b from `prove_with_multi_table_lookup` the second
    // fold's T_lookup is in general distinct from fold #1's).
    let t_lookup_after_fold_2 = recursive_snark
      .r_U
      .T_lookup
      .as_ref()
      .expect("second real fold preserves Some(T_lookup)");
    assert_eq!(
      t_lookup_after_fold_2.len(),
      K,
      "T_lookup length stable at K across folds"
    );

    // ── (G) Regression invariant: prove_step continues to compile + run
    //         byte-identical against the pre-Corrigendum-#14 vendor ──
    //
    // Captured by the existing vendor tests in this module: the
    // `test_pp_digest_with` callsites at `mod.rs:830-841` pass
    // unchanged under the `(vec![], 0, 0, None)` argument shape; the
    // `test_ivc_*` tests exercise `RecursiveSNARK::prove_step` (NOT
    // the new sibling) and continue to pass per the pre-
    // sub-ratification-#2 baseline. No additional in-test assertion
    // here — the regression is in the test suite as a whole.
  }

  // ===========================================================================
  // M.GH7.3b — negative-test triple
  //
  // Discharges `.claude/rules/cryptography.md` §70-78 (Constraint hygiene —
  // "a missing constraint is a forgery vector") for the M.GH7.3a public prover
  // API surface (`RecursiveSNARK::prove_step_with_lookup_fold` sibling +
  // `LookupStepCircuit<E>` trait + `RecursiveSNARK::running_lws` + the
  // `PublicParams::setup` `lookup_shape` extension), per Corrigendum #15 +
  // 2026-05-12 late-evening Sub-ratification in
  // `docs/research/cryptography/gh-7-stage-k-compressed-snark-design-pin-2026-05-11.md`
  // §0 entry 15 + §3.1.1 + §5.5 row M.GH7.3b.
  //
  // Sub-ratification disposition (fifth Halpert-class verify-don't-assume
  // failure named): Corrigendum #15's original test (i) corruption target
  // `bundles[j].running_lw = LookupRunningWitness::default(&shape)` is
  // structurally inadequate at the (C)-binding rejection mechanism because
  // `LookupSumcheckInstance::prove_step` at `vendor/nova/src/neutron/lookup_sumcheck.rs:278-285`
  // sets `evals[1] := t_lookup_running - evals[0]` BY CONSTRUCTION, making
  // `poly_lookup_j(0) + poly_lookup_j(1) = t_lookup_running_j` a TAUTOLOGY
  // for any honestly-produced polynomial regardless of U1-side internal state
  // corruption. The sub-ratification re-targets test (i) to the R1CS-side
  // (C)-binding (a structurally-distinct rejection mechanism in the same
  // `verify_with_multi_table_lookup` body, returning the same
  // `NovaError::InvalidSumcheckProof` variant). Test (ii)'s mutation shape is
  // amended from multiplicative `× 2` (preserves the (C)-binding tautology
  // under all-zero `T_lookup_running = 0`) to additive `coeffs[0] += ONE`
  // (breaks the (C)-binding equation in any field of characteristic ≠ 2;
  // BN254 scalar field characteristic is odd, verified). Test (iii) STANDS
  // verbatim modulo function-name suffix.
  //
  // Implementation pattern: all three tests follow the M.14 precedent at
  // `vendor/nova/src/neutron/nifs.rs:5719-6149`
  // (`m14_cross_table_cancellation_hard_rejects`): direct
  // `NIFS::prove_with_multi_table_lookup` invocation at vendor-test scope.
  // The sibling-method `prove_step_with_lookup_fold` consumes its produced
  // NIFS inline at `mod.rs:985` (verified at vendor HEAD `e3fe75d`;
  // `RecursiveSNARK` has no `nifs` field at `:520-569`) so post-prove NIFS
  // extraction from `RecursiveSNARK` state is structurally impossible — the
  // direct-NIFS-at-vendor-test-scope pattern is the structurally-valid
  // discipline that the (C)-binding-rejection detection mechanism for the
  // M.GH7.3a-introduced public prover API surfaces through. The (C)-binding
  // algebra at `nifs.rs:2000-2001` (R1CS-side) and `:2025-2028` (lookup-side)
  // is the SAME path `prove_step_with_lookup_fold` invokes internally;
  // M.GH7.3a's positive-path acceptance test at `:1466-1900` IS the
  // empirical-close that the sibling-method routes through the same NIFS
  // construction.
  //
  // Fixture: AbsentTableStepCircuit shape inherited from M.GH7.3a positive
  // path (K=1, TABLE_SIZE=4, TABLE_LOG2=2, Bn256EngineKZG + GrumpkinEngine).
  // Per Sub-ratification fixture pin, no fixture amendment required — the
  // three algebraic preconditions (BN254 scalar field characteristic odd;
  // rho ≠ ONE w.o.p. under RO-squeeze; prime field of size ≥ 2) hold under
  // this fixture. R1CS shape uses `DirectCircuit<E, NonTrivialCircuit>` per
  // M.14 precedent — what matters for the (C)-binding rejection is the
  // bundle/lookup-side construction, not the R1CS body content.
  //
  // Deterministic seeds (US-05 reviewer-reproducibility):
  //   SEED_M_GH7_3B_TEST_I  = 0xC1BE_5BAD_C0DE_7031
  //   SEED_M_GH7_3B_TEST_II = 0xC1BE_5BAD_C0DE_7032
  //   SEED_M_GH7_3B_TEST_III= 0xC1BE_5BAD_C0DE_7033
  // ===========================================================================

  /// Shared honest-baseline fixture builder for the M.GH7.3b negative-test
  /// triple. Mirrors the M.14 precedent at `nifs.rs:5719-6149` but with the
  /// AbsentTableStepCircuit lookup-shape per M.GH7.3a (`K = 1`, `TABLE_SIZE = 4`,
  /// `TABLE_LOG2 = 2`, empty `multi_column_tables` columns).
  ///
  /// Runs two honest direct `NIFS::prove_with_multi_table_lookup` folds:
  ///   - Fold #1: `default(&str_local) → (nifs1, (folded_U_s1, folded_W_s1), lws_s1)`
  ///   - Fold #2: `folded_U_s1 → (nifs2, _, _)` with `bundle_s2.running_lw =
  ///     lws_s1[0]`
  ///
  /// The honest fold #2 verify must accept (positive control, asserted
  /// before the per-test corruption pattern fires). The fixture's
  /// `t_lookup_running_j` at fold #2's verifier-input boundary
  /// (`folded_U_s1.T_lookup[0]`) is generically non-trivial after fold #1
  /// (per M.GH7.3a's `assert_ne!(next_running_lws_after_fold_1,
  /// bootstrap_running_lws, ...)` at `mod.rs:1802-1806`).
  #[cfg(feature = "lookup-fold")]
  fn m_gh7_3b_build_honest_two_step_fixture(seed: u64) -> Bn256EngineKZGFixture {
    use crate::frontend::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      Circuit,
    };
    use crate::neutron::nifs::PerTableBundle;
    use crate::neutron::relation::{
      LookupPayload, LookupPayloadPublicMultiTable, LookupRunningWitness, LookupShape,
      LookupTableHandle, MultiColumnLookupTable,
    };
    use crate::neutron::{FoldedInstance, FoldedWitness, Structure, NIFS};
    use crate::provider::hyperkzg::EvaluationEngine as HyperKZGEE;
    use crate::r1cs::R1CSShape;
    use crate::spartan::{direct::DirectCircuit, snark::RelaxedR1CSSNARK};
    use crate::traits::{
      circuit::NonTrivialCircuit, commitment::CommitmentEngineTrait,
      snark::RelaxedR1CSSNARKTrait, RO2Constants,
    };
    use crate::Commitment;
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    type E1Eng = Bn256EngineKZG;
    type Scalar = <E1Eng as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E1Eng, HyperKZGEE<E1Eng>>;

    const TABLE_SIZE: usize = 4;
    const TABLE_LOG2: usize = 2;

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let ro_consts = RO2Constants::<E1Eng>::default();
    let pp_digest = Scalar::ZERO;

    // R1CS shape: NonTrivialCircuit per M.14 precedent at `nifs.rs:5736-5742`.
    // The body content is immaterial to the (C)-binding rejection mechanism;
    // what matters is that we have a satisfying R1CS instance-witness pair to
    // feed `prove_with_multi_table_lookup` so the prover does not reject at
    // a pre-(C)-binding structural gate.
    let num_cons = 32usize;
    let circuit_shape: DirectCircuit<E1Eng, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E1Eng> = ShapeCS::new();
    let _ = circuit_shape.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // AbsentTableStepCircuit-style lookup shape: single table, empty value
    // columns (single-column-degenerate path per `nifs.rs:57-70`). The
    // `tables[0]` `LookupTableHandle.commitment` is an arbitrary identity
    // commitment to keep `pp_digest` deterministic across invocations.
    let identity: Vec<Scalar> = (0..TABLE_SIZE).map(|i| Scalar::from(i as u64)).collect();
    let lookup_shape = LookupShape::<E1Eng> {
      tables: vec![LookupTableHandle {
        table_id: 0,
        size: TABLE_SIZE,
        commitment: <E1Eng as Engine>::CE::commit(&ck, &identity, &Scalar::ZERO),
      }],
      multi_column_tables: vec![MultiColumnLookupTable {
        table_id: 0,
        size: TABLE_SIZE,
        columns: vec![],
        value_commitments: vec![],
      }],
      num_addr_columns: 1,
      num_witness_columns: 1,
      witness_ell_cached: TABLE_LOG2,
    };
    let str_local = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str_local.S.clone();

    // Two satisfying R1CS instance-witness pairs (mirrors M.14 at
    // `nifs.rs:5787-5798`).
    let make_r1cs = |x: u64| {
      let circuit: DirectCircuit<E1Eng, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
        Some(vec![Scalar::from(x)]),
        NonTrivialCircuit::<Scalar>::new(num_cons),
      );
      let mut cs = SatisfyingAssignment::<E1Eng>::new();
      let _ = circuit.synthesize(&mut cs);
      let (u, w) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
      (u, w.pad(&shape))
    };
    let (u_step1, w_step1) = make_r1cs(2);
    let (u_step2, w_step2) = make_r1cs(3);

    // Per-table eq vector half-lengths: TABLE_LOG2 = 2 → ell1 = ell2 = 1 →
    // w_left = w_right = 2.
    let ell1 = TABLE_LOG2.div_ceil(2);
    let ell2 = TABLE_LOG2 / 2;
    let w_left = 1usize << ell1;
    let w_right = 1usize << ell2;

    // Absent-table bundle: zero address, zero multiplicities, no value
    // columns (mirrors M.GH7.3a at `mod.rs:1561-1599`).
    let mk_absent_bundle = |running_lw: LookupRunningWitness<E1Eng>| -> PerTableBundle<E1Eng> {
      let zero_addr = vec![Scalar::ZERO; TABLE_SIZE];
      let zero_mult = vec![Scalar::ZERO; TABLE_SIZE];
      let payload = LookupPayload::<E1Eng> {
        comm_L: <E1Eng as Engine>::CE::commit(&ck, &zero_addr, &Scalar::ZERO),
        comm_ts: <E1Eng as Engine>::CE::commit(&ck, &zero_mult, &Scalar::ZERO),
        comm_inv_w: Commitment::<E1Eng>::default(),
        comm_inv_t: Commitment::<E1Eng>::default(),
        T2_lookup: Scalar::ZERO,
        comm_values: vec![],
      };
      PerTableBundle::<E1Eng> {
        table_id: 0,
        payload,
        fresh_witness_address: zero_addr,
        fresh_witness_value_columns: vec![],
        fresh_multiplicities: zero_mult,
        fresh_eq_w_left: vec![Scalar::ZERO; w_left],
        fresh_eq_w_right: vec![Scalar::ZERO; w_right],
        fresh_eq_t_left: vec![Scalar::ZERO; w_left],
        fresh_eq_t_right: vec![Scalar::ZERO; w_right],
        running_lw,
      }
    };

    // === Fold #1: honest prove (outer base → fold #1). ===
    let running_W_default = FoldedWitness::default(&str_local);
    let running_U_default = FoldedInstance::default(&str_local);
    let bundle_s1 = mk_absent_bundle(LookupRunningWitness::<E1Eng>::default(&lookup_shape));

    let (_nifs1, (folded_U_s1, folded_W_s1), folded_lws_s1) =
      NIFS::<E1Eng>::prove_with_multi_table_lookup(
        &ck,
        &ro_consts,
        &pp_digest,
        &str_local,
        &running_U_default,
        &running_W_default,
        &u_step1,
        &w_step1,
        &[bundle_s1],
      )
      .expect("fold #1 prove must succeed (positive baseline)");

    assert_eq!(folded_lws_s1.len(), 1, "K=1: folded_lws_s1 must have length 1");
    let t_lookup_after_fold_1 = folded_U_s1
      .T_lookup
      .as_ref()
      .expect("fold #1 populates T_lookup");
    assert_eq!(t_lookup_after_fold_1.len(), 1, "K=1: T_lookup length 1");

    // === Fold #2: honest prove using fold-#1 folded state. ===
    let bundle_s2 = mk_absent_bundle(folded_lws_s1[0].clone());
    let payload_s2 = bundle_s2.payload.clone();
    let (nifs2, _, _) = NIFS::<E1Eng>::prove_with_multi_table_lookup(
      &ck,
      &ro_consts,
      &pp_digest,
      &str_local,
      &folded_U_s1,
      &folded_W_s1,
      &u_step2,
      &w_step2,
      &[bundle_s2],
    )
    .expect("fold #2 prove must succeed (positive baseline)");

    let public_bundles_s2 = vec![LookupPayloadPublicMultiTable::<E1Eng> {
      table_id: 0,
      comm_L: payload_s2.comm_L,
      comm_values: payload_s2.comm_values.clone(),
      comm_ts: payload_s2.comm_ts,
    }];

    // Positive control: honest fold #2 verify MUST accept against honest
    // folded_U_s1. If this fires the fixture is broken; halt before
    // attributing failure to any corruption pattern. Suppress unused
    // warning by discarding the Ok payload.
    let _accepted = nifs2
      .verify_with_multi_table_lookup(
        &ro_consts,
        &pp_digest,
        &str_local,
        &folded_U_s1,
        &u_step2,
        &public_bundles_s2,
      )
      .expect(
        "M.GH7.3b positive control: honest fold #2 verify must accept against \
         the honest folded_U_s1. If this fires, the fixture is broken — halt \
         and surface BEFORE blaming any corruption pattern.",
      );

    // Burn the rng so the seed parameter is not unused (deterministic
    // reviewer-reproducibility per US-05).
    let _ = Scalar::random(&mut rng);

    Bn256EngineKZGFixture {
      ro_consts,
      pp_digest,
      str_local,
      folded_U_s1,
      u_step2,
      nifs2,
      public_bundles_s2,
    }
  }

  /// Honest-baseline fixture handle returned by
  /// `m_gh7_3b_build_honest_two_step_fixture`. Carries only what the three
  /// negative tests need at the verifier-call boundary.
  #[cfg(feature = "lookup-fold")]
  struct Bn256EngineKZGFixture {
    ro_consts: crate::traits::RO2Constants<Bn256EngineKZG>,
    pp_digest: <Bn256EngineKZG as Engine>::Scalar,
    str_local: crate::neutron::Structure<Bn256EngineKZG>,
    folded_U_s1: crate::neutron::FoldedInstance<Bn256EngineKZG>,
    u_step2: crate::r1cs::R1CSInstance<Bn256EngineKZG>,
    nifs2: crate::neutron::NIFS<Bn256EngineKZG>,
    public_bundles_s2:
      Vec<crate::neutron::relation::LookupPayloadPublicMultiTable<Bn256EngineKZG>>,
  }

  /// **M.GH7.3b test (i)** — `prove_step_with_lookup_fold` negative test:
  /// corrupt-`U1.T`-at-verifier-input-boundary rejects at the R1CS-side
  /// (C)-binding (`vendor/nova/src/neutron/nifs.rs:2000-2001`).
  ///
  /// Sub-ratification 2026-05-12 late evening: this test was RE-TARGETED from
  /// Corrigendum #15's original `bundles[j].running_lw = default(&shape)`
  /// corruption (structurally inadequate at the lookup-side (C)-binding per
  /// the `evals[1] := t_lookup_running - evals[0]` tautology in
  /// `lookup_sumcheck.rs:278-285`) to verifier-input `U1.T += Scalar::ONE`,
  /// which routes through a structurally-distinct rejection mechanism in the
  /// SAME `verify_with_multi_table_lookup` body (Step 7, R1CS-side, fires
  /// BEFORE the lookup-side loop at Step 8). Same `NovaError` variant,
  /// different rejection locus. Fifth Halpert-class verify-don't-assume
  /// failure named in Corrigendum #15 Sub-ratification §"Audit-trail
  /// honesty".
  ///
  /// Rejection mechanism: `self.poly.eval_at_zero() + self.poly.eval_at_one()
  /// != (ONE - rho) * U1.T` at `nifs.rs:2000-2001`. The prover's `self.poly`
  /// was committed against `T_honest_target = (ONE - rho) * U1_honest.T`;
  /// after `U1.T += ONE`, the verifier computes `T_corrupted_target = (ONE -
  /// rho) * (U1_honest.T + ONE) = T_honest_target + (ONE - rho)`. For `rho ≠
  /// ONE` (overwhelming probability under RO-squeeze + BN254 scalar field
  /// characteristic), `(ONE - rho) ≠ ZERO` so `T_honest_target ≠
  /// T_corrupted_target`; equation rejects deterministically with
  /// `NovaError::InvalidSumcheckProof`.
  ///
  /// Anchored on Nova 2021/370 v3 §3 R1CS sumcheck soundness (the per-step
  /// R1CS folding-step (C)-binding; M.4-landed in vendor at vendor HEAD
  /// `e3fe75d` and reachable independent of M.GH7.4 progress).
  ///
  /// STOP-AND-ASK trigger #5 (per Sub-ratification): the rejection MUST occur
  /// at the R1CS-side (C)-binding (`:2000-2001`, Step 7), NOT at the
  /// lookup-side (C)-binding loop (`:2025-2028`, Step 8). The two sites
  /// return the same `NovaError::InvalidSumcheckProof` variant, so the
  /// typed-variant assertion alone cannot distinguish them — the
  /// disambiguation is via the algebra: corrupting `U1.T` (NOT
  /// `U1.T_lookup`) flips the R1CS-side equation while leaving the
  /// lookup-side equation untouched, so Step 7 short-circuits with `Err`
  /// before Step 8 is reached. The empirical-close confirmation is the
  /// design pin §6.1 row M.GH7.3b SECONDARY falsifier.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn prove_step_with_lookup_fold_corrupt_t_at_index_in_consumed_u1_rejects_at_r1cs_c_binding() {
    const SEED_M_GH7_3B_TEST_I: u64 = 0xC1BE_5BAD_C0DE_7031;
    let fixture = m_gh7_3b_build_honest_two_step_fixture(SEED_M_GH7_3B_TEST_I);

    // Corrupt the verifier-input R1CS-side running target U1.T by additive
    // += ONE. `FoldedInstance.T` is `pub(crate)` at `relation.rs:254`;
    // accessible from this `#[cfg(test)] mod tests` because the test module
    // is a child of `crate::neutron` (sibling to `relation` and `nifs`).
    let mut folded_U_step1_corrupted = fixture.folded_U_s1.clone();
    folded_U_step1_corrupted.T += <Bn256EngineKZG as Engine>::Scalar::ONE;
    assert_ne!(
      folded_U_step1_corrupted.T, fixture.folded_U_s1.T,
      "test (i) corruption non-vacuous: corrupted T must differ from honest T \
       (BN254 scalar field is prime of size ≥ 2, so honest + ONE ≠ honest)"
    );

    let result = fixture.nifs2.verify_with_multi_table_lookup(
      &fixture.ro_consts,
      &fixture.pp_digest,
      &fixture.str_local,
      &folded_U_step1_corrupted,
      &fixture.u_step2,
      &fixture.public_bundles_s2,
    );

    // Primary assertion: typed-variant rejection (M.14 / M.GH5.7 pattern at
    // `nifs.rs:6140`). The locus disambiguation (R1CS-side vs lookup-side
    // (C)-binding) is via the algebra of the corruption — `U1.T` corruption
    // can ONLY flip the R1CS-side equation at Step 7; the lookup-side
    // equation at Step 8 reads `t_lookup_running_j = U1.T_lookup[j]`, which
    // is UNCHANGED by this test's corruption.
    assert!(
      matches!(
        result,
        Err(crate::errors::NovaError::InvalidSumcheckProof)
      ),
      "M.GH7.3b test (i): verify_with_multi_table_lookup must REJECT the \
       R1CS-side `U1.T += ONE` corruption at the R1CS-side (C)-binding \
       (`vendor/nova/src/neutron/nifs.rs:2000-2001`, Step 7 of \
       verify_with_multi_table_lookup) with `NovaError::InvalidSumcheckProof`. \
       The prover's `self.poly` was committed against \
       `T_honest_target = (ONE - rho) * U1_honest.T`; the corrupted verifier \
       computes `T_corrupted_target = (ONE - rho) * (U1_honest.T + ONE)`. For \
       `rho ≠ ONE` (overwhelming probability under RO-squeeze + BN254 \
       characteristic), the equation rejects deterministically. \
       Anchored on Nova 2021/370 v3 §3 R1CS sumcheck soundness. If this \
       fires, halt and surface — this is the Constraint hygiene §70-78 \
       discharge for the M.GH7.3a public prover API. Got: {:?}",
      result.as_ref().err()
    );
  }

  /// **M.GH7.3b test (ii)** — `prove_step_with_lookup_fold` negative test:
  /// post-prove additive mutation of `nifs.poly_lookup[0].coeffs[0]` rejects
  /// at the lookup-side (C)-binding (`vendor/nova/src/neutron/nifs.rs:2025-2028`).
  ///
  /// Sub-ratification 2026-05-12 late evening: mutation shape AMENDED from
  /// Corrigendum #15's original multiplicative `× Scalar::from(2)` (which
  /// preserves the (C)-binding tautology under all-zero `t_lookup_running =
  /// 0`: `2·0 + 2·0 = 0 = 0`, per the crafter dispatch's empirical
  /// "Gap 1" finding) to additive `coeffs[0] += Scalar::ONE`.
  ///
  /// Algebra of the additive mutation: for any `UniPoly<Scalar>`, the load-
  /// bearing invariants at `vendor/nova/src/spartan/polys/univariate.rs` are
  /// `eval_at_zero() = coeffs[0]` and `eval_at_one() = Σ coeffs[i]`. After
  /// `coeffs[0] += ONE`:
  ///   - eval_at_zero shifts by +ONE
  ///   - eval_at_one  shifts by +ONE  (because coeffs[0] is one of the
  ///                                   summands of `Σ coeffs[i]`)
  ///   - mutated_sum = (eval_at_zero + ONE) + (eval_at_one + ONE)
  ///                 = original_sum + 2·ONE
  /// For any field of characteristic ≠ 2, `2·ONE ≠ ZERO` so `mutated_sum ≠
  /// original_sum`. BN254 scalar field characteristic is the prime
  /// `r ≈ 2^254` (odd, verified) — the additive mutation breaks the
  /// (C)-binding equation under ANY fixture, including the all-zero
  /// `t_lookup_running = 0` baseline inherited from the AbsentTableStepCircuit
  /// shape. The empirical-close holds without fixture amendment.
  ///
  /// Anchored on FS-NIZK soundness for sumcheck-based protocols (Fischlin-
  /// Fischlin 2005 / Bernhard-Pereira-Warinschi 2012). The `UniPoly`
  /// `poly_lookup_j` is FS-bound at `nifs.rs:2003` and `:2030` via the prover
  /// transcript; any post-prove mutation of `coeffs[0]` changes the
  /// evaluations at `0` and `1` (deterministically by the invariants above),
  /// and the verifier's (C)-binding check at `:2025-2028` reads the mutated
  /// polynomial directly.
  ///
  /// STOP-AND-ASK trigger #6 (per Sub-ratification): under the all-zero
  /// `t_lookup_running_j = 0` baseline, the additive mutation MUST produce
  /// `eval_at_zero + eval_at_one = 2·ONE ≠ 0`. If the additive mutation
  /// fails to reject, the coefficient-to-evaluation invariants are violated
  /// at the vendor-side `polys/univariate.rs` — diagnose against the
  /// load-bearing invariants `eval_at_zero = coeffs[0]` and `eval_at_one = Σ
  /// coeffs[i]` before declaring the mutation shape wrong.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn prove_step_with_lookup_fold_post_prove_mutate_poly_lookup_via_nifs_rebuild_rejects_at_lookup_c_binding(
  ) {
    const SEED_M_GH7_3B_TEST_II: u64 = 0xC1BE_5BAD_C0DE_7032;
    let fixture = m_gh7_3b_build_honest_two_step_fixture(SEED_M_GH7_3B_TEST_II);

    // Mutate the NIFS's `poly_lookup[0]` constant term by additive += ONE.
    // `NIFS::poly_lookup` is `pub` per the M.GH5.1 visibility bump at
    // `nifs.rs:252`; mutation is mechanically feasible from outside the
    // `neutron` module. We clone the NIFS to keep the fixture's honest copy
    // available for symmetry with tests (i) and (iii).
    let mut nifs_corrupted = fixture.nifs2.clone();
    let poly_lookup_vec = nifs_corrupted
      .poly_lookup
      .as_mut()
      .expect("fold #2 prove populates Some(poly_lookup) under lookup-fold");
    assert_eq!(
      poly_lookup_vec.len(),
      1,
      "K=1: poly_lookup must be a length-1 Vec"
    );
    let poly0 = &mut poly_lookup_vec[0];
    // Snapshot the honest eval_at_zero + eval_at_one for the post-mutation
    // sanity check below.
    let honest_eval_sum = poly0.eval_at_zero() + poly0.eval_at_one();

    // Additive mutation on the constant term: coeffs[0] += ONE.
    // `UniPoly.coeffs` is `pub` per the existing vendor-side polynomial API
    // (verified by the use in test (ii) compilation; see UniPoly source at
    // `spartan/polys/univariate.rs`). If the field is not `pub`, the
    // compilation will fail — STOP-AND-ASK at that point per the
    // visibility-gap framing of trigger #1.
    poly0.coeffs[0] += <Bn256EngineKZG as Engine>::Scalar::ONE;

    let mutated_eval_sum = poly0.eval_at_zero() + poly0.eval_at_one();
    let two_one = <Bn256EngineKZG as Engine>::Scalar::ONE
      + <Bn256EngineKZG as Engine>::Scalar::ONE;
    assert_eq!(
      mutated_eval_sum,
      honest_eval_sum + two_one,
      "test (ii) additive-mutation effectiveness (STOP-AND-ASK trigger #6): \
       additive `coeffs[0] += ONE` must shift eval_at_zero and eval_at_one \
       each by +ONE (because `eval_at_zero = coeffs[0]` and `eval_at_one = \
       Σ coeffs[i]`), so the sum shifts by +2·ONE. If this fires, the \
       UniPoly coefficient-to-evaluation invariants at \
       `spartan/polys/univariate.rs` are violated; halt before claiming \
       additive mutation is wrong."
    );
    assert_ne!(
      mutated_eval_sum, honest_eval_sum,
      "test (ii) mutation non-vacuous: BN254 scalar field characteristic is \
       odd (≠ 2), so 2·ONE ≠ ZERO; mutated_sum ≠ honest_sum"
    );

    let result = nifs_corrupted.verify_with_multi_table_lookup(
      &fixture.ro_consts,
      &fixture.pp_digest,
      &fixture.str_local,
      &fixture.folded_U_s1,
      &fixture.u_step2,
      &fixture.public_bundles_s2,
    );

    assert!(
      matches!(
        result,
        Err(crate::errors::NovaError::InvalidSumcheckProof)
      ),
      "M.GH7.3b test (ii): verify_with_multi_table_lookup must REJECT the \
       additive `poly_lookup[0].coeffs[0] += ONE` mutation at the lookup-side \
       (C)-binding loop (`vendor/nova/src/neutron/nifs.rs:2025-2028`, Step 8 \
       of verify_with_multi_table_lookup) with `NovaError::InvalidSumcheckProof`. \
       The prover committed `poly_lookup_0` against the honest \
       `t_lookup_running_0 = folded_U_s1.T_lookup[0]`; the additive constant- \
       term mutation shifts `eval_at_zero() + eval_at_one()` by +2·ONE (≠ 0 \
       in BN254's odd-characteristic prime field), breaking the (C)-binding \
       equation. Anchored on FS-NIZK soundness (Fischlin-Fischlin 2005). If \
       this fires, halt and surface — Constraint hygiene §70-78 discharge \
       for the post-prove NIFS-message-tamper attack surface introduced by \
       the M.GH7.3a public prover API. Got: {:?}",
      result.as_ref().err()
    );
  }

  /// **M.GH7.3b test (iii)** — `prove_step_with_lookup_fold` negative test:
  /// corrupt-`U1.T_lookup`-at-verifier-input-boundary rejects at the
  /// lookup-side (C)-binding (`vendor/nova/src/neutron/nifs.rs:2025-2028`).
  ///
  /// STANDS verbatim from Corrigendum #15 (modulo function-name suffix
  /// `_at_c_binding → _at_lookup_c_binding` for symmetry with tests (i) and
  /// (ii)). This is the direct dual of M.14's forward-cancellation pattern
  /// at `nifs.rs:5719-6149` at the per-step-NIFS-routed-through-M.GH7.3a
  /// surface (K=1 single-table variant; M.14 is K=2 cross-table).
  ///
  /// Rejection mechanism: `poly_lookup_0.eval_at_zero() +
  /// poly_lookup_0.eval_at_one() != t_lookup_running_0` at `nifs.rs:2025-2028`.
  /// The prover's `poly_lookup_0` was produced against
  /// `U1_honest.T_lookup[0]` (the honest fold-#1 accumulated running
  /// scalar); after `r_U_pre.T_lookup[0] += ONE`, the verifier reads
  /// `t_lookup_running_0 = U1_corrupted.T_lookup[0] = honest + ONE`, but
  /// `poly_lookup_0(0) + poly_lookup_0(1) = honest` (per the prover-side
  /// `evals[1] := t_lookup_running − evals[0]` tautology — note that THIS
  /// tautology is what makes the corruption observable here: the polynomial
  /// is FROZEN at the honest running scalar, and the verifier-side
  /// substitution surfaces the divergence). Equation rejects:
  /// `honest ≠ honest + ONE` holds in any prime field of size ≥ 2.
  ///
  /// Anchored on Nova 2021/370 v3 §4 binding-by-hash composition.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn prove_step_with_lookup_fold_corrupt_t_lookup_at_index_in_consumed_u1_rejects_at_lookup_c_binding(
  ) {
    const SEED_M_GH7_3B_TEST_III: u64 = 0xC1BE_5BAD_C0DE_7033;
    let fixture = m_gh7_3b_build_honest_two_step_fixture(SEED_M_GH7_3B_TEST_III);

    // Corrupt the verifier-input lookup-side running target
    // U1.T_lookup[0] by additive += ONE. `FoldedInstance.T_lookup` is
    // `pub(crate)` at `relation.rs:294`; accessible from this sibling test
    // module.
    let mut folded_U_step1_corrupted_lookup = fixture.folded_U_s1.clone();
    {
      let t_lookup = folded_U_step1_corrupted_lookup
        .T_lookup
        .as_mut()
        .expect("fold #1 populates Some(T_lookup) under lookup-fold");
      assert_eq!(t_lookup.len(), 1, "K=1: T_lookup must have length 1");
      let honest_value = t_lookup[0];
      t_lookup[0] += <Bn256EngineKZG as Engine>::Scalar::ONE;
      assert_ne!(
        t_lookup[0], honest_value,
        "test (iii) corruption non-vacuous: corrupted T_lookup[0] must differ \
         from honest T_lookup[0] (BN254 scalar field is prime of size ≥ 2, \
         so honest + ONE ≠ honest)"
      );
    }

    let result = fixture.nifs2.verify_with_multi_table_lookup(
      &fixture.ro_consts,
      &fixture.pp_digest,
      &fixture.str_local,
      &folded_U_step1_corrupted_lookup,
      &fixture.u_step2,
      &fixture.public_bundles_s2,
    );

    assert!(
      matches!(
        result,
        Err(crate::errors::NovaError::InvalidSumcheckProof)
      ),
      "M.GH7.3b test (iii): verify_with_multi_table_lookup must REJECT the \
       lookup-side `U1.T_lookup[0] += ONE` corruption at the lookup-side \
       (C)-binding loop (`vendor/nova/src/neutron/nifs.rs:2025-2028`, Step 8 \
       of verify_with_multi_table_lookup, j=0 first-fire under K=1) with \
       `NovaError::InvalidSumcheckProof`. The prover's `poly_lookup_0` was \
       produced against `U1_honest.T_lookup[0]`; after corruption, the \
       verifier reads `t_lookup_running_0 = honest + ONE`, but the \
       polynomial's (C)-binding sum equals `honest` (frozen at prove time by \
       the `evals[1] := t_lookup_running − evals[0]` construction in \
       `lookup_sumcheck.rs:278-285`); equation `honest ≠ honest + ONE` in \
       any prime field of size ≥ 2 rejects deterministically. Anchored on \
       Nova 2021/370 v3 §4 binding-by-hash composition (direct dual of M.14 \
       forward-cancellation at `nifs.rs:5719-6149` at the K=1 single-table \
       variant). If this fires, halt and surface — Constraint hygiene \
       §70-78 discharge for the verifier-input running-state-carry \
       substitution attack surface. Got: {:?}",
      result.as_ref().err()
    );
  }
}
