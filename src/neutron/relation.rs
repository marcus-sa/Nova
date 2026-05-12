//! This module defines relations used in the Neutron folding scheme
use crate::{
  errors::NovaError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
  spartan::math::Math,
  traits::{commitment::CommitmentEngineTrait, AbsorbInRO2Trait, Engine, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A handle to a lookup table that has been registered with a commitment key.
///
/// Carried in [`Structure::lookups`] (via [`LookupShape`]) and absorbed into the
/// pp-digest as part of the running `Structure<E>`. The commitment is computed
/// once at table-registration time and pins the table's identity for every
/// subsequent fold step on the IVC chain.
///
/// Pinned by `c1-beta-lookup-fold-soundness-sketch-addendum-spike-design-pins.md`
/// §A.1.1 property 4 ("table commitment provenance — public input, not per-step
/// witness").
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupTableHandle<E: Engine> {
  /// Stable identifier for the table; ordering on `LookupShape::tables` is
  /// canonicalised by sorting on this id (addendum §A.1.1 property 4 line "sorted by table-id").
  pub table_id: u64,
  /// Number of entries in the table.
  pub size: usize,
  /// Commitment to the table contents under the table-derivation generator.
  pub commitment: Commitment<E>,
}

/// A multi-column lookup table.
///
/// Stage I-pri (Lasso §6.2 address-value combine): a multi-column table
/// has an implicit address column `i ∈ [0, size)` plus `c >= 0` value
/// columns. The single-column case (`columns.is_empty()`) collapses to
/// the prior `LookupTableHandle` shape — the table is the identity
/// `{0, 1, ..., size-1}` and the LogUp combine reduces to the identity.
///
/// `value_commitments[i]` is the commitment to `columns[i]` (parallel
/// vectors). The address column has no separate commitment because the
/// address `i` is implicit (the LogUp combined-witness preprocessing
/// emits `combined[k] = address[k] + α·v₁[k] + ... + α^c·v_c[k]` and
/// the verifier reconstructs `comm_combined` by linear homomorphism).
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MultiColumnLookupTable<E: Engine> {
  /// Stable identifier for the table.
  pub table_id: u64,
  /// Number of entries (rows) in the table.
  pub size: usize,
  /// Value columns. `columns[c][i]` is the c-th column's value at row i.
  pub columns: Vec<Vec<E::Scalar>>,
  /// Per-column commitments. `value_commitments[c]` commits to `columns[c]`.
  pub value_commitments: Vec<Commitment<E>>,
}

/// Lookup-side shape for a [`Structure`] that uses the lookup-fold extension.
///
/// Pinned by addendum §A.2.1: `comm_L`, `comm_ts`, `comm_inv_w`, `comm_inv_t`
/// per-step witness shapes are described here, and the table identities are
/// fixed via `tables` (which is what binds the LogUp identity's table side).
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupShape<E: Engine> {
  /// Tables registered for this `Structure`. Order canonicalised by ascending
  /// `table_id` for deterministic `pp_digest` derivation (addendum §A.1.1
  /// property 4).
  pub tables: Vec<LookupTableHandle<E>>,
  /// Multi-column lookup tables registered for this `Structure` (Stage I-app.2).
  ///
  /// Each entry's `table_id`, `size`, column count, and per-column commitments
  /// are absorbed into `pp_digest` via the serde `Serialize` derive on
  /// [`MultiColumnLookupTable`] — that pins the table identity globally across
  /// every fold step on the IVC chain. The per-step
  /// [`crate::neutron::nifs::NIFS::prove_with_multi_column_lookup`] /
  /// [`crate::neutron::nifs::NIFS::verify_with_multi_column_lookup`]
  /// take a `table_id: u64` argument that resolves into this vec — the
  /// table contents are NEVER passed per-call.
  ///
  /// Order canonicalised by ascending `table_id` (matches the
  /// [`tables`] vec discipline; addendum §A.1.1 property 4).
  ///
  /// Pinned by Stage I-app.2 (pp_digest binding for table identity).
  #[serde(default = "Vec::new")]
  pub multi_column_tables: Vec<MultiColumnLookupTable<E>>,
  /// Number of address columns in the per-step lookup witness layout.
  pub num_addr_columns: usize,
  /// Number of witness columns in the per-step lookup witness layout.
  pub num_witness_columns: usize,
  /// Number of variables in the witness-side `eq` polynomial; the pooled
  /// per-step witness vector is padded to length `2^witness_ell_cached`.
  ///
  /// Populated by `LookupConstraintSystem::finalize()` at end-of-step (Stage 3,
  /// §H.3); set by callers explicitly at construction time for the spike
  /// scope. Pinned by §B.4.3 of the implementation outline (amendment
  /// 2026-05-07).
  pub(crate) witness_ell_cached: usize,
}

#[cfg(feature = "lookup-fold")]
impl<E: Engine> LookupShape<E> {
  /// Construct a `LookupShape` from caller-supplied components.
  ///
  /// `witness_ell_cached` is `pub(crate)` to keep the Stage 3
  /// `LookupConstraintSystem::finalize()` discipline in the vendor crate;
  /// this constructor is the spike-scope escape hatch for direct callers
  /// (e.g. proving-time benchmarks in `inumbra-spend-harness`) that need
  /// to instantiate a `LookupShape` without rerunning the full Stage 3
  /// pooling pipeline.
  ///
  /// Pinned by §B.4.3 of the implementation outline (amendment 2026-05-07).
  pub fn new(
    tables: Vec<LookupTableHandle<E>>,
    multi_column_tables: Vec<MultiColumnLookupTable<E>>,
    num_addr_columns: usize,
    num_witness_columns: usize,
    witness_ell_cached: usize,
  ) -> Self {
    Self {
      tables,
      multi_column_tables,
      num_addr_columns,
      num_witness_columns,
      witness_ell_cached,
    }
  }

  /// Number of variables in the witness-side `eq` polynomial. The pooled
  /// per-step witness vector is padded to length `2^witness_ell()`.
  ///
  /// Populated by `LookupConstraintSystem::finalize()` at end-of-step
  /// (Stage 3); for direct callers (Stage 1 spike), supplied via the
  /// `witness_ell_cached` field at construction time.
  ///
  /// Pinned by §B.4.3 of the implementation outline (amendment 2026-05-07).
  pub fn witness_ell(&self) -> usize {
    self.witness_ell_cached
  }

  /// Number of variables in the table-side `eq` polynomial.
  ///
  /// Defined as `ceil(log2(Σ table.size for table in tables))` for the
  /// merged-table pool (addendum §A.3.2). For single-table use (the spike
  /// default), this is `log2(self.tables[0].size)`.
  ///
  /// Pinned by §B.4.3 of the implementation outline (amendment 2026-05-07).
  pub fn table_ell(&self) -> usize {
    let total = self.tables.iter().map(|h| h.size).sum::<usize>();
    if total == 0 {
      0
    } else {
      total.next_power_of_two().trailing_zeros() as usize
    }
  }

  /// Split of `witness_ell()` into `(left, right)` matching `Structure`'s split
  /// for tensor-form `eq` reuse with `evaluation_points_cubic_with_two_inputs`
  /// (mirrors the R1CS-side split at `Structure::new`).
  ///
  /// Returns `(2^ell1, 2^ell2)` with `ell1 = ⌈ell/2⌉`, `ell2 = ⌊ell/2⌋`,
  /// and `ell1 + ell2 = witness_ell()`.
  ///
  /// Pinned by §B.4.3 of the implementation outline (amendment 2026-05-07).
  pub fn witness_split(&self) -> (usize, usize) {
    let ell = self.witness_ell();
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    (1 << ell1, 1 << ell2)
  }

  /// Split of `table_ell()` into `(left, right)` for tensor-form `eq` reuse.
  ///
  /// Pinned by §B.4.3 of the implementation outline (amendment 2026-05-07).
  pub fn table_split(&self) -> (usize, usize) {
    let ell = self.table_ell();
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    (1 << ell1, 1 << ell2)
  }
}

/// A type that holds structure information for a zero-fold relation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Structure<E: Engine> {
  /// Shape of the R1CS relation
  pub(crate) S: R1CSShape<E>,

  /// the number of variables in the Eq polynomial
  pub(crate) ell: usize,
  pub(crate) left: usize,
  pub(crate) right: usize,

  /// Lookup-side shape, present iff the `Structure` participates in the
  /// lookup-fold extension. `None` for the legacy zero-fold path so non-lookup
  /// `StepCircuit`s pay zero serde / per-step cost. Pinned by addendum §A.2.1
  /// (the `Option<>` wrapper is the backward-compatibility pin).
  ///
  /// Because `Structure<E>` is included in `PublicParams<E1, E2, C>` and
  /// `PublicParams` participates in `SimpleDigestible` (see
  /// `vendor/nova/src/neutron/mod.rs:60`), the `Serialize`/`Deserialize`
  /// derive on this field makes both single-column `LookupTableHandle`s
  /// AND `MultiColumnLookupTable`s (`LookupShape::tables` and
  /// `LookupShape::multi_column_tables`) part of the `pp_digest`
  /// automatically. This satisfies addendum §A.1.1 property 4 (single-
  /// column) and Stage I-app.2 (multi-column) — the table contents are
  /// pinned globally at IVC initialisation, not per-call.
  #[cfg(feature = "lookup-fold")]
  pub(crate) lookups: Option<LookupShape<E>>,
}

/// A type that holds witness information for a zero-fold relation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedWitness<E: Engine> {
  /// Running witness of the main relation
  pub(crate) W: Vec<E::Scalar>,
  /// Witness blinding factor.
  ///
  /// GH-#7 M.GH7.0.1 visibility bump (Corrigendum #8 §1.2(a)): bumped from
  /// fully-private to `pub(crate)` so the sibling `neutron::compressed_snark`
  /// module can read/write `r_W` when constructing test fixtures for the
  /// Pedersen MSM-linearity split-E commitment helper. Co-classified with
  /// the M.GH5.1 / M.GH5.7 lookup-fold-surface visibility-bump bracket
  /// noted at `neutron/mod.rs:30-32`.
  pub(crate) r_W: E::Scalar,

  /// eq polynomial in tensor form
  pub(crate) E: Vec<E::Scalar>,
  /// `E`-blinding factor.
  ///
  /// GH-#7 M.GH7.0.1 visibility bump (Corrigendum #8 §1.2(a)): bumped from
  /// fully-private to `pub(crate)` so the sibling `neutron::compressed_snark`
  /// module can read `r_E` when computing the blinding split
  /// `r_E1 + r_E2 == W.r_E` for `split_E_commitments`. Co-classified with
  /// the `r_W` bump above and the M.GH5.1 / M.GH5.7 visibility-bump
  /// bracket noted at `neutron/mod.rs:30-32`.
  pub(crate) r_E: E::Scalar,
}

/// A type that holds instance information for a zero-fold relation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedInstance<E: Engine> {
  pub(crate) comm_W: Commitment<E>,
  pub(crate) comm_E: Commitment<E>,
  pub(crate) T: E::Scalar,
  pub(crate) u: E::Scalar,
  pub(crate) X: Vec<E::Scalar>,

  /// Folded per-step lookup-witness commitment, **per-table**. `None` until
  /// a non-default running instance has been produced via a fold step that
  /// carried lookup data; otherwise `Some(Vec<Commitment<E>>)` with one
  /// entry per registered table in `table_id`-canonical position (mirroring
  /// `T_lookup_per_table` at `relation.rs:265`). Single-table use
  /// degenerates cleanly to a one-element Vec.
  ///
  /// **M.GH7.0a widening** (GH-#7 design pin Corrigendum #1 + Corrigendum
  /// #6 §1.3): widened from `Option<Commitment<E>>` to
  /// `Option<Vec<Commitment<E>>>` so the §1.2(b) bridge-reduction (Spartan
  /// close) can consume per-table running commitments. The per-`j`
  /// independent fold under common `r_b` (Corrigendum #6 primitive 3)
  /// preserves PCS binding for each `(X, j)` pair against (PCS binding) +
  /// (RO2 collision-resistance) + (Path β per-step binding). Originally
  /// pinned by addendum §A.2.1.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_L: Option<Vec<Commitment<E>>>,
  /// Folded multiplicity-vector commitment, **per-table**. See `comm_L`
  /// for the M.GH7.0a widening rationale.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_ts: Option<Vec<Commitment<E>>>,
  /// Folded inverse-witness commitment for `1/(w_i + r)`, **per-table**.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_w: Option<Vec<Commitment<E>>>,
  /// Folded inverse-table commitment for `1/(T_j + r)`, **per-table**.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_t: Option<Vec<Commitment<E>>>,
  /// Running target for the lookup-zero side. Mirrors the existing `T` for
  /// the R1CS-zero side (§A.2.1).
  ///
  /// Multi-table extension (GH-#2, design pin §1.3): widened from
  /// `Option<E::Scalar>` to `Option<Vec<E::Scalar>>`. Each entry is the
  /// per-table running scalar `T_lookup_j` indexed by `table_id`-canonical
  /// position in `LookupShape::multi_column_tables`. Single-table use
  /// degenerates cleanly to a one-element Vec.
  #[cfg(feature = "lookup-fold")]
  pub(crate) T_lookup: Option<Vec<E::Scalar>>,
}

/// Verifier-side projection of [`LookupPayload`] carried as a public input
/// to the augmented circuit (Stage G, §G.4).
///
/// The augmented-circuit verifier needs the per-step `comm_L` and `comm_ts`
/// commitments to absorb them in the FS transcript at the same point as the
/// native `verify_with_lookup` (mirroring `nifs.rs:609-610`). It does NOT
/// need `comm_inv_w` / `comm_inv_t` (which the native code re-derives from
/// the [`NIFS`] message) nor `T2_lookup` (which is verifier-computed).
///
/// Pinned by §G.4 of the implementation outline.
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupPayloadPublic<E: Engine> {
  /// Per-step lookup-witness commitment.
  pub comm_L: Commitment<E>,
  /// Per-step multiplicity-vector commitment.
  pub comm_ts: Commitment<E>,
}

/// Verifier-side per-table public projection of the multi-table lookup payload.
///
/// GH-#2 M.4 (design pin §5.1): the multi-table verifier consumes a slice of
/// these — one entry per registered table in `table_id`-canonical order —
/// instead of a single [`LookupPayload`]. Each entry carries the public
/// commitments the verifier needs to absorb at FS-transcript step 2 (per pin
/// §2.2): the address-column commitment `comm_L`, the per-column value
/// commitments `comm_values` (Lasso §6.2 multi-column extension), and the
/// multiplicity commitment `comm_ts`. The verifier never needs `comm_inv_w` /
/// `comm_inv_t` (recovered from the [`NIFS`] message's per-table Vec fields)
/// nor `T2_lookup` (per-table running scalar threading is via the
/// `FoldedInstance::T_lookup` Vec, not this payload).
///
/// For "absent" tables at a fold step (queries did not touch the table this
/// step), each commitment is the zero-payload commitment per pin §1.5.3
/// ("zero-payload commitments, not skipped absorptions") — the FS transcript
/// schedule is constant-shape across present/absent regardless.
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupPayloadPublicMultiTable<E: Engine> {
  /// Stable identifier for the table this entry corresponds to.
  pub table_id: u64,
  /// Per-step lookup-witness commitment for the table's address column.
  pub comm_L: Commitment<E>,
  /// Per-step value-column commitments (Lasso §6.2). Empty for the
  /// single-column degenerate path (pin §2.2 step 5a empty-skip rule).
  #[serde(default = "Vec::new")]
  pub comm_values: Vec<Commitment<E>>,
  /// Per-step multiplicity-vector commitment.
  pub comm_ts: Commitment<E>,
}

/// Per-step lookup-side payload delivered to [`NIFS::prove`] alongside the
/// incoming [`R1CSInstance`].
///
/// Carries the per-step lookup commitments and the per-step `T2_lookup` evaluation
/// target. `NIFS::prove` absorbs the `comm_L` / `comm_ts` fields at addendum
/// §A.1.1 step (2), squeezes `r`, and absorbs `comm_inv_w` / `comm_inv_t` at
/// §A.1.1 step (6).
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupPayload<E: Engine> {
  /// Per-step lookup-witness commitment for the **address column** (the
  /// pooled query vector under the `LookupConstraintSystem` collector —
  /// addendum §A.3.2). For single-column lookups (`comm_values.is_empty()`)
  /// this is the only witness commitment and `address ≡ value`.
  pub comm_L: Commitment<E>,
  /// Per-step multiplicity-vector commitment.
  pub comm_ts: Commitment<E>,
  /// Per-step inverse-witness commitment for `1/(w_i + r)`. Computed AFTER
  /// `r` is squeezed in §A.1.1 step (5); supplied at this point as a fully
  /// committed value so `NIFS::prove` can absorb it at step (6).
  pub comm_inv_w: Commitment<E>,
  /// Per-step inverse-table commitment for `1/(T_j + r)`.
  pub comm_inv_t: Commitment<E>,
  /// Per-step lookup-zero evaluation target. For a fresh per-step `R1CSInstance`
  /// satisfying the lookup relation, this is the per-step LogUp residual
  /// evaluated at the bound randomness (the analogue of `T2 = 0` for the
  /// R1CS-zero side at line 233 of `vendor/nova/src/neutron/nifs.rs`).
  pub T2_lookup: E::Scalar,
  /// Per-step value-column commitments (Stage I-pri, Lasso §6.2 address-
  /// value combine). Empty for the single-column path (Stage H), in which
  /// case the multi-column transcript extension (per-column absorption +
  /// α squeeze) is skipped and the FS transcript is byte-identical to the
  /// Stage H pin. Pinned by Stage I-pri §I.4.
  ///
  /// For multi-column queries this carries the commitments to the per-step
  /// value-column vectors `v₁[k], ..., v_c[k]`. The verifier reconstructs
  /// the combined-witness commitment by linear homomorphism:
  ///
  /// ```text
  /// comm_W_combined = comm_L + α·comm_values[0] + α²·comm_values[1] + ...
  /// ```
  #[serde(default = "Vec::new")]
  pub comm_values: Vec<Commitment<E>>,
}

/// Running lookup witness vectors accumulated across fold steps.
///
/// Carries the witness-side data that `NIFS::prove_with_lookup` needs for the
/// U1 (running) instance in the dual-instance `LookupSumcheckInstance`. At
/// outer base (first fold), all vectors are zeros. At subsequent steps, they
/// carry the linear-combination-folded values from prior fold steps.
///
/// These vectors are NOT recomputed from the running `FoldedWitness::W`; they
/// are stored alongside it and folded independently using the same `r_b` weight.
///
/// Option (b) from the task spec: passed separately through the prove API,
/// not embedded in `FoldedWitness`, to keep the non-lookup path untouched.
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupRunningWitness<E: Engine> {
  /// Running lookup-witness vector (pooled query values).
  pub witness: Vec<E::Scalar>,
  /// Running inverse-witness vector `1/(w_i + r)`.
  pub inv_w: Vec<E::Scalar>,
  /// Running table vector.
  pub table: Vec<E::Scalar>,
  /// Running multiplicity vector.
  pub multiplicities: Vec<E::Scalar>,
  /// Running inverse-table vector `ts_j/(T_j + r)`.
  pub inv_t: Vec<E::Scalar>,
  /// Running eq polynomial (witness side) in split-tensor form, left half.
  pub eq_w_left: Vec<E::Scalar>,
  /// Running eq polynomial (witness side) in split-tensor form, right half.
  pub eq_w_right: Vec<E::Scalar>,
  /// Running eq polynomial (table side) in split-tensor form, left half.
  pub eq_t_left: Vec<E::Scalar>,
  /// Running eq polynomial (table side) in split-tensor form, right half.
  pub eq_t_right: Vec<E::Scalar>,
}

#[cfg(feature = "lookup-fold")]
impl<E: Engine> LookupRunningWitness<E> {
  /// Creates a default (all-zeros) running witness for the first fold step.
  pub fn default(shape: &LookupShape<E>) -> Self {
    let (w_left, w_right) = shape.witness_split();
    let (t_left, t_right) = shape.table_split();
    let n_w = w_left * w_right;
    let n_t = t_left * t_right;
    Self {
      witness: vec![E::Scalar::ZERO; n_w],
      inv_w: vec![E::Scalar::ZERO; n_w],
      table: vec![E::Scalar::ZERO; n_t],
      multiplicities: vec![E::Scalar::ZERO; n_t],
      inv_t: vec![E::Scalar::ZERO; n_t],
      eq_w_left: vec![E::Scalar::ZERO; w_left],
      eq_w_right: vec![E::Scalar::ZERO; w_right],
      eq_t_left: vec![E::Scalar::ZERO; t_left],
      eq_t_right: vec![E::Scalar::ZERO; t_right],
    }
  }

  /// Fold with per-step fresh data using the same `r_b` weight.
  ///
  /// Mirrors `FoldedWitness::fold`: `new = (1-r_b)*self + r_b*fresh`.
  pub fn fold(
    &self,
    fresh: &LookupFreshWitness<E>,
    r_b: &E::Scalar,
  ) -> Self {
    let one_minus = E::Scalar::ONE - r_b;
    let interp = |a: &[E::Scalar], b: &[E::Scalar]| -> Vec<E::Scalar> {
      a.iter()
        .zip(b.iter())
        .map(|(x, y)| one_minus * x + *r_b * y)
        .collect()
    };
    Self {
      witness: interp(&self.witness, &fresh.witness),
      inv_w: interp(&self.inv_w, &fresh.inv_w),
      table: interp(&self.table, &fresh.table),
      multiplicities: interp(&self.multiplicities, &fresh.multiplicities),
      inv_t: interp(&self.inv_t, &fresh.inv_t),
      eq_w_left: interp(&self.eq_w_left, &fresh.eq_w_left),
      eq_w_right: interp(&self.eq_w_right, &fresh.eq_w_right),
      eq_t_left: interp(&self.eq_t_left, &fresh.eq_t_left),
      eq_t_right: interp(&self.eq_t_right, &fresh.eq_t_right),
    }
  }
}

/// Per-step fresh lookup witness data for the U2 (fresh) side of a fold step.
///
/// This carries the raw polynomial vectors that `LookupSumcheckInstance::new`
/// needs for the U2 side. The inverse witnesses (`inv_w`, `inv_t`) are
/// computed by `LookupSumcheckInstance::new` from the raw `witness` and
/// `table` + `multiplicities` vectors, but for folding purposes we need
/// them post-computation too — so they're stored back here after construction.
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupFreshWitness<E: Engine> {
  /// Per-step lookup-witness vector (pooled query values).
  pub witness: Vec<E::Scalar>,
  /// Per-step inverse-witness vector `1/(w_i + r)`.
  /// Populated AFTER `LookupSumcheckInstance::new` computes it.
  pub inv_w: Vec<E::Scalar>,
  /// Per-step table vector.
  pub table: Vec<E::Scalar>,
  /// Per-step multiplicity vector.
  pub multiplicities: Vec<E::Scalar>,
  /// Per-step inverse-table vector `ts_j/(T_j + r)`.
  /// Populated AFTER `LookupSumcheckInstance::new` computes it.
  pub inv_t: Vec<E::Scalar>,
  /// Per-step eq polynomial (witness side) in split-tensor form, left half.
  pub eq_w_left: Vec<E::Scalar>,
  /// Per-step eq polynomial (witness side) in split-tensor form, right half.
  pub eq_w_right: Vec<E::Scalar>,
  /// Per-step eq polynomial (table side) in split-tensor form, left half.
  pub eq_t_left: Vec<E::Scalar>,
  /// Per-step eq polynomial (table side) in split-tensor form, right half.
  pub eq_t_right: Vec<E::Scalar>,
}

impl<E: Engine> Structure<E> {
  /// GH-#5 M.GH5.7 / Pin Corrigendum #10 Q2 (Halpert 2026-05-10):
  /// accessor for the canonical (padded) `R1CSShape` carried by this
  /// structure. `Structure::new` calls `S.pad()` internally
  /// (`relation.rs:490`); harness-side consumers building witnesses
  /// against `Structure`-bound provers (M.GH5.7 negative-test triple)
  /// need this shape, not the unpadded original, for
  /// `r1cs_instance_and_witness` to produce a witness whose length
  /// agrees with the prover's `multiply_vec` invariant
  /// (`r1cs/mod.rs:384` `z.len() != self.num_io + self.num_vars + 1
  /// → InvalidWitnessLength`).
  ///
  /// Co-classified with the `comm_W()` / `comm_E()` accessors on
  /// `FoldedInstance` (same M.GH5.7 visibility-bump bracket).
  pub fn shape(&self) -> &R1CSShape<E> {
    &self.S
  }

  /// Create a new structure using the provided shape
  pub fn new(S: &R1CSShape<E>) -> Self {
    // pad to the regular shape
    let S = S.pad();

    let ell = S.num_cons.next_power_of_two().log_2();

    // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
    let ell1 = ell.div_ceil(2); // This ensures ell1 >= ell2
    let ell2 = ell / 2;

    Structure {
      S: S.clone(),
      ell,
      left: 1 << ell1,
      right: 1 << ell2,
      #[cfg(feature = "lookup-fold")]
      lookups: None,
    }
  }

  /// Create a new structure with a lookup-side shape attached.
  ///
  /// Constructs a `Structure<E>` whose `lookups` field is `Some(shape)`. The
  /// `tables` vec is canonicalised by sorting on `table_id` in place so the
  /// `pp_digest` derivation is deterministic regardless of caller-side
  /// insertion order (addendum §A.1.1 property 4).
  #[cfg(feature = "lookup-fold")]
  pub fn new_with_lookups(S: &R1CSShape<E>, mut shape: LookupShape<E>) -> Self {
    let mut s = Self::new(S);
    shape.tables.sort_by_key(|h| h.table_id);
    // Stage I-app.2: canonicalise multi-column tables by ascending `table_id`
    // so the `pp_digest` derivation (via serde on `Structure → LookupShape →
    // multi_column_tables`) is independent of caller-side insertion order.
    shape.multi_column_tables.sort_by_key(|t| t.table_id);
    s.lookups = Some(shape);
    s
  }

  /// Check if the witness is satisfying
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<E>,
    U: &FoldedInstance<E>,
    W: &FoldedWitness<E>,
  ) -> Result<(), NovaError> {
    // check if the witness is satisfying
    let z = [W.W.clone(), vec![U.u], U.X.clone()].concat();
    let (Az, Bz, Cz) = self.S.multiply_vec(&z)?;

    // full_E is the outer product of E1 and E2
    // E1 and E2 are splits of E
    let (E1, E2) = W.E.split_at(self.left);
    let mut full_E = vec![E::Scalar::ONE; self.left * self.right];
    for i in 0..self.right {
      for j in 0..self.left {
        full_E[i * self.left + j] = E2[i] * E1[j];
      }
    }

    let sum = full_E
      .par_iter()
      .zip(Az.par_iter())
      .zip(Bz.par_iter())
      .zip(Cz.par_iter())
      .map(|(((e, a), b), c)| *e * ((*a) * (*b) - *c))
      .reduce(|| E::Scalar::ZERO, |acc, x| acc + x);

    if sum != U.T {
      return Err(NovaError::UnSat {
        reason: format!("sum != U.T\n    sum: {:?}\n    U.T: {:?}", sum, U.T),
      });
    }

    // check the validity of the commitments
    let comm_W = E::CE::commit(ck, &W.W, &W.r_W);
    let comm_E = E::CE::commit(ck, &W.E, &W.r_E);

    if comm_W != U.comm_W || comm_E != U.comm_E {
      return Err(NovaError::UnSat {
        reason: "comm_W != U.comm_W || comm_E != U.comm_E".to_string(),
      });
    }

    Ok(())
  }
}

impl<E: Engine> FoldedWitness<E> {
  /// Create a default witness
  pub fn default(S: &Structure<E>) -> Self {
    FoldedWitness {
      W: vec![E::Scalar::ZERO; S.S.num_vars],
      r_W: E::Scalar::ZERO,
      E: vec![E::Scalar::ZERO; S.left + S.right],
      r_E: E::Scalar::ZERO,
    }
  }

  /// Fold the witness with another witness
  pub fn fold(
    &self,
    W2: &R1CSWitness<E>,
    E2: &Vec<E::Scalar>,
    r_E2: &E::Scalar,
    r_b: &E::Scalar,
  ) -> Result<Self, NovaError> {
    // we need to compute the weighted sum using weights of (1-r_b) and r_b
    let W = self
      .W
      .par_iter()
      .zip(W2.W.par_iter())
      .map(|(w1, w2)| *w1 + *r_b * (*w2 - *w1))
      .collect::<Vec<_>>();
    let r_W = (E::Scalar::ONE - r_b) * self.r_W + *r_b * W2.r_W;

    let E = self
      .E
      .par_iter()
      .zip(E2.par_iter())
      .map(|(e1, e2)| *e1 + *r_b * (*e2 - *e1))
      .collect::<Vec<_>>();
    let r_E = (E::Scalar::ONE - r_b) * self.r_E + *r_b * r_E2;

    Ok(Self { W, r_W, E, r_E })
  }
}

impl<E: Engine> FoldedInstance<E> {
  /// Create a default instance
  pub fn default(S: &Structure<E>) -> Self {
    FoldedInstance {
      comm_W: Commitment::<E>::default(),
      comm_E: Commitment::<E>::default(),
      T: E::Scalar::ZERO,
      u: E::Scalar::ZERO,
      X: vec![E::Scalar::ZERO; S.S.num_io],
      // Pinned by addendum §A.2.1: defaults to `None` so non-lookup `StepCircuit`s
      // pay zero per-step cost. The lookup-side fields become `Some` only after
      // a fold step that carried a lookup payload.
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
    }
  }

  /// Per-table running lookup-claim accumulator, exposed for differential
  /// harnesses that need to compare against an off-circuit reference.
  ///
  /// Returns `None` at outer base (no lookup payload has been folded in
  /// yet) and `Some(&[E::Scalar])` after the first fold step with a
  /// lookup payload. The returned slice has one entry per table in the
  /// canonical sort order (`sort_by_key(|t| t.table_id)` per
  /// `relation.rs:482`).
  ///
  /// GH-#2 M.11 enabler: the inumbra-side fold-of-four absent-table
  /// differential needs this projection to assert the running-scalar
  /// pipeline equals the off-circuit `fold_lookup_running_claim_naive`
  /// reference at every step (cryptographer review §D.2 / §D.4 #2).
  #[cfg(feature = "lookup-fold")]
  pub fn t_lookup(&self) -> Option<&[E::Scalar]> {
    self.T_lookup.as_deref()
  }

  /// GH-#5 M.GH5.7 / Pin Corrigendum #10 Q2 (Halpert 2026-05-10):
  /// folded R1CS-side `comm_W` accessor. The field is `pub(crate)` so
  /// that the augmented-circuit's `alloc_witness` site (in vendor
  /// `circuit/mod.rs`) can read it via field access; harness-side
  /// consumers (M.GH5.7 negative-test triple) need the accessor to
  /// build `NeutronAugmentedCircuitInputs::new(... Some(comm_W_fold), ...)`
  /// for the non-base-case fold-depth-≥1 augmented-circuit synthesis.
  ///
  /// Co-classified with `t_lookup()` above (a similar M.11 / M.GH5.6
  /// enabler accessor pattern). Zero-algebra-zero-FS-zero-digest;
  /// audit envelope folded into the M.GH5.1 visibility-bump bracket.
  pub fn comm_W(&self) -> Commitment<E> {
    self.comm_W
  }

  /// GH-#5 M.GH5.7 / Pin Corrigendum #10 Q2 (Halpert 2026-05-10):
  /// folded R1CS-side `comm_E` accessor. Co-classified with
  /// [`Self::comm_W`]; same M.GH5.1-visibility-bump bracket.
  pub fn comm_E(&self) -> Commitment<E> {
    self.comm_E
  }

  /// Fold the instance with another instance
  pub fn fold(
    &self,
    U2: &R1CSInstance<E>,
    comm_E: &Commitment<E>,
    r_b: &E::Scalar,
    T_out: &E::Scalar,
  ) -> Result<Self, NovaError> {
    // we need to compute the weighted sum using weights of (1-r_b) and r_b
    let comm_W = self.comm_W * (E::Scalar::ONE - r_b) + U2.comm_W * *r_b;
    let comm_E = self.comm_E * (E::Scalar::ONE - r_b) + *comm_E * *r_b;
    let X = self
      .X
      .par_iter()
      .zip(U2.X.par_iter())
      .map(|(x1, x2)| (E::Scalar::ONE - r_b) * x1 + *r_b * x2)
      .collect::<Vec<_>>();
    let u = (E::Scalar::ONE - r_b) * self.u + r_b;

    Ok(Self {
      comm_W,
      comm_E,
      T: *T_out,
      u,
      X,
      // M.GH7.0a widening (GH-#7 design pin Corrigendum #1 + #6): the
      // four comm_X fields are now `Option<Vec<Commitment<E>>>` per-table;
      // R1CS-only fold passes them through unchanged. `.clone()` required
      // since `Vec` is not `Copy`.
      #[cfg(feature = "lookup-fold")]
      comm_L: self.comm_L.clone(),
      #[cfg(feature = "lookup-fold")]
      comm_ts: self.comm_ts.clone(),
      #[cfg(feature = "lookup-fold")]
      comm_inv_w: self.comm_inv_w.clone(),
      #[cfg(feature = "lookup-fold")]
      comm_inv_t: self.comm_inv_t.clone(),
      // Multi-table extension (GH-#2, design pin §5.1): `T_lookup` is now
      // `Option<Vec<E::Scalar>>`; passthrough needs `.clone()` since Vec is
      // not `Copy`.
      #[cfg(feature = "lookup-fold")]
      T_lookup: self.T_lookup.clone(),
    })
  }

  /// Fold the instance together with a per-step lookup payload.
  ///
  /// This is the lookup-fold extension's analogue of [`Self::fold`]. The
  /// R1CS-zero side is folded identically to [`Self::fold`]; the lookup-zero
  /// side folds the four per-step lookup commitments into the running ones
  /// using the same `r_b` weight (Path β common-challenge composition,
  /// addendum §A.2.4).
  ///
  /// On the FIRST fold of a default running instance, `self.comm_L` is `None`
  /// and the lookup-side weighted sum degenerates to `0 * (1 - r_b) +
  /// payload * r_b = payload * r_b` (treating absent commitments as the
  /// identity / zero element). This matches the invariant that a default
  /// running instance carries no committed lookup data.
  ///
  /// Multi-table extension (GH-#2 M.2, design pin §1.3): `T_lookup_out` is
  /// a slice of length `k` carrying one running scalar per registered
  /// `MultiColumnLookupTable` entry, in `table_id`-canonical order. The
  /// resulting running instance stores `T_lookup = Some(Vec<…>)` of the
  /// same length, threading each per-table running claim independently per
  /// the VECTOR (C)-invariant. Single-table callers pass a one-element
  /// slice; the k=1 storage shape is byte-identical to M.1's
  /// `Some(vec![T_lookup_out])` wrap, preserving the regression contract
  /// validated by `stage_i_pri_multi_column_fold_of_two`.
  ///
  /// Per orchestrator A1: this method threads the running-scalar VECTOR
  /// only. The per-table (C)-binding `∀ j: poly_lookup_j(0) +
  /// poly_lookup_j(1) == T_lookup_running_j` is enforced at verify time
  /// (M.4 native, M.6 in-circuit) — `fold_with_lookup` itself does NOT
  /// re-check the binding.
  ///
  /// M.GH7.0a extension (GH-#7 design pin Corrigendum #1 + #6, §1.3 lines
  /// 166-241): the `payload_per_table` argument is a slice of length
  /// `k = T_lookup_out.len()`, one [`LookupPayload`] per registered table
  /// in `table_id`-canonical position. Each of the four `comm_X` running
  /// fields on the resulting `FoldedInstance` is `Some(Vec<Commitment<E>>)`
  /// of the same length, with each per-`j` entry produced by the per-
  /// table independent fold body `comm_X_j_fold = (1 - r_b) * comm_X_j_run
  /// + r_b * payload_per_table[j].comm_X` (Corrigendum #6 primitive 3 —
  /// no cross-`j` term under common `r_b`; FoldedInstance-level binding
  /// reduces to per-`(X, j)` PCS binding + RO2 collision-resistance +
  /// Path-β per-step composition, primitives 1-4). The k=1 single-table
  /// caller passes a 1-element slice; the storage shape is
  /// `Some(vec![comm_X_fold])`, byte-identical to pre-M.GH7.0a's
  /// `Some(comm_X_fold)` wrap for k=1 by §5.2 #2 (modulo the type-level
  /// `Option<Commitment<E>> → Option<Vec<Commitment<E>>>` widening).
  #[cfg(feature = "lookup-fold")]
  pub fn fold_with_lookup(
    &self,
    U2: &R1CSInstance<E>,
    comm_E: &Commitment<E>,
    r_b: &E::Scalar,
    T_out: &E::Scalar,
    payload_per_table: &[LookupPayload<E>],
    T_lookup_out: &[E::Scalar],
  ) -> Result<Self, NovaError> {
    let k = T_lookup_out.len();
    if payload_per_table.len() != k {
      return Err(NovaError::InvalidStructure);
    }
    // R1CS-zero side: identical to `Self::fold`.
    let comm_W = self.comm_W * (E::Scalar::ONE - r_b) + U2.comm_W * *r_b;
    let comm_E = self.comm_E * (E::Scalar::ONE - r_b) + *comm_E * *r_b;
    let X = self
      .X
      .par_iter()
      .zip(U2.X.par_iter())
      .map(|(x1, x2)| (E::Scalar::ONE - r_b) * x1 + *r_b * x2)
      .collect::<Vec<_>>();
    let u = (E::Scalar::ONE - r_b) * self.u + r_b;

    // Lookup-zero side: M.GH7.0a per-`j` independent fold under common r_b
    // (Corrigendum #6 primitive 3). For each j ∈ {0, ..., k-1}, fold the
    // running per-table commitment with the fresh per-table commitment
    // using the SAME r_b weight (the common Fiat-Shamir challenge — Path β,
    // addendum §A.2.4). The fold body has NO cross-`j` term: `comm_X_j_fold`
    // depends only on `running.comm_X[j]` and `payload_per_table[j].comm_X`,
    // not on any `j' ≠ j` — this is the per-table independence property
    // Corrigendum #6 primitive 3 reduces FoldedInstance-level binding to.
    //
    // At outer base (running side `None`), the per-`j` entry produces
    // `r_b * fresh_j` per `running.unwrap_or(zero)` semantics (zero
    // contributes nothing to the `(1 - r_b)` weight). This matches the
    // pre-M.GH7.0a single-table behaviour at k=1 for byte-equivalence.
    let one_minus_rb = E::Scalar::ONE - r_b;
    let zero = Commitment::<E>::default();
    // Per-`X` Vec accessor with `None → empty` semantics: at outer base
    // (running side `None`), each `running.comm_X` slot reads as `zero`,
    // matching the pre-M.GH7.0a `fold_one` `running.unwrap_or(zero)` body.
    let running_or_zero = |opt: &Option<Vec<Commitment<E>>>, j: usize| -> Commitment<E> {
      opt.as_ref().and_then(|v| v.get(j)).copied().unwrap_or(zero)
    };
    let mut comm_L_per_table_new: Vec<Commitment<E>> = Vec::with_capacity(k);
    let mut comm_ts_per_table_new: Vec<Commitment<E>> = Vec::with_capacity(k);
    let mut comm_inv_w_per_table_new: Vec<Commitment<E>> = Vec::with_capacity(k);
    let mut comm_inv_t_per_table_new: Vec<Commitment<E>> = Vec::with_capacity(k);
    for j in 0..k {
      let p = &payload_per_table[j];
      comm_L_per_table_new
        .push(running_or_zero(&self.comm_L, j) * one_minus_rb + p.comm_L * *r_b);
      comm_ts_per_table_new
        .push(running_or_zero(&self.comm_ts, j) * one_minus_rb + p.comm_ts * *r_b);
      comm_inv_w_per_table_new
        .push(running_or_zero(&self.comm_inv_w, j) * one_minus_rb + p.comm_inv_w * *r_b);
      comm_inv_t_per_table_new
        .push(running_or_zero(&self.comm_inv_t, j) * one_minus_rb + p.comm_inv_t * *r_b);
    }

    Ok(Self {
      comm_W,
      comm_E,
      T: *T_out,
      u,
      X,
      comm_L: Some(comm_L_per_table_new),
      comm_ts: Some(comm_ts_per_table_new),
      comm_inv_w: Some(comm_inv_w_per_table_new),
      comm_inv_t: Some(comm_inv_t_per_table_new),
      // Multi-table extension (GH-#2 M.2, design pin §1.3): `T_lookup` is
      // Vec-typed at the FoldedInstance level. The slice is copied
      // verbatim into a fresh Vec so the storage shape mirrors the
      // (C)-invariant exactly (one entry per registered table). For the
      // k=1 single-table caller (e.g. `prove_with_multi_column_lookup` /
      // `prove_with_lookup` passing `&[T_lookup_out]`), this produces
      // `Some(vec![T_lookup_out])`, byte-identical to M.1's storage.
      T_lookup: Some(T_lookup_out.to_vec()),
    })
  }
}

impl<E: Engine> AbsorbInRO2Trait<E> for FoldedInstance<E> {
  fn absorb_in_ro2(&self, ro: &mut E::RO2) {
    self.comm_W.absorb_in_ro2(ro);
    self.comm_E.absorb_in_ro2(ro);

    ro.absorb(self.T);

    // GH-#5 design pin §1.4 / §3.2 (W1) binding-via-hash NATIVE-SIDE
    // MIRROR: each `T_lookup[j]` is absorbed in `table_id`-canonical
    // order BETWEEN the existing `T` absorption and the `u`/`X`
    // absorptions. This is the byte-equivalent counterpart to the
    // in-circuit `AllocatedFoldedInstance::absorb_in_ro` extension at
    // `vendor/nova/src/neutron/circuit/relation.rs`. The extension is
    // gated on `feature = "lookup-fold"`: non-`lookup-fold` builds emit
    // the original absorption shape (T, u, X) without the new branch.
    //
    // At outer base where `T_lookup == None` no scalars are absorbed
    // (the sequence is empty, not skipped). The `LookupPayloadPublicMultiTable`
    // canonical-sort discipline at `relation.rs:482` (sort_by_key
    // table_id) is what makes "table_id-canonical order" a stable
    // structural pin — the order pin survives any future addition of
    // tables to the registry as long as the sort key remains.
    //
    // M.GH5.0 STAGE 0 acceptance criterion: byte-equivalent
    // squeeze output between this trait and the in-circuit
    // `absorb_in_ro` for the SAME `FoldedInstance` content.
    #[cfg(feature = "lookup-fold")]
    if let Some(t_lookup) = &self.T_lookup {
      for t_j in t_lookup {
        ro.absorb(*t_j);
      }
    }

    // GH-#7 design pin Corrigendum #18 M.GH7.5.0a path α NATIVE-SIDE MIRROR:
    // bind per-table running `comm_L` and `comm_ts` into the IVC public-input
    // hash chain. Byte-equivalent counterpart to the in-circuit
    // `AllocatedFoldedInstance::absorb_in_ro` extension at
    // `vendor/nova/src/neutron/circuit/relation.rs`. Inserted AFTER the
    // `T_lookup` block above and BEFORE the `u` / `X` absorbs below — full
    // per-table `comm_L` block first, then full per-table `comm_ts` block,
    // each in `table_id`-canonical order (which the off-circuit `Vec<...>`
    // storage shape already preserves via the
    // `LookupPayloadPublicMultiTable::sort_by_key(|t| t.table_id)` discipline
    // upstream of `prove_with_multi_table_lookup`'s Vec construction).
    //
    // `comm_inv_w` / `comm_inv_t` are intentionally NOT absorbed here per
    // Corrigendum #17 chicken-and-egg resolution (envelope-fresh against
    // envelope-`r_logup_j`, NOT bound into IVC hash chain).
    //
    // At outer base where `comm_L == None` / `comm_ts == None` no
    // commitments are absorbed (the sequence is empty, not skipped). The
    // gating is on `feature = "lookup-fold"`: non-`lookup-fold` builds emit
    // the original absorption shape (T, u, X) without the new branch.
    //
    // M.GH7.5.0a STORAGE + ABSORB only; the per-table fold update of
    // `comm_L` / `comm_ts` in the off-circuit `fold` body at
    // `relation.rs:862-884` already lands per M.GH7.0a, so the binding is
    // load-bearing immediately on the native side.
    #[cfg(feature = "lookup-fold")]
    if let Some(comm_L) = &self.comm_L {
      for c in comm_L {
        c.absorb_in_ro2(ro);
      }
    }
    #[cfg(feature = "lookup-fold")]
    if let Some(comm_ts) = &self.comm_ts {
      for c in comm_ts {
        c.absorb_in_ro2(ro);
      }
    }

    ro.absorb(self.u);
    for x in &self.X {
      ro.absorb(*x);
    }
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
      Circuit, ConstraintSystem,
    },
    provider::{hyperkzg::EvaluationEngine, Bn256EngineKZG},
    r1cs::R1CSShape,
    spartan::{direct::DirectCircuit, snark::RelaxedR1CSSNARK},
    spartan::{math::Math, polys::eq::EqPolynomial},
    traits::{circuit::NonTrivialCircuit, snark::RelaxedR1CSSNARKTrait},
  };
  use rand::rngs::OsRng;

  fn test_sat_inner<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() -> Result<(), NovaError> {
    // generate a non-trivial circuit
    let num_cons: usize = 16;
    let log_num_cons = num_cons.log_2();

    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<E::Scalar>::new(num_cons));

    // synthesize the circuit's shape
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();
    let S = Structure::new(&shape);

    // test default instance-witness pair under the structure
    let W = FoldedWitness::default(&S);
    let U = FoldedInstance::default(&S);
    S.is_sat(&ck, &U, &W)?;

    // generate a satisfying instance-witness for the r1cs
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(2)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (u, w) = cs
      .r1cs_instance_and_witness(&shape, &ck)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })?;

    // generate a random eq polynomial
    let coords = (0..log_num_cons)
      .map(|_| E::Scalar::random(&mut OsRng))
      .collect::<Vec<_>>();
    let E = EqPolynomial::new(coords).evals();

    // pad witness
    let mut W = w.W.clone();
    W.resize(S.S.num_vars, E::Scalar::ZERO);

    let W = FoldedWitness {
      W,
      r_W: w.r_W,
      E: E.clone(),
      r_E: E::Scalar::random(&mut OsRng),
    };

    let U = FoldedInstance {
      comm_W: u.comm_W,
      comm_E: E::CE::commit(&ck, &E, &W.r_E),
      T: E::Scalar::ZERO,
      X: u.X.clone(),
      u: E::Scalar::ONE,
      // Stage 1.A added these `Option<>` lookup-fold fields. The legacy
      // `test_sat_inner` exercises the R1CS-zero side only; the lookup-side
      // is `None` to mirror a default running instance with no
      // `LookupShape` attached (cf. `FoldedInstance::default`). Without
      // these initialisers the test fails to compile under
      // `--features lookup-fold`.
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

    S.is_sat(&ck, &U, &W)
  }

  #[test]
  fn test_sat() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;
    let res = test_sat_inner::<E, S>();
    assert!(res.is_ok());
  }

  /// GH-#2 M.2: `fold_with_lookup` per-table extension — k=2 fold-of-two
  /// unit test demonstrating per-table threading at the `FoldedInstance`
  /// level under the Vec-typed `T_lookup` field.
  ///
  /// Per design pin §1.3, the running-claim invariant is a VECTOR
  /// `[T_lookup_1, ..., T_lookup_k]` rather than a sum. M.1 widened
  /// `FoldedInstance::T_lookup` to `Option<Vec<E::Scalar>>`; M.2 extends
  /// `fold_with_lookup` to accept a per-table running-scalar slice and
  /// thread each scalar independently into the next-step running
  /// instance.
  ///
  /// k=1 byte-equivalence (the k=1 path produces an output where
  /// `T_lookup = Some(vec![T_lookup_out])`, matching M.1's storage
  /// shape) is preserved by the existing
  /// `stage_i_pri_multi_column_fold_of_two` regression test. This
  /// k=2 test exercises the structural-threading property the pin
  /// requires.
  ///
  /// Per orchestrator A1: M.2's positive test demonstrates per-table
  /// threading works structurally; the cross-table-cancellation
  /// negative test (pin §5.2 #6) lands at M.14 in inumbra harness, NOT
  /// here.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn fold_with_lookup_threads_per_table_running_scalars_k2() {
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

    let _rng = ChaCha20Rng::seed_from_u64(0xC1B2_F002);

    // Minimal R1CS shape via DirectCircuit (mirrors `test_sat_inner`).
    let num_cons: usize = 16;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();
    let str = Structure::new(&shape);

    // Default running U1 (T_lookup = None at outer base).
    let U1 = FoldedInstance::default(&str);

    // A satisfying fresh R1CSInstance for U2.
    let circuit2: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs2 = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut cs2);
    let (U2, _W2) = cs2.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Per-table LookupPayload with default (zero) commitments — this
    // test exercises only the `T_lookup` per-table threading path. After
    // M.GH7.0a (GH-#7 Corrigendum #1+#6) the four `comm_*` fields on
    // `FoldedInstance` are now `Option<Vec<Commitment<E>>>`; `fold_with_lookup`
    // takes a `&[LookupPayload<E>]` slice of length k.
    let payload_per_table: Vec<LookupPayload<E>> = (0..2).map(|_| LookupPayload::<E> {
      comm_L: Commitment::<E>::default(),
      comm_ts: Commitment::<E>::default(),
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![],
    }).collect();

    let comm_E = Commitment::<E>::default();
    let r_b = Scalar::from(7u64);
    let T_out = Scalar::ZERO;

    // Two distinct per-table running scalars. The choice of distinct
    // non-zero values makes the per-table threading observable:
    // a buggy implementation that collapsed both into one scalar (or
    // dropped the second) would fail the post-condition.
    let T_lookup_out_1 = Scalar::from(0xAA_AAu64);
    let T_lookup_out_2 = Scalar::from(0x55_55u64);
    let T_lookup_out: [Scalar; 2] = [T_lookup_out_1, T_lookup_out_2];

    let folded = U1
      .fold_with_lookup(&U2, &comm_E, &r_b, &T_out, &payload_per_table, &T_lookup_out)
      .expect("fold_with_lookup must succeed under k=2 per-table threading");

    // Per-table threading assertion: the resulting running instance
    // carries BOTH per-table scalars in canonical order.
    let stored = folded
      .T_lookup
      .as_ref()
      .expect("T_lookup must be populated after a fold step that carried lookup data");
    assert_eq!(
      stored.len(),
      2,
      "M.2: T_lookup must thread one scalar per registered table (k=2 here)"
    );
    assert_eq!(
      stored[0], T_lookup_out_1,
      "M.2: per-table threading must preserve table-0 running scalar"
    );
    assert_eq!(
      stored[1], T_lookup_out_2,
      "M.2: per-table threading must preserve table-1 running scalar"
    );
  }

  // ==========================================================================
  // M.GH7.0a — FoldedInstance per-table commitment Vec extension tests
  // ==========================================================================
  //
  // Discharges the slice `01-01` per `docs/research/cryptography/
  // gh-7-stage-k-compressed-snark-design-pin-2026-05-11.md` Corrigendum #1 +
  // Corrigendum #6 (§1.3 lines 166-241 — FoldedInstance-level per-table-
  // independent-fold-under-common-r_b soundness sketch).
  //
  // M.GH7.0a widens `FoldedInstance::{comm_L, comm_ts, comm_inv_w, comm_inv_t}`
  // from `Option<Commitment<E>>` to `Option<Vec<Commitment<E>>>` and lifts
  // `fold_with_lookup`'s per-`X` scalar `fold_one` body to a per-`j` loop
  // (Corrigendum #6 primitive 3). The structural mirror is the already-landed
  // `T_lookup: Option<Vec<E::Scalar>>` extension (GH-#2 M.2; pin §1.3) —
  // scalar leaf type for `T_lookup_per_table` ↔ group leaf type (`Commitment<E>`)
  // for the four `comm_X_per_table` fields; same per-`j` independent fold body.
  //
  // Tests:
  // - `test_folded_instance_per_table_vec_extension_binding` (acceptance):
  //   k=2 single fold from outer base; asserts the four `comm_X` fields are
  //   Vec-typed with two entries; per-`j` outer-base body `comm_X_j_fold =
  //   r_b * fresh_per_table[j].comm_X` (since running side is `None` → zero
  //   per `fold_one`'s `running.unwrap_or(zero)` semantics, the `(1 - r_b)`
  //   weight contributes zero at outer base).
  // - `fold_with_lookup_per_table_no_cross_j_interference` (Falsifier B
  //   negative test): corrupting `fresh_per_table[1].comm_X` (for each
  //   `X ∈ {L, ts, inv_w, inv_t}`) leaves `folded.comm_X_per_table[0]`
  //   unchanged — per-table independence per Corrigendum #6 primitive 3.
  // - `fold_with_lookup_per_table_differential_against_reference_1000_iter`:
  //   1000-iter ChaCha20Rng-seeded differential against the off-circuit
  //   reference formula `comm_X_j_fold = (1 - r_b) * running_j + r_b *
  //   fresh_j` for each `(X, j)` pair, per `.claude/rules/cryptography.md`
  //   Gadget contract ≥ 1000-iter behavioural earned-trust threshold.

  /// M.GH7.0a acceptance test — Corrigendum #6 primitive 1 + primitive 3.
  ///
  /// k=2 multi-table fold from outer base (`U1 = FoldedInstance::default`,
  /// all four `comm_X` running fields are `None`). Per-table fresh inputs
  /// have distinct commitments at each `j`; assert the folded
  /// `FoldedInstance` carries per-table `Vec<Commitment<E>>` with two
  /// entries, each equal to `r_b * fresh_per_table[j].comm_X` (Corrigendum
  /// #6 §1.3 line 200: at outer base, `comm_X_j_running_0 = identity`).
  ///
  /// This test FAILS to compile until the M.GH7.0a widening lands:
  /// (a) `FoldedInstance::{comm_L, comm_ts, comm_inv_w, comm_inv_t}` widened
  /// from `Option<Commitment<E>>` to `Option<Vec<Commitment<E>>>`; AND
  /// (b) `fold_with_lookup` accepts per-table `payload_per_table:
  /// &[LookupPayload<E>]` and runs a per-`j` independent fold loop.
  ///
  /// The compile failure IS the RED for this slice (per nWave TDD Phase 1
  /// RED_ACCEPTANCE: "business logic reason — not import/syntax error";
  /// here the failure is `FoldedInstance` shape divergence, a soundness-
  /// substrate gap surfaced by the test).
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn test_folded_instance_per_table_vec_extension_binding() {
    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;
    use crate::traits::commitment::CommitmentEngineTrait;
    type CE = <E as Engine>::CE;

    // Minimal R1CS shape via DirectCircuit (mirrors `test_sat_inner`).
    let num_cons: usize = 16;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();
    let str = Structure::new(&shape);

    // Default running U1 (all four comm_X = None at outer base).
    let U1 = FoldedInstance::<E>::default(&str);

    // A satisfying fresh R1CSInstance for U2.
    let circuit2: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs2 = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut cs2);
    let (U2, _W2) = cs2.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Two per-table payloads with DISTINCT commitments. Using committed
    // singletons under distinct scalars (`100j+1`, `100j+2`, ...) makes
    // each per-`(X, j)` slot byte-distinguishable.
    let mk_comm = |x: u64| -> Commitment<E> {
      CE::commit(&ck, &[Scalar::from(x)], &Scalar::ZERO)
    };
    let payload_per_table: Vec<LookupPayload<E>> = (0..2u64)
      .map(|j| LookupPayload::<E> {
        comm_L: mk_comm(100 * j + 1),
        comm_ts: mk_comm(100 * j + 2),
        comm_inv_w: mk_comm(100 * j + 3),
        comm_inv_t: mk_comm(100 * j + 4),
        T2_lookup: Scalar::ZERO,
        comm_values: vec![],
      })
      .collect();

    let comm_E = Commitment::<E>::default();
    let r_b = Scalar::from(7u64);
    let T_out = Scalar::ZERO;
    let T_lookup_out: [Scalar; 2] = [Scalar::from(0xAA_AAu64), Scalar::from(0x55_55u64)];

    // Per-table fold under common r_b (Corrigendum #6 primitive 3).
    let folded = U1
      .fold_with_lookup(
        &U2,
        &comm_E,
        &r_b,
        &T_out,
        &payload_per_table,
        &T_lookup_out,
      )
      .expect("fold_with_lookup must succeed under k=2 per-table commitment Vec extension");

    // --- Per-table commitment Vec extension assertions ---
    let comm_L_per_table = folded
      .comm_L
      .as_ref()
      .expect("comm_L must be populated after a fold step with lookup data");
    let comm_ts_per_table = folded
      .comm_ts
      .as_ref()
      .expect("comm_ts must be populated after a fold step with lookup data");
    let comm_inv_w_per_table = folded
      .comm_inv_w
      .as_ref()
      .expect("comm_inv_w must be populated after a fold step with lookup data");
    let comm_inv_t_per_table = folded
      .comm_inv_t
      .as_ref()
      .expect("comm_inv_t must be populated after a fold step with lookup data");

    assert_eq!(
      comm_L_per_table.len(),
      2,
      "M.GH7.0a: comm_L must be Vec<Commitment<E>> with one entry per registered table (k=2)"
    );
    assert_eq!(
      comm_ts_per_table.len(),
      2,
      "M.GH7.0a: comm_ts must be Vec<Commitment<E>> with one entry per registered table (k=2)"
    );
    assert_eq!(
      comm_inv_w_per_table.len(),
      2,
      "M.GH7.0a: comm_inv_w must be Vec<Commitment<E>> with one entry per registered table (k=2)"
    );
    assert_eq!(
      comm_inv_t_per_table.len(),
      2,
      "M.GH7.0a: comm_inv_t must be Vec<Commitment<E>> with one entry per registered table (k=2)"
    );

    // Per Corrigendum #6 primitive 1: at outer base, the per-`j` fold body
    // collapses to `comm_X_j_fold = r_b * fresh_per_table[j].comm_X` because
    // `(1 - r_b) * None` evaluates to zero via `fold_one`'s
    // `running.unwrap_or(zero)` semantics (per `relation.rs:802-803`).
    for j in 0..2 {
      assert_eq!(
        comm_L_per_table[j],
        payload_per_table[j].comm_L * r_b,
        "M.GH7.0a: comm_L_per_table[{}] must equal r_b * fresh[{}].comm_L at outer base \
         (per Corrigendum #6 primitive 1)",
        j,
        j
      );
      assert_eq!(
        comm_ts_per_table[j],
        payload_per_table[j].comm_ts * r_b,
        "M.GH7.0a: comm_ts_per_table[{}] must equal r_b * fresh[{}].comm_ts at outer base",
        j,
        j
      );
      assert_eq!(
        comm_inv_w_per_table[j],
        payload_per_table[j].comm_inv_w * r_b,
        "M.GH7.0a: comm_inv_w_per_table[{}] must equal r_b * fresh[{}].comm_inv_w at outer base",
        j,
        j
      );
      assert_eq!(
        comm_inv_t_per_table[j],
        payload_per_table[j].comm_inv_t * r_b,
        "M.GH7.0a: comm_inv_t_per_table[{}] must equal r_b * fresh[{}].comm_inv_t at outer base",
        j,
        j
      );
    }

    // T_lookup Vec threading must continue to work alongside the four new
    // per-table commitment Vecs (regression guard).
    let stored_t_lookup = folded
      .T_lookup
      .as_ref()
      .expect("T_lookup must still thread alongside the new comm_X per-table Vecs");
    assert_eq!(stored_t_lookup.len(), 2);
    assert_eq!(stored_t_lookup[0], T_lookup_out[0]);
    assert_eq!(stored_t_lookup[1], T_lookup_out[1]);
  }

  /// M.GH7.0a — Falsifier B falsifier: corrupting fresh input at `j=1`
  /// must NOT change the folded output at `j=0` (per-table independence
  /// under common `r_b`; Corrigendum #6 primitive 3).
  ///
  /// Parametrized over `X ∈ {L, ts, inv_w, inv_t}` per Mandate 5 — four
  /// input-variation runs of the SAME behavior (cross-`j` independence).
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn fold_with_lookup_per_table_no_cross_j_interference() {
    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;
    use crate::traits::commitment::CommitmentEngineTrait;
    type CE = <E as Engine>::CE;

    let num_cons: usize = 16;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();
    let str = Structure::new(&shape);

    let U1 = FoldedInstance::<E>::default(&str);

    let circuit2: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs2 = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut cs2);
    let (U2, _W2) = cs2.r1cs_instance_and_witness(&shape, &ck).unwrap();

    let mk_comm = |x: u64| -> Commitment<E> {
      CE::commit(&ck, &[Scalar::from(x)], &Scalar::ZERO)
    };
    let comm_E = Commitment::<E>::default();
    let r_b = Scalar::from(11u64);
    let T_out = Scalar::ZERO;
    let T_lookup_out: [Scalar; 2] = [Scalar::from(1u64), Scalar::from(2u64)];

    // Honest k=2 payload-per-table baseline.
    let honest = |j: u64| -> LookupPayload<E> {
      LookupPayload::<E> {
        comm_L: mk_comm(1000 * j + 1),
        comm_ts: mk_comm(1000 * j + 2),
        comm_inv_w: mk_comm(1000 * j + 3),
        comm_inv_t: mk_comm(1000 * j + 4),
        T2_lookup: Scalar::ZERO,
        comm_values: vec![],
      }
    };
    let baseline_per_table: Vec<LookupPayload<E>> = (0..2u64).map(honest).collect();
    let baseline_folded = U1
      .fold_with_lookup(
        &U2,
        &comm_E,
        &r_b,
        &T_out,
        &baseline_per_table,
        &T_lookup_out,
      )
      .expect("baseline k=2 fold must succeed");
    let baseline_L = baseline_folded.comm_L.as_ref().unwrap()[0];
    let baseline_ts = baseline_folded.comm_ts.as_ref().unwrap()[0];
    let baseline_inv_w = baseline_folded.comm_inv_w.as_ref().unwrap()[0];
    let baseline_inv_t = baseline_folded.comm_inv_t.as_ref().unwrap()[0];

    // For each X ∈ {L, ts, inv_w, inv_t}, corrupt fresh[1].comm_X and
    // assert folded[0] is unchanged in every component.
    let corruption_offset = mk_comm(0xDEAD_BEEF);
    let corruptions: [(&str, fn(&mut LookupPayload<E>, Commitment<E>)); 4] = [
      ("comm_L", |p, off| p.comm_L = p.comm_L + off),
      ("comm_ts", |p, off| p.comm_ts = p.comm_ts + off),
      ("comm_inv_w", |p, off| p.comm_inv_w = p.comm_inv_w + off),
      ("comm_inv_t", |p, off| p.comm_inv_t = p.comm_inv_t + off),
    ];

    for (label, corrupt) in &corruptions {
      let mut corrupted_per_table = baseline_per_table.clone();
      corrupt(&mut corrupted_per_table[1], corruption_offset);
      let corrupted_folded = U1
        .fold_with_lookup(
          &U2,
          &comm_E,
          &r_b,
          &T_out,
          &corrupted_per_table,
          &T_lookup_out,
        )
        .expect("corrupted-fresh[1] fold must structurally succeed (no rejection at fold body)");

      // j=0 invariant: all four comm_X_per_table[0] entries unchanged by
      // any corruption at j=1. (Corrigendum #6 primitive 3 — no cross-`j`
      // term in the linear combination.)
      assert_eq!(
        corrupted_folded.comm_L.as_ref().unwrap()[0],
        baseline_L,
        "Falsifier B: corrupting fresh[1].{} must not affect folded.comm_L_per_table[0] \
         (per-table independence under common r_b)",
        label
      );
      assert_eq!(
        corrupted_folded.comm_ts.as_ref().unwrap()[0],
        baseline_ts,
        "Falsifier B: corrupting fresh[1].{} must not affect folded.comm_ts_per_table[0]",
        label
      );
      assert_eq!(
        corrupted_folded.comm_inv_w.as_ref().unwrap()[0],
        baseline_inv_w,
        "Falsifier B: corrupting fresh[1].{} must not affect folded.comm_inv_w_per_table[0]",
        label
      );
      assert_eq!(
        corrupted_folded.comm_inv_t.as_ref().unwrap()[0],
        baseline_inv_t,
        "Falsifier B: corrupting fresh[1].{} must not affect folded.comm_inv_t_per_table[0]",
        label
      );

      // j=1 sensitivity: at least one of the four comm_X_per_table[1]
      // entries MUST change (otherwise the corruption did not propagate,
      // which would also indicate a bug — the per-`j` fold body is
      // supposed to homomorphically combine the fresh input).
      let j1_changed = corrupted_folded.comm_L.as_ref().unwrap()[1] != baseline_folded.comm_L.as_ref().unwrap()[1]
        || corrupted_folded.comm_ts.as_ref().unwrap()[1] != baseline_folded.comm_ts.as_ref().unwrap()[1]
        || corrupted_folded.comm_inv_w.as_ref().unwrap()[1] != baseline_folded.comm_inv_w.as_ref().unwrap()[1]
        || corrupted_folded.comm_inv_t.as_ref().unwrap()[1] != baseline_folded.comm_inv_t.as_ref().unwrap()[1];
      assert!(
        j1_changed,
        "Falsifier B counter-check: corrupting fresh[1].{} must change at least one \
         folded.comm_*_per_table[1] entry (homomorphic propagation of fresh input)",
        label
      );
    }
  }

  /// M.GH7.0a — 1000-iter ChaCha20-seeded differential against off-circuit
  /// reference per `.claude/rules/cryptography.md` Gadget contract.
  ///
  /// For randomized k=2 inputs (running side `None` at outer base; varied
  /// `r_b`, varied `fresh_per_table[j].comm_X`), the in-circuit
  /// `fold_with_lookup` body must agree byte-for-byte with the off-circuit
  /// reference formula `comm_X_j_fold = (1 - r_b) * running_j + r_b *
  /// fresh_j` per Corrigendum #6 primitive 1 unrolled at outer base
  /// (where `running_j` reduces to identity).
  ///
  /// Deterministic ChaCha20Rng-seeded driver per US-05 reviewer-
  /// reproducibility (`.claude/rules/cryptography.md`). 1000 iterations
  /// per the behavioural earned-trust threshold (≥ 1000 iters required;
  /// CI fails below).
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn fold_with_lookup_per_table_differential_against_reference_1000_iter() {
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
    use ff::Field as _;

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;
    use crate::traits::commitment::CommitmentEngineTrait;
    type CE = <E as Engine>::CE;

    const N_ITERATIONS: usize = 1_000;
    const SEED_M_GH7_0A: u64 = 0xC1BE_5BAD_C0DE_700A;

    let num_cons: usize = 16;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();
    let str = Structure::new(&shape);

    let U1 = FoldedInstance::<E>::default(&str);
    let circuit2: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs2 = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut cs2);
    let (U2, _W2) = cs2.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let comm_E = Commitment::<E>::default();
    let T_out = Scalar::ZERO;

    let mut rng = ChaCha20Rng::seed_from_u64(SEED_M_GH7_0A);
    for iter in 0..N_ITERATIONS {
      let r_b = Scalar::random(&mut rng);
      let payload_per_table: Vec<LookupPayload<E>> = (0..2u64)
        .map(|_j| {
          let mk_random_commit = |rng: &mut ChaCha20Rng| -> Commitment<E> {
            CE::commit(&ck, &[Scalar::random(rng)], &Scalar::ZERO)
          };
          LookupPayload::<E> {
            comm_L: mk_random_commit(&mut rng),
            comm_ts: mk_random_commit(&mut rng),
            comm_inv_w: mk_random_commit(&mut rng),
            comm_inv_t: mk_random_commit(&mut rng),
            T2_lookup: Scalar::ZERO,
            comm_values: vec![],
          }
        })
        .collect();
      let T_lookup_out: [Scalar; 2] = [Scalar::random(&mut rng), Scalar::random(&mut rng)];

      let folded = U1
        .fold_with_lookup(
          &U2,
          &comm_E,
          &r_b,
          &T_out,
          &payload_per_table,
          &T_lookup_out,
        )
        .expect("fold_with_lookup k=2 must succeed");

      // Off-circuit reference per Corrigendum #6 primitive 1, outer-base
      // unroll: `(1 - r_b) * None_j + r_b * fresh_j = r_b * fresh_j`.
      // The in-circuit `fold_with_lookup` body must agree byte-for-byte.
      for j in 0..2 {
        let expected_L = payload_per_table[j].comm_L * r_b;
        let expected_ts = payload_per_table[j].comm_ts * r_b;
        let expected_inv_w = payload_per_table[j].comm_inv_w * r_b;
        let expected_inv_t = payload_per_table[j].comm_inv_t * r_b;
        assert_eq!(
          folded.comm_L.as_ref().unwrap()[j],
          expected_L,
          "M.GH7.0a differential iter {} j {}: comm_L mismatch (seed = {:#x})",
          iter,
          j,
          SEED_M_GH7_0A
        );
        assert_eq!(
          folded.comm_ts.as_ref().unwrap()[j],
          expected_ts,
          "M.GH7.0a differential iter {} j {}: comm_ts mismatch (seed = {:#x})",
          iter,
          j,
          SEED_M_GH7_0A
        );
        assert_eq!(
          folded.comm_inv_w.as_ref().unwrap()[j],
          expected_inv_w,
          "M.GH7.0a differential iter {} j {}: comm_inv_w mismatch (seed = {:#x})",
          iter,
          j,
          SEED_M_GH7_0A
        );
        assert_eq!(
          folded.comm_inv_t.as_ref().unwrap()[j],
          expected_inv_t,
          "M.GH7.0a differential iter {} j {}: comm_inv_t mismatch (seed = {:#x})",
          iter,
          j,
          SEED_M_GH7_0A
        );
      }
    }
  }
}
