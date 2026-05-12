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
#[cfg(feature = "lookup-fold")]
use crate::Commitment;
use ff::Field;

/// An in-circuit representation of NeutronNova's FoldedInstance
///
/// The `X` field is a `Vec<AllocatedNum<E::Scalar>>` mirroring the native
/// `FoldedInstance::X: Vec<E::Scalar>` (length `S.S.num_io`). Per pin §1.4
/// Corrigendum #2 (Path W ratification, Halpert 2026-05-09): `absorb_in_ro`
/// absorbs each X scalar in canonical order, byte-equivalent to native at
/// any `num_io >= 1`. Per Corrigendum #3 (fold-arity invariant under §1.6
/// anchor): `fold` requires `X.len() == 1` because U2 is structurally
/// single-IO (single `hash.inputize` site at `circuit/mod.rs:428`); fold
/// fails-closed at `X.len() >= 2`. The arity-agnostic primitives
/// (`absorb_in_ro`, `alloc`, `default`, `default_with_lookup_k`,
/// `conditionally_select`) operate at any N.
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

  /// Per-table running lookup-witness commitment VECTOR. `None` only at outer
  /// base when no prior fold step has carried lookup data; for fold-depth >= 1
  /// always `Some` (length pinned by the shape registry's
  /// `multi_column_tables.len()`).
  ///
  /// GH-#7 design pin Corrigendum #18 (M.GH7.5.0 path α): in-circuit mirror of
  /// the off-circuit `FoldedInstance::comm_L: Option<Vec<Commitment<E>>>` at
  /// `vendor/nova/src/neutron/relation.rs:273-274`. Absorbed in `absorb_in_ro`
  /// between the `T_lookup_per_table` block and `u` in `table_id`-canonical
  /// order; the per-`j` independent fold update is M.GH7.5.0b scope (the
  /// `verify_with_multi_table_lookup` in-circuit per-table fold body that
  /// mirrors the off-circuit fold at `relation.rs:862-884` under Corrigendum
  /// #6 primitive 3). At M.GH7.5.0a (this commit) the field is purely
  /// load-bearing for STORAGE + ABSORB; `fold` and `from_lookup_fold_output`
  /// propagate it unchanged from `self` (passthrough analogous to
  /// `T_lookup_per_table` passthrough at `fold` line ~488).
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_L_per_table: Option<Vec<AllocatedNonnativePoint<E>>>,
  /// Per-table running multiplicity-vector commitment VECTOR. See
  /// `comm_L_per_table` for the Corrigendum #18 path α rationale. Mirrors
  /// off-circuit `FoldedInstance::comm_ts: Option<Vec<Commitment<E>>>` at
  /// `vendor/nova/src/neutron/relation.rs:277-278`.
  ///
  /// `comm_inv_w` / `comm_inv_t` are intentionally NOT mirrored on the
  /// in-circuit side: per Corrigendum #17 chicken-and-egg resolution they are
  /// envelope-fresh against envelope-`r_logup_j`, NOT bound into the IVC hash
  /// chain.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_ts_per_table: Option<Vec<AllocatedNonnativePoint<E>>>,

  pub(crate) u: AllocatedNum<E::Scalar>,

  /// Public IO scalar VECTOR. Length matches the underlying R1CS shape's
  /// `S.num_io`. Path W (pin §1.4 Corrigendum #2): per-element absorption
  /// in `absorb_in_ro` is byte-equivalent to native `for x in &self.X {
  /// ro.absorb(*x) }` at any `num_io >= 1`. Per Corrigendum #3, `fold`
  /// requires `len() == 1` (fail-closes at `>= 2`).
  pub(crate) X: Vec<AllocatedNum<E::Scalar>>,
}

impl<E: Engine> AllocatedFoldedInstance<E> {
  /// Allocates the given `FoldedInstance` as a witness of the circuit.
  ///
  /// GH-#5 M.GH5.3: thin wrapper for the legacy 2-arg call sites
  /// (vendor-internal tests in `circuit/lookup.rs` and the non-`lookup-fold`
  /// build of `circuit/mod.rs::alloc_witness`). Under `lookup-fold`, the
  /// `inst == None` path produces `T_lookup_per_table = None` — pre-M.GH5.3
  /// behaviour. Augmented-circuit `alloc_witness` at fold-depth-≥1
  /// lookup-fold paths invokes the explicit-k [`Self::alloc_with_k_hint`]
  /// instead so shape derivation produces a `T_lookup_per_table` of the
  /// structurally-pinned length k.
  // `#[allow(dead_code)]`: under `--features lookup-fold`, the production
  // augmented-circuit path uses `alloc_with_k_hint`; this wrapper only
  // serves the `cfg(not(feature = "lookup-fold"))` branch + vendor-internal
  // test fixtures (`circuit/lookup.rs::*`). Without the allow, the lib
  // build under lookup-fold flags this as dead.
  #[allow(dead_code)]
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    cs: CS,
    inst: Option<&FoldedInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    Self::alloc_with_k_hint(cs, inst, 0)
  }

  /// GH-#5 M.GH5.3: allocate with an explicit `lookup_fold_k_hint` for the
  /// `inst == None` path.
  ///
  /// Semantics:
  /// - When `inst` is `Some`: length of `T_lookup_per_table` is derived
  ///   from `inst.t_lookup()` (which carries the structurally-pinned k
  ///   from prior fold steps); the hint is unused.
  /// - When `inst` is `None` AND `lookup_fold_k_hint > 0`: under
  ///   `lookup-fold`, allocate `T_lookup_per_table = Some(vec![alloc; k])`
  ///   so the shape-derivation path matches the non-base-case shape that
  ///   `verify_with_multi_table_lookup` produces. Non-`lookup-fold` builds
  ///   ignore the hint (the field does not exist).
  /// - When `inst` is `None` AND `lookup_fold_k_hint == 0`: legacy path,
  ///   `T_lookup_per_table = None`. This is the path that `Self::alloc`
  ///   delegates to.
  pub fn alloc_with_k_hint<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    inst: Option<&FoldedInstance<E>>,
    lookup_fold_k_hint: usize,
  ) -> Result<Self, SynthesisError> {
    // Suppress the `unused_variables` lint when `lookup-fold` is off — the
    // hint is only consulted under `lookup-fold` (the field doesn't exist
    // in non-`lookup-fold` builds, so no None-path k-hint allocation is
    // emitted).
    #[cfg(not(feature = "lookup-fold"))]
    let _ = lookup_fold_k_hint;
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

    // GH-#5 design pin §1.4 Corrigendum #2 (Halpert 2026-05-09, Path W
    // ratification under Core Principle 7): allocate X as a Vec mirroring
    // native `FoldedInstance::X: Vec<E::Scalar>` (length `S.S.num_io`).
    // Per-element allocation is byte-equivalent to native at any N>=1 in
    // `absorb_in_ro`. Corrigendum #3 (fold-arity invariant) constrains
    // `fold` (not `alloc`) to `len() == 1`; `alloc` is arity-agnostic.
    //
    // At `inst == None` (outer base / shape derivation / base-case satisfying
    // witness), the outer-base default mirrors the augmented circuit's
    // structural `num_io == 1` invariant per Corrigendum #3 / §1.6 anchor
    // (single `hash.inputize` site at `circuit/mod.rs:431`). A length-0 Vec
    // here would trigger the γ.1 fold-arity fail-close on the honest path
    // (where `fold` is invoked against this `alloc`-produced `None`-arm
    // instance, e.g. at i=0 base case via `inputs.U == None`). The length-1
    // zero default is byte-equivalent to the prior single-AllocatedNum
    // behavior at N==1 and matches `default(cs, 1)` semantics at the
    // augmented circuit's structural `num_io`.
    let X = match inst {
      Some(inst) => inst
        .X
        .iter()
        .enumerate()
        .map(|(i, x)| {
          AllocatedNum::alloc(cs.namespace(|| format!("allocate X[{i}]")), || Ok(*x))
        })
        .collect::<Result<Vec<_>, _>>()?,
      // Halpert's third triage ruling (cluster B follow-up): use bare
      // `AllocatedNum::alloc` here, NOT `alloc_zero`. `alloc_zero` emits an
      // additional `0*0 = var` constraint (see `gadgets/utils.rs:45-54`),
      // which would force `var == 0` in the *None*-arm shape. The *Some*-arm
      // pattern above is bare `AllocatedNum::alloc` (unconstrained), so at
      // non-base IVC steps where `inst.X[0]` carries a non-zero hash output,
      // the shape derived from `alloc_zero` would fail to satisfy the witness.
      // Bare `AllocatedNum::alloc` here restores Some/None shape parity and
      // preserves Corrigendum #3 honest-path semantics for length-1 `num_io`.
      None => vec![AllocatedNum::alloc(cs.namespace(|| "allocate X[0]"), || Ok(E::Scalar::ZERO))?],
    };

    // GH-#5 design pin §3.2 / M.GH5.3: allocate `T_lookup_per_table`.
    //
    // - When `inst.t_lookup() == Some(slice)`: allocate per-element from
    //   the slice. Length is implicit (whatever the prior running U1
    //   carried). This is the fold-depth-≥1 honest path.
    // - When `inst.t_lookup() == None` AND `lookup_fold_k_hint > 0`:
    //   M.GH5.3 shape-derivation path. Allocate `Some(vec![alloc(0); k])`
    //   so `synthesize_non_base_case`'s `verify_with_multi_table_lookup`
    //   invocation finds a length-k slice when synthesising the shape via
    //   `inputs == None`. The witness values are zero (placeholder); the
    //   shape is what matters.
    // - When `inst.t_lookup() == None` AND `lookup_fold_k_hint == 0`:
    //   legacy path, produces `T_lookup_per_table = None`. Used by the
    //   non-lookup-fold internal vendor tests (TrivialCircuit /
    //   CubicCircuit `RecursiveSNARK::new`) and shape derivation when
    //   `lookup_fold_k == 0` on the `NeutronAugmentedCircuit`.
    #[cfg(feature = "lookup-fold")]
    let T_lookup_per_table = {
      let t_lookup_slice: Option<Vec<E::Scalar>> = inst
        .and_then(|inst| inst.t_lookup())
        .map(|s| s.to_vec());
      match t_lookup_slice {
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
        None if lookup_fold_k_hint > 0 => {
          let allocated = (0..lookup_fold_k_hint)
            .map(|j| {
              AllocatedNum::alloc(
                cs.namespace(|| format!("allocate T_lookup_per_table[{j}] (k-hint)")),
                || Ok(E::Scalar::ZERO),
              )
            })
            .collect::<Result<Vec<_>, _>>()?;
          Some(allocated)
        }
        None => None,
      }
    };

    // GH-#7 design pin Corrigendum #18 (M.GH7.5.0a path α): allocate
    // `comm_L_per_table` and `comm_ts_per_table` mirroring the off-circuit
    // `FoldedInstance::comm_L` / `comm_ts` Vec fields at
    // `vendor/nova/src/neutron/relation.rs:273-278`. Shape discipline matches
    // `T_lookup_per_table` above:
    //
    // - `inst.comm_L == Some(vec)`: allocate per-element from the vec.
    // - `inst.comm_L == None` AND `lookup_fold_k_hint > 0`: M.GH5.3-style
    //   shape-derivation path, allocate `Some(vec![default; k])` so
    //   `synthesize_non_base_case`'s shape matches the non-base-case shape that
    //   M.GH7.5.0b's `verify_with_multi_table_lookup` will produce.
    // - `inst.comm_L == None` AND `lookup_fold_k_hint == 0`: legacy path,
    //   `None`. Used by non-lookup-fold vendor-internal tests / shape derivation
    //   at `lookup_fold_k == 0`.
    //
    // The values are `AllocatedNonnativePoint::default(cs)` (which is
    // `AllocatedNonnativePoint::alloc(cs, None)` with no infinity constraint
    // beyond what the gadget enforces internally) for the k-hint path; this
    // mirrors the `default_with_lookup_k` `comm_W` allocation pattern below.
    #[cfg(feature = "lookup-fold")]
    let comm_L_per_table = {
      let comm_L_slice: Option<&[Commitment<E>]> = inst.and_then(|inst| inst.comm_L.as_deref());
      match comm_L_slice {
        Some(slice) => {
          let allocated = slice
            .iter()
            .enumerate()
            .map(|(j, c)| {
              AllocatedNonnativePoint::alloc(
                cs.namespace(|| format!("allocate comm_L_per_table[{j}]")),
                Some(c.to_coordinates()),
              )
            })
            .collect::<Result<Vec<_>, _>>()?;
          Some(allocated)
        }
        None if lookup_fold_k_hint > 0 => {
          let allocated = (0..lookup_fold_k_hint)
            .map(|j| {
              AllocatedNonnativePoint::alloc(
                cs.namespace(|| format!("allocate comm_L_per_table[{j}] (k-hint)")),
                None,
              )
            })
            .collect::<Result<Vec<_>, _>>()?;
          Some(allocated)
        }
        None => None,
      }
    };

    #[cfg(feature = "lookup-fold")]
    let comm_ts_per_table = {
      let comm_ts_slice: Option<&[Commitment<E>]> = inst.and_then(|inst| inst.comm_ts.as_deref());
      match comm_ts_slice {
        Some(slice) => {
          let allocated = slice
            .iter()
            .enumerate()
            .map(|(j, c)| {
              AllocatedNonnativePoint::alloc(
                cs.namespace(|| format!("allocate comm_ts_per_table[{j}]")),
                Some(c.to_coordinates()),
              )
            })
            .collect::<Result<Vec<_>, _>>()?;
          Some(allocated)
        }
        None if lookup_fold_k_hint > 0 => {
          let allocated = (0..lookup_fold_k_hint)
            .map(|j| {
              AllocatedNonnativePoint::alloc(
                cs.namespace(|| format!("allocate comm_ts_per_table[{j}] (k-hint)")),
                None,
              )
            })
            .collect::<Result<Vec<_>, _>>()?;
          Some(allocated)
        }
        None => None,
      }
    };

    Ok(Self {
      comm_W,
      comm_E,
      T,
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table,
      #[cfg(feature = "lookup-fold")]
      comm_L_per_table,
      #[cfg(feature = "lookup-fold")]
      comm_ts_per_table,
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
  #[cfg(feature = "lookup-fold")]
  pub fn default_with_lookup_k<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    num_io: usize,
    k: usize,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::default(cs.namespace(|| "allocate W"))?;
    let comm_E = comm_W.clone();
    let T = alloc_zero(cs.namespace(|| "allocate T"));
    let u = T.clone();
    // Path Y (pin §1.4 Corrigendum #4, Halpert 2026-05-09): X is a Vec
    // mirroring native `FoldedInstance::X = vec![Scalar::ZERO; S.S.num_io]`,
    // BUT each slot is `T.clone()` (sharing T's variable). This inherits
    // `alloc_zero`'s `(0)·(0) = T` zero-binding constraint for free, closing
    // the missing-constraint forgery vector that Path W's per-element
    // `AllocatedNum::alloc(... || Ok(ZERO))` had introduced (allocated a
    // fresh aux variable with witness-closure value zero but no constraint
    // enforcing it). Restores upstream microsoft/Nova's `let X = T.clone()`
    // pattern under Path W's Vec shape.
    let X = (0..num_io).map(|_| T.clone()).collect::<Vec<_>>();

    // Path Y extension to T_lookup_per_table (pin §1.4 Corrigendum #6,
    // Halpert ratification 2026-05-09): allocate a single
    // `T_lookup_zero` via `alloc_zero` and clone across all k slots.
    // Each `T_lookup_per_table[j]` shares `T_lookup_zero`'s variable,
    // inheriting `alloc_zero`'s `(0)·(0) = T_lookup_zero` zero-binding
    // constraint for free. Closes the IVC-anchor (i==0) missing-
    // constraint forgery vector for the running lookup scalar — same
    // class as Corrigendum #4's X-side gap, extended to the
    // `T_lookup_per_table` field that Corrigendum #4-A implicitly
    // omitted. The separate `T_lookup_zero` (rather than aliasing to
    // `T`) is preferred for namespace and audit-surface clarity per
    // Corrigendum #6 ("Why a dedicated `T_lookup_zero` rather than
    // aliasing to `T`"); X-T aliasing is upstream-precedented but
    // T_lookup-T aliasing would be novel cross-field reuse. Native-
    // side mirror: no change required (native `FoldedInstance::default`
    // produces `T_lookup: None` — the structurally-empty outer base
    // has no zeroed-Vec to bind; the in-circuit `Some(vec![alloc; k])`
    // shape is intentional per §2.3 / §3.2 for the constant-shape FS
    // schedule across base/non-base, and the binding obligation is
    // in-circuit-only).
    let T_lookup_zero = alloc_zero(cs.namespace(|| "allocate T_lookup_zero"));
    let T_lookup_per_table = Some((0..k).map(|_| T_lookup_zero.clone()).collect::<Vec<_>>());

    // GH-#7 design pin Corrigendum #18 (M.GH7.5.0a path α): default-zero
    // `comm_L_per_table` and `comm_ts_per_table` Vecs of length `k`,
    // analogous to `T_lookup_per_table` outer-base discipline. Use a single
    // `AllocatedNonnativePoint::default(...)` and clone across all k slots so
    // each slot shares the same zero-binding allocation (audit-surface
    // mirror of the `T_lookup_zero` discipline above and of the
    // `comm_E = comm_W.clone()` pattern at line ~251 above).
    //
    // The native-side `FoldedInstance::default` produces `comm_L = None` /
    // `comm_ts = None` (structurally-empty outer base); the in-circuit
    // `Some(vec![default; k])` shape is intentional per §2.3 / §3.2 for the
    // constant-shape FS schedule across base / non-base, consistent with the
    // existing `T_lookup_per_table` constant-shape pin (Corrigendum #6).
    let comm_L_zero = AllocatedNonnativePoint::default(cs.namespace(|| "allocate comm_L_zero"))?;
    let comm_L_per_table = Some((0..k).map(|_| comm_L_zero.clone()).collect::<Vec<_>>());
    let comm_ts_zero = AllocatedNonnativePoint::default(cs.namespace(|| "allocate comm_ts_zero"))?;
    let comm_ts_per_table = Some((0..k).map(|_| comm_ts_zero.clone()).collect::<Vec<_>>());

    Ok(Self {
      comm_W,
      comm_E,
      T,
      T_lookup_per_table,
      comm_L_per_table,
      comm_ts_per_table,
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
    num_io: usize,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::default(cs.namespace(|| "allocate W"))?;
    let comm_E = comm_W.clone();

    // Allocate T = 0. Similar to X, we do not need to check that T is well-formed
    let T = alloc_zero(cs.namespace(|| "allocate T"));

    let u = T.clone();

    // Path Y (pin §1.4 Corrigendum #4, Halpert 2026-05-09): X is a Vec
    // mirroring native `FoldedInstance::X = vec![Scalar::ZERO; S.S.num_io]`,
    // BUT each slot is `T.clone()` (sharing T's variable). This inherits
    // `alloc_zero`'s `(0)·(0) = T` zero-binding constraint for free, closing
    // the missing-constraint forgery vector that Path W's per-element
    // `AllocatedNum::alloc(... || Ok(ZERO))` had introduced (allocated a
    // fresh aux variable with witness-closure value zero but no constraint
    // enforcing it). Restores upstream microsoft/Nova's `let X = T.clone()`
    // pattern under Path W's Vec shape.
    let X = (0..num_io).map(|_| T.clone()).collect::<Vec<_>>();

    Ok(Self {
      comm_W,
      comm_E,
      T,
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table: None,
      // GH-#7 design pin Corrigendum #18 (M.GH7.5.0a path α): outer-base
      // `None` mirrors the off-circuit `FoldedInstance::default` produces
      // `comm_L = None` / `comm_ts = None`. Co-defaulted with
      // `T_lookup_per_table` per Corrigendum #6 outer-base discipline.
      #[cfg(feature = "lookup-fold")]
      comm_L_per_table: None,
      #[cfg(feature = "lookup-fold")]
      comm_ts_per_table: None,
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

    // GH-#7 design pin Corrigendum #18 M.GH7.5.0a path α: bind per-table
    // running `comm_L` and `comm_ts` into the IVC public-input hash chain.
    // Inserted AFTER the `T_lookup_per_table` block above and BEFORE the
    // `u` / `X` absorbs below — `table_id`-canonical order. comm_L's full
    // per-table block is absorbed first, then comm_ts's full per-table
    // block (mirroring the off-circuit `FoldedInstance::absorb_in_ro2`
    // byte-equivalent counterpart at
    // `vendor/nova/src/neutron/relation.rs`).
    //
    // `comm_inv_w` / `comm_inv_t` are intentionally NOT absorbed here per
    // Corrigendum #17 chicken-and-egg resolution (envelope-fresh against
    // envelope-`r_logup_j`, NOT bound into IVC hash chain).
    //
    // Outer-base `None`-skip mirrors the `T_lookup_per_table` discipline
    // above: at outer base (`comm_L_per_table == None`) no commitments
    // are absorbed (the sequence is empty, not skipped). At k > 0 with
    // `default_with_lookup_k` the field is `Some(vec![default; k])` and k
    // zero-commitments are absorbed — constant-shape FS schedule across
    // base / non-base per Corrigendum #6.
    //
    // M.GH7.5.0a STORAGE + ABSORB only; the in-circuit per-table fold
    // update (`verify_with_multi_table_lookup` body) is M.GH7.5.0b scope.
    #[cfg(feature = "lookup-fold")]
    if let Some(comm_L_pt) = &self.comm_L_per_table {
      for (j, c) in comm_L_pt.iter().enumerate() {
        c.absorb_in_ro(cs.namespace(|| format!("absorb running comm_L[{j}]")), ro)?;
      }
    }
    #[cfg(feature = "lookup-fold")]
    if let Some(comm_ts_pt) = &self.comm_ts_per_table {
      for (j, c) in comm_ts_pt.iter().enumerate() {
        c.absorb_in_ro(cs.namespace(|| format!("absorb running comm_ts[{j}]")), ro)?;
      }
    }

    ro.absorb(&self.u);
    // Path W (pin §1.4 Corrigendum #2): per-element X absorption mirrors
    // native `for x in &self.X { ro.absorb(*x) }` (relation.rs:828-830).
    // Byte-equivalent at any `num_io >= 1`. At `num_io == 1` this is
    // byte-equivalent to the prior single-AllocatedNum absorption.
    for x in &self.X {
      ro.absorb(x);
    }
    Ok(())
  }

  /// Folds self with an r1cs instance and returns the result
  ///
  /// Per pin §1.4 Corrigendum #3 (Halpert 2026-05-09, fold-arity invariant
  /// under §1.6 anchor): the legal U1.X arity at fold time is **1**. The
  /// augmented circuit's compiled R1CS shape MUST have `num_io == 1`,
  /// structurally enforced by `circuit/mod.rs:428`'s single
  /// `hash.inputize` site. This method fail-closes at `self.X.len() != 1`
  /// because U2 is structurally single-IO per §1.6 (single AllocatedNum
  /// at `AllocatedNonnativeR1CSInstance::X`); there is no native algebra
  /// to mirror at N>=2 without breaking the single `hash.inputize` IVC
  /// pattern (γ.1 ratification — α/β/δ alternatives all fail Principle 7
  /// (a/b/c) grounds). Path W's Vec-shape applies to the arity-agnostic
  /// primitives (`absorb_in_ro`, `alloc`, `default`, `select`); fold is
  /// the single-arity exception.
  pub fn fold<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    U2: &AllocatedNonnativeR1CSInstance<E>,
    r_b: &AllocatedNum<E::Scalar>,
    T_out: &AllocatedNum<E::Scalar>,
    comm_W_fold: &AllocatedNonnativePoint<E>,
    comm_E_fold: &AllocatedNonnativePoint<E>,
  ) -> Result<Self, SynthesisError> {
    // GH-#5 design pin §1.4 Corrigendum #3 (γ.1 fail-close): assert U1.X
    // arity == 1 BEFORE any algebra. U2 is structurally single-IO per
    // §1.6 anchor (augmented-circuit single hash.inputize at
    // circuit/mod.rs:428). The legal U1.X arity at fold time is 1 per
    // pin §1.4 Corrigendum #3. Path W's Vec-shape applies to
    // absorb_in_ro / select / alloc only; fold requires arity-match per
    // native FoldedInstance::fold at relation.rs:669-674.
    if self.X.len() != 1 {
      return Err(SynthesisError::Unsatisfiable(
        "AllocatedFoldedInstance::fold: self.X.len() != 1; \
         U2 is structurally single-IO per §1.6 anchor (augmented-circuit \
         single hash.inputize at circuit/mod.rs:428). The legal U1.X arity \
         at fold time is 1 per pin §1.4 Corrigendum #3. Path W's Vec-shape \
         applies to absorb_in_ro / select / alloc only; fold requires \
         arity-match per native FoldedInstance::fold at relation.rs:669-674."
          .to_string(),
      ));
    }
    let self_X = &self.X[0];

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

    // Fold the IO (single-arity per Corrigendum #3):
    // X_fold[0] = self.X[0] + r_b (U2.X - self.X[0])
    let X_fold = AllocatedNum::alloc(cs.namespace(|| "allocate X_fold"), || {
      let X = self_X
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U2_X = U2.X.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(X + r_b * (U2_X - X))
    })?;
    cs.enforce(
      || "enforce X_fold - self.X[0] = r_b (U2.X - self.X[0])",
      |lc| lc + r_b.get_variable(),
      |lc| lc + U2.X.get_variable() - self_X.get_variable(),
      |lc| lc + X_fold.get_variable() - self_X.get_variable(),
    );

    // GH-#5 design pin §3.2: `fold` passes `T_lookup_per_table` through
    // unchanged from `self`. Post-fold update of the per-table running
    // scalar vector is the responsibility of the lookup verifier path
    // (`verify_with_multi_table_lookup`); the augmented-circuit caller at
    // M.GH5.3 wires the post-fold T_lookup values into the post-fold
    // `AllocatedFoldedInstance` via a sibling helper. Mirrors the native
    // `FoldedInstance::fold` passthrough at
    // `vendor/nova/src/neutron/relation.rs:695`.
    // Path W: wrap single-arity X_fold as length-1 Vec to satisfy the
    // Vec-shaped X invariant. Byte-equivalent to the prior single-AllocatedNum
    // result at N==1 (per Corrigendum #2's "row 6 generalises to N==1").
    Ok(Self {
      comm_W: comm_W_fold.clone(),
      comm_E: comm_E_fold.clone(),
      T: T_out.clone(),
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table: self.T_lookup_per_table.clone(),
      // GH-#7 design pin Corrigendum #18 (M.GH7.5.0a path α): `fold` passes
      // `comm_L_per_table` / `comm_ts_per_table` through unchanged from
      // `self`, analogous to the `T_lookup_per_table` passthrough above.
      // The post-fold update of the per-table running commitments is the
      // responsibility of the lookup verifier path
      // (`verify_with_multi_table_lookup`); M.GH7.5.0b's augmented-circuit
      // caller will wire the post-fold commitments into the post-fold
      // `AllocatedFoldedInstance` via the widened `from_lookup_fold_output`
      // sibling. Mirrors the off-circuit `FoldedInstance::fold` post-fold
      // override at `vendor/nova/src/neutron/relation.rs:862-884`.
      #[cfg(feature = "lookup-fold")]
      comm_L_per_table: self.comm_L_per_table.clone(),
      #[cfg(feature = "lookup-fold")]
      comm_ts_per_table: self.comm_ts_per_table.clone(),
      u: u_fold,
      X: vec![X_fold],
    })
  }

  /// GH-#5 M.GH5.3 / pin §3.2: construct a post-fold `AllocatedFoldedInstance`
  /// from `verify_with_multi_table_lookup`'s output by overriding the
  /// `T_lookup_per_table` field with the per-table post-fold running
  /// scalars (`T_lookup_out_per_table`).
  ///
  /// The lookup verifier returns `LookupVerifyOutputMultiTable { U_fold,
  /// T_lookup_out_per_table }` where `U_fold` was produced via `U1.fold(...)`
  /// — and `fold` propagates `T_lookup_per_table` from `self` (U1) unchanged
  /// per pin §3.2 ("`fold` passes `T_lookup_per_table` through unchanged").
  /// To bind the post-fold per-table running scalars into the next step's
  /// hash absorption (so the (W1) binding-via-hash chain extends to the
  /// new running U), the augmented-circuit replaces `U_fold.T_lookup_per_table`
  /// with `T_lookup_out_per_table`.
  ///
  /// The other fields (`comm_W`, `comm_E`, `T`, `u`, `X`) are passed
  /// through from `u_fold` unchanged. This helper performs no constraint
  /// emission — it is purely a structural rewiring of the existing
  /// allocations.
  #[cfg(feature = "lookup-fold")]
  pub fn from_lookup_fold_output(
    u_fold: Self,
    t_lookup_out_per_table: Vec<AllocatedNum<E::Scalar>>,
    comm_L_fold_per_table: Vec<AllocatedNonnativePoint<E>>,
    comm_ts_fold_per_table: Vec<AllocatedNonnativePoint<E>>,
  ) -> Self {
    Self {
      comm_W: u_fold.comm_W,
      comm_E: u_fold.comm_E,
      T: u_fold.T,
      T_lookup_per_table: Some(t_lookup_out_per_table),
      // GH-#7 design pin Corrigendum #19 (M.GH7.5.0b path α.2): OVERRIDE
      // the M.GH7.5.0a passthrough of `u_fold.comm_L_per_table` /
      // `u_fold.comm_ts_per_table` (which inherits the PRE-fold per-table
      // commitments via `AllocatedFoldedInstance::fold`'s
      // `self.comm_L_per_table.clone()` passthrough) with the POST-fold
      // hint Vecs supplied by the per-step NIFS message and propagated
      // through `verify_with_multi_table_lookup`'s
      // `LookupVerifyOutputMultiTable`. This is the load-bearing fix for
      // the IVC↔envelope binding closure: the augmented circuit's final-
      // step hash absorption at `circuit/mod.rs:932` invokes
      // `Unew.absorb_in_ro` (which absorbs `Unew.comm_L_per_table` per
      // M.GH7.5.0a `absorb_in_ro` extension), and the off-circuit
      // `RecursiveSNARK::verify` at `mod.rs:753-773` invokes
      // `self.r_U.absorb_in_ro2` (which absorbs `self.r_U.comm_L` per
      // M.GH7.5.0a off-circuit mirror at `relation.rs:958-969`); for the
      // hash-chain check to pass at the next step, the in-circuit
      // `Unew.comm_L_per_table[j]` MUST equal the off-circuit
      // `self.r_U.comm_L[j]` byte-for-byte — which is exactly what these
      // hint Vecs supply (sourced off-circuit from
      // `NIFS::prove_with_multi_table_lookup_inner`'s `U.comm_L` /
      // `U.comm_ts` post-fold extraction, where `U = U1.fold_with_lookup`
      // at `relation.rs:862-884`). Soundness via parallel reasoning to
      // the existing `comm_W_fold` / `comm_E_fold` discipline at
      // `circuit/nifs.rs:46-55` / `:649-664`; the rejection mechanism
      // for malicious hints is (i) IVC hash-chain divergence at the next
      // step's Phase-1 hash check OR at the off-circuit
      // `RecursiveSNARK::verify` hash reconstruction, PLUS (ii)
      // envelope-side Pedersen-binding rejection at Spartan-close LogUp
      // identity verify per Corrigendum #17 path (b).
      comm_L_per_table: Some(comm_L_fold_per_table),
      comm_ts_per_table: Some(comm_ts_fold_per_table),
      u: u_fold.u,
      X: u_fold.X,
    }
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

    // Path W (pin §1.4 Corrigendum #2): per-element matched-shape select
    // mirrors native `FoldedInstance::X: Vec<E::Scalar>`. Length-mismatch
    // is a wire-up bug at the caller; fail-close rather than panicking.
    if self.X.len() != other.X.len() {
      return Err(SynthesisError::Unsatisfiable(
        "AllocatedFoldedInstance::conditionally_select: X shape mismatch"
          .to_string(),
      ));
    }
    let X = self
      .X
      .iter()
      .zip(other.X.iter())
      .enumerate()
      .map(|(i, (s, o))| {
        conditionally_select(
          cs.namespace(|| format!("X[{i}] = cond ? self.X[{i}] : other.X[{i}]")),
          s,
          o,
          condition,
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

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

    // GH-#7 design pin Corrigendum #18 M.GH7.5.0a path α: per-table
    // shape-matched select for `comm_L_per_table` / `comm_ts_per_table`,
    // identical shape-mismatch-invariant discipline as `T_lookup_per_table`
    // above. The augmented-circuit's `synthesize_base_case` allocates via
    // `default_with_lookup_k(cs, k)` (which produces `Some(vec![default; k])`
    // for both new fields) and `synthesize_non_base_case` will (at M.GH7.5.0b)
    // produce a post-fold instance with matching length-k Vecs; honest
    // synthesis preserves the invariant. Shape-mismatch indicates a wire-up
    // bug at the caller; fail-close at synthesis time rather than panic.
    #[cfg(feature = "lookup-fold")]
    let comm_L_per_table = match (&self.comm_L_per_table, &other.comm_L_per_table) {
      (None, None) => None,
      (Some(self_v), Some(other_v)) if self_v.len() == other_v.len() => {
        let selected = self_v
          .iter()
          .zip(other_v.iter())
          .enumerate()
          .map(|(j, (s, o))| {
            AllocatedNonnativePoint::conditionally_select(
              cs.namespace(|| {
                format!("comm_L_per_table[{j}] = cond ? self.comm_L_per_table[{j}] : other.comm_L_per_table[{j}]")
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
          "AllocatedFoldedInstance::conditionally_select: comm_L_per_table shape mismatch (None/Some or unequal lengths) — invariant violated by caller".to_string(),
        ));
      }
    };

    #[cfg(feature = "lookup-fold")]
    let comm_ts_per_table = match (&self.comm_ts_per_table, &other.comm_ts_per_table) {
      (None, None) => None,
      (Some(self_v), Some(other_v)) if self_v.len() == other_v.len() => {
        let selected = self_v
          .iter()
          .zip(other_v.iter())
          .enumerate()
          .map(|(j, (s, o))| {
            AllocatedNonnativePoint::conditionally_select(
              cs.namespace(|| {
                format!("comm_ts_per_table[{j}] = cond ? self.comm_ts_per_table[{j}] : other.comm_ts_per_table[{j}]")
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
          "AllocatedFoldedInstance::conditionally_select: comm_ts_per_table shape mismatch (None/Some or unequal lengths) — invariant violated by caller".to_string(),
        ));
      }
    };

    Ok(Self {
      comm_W,
      comm_E,
      T,
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table,
      #[cfg(feature = "lookup-fold")]
      comm_L_per_table,
      #[cfg(feature = "lookup-fold")]
      comm_ts_per_table,
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
  use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha20Rng,
  };

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

    // ===========================================================
    // M.GH7.5.0b path α.2 — work-item 8 / Corrigendum #19 #M.GH7.5.5
    //
    // POST-FOLD PER-TABLE HINT PROPAGATION BYTE-EQUIVALENCE TEST.
    //
    // This block discharges Corrigendum #19's empirical-close obligation
    // for the path α.2 hint-based per-table fold-result propagation.
    //
    // Claim under test: the per-step NIFS message's
    // `comm_L_fold_per_table` / `comm_ts_fold_per_table` Vec hints
    // (populated off-circuit at `prove_with_multi_table_lookup_inner` from
    // `U.comm_L` / `U.comm_ts` — the post-fold per-table commitments
    // produced by the `fold_with_lookup` body at `relation.rs:862-884`)
    // are byte-equal to the post-fold per-table running commitments on
    // the verified `FoldedInstance`. Equivalently: the hint vector that
    // the augmented circuit consumes at allocation time IS the same
    // commitment that the off-circuit `RecursiveSNARK::verify`
    // reconstructs at `mod.rs:763` via `self.r_U.absorb_in_ro2`.
    //
    // This is THE load-bearing wiring check for path α.2: if the hint
    // Vec on the NIFS message diverges from the post-fold per-table
    // commitments on `running_u_after_step_N`, then the in-circuit
    // `Unew.comm_L_per_table[j]` allocated from the hint (per the new
    // `from_lookup_fold_output` override at `circuit/relation.rs`) will
    // NOT byte-equal the off-circuit `r_U.comm_L[j]` absorbed at
    // `RecursiveSNARK::verify` time, and the IVC hash-chain check at
    // the next step's Phase-1 hash will reject. Per Corrigendum #19
    // second-order issue #6, this test fails ONLY if (a) the per-step
    // NIFS hint Vec is not byte-equal to the off-circuit fold output
    // (crafter-side bug at hint-attachment locus); (b) the
    // `from_lookup_fold_output` override does not propagate the hint
    // Vecs (crafter-side bug at the consumer site); (c) the per-table
    // absorb sequence in `absorb_in_ro` / `absorb_in_ro2` is not
    // byte-equivalent (M.GH7.5.0a regression).
    //
    // Soundness wiring: at honest synthesis, every hint must equal the
    // off-circuit fold output by construction of
    // `prove_with_multi_table_lookup_inner`'s `let
    // comm_L_fold_per_table = U.comm_L.clone();` at `nifs.rs` (where
    // `U` IS the off-circuit folded instance). The test asserts the
    // contract that the prover hint Vec equals the verifier's
    // post-fold commitment Vec — the structural equality that the
    // augmented circuit assumes when allocating the hint as
    // `AllocatedNonnativePoint`.

    // Step 1: hint Vec on `nifs1` must equal `folded_U_s1.comm_L` /
    // `comm_ts` (Vec of length k=2, table_id-canonical order).
    let nifs1_hint_comm_L = nifs1
      .comm_L_fold_per_table
      .as_ref()
      .expect(
        "M.GH7.5.5 (Corrigendum #19 path α.2): NIFS::prove_with_multi_table_lookup must \
         populate `comm_L_fold_per_table` on the per-step NIFS message under \
         lookup-fold. If None, the off-circuit hint-attachment locus is broken — \
         halt at work-item 1 in Corrigendum #19 §'Revised implementation roadmap \
         under path α.2'.",
      );
    let nifs1_hint_comm_ts = nifs1
      .comm_ts_fold_per_table
      .as_ref()
      .expect("M.GH7.5.5: comm_ts_fold_per_table must be populated on per-step NIFS message");
    let folded_U_s1_comm_L = folded_U_s1
      .comm_L
      .as_ref()
      .expect("M.GH7.0a fold-with-lookup populates folded_U_s1.comm_L");
    let folded_U_s1_comm_ts = folded_U_s1
      .comm_ts
      .as_ref()
      .expect("M.GH7.0a fold-with-lookup populates folded_U_s1.comm_ts");
    assert_eq!(
      nifs1_hint_comm_L.len(),
      folded_U_s1_comm_L.len(),
      "M.GH7.5.5: nifs1.comm_L_fold_per_table length must equal folded_U_s1.comm_L length (k=2)"
    );
    assert_eq!(
      nifs1_hint_comm_ts.len(),
      folded_U_s1_comm_ts.len(),
      "M.GH7.5.5: nifs1.comm_ts_fold_per_table length must equal folded_U_s1.comm_ts length (k=2)"
    );
    for j in 0..nifs1_hint_comm_L.len() {
      assert_eq!(
        nifs1_hint_comm_L[j], folded_U_s1_comm_L[j],
        "M.GH7.5.5 step-1 path α.2 byte-equivalence: nifs1.comm_L_fold_per_table[{j}] \
         MUST equal folded_U_s1.comm_L[{j}] (the off-circuit fold output at \
         relation.rs:862-884 produces the SAME per-table commitment that the \
         augmented circuit consumes as a hint). If this fires, the off-circuit \
         hint-attachment site at `prove_with_multi_table_lookup_inner` is wired \
         WRONG — re-verify the `let comm_L_fold_per_table = U.comm_L.clone()` \
         extraction is sourcing from the POST-fold `U`, not from `U1` or from \
         any per-step bundle's pre-fold `payload.comm_L`. \
         Per Corrigendum #19 STOP-AND-ASK trigger #M.GH7.5.5: halt at work-item \
         1 and re-audit.",
        j = j
      );
      assert_eq!(
        nifs1_hint_comm_ts[j], folded_U_s1_comm_ts[j],
        "M.GH7.5.5 step-1 path α.2 byte-equivalence: nifs1.comm_ts_fold_per_table[{j}] \
         MUST equal folded_U_s1.comm_ts[{j}]. See comm_L assertion above for \
         disposition.",
        j = j
      );
    }
    // Non-vacuous fixture: at least one per-table post-fold commitment must
    // be NON-default (default = point at infinity). Otherwise the
    // byte-equivalence test holds vacuously (zero on both sides).
    let zero_commitment = Commitment::<E>::default();
    assert!(
      nifs1_hint_comm_L
        .iter()
        .any(|c| *c != zero_commitment),
      "M.GH7.5.5 fixture non-vacuity: at least one nifs1.comm_L_fold_per_table[j] \
       must be NON-default (the k=2 multi-table fixture queries non-trivial table \
       entries; the off-circuit `fold_with_lookup` body produces \
       `r_b * payload.comm_L` at outer base which is non-default for non-zero \
       payload.comm_L). If all entries are default, the test holds vacuously — \
       fixture is broken; halt before claiming the path α.2 wiring is sound."
    );

    // Step 2: hint Vec on `nifs2` must equal `folded_U_s2.comm_L` /
    // `comm_ts` (post-fold #2 state). This exercises the mid-fold path
    // where the running side IS non-trivial (folded_U_s1 carries
    // non-zero per-table commitments).
    let nifs2_hint_comm_L = nifs2
      .comm_L_fold_per_table
      .as_ref()
      .expect("M.GH7.5.5 step-2: nifs2.comm_L_fold_per_table must be populated");
    let nifs2_hint_comm_ts = nifs2
      .comm_ts_fold_per_table
      .as_ref()
      .expect("M.GH7.5.5 step-2: nifs2.comm_ts_fold_per_table must be populated");
    let folded_U_s2_comm_L = folded_U_s2
      .comm_L
      .as_ref()
      .expect("M.GH7.0a fold-with-lookup populates folded_U_s2.comm_L");
    let folded_U_s2_comm_ts = folded_U_s2
      .comm_ts
      .as_ref()
      .expect("M.GH7.0a fold-with-lookup populates folded_U_s2.comm_ts");
    for j in 0..nifs2_hint_comm_L.len() {
      assert_eq!(
        nifs2_hint_comm_L[j], folded_U_s2_comm_L[j],
        "M.GH7.5.5 step-2 path α.2 byte-equivalence: nifs2.comm_L_fold_per_table[{j}] \
         MUST equal folded_U_s2.comm_L[{j}] under non-trivial running-side \
         (folded_U_s1 carries non-zero per-table commitments per the fold-#1 \
         output). This exercises the `(1-r_b) * running + r_b * payload` per-`j` \
         independent fold body at `relation.rs:862-884` (Corrigendum #6 \
         primitive 3) under non-zero running side. If this fires, the \
         hint-attachment site is wired wrong OR the off-circuit fold body has \
         drifted from the Corrigendum #6 algebra.",
        j = j
      );
      assert_eq!(
        nifs2_hint_comm_ts[j], folded_U_s2_comm_ts[j],
        "M.GH7.5.5 step-2 path α.2 byte-equivalence: nifs2.comm_ts_fold_per_table[{j}] \
         MUST equal folded_U_s2.comm_ts[{j}].",
        j = j
      );
    }
  }

  /// M.GH7.5.0b — Corrigendum #19 path α.2: in-circuit consumer test.
  ///
  /// Directly synthesises `AllocatedFoldedInstance::from_lookup_fold_output`
  /// with non-trivial `comm_L_fold_per_table` / `comm_ts_fold_per_table`
  /// hint Vecs and asserts:
  ///
  /// 1. The resulting `AllocatedFoldedInstance::comm_L_per_table` /
  ///    `comm_ts_per_table` carry the HINT witness values, NOT the
  ///    pre-fold passthrough values (which would equal `u_fold.comm_L_per_table`
  ///    inherited from the pre-fold `fold` body's
  ///    `self.comm_L_per_table.clone()` passthrough).
  /// 2. The in-circuit `absorb_in_ro` squeeze on the resulting instance
  ///    equals the off-circuit `absorb_in_ro2` squeeze on a native
  ///    `FoldedInstance` constructed with the hint values in the
  ///    `comm_L` / `comm_ts` Vec slots.
  ///
  /// This is the load-bearing wiring check for work-items 4-5 of
  /// Corrigendum #19's revised implementation roadmap. If this test fails,
  /// the `from_lookup_fold_output` override at `circuit/relation.rs:709`
  /// is NOT propagating the hint Vecs — surface as crafter-side bug per
  /// Corrigendum #19 second-order issue #6 disposition (b).
  ///
  /// Distinguishes M.GH7.5.0b from M.GH7.5.0a:
  /// - M.GH7.5.0a (landed): post-fold `Unew.comm_L_per_table` is the
  ///   PRE-fold passthrough (`self.comm_L_per_table.clone()`).
  /// - M.GH7.5.0b (this): post-fold `Unew.comm_L_per_table` is the
  ///   POST-fold hint Vec (overridden via `from_lookup_fold_output`).
  ///
  /// The witness-value comparison below would FAIL under M.GH7.5.0a's
  /// passthrough (which would put PRE-fold values into
  /// `result.comm_L_per_table`, NOT the hint values). The pass is
  /// load-bearing for the IVC↔envelope binding closure.
  #[test]
  fn m_gh7_5_0b_path_alpha2_from_lookup_fold_output_overrides_with_hint_vecs() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_5B_C0DE_5050);
    let ro_consts = RO2Constants::<E>::default();
    let ro_consts_circuit = RO2ConstantsCircuit::<E>::default();

    // Build a non-default base CommitmentKey from the augmented shape so
    // we can produce non-default Commitments. Mirrors stage0 fixture
    // bootstrapping (lines 1058-1071).
    use crate::traits::circuit::NonTrivialCircuit;
    let num_cons = 32usize;
    let step_circuit = NonTrivialCircuit::<Scalar>::new(num_cons);
    let augmented_shape_builder: NeutronAugmentedCircuit<'_, E, NonTrivialCircuit<Scalar>> =
      NeutronAugmentedCircuit::new(None, &step_circuit, ro_consts_circuit.clone());
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = augmented_shape_builder.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // Non-trivial hint commitments. k=2 multi-table (production
    // chunked-Strauss-Shamir shape per ADR-0021).
    let k = 2usize;
    let mk_commit = |rng: &mut ChaCha20Rng, len: usize| -> Commitment<E> {
      let scalars: Vec<Scalar> = (0..len).map(|_| Scalar::random(&mut *rng)).collect();
      <E as Engine>::CE::commit(&ck, &scalars, &Scalar::ZERO)
    };

    let comm_L_fold_hint_native: Vec<Commitment<E>> =
      (0..k).map(|_| mk_commit(&mut rng, 8)).collect();
    let comm_ts_fold_hint_native: Vec<Commitment<E>> =
      (0..k).map(|_| mk_commit(&mut rng, 8)).collect();

    // Distinct "pre-fold" passthrough commitments that the M.GH7.5.0a
    // discipline would have propagated. These MUST be different from the
    // hint values so the test can falsify a regression where
    // `from_lookup_fold_output` accidentally restores the passthrough
    // behaviour.
    let comm_L_pre_fold_passthrough: Vec<Commitment<E>> =
      (0..k).map(|_| mk_commit(&mut rng, 8)).collect();
    let comm_ts_pre_fold_passthrough: Vec<Commitment<E>> =
      (0..k).map(|_| mk_commit(&mut rng, 8)).collect();
    let zero_commitment = Commitment::<E>::default();
    for j in 0..k {
      assert_ne!(
        comm_L_fold_hint_native[j], comm_L_pre_fold_passthrough[j],
        "fixture: hint[{j}] and pre-fold passthrough must differ to falsify a \
         regression to M.GH7.5.0a passthrough behaviour",
        j = j
      );
      assert_ne!(
        comm_L_fold_hint_native[j], zero_commitment,
        "fixture non-vacuity: hint[{j}] must be non-default",
        j = j
      );
    }

    // Build a synthetic `u_fold` carrying the pre-fold passthrough
    // commitments in its `comm_L_per_table` / `comm_ts_per_table` fields
    // (this is what `AllocatedFoldedInstance::fold`'s
    // `self.comm_L_per_table.clone()` would have produced under
    // M.GH7.5.0a's passthrough discipline).
    let pre_fold_native = FoldedInstance::<E> {
      comm_W: Commitment::<E>::default(),
      comm_E: Commitment::<E>::default(),
      T: Scalar::ZERO,
      u: Scalar::ZERO,
      X: vec![Scalar::ZERO],
      comm_L: Some(comm_L_pre_fold_passthrough.clone()),
      comm_ts: Some(comm_ts_pre_fold_passthrough.clone()),
      comm_inv_w: None,
      comm_inv_t: None,
      T_lookup: Some(vec![Scalar::ZERO; k]),
    };

    // Synthesise: allocate `u_fold` with the pre-fold passthrough,
    // allocate hint vectors as `AllocatedNonnativePoint`, invoke
    // `from_lookup_fold_output`, and assert the result's
    // `comm_L_per_table` witness values match the HINT (not the
    // passthrough).
    let mut tcs = TestConstraintSystem::<Scalar>::new();
    let u_fold_alloc = AllocatedFoldedInstance::<E>::alloc_with_k_hint(
      tcs.namespace(|| "u_fold (pre-fold passthrough)"),
      Some(&pre_fold_native),
      k,
    )
    .unwrap();
    let comm_L_fold_hint_alloc: Vec<AllocatedNonnativePoint<E>> = comm_L_fold_hint_native
      .iter()
      .enumerate()
      .map(|(j, c)| {
        AllocatedNonnativePoint::alloc(
          tcs.namespace(|| format!("comm_L_fold_hint[{j}]")),
          Some(c.to_coordinates()),
        )
        .unwrap()
      })
      .collect();
    let comm_ts_fold_hint_alloc: Vec<AllocatedNonnativePoint<E>> = comm_ts_fold_hint_native
      .iter()
      .enumerate()
      .map(|(j, c)| {
        AllocatedNonnativePoint::alloc(
          tcs.namespace(|| format!("comm_ts_fold_hint[{j}]")),
          Some(c.to_coordinates()),
        )
        .unwrap()
      })
      .collect();
    let t_lookup_out_alloc: Vec<AllocatedNum<Scalar>> = (0..k)
      .map(|j| {
        AllocatedNum::alloc(tcs.namespace(|| format!("t_lookup_out[{j}]")), || {
          Ok(Scalar::ZERO)
        })
        .unwrap()
      })
      .collect();

    let result = AllocatedFoldedInstance::<E>::from_lookup_fold_output(
      u_fold_alloc,
      t_lookup_out_alloc,
      comm_L_fold_hint_alloc.clone(),
      comm_ts_fold_hint_alloc.clone(),
    );

    // Assertion (1): result's comm_L_per_table / comm_ts_per_table point
    // to the HINT allocations, NOT the pre-fold passthrough. We compare
    // BigNat limb witness values (the in-circuit representation).
    //
    // Borrow-pattern note: `result` is needed BOTH for the per-`j` witness-
    // equality assertions below AND for the subsequent `result.absorb_in_ro`
    // synthesise call in assertion (2). Clone the per-table option-vec
    // extractions so `result` itself is not partially moved.
    let result_comm_L = result
      .comm_L_per_table
      .clone()
      .expect("path α.2 sets Some");
    let result_comm_ts = result
      .comm_ts_per_table
      .clone()
      .expect("path α.2 sets Some");
    assert_eq!(
      result_comm_L.len(),
      k,
      "M.GH7.5.0b: result.comm_L_per_table.len() must equal k={k}",
      k = k
    );
    for j in 0..k {
      // Check x-coordinate BigNat limb 0 witness value matches the hint,
      // NOT the passthrough. This is a structural witness-value equality
      // check that confirms the OVERRIDE is wired correctly.
      let hint_x_limb0 = comm_L_fold_hint_alloc[j].x.as_limbs()[0]
        .value
        .clone()
        .expect("hint x limb 0 must have witness value");
      let result_x_limb0 = result_comm_L[j].x.as_limbs()[0]
        .value
        .clone()
        .expect("result x limb 0 must have witness value");
      assert_eq!(
        hint_x_limb0, result_x_limb0,
        "M.GH7.5.0b path α.2 #M.GH7.5.5 work-item 9: \
         `from_lookup_fold_output` MUST propagate the hint `comm_L_fold_per_table[{j}]` \
         into `result.comm_L_per_table[{j}]`, NOT the pre-fold passthrough \
         `u_fold.comm_L_per_table[{j}]`. The witness x-limb-0 BigInt of the \
         result MUST equal the hint's witness x-limb-0 BigInt. \
         If this fires, the override at `circuit/relation.rs:709` regressed to \
         the M.GH7.5.0a passthrough behaviour — re-audit `comm_L_per_table: \
         Some(comm_L_fold_per_table)` in the struct literal.",
        j = j
      );
    }
    for j in 0..k {
      let hint_x_limb0 = comm_ts_fold_hint_alloc[j].x.as_limbs()[0]
        .value
        .clone()
        .expect("hint comm_ts x limb 0 must have witness value");
      let result_x_limb0 = result_comm_ts[j].x.as_limbs()[0]
        .value
        .clone()
        .expect("result comm_ts x limb 0 must have witness value");
      assert_eq!(
        hint_x_limb0, result_x_limb0,
        "M.GH7.5.0b path α.2: `from_lookup_fold_output` MUST propagate \
         `comm_ts_fold_per_table[{j}]` into `result.comm_ts_per_table[{j}]`.",
        j = j
      );
    }

    // Assertion (2): the resulting in-circuit instance's `absorb_in_ro`
    // squeeze byte-equals the off-circuit `absorb_in_ro2` squeeze on the
    // SAME hint values. This validates that the OVERRIDE flows through
    // the IVC hash-chain absorption pattern correctly.
    let post_fold_native_with_hints = FoldedInstance::<E> {
      comm_W: Commitment::<E>::default(),
      comm_E: Commitment::<E>::default(),
      T: Scalar::ZERO,
      u: Scalar::ZERO,
      X: vec![Scalar::ZERO],
      comm_L: Some(comm_L_fold_hint_native.clone()),
      comm_ts: Some(comm_ts_fold_hint_native.clone()),
      comm_inv_w: None,
      comm_inv_t: None,
      T_lookup: Some(vec![Scalar::ZERO; k]),
    };
    let mut ro = <E as Engine>::RO2::new(ro_consts.clone());
    post_fold_native_with_hints.absorb_in_ro2(&mut ro);
    let h_native = ro.squeeze(NUM_HASH_BITS, false);

    let mut ro_circ = <E as Engine>::RO2Circuit::new(ro_consts_circuit.clone());
    result
      .absorb_in_ro(tcs.namespace(|| "absorb result"), &mut ro_circ)
      .expect("in-circuit absorb_in_ro must synthesise cleanly");
    let h_bits = ro_circ
      .squeeze(tcs.namespace(|| "squeeze"), NUM_HASH_BITS, false)
      .expect("squeeze must synthesise");
    let h_circuit = crate::gadgets::utils::le_bits_to_num(
      tcs.namespace(|| "bits to num"),
      &h_bits,
    )
    .expect("le_bits_to_num must synthesise");
    let h_circuit_val = h_circuit
      .get_value()
      .expect("hash witness must be assigned");

    assert!(
      tcs.is_satisfied(),
      "M.GH7.5.0b: in-circuit absorb_in_ro on the path α.2 result MUST produce a \
       satisfied CS. First unsatisfied: {:?}",
      tcs.which_is_unsatisfied()
    );
    assert_eq!(
      h_native, h_circuit_val,
      "M.GH7.5.0b path α.2 #M.GH7.5.5 byte-equivalence: in-circuit \
       `result.absorb_in_ro` squeeze MUST byte-equal the off-circuit \
       `FoldedInstance::absorb_in_ro2` squeeze on a native instance carrying \
       the SAME hint Vec values in `comm_L` / `comm_ts`. \
       (h_native, h_circuit) = ({:?}, {:?}). \
       This is the load-bearing IVC↔envelope binding closure: if these diverge, \
       the next step's Phase-1 hash check at `circuit/mod.rs:580-599` would \
       reject the honest fold under path α.2 because in-circuit `Unew` and \
       off-circuit `r_U` hash to different values. Per Corrigendum #19 \
       second-order issue #6 disposition (c): re-audit the M.GH7.5.0a-landed \
       absorb-extension wiring at `circuit/relation.rs:549-560` AND \
       `relation.rs:958-969` for per-table absorb-order byte-equivalence.",
      h_native, h_circuit_val
    );
  }

  /// M.GH5.0 STAGE 0 (0b) — Path W byte-equivalence at `num_io == 2`.
  ///
  /// Per pin §1.4 Corrigendum #2 (Halpert 2026-05-09, Path W ratification
  /// under Core Principle 7): the (W1) binding-via-hash extension is
  /// sound at any `num_io >= 1`. RO2 collision-resistance is a function
  /// of the input bitstring, not the semantic length of any sub-field.
  /// Per-element X absorption in canonical order produces a byte-equivalent
  /// hash schema at any N. This test directly constructs an
  /// `AllocatedFoldedInstance` with `X: Vec<AllocatedNum>` of length 2
  /// and a matching native `FoldedInstance` with `X: Vec<E::Scalar>` of
  /// length 2 (same scalar values), then asserts byte-equivalent RO
  /// output. Exercises only `absorb_in_ro` (arity-agnostic primitive);
  /// does NOT exercise `fold` (which is N==1-only per Corrigendum #3).
  ///
  /// Twin-substring negative test pattern not applicable here (positive
  /// equivalence test). The (0c) test below covers the fail-close path
  /// for `fold` at N>=2.
  #[test]
  fn m_gh5_0_stage0_absorb_in_ro_byte_equivalence_at_num_io_2() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_5BAD_C0DE_0B02);
    let ro_consts = RO2Constants::<E>::default();
    let ro_consts_circuit = RO2ConstantsCircuit::<E>::default();

    // Two distinct non-zero X scalars. Random per ChaCha20Rng so the test
    // exercises non-trivial absorption (not just zeros).
    let x0 = Scalar::random(&mut rng);
    let x1 = Scalar::random(&mut rng);
    assert_ne!(x0, x1, "fixture: x0 and x1 must differ");
    assert_ne!(x0, Scalar::ZERO, "fixture: x0 must be non-zero");
    assert_ne!(x1, Scalar::ZERO, "fixture: x1 must be non-zero");

    // Native FoldedInstance with X.len() == 2.
    let native_inst = FoldedInstance::<E> {
      comm_W: Commitment::<E>::default(),
      comm_E: Commitment::<E>::default(),
      T: Scalar::ZERO,
      u: Scalar::ZERO,
      X: vec![x0, x1],
      comm_L: None,
      comm_ts: None,
      comm_inv_w: None,
      comm_inv_t: None,
      T_lookup: None,
    };

    // Native squeeze.
    let mut ro = <E as Engine>::RO2::new(ro_consts.clone());
    native_inst.absorb_in_ro2(&mut ro);
    let h_native = ro.squeeze(NUM_HASH_BITS, false);

    // In-circuit squeeze via Path W's per-element absorption.
    let (h_circuit, satisfied) = circuit_absorb_squeeze(&ro_consts_circuit, &native_inst);
    assert!(
      satisfied,
      "STAGE 0 (0b) num_io==2: in-circuit absorb_in_ro CS must be satisfied. \
       Path W per-element X absorption MUST synthesise cleanly at any N>=1."
    );
    assert_eq!(
      h_native, h_circuit,
      "STAGE 0 (0b) num_io==2: native FoldedInstance::absorb_in_ro2 squeeze MUST equal \
       in-circuit AllocatedFoldedInstance::absorb_in_ro derived squeeze under Path W. \
       (h_native, h_circuit) = ({:?}, {:?}). \
       Per pin §1.4 Corrigendum #2: byte-equivalence holds for ANY N>=1.",
      h_native, h_circuit
    );
  }

  /// M.GH5.0 STAGE 0 (0c) — Corrigendum #3 fold-arity fail-close.
  ///
  /// Per pin §1.4 Corrigendum #3 (Halpert 2026-05-09, fold-arity invariant
  /// under §1.6 anchor): the legal U1.X arity at fold time is **1**
  /// (because U2 is structurally single-IO per §1.6). At N>=2 there is no
  /// native algebra to mirror without breaking the single `hash.inputize`
  /// IVC pattern; γ.1 ratification mandates `fold` fail-close with
  /// `SynthesisError::Unsatisfiable` citing the corrigendum.
  ///
  /// This test constructs an `AllocatedFoldedInstance` with `X.len() == 2`
  /// (legal under Path W's Vec-shape for `alloc` / `absorb_in_ro` /
  /// `select`), invokes `fold`, and asserts the fail-close fires BEFORE
  /// any U2 usage with the twin-substring match per pin §3.5
  /// negative-test discipline (`self.X.len() != 1` AND `§1.6`).
  #[test]
  fn m_gh5_0_fold_rejects_x_arity_two() {
    // Build a native FoldedInstance with X.len() == 2 — exercising Path
    // W's arity-agnostic alloc.
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

    // Alloc must succeed under Path W (arity-agnostic).
    let allocated = AllocatedFoldedInstance::<E>::alloc(
      cs.namespace(|| "U with X.len()==2"),
      Some(&inst_x_len_two),
    )
    .expect(
      "Path W: AllocatedFoldedInstance::alloc must accept any X.len() per Corrigendum #2; \
       only `fold` fail-closes at N>=2 per Corrigendum #3.",
    );
    assert_eq!(
      allocated.X.len(),
      2,
      "Path W: alloc preserves native X.len() == 2"
    );

    // Construct minimal U2, r_b, T_out, comm_W_fold, comm_E_fold for the
    // fold call. The γ.1 fail-close fires BEFORE any U2 usage, so the U2
    // contents don't affect the test outcome — but the call signature
    // requires concrete arguments.
    let U2 = AllocatedNonnativeR1CSInstance::<E>::alloc(cs.namespace(|| "U2"), None)
      .expect("U2 alloc must succeed");
    let r_b = AllocatedNum::alloc(cs.namespace(|| "r_b"), || Ok(Scalar::ZERO))
      .expect("r_b alloc must succeed");
    let T_out = AllocatedNum::alloc(cs.namespace(|| "T_out"), || Ok(Scalar::ZERO))
      .expect("T_out alloc must succeed");
    let comm_W_fold = AllocatedNonnativePoint::<E>::default(cs.namespace(|| "comm_W_fold"))
      .expect("comm_W_fold default must succeed");
    let comm_E_fold = AllocatedNonnativePoint::<E>::default(cs.namespace(|| "comm_E_fold"))
      .expect("comm_E_fold default must succeed");

    let result = allocated.fold(
      cs.namespace(|| "fold rejects X.len()==2"),
      &U2,
      &r_b,
      &T_out,
      &comm_W_fold,
      &comm_E_fold,
    );

    match result {
      Err(SynthesisError::Unsatisfiable(msg)) => {
        assert!(
          msg.contains("self.X.len() != 1"),
          "Corrigendum #3 fold-arity fail-close: error message must mention \
           `self.X.len() != 1`. Got: {msg}"
        );
        assert!(
          msg.contains("§1.6"),
          "Corrigendum #3 fold-arity fail-close: error message must cite the \
           §1.6 anchor (single-IO `AllocatedNonnativeR1CSInstance` invariant). \
           Got: {msg}"
        );
      }
      Ok(_) => panic!(
        "Corrigendum #3: AllocatedFoldedInstance::fold accepted X.len() == 2 — \
         fold-arity fail-close regression. fold MUST return \
         SynthesisError::Unsatisfiable at N>=2 per pin §1.4 Corrigendum #3."
      ),
      Err(other) => panic!(
        "Corrigendum #3: expected SynthesisError::Unsatisfiable; got {other:?}"
      ),
    }
  }

  /// M.GH5.0 STAGE 0 — Path Y structural-binding negative test
  /// (pin §1.4 Corrigendum #4-A, Halpert ratification 2026-05-09).
  /// M.GH5.2 extension: Corrigendum #6 (Halpert ratification 2026-05-09)
  /// — Path Y extension to `T_lookup_per_table` base-case zero-binding.
  ///
  /// Path Y replaces Path W's per-slot
  /// `AllocatedNum::alloc(... || Ok(F::ZERO))` for `X[i]` with `T.clone()`
  /// (`circuit/relation.rs:185, 195` for `default_with_lookup_k`;
  /// `:238, 249` for `default`). Under Path Y's variable-aliasing
  /// semantics, every `X[i].get_variable() == T.get_variable()`, and
  /// `T` is bound by `alloc_zero`'s `(0)·(0) = T` constraint
  /// (`vendor/nova/src/gadgets/utils.rs`). Variable identity IS the
  /// algebraic binding — `eval_lc` resolves `X[i]` to `T`'s witness
  /// value, which `alloc_zero` pins to `F::ZERO` unforgeably.
  ///
  /// The original Corrigendum #4 negative-test obligation
  /// (`Unew_base.X[0] = N ≠ 0` then assert `cs.is_satisfied() == false`)
  /// is structurally meaningless against Path Y: there is no
  /// independent aux index for `X[i]` to attack — setting `X[0] = N ≠ 0`
  /// would require `T = N ≠ 0`, which `alloc_zero` rejects directly,
  /// not the missing per-slot zero-binding the Path-W test would have
  /// exposed. `WitnessCS::enforce` is a no-op (option 4 dead) per
  /// `frontend/util_cs/witness_cs.rs:115-124`.
  ///
  /// Per Corrigendum #4-A, the amended obligation is a
  /// **structural-binding test** that catches the only regression class
  /// capable of re-opening the original missing-constraint forgery
  /// vector — namely, an edit that reverts Path Y's `T.clone()` to a
  /// fresh `AllocatedNum::alloc(..., || Ok(F::ZERO))` (or equivalent)
  /// without an accompanying `enforce(X[i] - T == 0)` constraint. The
  /// test asserts:
  ///
  ///   1. **Variable-identity binding** — `X[i].get_variable() ==
  ///      T.get_variable()` for every `i ∈ [0, num_io)` AND
  ///      `u.get_variable() == T.get_variable()`, exercised at the
  ///      structurally-relevant arities for both `default` and
  ///      `default_with_lookup_k`.
  ///   2. **Constraint-count floor** — `cs.num_constraints() >= 1` to
  ///      anchor `alloc_zero`'s `(0)·(0) = T` row. Floor (not exact)
  ///      so future `alloc_zero` refactors do not bisect through this
  ///      test.
  ///   3. **Positive `is_satisfied`** — exercises `alloc_zero`'s
  ///      `(0)·(0) = T` end-to-end against `eval_lc`; provides
  ///      regression-bisect ergonomics for sibling negative tests.
  ///
  /// If a future edit re-introduces an independent aux index for
  /// `X[i]` (whether by reverting Path Y, by lattice-folding, or by a
  /// Stwo migration), the variable-identity assertion fires
  /// algebraically before any cross-step substitution can be
  /// exercised. At that point, a new corrigendum (Corrigendum #5)
  /// MUST replace this amendment with an explicit malicious-witness
  /// test against whatever then-current CS surface supports
  /// witness-injection.
  ///
  /// Engine variants mirror `neutron/mod.rs::test_pp_digest`'s
  /// (Pallas, Bn256, Secp) sweep so soundness-class assertions are
  /// exercised across all production curve families.
  fn m_gh5_0_path_y_x_t_aliasing_binding_with<EngineForTest: Engine>() {
    type Cases = &'static [(usize, Option<usize>)];
    // Cases mirror Corrigendum #4-A item 1 — production-relevant arities
    // for both `default` and `default_with_lookup_k`. `num_io == 1` is
    // the production base case (pin §1.6 single-IO anchor); `num_io == 2`
    // is Path W's Vec at multi-IO (Corrigendum #2). `k ∈ {0, 1}` covers
    // the lookup-fold k-typed running instance at degenerate (k=0) and
    // minimal-non-trivial (k=1) shapes.
    let cases: Cases = &[
      (1, None),    // default(cs, num_io=1) — production base case
      (2, None),    // default(cs, num_io=2) — Path W's Vec at multi-IO
      (1, Some(0)), // default_with_lookup_k(cs, num_io=1, k=0)
      (1, Some(1)), // default_with_lookup_k(cs, num_io=1, k=1)
      (2, Some(1)), // default_with_lookup_k(cs, num_io=2, k=1)
      // Corrigendum #6 item 1 — production-relevant T_lookup arities
      // per ADR-0021 (k=2: T_1 merged window table + T_2 chunk-lookup
      // identity table). The intra-Vec aliasing assertion at (4) only
      // fires algebraically at k >= 2 — these cases are the load-bearing
      // ones for the regression-detection of per-slot fresh aux.
      (1, Some(2)), // default_with_lookup_k(cs, num_io=1, k=2) — production
      (2, Some(2)), // default_with_lookup_k(cs, num_io=2, k=2)
    ];

    for &(num_io, k_opt) in cases {
      let mut cs = TestConstraintSystem::<<EngineForTest as Engine>::Scalar>::new();
      let unew_base = match k_opt {
        None => AllocatedFoldedInstance::<EngineForTest>::default(
          cs.namespace(|| format!("default num_io={num_io}")),
          num_io,
        )
        .expect(
          "Path Y: AllocatedFoldedInstance::default must synthesize cleanly \
           per pin §1.4 Corrigendum #4 / #4-A",
        ),
        Some(k) => AllocatedFoldedInstance::<EngineForTest>::default_with_lookup_k(
          cs.namespace(|| format!("default_with_lookup_k num_io={num_io} k={k}")),
          num_io,
          k,
        )
        .expect(
          "Path Y: AllocatedFoldedInstance::default_with_lookup_k must synthesize \
           cleanly per pin §1.4 Corrigendum #4 / #4-A",
        ),
      };

      // (1) Variable-identity binding — Path Y's algebraic binding under
      // Corrigendum #4-A. Each `X[i]` MUST share `T`'s variable; `u`
      // MUST share `T`'s variable. Variable identity IS the binding
      // under T.clone() semantics: `eval_lc` resolves these allocations
      // to `T`'s witness value, which `alloc_zero`'s `(0)·(0) = T` row
      // pins to `F::ZERO` unforgeably.
      for i in 0..num_io {
        assert_eq!(
          unew_base.X[i].get_variable(),
          unew_base.T.get_variable(),
          "Path Y / Corrigendum #4-A item 1 (case num_io={num_io}, k={k_opt:?}): \
           `X[{i}].get_variable()` MUST equal `T.get_variable()`. A regression \
           that allocates `X[{i}]` as a fresh aux variable (e.g., reverting \
           Path Y's `T.clone()` to `AllocatedNum::alloc(... || Ok(F::ZERO))`) \
           without an accompanying `enforce(X[{i}] - T == 0)` constraint \
           re-opens the missing-constraint forgery vector closed by Path Y. \
           See pin §1.4 Corrigendum #4-A item 1.",
        );
      }
      assert_eq!(
        unew_base.u.get_variable(),
        unew_base.T.get_variable(),
        "Path Y / Corrigendum #4-A item 1 (case num_io={num_io}, k={k_opt:?}): \
         `u.get_variable()` MUST equal `T.get_variable()` per upstream \
         microsoft/Nova's clone-of-T pattern (`let u = T.clone()` at \
         circuit/relation.rs:185, 238). Same forgery-vector class as the \
         X[i] aliasing.",
      );

      // (2) Constraint-count floor — `alloc_zero` contributes the
      // `(0)·(0) = T` row that pins `T`'s witness value to `F::ZERO`.
      // After Path Y's `T.clone()`, no fresh aux for `X[i]` / `u` ⇒
      // `num_aux()` matches the upstream clone-of-T baseline. We assert
      // a floor (≥ 1 constraint) rather than exact count to avoid
      // brittle drift under future `alloc_zero` refactors. Per pin §1.4
      // Corrigendum #4-A item 2.
      // Constraint-count floor — Corrigendum #4-A item 2 + Corrigendum
      // #6 item 2 unified obligation. `alloc_zero` for `T` contributes
      // the `(0)·(0) = T` row (X-side zero-binding); under
      // `default_with_lookup_k`, `alloc_zero` for `T_lookup_zero`
      // contributes a second `(0)·(0) = T_lookup_zero` row. Floor (not
      // exact) so future `alloc_zero` refactors do not bisect through
      // this test. Empirically anchored: pre-Corrigendum-#6 baseline at
      // M.GH5.1 close was `num_constraints == 5` for `default(num_io)`
      // (k_opt == None) and `num_constraints == 5` for
      // `default_with_lookup_k(num_io, k)` (k slots had fresh aux but
      // ZERO new constraints — the gap Corrigendum #6 closes); post-
      // Corrigendum-#6: `default(num_io)` stays at 5, but
      // `default_with_lookup_k(num_io, k)` advances to 6 (one new
      // constraint from `T_lookup_zero`'s `alloc_zero`). The floor
      // enforces: lookup-aware constructor MUST emit ≥ 1 more
      // constraint than the default constructor at the same num_io.
      assert!(
        cs.num_constraints() >= 1,
        "Path Y / Corrigendum #4-A item 2 (case num_io={num_io}, k={k_opt:?}): \
         `cs.num_constraints()` MUST be >= 1 to anchor `alloc_zero`'s \
         `(0)·(0) = T` zero-binding row. A regression that drops the \
         `alloc_zero` constraint (or replaces it with a constraint-free \
         allocation) re-opens the forgery vector even with variable \
         aliasing intact — `T`'s witness would be unconstrained, and \
         every cloning `X[i]` / `u` would inherit the unconstraint. \
         Got num_constraints() = {}.",
        cs.num_constraints(),
      );
      // Corrigendum #6 item 2 — lookup-aware constructor MUST emit at
      // least one more constraint than the unallocated `T_lookup_zero`
      // baseline. The `T_lookup_zero` `alloc_zero` row is unconditional
      // in `default_with_lookup_k` (allocated before the `(0..k).map`),
      // so even at `k == 0` the constraint count strictly exceeds the
      // `default(num_io)` baseline (no `T_lookup_zero` allocation).
      // Empirical: 5 (default) → 6 (default_with_lookup_k) at M.GH5.2.
      if k_opt.is_some() {
        assert!(
          cs.num_constraints() >= 6,
          "Corrigendum #6 item 2 (case num_io={num_io}, k={k_opt:?}): \
           `default_with_lookup_k` MUST emit at least 6 constraints (the \
           pre-Corrigendum-#6 `default(num_io)` baseline of 5 + one new \
           `(0)·(0) = T_lookup_zero` row from `alloc_zero(cs.namespace(|| \
           \"allocate T_lookup_zero\"))`). A regression that drops the \
           `T_lookup_zero` `alloc_zero` (e.g., reverting to per-slot \
           `AllocatedNum::alloc(... || Ok(F::ZERO))`) re-opens the IVC- \
           anchor missing-constraint forgery vector for the running \
           lookup scalar. Got num_constraints() = {}.",
          cs.num_constraints(),
        );
      }

      // (3) Positive `is_satisfied` — exercises `alloc_zero`'s
      // `(0)·(0) = T` end-to-end against `eval_lc`. The full-stack
      // `test_pp_digest` and `test_neutron_recursive_circuit_pasta`
      // runs already provide this coverage transitively, but the
      // sibling assertion provides regression-bisect ergonomics. Per
      // pin §1.4 Corrigendum #4-A item 3.
      assert!(
        cs.is_satisfied(),
        "Path Y / Corrigendum #4-A item 3 (case num_io={num_io}, k={k_opt:?}): \
         `default` / `default_with_lookup_k` MUST produce a satisfying \
         base-case constraint system. Failure here means `alloc_zero`'s \
         `(0)·(0) = T` row evaluates inconsistently against the \
         `T.clone()`-aliased `X[i]` / `u` — a synthesis bug introduced \
         by Path Y or a downstream Corrigendum.",
      );

      // (4) Path Y extension to T_lookup_per_table — pin §1.4
      // Corrigendum #6 (Halpert ratification 2026-05-09). At
      // `default_with_lookup_k` only: the per-slot
      // `T_lookup_per_table[j]` allocations MUST share a single
      // `T_lookup_zero` aux variable (clone-across-k aliasing),
      // inheriting `alloc_zero`'s `(0)·(0) = T_lookup_zero`
      // zero-binding constraint for free. The X-side aliasing in
      // Corrigendum #4 is precedented by upstream microsoft/Nova;
      // this extension closes the analogous IVC-anchor binding gap
      // for the `T_lookup_per_table` field that Corrigendum #4-A
      // implicitly omitted. A regression that reverts to per-slot
      // `AllocatedNum::alloc(... || Ok(F::ZERO))` re-opens the
      // missing-constraint forgery vector for the running lookup
      // scalar at the IVC anchor (i==0), which subsequent fold steps
      // would absorb honestly via the (W1) §1.4 hash extension —
      // producing a corrupt running scalar throughout the chain.
      //
      // Algebraic binding under Corrigendum #6: variable identity
      // across all k slots IS the binding. `eval_lc` resolves every
      // `T_lookup_per_table[j]` to the shared `T_lookup_zero`'s
      // witness value, which `alloc_zero` pins to `F::ZERO`
      // unforgeably. Per pin §1.4 Corrigendum #6 item 1
      // (variable-identity binding) — the dispatch's adapted
      // intra-Vec form: assert all slots share a single variable,
      // catching a regression to per-slot fresh aux.
      if let Some(k) = k_opt {
        if k > 0 {
          let t_lookup = unew_base
            .T_lookup_per_table
            .as_ref()
            .expect(
              "Corrigendum #6: default_with_lookup_k MUST produce \
               T_lookup_per_table = Some(...) — None at this \
               constructor would itself be a regression.",
            );
          assert_eq!(
            t_lookup.len(),
            k,
            "Corrigendum #6 (case num_io={num_io}, k={k}): \
             T_lookup_per_table.len() MUST equal k.",
          );
          let v0 = t_lookup[0].get_variable();
          for j in 1..k {
            assert_eq!(
              t_lookup[j].get_variable(),
              v0,
              "Corrigendum #6 / pin §1.4 — T_lookup_zero variable-identity \
               binding (case num_io={num_io}, k={k}): \
               `T_lookup_per_table[{j}].get_variable()` MUST equal \
               `T_lookup_per_table[0].get_variable()` — all k slots MUST \
               share the single `T_lookup_zero` aux variable allocated \
               via `alloc_zero(cs.namespace(|| \"allocate T_lookup_zero\"))`. \
               A regression that allocates each slot via \
               `AllocatedNum::alloc(... || Ok(F::ZERO))` (per-slot fresh \
               aux without an enforcing constraint) re-opens the IVC-anchor \
               missing-constraint forgery vector for the running lookup \
               scalar — same class as Corrigendum #4's X-side gap. See pin \
               §1.4 Corrigendum #6 item 1 (\"Path Y extension to \
               T_lookup_per_table\").",
            );
          }
          // (4b) Distinctness from `T` — Halpert's recommended algebra
          // prefers a SEPARATE `T_lookup_zero` rather than aliasing to
          // `T` for namespace and audit-surface clarity (the X-T
          // aliasing is upstream-precedented; T_lookup-T aliasing would
          // be novel cross-field reuse — pin §1.4 Corrigendum #6, "Why
          // a dedicated `T_lookup_zero` rather than aliasing to `T`").
          // This assertion fires if a future edit collapses
          // `T_lookup_zero` into `T` sharing — soundness-equivalent but
          // corrigendum-departing (would require a new corrigendum).
          assert_ne!(
            v0,
            unew_base.T.get_variable(),
            "Corrigendum #6 / pin §1.4 — T_lookup_zero distinct-from-T \
             binding (case num_io={num_io}, k={k}): \
             `T_lookup_per_table[0].get_variable()` MUST NOT equal \
             `T.get_variable()` — Halpert's recommended algebra is a \
             SEPARATE `T_lookup_zero` allocation, not aliasing to `T`. \
             See pin §1.4 Corrigendum #6 (\"Why a dedicated `T_lookup_zero` \
             rather than aliasing to `T`\"): namespace and audit-surface \
             clarity. If this assertion fires, EITHER the alias was \
             collapsed into `T` (soundness-equivalent but corrigendum- \
             departing — requires a new corrigendum) OR a per-slot \
             fresh-aux regression also produced `v0 == T.get_variable()` \
             by accident (extremely unlikely but worth catching).",
          );
        }
      }
    }
  }

  #[test]
  fn m_gh5_0_path_y_x_t_aliasing_binding() {
    // Mirror `neutron/mod.rs::test_pp_digest`'s engine sweep — soundness-
    // class assertions are exercised across all production curve families
    // (Pallas, Bn256, Secp). The test is engine-generic; the helper
    // takes any `E: Engine` and constructs `AllocatedFoldedInstance::<E>`
    // via the public `default` / `default_with_lookup_k` constructors.
    m_gh5_0_path_y_x_t_aliasing_binding_with::<crate::provider::PallasEngine>();
    m_gh5_0_path_y_x_t_aliasing_binding_with::<crate::provider::Bn256EngineKZG>();
    m_gh5_0_path_y_x_t_aliasing_binding_with::<crate::provider::Secp256k1Engine>();
  }

  // =========================================================================
  // M.GH5.5 (per Pin Corrigendum #9, 2026-05-10) — Fold-of-N happy-path
  // augmented-circuit-shape differential.
  //
  // Source-of-truth: `docs/research/cryptography/gh-5-augmented-circuit-wireup-design-pin-2026-05-08.md`
  //   §1.4 — (W1) binding-via-hash + absorption-order pin.
  //   §5.5 row M.GH5.5 (Corrigendum #9) — milestone contract.
  //   §5.3 #1 — (W1)/(W2)/(W3) end-to-end discharge.
  //
  // Scope per Corrigendum #9: ≥ 4 fold steps, multi-table k=2 absent-table
  // pattern (M.11 §D.1 template), driven via direct
  // `NIFS::prove_with_multi_table_lookup` + `verify_with_multi_table_lookup`,
  // NOT via `RecursiveSNARK::prove_step` (which is GH-#7 / Stage K scope per
  // the chunk_index_in_z routing carry-forward).
  //
  // Per-step assertions per Corrigendum #9:
  //   (a) `cs.is_satisfied() == true` on augmented-circuit `absorb_in_ro`
  //        synthesis at every fold step ≥ 1.
  //   (b) `T_lookup_per_table` non-zero at every fold-depth ≥ 1 (empirical
  //        signal that (W1) binding-via-hash is being exercised end-to-end).
  //   (c) byte-equivalent native ↔ in-circuit hash output (the (W1)
  //        absorption-order invariant survives at fold-depth ≥ 4).
  //
  // Co-located with `m_gh5_0_stage0_absorb_in_ro_byte_equivalence_at_fold_depth_1_and_2`
  // because `AllocatedFoldedInstance`, `circuit_absorb_squeeze`,
  // `native_absorb_squeeze`, and `NeutronAugmentedCircuit` are all
  // vendor-internal (`mod circuit;` is private at `neutron/mod.rs:25`,
  // and `pub use` re-exports of `AllocatedFoldedInstance` are absent).
  // The Halpert-authored M.11 file header established the precedent of
  // deferring vendor visibility bumps until consumers exist (Finding 1
  // ruling §1.4); the natural sequencing for the augmented-circuit-shape
  // byte-equivalence at fold-depth ≥ 4 is to extend the vendor-internal
  // STAGE 0 fixture rather than introduce a new visibility surface.
  // =========================================================================

  /// Number of fold steps (≥ 4 per Corrigendum #9 contract).
  const M_GH5_5_NUM_FOLD_STEPS: usize = 4;

  /// k = 2 multi-table arity (production per ADR-0021).
  const M_GH5_5_K_TABLES: usize = 2;

  /// M.11 §D.1 step pattern: T_2 absent at steps 1 and 4 (1-based);
  /// 0-indexed here. The absent table is T_2 (table_id = 1) per the
  /// chunk-lookup identity-table production shape.
  const M_GH5_5_T2_ABSENT_AT_STEP: [bool; M_GH5_5_NUM_FOLD_STEPS] =
    [true, false, false, true];

  /// Deterministic seed (US-05 reviewer-reproducibility).
  const M_GH5_5_SEED: u64 = 0xC1BE_5BAD_C0DE_0055;

  /// M.GH5.5 — Fold-of-4 happy-path augmented-circuit-shape differential.
  ///
  /// Per Pin Corrigendum #9 (2026-05-10) §5.5 row M.GH5.5: discharges
  /// pin §5.3 #1 — (W1)/(W2)/(W3) end-to-end at fold-depth ≥ 4 under the
  /// multi-table k=2 absent-table pattern.
  ///
  /// At each of 4 fold steps, the test:
  ///   1. Drives one direct `NIFS::prove_with_multi_table_lookup` call
  ///      with a per-step bundle pair (T_1 always queried; T_2 absent at
  ///      steps 1 and 4, queried at steps 2 and 3 — M.11 §D.1).
  ///   2. Verifies the resulting NIFS via `verify_with_multi_table_lookup`
  ///      and asserts `verified_U == folded_U`.
  ///   3. Mirrors the post-step running U into an `AllocatedFoldedInstance`
  ///      via `circuit_absorb_squeeze` and asserts:
  ///      (a) `cs.is_satisfied() == true`
  ///      (b) byte-equivalent native ↔ in-circuit hash output
  ///      (c) `T_lookup_per_table` is `Some(&[..])` at fold-depth ≥ 1 with
  ///          at least one non-zero entry (the (W1) extension is being
  ///          exercised).
  ///
  /// STOP-AND-ASK on any failure per Corrigendum #9. `cargo test --release`
  /// per `.claude/rules/testing.md` (cryptography-tests release-mode rule).
  #[test]
  fn m_gh5_5_fold_of_n_augmented_circuit_shape_differential() {
    let mut rng = ChaCha20Rng::seed_from_u64(M_GH5_5_SEED);
    let ro_consts = RO2Constants::<E>::default();
    let ro_consts_circuit = RO2ConstantsCircuit::<E>::default();
    let pp_digest = Scalar::ZERO;

    // === R1CS shape from `NeutronAugmentedCircuit` (`num_io == 1`). ===
    // Path A per pin §1.4 corrigendum: the (W1) binding-via-hash
    // extension's intended deployment shape is `num_io == 1` (single
    // `hash.inputize` site at `circuit/mod.rs:428`). Per Corrigendum #2
    // (Path W) byte-equivalence holds at any `num_io >= 1`; we still
    // exercise the production-shape invariant.
    let num_cons = 32usize;
    let step_circuit = NonTrivialCircuit::<Scalar>::new(num_cons);
    let augmented_shape_builder: NeutronAugmentedCircuit<'_, E, NonTrivialCircuit<Scalar>> =
      NeutronAugmentedCircuit::new(None, &step_circuit, ro_consts_circuit.clone());
    let mut shape_cs: ShapeCS<E> = ShapeCS::new();
    let _ = augmented_shape_builder.synthesize(&mut shape_cs);
    let shape = shape_cs.r1cs_shape().unwrap();
    assert_eq!(
      shape.num_io(),
      1,
      "M.GH5.5: production-shape invariant — `NeutronAugmentedCircuit` \
       must produce R1CS shape with `num_io == 1` per pin §1.6 anchor."
    );

    // === Multi-table k=2 fixture (M.11 §D.1 + STAGE 0 size). ===
    // Both tables sized to 64 (production merged-table arity stand-in;
    // M.11's 1024-row T_2 is unnecessary for the byte-equivalence
    // contract — STAGE 0 used 64 and the algebra is symmetric in size).
    let table_size = 64usize;
    let table_log2 = 6usize;
    let t1_col0: Vec<Scalar> = (0..table_size).map(|_| Scalar::random(&mut rng)).collect();
    let identity: Vec<Scalar> =
      (0..table_size).map(|i| Scalar::from(i as u64)).collect();

    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

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
        // T_1: 1 value column (M.11 §D.1 — synthetic stand-in for the
        // 9-column production merged table; algebra is symmetric).
        MultiColumnLookupTable {
          table_id: 0,
          size: table_size,
          columns: vec![t1_col0.clone()],
          value_commitments: vec![<E as Engine>::CE::commit(
            &ck,
            &t1_col0,
            &Scalar::ZERO,
          )],
        },
        // T_2: 0 value columns (M.11 §D.1 — chunk-lookup identity table
        // shape per `chunk_table_as_multi_column`).
        MultiColumnLookupTable {
          table_id: 1,
          size: table_size,
          columns: Vec::new(),
          value_commitments: Vec::new(),
        },
      ],
      num_addr_columns: 1,
      num_witness_columns: M_GH5_5_K_TABLES,
      witness_ell_cached: table_log2,
    };
    let str_local = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str_local.S.clone();

    // === Per-table eq dimensions. ===
    let per_table_log2 = table_log2;
    let ell1 = per_table_log2.div_ceil(2);
    let ell2 = per_table_log2 / 2;
    let per_table_w_left = 1usize << ell1;
    let per_table_w_right = 1usize << ell2;

    // === Helper: build a satisfying witness for T_1 (1 value column). ===
    let build_present_t1 =
      |rng: &mut ChaCha20Rng,
       column: &[Scalar],
       query_indices: &[usize]|
       -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
        let mut witness_addr = vec![Scalar::ZERO; table_size];
        let mut witness_v0 = vec![Scalar::ZERO; table_size];
        let mut multiplicities = vec![Scalar::ZERO; table_size];
        for (i, &idx) in query_indices.iter().enumerate() {
          witness_addr[i] = Scalar::from(idx as u64);
          witness_v0[i] = column[idx];
          multiplicities[idx] += Scalar::ONE;
        }
        for i in query_indices.len()..table_size {
          witness_addr[i] = Scalar::ZERO;
          witness_v0[i] = column[0];
          multiplicities[0] += Scalar::ONE;
        }
        // suppress unused-rng warning on this branch — caller drives RNG.
        let _ = rng;
        (witness_addr, witness_v0, multiplicities)
      };

    // === Helper: build a "present" T_2 witness (0 value columns). ===
    // Mirrors M.11 `build_present_bundle` for `column_count == 0`:
    // witness_v0 stays zero, multiplicities still account for queries.
    let build_present_t2 = |query_indices: &[usize]| -> (Vec<Scalar>, Vec<Scalar>) {
      let mut witness_addr = vec![Scalar::ZERO; table_size];
      let mut multiplicities = vec![Scalar::ZERO; table_size];
      for (i, &idx) in query_indices.iter().enumerate() {
        witness_addr[i] = Scalar::from(idx as u64);
        multiplicities[idx] += Scalar::ONE;
      }
      for _ in query_indices.len()..table_size {
        multiplicities[0] += Scalar::ONE;
      }
      (witness_addr, multiplicities)
    };

    // === Helper: build per-step eqs. ===
    let mk_eqs = |rng: &mut ChaCha20Rng| -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
      let tau_w = Scalar::random(&mut *rng);
      let pow_w = PowPolynomial::new(&tau_w, per_table_log2);
      let combined_w = pow_w.split_evals(per_table_w_left, per_table_w_right);
      let (eq_w_left, eq_w_right) = combined_w.split_at(per_table_w_left);
      let tau_t = Scalar::random(&mut *rng);
      let pow_t = PowPolynomial::new(&tau_t, per_table_log2);
      let combined_t = pow_t.split_evals(per_table_w_left, per_table_w_right);
      let (eq_t_left, eq_t_right) = combined_t.split_at(per_table_w_left);
      (
        eq_w_left.to_vec(),
        eq_w_right.to_vec(),
        eq_t_left.to_vec(),
        eq_t_right.to_vec(),
      )
    };

    // === Helper: outer-base running_lw (sized to n_j). ===
    let mk_running_lw = || LookupRunningWitness::<E> {
      witness: vec![Scalar::ZERO; table_size],
      inv_w: vec![Scalar::ZERO; table_size],
      table: vec![Scalar::ZERO; table_size],
      multiplicities: vec![Scalar::ZERO; table_size],
      inv_t: vec![Scalar::ZERO; table_size],
      eq_w_left: vec![Scalar::ZERO; per_table_w_left],
      eq_w_right: vec![Scalar::ZERO; per_table_w_right],
      eq_t_left: vec![Scalar::ZERO; per_table_w_left],
      eq_t_right: vec![Scalar::ZERO; per_table_w_right],
    };

    // === Helper: distinct R1CS witness per step from the augmented
    // circuit, varying `r_next` to give distinct U2 instances (mirrors
    // the STAGE 0 fixture pattern at `relation.rs:846-870`). ===
    let make_r1cs = |seed: u64| {
      let inputs: NeutronAugmentedCircuitInputs<E> = NeutronAugmentedCircuitInputs::new(
        pp_digest,          // pp_digest
        Scalar::ZERO,       // i = 0 (base case)
        vec![Scalar::ZERO], // z0 (arity == 1)
        None,               // zi (base case)
        None,               // U (base case)
        None,               // ri
        Scalar::from(seed), // r_next — distinct per step
        None,               // u (base case)
        None,               // nifs (base case)
        None,               // comm_W_fold
        None,               // comm_E_fold
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

    // === Outer-base running U/W. ===
    let mut running_U = FoldedInstance::default(&str_local);
    let mut running_W = FoldedWitness::default(&str_local);
    let mut running_lws: Vec<LookupRunningWitness<E>> =
      (0..M_GH5_5_K_TABLES).map(|_| mk_running_lw()).collect();

    // === Per-step T_lookup snapshot for the audit trail. ===
    let mut t_lookup_per_step: Vec<Vec<Scalar>> = Vec::with_capacity(M_GH5_5_NUM_FOLD_STEPS);

    // === N-step fold loop. ===
    for step in 0..M_GH5_5_NUM_FOLD_STEPS {
      // ----- Build per-step bundles (T_1 always queried; T_2 per §D.1). -----
      // T_1: random query count in [1, table_size].
      let t1_query_count = ((rng.next_u32() as usize) % table_size) + 1;
      let t1_query_indices: Vec<usize> = {
        let mut pool: Vec<usize> = (0..table_size).collect();
        for i in (1..pool.len()).rev() {
          let j = (rng.next_u32() as usize) % (i + 1);
          pool.swap(i, j);
        }
        pool.truncate(t1_query_count);
        pool
      };
      let (wa_1, wv_1, m_1) = build_present_t1(&mut rng, &t1_col0, &t1_query_indices);
      let (eq_w1l, eq_w1r, eq_t1l, eq_t1r) = mk_eqs(&mut rng);

      let payload_1 = LookupPayload::<E> {
        comm_L: <E as Engine>::CE::commit(&ck, &wa_1, &Scalar::ZERO),
        comm_ts: <E as Engine>::CE::commit(&ck, &m_1, &Scalar::ZERO),
        comm_inv_w: Commitment::<E>::default(),
        comm_inv_t: Commitment::<E>::default(),
        T2_lookup: Scalar::ZERO,
        comm_values: vec![<E as Engine>::CE::commit(&ck, &wv_1, &Scalar::ZERO)],
      };
      let bundle_1 = crate::neutron::nifs::PerTableBundle::<E> {
        table_id: 0,
        payload: payload_1.clone(),
        fresh_witness_address: wa_1,
        fresh_witness_value_columns: vec![wv_1],
        fresh_multiplicities: m_1,
        fresh_eq_w_left: eq_w1l,
        fresh_eq_w_right: eq_w1r,
        fresh_eq_t_left: eq_t1l,
        fresh_eq_t_right: eq_t1r,
        running_lw: running_lws[0].clone(),
      };

      // T_2: absent at steps 1 and 4 (0-indexed 0 and 3), queried at 2 and 3.
      let (wa_2, m_2) = if M_GH5_5_T2_ABSENT_AT_STEP[step] {
        (vec![Scalar::ZERO; table_size], vec![Scalar::ZERO; table_size])
      } else {
        let t2_query_count = ((rng.next_u32() as usize) % table_size) + 1;
        let t2_query_indices: Vec<usize> = {
          let mut pool: Vec<usize> = (0..table_size).collect();
          for i in (1..pool.len()).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            pool.swap(i, j);
          }
          pool.truncate(t2_query_count);
          pool
        };
        build_present_t2(&t2_query_indices)
      };
      let (eq_w2l, eq_w2r, eq_t2l, eq_t2r) = mk_eqs(&mut rng);

      let payload_2 = LookupPayload::<E> {
        comm_L: <E as Engine>::CE::commit(&ck, &wa_2, &Scalar::ZERO),
        comm_ts: <E as Engine>::CE::commit(&ck, &m_2, &Scalar::ZERO),
        comm_inv_w: Commitment::<E>::default(),
        comm_inv_t: Commitment::<E>::default(),
        T2_lookup: Scalar::ZERO,
        comm_values: Vec::new(),
      };
      let bundle_2 = crate::neutron::nifs::PerTableBundle::<E> {
        table_id: 1,
        payload: payload_2.clone(),
        fresh_witness_address: wa_2,
        fresh_witness_value_columns: Vec::new(),
        fresh_multiplicities: m_2,
        fresh_eq_w_left: eq_w2l,
        fresh_eq_w_right: eq_w2r,
        fresh_eq_t_left: eq_t2l,
        fresh_eq_t_right: eq_t2r,
        running_lw: running_lws[1].clone(),
      };

      // ----- R1CS U2 for this step (distinct seed per step). -----
      let (u_step, w_step) = make_r1cs((step as u64) + 2);

      // ----- Direct NIFS prove (multi-table). -----
      let (nifs, (folded_U, folded_W), folded_lw_per_table) =
        NIFS::<E>::prove_with_multi_table_lookup(
          &ck,
          &ro_consts,
          &pp_digest,
          &str_local,
          &running_U,
          &running_W,
          &u_step,
          &w_step,
          &[bundle_1, bundle_2],
        )
        .unwrap_or_else(|e| {
          panic!(
            "M.GH5.5 STOP-AND-ASK: prove FAILED at step {} (1-based {}): \
             {:?} — most likely cause: absent-on-running mid-fold \
             rejection at LookupSumcheckInstance::new (M.11 §D.3 \
             Trigger 2). Halt; surface to orchestrator.",
            step,
            step + 1,
            e,
          )
        });

      // ----- Direct NIFS verify (multi-table). -----
      let public_bundles = vec![
        LookupPayloadPublicMultiTable::<E> {
          table_id: 0,
          comm_L: payload_1.comm_L,
          comm_values: payload_1.comm_values.clone(),
          comm_ts: payload_1.comm_ts,
        },
        LookupPayloadPublicMultiTable::<E> {
          table_id: 1,
          comm_L: payload_2.comm_L,
          comm_values: payload_2.comm_values.clone(),
          comm_ts: payload_2.comm_ts,
        },
      ];
      let verified_U = nifs
        .verify_with_multi_table_lookup(
          &ro_consts,
          &pp_digest,
          &str_local,
          &running_U,
          &u_step,
          &public_bundles,
        )
        .unwrap_or_else(|e| {
          panic!(
            "M.GH5.5 STOP-AND-ASK: verify FAILED at step {} (1-based {}): \
             {:?} — most likely cause: fixture-builder miscoded the \
             absent shape per pin §1.5.2 (M.11 §D.3 Trigger 4). Halt; \
             surface to orchestrator.",
            step,
            step + 1,
            e,
          )
        });
      assert_eq!(
        verified_U, folded_U,
        "M.GH5.5 step {} (1-based {}): prove/verify must agree on folded U",
        step,
        step + 1,
      );

      // ----- Per-step assertions per Corrigendum #9. -----

      // (b) T_lookup_per_table is Some(&[..]) at fold-depth ≥ 1 with at
      // least one non-zero entry. T_1 is always queried, so T_1's
      // running scalar accumulates non-zero contributions; T_2's scalar
      // also accumulates because the T_2 LogUp identity contributes
      // Σ inv_w − Σ inv_t = n_j / r_logup_j even on the absent-shape side.
      let t_lookup_slice = verified_U.t_lookup().unwrap_or_else(|| {
        panic!(
          "M.GH5.5 STOP-AND-ASK: T_lookup_per_table is None at fold-depth \
           {} (1-based step {}) — the (W1) extension is not being \
           exercised end-to-end. Halt; surface to orchestrator.",
          step + 1,
          step + 1,
        )
      });
      assert_eq!(
        t_lookup_slice.len(),
        M_GH5_5_K_TABLES,
        "M.GH5.5 step {} (1-based {}): T_lookup_per_table.len() must equal k = {}; got {}",
        step,
        step + 1,
        M_GH5_5_K_TABLES,
        t_lookup_slice.len(),
      );
      let any_nonzero = t_lookup_slice.iter().any(|s| *s != Scalar::ZERO);
      assert!(
        any_nonzero,
        "M.GH5.5 STOP-AND-ASK: at fold-depth {} (1-based step {}) all \
         T_lookup_per_table entries are zero. T_1 was queried at every \
         step, so its running scalar must be non-zero by construction. \
         All-zero indicates the running-scalar pipeline silently dropped \
         state — the (W1) binding-via-hash is not being exercised. Halt; \
         surface to orchestrator.",
        step + 1,
        step + 1,
      );
      let t_lookup_owned: Vec<Scalar> = t_lookup_slice.to_vec();
      t_lookup_per_step.push(t_lookup_owned);

      // (a) cs.is_satisfied() == true on augmented-circuit absorb_in_ro
      // synthesis AND (c) byte-equivalent native ↔ in-circuit hash output.
      let h_native = native_absorb_squeeze(&ro_consts, &verified_U);
      let (h_circuit, satisfied) = circuit_absorb_squeeze(&ro_consts_circuit, &verified_U);
      assert!(
        satisfied,
        "M.GH5.5 STOP-AND-ASK: at fold-depth {} (1-based step {}) the \
         in-circuit `absorb_in_ro` CS is NOT satisfied. The augmented- \
         circuit synthesis at fold-depth ≥ 1 with non-trivial \
         T_lookup_per_table has a constraint-system bug. Per pin §6.2 \
         halt all further GH-#5 milestones.",
        step + 1,
        step + 1,
      );
      assert_eq!(
        h_native, h_circuit,
        "M.GH5.5 STOP-AND-ASK: at fold-depth {} (1-based step {}) the \
         native `FoldedInstance::absorb_in_ro2` squeeze MUST equal the \
         in-circuit `AllocatedFoldedInstance::absorb_in_ro` derived \
         squeeze. (h_native, h_circuit) = ({:?}, {:?}). Most likely \
         cause: native and in-circuit `T_lookup_per_table` absorption \
         order disagrees, or one side handles outer-base differently \
         from the other. Per pin §6.2 halt all further GH-#5 milestones \
         and re-derive §1.4.",
        step + 1,
        step + 1,
        h_native,
        h_circuit,
      );

      // ----- Advance state for next iteration. -----
      running_U = folded_U;
      running_W = folded_W;
      for j in 0..M_GH5_5_K_TABLES {
        running_lws[j] = folded_lw_per_table[j].clone();
      }
    }

    // === Final-step audit-trail assertion. ===
    assert_eq!(
      t_lookup_per_step.len(),
      M_GH5_5_NUM_FOLD_STEPS,
      "M.GH5.5: must have collected one T_lookup snapshot per fold step",
    );

    // Defensive: the (W1) running scalar must visibly evolve across steps.
    // Two consecutive steps with all-equal T_lookup would indicate the
    // fold prover is ignoring fresh-side contributions silently. (T_2's
    // entry can stay equal across consecutive absent-on-fresh steps — the
    // n_j / r_logup_j contribution depends on the per-step rho_logup
    // challenge, which is FS-derived; we assert at least ONE pair of
    // consecutive snapshots differs in T_1's entry.)
    let any_pair_t1_evolves = (1..M_GH5_5_NUM_FOLD_STEPS)
      .any(|s| t_lookup_per_step[s][0] != t_lookup_per_step[s - 1][0]);
    assert!(
      any_pair_t1_evolves,
      "M.GH5.5: T_1's running scalar must evolve across consecutive fold \
       steps (T_1 is queried at every step with random per-step indices); \
       all-equal T_1 across 4 steps indicates the (W1) running-scalar \
       pipeline silently dropped state. T_lookup snapshots: {:?}",
      t_lookup_per_step,
    );

    // Audit-trail eprintln (non-fatal, helps reviewer-journey logs).
    eprintln!(
      "M.GH5.5 fold-of-{} happy-path GREEN (seed={M_GH5_5_SEED:#x}): \
       num_io={}, k={}, table_size={}, absent_pattern={:?}",
      M_GH5_5_NUM_FOLD_STEPS,
      str_local.S.num_io(),
      M_GH5_5_K_TABLES,
      table_size,
      M_GH5_5_T2_ABSENT_AT_STEP,
    );
  }
}
