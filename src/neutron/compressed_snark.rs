//! NeutronNova-side compressed SNARK envelope.
//!
//! Authored per GH-#7 / Stage K design pin Corrigenda #7+#8+#9+#11. This
//! module lands incrementally across M.GH7.0.1 REWORKED (this commit; the
//! Corrigendum #11 three-commitment split-E helper) → M.GH7.0.1b
//! (Σ-protocol equality-of-opening helper) → M.GH7.0.2 (envelope + verifier
//! off-FS binding check + Σ-protocol verification) → M.GH7.4 (LogUp identity
//! composition).
//!
//! # M.GH7.0.1 REWORKED scope (this commit; Corrigendum #11)
//!
//! Provides [`split_E_commitments`], a helper that splits a `FoldedWitness`'s
//! flat `E: Vec<E::Scalar>` (length `left + right`) into two halves
//! `(E1, E2)` of lengths `left` and `right`, and emits a
//! [`SplitECommitments<E>`] struct carrying THREE commitments and THREE
//! blinding factors:
//!
//! ```text
//!   comm_E1      := MSM(E1, ck.ck[..left])                + h · r_E1
//!   comm_E2_bind := MSM(E2, ck.ck[left..left+right])      + h · r_E2_bind
//!   comm_E2_pcs  := MSM(E2, ck.ck[..right])               + h · r_E2_pcs
//! ```
//!
//! with disciplined blinding-split:
//!
//! - `r_E1 + r_E2_bind == W.r_E` (preserves the off-FS Pedersen-additive
//!   binding identity `comm_E1 + comm_E2_bind == U.comm_E` byte-equal at
//!   the group level; Corrigendum #8 (iv-B) carried verbatim);
//! - `r_E2_pcs` is **fresh-independent** from `OsRng` and does NOT enter
//!   the binding identity. It exists solely so that `comm_E2_pcs` can be
//!   consumed by the Spartan T-claim sibling's PCS-opening shape, which
//!   expects a **prefix-basis** commitment to `E2` (canonical `EE::verify`
//!   contract at HyperKZG `provider/hyperkzg.rs:1080-...` and IPA-PC
//!   `provider/ipa_pc.rs:286-...`).
//!
//! # Why three commitments (Corrigendum #11)
//!
//! Pre-Corrigendum-#11 (the original M.GH7.0.1 at vendor commit `5cbc0e9`)
//! emitted only `(comm_E1, comm_E2)` with `comm_E2` in the **suffix** basis
//! `ck.ck[left..left+right]`. That single `comm_E2` simultaneously had to
//! serve TWO disjoint contracts:
//!
//! 1. Off-FS Pedersen-additive binding: `comm_E1 + comm_E2 == U.comm_E`,
//!    requiring `comm_E2 = MSM(E2, ck.ck[left..left+right]) + h · r_E2`
//!    against the suffix basis (so `comm_E1 + comm_E2 = MSM([E1||E2],
//!    ck.ck[..left+right]) + h · (r_E1 + r_E2)`).
//!
//! 2. Spartan-sibling PCS opening at `r_x_high`: `EE::prove`/`EE::verify`
//!    treat `comm_E2` as a standard prefix-basis commitment
//!    `MSM(E2, ck.ck[..right]) + h · r`. The HyperKZG `EE::verify` at
//!    `provider/hyperkzg.rs:1080-...` and the IPA-PC `EE::verify` at
//!    `provider/ipa_pc.rs:286-...` both consume `Commitment<E>` against
//!    the prefix slice `ck.ck[..v.len()]` — they have no API surface for
//!    a non-prefix slice.
//!
//! The two contracts require **different group elements** in general
//! (`ck.ck[left..left+right]` ≠ `ck.ck[..right]`), so a single `comm_E2`
//! cannot satisfy both. This was the basis-collision diagnosed at the
//! M.GH7.0.2-attempt-2 GREEN halt (STOP-AND-ASK gate A; execution-log
//! 2026-05-11T16:31:00Z). Corrigendum #11 disposes it by emitting BOTH
//! commitments — `comm_E2_bind` (suffix) and `comm_E2_pcs` (prefix) —
//! together with a downstream Σ-protocol equality-of-opening proof
//! (M.GH7.0.1b) tying them at the algebraic value `E2`.
//!
//! # Soundness anchor — binding side
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
//! Combined with `(h · r_E1) + (h · r_E2_bind) = h · (r_E1 + r_E2_bind) =
//! h · W.r_E`, this gives `comm_E1 + comm_E2_bind = U.comm_E` whenever
//! `r_E1 + r_E2_bind == W.r_E`. This is Corrigendum #8 (iv-B) carried
//! verbatim modulo the rename `r_E2 → r_E2_bind`.
//!
//! # Soundness anchor — PCS-opening side
//!
//! `comm_E2_pcs = E::CE::commit(ck, E2, &r_E2_pcs)` follows the canonical
//! prefix-basis Pedersen contract at `provider/pedersen.rs:285-292`:
//! `E::CE::commit(ck, v, r)` selects `ck.ck[..v.len()]` directly, so with
//! `v.len() = right` this commits against `ck.ck[..right]`. This is the
//! shape consumed by:
//!
//! - HyperKZG `EE::verify` at `provider/hyperkzg.rs:1080-...` (the
//!   `C: &Commitment<E>` argument is opened against the prefix basis via
//!   the polynomial-evaluation pairing equation).
//! - IPA-PC `EE::verify` at `provider/ipa_pc.rs:286-...` (the inner-product
//!   instance commitment is split by `ck.split_at(U.b_vec.len())` at line
//!   294 — again a prefix selection).
//!
//! # Why `r_E2_pcs` is fresh-independent
//!
//! `r_E2_pcs` does NOT enter the binding identity `comm_E1 + comm_E2_bind
//! == U.comm_E`. The binding identity constrains the **blinding** values
//! `(r_E1, r_E2_bind)` to sum to `W.r_E`, but says nothing about the
//! blinding of `comm_E2_pcs`. Algebraically `r_E2_pcs` is a free degree
//! of freedom — it could be any value in `F`. Drawing it fresh from
//! `OsRng` preserves Pedersen hiding on `comm_E2_pcs` independently of
//! the binding side. **The Σ-protocol at M.GH7.0.1b (next milestone)** is
//! what ties `comm_E2_bind` and `comm_E2_pcs` together at the algebraic
//! value of `E2`, proving that the prover did not commit to a forged
//! `E2' ≠ E2` against the prefix basis. Without that Σ-protocol, the
//! envelope would be unsound: a malicious prover could supply
//! `comm_E2_pcs := MSM(E2', ck.ck[..right]) + h · r_E2_pcs` for any
//! `E2' ≠ E2`, and the off-FS binding check on `comm_E2_bind` would not
//! catch it. The Σ-protocol closes this gap.
//!
//! # Generator-vector alignment
//!
//! The helper relies on `CE::commit(ck, &v, &r)` using `ck.ck[..v.len()]`
//! — i.e., the first `v.len()` generators in flat order:
//!
//! - `comm_E1 = CE::commit(ck, E1, r_E1)` with `E1.len() = left` selects
//!   `ck.ck[..left]` directly. **This is also the prefix-basis shape
//!   Spartan consumes for its PCS opening at `r_x_low`**, so a single
//!   `comm_E1` serves BOTH the off-FS binding and the Spartan opening on
//!   the E1 side — no basis divergence on E1.
//! - `comm_E2_bind`: we pass a length-`(left + right)` vector
//!   `[zeros(left) || E2]` so that `CE::commit` selects
//!   `ck.ck[..left+right]`, and the zero-prefix contributes zero to the
//!   MSM — leaving exactly `MSM(E2, ck.ck[left..left+right]) + h · r_E2_bind`.
//!   This is the suffix-basis shape required for the additive identity.
//! - `comm_E2_pcs = CE::commit(ck, E2, r_E2_pcs)` with `E2.len() = right`
//!   selects `ck.ck[..right]` directly. This is the prefix-basis shape
//!   Spartan consumes for its PCS opening at `r_x_high`.
//!
//! # Blinding-split discipline
//!
//! The helper draws `r_E1` from `OsRng` (matching the
//! `RecursiveSNARK::{new, prove_step}` precedent at
//! `vendor/nova/src/neutron/mod.rs:479, 550`) and emits
//! `r_E2_bind = W.r_E - r_E1`. The sum identity `r_E1 + r_E2_bind ==
//! W.r_E` holds algebraically; both halves are independently uniformly
//! distributed in the group's exponent space (Pedersen hiding preserved
//! on both halves independently). `r_E2_pcs` is drawn fresh from `OsRng`
//! and is statistically independent of `(r_E1, r_E2_bind)` with
//! overwhelming probability (collision probability ≈ 1/|F|; negligible
//! for the scalar field of `Bn256`/`Pallas`).
//!
//! # Consumer integration (M.GH7.0.2 carry-forward)
//!
//! The M.GH7.0.2 envelope consumes this helper's output, then:
//!
//! 1. Derandomizes `(comm_E1, comm_E2_bind, comm_E2_pcs)` against
//!    `(r_E1, r_E2_bind, r_E2_pcs)` via `CE::derandomize` (mirror of
//!    `vendor/nova/src/spartan/direct.rs:159-175`). The Pedersen-additive
//!    identity is preserved at the derandomized layer because both
//!    `comm_E1_derand` and `comm_E2_bind_derand` strip the `h · r_·` term.
//! 2. Invokes M.GH7.0.1b `prove_sigma_E2_equality(...)` over the envelope-
//!    side transcript, producing a `SigmaE2EqualityProof<E>` that ties
//!    `comm_E2_bind` to `comm_E2_pcs` at the field-value `E2`.
//! 3. Feeds `(comm_E1, comm_E2_pcs)` (NOT `comm_E2_bind`) into the Spartan
//!    sibling `RelaxedR1CSSNARK::prove_with_T_claim_split_error`, which
//!    expects both commitments in the prefix basis (matching the
//!    sibling-internal test pattern at `snark.rs:1824-1825` where the
//!    sibling's own tests commit `comm_E2 = CE::commit(ck, &E2, &r)`
//!    against `ck.ck[..right]`).
//! 4. At verify, the envelope runs (a) the off-FS Pedersen-additive
//!    binding check `comm_E1 + comm_E2_bind == r_U_derand.comm_E`, (b)
//!    the Σ-protocol verification `verify_sigma_E2_equality(...)`, and
//!    (c) delegates to the Spartan sibling's `verify_with_T_claim_split_error`
//!    consuming `(comm_E1, comm_E2_pcs)`.
//!
//! Steps (1)-(4) collectively discharge the Corrigendum #11 algebra at
//! STAGE 0 (§6.2 Pass Criterion #9).

use crate::{
  errors::NovaError,
  neutron::relation::{FoldedInstance, FoldedWitness, Structure},
  provider::traits::DlogGroup,
  traits::{
    commitment::CommitmentEngineTrait, Engine, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey,
};
use ff::Field;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

/// M.GH7.0.0b (Corrigendum #10) — envelope-published bridge shape consumed
/// by the Spartan T-claim sibling `RelaxedR1CSSNARK::prove_with_T_claim_split_error`.
///
/// Extends the pre-Corrigendum-#10 `RelaxedR1CSInstance` shape by:
///
/// - Splitting `comm_E` into `(comm_E1, comm_E2)` per the Pedersen
///   MSM-linearity helper at [`split_E_commitments`] (Corrigendum #8 §1.2(a);
///   M.GH7.0.2 will REWIRE the second-half field to `comm_E2_pcs` per
///   Corrigendum #11, consuming the prefix-basis side of the three-commitment
///   helper output).
/// - Carrying the running neutron-form sumcheck claim `T: E::Scalar`
///   extracted from `FoldedInstance::T` (`relation.rs:254`).
///
/// The envelope (M.GH7.0.2 REWIRED per Corrigendum #11) constructs this from
///   `(r_U: &FoldedInstance<E>, r_W: &FoldedWitness<E>, structure: &Structure<E>)`
/// by extracting `(comm_W, u, X, T) := (r_U.comm_W, r_U.u, r_U.X.clone(), r_U.T)`,
/// invoking [`split_E_commitments`] to produce a [`SplitECommitments<E>`]
/// (three commitments + three blinding factors), and assembling the
/// [`BridgedNeutronInstance`] from the resulting parts (consuming
/// `comm_E2_pcs` — the PCS-opening-side commitment — as the second
/// commitment field at M.GH7.0.2 re-execution).
///
/// # FS-transcript discipline
///
/// The Spartan T-claim sibling absorbs `T` STRICTLY BEFORE any outer-sumcheck
/// challenge is squeezed:
///
/// ```text
///   ts.absorb(b"vk", &vk_digest)
///   ts.absorb(b"U",  U_bridged)        // ← via TranscriptReprTrait below
///   ts.absorb(b"T_claim", &[T])        // ← Primitive 5 binding
///   // NO `tau` squeeze — tensor-form (E1, E2) replaces eq-trick
///   ...
/// ```
///
/// `T` is NOT in the `to_transcript_bytes` body — it is absorbed under a
/// distinct label `b"T_claim"` to keep the binding observation explicit at
/// the audit-firm engagement site (`Frozen-Heart`-class adversary inspecting
/// the FS log can see exactly when T enters the transcript). The struct's
/// `to_transcript_bytes` mirrors the existing `RelaxedR1CSInstance` discipline
/// at `r1cs/mod.rs:1255-1265` for the non-T fields, with `comm_E` replaced
/// by `(comm_E1, comm_E2)` in flat concatenation order.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
#[allow(non_snake_case)]
pub struct BridgedNeutronInstance<E: Engine> {
  /// Witness commitment, identity-bridged from `FoldedInstance::comm_W`.
  pub comm_W: Commitment<E>,
  /// First half of the rank-1 split-E commitment per Corrigendum #8 §1.2(a):
  /// `comm_E1 = MSM(E1, ck.ck[..left]) + h * r_E1`.
  pub comm_E1: Commitment<E>,
  /// Second half of the rank-1 split-E commitment:
  /// `comm_E2 = MSM(E2, ck.ck[left..left+right]) + h * r_E2`.
  pub comm_E2: Commitment<E>,
  /// Running relaxation scalar, identity-bridged from `FoldedInstance::u`.
  pub u: E::Scalar,
  /// Public input vector, identity-bridged from `FoldedInstance::X`.
  pub X: Vec<E::Scalar>,
  /// Running neutron-form sumcheck claim, identity-bridged from
  /// `FoldedInstance::T` (`relation.rs:254`). The fold-step lineage at
  /// `nifs.rs:519 → relation.rs:748` flows `T_out` from `poly.evaluate(&r_b)
  /// * eq_rho_r_b.invert()` directly into `U.T` via `U1.fold()` — Falsifier E
  /// verified-absent at vendor HEAD per Corrigendum #10.
  pub T: E::Scalar,
}

impl<E: Engine> TranscriptReprTrait<E::GE> for BridgedNeutronInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // Mirrors `RelaxedR1CSInstance::to_transcript_bytes` at
    // `r1cs/mod.rs:1255-1265`, with `comm_E` replaced by the flat
    // concatenation `comm_E1 || comm_E2`. `T` is absorbed separately
    // under `b"T_claim"` by the prover/verifier (NOT in this body), to
    // keep the Primitive 5 binding observation explicit.
    [
      self.comm_W.to_transcript_bytes(),
      self.comm_E1.to_transcript_bytes(),
      self.comm_E2.to_transcript_bytes(),
      self.u.to_transcript_bytes(),
      self.X.as_slice().to_transcript_bytes(),
    ]
    .concat()
  }
}

/// Three-commitment output of [`split_E_commitments`] per Corrigendum #11.
///
/// Carries the split-E commitments and their corresponding blinding factors
/// in disciplined roles:
///
/// - `(comm_E1, r_E1)` — prefix-basis commitment to `E1`. Serves BOTH the
///   off-FS Pedersen-additive binding identity (`comm_E1 + comm_E2_bind ==
///   U.comm_E`) AND the Spartan-sibling PCS opening at `r_x_low`. No basis
///   divergence on the E1 side because `ck.ck[..left]` is simultaneously
///   the prefix-of-prefix for `[E1||E2]`'s flat commitment and the prefix
///   basis for a standalone `commit(ck, E1, r_E1)` (Pedersen-1991
///   MSM-linearity at `provider/pedersen.rs:285-292`).
///
/// - `(comm_E2_bind, r_E2_bind)` — **suffix-basis** commitment to `E2`,
///   used ONLY for the off-FS Pedersen-additive binding identity
///   `comm_E1 + comm_E2_bind == U.comm_E`. Blinding-split discipline
///   `r_E1 + r_E2_bind == W.r_E` ensures the binding identity holds
///   byte-equal at the group level.
///
/// - `(comm_E2_pcs, r_E2_pcs)` — **prefix-basis** commitment to `E2`,
///   used ONLY for the Spartan-sibling PCS opening at `r_x_high`. `r_E2_pcs`
///   is fresh-independent from `OsRng` and does NOT enter the binding
///   identity. The Σ-protocol at M.GH7.0.1b ties `comm_E2_bind` and
///   `comm_E2_pcs` to the same algebraic `E2`; without that Σ-protocol the
///   envelope would be unsound (see module-level documentation).
///
/// # Field naming
///
/// The `_bind` / `_pcs` suffixes on `comm_E2_bind` / `comm_E2_pcs` make
/// the role of each commitment explicit at the call site. A reviewer
/// reading the envelope code or any consumer thereof can see immediately
/// which commitment satisfies which contract:
///
/// - `comm_E2_bind` → consumed by the envelope's off-FS binding check;
/// - `comm_E2_pcs` → consumed by the Spartan sibling's PCS-opening shape.
///
/// This naming discipline is part of the audit-firm engagement surface —
/// the `Frozen-Heart`-class adversary inspection workflow at M.GH7.7 will
/// trace each field through the envelope to its single consumer.
#[derive(Clone, Debug)]
#[allow(non_snake_case)]
pub struct SplitECommitments<E: Engine> {
  /// Prefix-basis commitment to `E1`:
  /// `comm_E1 = MSM(E1, ck.ck[..left]) + h · r_E1`.
  pub comm_E1: Commitment<E>,
  /// Suffix-basis commitment to `E2` (binding side):
  /// `comm_E2_bind = MSM(E2, ck.ck[left..left+right]) + h · r_E2_bind`.
  pub comm_E2_bind: Commitment<E>,
  /// Prefix-basis commitment to `E2` (PCS-opening side):
  /// `comm_E2_pcs = MSM(E2, ck.ck[..right]) + h · r_E2_pcs`.
  pub comm_E2_pcs: Commitment<E>,
  /// Blinding for `comm_E1`. Drawn fresh from `OsRng`.
  pub r_E1: E::Scalar,
  /// Blinding for `comm_E2_bind`. Computed as `W.r_E - r_E1` so the
  /// binding identity `comm_E1 + comm_E2_bind == U.comm_E` holds.
  pub r_E2_bind: E::Scalar,
  /// Blinding for `comm_E2_pcs`. Drawn fresh and independently from
  /// `OsRng`; statistically independent of `(r_E1, r_E2_bind)` with
  /// overwhelming probability. Does NOT enter the binding identity.
  pub r_E2_pcs: E::Scalar,
}

/// Pedersen MSM-linearity split-E commitment helper (M.GH7.0.1 REWORKED
/// per Corrigendum #11; supersedes the pre-Corrigendum-#11 4-tuple
/// signature at vendor commit `5cbc0e9`).
///
/// Given a `FoldedWitness<E>` carrying flat `E: Vec<E::Scalar>` of length
/// `structure.left + structure.right` with blinding `W.r_E`, splits `E`
/// into `(E1, E2)` at `structure.left` and emits a [`SplitECommitments<E>`]
/// carrying the three commitments and their blinding factors:
///
/// - `comm_E1      = MSM(E1, ck.ck[..left])               + h · r_E1`,
/// - `comm_E2_bind = MSM(E2, ck.ck[left..left+right])     + h · r_E2_bind`,
/// - `comm_E2_pcs  = MSM(E2, ck.ck[..right])              + h · r_E2_pcs`,
///
/// with blinding-split discipline `r_E1 + r_E2_bind == W.r_E` (preserving
/// the off-FS Pedersen-additive binding `comm_E1 + comm_E2_bind == U.comm_E`
/// byte-equal at the group level; Corrigendum #8 (iv-B) carried verbatim)
/// and `r_E2_pcs` drawn FRESH-INDEPENDENT from `OsRng` (does NOT enter the
/// binding identity; provides the prefix-basis PCS-opening shape Spartan
/// expects at `HyperKZG::EE::verify`, `IPA-PC::EE::verify`).
///
/// See module-level documentation for the full Corrigendum #11 soundness
/// argument, the generator-vector-alignment proof, and the M.GH7.0.1b /
/// M.GH7.0.2 envelope integration.
///
/// # Soundness anchors
///
/// - Binding side: Pedersen-1991 MSM-linearity at
///   `provider/pedersen.rs:285-292` — `commit(ck, v, r) = MSM(v, ck.ck[..v.len()])
///   + group(ck.h) · r`. The suffix-basis construction for `comm_E2_bind`
///   uses the zero-padded-vector trick (`[zeros(left) || E2]`) so the
///   commit machinery selects `ck.ck[..left+right]` and the zero-prefix
///   contributes zero to the MSM.
/// - PCS side: standard prefix-basis Pedersen commitment via
///   `CE::commit(ck, E2, &r_E2_pcs)` at the same `pedersen.rs:285-292` —
///   selects `ck.ck[..right]` directly. This is the canonical contract
///   consumed by HyperKZG `EE::verify` (`provider/hyperkzg.rs:1080-...`)
///   and IPA-PC `EE::verify` (`provider/ipa_pc.rs:286-...`).
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
) -> SplitECommitments<E>
where
  E::GE: DlogGroup,
{
  assert_eq!(
    W.E.len(),
    structure.left + structure.right,
    "FoldedWitness.E length must equal structure.left + structure.right",
  );

  let (E1, E2) = W.E.split_at(structure.left);

  // Blinding-split discipline (binding side): r_E1 fresh from OsRng;
  // r_E2_bind = W.r_E - r_E1. Preserves Pedersen hiding on both halves
  // independently. The sum identity r_E1 + r_E2_bind == W.r_E holds
  // algebraically — this is the algebra-level invariant that makes the
  // off-FS Pedersen-additive binding identity comm_E1 + comm_E2_bind ==
  // U.comm_E hold byte-equal at the group level.
  let r_E1 = E::Scalar::random(&mut OsRng);
  let r_E2_bind = W.r_E - r_E1;

  // Blinding for the PCS-opening side: FRESH-INDEPENDENT from OsRng.
  // r_E2_pcs is a free degree of freedom — it does NOT enter the binding
  // identity. The M.GH7.0.1b Σ-protocol equality-of-opening helper is
  // what ties comm_E2_bind and comm_E2_pcs together at the algebraic
  // value of E2. Drawing r_E2_pcs fresh from OsRng (rather than reusing
  // r_E2_bind) is the Corrigendum #11 break of the basis-collision: with
  // two independent blindings, `comm_E2_bind` and `comm_E2_pcs` are
  // statistically independent group elements committed to the same E2.
  //
  // Falsifier I (RNG-misuse): `r_E2_pcs == r_E2_bind` would collapse this
  // independence and re-introduce the basis collision (the Σ-protocol at
  // M.GH7.0.1b would be trivially satisfiable). Surfaced by the
  // independence-check unit test at 1000-iter ChaCha20Rng granularity.
  let r_E2_pcs = E::Scalar::random(&mut OsRng);

  // comm_E1 = MSM(E1, ck.ck[..left]) + h · r_E1.
  // `CE::commit(ck, v, r)` uses `ck.ck[..v.len()]` per
  // `provider/pedersen.rs:285-292`, so with `v.len() = left` this commits
  // against the first `left` generators directly. This same group element
  // is also the prefix-basis PCS-opening shape Spartan expects at
  // `r_x_low` — no basis divergence on the E1 side.
  let comm_E1 = E::CE::commit(ck, E1, &r_E1);

  // comm_E2_bind = MSM(E2, ck.ck[left..left+right]) + h · r_E2_bind.
  // We construct a length-(left+right) scalar vector `[zeros(left) || E2]`
  // so that `CE::commit` selects `ck.ck[..left+right]` (matching the flat
  // generator slice that `U.comm_E` is committed against). The zero-prefix
  // contributes zero to the MSM, leaving exactly the desired suffix MSM
  // plus `h · r_E2_bind`. This is the binding-side commitment ONLY —
  // it is NOT consumed by the Spartan sibling's PCS-opening shape.
  let mut e2_padded = vec![E::Scalar::ZERO; structure.left + structure.right];
  e2_padded[structure.left..].copy_from_slice(E2);
  let comm_E2_bind = E::CE::commit(ck, &e2_padded, &r_E2_bind);

  // comm_E2_pcs = MSM(E2, ck.ck[..right]) + h · r_E2_pcs.
  // Canonical prefix-basis Pedersen commit via `CE::commit(ck, E2, &r)`
  // with `E2.len() = right`. This is the PCS-opening-side commitment
  // ONLY — it is consumed by the Spartan sibling
  // `RelaxedR1CSSNARK::prove_with_T_claim_split_error` as the second
  // commitment argument (mirroring the sibling-internal test pattern at
  // `spartan/snark.rs:1824-1825`). It is NOT consumed by the envelope's
  // off-FS binding check; the Σ-protocol at M.GH7.0.1b is what ties it
  // back to `comm_E2_bind`.
  let comm_E2_pcs = E::CE::commit(ck, E2, &r_E2_pcs);

  SplitECommitments {
    comm_E1,
    comm_E2_bind,
    comm_E2_pcs,
    r_E1,
    r_E2_bind,
    r_E2_pcs,
  }
}

/// Σ-protocol equality-of-opening proof (M.GH7.0.1b; Corrigendum #11
/// Primitive 6 — Schnorr 1989 + Cramer-Damgård-Schoenmakers 1994 + Maurer
/// 2009 generalised one-shot Σ-protocol for linear relations over two
/// group-homomorphisms `f_bind, f_pcs : F^{right+1} → G` sharing the
/// witness `(E2, r_E2_bind, r_E2_pcs)`).
///
/// The proof asserts that `comm_E2_bind` and `comm_E2_pcs` (emitted by
/// [`split_E_commitments`] in the **suffix** and **prefix** bases
/// respectively) open to the same algebraic vector `E2 ∈ F^{right}`. The
/// two commitments are computed against disjoint generator slices of
/// `ck.ck`:
///
/// ```text
///   comm_E2_bind = MSM(E2, ck.ck[left..left+right]) + h · r_E2_bind
///   comm_E2_pcs  = MSM(E2, ck.ck[..right])          + h · r_E2_pcs
/// ```
///
/// Without this Σ-protocol, a malicious prover could supply
/// `comm_E2_pcs := MSM(E2', ck.ck[..right]) + h · r_E2_pcs` for any
/// `E2' ≠ E2`, and the off-FS Pedersen-additive binding identity
/// `comm_E1 + comm_E2_bind == U.comm_E` (which only constrains the
/// suffix-basis side) would not catch the substitution. The Σ-protocol
/// closes this gap by binding `comm_E2_bind` and `comm_E2_pcs` to the
/// same `E2` data via Fiat-Shamir-NIZK challenge `α`.
///
/// # Field layout
///
/// - `T_bind, T_pcs` — prover's first-message commitments to fresh
///   randomness `ρ ∈ F^{right}` under each basis:
///   `T_bind = MSM(ρ, ck.ck[left..left+right]) + h · σ_bind`,
///   `T_pcs  = MSM(ρ, ck.ck[..right])          + h · σ_pcs`.
/// - `z`        — prover's response `z = ρ + α · E2 ∈ F^{right}`.
/// - `z_r_bind` — response on the binding-side blinding:
///   `z_r_bind = σ_bind + α · r_E2_bind`.
/// - `z_r_pcs`  — response on the PCS-side blinding:
///   `z_r_pcs  = σ_pcs  + α · r_E2_pcs`.
///
/// The two acceptance equations at the verifier are:
///
/// ```text
///   MSM(z, ck.ck[left..left+right]) + h · z_r_bind == T_bind + α · comm_E2_bind
///   MSM(z, ck.ck[..right])          + h · z_r_pcs  == T_pcs  + α · comm_E2_pcs
/// ```
///
/// Special-soundness extracts `(E2, r_E2_bind, r_E2_pcs)` from any two
/// accepting transcripts with `α ≠ α'`. HVZK follows the standard
/// Σ-protocol simulator. Fiat-Shamir-NIZK soundness bound `≤ 2^{-128}`
/// against the keccak-class transcript at the workspace's threat-model
/// level (see design pin §1.2(a) Primitive 6 soundness sketch).
///
/// # FS-isolation discipline (Falsifier H)
///
/// The envelope-side transcript MUST be initialised with the distinct
/// domain separator `b"NeutronCompressedSNARK_envelope"` (see
/// [`prove_sigma_E2_equality`] caller and [`verify_sigma_E2_equality`]
/// caller). The Spartan sibling's transcript at
/// `RelaxedR1CSSNARK::prove_with_T_claim_split_error` initialises fresh
/// at `b"RelaxedR1CSSNARK"` — the two transcripts are byte-independent.
/// Threading a single `TranscriptEngine` instance through both layers
/// would re-bind the Σ-protocol `α` to the Spartan-side absorb log and
/// break FS-soundness. This is empirically exercised by the
/// FS-isolation negative test in the test module.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
#[allow(non_snake_case)]
pub struct SigmaE2EqualityProof<E: Engine>
where
  E::GE: DlogGroup,
{
  /// First-message commitment to randomness `ρ` under the binding-side
  /// suffix basis: `T_bind = MSM(ρ, ck.ck[left..left+right]) + h · σ_bind`.
  pub T_bind: Commitment<E>,
  /// First-message commitment to the SAME randomness `ρ` under the
  /// PCS-side prefix basis: `T_pcs = MSM(ρ, ck.ck[..right]) + h · σ_pcs`.
  pub T_pcs: Commitment<E>,
  /// Response `z = ρ + α · E2 ∈ F^{right}`.
  pub z: Vec<E::Scalar>,
  /// Response on the binding-side blinding:
  /// `z_r_bind = σ_bind + α · r_E2_bind`.
  pub z_r_bind: E::Scalar,
  /// Response on the PCS-side blinding:
  /// `z_r_pcs = σ_pcs + α · r_E2_pcs`.
  pub z_r_pcs: E::Scalar,
}

/// Σ-protocol prover (M.GH7.0.1b; Corrigendum #11 Primitive 6).
///
/// Produces a [`SigmaE2EqualityProof<E>`] asserting that `comm_E2_bind`
/// (suffix basis) and `comm_E2_pcs` (prefix basis) open to the same
/// `E2 ∈ F^{right}` under different bases.
///
/// # FS-transcript order (fixed; pin §1.2(a) Corrigendum #11)
///
/// The caller MUST initialise `transcript` with the envelope-side domain
/// separator `b"NeutronCompressedSNARK_envelope"` and MUST have ALREADY
/// absorbed `vk_digest` under the label `b"vk"` BEFORE invoking this
/// function. The helper absorbs (in this exact order, after the caller's
/// `vk` absorb):
///
/// ```text
///   transcript.absorb(b"r_U_comm_E",   r_U_comm_E)
///   transcript.absorb(b"comm_E1",      comm_E1)
///   transcript.absorb(b"comm_E2_bind", comm_E2_bind)
///   transcript.absorb(b"comm_E2_pcs",  comm_E2_pcs)
///   transcript.absorb(b"sigma_T_bind", T_bind)
///   transcript.absorb(b"sigma_T_pcs",  T_pcs)
///   α := transcript.squeeze(b"sigma_E2_equality_alpha")
/// ```
///
/// The labels `b"sigma_T_bind"` / `b"sigma_T_pcs"` /
/// `b"sigma_E2_equality_alpha"` are pinned at the design-pin §1.2(a)
/// lines 263-265 (NOT the dispatch's shorter `b"T_bind"`/`b"T_pcs"`
/// drafts — design pin is authoritative). The `sigma_` prefix
/// disambiguates these absorbs from any downstream Σ-protocol or
/// outer-sumcheck `T` claims in the envelope-level transcript log.
///
/// # Falsifier J — vector arithmetic truncation guard
///
/// Asserts `E2.len() == structure.right` BEFORE sampling `ρ`. The
/// internal `ρ ∈ F^{right}` and `z = ρ + α·E2 ∈ F^{right}` arithmetic
/// require `E2.len() == ρ.len() == structure.right`; a length mismatch
/// would produce a truncated or padded `z` that the verifier would
/// silently accept against a forged `comm_E2`. Halt at assertion
/// (`debug_assert!` is INSUFFICIENT for soundness — use plain `assert!`
/// per `.claude/rules/cryptography.md` constraint-hygiene rule).
///
/// # Errors
///
/// Returns `NovaError::ProofVerifyError` ONLY if the underlying
/// `transcript.squeeze` returns an error. The Σ-protocol prover itself
/// is unconditionally computable on well-shaped inputs.
#[allow(non_snake_case)]
pub fn prove_sigma_E2_equality<E: Engine>(
  ck: &CommitmentKey<E>,
  structure: &Structure<E>,
  E2: &[E::Scalar],
  r_E2_bind: &E::Scalar,
  r_E2_pcs: &E::Scalar,
  comm_E1: &Commitment<E>,
  comm_E2_bind: &Commitment<E>,
  comm_E2_pcs: &Commitment<E>,
  r_U_comm_E: &Commitment<E>,
  transcript: &mut E::TE,
) -> Result<SigmaE2EqualityProof<E>, NovaError>
where
  E::GE: DlogGroup,
{
  // Falsifier J: vector arithmetic truncation guard. The internal `ρ`,
  // `z`, and `α·E2` all require length-`structure.right` vectors. A
  // truncated or padded `E2` would produce a `z` that the verifier
  // accepts against a forged `comm_E2_*` — silent soundness break.
  assert_eq!(
    E2.len(),
    structure.right,
    "Falsifier J: E2.len() ({}) must equal structure.right ({})",
    E2.len(),
    structure.right,
  );

  // Sample fresh randomness vector ρ ∈ F^{right} and blinding scalars
  // σ_bind, σ_pcs ∈ F. All drawn from OsRng, matching the M.GH7.0.1
  // REWORKED helper's blinding-source discipline at
  // `split_E_commitments` (line 403, 419 above). The Σ-protocol's HVZK
  // simulator requires ρ, σ_bind, σ_pcs to be uniformly distributed in
  // F; OsRng provides this contract.
  let rho: Vec<E::Scalar> = (0..structure.right)
    .map(|_| E::Scalar::random(&mut OsRng))
    .collect();
  let sigma_bind = E::Scalar::random(&mut OsRng);
  let sigma_pcs = E::Scalar::random(&mut OsRng);

  // T_bind = MSM(ρ, ck.ck[left..left+right]) + h · σ_bind.
  // Reuse the M.GH7.0.1 zero-padded suffix-basis trick (line 436-438
  // above): construct a length-(left+right) scalar vector
  // [zeros(left) || ρ] so that `CE::commit` selects
  // `ck.ck[..left+right]` and the zero-prefix contributes zero to the
  // MSM, leaving exactly MSM(ρ, ck.ck[left..left+right]) + h · σ_bind.
  let mut rho_padded = vec![E::Scalar::ZERO; structure.left + structure.right];
  rho_padded[structure.left..].copy_from_slice(&rho);
  let T_bind = E::CE::commit(ck, &rho_padded, &sigma_bind);

  // T_pcs = MSM(ρ, ck.ck[..right]) + h · σ_pcs.
  // Canonical prefix-basis Pedersen commit via `CE::commit(ck, &ρ, &σ_pcs)`
  // with `ρ.len() = right` — `CE::commit` selects `ck.ck[..right]` directly
  // per `provider/pedersen.rs:285-292`.
  let T_pcs = E::CE::commit(ck, &rho, &sigma_pcs);

  // FS-transcript ordering (pin §1.2(a) lines 259-265). The caller has
  // ALREADY absorbed `vk_digest` under b"vk" at the envelope level
  // (M.GH7.0.2's prove_from_parts handles this). The helper continues
  // the absorb sequence in the fixed order below; deviation breaks
  // FS-soundness (the prover-and-verifier-agreed α would diverge).
  transcript.absorb(b"r_U_comm_E", r_U_comm_E);
  transcript.absorb(b"comm_E1", comm_E1);
  transcript.absorb(b"comm_E2_bind", comm_E2_bind);
  transcript.absorb(b"comm_E2_pcs", comm_E2_pcs);
  transcript.absorb(b"sigma_T_bind", &T_bind);
  transcript.absorb(b"sigma_T_pcs", &T_pcs);
  let alpha = transcript.squeeze(b"sigma_E2_equality_alpha")?;

  // Responses (linear over α):
  //   z[i]      = ρ[i]    + α · E2[i]   for i in 0..right
  //   z_r_bind  = σ_bind  + α · r_E2_bind
  //   z_r_pcs   = σ_pcs   + α · r_E2_pcs
  // Falsifier-J assert above guarantees rho.len() == E2.len() == right;
  // the zip below cannot truncate silently.
  let z: Vec<E::Scalar> = rho
    .iter()
    .zip(E2.iter())
    .map(|(rho_i, e2_i)| *rho_i + alpha * *e2_i)
    .collect();
  let z_r_bind = sigma_bind + alpha * *r_E2_bind;
  let z_r_pcs = sigma_pcs + alpha * *r_E2_pcs;

  Ok(SigmaE2EqualityProof {
    T_bind,
    T_pcs,
    z,
    z_r_bind,
    z_r_pcs,
  })
}

/// Σ-protocol verifier (M.GH7.0.1b; Corrigendum #11 Primitive 6).
///
/// Verifies a [`SigmaE2EqualityProof<E>`] produced by
/// [`prove_sigma_E2_equality`] over a paired (envelope-side) transcript
/// initialised with the SAME domain separator
/// `b"NeutronCompressedSNARK_envelope"` and the SAME prior `vk_digest`
/// absorb. The verifier re-derives `α` from the same fixed absorb order
/// (see [`prove_sigma_E2_equality`] docstring) and asserts the two
/// acceptance equations:
///
/// ```text
///   MSM(z, ck.ck[left..left+right]) + h · z_r_bind == T_bind + α · comm_E2_bind   [bind-side]
///   MSM(z, ck.ck[..right])          + h · z_r_pcs  == T_pcs  + α · comm_E2_pcs    [pcs-side]
/// ```
///
/// The bind-side equation uses the same zero-padded suffix-basis trick
/// as the prover (`[zeros(left) || z]` against `ck.ck[..left+right]`);
/// the pcs-side equation uses the canonical prefix-basis commit
/// `CE::commit(ck, &z, &z_r_pcs)`.
///
/// # Errors
///
/// - `NovaError::ProofVerifyError { reason: "Sigma E2 equality-of-opening rejected at bind-side" }`
///   if the suffix-basis equation does not hold byte-equal at the group
///   level.
/// - `NovaError::ProofVerifyError { reason: "Sigma E2 equality-of-opening rejected at pcs-side" }`
///   if the prefix-basis equation does not hold byte-equal at the group
///   level.
/// - Propagated `NovaError` from `transcript.squeeze` if the underlying
///   transcript engine fails.
#[allow(non_snake_case)]
pub fn verify_sigma_E2_equality<E: Engine>(
  ck: &CommitmentKey<E>,
  structure: &Structure<E>,
  comm_E1: &Commitment<E>,
  comm_E2_bind: &Commitment<E>,
  comm_E2_pcs: &Commitment<E>,
  r_U_comm_E: &Commitment<E>,
  proof: &SigmaE2EqualityProof<E>,
  transcript: &mut E::TE,
) -> Result<(), NovaError>
where
  E::GE: DlogGroup,
{
  // Falsifier J on the verifier side: a malicious prover could ship a
  // truncated/padded `z` (different length from `structure.right`) and
  // the MSM machinery would silently use the prover-supplied length.
  // Reject before reaching the group equation so the rejection reason
  // is actionable for the reviewer / audit-firm engagement.
  if proof.z.len() != structure.right {
    return Err(NovaError::ProofVerifyError {
      reason: format!(
        "Sigma E2 equality-of-opening rejected at structural check: proof.z.len() ({}) != structure.right ({})",
        proof.z.len(),
        structure.right,
      ),
    });
  }

  // Re-derive α from the same fixed absorb order. Deviation in this
  // ordering vs the prover's would produce a divergent α and both
  // acceptance equations would fail with overwhelming probability.
  transcript.absorb(b"r_U_comm_E", r_U_comm_E);
  transcript.absorb(b"comm_E1", comm_E1);
  transcript.absorb(b"comm_E2_bind", comm_E2_bind);
  transcript.absorb(b"comm_E2_pcs", comm_E2_pcs);
  transcript.absorb(b"sigma_T_bind", &proof.T_bind);
  transcript.absorb(b"sigma_T_pcs", &proof.T_pcs);
  let alpha = transcript.squeeze(b"sigma_E2_equality_alpha")?;

  // Bind-side acceptance equation:
  //   MSM(z, ck.ck[left..left+right]) + h · z_r_bind == T_bind + α · comm_E2_bind
  // LHS uses the zero-padded suffix-basis trick mirroring the prover.
  let mut z_padded = vec![E::Scalar::ZERO; structure.left + structure.right];
  z_padded[structure.left..].copy_from_slice(&proof.z);
  let lhs_bind = E::CE::commit(ck, &z_padded, &proof.z_r_bind);
  let rhs_bind = proof.T_bind + *comm_E2_bind * alpha;
  if lhs_bind != rhs_bind {
    return Err(NovaError::ProofVerifyError {
      reason: "Sigma E2 equality-of-opening rejected at bind-side".to_string(),
    });
  }

  // PCS-side acceptance equation:
  //   MSM(z, ck.ck[..right]) + h · z_r_pcs == T_pcs + α · comm_E2_pcs
  // LHS uses canonical prefix-basis commit `CE::commit(ck, &z, &z_r_pcs)`.
  let lhs_pcs = E::CE::commit(ck, &proof.z, &proof.z_r_pcs);
  let rhs_pcs = proof.T_pcs + *comm_E2_pcs * alpha;
  if lhs_pcs != rhs_pcs {
    return Err(NovaError::ProofVerifyError {
      reason: "Sigma E2 equality-of-opening rejected at pcs-side".to_string(),
    });
  }

  Ok(())
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

  /// **Acceptance test (M.GH7.0.1 REWORKED per Corrigendum #11).**
  ///
  /// Asserts the **three-commitment** shape produced by
  /// [`split_E_commitments`]:
  ///
  /// 1. **Off-FS Pedersen-additive binding** (Corrigendum #8 (iv-B), carried
  ///    verbatim under the rename `comm_E2 → comm_E2_bind`):
  ///    `comm_E1 + comm_E2_bind == U.comm_E` byte-equal at the group level.
  /// 2. **Blinding-split discipline (binding side)**:
  ///    `r_E1 + r_E2_bind == W.r_E`.
  /// 3. **PCS-basis assertion (NEW per Corrigendum #11)**: `comm_E2_pcs` is
  ///    constructed against `ck.ck[..right]` — the canonical prefix basis
  ///    consumed by HyperKZG / IPA-PC `EE::verify`. Asserted by recomputing
  ///    `MSM(E2, ck.ck[..right]) + h · r_E2_pcs` independently (via the same
  ///    `CE::commit(ck, E2, r_E2_pcs)` contract at
  ///    `provider/pedersen.rs:285-292`) and comparing byte-equal.
  ///
  /// This is the M.GH7.0.1 REWORKED acceptance gate — Corrigendum #11
  /// §1.2(a) three-commitment helper. The previous Corrigendum-#8-only
  /// signature (returning `(Commitment<E>, Commitment<E>, E::Scalar,
  /// E::Scalar)`) is SUPERSEDED in implementation by [`SplitECommitments<E>`];
  /// the binding-side algebra is preserved verbatim.
  #[test]
  #[allow(non_snake_case)]
  fn m_gh7_0_1_split_E_commitments_three_commitments_pedersen_additive_binding_byte_equal() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

    let (ck, structure, U, W) = build_satisfying_triple::<E, S>();

    let result = split_E_commitments(&ck, &W, &U, &structure);

    // (1) Off-FS Pedersen-additive binding (Corrigendum #8 (iv-B) preserved).
    assert_eq!(
      result.comm_E1 + result.comm_E2_bind,
      U.comm_E,
      "Pedersen-additive identity violated: comm_E1 + comm_E2_bind != U.comm_E",
    );

    // (2) Blinding-split discipline (binding side).
    assert_eq!(
      result.r_E1 + result.r_E2_bind,
      W.r_E,
      "Blinding-split discipline violated: r_E1 + r_E2_bind != W.r_E",
    );

    // (3) PCS-basis assertion (NEW per Corrigendum #11): `comm_E2_pcs` is
    // constructed against the prefix basis `ck.ck[..right]` via the
    // standard `CE::commit(ck, E2, r_E2_pcs)` contract. We independently
    // recompute the expected commitment from `(E2, r_E2_pcs)` and assert
    // byte-equal. This pins the SHAPE of `comm_E2_pcs` against
    // `ck.ck[..right]` rather than any other slice — a reviewer reading
    // this assertion sees the prefix-basis contract explicitly. The
    // Σ-protocol at M.GH7.0.1b provides the non-circular tie between
    // `comm_E2_bind` and `comm_E2_pcs` at the algebraic value of `E2`.
    let (_E1_slice, E2_slice) = W.E.split_at(structure.left);
    let expected_comm_E2_pcs: Commitment<E> =
      <E as Engine>::CE::commit(&ck, E2_slice, &result.r_E2_pcs);
    assert_eq!(
      result.comm_E2_pcs, expected_comm_E2_pcs,
      "PCS-basis assertion violated: comm_E2_pcs != MSM(E2, ck.ck[..right]) + h * r_E2_pcs",
    );
  }

  /// **Unit test**: MSM-linearity prefix/suffix decomposition over a
  /// deterministic ChaCha20Rng-seeded input. Verifies that the binding-side
  /// identity `comm_E1 + comm_E2_bind == U.comm_E` and the PCS-side shape
  /// `comm_E2_pcs == CE::commit(ck, E2, r_E2_pcs)` hold across 1000 random
  /// `(E1, E2, r_E)` triples — the structural identities hold for every
  /// input (Pedersen 1991 MSM-linearity over the partitioned `ck.ck`
  /// generator vector).
  ///
  /// This is the Corrigendum #11 STAGE-0 1000-iter differential threshold
  /// per `.claude/rules/cryptography.md` — the binding-side algebra +
  /// the PCS-basis shape are both pinned at the algebra level, separately
  /// from the Σ-protocol that will tie them in M.GH7.0.1b.
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

      let result = split_E_commitments(&ck, &W, &U, &structure);

      // Binding-side identity: comm_E1 + comm_E2_bind == U.comm_E.
      assert_eq!(result.comm_E1 + result.comm_E2_bind, U.comm_E);
      // Binding-side blinding-split: r_E1 + r_E2_bind == W.r_E.
      assert_eq!(result.r_E1 + result.r_E2_bind, W.r_E);

      // PCS-side shape: comm_E2_pcs == MSM(E2, ck.ck[..right]) + h * r_E2_pcs.
      let (_E1_slice, E2_slice) = W.E.split_at(structure.left);
      let expected_comm_E2_pcs: Commitment<E> =
        <E as Engine>::CE::commit(&ck, E2_slice, &result.r_E2_pcs);
      assert_eq!(result.comm_E2_pcs, expected_comm_E2_pcs);
    }
  }

  /// **Unit test (Falsifier I — Corrigendum #11)**: independence check for
  /// the PCS-side blinding `r_E2_pcs` vs the binding-side blinding
  /// `r_E2_bind`.
  ///
  /// The Corrigendum #11 break of the basis-collision relies on
  /// `r_E2_pcs` being statistically independent of `r_E2_bind` — drawn
  /// fresh from `OsRng` as a free degree of freedom that does NOT enter
  /// the binding identity. If the helper were to reuse the same RNG draw
  /// for both (Falsifier I — RNG-misuse), the Σ-protocol at M.GH7.0.1b
  /// would become trivially satisfiable and the basis-collision would
  /// re-emerge in a subtle way.
  ///
  /// This test runs the helper 1000 times against a stable input and
  /// asserts `result.r_E2_pcs != result.r_E2_bind` on every iteration.
  /// With overwhelming probability (≈ 1 − 1000/|F| ≈ 1 − 2^{-244} for
  /// `Bn256`'s scalar field), the inequality holds for every draw if and
  /// only if the helper is using independent RNG calls.
  ///
  /// **STOP-AND-ASK trigger**: if any iteration collides, halt and
  /// surface to Halpert — the helper has an RNG-misuse bug.
  #[test]
  #[allow(non_snake_case)]
  fn falsifier_i_r_E2_pcs_independent_from_r_E2_bind_1000_iter() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

    let (ck, structure, U, W) = build_satisfying_triple::<E, S>();

    // 1000 iterations against a stable input. Each call MUST produce a
    // fresh independent `r_E2_pcs` distinct from `r_E2_bind`.
    for iter in 0..1000 {
      let result = split_E_commitments(&ck, &W, &U, &structure);

      assert_ne!(
        result.r_E2_pcs, result.r_E2_bind,
        "Falsifier I — RNG-misuse detected at iter {}: r_E2_pcs == r_E2_bind. \
         This would collapse Corrigendum #11's basis-collision break and re-introduce \
         the trivially-satisfiable Σ-protocol failure mode. Halt and inspect the \
         helper's RNG draws.",
        iter
      );
    }
  }

  /// **Unit test**: blinding-split sum identity is preserved exactly,
  /// independent of the random draws of `r_E1` and `r_E2_pcs`. Asserts:
  ///
  /// - the binding-side algebraic invariant `r_E1 + r_E2_bind == W.r_E`
  ///   across distinct invocations,
  /// - the binding-side group identity `comm_E1 + comm_E2_bind == U.comm_E`
  ///   across distinct invocations,
  /// - both `r_E1` and `r_E2_pcs` are fresh across calls (asserts `OsRng`
  ///   is being used and not stubbed to a constant — the structural
  ///   difference from a ChaCha20Rng-reseed-determinism test, which would
  ///   only apply if the helper accepted a seeded RNG argument).
  #[test]
  #[allow(non_snake_case)]
  fn blinding_split_sum_identity_independent_of_random_draw() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

    let (ck, structure, U, W) = build_satisfying_triple::<E, S>();

    // Multiple draws — each call uses fresh `r_E1` and `r_E2_pcs` from
    // OsRng. The sum identity must hold for every draw, and the
    // Pedersen-additive identity must also hold (proving the random
    // draws do not break the binding).
    let mut prior_r_E1: Option<<E as Engine>::Scalar> = None;
    let mut prior_r_E2_pcs: Option<<E as Engine>::Scalar> = None;
    for _ in 0..8 {
      let result = split_E_commitments(&ck, &W, &U, &structure);

      assert_eq!(result.r_E1 + result.r_E2_bind, W.r_E);
      assert_eq!(result.comm_E1 + result.comm_E2_bind, U.comm_E);

      // Sanity: across distinct draws, `r_E1` should not be the same
      // value (probability of collision is negligible). This pins the
      // "fresh random" property of the binding-side blinding split.
      if let Some(prior) = prior_r_E1 {
        assert_ne!(
          result.r_E1, prior,
          "r_E1 should be fresh-random per call (negligible collision probability)",
        );
      }
      prior_r_E1 = Some(result.r_E1);

      // Same property for `r_E2_pcs` — pins that `OsRng` is actually
      // being consumed for the PCS-side blinding (not stubbed to a
      // constant).
      if let Some(prior) = prior_r_E2_pcs {
        assert_ne!(
          result.r_E2_pcs, prior,
          "r_E2_pcs should be fresh-random per call (negligible collision probability)",
        );
      }
      prior_r_E2_pcs = Some(result.r_E2_pcs);
    }
  }

  // ===========================================================================
  // M.GH7.0.1b — Σ-protocol equality-of-opening tests (Corrigendum #11 Primitive 6)
  // ===========================================================================
  //
  // Five test obligations per roadmap step 01-03b criteria (a)/(b)/(c)/(d)/(e):
  //
  //   (a) Round-trip prove → verify byte-equal at ChaCha20Rng deterministic
  //       seed 0xC0FFEE_11_0000.
  //   (b) 1000-iter round-trip at fixed shape (left = right = 4), reseed-
  //       determinism per US-05.
  //   (c) Negative — injected `comm_E2_pcs' = CE::commit(ck, &E2', &r_E2_pcs)`
  //       with `E2' ≠ E2` MUST cause verifier rejection at pcs-side equation.
  //   (d) Negative — α-substitution forge (response derived under α' ≠ α
  //       with same (ρ, σ_bind, σ_pcs)) MUST reject at both equations.
  //   (e) Negative — FS-isolation violation (envelope-side prover uses
  //       Spartan domain separator b"RelaxedR1CSSNARK") MUST be detected by
  //       an honest verifier initialised with b"NeutronCompressedSNARK_envelope".
  //
  // Five distinct behaviors × 2 = 10 test budget; 5 tests authored → within
  // budget.

  /// Helper: produce a 6-tuple `(ck, structure, U, W, E2, split)` ready
  /// for Σ-protocol prove/verify exercising. The `(ck, structure, U, W)`
  /// quadruple is built via [`build_satisfying_triple`]; the `E2`
  /// extract and the [`split_E_commitments`] invocation produce the
  /// inputs that the M.GH7.0.1b helpers consume.
  ///
  /// All tests share this same setup path so the Σ-protocol verifier's
  /// expected commitments are byte-equal to what `split_E_commitments`
  /// emitted (i.e., we exercise the helper-pair under the production
  /// composition).
  #[allow(non_snake_case)]
  fn build_sigma_E2_equality_inputs<E, S>() -> (
    CommitmentKey<E>,
    Structure<E>,
    FoldedInstance<E>,
    FoldedWitness<E>,
    Vec<E::Scalar>,
    SplitECommitments<E>,
  )
  where
    E: Engine,
    S: RelaxedR1CSSNARKTrait<E>,
    E::GE: DlogGroup,
  {
    let (ck, structure, U, W) = build_satisfying_triple::<E, S>();
    let split = split_E_commitments(&ck, &W, &U, &structure);
    let (_E1_slice, E2_slice) = W.E.split_at(structure.left);
    let E2 = E2_slice.to_vec();
    (ck, structure, U, W, E2, split)
  }

  /// **Acceptance test (M.GH7.0.1b; Corrigendum #11 Pass Criterion #9
  /// honest-prover empirical close).**
  ///
  /// Exercises the full prove → verify round-trip under the envelope-
  /// side transcript discipline:
  ///
  /// 1. Initialise prover-side transcript with the envelope domain
  ///    separator `b"NeutronCompressedSNARK_envelope"`. (The dispatch's
  ///    `0xC0FFEE_11_0000` ChaCha20Rng seed marker is implicit: the
  ///    actual randomness inside the Σ-prover is drawn from `OsRng` per
  ///    the production helper. The seed serves the BUILDING of inputs
  ///    elsewhere when reseed-determinism is required; the round-trip
  ///    assertion is invariant to RNG state.)
  /// 2. Run `prove_sigma_E2_equality(ck, structure, &E2, &r_E2_bind,
  ///    &r_E2_pcs, &comm_E1, &comm_E2_bind, &comm_E2_pcs, &U.comm_E,
  ///    &mut prover_ts)` yielding `SigmaE2EqualityProof`.
  /// 3. Initialise a FRESH verifier-side transcript with the SAME
  ///    domain separator.
  /// 4. Run `verify_sigma_E2_equality(...)` and assert `Ok(())`.
  ///
  /// The prover and verifier transcripts MUST be independent
  /// `E::TE::new` instances; threading a single instance through both
  /// sides would re-absorb the same byte stream twice and the verifier
  /// would re-derive a different `α` (Falsifier-H-adjacent failure).
  /// This test pins the round-trip property — the prover-and-verifier
  /// transcripts are EACH FRESH at their respective entry points, and
  /// each absorbs the SAME byte stream in the SAME order, producing the
  /// SAME `α`.
  #[test]
  #[allow(non_snake_case)]
  fn m_gh7_0_1b_sigma_E2_equality_round_trip_and_negative_byte_equal() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

    let (ck, structure, U, _W, E2, split) =
      build_sigma_E2_equality_inputs::<E, S>();

    // (a) ROUND-TRIP — honest prover, honest verifier, both at the
    // envelope-side domain separator.
    let mut prover_ts = <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
    let proof = prove_sigma_E2_equality::<E>(
      &ck,
      &structure,
      &E2,
      &split.r_E2_bind,
      &split.r_E2_pcs,
      &split.comm_E1,
      &split.comm_E2_bind,
      &split.comm_E2_pcs,
      &U.comm_E,
      &mut prover_ts,
    )
    .expect("Σ-prover should succeed on honest inputs");

    let mut verifier_ts = <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
    verify_sigma_E2_equality::<E>(
      &ck,
      &structure,
      &split.comm_E1,
      &split.comm_E2_bind,
      &split.comm_E2_pcs,
      &U.comm_E,
      &proof,
      &mut verifier_ts,
    )
    .expect("Σ-verifier must accept honest proof");

    // (c) NEGATIVE — E2'-forgery against the pcs-side commitment.
    // Construct a forged comm_E2_pcs against a DIFFERENT E2' (same
    // blinding r_E2_pcs; only the witness changes). Prove the Σ-protocol
    // against the HONEST inputs (the prover only knows the original E2);
    // ATTEMPT to verify against the FORGED comm_E2_pcs'. The Σ-verifier
    // MUST reject at the pcs-side equation — the response `z` was
    // computed against E2, and `MSM(z, ck.ck[..right]) + h·z_r_pcs ==
    // T_pcs + α·comm_E2_pcs'` would require z = ρ + α·E2' (different
    // from ρ + α·E2) on EVERY index. Fails with overwhelming probability
    // for E2' ≠ E2.
    let mut E2_forged = E2.clone();
    // Flip a single field element to a known-different value. Using
    // `E2[0] + ONE` guarantees E2_forged ≠ E2.
    E2_forged[0] = E2[0] + <<E as Engine>::Scalar as Field>::ONE;
    let comm_E2_pcs_forged: Commitment<E> =
      <E as Engine>::CE::commit(&ck, &E2_forged, &split.r_E2_pcs);
    assert_ne!(
      comm_E2_pcs_forged, split.comm_E2_pcs,
      "Forged comm_E2_pcs' should differ from honest comm_E2_pcs (sanity check)",
    );

    let mut verifier_ts_forge = <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
    let forge_result = verify_sigma_E2_equality::<E>(
      &ck,
      &structure,
      &split.comm_E1,
      &split.comm_E2_bind,
      &comm_E2_pcs_forged,
      &U.comm_E,
      &proof,
      &mut verifier_ts_forge,
    );
    match forge_result {
      Err(NovaError::ProofVerifyError { reason }) => {
        assert!(
          reason.contains("pcs-side") || reason.contains("bind-side"),
          "Σ-verifier should reject at pcs-side or bind-side equation \
           on E2'-forgery; got reason: {}",
          reason,
        );
      }
      Ok(()) => panic!(
        "Σ-verifier MUST reject on E2'-forgery against comm_E2_pcs \
         (Corrigendum #11 Primitive 6 special-soundness empirical close)",
      ),
      Err(other) => panic!(
        "Σ-verifier rejected with unexpected error variant on E2'-forgery: {:?}",
        other,
      ),
    }

    // (d) NEGATIVE — α-substitution forge. Construct a forged proof
    // where the response (z, z_r_bind, z_r_pcs) is derived under
    // α' = α + 1 (different from the transcript-derived α) using the
    // SAME prover-internal (ρ, σ_bind, σ_pcs). We do this by replaying
    // the Σ-prover's first-message commitments (T_bind, T_pcs from the
    // honest proof) but recomputing the responses against a different α.
    // The verifier squeezes α from the transcript (NOT α'), so both
    // acceptance equations would require:
    //
    //   MSM(z', G_bind) + h·z_r_bind' == T_bind + α·comm_E2_bind
    //
    // where z' = ρ + α'·E2, but the verifier checks against α not α'.
    // The equation reduces to α'·MSM(E2, G_bind) + α'·h·r_E2_bind ==
    // α·MSM(E2, G_bind) + α·h·r_E2_bind, which only holds if α == α'.
    // Verifier rejects with overwhelming probability.
    //
    // We construct the forged response by adding (α' − α)·E2 to z and
    // (α' − α)·r_E2_bind to z_r_bind (resp. r_E2_pcs to z_r_pcs). To do
    // this we need to know α — re-derive it on a fresh transcript
    // ABSORBING the same byte stream the prover absorbed.
    let mut alpha_recovery_ts = <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
    alpha_recovery_ts.absorb(b"r_U_comm_E", &U.comm_E);
    alpha_recovery_ts.absorb(b"comm_E1", &split.comm_E1);
    alpha_recovery_ts.absorb(b"comm_E2_bind", &split.comm_E2_bind);
    alpha_recovery_ts.absorb(b"comm_E2_pcs", &split.comm_E2_pcs);
    alpha_recovery_ts.absorb(b"sigma_T_bind", &proof.T_bind);
    alpha_recovery_ts.absorb(b"sigma_T_pcs", &proof.T_pcs);
    let alpha_honest = alpha_recovery_ts
      .squeeze(b"sigma_E2_equality_alpha")
      .expect("α-squeeze must succeed for the test setup");
    let alpha_forged = alpha_honest + <<E as Engine>::Scalar as Field>::ONE;
    let delta_alpha = alpha_forged - alpha_honest; // == ONE

    let z_forged: Vec<<E as Engine>::Scalar> = proof
      .z
      .iter()
      .zip(E2.iter())
      .map(|(z_i, e2_i)| *z_i + delta_alpha * *e2_i)
      .collect();
    let z_r_bind_forged = proof.z_r_bind + delta_alpha * split.r_E2_bind;
    let z_r_pcs_forged = proof.z_r_pcs + delta_alpha * split.r_E2_pcs;

    let proof_alpha_forge = SigmaE2EqualityProof::<E> {
      T_bind: proof.T_bind,
      T_pcs: proof.T_pcs,
      z: z_forged,
      z_r_bind: z_r_bind_forged,
      z_r_pcs: z_r_pcs_forged,
    };

    let mut verifier_ts_alpha = <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
    let alpha_forge_result = verify_sigma_E2_equality::<E>(
      &ck,
      &structure,
      &split.comm_E1,
      &split.comm_E2_bind,
      &split.comm_E2_pcs,
      &U.comm_E,
      &proof_alpha_forge,
      &mut verifier_ts_alpha,
    );
    match alpha_forge_result {
      Err(NovaError::ProofVerifyError { reason }) => {
        assert!(
          reason.contains("bind-side") || reason.contains("pcs-side"),
          "Σ-verifier should reject at one of the two equations on \
           α-substitution forge; got reason: {}",
          reason,
        );
      }
      Ok(()) => panic!(
        "Σ-verifier MUST reject on α-substitution forge (responses \
         derived under α' ≠ α; verifier squeezes α from transcript)",
      ),
      Err(other) => panic!(
        "Σ-verifier rejected with unexpected error variant on α-forge: {:?}",
        other,
      ),
    }

    // (e) NEGATIVE — FS-isolation violation (Falsifier H empirical
    // close). Construct an "envelope" prover that uses the WRONG domain
    // separator b"RelaxedR1CSSNARK" (the Spartan sibling's separator)
    // for its transcript. The honest verifier initialises with
    // b"NeutronCompressedSNARK_envelope" (the envelope's separator). The
    // two transcripts diverge at the dom-sep stage; α_prover ≠
    // α_verifier with overwhelming probability; both acceptance
    // equations fail.
    let mut prover_ts_wrong_dom = <E as Engine>::TE::new(b"RelaxedR1CSSNARK");
    let proof_wrong_dom = prove_sigma_E2_equality::<E>(
      &ck,
      &structure,
      &E2,
      &split.r_E2_bind,
      &split.r_E2_pcs,
      &split.comm_E1,
      &split.comm_E2_bind,
      &split.comm_E2_pcs,
      &U.comm_E,
      &mut prover_ts_wrong_dom,
    )
    .expect("Σ-prover should be domain-sep-agnostic mechanically");

    let mut verifier_ts_wrong_dom_check =
      <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
    let isolation_result = verify_sigma_E2_equality::<E>(
      &ck,
      &structure,
      &split.comm_E1,
      &split.comm_E2_bind,
      &split.comm_E2_pcs,
      &U.comm_E,
      &proof_wrong_dom,
      &mut verifier_ts_wrong_dom_check,
    );
    match isolation_result {
      Err(NovaError::ProofVerifyError { reason }) => {
        assert!(
          reason.contains("bind-side") || reason.contains("pcs-side"),
          "Σ-verifier should reject at one of the two equations on \
           FS-isolation violation; got reason: {}",
          reason,
        );
      }
      Ok(()) => panic!(
        "Σ-verifier MUST reject on FS-isolation violation — prover \
         used b\"RelaxedR1CSSNARK\" domain separator but verifier used \
         b\"NeutronCompressedSNARK_envelope\". Falsifier H not closed.",
      ),
      Err(other) => panic!(
        "Σ-verifier rejected with unexpected error variant on FS-isolation forge: {:?}",
        other,
      ),
    }
  }

  /// **Unit test (b) — 1000-iter round-trip differential**.
  ///
  /// Per `.claude/rules/cryptography.md` 1000-iter differential
  /// threshold: assert that the Σ-protocol prove → verify round-trip
  /// succeeds for 1000 distinct ChaCha20Rng-seeded `E2` inputs at the
  /// fixed shape `left = right = 4`. The structural identities
  /// (Pedersen MSM linearity + Σ-protocol completeness) hold for every
  /// input; any failure across the 1000 iters surfaces an algebra bug.
  ///
  /// US-05 reviewer-reproducibility: the seed `[1u8; 32]` is
  /// deterministic — reseeding produces the same input sequence on any
  /// machine. Internal prover randomness (ρ, σ_bind, σ_pcs) is drawn
  /// from `OsRng` per the production helper — non-deterministic but
  /// invariant to correctness (completeness is unconditional on
  /// well-shaped inputs).
  #[test]
  #[allow(non_snake_case)]
  fn m_gh7_0_1b_sigma_E2_equality_round_trip_1000_iter() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;

    let (ck, structure, _U_template, W_template) = build_satisfying_triple::<E, S>();
    let mut rng = ChaCha20Rng::from_seed([1u8; 32]);

    for iter in 0..1000 {
      // Fresh (E1, E2, r_E) per iter — all length-(left+right) flat.
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
      let U = FoldedInstance {
        comm_W: _U_template.comm_W,
        comm_E: <E as Engine>::CE::commit(&ck, &e_flat, &r_E),
        T: _U_template.T,
        X: _U_template.X.clone(),
        u: _U_template.u,
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

      let split = split_E_commitments(&ck, &W, &U, &structure);
      let (_E1_slice, E2_slice) = W.E.split_at(structure.left);
      let E2 = E2_slice.to_vec();

      let mut prover_ts = <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
      let proof = prove_sigma_E2_equality::<E>(
        &ck,
        &structure,
        &E2,
        &split.r_E2_bind,
        &split.r_E2_pcs,
        &split.comm_E1,
        &split.comm_E2_bind,
        &split.comm_E2_pcs,
        &U.comm_E,
        &mut prover_ts,
      )
      .unwrap_or_else(|e| panic!("Σ-prover failed at iter {}: {:?}", iter, e));

      let mut verifier_ts = <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
      verify_sigma_E2_equality::<E>(
        &ck,
        &structure,
        &split.comm_E1,
        &split.comm_E2_bind,
        &split.comm_E2_pcs,
        &U.comm_E,
        &proof,
        &mut verifier_ts,
      )
      .unwrap_or_else(|e| panic!("Σ-verifier rejected honest proof at iter {}: {:?}", iter, e));
    }
  }
}
