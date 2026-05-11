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
  neutron::relation::{FoldedInstance, FoldedWitness, Structure},
  provider::traits::DlogGroup,
  traits::{commitment::CommitmentEngineTrait, Engine, TranscriptReprTrait},
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
}
