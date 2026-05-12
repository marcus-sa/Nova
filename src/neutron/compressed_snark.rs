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
  neutron::{
    relation::{FoldedInstance, FoldedWitness, Structure},
    PublicParams, RecursiveSNARK,
  },
  provider::traits::DlogGroup,
  r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness},
  spartan::snark::{
    ProverKey as SpartanProverKey, RelaxedR1CSSNARK, VerifierKey as SpartanVerifierKey,
  },
  traits::{
    circuit::StepCircuit, commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait,
    snark::RelaxedR1CSSNARKTrait, Engine, RO2Constants, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, DerandKey,
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
  ///
  /// M.GH7.0.2 (Corrigendum #11) — prefix-basis: served by `SplitECommitments::comm_E1`
  /// without change. The same group element satisfies BOTH the off-FS Pedersen-additive
  /// binding identity AND the Spartan sibling's PCS-opening shape on the E1 side
  /// (`ck.ck[..left]` is simultaneously the prefix-of-prefix for `[E1||E2]`'s flat
  /// commitment and the prefix basis for a standalone `commit(ck, E1, r_E1)`).
  pub comm_E1: Commitment<E>,
  /// PCS-opening-side commitment to `E2`, in the **prefix** basis (Corrigendum #11):
  /// `comm_E2_pcs = MSM(E2, ck.ck[..right]) + h * r_E2_pcs`.
  ///
  /// **Field-rename from `comm_E2` at M.GH7.0.2 (Corrigendum #11 REWIRE).**
  /// The pre-Corrigendum-#11 single `comm_E2` was suffix-basis and had to serve
  /// BOTH the off-FS binding check AND the Spartan PCS-opening shape — which
  /// required different generator slices (`ck.ck[left..left+right]` vs
  /// `ck.ck[..right]`), an algebraic impossibility. Per Corrigendum #11 the
  /// envelope now ships THREE commitments via [`SplitECommitments`]:
  ///
  /// - `comm_E1` (prefix-basis on E1) — this struct's `comm_E1` field above;
  /// - `comm_E2_bind` (suffix-basis on E2) — published off-band on
  ///   [`CompressedSNARK::comm_E2_bind`] and used ONLY for the off-FS
  ///   Pedersen-additive binding check `comm_E1 + comm_E2_bind == r_U_derand.comm_E`;
  /// - `comm_E2_pcs` (prefix-basis on E2) — THIS field, consumed by the
  ///   Spartan T-claim sibling `RelaxedR1CSSNARK::prove_with_T_claim_split_error`
  ///   as the second `comm_E2` argument (matching the sibling-internal contract
  ///   at `snark.rs:1824-1825`).
  ///
  /// `to_transcript_bytes` byte-stream length is UNCHANGED by this rename
  /// (still concatenates `comm_W || comm_E1 || comm_E2_pcs || u || X` in the
  /// same order). The Σ-protocol equality-of-opening proof at M.GH7.0.1b
  /// (`SigmaE2EqualityProof`) is what ties `comm_E2_bind` and `comm_E2_pcs`
  /// to the same algebraic `E2` — without that Σ-protocol the envelope would
  /// be unsound (see module-level documentation).
  pub comm_E2_pcs: Commitment<E>,
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
      self.comm_E2_pcs.to_transcript_bytes(),
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

// ---------------------------------------------------------------------------
// M.GH7.0.2 — `neutron::CompressedSNARK<E, EE>` envelope.
//
// Authored per GH-#7 / Stage K design pin Corrigenda #8 + #10 + #11 REWIRE.
// Single-curve 2-parameter envelope (NOT the classic-Nova 5-parameter shape —
// the secondary-curve `E2` collapses under the NeutronNova IVC discipline,
// which runs a single Pasta-style cycle entirely on `E1`).
//
// # Bridge architecture (Corrigendum #10 §1.2(a) β′ + Corrigendum #11 REWIRE)
//
// The envelope bridges the IVC-final folded state
// `(r_U: FoldedInstance<E>, r_W: FoldedWitness<E>)` into the
// `BridgedNeutronInstance<E>` consumed by Spartan's T-claim sibling
// `RelaxedR1CSSNARK::prove_with_T_claim_split_error`, with a Σ-protocol
// equality-of-opening proof closing the basis-collision gap between the
// off-FS binding side (`comm_E2_bind` against the suffix basis
// `ck.ck[left..left+right]`) and the Spartan PCS-opening side
// (`comm_E2_pcs` against the prefix basis `ck.ck[..right]`):
//
//   1. Build BLINDED `SplitECommitments { comm_E1, comm_E2_bind,
//      comm_E2_pcs, r_E1, r_E2_bind, r_E2_pcs }` via [`split_E_commitments`]
//      against the (still-blinded) `(FoldedInstance, FoldedWitness,
//      Structure)` triple. By M.GH7.0.1 REWORKED helper contract,
//      `comm_E1 + comm_E2_bind == r_U.comm_E` and `r_E1 + r_E2_bind ==
//      r_W.r_E`. The third commitment `comm_E2_pcs` opens to the SAME
//      algebraic E2 in the prefix basis (Corrigendum #11).
//
//   2. Bridge `r_W → RelaxedR1CSWitness<E>` by field-copy
//      (W/r_W/E/r_E). Bridge `r_U → RelaxedR1CSInstance<E>` by field-copy
//      (comm_W/comm_E/u/X — T is held separately and threaded into
//      `BridgedNeutronInstance.T`).
//
//   3. Derandomize the witness via `RelaxedR1CSWitness::derandomize()`,
//      yielding `(W_derand, blind_W, blind_E)`. Derandomize the instance
//      via `RelaxedR1CSInstance::derandomize(&dk, &blind_W, &blind_E)`
//      against the same dk, mirror of `vendor/nova/src/nova/mod.rs:841-848`.
//
//   4. Derandomize all THREE split-E commitments against their respective
//      blindings via `E::CE::derandomize(&dk, &comm, &r)`. Pedersen-
//      additive identity is preserved at the derandomized layer:
//          comm_E1_derand + comm_E2_bind_derand
//        = MSM(E1, ck[..left]) + MSM(E2_pad, ck[..left+right])
//        = MSM([E1 || E2], ck[..left+right])
//        = r_U_derand.comm_E
//      i.e., the binding check `comm_E1 + comm_E2_bind == r_U_derand.comm_E`
//      holds against the derandomized envelope's `r_U_derand.comm_E`.
//
//   5. Construct an **envelope-side transcript** with the distinct domain
//      separator `b"NeutronCompressedSNARK_envelope"` (FS-isolation from
//      the Spartan sibling's `b"RelaxedR1CSSNARK"` transcript per
//      Falsifier H). Absorb `(vk, ...)` under the SAME ordering pin
//      §1.2(a) Primitive 6 requires before invoking the Σ-protocol
//      prover. The Σ-protocol helper itself absorbs `(T_bind, T_pcs)` and
//      squeezes `α` — the helper requires the caller to absorb
//      `(vk_digest, r_U_comm_E, comm_E1, comm_E2_bind, comm_E2_pcs)`
//      BEFORE calling, which we do.
//
//   6. Invoke `prove_sigma_E2_equality(...)` over the envelope-side
//      transcript, producing a `SigmaE2EqualityProof<E>` that ties
//      `comm_E2_bind` and `comm_E2_pcs` to the SAME algebraic E2.
//
//   7. Construct `U_bridged: BridgedNeutronInstance` carrying
//      `(comm_W: r_U_derand.comm_W, comm_E1: comm_E1_derand,
//        comm_E2_pcs: comm_E2_pcs_derand, u: r_U.u, X: r_U.X.clone(),
//        T: r_U.T)`. The scalars `(u, X, T)` are NOT affected by
//      derandomization (only group elements are).
//
//   8. Invoke `RelaxedR1CSSNARK::prove_with_T_claim_split_error(
//          &ck, &pk.pk_spartan, &S, &U_bridged, &W_derand,
//          comm_E1_derand, comm_E2_pcs_derand, &E1, &E2,
//          r_U.T, r_E1, r_E2_pcs)`. The sibling receives `(comm_E1,
//      comm_E2_pcs)` in the prefix basis on both sides (matches sibling-
//      internal contract at `snark.rs:1445-1446` + `:1824-1825`); the
//      sibling initialises its OWN fresh `b"RelaxedR1CSSNARK"` transcript
//      at entry — FS-isolated from the envelope-side transcript at the
//      domain-separator stage.
//
// # Verifier discipline (Corrigendum #8 (iv-B) + Corrigendum #10 + #11)
//
// (1) Off-FS Pedersen-additive binding check (Corrigendum #8 (iv-B)):
//     `self.U_bridged.comm_E1 + self.comm_E2_bind == self.r_U_derand_comm_E`
//     at the group level. If false, reject with
//     `NovaError::ProofVerifyError { reason: "off-FS Pedersen-additive
//     binding rejected" }`.
//
// (2) Σ-protocol equality-of-opening (Corrigendum #11): build a FRESH
//     `E::TE::new(b"NeutronCompressedSNARK_envelope")` transcript,
//     absorb `(vk, r_U_derand_comm_E, comm_E1, comm_E2_bind, comm_E2_pcs)`
//     in the SAME fixed order as the prover, invoke
//     `verify_sigma_E2_equality(...)`. Rejection delegates to the helper's
//     own reason strings ("Sigma E2 equality-of-opening rejected at
//     bind-side" / "pcs-side").
//
// (3) Spartan T-claim sibling verify (Corrigendum #10): delegate to
//     `self.snark_spartan.verify_with_T_claim_split_error(
//         &vk.vk_spartan, &self.U_bridged,
//         self.U_bridged.comm_E1, self.U_bridged.comm_E2_pcs)`.
//     The sibling initialises its OWN fresh `b"RelaxedR1CSSNARK"`
//     transcript at entry (verify-don't-assume against
//     `snark.rs::verify_with_T_claim_split_error` body, line 1149) — the
//     envelope MUST NOT construct a transcript that the sibling reads.
//
// (4) IVC final state check: assert `zn == self.zn`.
//
// # FS-isolation discipline (Falsifier H)
//
// Two independent `E::TE` instances exist in the verify flow:
//   - envelope-side: `b"NeutronCompressedSNARK_envelope"` (used only for
//     the Σ-protocol verify; NEVER threaded into the Spartan sibling);
//   - Spartan-side: `b"RelaxedR1CSSNARK"` (constructed fresh inside
//     `RelaxedR1CSSNARK::verify_with_T_claim_split_error` line 1149;
//     NEVER threaded out to the envelope).
// The two transcripts are byte-independent at the domain-separator stage.
// A future maintainer who threads a single `E::TE` through both layers
// would re-bind the Σ-protocol `α` to the Spartan-side absorb log and
// break FS-soundness — see Falsifier H STOP-AND-ASK at dispatch §"STOP-
// AND-ASK triggers". The Σ-protocol's b"...envelope" dom-sep test (case
// (e) at compressed_snark.rs:1387-1432) is the empirical close.
// ---------------------------------------------------------------------------

/// `CompressedSNARK` envelope for the NeutronNova IVC produced by
/// [`RecursiveSNARK`]. Wraps a Spartan-side `RelaxedR1CSSNARK<E, EE>` proof
/// produced via the T-claim sibling
/// [`RelaxedR1CSSNARK::prove_with_T_claim_split_error`] (Corrigendum #10),
/// together with the derandomized [`BridgedNeutronInstance`], the off-band
/// `comm_E2_bind` (suffix-basis binding-side commitment), the
/// [`SigmaE2EqualityProof`] that ties `comm_E2_bind` to `comm_E2_pcs` at
/// the same algebraic E2 (Corrigendum #11), and the IVC final state `zn`.
///
/// Single-curve 2-parameter (`E`, `EE`) — the secondary curve `E2` of the
/// classic-Nova 5-parameter envelope collapses under NeutronNova's single-
/// curve IVC discipline.
///
/// # Soundness anchors (Corrigenda #8 + #10 + #11)
///
/// - Off-FS Pedersen-additive binding `comm_E1 + comm_E2_bind ==
///   r_U_derand.comm_E` enforced at the envelope (Corrigendum #8 (iv-B)).
/// - Σ-protocol equality-of-opening ties `comm_E2_bind` (suffix-basis,
///   used only for the binding identity) to `comm_E2_pcs` (prefix-basis,
///   consumed by Spartan PCS-opening) at the same algebraic E2
///   (Corrigendum #11 Primitive 6).
/// - Tensor-form factorisation `eval_E1 · eval_E2 == eval_E` enforced
///   inside the Spartan sibling's verifier (`snark.rs:1144-1147`).
/// - Neutron-form residue `sum_x full_E(x)·(Az·Bz − Cz) = T` discharged
///   by the Spartan sibling's outer sumcheck with claim = T (Corrigendum
///   #10 §1.2(a)).
/// - `T_claim` absorb-before-squeeze discipline enforced inside the
///   Spartan sibling (`snark.rs:953-954`).
// `Clone` / `Debug` not derived to match the upstream Spartan sibling
// (`spartan::snark::RelaxedR1CSSNARK` does not derive `Clone`; its
// `EE::EvaluationArgument` associated type is not `Debug`). The M.GH7.1
// polish step will reconsider if downstream consumers need these.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
#[allow(non_snake_case)]
pub struct CompressedSNARK<E, EE>
where
  E: Engine,
  EE: EvaluationEngineTrait<E>,
  E::GE: DlogGroup,
{
  /// Derandomized bridged neutron instance consumed by the Spartan sibling.
  /// Carries `(comm_W_derand, comm_E1_derand, comm_E2_pcs_derand, u, X, T)`.
  /// The verifier runs the off-FS Pedersen-additive binding `comm_E1 +
  /// comm_E2_bind == r_U_derand_comm_E` against the envelope-published
  /// `r_U_derand_comm_E` field below, with `comm_E2_bind` held separately
  /// (NOT in `U_bridged`, which carries `comm_E2_pcs`).
  pub U_bridged: BridgedNeutronInstance<E>,

  /// Derandomized R1CS-side error-vector commitment `r_U_derand.comm_E`.
  /// Published so the verifier can run the off-FS Pedersen-additive
  /// binding check `comm_E1 + comm_E2_bind == r_U_derand_comm_E` without
  /// re-deriving the derandomization on the verifier side. (The blinding
  /// factors used to derandomize are PROVER-SIDE; the verifier cannot
  /// reconstruct them from `vk` alone.)
  pub r_U_derand_comm_E: Commitment<E>,

  /// Derandomized SUFFIX-basis commitment to `E2`. Used ONLY for the
  /// off-FS Pedersen-additive binding identity `comm_E1 + comm_E2_bind ==
  /// r_U_derand_comm_E` per Corrigendum #8 (iv-B). The Σ-protocol at
  /// [`Self::sigma_E2_equality`] ties this to `U_bridged.comm_E2_pcs` at
  /// the same algebraic E2 (Corrigendum #11).
  pub comm_E2_bind: Commitment<E>,

  /// Σ-protocol equality-of-opening proof (Corrigendum #11 Primitive 6).
  /// Asserts `comm_E2_bind` and `U_bridged.comm_E2_pcs` open to the same
  /// `E2 ∈ F^{right}` under their respective bases. Without this proof
  /// the envelope would be unsound (a malicious prover could supply
  /// `comm_E2_pcs := MSM(E2', ck.ck[..right]) + h · r_E2_pcs` for any
  /// `E2' ≠ E2`, and the off-FS binding check would not catch the
  /// substitution).
  pub sigma_E2_equality: SigmaE2EqualityProof<E>,

  /// Spartan-side T-claim sibling proof produced by
  /// [`RelaxedR1CSSNARK::prove_with_T_claim_split_error`].
  pub snark_spartan: RelaxedR1CSSNARK<E, EE>,

  /// IVC final state, mirrors `nova::CompressedSNARK::zn` precedent.
  pub zn: Vec<E::Scalar>,
}

/// Prover key for [`CompressedSNARK`]. Wraps the Spartan-side prover key
/// produced by `<RelaxedR1CSSNARK<E, EE>>::setup`. Single-curve 2-parameter
/// minimum surface; M.GH7.1 polishes the surrounding fields.
// `Clone` / `Debug` not derived: matches upstream `spartan::snark::ProverKey`.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E, EE>
where
  E: Engine,
  EE: EvaluationEngineTrait<E>,
{
  pub(crate) pk_spartan: SpartanProverKey<E, EE>,
}

/// Verifier key for [`CompressedSNARK`]. Wraps the Spartan-side verifier
/// key + the `F_arity` / `ro_consts` / `pp_digest` / `lookup_fold_k` /
/// `shape_registry_digest` fields pinned by GH-#7 design pin §3.3 +
/// the `DerandKey<E>` / `Structure<E>` / `CommitmentKey<E>` carried for
/// the off-FS Pedersen-additive binding check and the Σ-protocol verify.
///
/// # Field set per pin §3.3 (verbatim ordering)
///
/// The pin (`docs/research/cryptography/gh-7-stage-k-compressed-snark-design-pin-2026-05-11.md`
/// §3.3 lines 1318-1326) names the following six fields plus a `...`
/// trail covering pragmatic additions:
///
/// - `F_arity: usize`
/// - `ro_consts: RO2Constants<E>`
/// - `pp_digest: E::Scalar`
/// - `vk_spartan: <RelaxedR1CSSNARK<E, EE> as RelaxedR1CSSNARKTrait<E>>::VerifierKey`
/// - `lookup_fold_k: usize`
/// - `shape_registry_digest: E::Scalar`
///
/// Plus the pin's `...` trail, this VK additionally carries the
/// M.GH7.0.2-pragmatic fields needed by the verifier body:
///
/// - `dk: DerandKey<E>` — consumed by the verifier's off-FS Pedersen-
///   additive binding reconstruction.
/// - `structure: Structure<E>` — needed by [`verify_sigma_E2_equality`]
///   for the zero-padded suffix-basis trick (`structure.left` /
///   `structure.right`).
/// - `ck: CommitmentKey<E>` — needed by the Σ-protocol verifier's
///   `CE::commit` calls.
///
/// # `shape_registry_digest` populate-state
///
/// For M.GH7.1 we surface the field at the type-signature level and
/// initialise it from `pp.shape_registry` as a Poseidon-or-equivalent
/// digest in the `setup` call; at the trivial / non-lookup-fold default
/// (`shape_registry.is_empty()`), the field is `E::Scalar::ZERO`. The
/// authoritative digest construction is wired by M.GH7.2 (M.7 shape-
/// registry assertion) — for STAGE 0 the simpler `digest_via_absorb`
/// shape suffices.
///
/// # `pp_digest`
///
/// Populated from `PublicParams::digest()` at `setup`. Used by
/// downstream consumers for `b"vk"` absorb discipline; the M.GH7.0.2
/// envelope absorbs `vk_spartan.digest()` directly (Corrigendum #11
/// Primitive 6 absorb-order), so the envelope-level `pp_digest` is
/// carried for the M.GH7.1+ extended `verify` post-polish (per the
/// `nova::CompressedSNARK::verify` precedent at `nova/mod.rs:935-960`).
// `Clone` / `Debug` not derived: matches upstream `spartan::snark::VerifierKey`.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
#[allow(non_snake_case)]
pub struct VerifierKey<E, EE>
where
  E: Engine,
  EE: EvaluationEngineTrait<E>,
  E::GE: DlogGroup,
{
  // === Pin §3.3 verbatim fields ===
  pub(crate) F_arity: usize,
  pub(crate) ro_consts: RO2Constants<E>,
  pub(crate) pp_digest: E::Scalar,
  pub(crate) vk_spartan: SpartanVerifierKey<E, EE>,
  pub(crate) lookup_fold_k: usize,
  pub(crate) shape_registry_digest: E::Scalar,

  // === Pin §3.3 `...` trail — pragmatic additions for M.GH7.0.2 verifier ===
  pub(crate) dk: DerandKey<E>,
  /// Carried for the Σ-protocol verify (which needs `structure.left`
  /// + `structure.right` to perform the zero-padded suffix-basis trick
  /// inside `verify_sigma_E2_equality`). Cloned from `PublicParams::structure`
  /// at `setup`.
  pub(crate) structure: Structure<E>,
  /// Carried for the off-FS Pedersen-additive binding check (which needs
  /// access to `ck` for the Σ-protocol verifier's `CE::commit` calls).
  pub(crate) ck: CommitmentKey<E>,
}

impl<E, EE> CompressedSNARK<E, EE>
where
  E: Engine,
  EE: EvaluationEngineTrait<E>,
  E::GE: DlogGroup,
{
  /// Creates prover and verifier keys for [`CompressedSNARK`].
  ///
  /// Forwards to `<RelaxedR1CSSNARK<E, EE> as RelaxedR1CSSNARKTrait<E>>::setup`
  /// against the neutron R1CS shape and commitment key, mirroring
  /// `vendor/nova/src/nova/mod.rs:765-766` for the single-curve case.
  ///
  /// # M.GH7.1 polish: pin-§3.3 verifier-key field set
  ///
  /// Per `docs/research/cryptography/gh-7-stage-k-compressed-snark-design-pin-2026-05-11.md`
  /// §3.3 lines 1318-1326, the verifier key populates:
  ///
  /// - `F_arity` ← `pp.F_arity`
  /// - `ro_consts` ← `pp.ro_consts.clone()`
  /// - `pp_digest` ← `pp.digest()` (forces `OnceCell` initialisation per
  ///   the upstream `nova/mod.rs:181` precedent)
  /// - `vk_spartan` ← `<RelaxedR1CSSNARK<E, EE>>::setup` output
  /// - `lookup_fold_k` ← `pp.lookup_fold_k` (under `feature = "lookup-fold"`;
  ///   `0` ⇒ no lookup-fold path)
  /// - `shape_registry_digest` ← `E::Scalar::ZERO` for STAGE 0 / non-
  ///   lookup-fold paths. The authoritative digest construction over
  ///   `pp.shape_registry` is wired by M.GH7.2 (M.7 shape-registry
  ///   assertion); STAGE 0 (M.GH7.0.3 + M.GH7.0.4) exercises the
  ///   trivial `(vec![], 0, 0)` `PublicParams::setup` shape per
  ///   Corrigendum #13, so the digest is structurally `ZERO`.
  ///
  /// The pragmatic `...`-trail fields (`dk`, `structure`, `ck`) are
  /// populated as in M.GH7.0.2 — they are load-bearing for the off-FS
  /// binding check + Σ-protocol verify.
  pub fn setup<E2, C>(
    pp: &PublicParams<E, E2, C>,
  ) -> Result<(ProverKey<E, EE>, VerifierKey<E, EE>), NovaError>
  where
    E2: Engine<Base = <E as Engine>::Scalar>,
    E: Engine<Base = <E2 as Engine>::Scalar>,
    C: StepCircuit<E::Scalar>,
  {
    let (pk_spartan, vk_spartan) =
      <RelaxedR1CSSNARK<E, EE> as RelaxedR1CSSNARKTrait<E>>::setup(&pp.ck, &pp.structure.S)?;

    // Populate `lookup_fold_k` from `pp` under `feature = "lookup-fold"`;
    // outside that feature it is unconditionally `0` (the lookup-fold
    // path is absent from the augmented circuit). The `cfg`-gate here
    // matches the `cfg`-gate on the corresponding `PublicParams` field
    // at `neutron/mod.rs:130-132`.
    #[cfg(feature = "lookup-fold")]
    let lookup_fold_k = pp.lookup_fold_k;
    #[cfg(not(feature = "lookup-fold"))]
    let lookup_fold_k = 0usize;

    // STAGE 0 / M.GH7.1 polish: shape_registry_digest is structurally
    // ZERO at the default `shape_registry.is_empty()` state. M.GH7.2 /
    // M.GH7.4 will replace this with the Poseidon-or-equivalent digest
    // construction binding the M.7 shape-registry assertion.
    let shape_registry_digest = <E as Engine>::Scalar::ZERO;

    let pk = ProverKey { pk_spartan };
    let vk = VerifierKey {
      F_arity: pp.F_arity,
      ro_consts: pp.ro_consts.clone(),
      pp_digest: pp.digest(),
      vk_spartan,
      lookup_fold_k,
      shape_registry_digest,
      dk: <E::CE as CommitmentEngineTrait<E>>::derand_key(&pp.ck),
      structure: pp.structure.clone(),
      ck: pp.ck.clone(),
    };
    Ok((pk, vk))
  }

  /// Create a new [`CompressedSNARK`] from an IVC-final [`RecursiveSNARK`].
  ///
  /// Thin wrapper around [`CompressedSNARK::prove_from_parts`] — extracts
  /// the final folded state `(r_U, r_W, zi)` from the [`RecursiveSNARK`]
  /// and forwards to the parts-based prover. M.GH7.1 polishes the
  /// `RecursiveSNARK` field-access discipline.
  pub fn prove<E2, C>(
    pp: &PublicParams<E, E2, C>,
    pk: &ProverKey<E, EE>,
    recursive_snark: &RecursiveSNARK<E, E2, C>,
  ) -> Result<Self, NovaError>
  where
    E2: Engine<Base = <E as Engine>::Scalar>,
    E: Engine<Base = <E2 as Engine>::Scalar>,
    C: StepCircuit<E::Scalar>,
  {
    Self::prove_from_parts(
      &pp.ck,
      &pp.structure,
      pk,
      &recursive_snark.r_U,
      &recursive_snark.r_W,
      recursive_snark.zi.clone(),
    )
  }

  /// Parts-based prover (M.GH7.0.2 acceptance-test entry point).
  ///
  /// Authored as a `pub(crate)` parts-based prover so the in-crate
  /// acceptance test can exercise the envelope without needing to drive a
  /// full `RecursiveSNARK::prove_step` cycle (which hits the pre-existing
  /// `batch_diff_size` vendor shape issue at `spartan/mod.rs:175-189` when
  /// `W.W.len() != max(left, right)`). The path through `prove` above is
  /// the production-facing surface; this is the testable surface.
  ///
  /// Executes the Corrigendum #10 §1.2(a) β′ bridge + Corrigendum #11
  /// Σ-protocol equality-of-opening composition as documented in the
  /// module-level docblock above.
  #[allow(non_snake_case)]
  pub(crate) fn prove_from_parts(
    ck: &CommitmentKey<E>,
    structure: &Structure<E>,
    pk: &ProverKey<E, EE>,
    r_U: &FoldedInstance<E>,
    r_W: &FoldedWitness<E>,
    zn: Vec<E::Scalar>,
  ) -> Result<Self, NovaError> {
    // (1) Build BLINDED SplitECommitments via M.GH7.0.1 REWORKED helper.
    let SplitECommitments {
      comm_E1: comm_E1_blinded,
      comm_E2_bind: comm_E2_bind_blinded,
      comm_E2_pcs: comm_E2_pcs_blinded,
      r_E1,
      r_E2_bind,
      r_E2_pcs,
    } = split_E_commitments(ck, r_W, r_U, structure);

    // (2) Bridge FoldedWitness → RelaxedR1CSWitness by field-copy.
    // The `E` field is carried verbatim (flat `[E1 || E2]` of length
    // `left + right`); `RelaxedR1CSWitness::pad(&S)` extends it with zeros
    // inside the Spartan sibling. The pad zeros do NOT participate in the
    // prover's algebra — `prove_with_T_claim_split_error` reads only `W.W`
    // (for `z`) and the externally-supplied `(E1, E2)` slices; `W.E` is
    // unused post-pad by the prover (and the verifier never sees it).
    let bridged_witness: RelaxedR1CSWitness<E> = RelaxedR1CSWitness {
      W: r_W.W.clone(),
      r_W: r_W.r_W,
      E: r_W.E.clone(),
      r_E: r_W.r_E,
    };

    // Bridge FoldedInstance → RelaxedR1CSInstance by field-copy. `T` is
    // NOT carried on `RelaxedR1CSInstance` — it threads through
    // `U_bridged.T` separately.
    let bridged_instance: RelaxedR1CSInstance<E> = RelaxedR1CSInstance {
      comm_W: r_U.comm_W,
      comm_E: r_U.comm_E,
      u: r_U.u,
      X: r_U.X.clone(),
    };

    // (3) Derandomize the witness and instance (mirror of
    // `vendor/nova/src/nova/mod.rs:841-848`).
    let dk = <E::CE as CommitmentEngineTrait<E>>::derand_key(ck);
    let (W_derand, blind_W, blind_E) = bridged_witness.derandomize();
    let U_derand = bridged_instance.derandomize(&dk, &blind_W, &blind_E);

    // (4) Derandomize all THREE split-E commitments. The Pedersen-additive
    // identity is preserved at the derandomized layer because the
    // derandomize op strips the `h · r_·` term symmetrically across both
    // halves of the binding sum:
    //   comm_E1_derand + comm_E2_bind_derand
    //     = (comm_E1_blinded - h·r_E1) + (comm_E2_bind_blinded - h·r_E2_bind)
    //     = comm_E1_blinded + comm_E2_bind_blinded - h·(r_E1 + r_E2_bind)
    //     = U.comm_E - h·W.r_E
    //     = U_derand.comm_E.
    let comm_E1_derand =
      <E::CE as CommitmentEngineTrait<E>>::derandomize(&dk, &comm_E1_blinded, &r_E1);
    let comm_E2_bind_derand =
      <E::CE as CommitmentEngineTrait<E>>::derandomize(&dk, &comm_E2_bind_blinded, &r_E2_bind);
    let comm_E2_pcs_derand =
      <E::CE as CommitmentEngineTrait<E>>::derandomize(&dk, &comm_E2_pcs_blinded, &r_E2_pcs);

    // Split `(E1, E2)` from the FLAT `r_W.E` per Structure's `left`
    // partition. The slices are length `structure.left` and
    // `structure.right` respectively, matching the Σ-protocol's vector-
    // arithmetic guards (Falsifier J) and the Spartan sibling's
    // `assert_eq!(E1.len(), left)` / `assert_eq!(E2.len(), right)` at
    // `snark.rs:965-966`.
    let (E1_slice, E2_slice) = r_W.E.split_at(structure.left);
    let E1: Vec<E::Scalar> = E1_slice.to_vec();
    let E2: Vec<E::Scalar> = E2_slice.to_vec();

    // (5) Initialise the ENVELOPE-side transcript with the distinct
    // domain separator `b"NeutronCompressedSNARK_envelope"`. This is the
    // FS-isolation discipline (Falsifier H): the Spartan sibling will
    // initialise its OWN fresh `b"RelaxedR1CSSNARK"` transcript inside
    // `prove_with_T_claim_split_error` — the two transcripts NEVER share
    // bytes.
    //
    // Pre-helper absorb order (pin §1.2(a) Primitive 6):
    //   ts_env.absorb(b"vk",            &pk.vk_spartan.vk_digest)
    //   ts_env.absorb(b"r_U_comm_E",    &r_U_derand_comm_E)   -- via helper
    //   ts_env.absorb(b"comm_E1",       &comm_E1_derand)      -- via helper
    //   ts_env.absorb(b"comm_E2_bind",  &comm_E2_bind_derand) -- via helper
    //   ts_env.absorb(b"comm_E2_pcs",   &comm_E2_pcs_derand)  -- via helper
    //   ts_env.absorb(b"sigma_T_bind",  &T_bind)              -- inside helper
    //   ts_env.absorb(b"sigma_T_pcs",   &T_pcs)               -- inside helper
    //   α := ts_env.squeeze(b"sigma_E2_equality_alpha")       -- inside helper
    //
    // The envelope owns the `b"vk"` absorb (pin line 258). The helper
    // body owns the rest (compressed_snark.rs:656-662 above).
    let mut ts_env = <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
    ts_env.absorb(b"vk", &pk.pk_spartan.vk_digest);

    // (6) Σ-protocol equality-of-opening prove (Corrigendum #11
    // Primitive 6). Ties `comm_E2_bind_derand` and `comm_E2_pcs_derand` to
    // the same `E2 ∈ F^{right}` under their respective bases.
    //
    // BLINDING CONSISTENCY: the Σ-protocol's bind-side acceptance equation
    //   MSM(z, ck.ck[left..left+right]) + h·z_r_bind == T_bind + α·comm_E2_bind
    // requires the published `comm_E2_bind` to be the commit-with-blinding-
    // `r_E2_bind` Pedersen value. We publish the DERANDOMIZED
    // `comm_E2_bind_derand` (the h·r_E2_bind term stripped), so the
    // effective blinding on the published commitment is ZERO. Same logic
    // on the pcs-side. Threading the original `(r_E2_bind, r_E2_pcs)`
    // would make LHS exceed RHS by `α·h·r_E2_bind` (resp.
    // `α·h·r_E2_pcs`) on each equation, and the verifier would reject
    // honest proofs — empirically observed at the 2026-05-12 GREEN-halt
    // before the blinding-consistency fix. Pass ZERO blindings to match
    // the derandomized commitments published in the envelope.
    let sigma_E2_equality = prove_sigma_E2_equality::<E>(
      ck,
      structure,
      &E2,
      &E::Scalar::ZERO,
      &E::Scalar::ZERO,
      &comm_E1_derand,
      &comm_E2_bind_derand,
      &comm_E2_pcs_derand,
      &U_derand.comm_E,
      &mut ts_env,
    )?;

    // (7) Construct U_bridged carrying the derandomized prefix-basis
    // commitments (comm_E1, comm_E2_pcs) and the running neutron-form
    // claim T. `(u, X, T)` are unaffected by derandomization.
    let U_bridged = BridgedNeutronInstance::<E> {
      comm_W: U_derand.comm_W,
      comm_E1: comm_E1_derand,
      comm_E2_pcs: comm_E2_pcs_derand,
      u: r_U.u,
      X: r_U.X.clone(),
      T: r_U.T,
    };

    // (8) Invoke the Spartan T-claim sibling (Corrigendum #10 + #11).
    // The sibling consumes `(comm_E1_derand, comm_E2_pcs_derand)` — BOTH
    // in the prefix basis — matching the sibling-internal contract at
    // `snark.rs:1824-1825` where the sibling's own tests commit via
    // `CE::commit(ck, &E2, &r)` against `ck.ck[..right]` with ZERO
    // blinding. The trailing blinding scalars `(_r_E1, _r_E2)` are
    // underscored in the sibling body (`snark.rs:936-937`) — they are
    // formal arguments only and pass ZERO for blinding-consistency with
    // the derandomized commitments (`comm_E1_derand` and
    // `comm_E2_pcs_derand` have their `h·r` terms stripped, so the
    // effective blinding on the published commitment is ZERO; this
    // matches what the M.GH7.0.0b sibling test passes at `snark.rs:1822-
    // 1823` where `r_E1 = r_E2 = E::Scalar::ZERO`).
    let snark_spartan = RelaxedR1CSSNARK::<E, EE>::prove_with_T_claim_split_error(
      ck,
      &pk.pk_spartan,
      &structure.S,
      &U_bridged,
      &W_derand,
      comm_E1_derand,
      comm_E2_pcs_derand,
      &E1,
      &E2,
      r_U.T,
      E::Scalar::ZERO,
      E::Scalar::ZERO,
    )?;

    Ok(CompressedSNARK {
      U_bridged,
      r_U_derand_comm_E: U_derand.comm_E,
      comm_E2_bind: comm_E2_bind_derand,
      sigma_E2_equality,
      snark_spartan,
      zn,
    })
  }

  /// Verify the [`CompressedSNARK`] envelope.
  ///
  /// (1) **Off-FS Pedersen-additive binding check** (LOAD-BEARING per
  ///     Corrigendum #8 (iv-B)): assert
  ///     `self.U_bridged.comm_E1 + self.comm_E2_bind ==
  ///      self.r_U_derand_comm_E` at the GROUP level. If false, reject
  ///     with [`NovaError::ProofVerifyError`].
  /// (2) **Σ-protocol equality-of-opening verify** (Corrigendum #11
  ///     Primitive 6): build a FRESH `E::TE` initialised with
  ///     `b"NeutronCompressedSNARK_envelope"`, absorb `(vk, r_U_derand_comm_E,
  ///     comm_E1, comm_E2_bind, comm_E2_pcs)` in the SAME fixed order as
  ///     the prover, invoke
  ///     [`verify_sigma_E2_equality`]. The helper's own rejections
  ///     ("bind-side" / "pcs-side") propagate up.
  /// (3) **Spartan T-claim sibling verify**: delegate to
  ///     [`RelaxedR1CSSNARK::verify_with_T_claim_split_error`] for the
  ///     outer-sumcheck-with-claim-T + tensor-factorisation +
  ///     inner-sumcheck + batch-eval + PCS opening (Corrigendum #10).
  /// (4) **IVC final state check**: assert `zn == self.zn` (caller-
  ///     supplied final state matches prover-claimed final state).
  ///
  /// Note: full `pp_digest` / `num_steps` / output-hash check (the
  /// `nova::CompressedSNARK::verify` precedent at `nova/mod.rs:935-960`)
  /// is M.GH7.1 polish. For M.GH7.0.2 STAGE 0 we check the binding +
  /// the Σ-protocol + the sibling proof + the zn match — which is
  /// sufficient for the envelope-shape acceptance gate.
  ///
  /// # Arguments
  /// - `vk`: the verifier key produced by [`CompressedSNARK::setup`].
  /// - `_num_steps`: reserved for the M.GH7.1 polish; currently unused.
  /// - `_z0`: reserved for the M.GH7.1 polish; currently unused.
  /// - `zn`: caller-supplied final state, must match `self.zn`.
  ///
  /// # Returns
  /// `Ok(self.zn.clone())` on success.
  pub fn verify(
    &self,
    vk: &VerifierKey<E, EE>,
    _num_steps: usize,
    _z0: &[E::Scalar],
    zn: &[E::Scalar],
  ) -> Result<Vec<E::Scalar>, NovaError> {
    // (1) Off-FS Pedersen-additive binding check (Corrigendum #8 (iv-B)).
    // The group equation `comm_E1 + comm_E2_bind == r_U_derand_comm_E`
    // must hold byte-equal — `comm_E1` is the prefix-basis half on E1,
    // `comm_E2_bind` is the suffix-basis half on E2, and their flat sum
    // matches `r_U_derand.comm_E = MSM([E1||E2], ck.ck[..left+right])`
    // at the derandomized layer (no h·r terms remain).
    if self.U_bridged.comm_E1 + self.comm_E2_bind != self.r_U_derand_comm_E {
      return Err(NovaError::ProofVerifyError {
        reason:
          "off-FS Pedersen-additive binding rejected: comm_E1 + comm_E2_bind != r_U_derand_comm_E"
            .to_string(),
      });
    }

    // (2) Σ-protocol equality-of-opening verify (Corrigendum #11
    // Primitive 6). Initialise a FRESH envelope-side transcript at the
    // SAME domain separator as the prover; absorb `vk_digest` under the
    // SAME `b"vk"` label; the helper continues the absorb sequence in the
    // fixed order pinned at §1.2(a) lines 259-265.
    //
    // FS-isolation discipline: this transcript is NEVER threaded into the
    // Spartan sibling — the sibling constructs its OWN fresh
    // `b"RelaxedR1CSSNARK"` transcript at entry to
    // `verify_with_T_claim_split_error` (snark.rs:1149). Threading a
    // single instance through both layers would re-bind α to the
    // Spartan-side absorb log; Falsifier H STOP-AND-ASK in dispatch.
    use crate::traits::snark::DigestHelperTrait;
    let mut ts_env = <E as Engine>::TE::new(b"NeutronCompressedSNARK_envelope");
    ts_env.absorb(b"vk", &vk.vk_spartan.digest());
    verify_sigma_E2_equality::<E>(
      &vk.ck,
      &vk.structure,
      &self.U_bridged.comm_E1,
      &self.comm_E2_bind,
      &self.U_bridged.comm_E2_pcs,
      &self.r_U_derand_comm_E,
      &self.sigma_E2_equality,
      &mut ts_env,
    )?;

    // (3) Delegate the FS-discipline + sumcheck + PCS verify to the
    // Spartan sibling. The sibling internally absorbs `vk` → `U_bridged`
    // → `T_claim` BEFORE any squeeze (Corrigendum #10 Primitive 5
    // binding). The two commitments threaded are the prefix-basis
    // `(comm_E1, comm_E2_pcs)` matching the sibling-internal contract.
    self.snark_spartan.verify_with_T_claim_split_error(
      &vk.vk_spartan,
      &self.U_bridged,
      self.U_bridged.comm_E1,
      self.U_bridged.comm_E2_pcs,
    )?;

    // (4) IVC final state check.
    if zn != self.zn.as_slice() {
      return Err(NovaError::ProofVerifyError {
        reason: "Caller-supplied zn does not match prover-claimed self.zn".to_string(),
      });
    }

    Ok(self.zn.clone())
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

  // =========================================================================
  // M.GH7.0.2 tests (Corrigenda #8 + #10 + #11) — `neutron::CompressedSNARK`
  // envelope round-trip + binding-check + Σ-protocol verify.
  //
  // Per roadmap step 01-04 criteria:
  //
  // (a) **Acceptance**: round-trip prove/verify against a satisfying
  //     neutron-form `(FoldedInstance, FoldedWitness, Structure)` triple at
  //     `left = right = 2, num_cons = 4` (the shape floor that sidesteps the
  //     pre-existing `batch_diff_size` panic at `spartan/mod.rs:175-189` —
  //     mirrors the M.GH7.0.0b sibling test's shape choice). Asserts
  //     `CompressedSNARK::verify` returns `Ok(zn)` AND the off-FS Pedersen-
  //     additive binding holds as an INDEPENDENT structural assertion (not
  //     just verify happy-path — pins the binding check is wired).
  //
  // (b) **Negative — off-FS binding**: perturb `comm_E2_bind` by `+δ` post-
  //     prove; assert `verify` rejects at check (1) with the "off-FS
  //     Pedersen-additive binding rejected" reason. This pins the binding
  //     check is load-bearing and fires BEFORE the Σ-protocol verify.
  //
  // (c) **Negative — Σ-protocol**: perturb `sigma_E2_equality.z[0]` post-
  //     prove; assert `verify` rejects at check (2) with a Σ-protocol
  //     reason string ("Sigma E2 equality-of-opening rejected at bind-side"
  //     or "...pcs-side"). This pins the Σ-protocol verify is wired and
  //     load-bearing.
  //
  // (d) **1000-iter differential** (US-05 reviewer-reproducibility): 1000
  //     ChaCha20Rng-seeded round-trips at shape `(left=2, right=2)`. The
  //     shape sweep `{(4,2), (2,4), (4,4)}` is NOT exercised here — those
  //     shapes hit the `batch_diff_size` panic in the prove path. Coverage
  //     across shapes is provided by the algebra-only assertions in
  //     M.GH7.0.1 REWORKED (`msm_linearity_prefix_suffix_decomposition_*`)
  //     and M.GH7.0.1b (`m_gh7_0_1b_sigma_E2_equality_round_trip_1000_iter`).
  //
  // Test budget: 3 distinct behaviors × 2 = 6 unit-test budget; 2 tests
  // authored (one composite acceptance + 1000-iter differential, one
  // composite negative covering both rejection paths) — well within
  // budget.
  // =========================================================================

  /// Build a satisfying **neutron-form** `(ck, Structure, FoldedInstance,
  /// FoldedWitness, E1, E2, T)` 7-tuple at a fixed `(left, right)` shape
  /// with `num_cons = left * right`, `num_vars = num_cons`, `num_io = 1`,
  /// `u = 1`, `X = [0]`. Matrices: `A[i,i] = 1` (so `Az = W`), `B[i,
  /// num_vars] = 1` (so `Bz = [u; num_cons]`), `C = 0` (so `Cz = [0;
  /// num_cons]`). Witness `W` and `(E1, E2)` drawn uniform random; flat
  /// `W.E = [E1 || E2]` of length `left + right` per the
  /// `FoldedWitness::default` invariant (`relation.rs:611`). Running claim
  /// `T = sum_x full_E(x)·(Az·Bz − Cz) = sum_x full_E(x)·W(x)` (since
  /// `Bz[i] = u = 1` and `Cz = 0`), with `full_E[i*left+j] = E2[i]·E1[j]`
  /// per Corrigendum #9 (2-A).
  ///
  /// Mirrors the Spartan sibling's `build_T_form_satisfying_instance` at
  /// `vendor/nova/src/spartan/snark.rs:1660-1761`, but produces
  /// `FoldedInstance`/`FoldedWitness` directly (so the envelope's full
  /// bridge path is exercised — including
  /// `RelaxedR1CSWitness::derandomize` + `split_E_commitments` + Σ-protocol
  /// prove + Spartan T-claim sibling prove).
  ///
  /// Shape floor `left = right = 2` (`num_cons = 4`) sidesteps the
  /// pre-existing vendor `batch_diff_size` panic at
  /// `spartan/mod.rs:175-189` (`chunk_size = 0` heterogeneous polynomials
  /// case). The Spartan sibling test (`snark.rs:1797-1810`) makes the
  /// same shape-floor choice for the same reason.
  #[allow(non_snake_case)]
  fn build_T_form_folded_triple<E, S>(
    rng: &mut ChaCha20Rng,
    left: usize,
    right: usize,
  ) -> (
    CommitmentKey<E>,
    Structure<E>,
    FoldedInstance<E>,
    FoldedWitness<E>,
  )
  where
    E: Engine,
    E::GE: DlogGroup,
    S: RelaxedR1CSSNARKTrait<E>,
  {
    use crate::r1cs::SparseMatrix;

    let num_cons = left * right;
    let num_vars = num_cons;
    let num_io = 1usize;
    let ell = num_cons.log_2();
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    assert_eq!(1usize << ell1, left, "left must equal 2^ell1");
    assert_eq!(1usize << ell2, right, "right must equal 2^ell2");

    // Matrices: A[i,i] = 1; B[i, num_vars] = 1 (u column); C = 0.
    // With these, for any witness W and u = 1:
    //   (Az · Bz − Cz)[i] = W[i] · 1 − 0 = W[i]
    // so T = sum_x full_E(x) · W(x) — easy to compute honestly.
    let one = E::Scalar::ONE;
    let rows = num_cons;
    let cols = num_vars + num_io + 1;
    let A_entries: Vec<(usize, usize, E::Scalar)> = (0..num_cons).map(|i| (i, i, one)).collect();
    let B_entries: Vec<(usize, usize, E::Scalar)> =
      (0..num_cons).map(|i| (i, num_vars, one)).collect();
    let C_entries: Vec<(usize, usize, E::Scalar)> = vec![];

    let shape = R1CSShape::<E>::new(
      num_cons,
      num_vars,
      num_io,
      SparseMatrix::new(&A_entries, rows, cols),
      SparseMatrix::new(&B_entries, rows, cols),
      SparseMatrix::new(&C_entries, rows, cols),
    )
    .unwrap();

    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();
    let structure = Structure::new(&shape);
    assert_eq!(structure.left, left);
    assert_eq!(structure.right, right);

    // Draw (E1, E2) uniform random.
    let E1: Vec<E::Scalar> = (0..left).map(|_| E::Scalar::random(&mut *rng)).collect();
    let E2: Vec<E::Scalar> = (0..right).map(|_| E::Scalar::random(&mut *rng)).collect();

    // Flat W.E = [E1 || E2] of length left + right per the
    // `FoldedWitness::default` invariant at `relation.rs:611`. The
    // Pedersen-additive identity ties this to U.comm_E.
    let mut e_flat: Vec<E::Scalar> = Vec::with_capacity(left + right);
    e_flat.extend_from_slice(&E1);
    e_flat.extend_from_slice(&E2);

    // Witness W drawn uniform random (unrelated to (E1, E2)). With Az=W,
    // Bz=[1;.], Cz=[0;.]: T = sum_x full_E(x) · W(x) is non-degenerate
    // (generically non-zero), exercising the β′ outer-sumcheck claim path
    // with claim = T (NOT zero) per Corrigendum #10.
    let W_vec: Vec<E::Scalar> =
      (0..num_vars).map(|_| E::Scalar::random(&mut *rng)).collect();
    let X: Vec<E::Scalar> = vec![E::Scalar::ZERO; num_io];

    // Build full_E in flat layout per Corrigendum #9 (2-A):
    //   full_E[i * left + j] = E2[i] · E1[j].
    let mut full_E: Vec<E::Scalar> = Vec::with_capacity(num_cons);
    for i in 0..right {
      for j in 0..left {
        full_E.push(E2[i] * E1[j]);
      }
    }

    // T = sum_x full_E(x) · (Az·Bz − Cz)(x) = sum_x full_E(x) · W(x).
    let T: E::Scalar = full_E
      .iter()
      .zip(W_vec.iter())
      .map(|(e, w)| *e * *w)
      .sum();

    // Commit to W and to e_flat on `ck`, with random blindings. The
    // Pedersen-additive identity `comm_E1 + comm_E2_bind == U.comm_E`
    // is enforced by the M.GH7.0.1 REWORKED helper's blinding-split
    // discipline `r_E1 + r_E2_bind == W.r_E` against `W.r_E = r_E` here.
    let r_W = E::Scalar::random(&mut *rng);
    let r_E = E::Scalar::random(&mut *rng);
    let comm_W = <E::CE as CommitmentEngineTrait<E>>::commit(&ck, &W_vec, &r_W);
    let comm_E = <E::CE as CommitmentEngineTrait<E>>::commit(&ck, &e_flat, &r_E);

    let W = FoldedWitness::<E> {
      W: W_vec,
      r_W,
      E: e_flat,
      r_E,
    };

    let U = FoldedInstance::<E> {
      comm_W,
      comm_E,
      T,
      X,
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

  /// **Acceptance + 1000-iter differential test (M.GH7.0.2)**.
  ///
  /// Composite test exercising:
  ///
  /// 1. **Acceptance round-trip**: build a satisfying neutron-form folded
  ///    triple at `left=right=2`, run
  ///    `CompressedSNARK::prove_from_parts → CompressedSNARK::verify`,
  ///    assert `Ok(zn)`. INDEPENDENTLY assert the off-FS Pedersen-additive
  ///    binding `comm_E1 + comm_E2_bind == r_U_derand_comm_E` holds at the
  ///    group level (NOT just inside verify — pins the binding is
  ///    structurally wired).
  ///
  /// 2. **1000-iter differential** (US-05): same shape, 1000 ChaCha20Rng-
  ///    seeded fresh inputs, each round-trip must verify. The seed
  ///    `0xC0FFEE_0700_0200u64` is stable for reviewer-reproducibility.
  ///    Per `.claude/rules/cryptography.md` 1000-iter behavioral earned-
  ///    trust threshold.
  #[test]
  #[allow(non_snake_case)]
  fn m_gh7_0_2_compressed_snark_envelope_off_fs_pedersen_binding_and_sigma_E2_equality() {
    type E = Bn256EngineKZG;
    type EE = EvaluationEngine<E>;
    type S = RelaxedR1CSSNARK<E, EE>;

    let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEE_0700_0200u64);

    // (1) Acceptance round-trip at left=right=2.
    {
      let (ck, structure, U, W) =
        build_T_form_folded_triple::<E, S>(&mut rng, 2, 2);

      // Spartan keys directly — the envelope `setup` reads `PublicParams`,
      // which we don't construct here. The parts-based test path bypasses
      // `PublicParams` and exercises `prove_from_parts` against the
      // structure directly. Build a VerifierKey shape that matches the
      // envelope's verify path (carries vk_spartan + dk + F_arity +
      // structure + ck).
      let (pk_spartan, vk_spartan) =
        <S as RelaxedR1CSSNARKTrait<E>>::setup(&ck, &structure.S).unwrap();

      let pk = ProverKey::<E, EE> { pk_spartan };
      // M.GH7.1: VerifierKey now carries pin-§3.3 fields. The test
      // bypasses `setup` (no `PublicParams` available), so we populate
      // the §3.3 fields manually:
      //   - `F_arity = 1` matches the in-test fold shape (no IVC arity).
      //   - `ro_consts = RO2Constants::<E>::default()` matches
      //     `PublicParams::setup`'s initialisation precedent
      //     (neutron/mod.rs:227, 286, 352, 411).
      //   - `pp_digest = E::Scalar::ZERO` — the parts-based test path
      //     has no `PublicParams` and thus no IVC-level digest. The
      //     M.GH7.0.2 envelope verify body uses `vk_spartan.digest()`
      //     (not `pp_digest`) for the `b"vk"` absorb at the Σ-protocol
      //     site, so `ZERO` here is benign at the envelope verifier.
      //   - `lookup_fold_k = 0`, `shape_registry_digest = ZERO` — the
      //     test exercises the non-lookup-fold (STAGE 0 trivial) shape.
      let vk = VerifierKey::<E, EE> {
        F_arity: 1,
        ro_consts: <RO2Constants<E> as Default>::default(),
        pp_digest: <E as Engine>::Scalar::ZERO,
        vk_spartan,
        lookup_fold_k: 0,
        shape_registry_digest: <E as Engine>::Scalar::ZERO,
        dk: <<E as Engine>::CE as CommitmentEngineTrait<E>>::derand_key(&ck),
        structure: structure.clone(),
        ck: ck.clone(),
      };

      // Arbitrary IVC final state — only used by the M.GH7.0.2 verify
      // body for the `zn == self.zn` check.
      let zn = vec![<E as Engine>::Scalar::from(42u64)];

      let snark =
        CompressedSNARK::<E, EE>::prove_from_parts(&ck, &structure, &pk, &U, &W, zn.clone())
          .expect("CompressedSNARK::prove_from_parts must succeed for honest folded triple");

      // INDEPENDENT structural assertion (LOAD-BEARING off-FS check per
      // Corrigendum #8 (iv-B)): Pedersen-additive binding holds at the
      // group level.
      assert_eq!(
        snark.U_bridged.comm_E1 + snark.comm_E2_bind,
        snark.r_U_derand_comm_E,
        "Pedersen-additive binding violated at envelope: comm_E1 + comm_E2_bind != r_U_derand_comm_E",
      );

      let z0_unused: Vec<<E as Engine>::Scalar> = vec![<E as Engine>::Scalar::ZERO; 1];
      let returned_zn = snark
        .verify(&vk, 1, &z0_unused, &zn)
        .expect("CompressedSNARK::verify must accept an honest envelope proof");

      assert_eq!(returned_zn, zn, "verify must return self.zn on success");
    }

    // (2) 1000-iter differential. Per `.claude/rules/cryptography.md`,
    // ≥1000 ChaCha20Rng-seeded fresh inputs at the shape floor.
    let mut rng_diff = ChaCha20Rng::seed_from_u64(0xFAC1_0700_0200u64);
    for iter in 0..1000 {
      let (ck, structure, U, W) =
        build_T_form_folded_triple::<E, S>(&mut rng_diff, 2, 2);

      let (pk_spartan, vk_spartan) =
        <S as RelaxedR1CSSNARKTrait<E>>::setup(&ck, &structure.S).unwrap();
      let pk = ProverKey::<E, EE> { pk_spartan };
      // M.GH7.1: see the §3.3 field-population rationale at the first
      // VerifierKey construction site above.
      let vk = VerifierKey::<E, EE> {
        F_arity: 1,
        ro_consts: <RO2Constants<E> as Default>::default(),
        pp_digest: <E as Engine>::Scalar::ZERO,
        vk_spartan,
        lookup_fold_k: 0,
        shape_registry_digest: <E as Engine>::Scalar::ZERO,
        dk: <<E as Engine>::CE as CommitmentEngineTrait<E>>::derand_key(&ck),
        structure: structure.clone(),
        ck: ck.clone(),
      };

      let zn = vec![<E as Engine>::Scalar::from(iter as u64)];
      let snark =
        CompressedSNARK::<E, EE>::prove_from_parts(&ck, &structure, &pk, &U, &W, zn.clone())
          .unwrap_or_else(|e| {
            panic!("prove_from_parts failed at iter={}: {:?}", iter, e)
          });

      // Pedersen-additive binding at the group level (per iter).
      assert_eq!(
        snark.U_bridged.comm_E1 + snark.comm_E2_bind,
        snark.r_U_derand_comm_E,
        "Pedersen-additive binding violated at iter={}",
        iter
      );

      let z0_unused: Vec<<E as Engine>::Scalar> = vec![<E as Engine>::Scalar::ZERO; 1];
      snark
        .verify(&vk, 1, &z0_unused, &zn)
        .unwrap_or_else(|e| panic!("verify failed at iter={}: {:?}", iter, e));
    }
  }

  /// **Negative test (M.GH7.0.2) — composite rejection paths**.
  ///
  /// Two perturbations against an honest proof:
  ///
  /// (b) Off-FS binding: perturb `comm_E2_bind` by `+δ` (fresh group
  ///     element). Assert `verify` rejects at check (1) with reason
  ///     containing "off-FS Pedersen-additive binding rejected". Pins the
  ///     binding check is wired and fires BEFORE the Σ-protocol verify.
  ///
  /// (c) Σ-protocol: perturb `sigma_E2_equality.z[0]` by `+ONE`. Assert
  ///     `verify` rejects at check (2) with a Σ-protocol reason string
  ///     ("Sigma E2 equality-of-opening rejected at bind-side" or
  ///     "...pcs-side"). The response perturbation breaks BOTH equations
  ///     deterministically with overwhelming probability.
  #[test]
  #[allow(non_snake_case)]
  fn m_gh7_0_2_compressed_snark_envelope_rejects_corrupted_binding_and_sigma() {
    type E = Bn256EngineKZG;
    type EE = EvaluationEngine<E>;
    type S = RelaxedR1CSSNARK<E, EE>;

    let mut rng = ChaCha20Rng::seed_from_u64(0xDEADBEEF_0700_0200u64);
    let (ck, structure, U, W) = build_T_form_folded_triple::<E, S>(&mut rng, 2, 2);

    let (pk_spartan, vk_spartan) =
      <S as RelaxedR1CSSNARKTrait<E>>::setup(&ck, &structure.S).unwrap();
    let pk = ProverKey::<E, EE> { pk_spartan };
    // M.GH7.1: see the §3.3 field-population rationale at the first
    // VerifierKey construction site in the in-crate test module.
    let vk = VerifierKey::<E, EE> {
      F_arity: 1,
      ro_consts: <RO2Constants<E> as Default>::default(),
      pp_digest: <E as Engine>::Scalar::ZERO,
      vk_spartan,
      lookup_fold_k: 0,
      shape_registry_digest: <E as Engine>::Scalar::ZERO,
      dk: <<E as Engine>::CE as CommitmentEngineTrait<E>>::derand_key(&ck),
      structure: structure.clone(),
      ck: ck.clone(),
    };
    let zn = vec![<E as Engine>::Scalar::from(7u64)];

    // Sanity: honest proof verifies before perturbation.
    let z0_unused: Vec<<E as Engine>::Scalar> = vec![<E as Engine>::Scalar::ZERO; 1];
    {
      let snark =
        CompressedSNARK::<E, EE>::prove_from_parts(&ck, &structure, &pk, &U, &W, zn.clone())
          .expect("honest prove must succeed");
      snark
        .verify(&vk, 1, &z0_unused, &zn)
        .expect("honest proof must verify before perturbation");
    }

    // (b) NEGATIVE — off-FS binding. Perturb `comm_E2_bind` by adding a
    // fresh non-identity group element. The binding-check at envelope-
    // verify step (1) MUST reject.
    {
      let mut snark =
        CompressedSNARK::<E, EE>::prove_from_parts(&ck, &structure, &pk, &U, &W, zn.clone())
          .expect("honest prove must succeed");

      // Build a fresh non-identity Commitment by committing a non-zero
      // scalar with zero blinding — produces a `MSM([delta], ck.ck[..1])`
      // group element.
      let delta_scalar = <E as Engine>::Scalar::random(&mut rng);
      let delta: Commitment<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::commit(
        &ck,
        &[delta_scalar],
        &<E as Engine>::Scalar::ZERO,
      );
      snark.comm_E2_bind = snark.comm_E2_bind + delta;

      let result = snark.verify(&vk, 1, &z0_unused, &zn);
      match result {
        Err(NovaError::ProofVerifyError { reason }) => {
          assert!(
            reason.contains("off-FS Pedersen-additive binding rejected"),
            "Rejection must be from the off-FS Pedersen-additive binding \
             check (envelope verify step 1); got reason: {}",
            reason,
          );
        }
        Ok(_) => panic!(
          "CompressedSNARK::verify MUST reject when comm_E2_bind is perturbed off-binding"
        ),
        Err(other) => panic!(
          "Unexpected error variant on binding-check rejection: {:?}",
          other
        ),
      }
    }

    // (c) NEGATIVE — Σ-protocol. Perturb `sigma_E2_equality.z[0]` by
    // adding `ONE` to it. The verifier squeezes the SAME α from the
    // transcript (identical absorb stream as the honest case — only the
    // response field changes), so the bind-side acceptance equation
    //   MSM(z, ck.ck[left..left+right]) + h·z_r_bind == T_bind +
    //   α·comm_E2_bind
    // diverges by exactly `MSM([ONE, 0, ...], ck.ck[left..left+right]) =
    // ck.ck[left]` on the LHS — a non-zero non-identity group element —
    // breaking the equation. Asserts the verifier rejects with a
    // Σ-protocol-side reason string.
    {
      let mut snark =
        CompressedSNARK::<E, EE>::prove_from_parts(&ck, &structure, &pk, &U, &W, zn.clone())
          .expect("honest prove must succeed");
      snark.sigma_E2_equality.z[0] = snark.sigma_E2_equality.z[0] + <E as Engine>::Scalar::ONE;

      let result = snark.verify(&vk, 1, &z0_unused, &zn);
      match result {
        Err(NovaError::ProofVerifyError { reason }) => {
          assert!(
            reason.contains("Sigma E2 equality-of-opening rejected"),
            "Rejection must be from the Σ-protocol verify (envelope verify \
             step 2); got reason: {}",
            reason,
          );
        }
        Ok(_) => panic!(
          "CompressedSNARK::verify MUST reject when sigma_E2_equality.z[0] is perturbed"
        ),
        Err(other) => panic!(
          "Unexpected error variant on Σ-protocol rejection: {:?}",
          other
        ),
      }
    }
  }

  /// **M.GH7.1 type-signature polish typecheck** — confirms the
  /// `CompressedSNARK::<E, EE>::setup(&pp)` surface compiles at BOTH
  /// production PCS choices: HyperKZG (`Bn256EngineKZG` +
  /// `provider::hyperkzg::EvaluationEngine`) AND IPA-PC (`Bn256EngineIPA`
  /// + `provider::ipa_pc::EvaluationEngine`).
  ///
  /// # Scope per roadmap step 01-08 acceptance criteria
  ///
  /// This is a STRUCTURAL typecheck — the body only invokes
  /// `PublicParams::setup` + `CompressedSNARK::setup` against the
  /// trivial-circuit / non-lookup-fold (`(vec![], 0, 0)`) shape, then
  /// drops the resulting `(pk, vk)` and asserts trivial structural
  /// invariants (`vk.F_arity == 1`, `vk.lookup_fold_k == 0`,
  /// `vk.shape_registry_digest == ZERO`). End-to-end prove/verify is
  /// already exercised at:
  ///
  ///   - M.GH7.0.3 + M.GH7.0.4: `gh7_stage_k_compressor::stage_k_*`
  ///     in `crates/inumbra-spend-harness/tests/gh7_stage_k_compressor.rs`
  ///     (HyperKZG + IPA-PC happy-path at the production wrap pipeline).
  ///   - M.GH7.0.2: `m_gh7_0_2_compressed_snark_envelope_*` above
  ///     (parts-based prove/verify at HyperKZG).
  ///
  /// The purpose of this test is the type-signature gate ONLY: the new
  /// pin-§3.3 verifier-key fields (`ro_consts`, `pp_digest`,
  /// `lookup_fold_k`, `shape_registry_digest`) must populate correctly
  /// through `setup` at both EE choices without compile errors.
  ///
  /// # Why both EE choices in ONE test
  ///
  /// Per `nw-software-crafter` Mandate 5 (parametrise input variations),
  /// the HyperKZG and IPA-PC paths are SAME-behavior input variations of
  /// the same `setup` shape. They share the same field, the same
  /// `PublicParams::setup` signature, and the same VK population logic
  /// — only the `EE::EvaluationArgument` associated type differs. One
  /// test with two inline sections is more honest than two near-identical
  /// tests at separate names.
  #[test]
  #[allow(non_snake_case)]
  fn m_gh7_1_compressed_snark_prover_key_verifier_key_typecheck_both_ees() {
    use crate::{
      neutron::PublicParams,
      provider::{ipa_pc, Bn256EngineIPA, Bn256EngineKZG, GrumpkinEngine},
      traits::{circuit::TrivialCircuit, snark::default_ck_hint},
    };

    // === Section A: HyperKZG path ===
    {
      type E1 = Bn256EngineKZG;
      type E2 = GrumpkinEngine;
      type EE = EvaluationEngine<E1>; // hyperkzg via the in-test type alias
      type C = TrivialCircuit<<E1 as Engine>::Scalar>;

      let circuit = C::default();
      // Trivial-circuit / non-lookup-fold `(vec![], 0, 0)` shape per
      // Corrigendum #13 STAGE-0 disambiguation. Mirrors the vendor-
      // internal precedent at `neutron/mod.rs:852-861`.
      #[cfg(feature = "lookup-fold")]
      let pp = PublicParams::<E1, E2, C>::setup(
        &circuit,
        &*default_ck_hint(),
        &*default_ck_hint(),
        vec![],
        0,
        0,
        None,
      )
      .expect("PublicParams::setup must succeed at HyperKZG trivial-circuit shape");
      #[cfg(not(feature = "lookup-fold"))]
      let pp = PublicParams::<E1, E2, C>::setup(
        &circuit,
        &*default_ck_hint(),
        &*default_ck_hint(),
      )
      .expect("PublicParams::setup must succeed at HyperKZG trivial-circuit shape");

      let (_pk, vk) = CompressedSNARK::<E1, EE>::setup(&pp)
        .expect("CompressedSNARK::setup must succeed at HyperKZG");

      // Structural assertions on the pin-§3.3 fields populated by
      // `setup`. These are typecheck-load-bearing: they fail to compile
      // if any field is missing from `VerifierKey`, fail to typecheck if
      // a field has the wrong type, and fail at runtime if the
      // population logic in `setup` deviates from the pin's intent at
      // the trivial-circuit / non-lookup-fold shape.
      assert_eq!(vk.F_arity, 1, "trivial circuit F_arity is 1");
      assert_eq!(
        vk.lookup_fold_k, 0,
        "non-lookup-fold path: lookup_fold_k is 0"
      );
      assert_eq!(
        vk.shape_registry_digest,
        <E1 as Engine>::Scalar::ZERO,
        "STAGE 0 / M.GH7.1 polish: shape_registry_digest is ZERO at empty registry"
      );
      // `pp_digest` is initialised by `PublicParams::digest()`'s
      // `OnceCell`; reading it through `vk.pp_digest` confirms the
      // field exists and the type is `E::Scalar` (typecheck only — the
      // actual digest value is implementation-defined).
      let _: <E1 as Engine>::Scalar = vk.pp_digest;
    }

    // === Section B: IPA-PC path ===
    {
      type E1 = Bn256EngineIPA;
      type E2 = GrumpkinEngine;
      type EE = ipa_pc::EvaluationEngine<E1>;
      type C = TrivialCircuit<<E1 as Engine>::Scalar>;

      let circuit = C::default();
      #[cfg(feature = "lookup-fold")]
      let pp = PublicParams::<E1, E2, C>::setup(
        &circuit,
        &*default_ck_hint(),
        &*default_ck_hint(),
        vec![],
        0,
        0,
        None,
      )
      .expect("PublicParams::setup must succeed at IPA-PC trivial-circuit shape");
      #[cfg(not(feature = "lookup-fold"))]
      let pp = PublicParams::<E1, E2, C>::setup(
        &circuit,
        &*default_ck_hint(),
        &*default_ck_hint(),
      )
      .expect("PublicParams::setup must succeed at IPA-PC trivial-circuit shape");

      let (_pk, vk) = CompressedSNARK::<E1, EE>::setup(&pp)
        .expect("CompressedSNARK::setup must succeed at IPA-PC");

      assert_eq!(vk.F_arity, 1, "trivial circuit F_arity is 1");
      assert_eq!(
        vk.lookup_fold_k, 0,
        "non-lookup-fold path: lookup_fold_k is 0"
      );
      assert_eq!(
        vk.shape_registry_digest,
        <E1 as Engine>::Scalar::ZERO,
        "STAGE 0 / M.GH7.1 polish: shape_registry_digest is ZERO at empty registry"
      );
      let _: <E1 as Engine>::Scalar = vk.pp_digest;
    }
  }
}
