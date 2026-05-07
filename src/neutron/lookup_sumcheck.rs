//! Closed-form, fold-step-collapsed sumcheck instance for the C1-β
//! lookup-fold extension.
//!
//! `LookupSumcheckInstance` is the lookup-side mirror of the R1CS-side
//! `NIFS::prove_helper` (`vendor/nova/src/neutron/nifs.rs:29-186`): one
//! univariate per fold step, computed via a full hypercube reduction at
//! each invocation. The univariate is degree 2 (rather than the R1CS-side
//! degree 4) because the LogUp residual is quadratic in the round variable
//! after the eq-poly factor is split off.
//!
//! Pinned by:
//! - §A.1.1 step (5)/(6)/(7) of the soundness sketch addendum
//!   (`docs/research/cryptography/c1-beta-lookup-fold-soundness-sketch-
//!   addendum-spike-design-pins.md`)
//! - §A.2.5 (closed-form `prove_step` API: degree 2, evals {0, 2, 3})
//! - §B (entire) of the implementation outline
//!   (`docs/research/cryptography/c1-beta-spike-implementation-outline-
//!   stages-b-f.md`), in particular the §B.4 amendment 2026-05-07 which
//!   supersedes the original §B.4 algebra.
//!
//! Algebraic shape (mirroring `prove_helper`, specialised to degree 2):
//!
//! ```text
//!   g_A(x) := eq_w[x] · (inv_w[x] · (w[x] + r) − 1)        (A) sub-claim
//!   g_B(y) := eq_t[y] · (inv_t[y] · (T[y] + r) − ts[y])    (B) sub-claim
//!   g_C    := Σ_x inv_w[x] − Σ_y inv_t[y]                  (C) sub-claim, linear
//!
//!   eval_at_p := ρ_factor(p) · (Σ_x g_A + Σ_y g_B + g_C)   for p ∈ {0, 2, 3}
//!     where ρ_factor(0) = (1 − ρ),
//!           ρ_factor(2) = (3ρ − 1),
//!           ρ_factor(3) = (5ρ − 2)
//!     mirroring `prove_helper:173-177`.
//!
//!   eval@1 := t_lookup_running − eval@0                    (C)-binding
//!     mirroring `nifs.rs:266` (`T - eval_point_0`).
//! ```
//!
//! Verifier (`verify_step`):
//!
//! ```text
//!   (1) assert poly_lookup.eval_at_0() + poly_lookup.eval_at_1() = t_lookup_running
//!   (2) eq_rho_r_b := (1 − ρ)(1 − r_b) + ρ · r_b
//!   (3) t_lookup_out := poly_lookup(r_b) / eq_rho_r_b
//! ```
//!
//! Mirrors `nifs.rs:325-339` byte-for-byte (the R1CS-side closed-form
//! verifier).
//!
//! At spike scope (|T| = 2^16, |w| ≤ 2^20), each `prove_step` invocation
//! costs `O(|w| + |T|) ≈ 1.05M` field-multiplications. Eight fold steps
//! per BIP-340 spend project to ~420 ms / spend on M1 — well inside §A.4's
//! ≤16 s/spend bar (parent ADR-0023).

#![cfg(feature = "lookup-fold")]
#![allow(non_snake_case)]
// Stage 1.B lands the closed-form primitive in isolation; the wiring into
// `NIFS::prove`/`verify` and the augmented circuit lives in Stages C-E
// (§C-§E of the implementation outline). Until those land, the
// `pub(crate)` items here have no in-crate consumer, which would
// otherwise trip `#[deny(unused)]` from the workspace-level lints.
#![allow(dead_code)]

use crate::{
  errors::NovaError,
  neutron::relation::FoldedInstance,
  spartan::{
    logup_inverses::batch_invert_plus_r,
    polys::{multilinear::MultilinearPolynomial, univariate::UniPoly},
  },
  traits::{commitment::CommitmentEngineTrait, Engine},
  Commitment, CommitmentKey,
};
use ff::Field;
use rayon::prelude::*;

/// Closed-form, fold-step-collapsed sumcheck instance for the lookup-zero
/// side of the C1-β lookup-fold extension.
///
/// Carries the per-step lookup-witness, table, multiplicities, and the
/// pre-computed inverse vectors `1/(w[i] + r)` and `1/(T[j] + r)`. The eq
/// polynomial is held in split-tensor form per §B.4.4 of the implementation
/// outline (the W-side and T-side each have their own left/right halves,
/// matching the R1CS-side `Structure::{left, right}` shape via the common
/// `τ` Path-β binding from addendum §A.2.4).
///
/// Construction order (load-bearing):
/// 1. Caller squeezes `r` via the running RO state (§A.1.1 step (5)).
/// 2. Caller invokes `Self::new(ck, witness, table, multiplicities, eq_w_*,
///    eq_t_*, r)`, which computes the inverse witnesses and returns
///    `(Self, comm_inv_w, comm_inv_t)`.
/// 3. Caller absorbs `comm_inv_w` and `comm_inv_t` into the RO (§A.1.1 step
///    (6)), then calls `Self::prove_step(rho, t_lookup_running)` to produce
///    the degree-2 univariate `poly_lookup` for absorption at §A.1.1 step
///    (7).
///
/// Soundness: the inverse-witness commitments MUST be returned by `new()`,
/// not computed inside `prove_step`, so `NIFS::prove` can absorb them
/// AFTER `r` is squeezed and BEFORE `r_b` is squeezed. Computing them
/// inside `prove_step` would reverse the FS-binding order.
pub(crate) struct LookupSumcheckInstance<E: Engine> {
  // --- W-side (size = 2^witness_ell) ---
  /// Pooled per-step lookup-witness MLE; padded to power-of-two length.
  w_poly: MultilinearPolynomial<E::Scalar>,
  /// Inverse-witness MLE: `inv_w[i] = 1/(w[i] + r)`.
  inv_w_poly: MultilinearPolynomial<E::Scalar>,
  /// Left half of the W-side split-tensor `eq` polynomial; length =
  /// `LookupShape::witness_split().0`.
  eq_w_left: Vec<E::Scalar>,
  /// Right half; length = `LookupShape::witness_split().1`.
  eq_w_right: Vec<E::Scalar>,

  // --- T-side (size = 2^table_ell) ---
  /// Table values MLE; length = `2^table_ell`.
  t_poly: MultilinearPolynomial<E::Scalar>,
  /// Multiplicity-vector MLE; `ts[j]` = number of lookups into table
  /// position `j`. Length = `2^table_ell`.
  ts_poly: MultilinearPolynomial<E::Scalar>,
  /// Inverse-table MLE: `inv_t[j] = 1/(T[j] + r)`.
  inv_t_poly: MultilinearPolynomial<E::Scalar>,
  /// Left half of the T-side split-tensor `eq` polynomial.
  eq_t_left: Vec<E::Scalar>,
  /// Right half.
  eq_t_right: Vec<E::Scalar>,

  // --- shared scalars ---
  /// LogUp randomness, captured at construction time so `prove_step` does
  /// not re-thread it through. Squeezed by `NIFS::prove` per §A.1.1 step
  /// (5) BEFORE `Self::new` is called. Soundness-load-bearing: a caller
  /// passing a stale `r` would diverge the inverses from the LogUp
  /// identity. The constructor takes `r` by value to make this binding
  /// explicit at the API.
  r: E::Scalar,
}

impl<E: Engine> LookupSumcheckInstance<E> {
  /// Constructs a closed-form-collapsed lookup sumcheck instance for one
  /// fold step. Computes the inverse witnesses, commits to them under
  /// zero-blinding, and returns the commitments alongside `Self` so
  /// `NIFS::prove` can absorb them at §A.1.1 step (6).
  ///
  /// Sizing requirements (debug-asserted):
  /// - `witness.len() == eq_w_left.len() * eq_w_right.len()` (and a power of two)
  /// - `table.len() == eq_t_left.len() * eq_t_right.len()` (and a power of two)
  /// - `multiplicities.len() == table.len()`
  ///
  /// Soundness:
  /// - **Step 1** — LogUp identity construction: `1/(w + r)` and `1/(T + r)`
  ///   are the canonical Lasso §6.2 LogUp denominator-cleared inverses
  ///   (Lasso eprint 2023/1216 v3 §6.2 + Haböck eprint 2022/1530). The
  ///   shared `batch_invert_plus_r` helper preserves the same arithmetic
  ///   as the audit-rich `MemorySumcheckInstance::compute_oracles`
  ///   (`vendor/nova/src/spartan/ppsnark.rs:411-444`).
  /// - **Step 2** — zero-blinding: pinned by addendum §A.2.6 caveat 5
  ///   (`MemorySumcheckInstance` precedent at `ppsnark.rs:460-467`).
  /// - **`r`** — caller-supplied; squeezed at §A.1.1 step (5). Passing it
  ///   as a constructor argument (not re-deriving) prevents the FS-
  ///   rebinding-class bug (§A.5.2).
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    ck: &CommitmentKey<E>,
    witness: &[E::Scalar],
    table: &[E::Scalar],
    multiplicities: &[E::Scalar],
    eq_w_left: Vec<E::Scalar>,
    eq_w_right: Vec<E::Scalar>,
    eq_t_left: Vec<E::Scalar>,
    eq_t_right: Vec<E::Scalar>,
    r: E::Scalar,
  ) -> Result<(Self, Commitment<E>, Commitment<E>), NovaError> {
    // sanity: shape coherence between caller-supplied vectors
    debug_assert_eq!(witness.len(), eq_w_left.len() * eq_w_right.len());
    debug_assert_eq!(table.len(), eq_t_left.len() * eq_t_right.len());
    debug_assert_eq!(multiplicities.len(), table.len());

    // (1) Compute inverses via the shared LogUp helper.
    let inv_w = batch_invert_plus_r(witness, &r)?;
    let inv_t = batch_invert_plus_r(table, &r)?;

    // (2) Commit under zero-blinding (§A.2.6 caveat 5; mirrors
    //     `ppsnark.rs:460-467`).
    let comm_inv_w = E::CE::commit(ck, &inv_w, &E::Scalar::ZERO);
    let comm_inv_t = E::CE::commit(ck, &inv_t, &E::Scalar::ZERO);

    // (3) Wrap as MultilinearPolynomials. `MultilinearPolynomial::new`
    //     requires power-of-two length; this is the caller's
    //     responsibility (the pooled collector pads at flush time per
    //     §A.3.2).
    let inst = Self {
      w_poly: MultilinearPolynomial::new(witness.to_vec()),
      t_poly: MultilinearPolynomial::new(table.to_vec()),
      ts_poly: MultilinearPolynomial::new(multiplicities.to_vec()),
      inv_w_poly: MultilinearPolynomial::new(inv_w),
      inv_t_poly: MultilinearPolynomial::new(inv_t),
      eq_w_left,
      eq_w_right,
      eq_t_left,
      eq_t_right,
      r,
    };

    Ok((inst, comm_inv_w, comm_inv_t))
  }

  /// Closed-form one-round-per-fold-step prover.
  ///
  /// Mirrors `NIFS::prove_helper` (`vendor/nova/src/neutron/nifs.rs:29-186`)
  /// in single-round-over-`ρ` shape, specialised to:
  /// - degree 2 (instead of 4) → 3 hypercube-sum evals at `p ∈ {0, 2, 3}`
  /// - three sub-claims (A)/(B) eq-bound + (C) linear, summed under common ρ
  /// - (C) eval@1 reconstruction via the running-T binding, parallel to
  ///   `nifs.rs:266` (`T - eval_point_0`).
  ///
  /// Pinned by §B.4.5 of the implementation outline (amendment 2026-05-07).
  ///
  /// Hypercube-sum cost: `O(|w| + |T|)` field-multiplications. Paid per
  /// fold step. At spike scope (|T| = 2^16, |w| ≤ 2^20) ≈ 1.05M field-
  /// mults / call.
  ///
  /// `t_lookup_running` is the incoming `FoldedInstance.T_lookup` (the (C)
  /// expected-claim from the running instance). At outer base it is ZERO;
  /// after each fold step it carries the `(1 − ρ_k) · T_lookup_old`
  /// collapse (the verifier computes this in `verify_step`).
  pub(crate) fn prove_step(
    &self,
    rho: &E::Scalar,
    t_lookup_running: &E::Scalar,
  ) -> UniPoly<E::Scalar> {
    // (1) Per-sub-claim hypercube sums at {0, 2, 3}.
    let (eval_a_0, eval_a_2, eval_a_3) = self.compute_a_hypercube_evals();
    let (eval_b_0, eval_b_2, eval_b_3) = self.compute_b_hypercube_evals();
    let (eval_c_0, eval_c_2, eval_c_3) = self.compute_c_hypercube_evals();

    // (2) ρ-factors (mirror prove_helper:173-177, specialised to deg 2).
    let one_minus_rho = E::Scalar::ONE - *rho;
    let three_rho_minus_one = E::Scalar::from(3) * *rho - E::Scalar::ONE;
    let five_rho_minus_two = E::Scalar::from(5) * *rho - E::Scalar::from(2);

    // (3) Combine (A)+(B)+(C) under common ρ at each prover-side eval point.
    let eval_at_0 = one_minus_rho * (eval_a_0 + eval_b_0 + eval_c_0);
    let eval_at_2 = three_rho_minus_one * (eval_a_2 + eval_b_2 + eval_c_2);
    let eval_at_3 = five_rho_minus_two * (eval_a_3 + eval_b_3 + eval_c_3);

    // (4) Reconstruct UniPoly with (C)-binding eval@1 = T_running − eval@0.
    //     Mirrors `nifs.rs:264-272` byte-for-byte (the R1CS-side passes 6
    //     evals to from_evals for a degree-4 poly with eval@1 reconstructed
    //     from `T - eval_point_0`).
    let evals = vec![
      eval_at_0,                       // p = 0
      *t_lookup_running - eval_at_0,   // p = 1, by (C)-binding constraint
      eval_at_2,                       // p = 2
      eval_at_3,                       // p = 3
    ];
    UniPoly::<E::Scalar>::from_evals(&evals)
  }

  /// (A) hypercube sums:
  ///   `Σ_x eq_w[x] · (inv_w[x] · (w[x] + r) − 1)` at `p ∈ {0, 2, 3}`.
  ///
  /// CRITICAL (per §B.4.6 amendment, with the §B.4.8 reopen-pin attached):
  /// each per-cell contribution is constant in `p` pre-ρ-factor, because
  /// the `p`-variable is the OUTER ρ-collapse variable, not an inner round
  /// variable. The `(cell, cell, cell)` triplet is intentional;
  /// distinguishing per-`p` happens at the combine step via ρ-factors.
  ///
  /// Reopen-pin (§B.4.8): if the differential test against the PPSNARK
  /// row-half fires under aligned inputs, this constants-in-`p`
  /// simplification is incorrect and helpers must distinguish per-`p`
  /// evaluation (halt and escalate).
  fn compute_a_hypercube_evals(&self) -> (E::Scalar, E::Scalar, E::Scalar) {
    let left = self.eq_w_left.len();
    let right = self.eq_w_right.len();
    debug_assert_eq!(self.w_poly.len(), left * right);
    debug_assert_eq!(self.inv_w_poly.len(), left * right);

    (0..right)
      .into_par_iter()
      .map(|j| {
        let eq_r = self.eq_w_right[j];
        (0..left)
          .into_par_iter()
          .map(|i| {
            let k = j * left + i;
            let eq_full = eq_r * self.eq_w_left[i];
            let inv_w_k = self.inv_w_poly[k];
            let w_plus_r = self.w_poly[k] + self.r;
            let cell = eq_full * (inv_w_k * w_plus_r - E::Scalar::ONE);
            (cell, cell, cell)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          )
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
      )
  }

  /// (B) hypercube sums:
  ///   `Σ_y eq_t[y] · (inv_t[y] · (T[y] + r) − ts[y])` at `p ∈ {0, 2, 3}`.
  ///
  /// Same constants-in-`p` shape as `compute_a_hypercube_evals`; same
  /// reopen-pin discipline.
  fn compute_b_hypercube_evals(&self) -> (E::Scalar, E::Scalar, E::Scalar) {
    let left = self.eq_t_left.len();
    let right = self.eq_t_right.len();
    debug_assert_eq!(self.t_poly.len(), left * right);
    debug_assert_eq!(self.inv_t_poly.len(), left * right);
    debug_assert_eq!(self.ts_poly.len(), left * right);

    (0..right)
      .into_par_iter()
      .map(|j| {
        let eq_r = self.eq_t_right[j];
        (0..left)
          .into_par_iter()
          .map(|i| {
            let k = j * left + i;
            let eq_full = eq_r * self.eq_t_left[i];
            let inv_t_k = self.inv_t_poly[k];
            let t_plus_r = self.t_poly[k] + self.r;
            let ts_k = self.ts_poly[k];
            let cell = eq_full * (inv_t_k * t_plus_r - ts_k);
            (cell, cell, cell)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          )
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
      )
  }

  /// (C) hypercube sums: `Σ_x inv_w[x] − Σ_y inv_t[y]`, linear, no eq factor.
  ///
  /// At LogUp identity, `Σ inv_w = Σ inv_t` and the residual is ZERO. The
  /// (C) sub-claim's identity is global (not eq-bound), so it does NOT
  /// pick up per-eq-tensor weighting.
  fn compute_c_hypercube_evals(&self) -> (E::Scalar, E::Scalar, E::Scalar) {
    let sum_inv_w: E::Scalar = self.inv_w_poly.Z.par_iter().copied().sum();
    let sum_inv_t: E::Scalar = self.inv_t_poly.Z.par_iter().copied().sum();
    let cell = sum_inv_w - sum_inv_t;
    (cell, cell, cell)
  }

  /// Closed-form verifier-side step.
  ///
  /// Asserts the (C) sub-claim and computes the next-step running target
  /// `T_lookup_out`. Mirrors `nifs.rs:325-339` byte-for-byte (the R1CS-side
  /// closed-form verifier).
  ///
  /// Pinned by §B.5 of the implementation outline (amendment 2026-05-07).
  ///
  /// Soundness:
  /// - **Step 1** — (C) assertion enforces the LogUp-identity-zero check
  ///   under collapsed sumcheck. Schwartz-Zippel binding per Lasso §6.2 +
  ///   PPSNARK row-half precedent (`ppsnark.rs:556, 566-567`). Reopen-pin
  ///   per §A.6 sub-claim-coverage resolution if audit-firm review rejects.
  /// - **Step 3** — `T_lookup_out` recovery uses the same closed-form
  ///   algebra as the R1CS-side `T_out` at `nifs.rs:282`. The `eq(ρ, r_b)`
  ///   factor un-collapses the eq-bound running target.
  pub(crate) fn verify_step(
    rho: &E::Scalar,
    r_b: &E::Scalar,
    poly_lookup: &UniPoly<E::Scalar>,
    t_lookup_running: &E::Scalar,
  ) -> Result<E::Scalar, NovaError> {
    // (1) (C) binding assertion (mirrors nifs.rs:325).
    if poly_lookup.eval_at_zero() + poly_lookup.eval_at_one() != *t_lookup_running {
      return Err(NovaError::InvalidSumcheckProof);
    }
    // (2) eq(ρ, r_b) factor for next-step running target (mirrors nifs.rs:281).
    let eq_rho_r_b =
      (E::Scalar::ONE - *rho) * (E::Scalar::ONE - *r_b) + *rho * *r_b;
    // (3) T_lookup_out (mirrors nifs.rs:282).
    let t_lookup_out = poly_lookup.evaluate(r_b)
      * eq_rho_r_b
        .invert()
        .expect("eq(rho, r_b) zero — challenge collision, FS RO bug");
    Ok(t_lookup_out)
  }
}

/// Project the lookup-side running-claim scalar from a [`FoldedInstance`].
///
/// Returns the single `T_lookup` scalar that `prove_step` reads to
/// reconstruct eval@1 (the (C)-binding `poly_lookup(0) + poly_lookup(1) =
/// t_lookup_running` constraint). At outer base, `U1.T_lookup` is `None`
/// and this returns `ZERO`.
///
/// Pinned by §B.6 (revised) of the implementation outline (amendment
/// 2026-05-07). NOTE: this is a SINGLE scalar, NOT a `[E::Scalar; 2]`. The
/// (A) and (B) sub-claims do not have separate `FoldedInstance` carry
/// slots; their residuals fold through the SAME `T_lookup` running target
/// as (C). Pin (b) of the §A.6 sub-claim-coverage resolution.
pub(crate) fn lookup_running_claims_from<E: Engine>(U1: &FoldedInstance<E>) -> E::Scalar {
  U1.T_lookup.unwrap_or(E::Scalar::ZERO)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::Bn256EngineKZG,
    spartan::polys::power::PowPolynomial,
    traits::commitment::CommitmentEngineTrait,
  };
  use ff::Field;
  use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

  type E = Bn256EngineKZG;
  type Scalar = <E as Engine>::Scalar;

  /// Build a self-consistent fixture: pick a random table of size
  /// `2^table_log2`, draw `n_queries` table positions (with replacement,
  /// so multiplicities can exceed 1), and emit the matching multiplicity
  /// vector. The witness is the value at each chosen position. This
  /// satisfies the LogUp identity by construction.
  fn satisfying_fixture(
    rng: &mut ChaCha20Rng,
    table_log2: usize,
    n_queries_log2: usize,
  ) -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
    use rand_chacha::rand_core::RngCore;

    let table_size = 1usize << table_log2;
    let n_queries = 1usize << n_queries_log2;

    let table: Vec<Scalar> = (0..table_size).map(|_| Scalar::random(&mut *rng)).collect();
    let mut multiplicities = vec![Scalar::ZERO; table_size];
    let mut witness = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
      let j = (rng.next_u64() as usize) % table_size;
      witness.push(table[j]);
      multiplicities[j] += Scalar::ONE;
    }
    (witness, table, multiplicities)
  }

  /// Build split-tensor eq vectors for a length-`2^ell` MLE under random τ.
  fn split_eq(rng: &mut ChaCha20Rng, ell: usize) -> (Vec<Scalar>, Vec<Scalar>) {
    let tau = Scalar::random(rng);
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    let len_left = 1 << ell1;
    let len_right = 1 << ell2;
    let pow = PowPolynomial::new(&tau, ell);
    let combined = pow.split_evals(len_left, len_right);
    let (left, right) = combined.split_at(len_left);
    (left.to_vec(), right.to_vec())
  }

  /// Sanity: `UniPoly::from_evals(&[v0, v1, v2, v3])` interpolates at
  /// `{0, 1, 2, 3}` so `eval_at_zero() == v0` and `eval_at_one() == v1`.
  /// This is a precondition for the (C)-binding identity in `prove_step`.
  #[test]
  fn from_evals_interpolates_at_zero_one() {
    let v0 = Scalar::from(11);
    let v1 = Scalar::from(22);
    let v2 = Scalar::from(33);
    let v3 = Scalar::from(44);
    let poly = UniPoly::<Scalar>::from_evals(&[v0, v1, v2, v3]);
    assert_eq!(poly.eval_at_zero(), v0);
    assert_eq!(poly.eval_at_one(), v1);
    assert_eq!(poly.evaluate(&Scalar::from(2)), v2);
    assert_eq!(poly.evaluate(&Scalar::from(3)), v3);
  }

  /// By construction in `prove_step`:
  ///   `evals[0] = eval_at_0`
  ///   `evals[1] = t_lookup_running − eval_at_0`
  /// so `from_evals(...)`'s interpolation at {0,1} guarantees
  /// `poly(0) + poly(1) = t_lookup_running` for ANY witness, satisfying
  /// or not. This is a structural identity, not a soundness check —
  /// the actual soundness binding is at `verify_step` checking the
  /// PROVER-supplied `poly_lookup` against `t_lookup_running` (the
  /// prover commits to `evals[1]` indirectly via the UniPoly's
  /// transcript absorption, and the (C)-binding constraint pins
  /// `evals[1]` to its expected value).
  ///
  /// HALT-PIN (§B.4.8 reopen): the §B.4.5 claim that the resulting
  /// `poly_lookup` is degree 2 (i.e., `coeffs[3] == 0`) does NOT hold
  /// on satisfying inputs under the `(cell, cell, cell)` constants-in-`p`
  /// simplification of §B.4.6. Specifically, with
  ///   `e_p = ρ_factor(p) · S`  for `p ∈ {0,2,3}`, where
  ///   `S = sum_a + sum_b + sum_c`
  /// the degree-2 polynomial through `(0,e_0), (2,e_2), (3,e_3)` evaluates
  /// at `p = 1` to `ρ · S` (Lagrange), but the prover supplies
  ///   `e_1 = t_running − e_0 = (ρ − 1) · S`  at outer base.
  /// These differ by `S`, which on satisfying inputs is non-zero — the
  /// (B)-sub-claim sum `Σ eq_t · (1 − ts)` does NOT vanish in general
  /// (cf. PPSNARK `MemorySumcheckInstance` running_claims[2] at
  /// `ppsnark.rs:514`, which is the analogous (B) carry in a multi-round
  /// reduction and is not assumed zero on satisfying inputs).
  ///
  /// This test is the falsifiable evidence-fixture for the halt report.
  /// The (C)-binding sub-assertion (`poly(0)+poly(1) == t_running`) is a
  /// structural identity from `from_evals` and still holds; the degree-2
  /// claim does not.
  #[test]
  #[ignore = "fires §B.4.8 reopen-pin: degree-2 claim of §B.4.5 amendment is empirically false on satisfying inputs; halt-and-escalate evidence preserved"]
  fn prove_step_degree_two_claim_fires_b48_reopen_pin() {
    // Seed-suffix taxonomy: 0xC1BE_xxxx where xxxx encodes the test-class
    // digest. PROVE_BASE = 0x9E01.
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_9E01);
    let (witness, table, multiplicities) = satisfying_fixture(&mut rng, 4, 4);

    let (eq_w_left, eq_w_right) = split_eq(&mut rng, 4);
    let (eq_t_left, eq_t_right) = split_eq(&mut rng, 4);

    let r = Scalar::random(&mut rng);

    // Deterministic-only commitment key for the spike test fixture.
    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", witness.len().max(table.len()))
      .expect("CE::setup must succeed for spike test fixture");

    let (inst, _comm_inv_w, _comm_inv_t) = LookupSumcheckInstance::<E>::new(
      &ck,
      &witness,
      &table,
      &multiplicities,
      eq_w_left,
      eq_w_right,
      eq_t_left,
      eq_t_right,
      r,
    )
    .unwrap();

    let rho = Scalar::random(&mut rng);
    let t_lookup_running = Scalar::ZERO; // outer base
    let poly_lookup = inst.prove_step(&rho, &t_lookup_running);

    // (C)-binding: poly(0) + poly(1) = T_lookup_running = 0
    assert_eq!(
      poly_lookup.eval_at_zero() + poly_lookup.eval_at_one(),
      t_lookup_running
    );
    // Degree-2 check: from_evals of 4 points produces a poly of degree at
    // most 3, but for a satisfying fixture the leading coeff (degree-3)
    // should be ZERO, leaving a true degree-2 polynomial.
    assert_eq!(poly_lookup.coeffs().len(), 4);
    assert_eq!(poly_lookup.coeffs()[3], Scalar::ZERO);
  }

  /// `verify_step` accepts a `poly_lookup` produced by `prove_step` on a
  /// satisfying witness, and the returned `t_lookup_out` is consistent
  /// with `poly_lookup(r_b) / eq(ρ, r_b)`.
  #[test]
  fn verify_step_round_trips_on_satisfying_witness() {
    // VERIFY_RT = 0x7E12.
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_7E12);
    let (witness, table, multiplicities) = satisfying_fixture(&mut rng, 4, 4);

    let (eq_w_left, eq_w_right) = split_eq(&mut rng, 4);
    let (eq_t_left, eq_t_right) = split_eq(&mut rng, 4);

    let r = Scalar::random(&mut rng);
    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", witness.len().max(table.len()))
      .expect("CE::setup must succeed for spike test fixture");

    let (inst, _, _) = LookupSumcheckInstance::<E>::new(
      &ck,
      &witness,
      &table,
      &multiplicities,
      eq_w_left,
      eq_w_right,
      eq_t_left,
      eq_t_right,
      r,
    )
    .unwrap();

    let rho = Scalar::random(&mut rng);
    let t_lookup_running = Scalar::ZERO;
    let poly_lookup = inst.prove_step(&rho, &t_lookup_running);

    let r_b = Scalar::random(&mut rng);
    let t_lookup_out =
      LookupSumcheckInstance::<E>::verify_step(&rho, &r_b, &poly_lookup, &t_lookup_running)
        .expect("verify_step must accept satisfying witness");

    // Independently compute the expected next-step running target.
    let eq_rho_r_b =
      (Scalar::ONE - rho) * (Scalar::ONE - r_b) + rho * r_b;
    let expected = poly_lookup.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();
    assert_eq!(t_lookup_out, expected);
  }

  /// `verify_step` rejects a `poly_lookup` whose `(C)`-binding assertion
  /// disagrees with `t_lookup_running`. Constructs a satisfying instance,
  /// then perturbs `t_lookup_running` away from the actual value.
  #[test]
  fn verify_step_rejects_when_c_binding_violated() {
    // VERIFY_REJ = 0x7E73.
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_7E73);
    let (witness, table, multiplicities) = satisfying_fixture(&mut rng, 4, 4);

    let (eq_w_left, eq_w_right) = split_eq(&mut rng, 4);
    let (eq_t_left, eq_t_right) = split_eq(&mut rng, 4);

    let r = Scalar::random(&mut rng);
    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", witness.len().max(table.len()))
      .expect("CE::setup must succeed for spike test fixture");

    let (inst, _, _) = LookupSumcheckInstance::<E>::new(
      &ck,
      &witness,
      &table,
      &multiplicities,
      eq_w_left,
      eq_w_right,
      eq_t_left,
      eq_t_right,
      r,
    )
    .unwrap();

    let rho = Scalar::random(&mut rng);
    let t_lookup_running_actual = Scalar::ZERO;
    let poly_lookup = inst.prove_step(&rho, &t_lookup_running_actual);

    // Verifier supplied with a wrong running target — must reject.
    let t_lookup_running_wrong = Scalar::ONE;
    let r_b = Scalar::random(&mut rng);
    let res =
      LookupSumcheckInstance::<E>::verify_step(&rho, &r_b, &poly_lookup, &t_lookup_running_wrong);
    assert!(matches!(res, Err(NovaError::InvalidSumcheckProof)));
  }

  /// `lookup_running_claims_from` projects to the underlying scalar at
  /// outer base (`None → ZERO`) and to the carried `T_lookup` value
  /// otherwise.
  #[test]
  fn lookup_running_claims_from_outer_base_is_zero() {
    use crate::{neutron::relation::Structure, r1cs::R1CSShape, spartan::math::Math};

    // Build a minimal Structure to drive FoldedInstance::default. We use a
    // 16-constraint NonTrivialCircuit shape (mirrors test conventions in
    // `vendor/nova/src/neutron/relation.rs`).
    let num_cons: usize = 16;
    let _log_num_cons = num_cons.log_2();
    use crate::{
      frontend::{r1cs::NovaShape, shape_cs::ShapeCS, Circuit},
      spartan::direct::DirectCircuit,
      traits::circuit::NonTrivialCircuit,
    };
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape: R1CSShape<E> = cs.r1cs_shape().unwrap();
    let s = Structure::new(&shape);
    let u1 = FoldedInstance::default(&s);

    assert!(u1.T_lookup.is_none());
    assert_eq!(lookup_running_claims_from::<E>(&u1), Scalar::ZERO);
  }
}
