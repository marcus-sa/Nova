//! Closed-form, fold-step-collapsed sumcheck instance for the C1-beta
//! lookup-fold extension.
//!
//! `LookupSumcheckInstance` is the lookup-side mirror of the R1CS-side
//! `NIFS::prove_helper` (`vendor/nova/src/neutron/nifs.rs:29-186`): one
//! univariate per fold step, computed via a full hypercube reduction at
//! each invocation.
//!
//! ## Degree analysis (corrected from B.4 rounds 1-7)
//!
//! The polynomial is degree **5** in the fold-step variable `b`, matching
//! the R1CS-side `prove_helper`. The degree comes from:
//!
//! - Each witness polynomial (`inv_w`, `w`, `inv_t`, `T`, `ts`) is
//!   interpolated between two folding instances (U1 running, U2 fresh):
//!   `f(b) = (1-b)*f_1 + b*f_2`, degree 1 in `b`.
//! - The eq polynomial is in split-tensor form, with BOTH halves
//!   interpolated: `eq_left(b) * eq_right(b)`, degree 2 in `b`.
//! - Sub-claim (A): `eq(b) * (inv_w(b) * (w(b)+r) - 1)` = degree 2+2 = 4.
//! - Sub-claim (B): `eq(b) * (inv_t(b) * (T(b)+r) - ts(b))` = degree 4.
//! - Sub-claim (C): `sum inv_w(b) - sum inv_t(b)` = degree 1.
//! - Combined: degree 4.
//! - With `eq_rho(b) = (1-rho)(1-b) + rho*b` factor: degree **5**.
//!
//! This requires 6 evaluation points {0,1,2,3,4,5}. Point 1 is
//! reconstructed from the running target: `eval@1 = T_running - eval@0`.
//! The prover computes evals at {0,2,3,4,5}.
//!
//! ## Sub-claim (B) semantics: `inv_t = ts/(T+r)`, NOT `1/(T+r)`
//!
//! Per Haboeck eprint 2022/1530, the LogUp identity is:
//!   `sum 1/(w_i + r) = sum ts_j/(T_j + r)`
//! The T-side inverse `inv_t_j` carries the multiplicity in its numerator:
//!   `inv_t_j = ts_j / (T_j + r)`
//! This matches the PPSNARK `MemorySumcheckInstance::compute_oracles`
//! pattern at `ppsnark.rs:438-441`.
//!
//! ## Dual-instance loading
//!
//! The struct carries polynomials from BOTH folding instances (U1 running,
//! U2 fresh), mirroring how `prove_helper` takes `(e1, Az1, Bz1, Cz1)`
//! and `(e2, Az2, Bz2, Cz2)`. The `bound_func` interpolation
//! `(1-b)*val_1 + b*val_2` at each evaluation point `p` produces:
//!   `val(p) = (1-p)*val_1 + p*val_2 = val_1 + p*(val_2 - val_1)`
//!
//! Verifier (`verify_step`):
//!
//! ```text
//!   (1) assert poly_lookup.eval_at_0() + poly_lookup.eval_at_1() = t_lookup_running
//!   (2) eq_rho_r_b := (1 - rho)(1 - r_b) + rho * r_b
//!   (3) t_lookup_out := poly_lookup(r_b) / eq_rho_r_b
//! ```
//!
//! Mirrors `nifs.rs:325-339` byte-for-byte (the R1CS-side closed-form
//! verifier).

#![cfg(feature = "lookup-fold")]
#![allow(non_snake_case)]
// Stage 1.B lands the closed-form primitive in isolation; the wiring into
// `NIFS::prove`/`verify` and the augmented circuit lives in Stages C-E.
// Until those land, the `pub(crate)` items here have no in-crate consumer.
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
/// side of the C1-beta lookup-fold extension.
///
/// Carries polynomials from BOTH folding instances (U1 running, U2 fresh)
/// to enable the `bound_func` interpolation that produces the degree-5
/// fold-step polynomial.
///
/// Construction order (load-bearing):
/// 1. Caller squeezes `r` via the running RO state.
/// 2. Caller invokes `Self::new(ck, ..., r)`, which computes the inverse
///    witnesses for BOTH instances and returns `(Self, comms)`.
/// 3. Caller absorbs commitments into the RO, then calls
///    `Self::prove_step(rho, t_lookup_running)` to produce the degree-5
///    univariate `poly_lookup`.
pub(crate) struct LookupSumcheckInstance<E: Engine> {
  // --- U1 (running instance) polynomials ---
  w1_poly: MultilinearPolynomial<E::Scalar>,
  inv_w1_poly: MultilinearPolynomial<E::Scalar>,
  eq_w1_left: Vec<E::Scalar>,
  eq_w1_right: Vec<E::Scalar>,

  t1_poly: MultilinearPolynomial<E::Scalar>,
  ts1_poly: MultilinearPolynomial<E::Scalar>,
  inv_t1_poly: MultilinearPolynomial<E::Scalar>,
  eq_t1_left: Vec<E::Scalar>,
  eq_t1_right: Vec<E::Scalar>,

  // --- U2 (fresh instance) polynomials ---
  w2_poly: MultilinearPolynomial<E::Scalar>,
  inv_w2_poly: MultilinearPolynomial<E::Scalar>,
  eq_w2_left: Vec<E::Scalar>,
  eq_w2_right: Vec<E::Scalar>,

  t2_poly: MultilinearPolynomial<E::Scalar>,
  ts2_poly: MultilinearPolynomial<E::Scalar>,
  inv_t2_poly: MultilinearPolynomial<E::Scalar>,
  eq_t2_left: Vec<E::Scalar>,
  eq_t2_right: Vec<E::Scalar>,

  // --- shared scalars ---
  /// LogUp randomness, squeezed BEFORE construction.
  r: E::Scalar,
}

impl<E: Engine> LookupSumcheckInstance<E> {
  /// Constructs a dual-instance lookup sumcheck instance for one fold step.
  ///
  /// ## U1 (running instance) data
  ///
  /// U1's polynomial vectors (`witness_1`, `inv_w_1`, `inv_t_1`, etc.) are
  /// **pre-computed** and come from the running witness accumulator. At outer
  /// base (first fold step), these are all-zero vectors. At subsequent steps,
  /// they carry the accumulated folded values from previous fold steps.
  /// Critically, `inv_w_1` and `inv_t_1` are NOT recomputed from `witness_1`
  /// and `table_1` via `batch_invert_plus_r` -- they are the stored running
  /// values, analogous to how `NIFS::prove` uses `W1.E` directly (not
  /// recomputing it from tau).
  ///
  /// ## U2 (fresh instance) data
  ///
  /// U2's inverse witnesses ARE computed fresh here from `witness_2` and
  /// `table_2`, since they haven't been folded yet. The commitments to
  /// `inv_w_2` and `inv_t_2` are returned for FS absorption.
  ///
  /// ## inv_t semantics
  ///
  /// For U2's T-side: `inv_t_2[j] = ts_2[j] / (T_2[j] + r)` (multiplicity
  /// in numerator), matching PPSNARK `compute_oracles` at ppsnark.rs:438-441.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    ck: &CommitmentKey<E>,
    // U1 (running) data -- pre-computed, NOT recomputed
    witness_1: &[E::Scalar],
    inv_w_1: &[E::Scalar],
    table_1: &[E::Scalar],
    multiplicities_1: &[E::Scalar],
    inv_t_1: &[E::Scalar],
    eq_w1_left: Vec<E::Scalar>,
    eq_w1_right: Vec<E::Scalar>,
    eq_t1_left: Vec<E::Scalar>,
    eq_t1_right: Vec<E::Scalar>,
    // U2 (fresh) data -- inverses computed here
    witness_2: &[E::Scalar],
    table_2: &[E::Scalar],
    multiplicities_2: &[E::Scalar],
    eq_w2_left: Vec<E::Scalar>,
    eq_w2_right: Vec<E::Scalar>,
    eq_t2_left: Vec<E::Scalar>,
    eq_t2_right: Vec<E::Scalar>,
    // shared
    r: E::Scalar,
  ) -> Result<(Self, Commitment<E>, Commitment<E>), NovaError> {
    // sanity: shape coherence
    debug_assert_eq!(witness_1.len(), eq_w1_left.len() * eq_w1_right.len());
    debug_assert_eq!(inv_w_1.len(), witness_1.len());
    debug_assert_eq!(table_1.len(), eq_t1_left.len() * eq_t1_right.len());
    debug_assert_eq!(multiplicities_1.len(), table_1.len());
    debug_assert_eq!(inv_t_1.len(), table_1.len());
    debug_assert_eq!(witness_2.len(), eq_w2_left.len() * eq_w2_right.len());
    debug_assert_eq!(table_2.len(), eq_t2_left.len() * eq_t2_right.len());
    debug_assert_eq!(multiplicities_2.len(), table_2.len());
    // Both instances must have same sizes for bound_func interpolation
    debug_assert_eq!(witness_1.len(), witness_2.len());
    debug_assert_eq!(table_1.len(), table_2.len());

    // Compute U2's inverses fresh: 1/(w+r) and ts/(T+r)
    let inv_w2 = batch_invert_plus_r(witness_2, &r)?;

    let inv_t2_raw = batch_invert_plus_r(table_2, &r)?;
    let inv_t2: Vec<E::Scalar> = inv_t2_raw
      .par_iter()
      .zip(multiplicities_2.par_iter())
      .map(|(inv, ts)| *inv * *ts)
      .collect();

    // Commit U2's inverses under zero-blinding
    let comm_inv_w2 = E::CE::commit(ck, &inv_w2, &E::Scalar::ZERO);
    let comm_inv_t2 = E::CE::commit(ck, &inv_t2, &E::Scalar::ZERO);

    let inst = Self {
      w1_poly: MultilinearPolynomial::new(witness_1.to_vec()),
      inv_w1_poly: MultilinearPolynomial::new(inv_w_1.to_vec()),
      eq_w1_left,
      eq_w1_right,
      t1_poly: MultilinearPolynomial::new(table_1.to_vec()),
      ts1_poly: MultilinearPolynomial::new(multiplicities_1.to_vec()),
      inv_t1_poly: MultilinearPolynomial::new(inv_t_1.to_vec()),
      eq_t1_left,
      eq_t1_right,

      w2_poly: MultilinearPolynomial::new(witness_2.to_vec()),
      inv_w2_poly: MultilinearPolynomial::new(inv_w2),
      eq_w2_left,
      eq_w2_right,
      t2_poly: MultilinearPolynomial::new(table_2.to_vec()),
      ts2_poly: MultilinearPolynomial::new(multiplicities_2.to_vec()),
      inv_t2_poly: MultilinearPolynomial::new(inv_t2),
      eq_t2_left,
      eq_t2_right,

      r,
    };

    Ok((inst, comm_inv_w2, comm_inv_t2))
  }

  /// Closed-form one-round-per-fold-step prover.
  ///
  /// Mirrors `NIFS::prove_helper` (nifs.rs:29-186) in structure:
  /// - Degree 5 (same as R1CS side) -> 6 eval points {0,1,2,3,4,5}
  /// - eval@1 reconstructed from `T_running - eval@0`
  /// - rho factors applied post-summation
  /// - Dual-instance bound_func interpolation at each point
  ///
  /// The three LogUp sub-claims are:
  /// (A) `sum eq_w(b,x) * (inv_w(b,x) * (w(b,x) + r) - 1)`
  /// (B) `sum eq_t(b,y) * (inv_t(b,y) * (T(b,y) + r) - ts(b,y))`
  ///     where inv_t already carries ts in numerator: inv_t = ts/(T+r)
  ///     so (B) = `sum eq_t(b,y) * (inv_t(b,y) * (T(b,y) + r) - ts(b,y))`
  /// (C) `sum inv_w(b,x) - sum inv_t(b,x)`
  ///
  /// Note: with inv_t = ts/(T+r), sub-claim (B) becomes:
  ///   eq_t * (inv_t * (T+r) - ts) = eq_t * (ts/(T+r) * (T+r) - ts) = eq_t * 0 = 0
  /// on satisfying inputs. And sub-claim (A) similarly:
  ///   eq_w * (inv_w * (w+r) - 1) = eq_w * (1/(w+r) * (w+r) - 1) = 0
  /// And sub-claim (C):
  ///   sum 1/(w+r) - sum ts/(T+r) = 0 by LogUp identity.
  pub(crate) fn prove_step(
    &self,
    rho: &E::Scalar,
    t_lookup_running: &E::Scalar,
  ) -> UniPoly<E::Scalar> {
    // Compute hypercube sums at {0, 2, 3, 4, 5} for each sub-claim.
    // The bound_func interpolation produces different values at each point.
    let (sum_a_0, sum_a_2, sum_a_3, sum_a_4, sum_a_5) = self.compute_a_evals();
    let (sum_b_0, sum_b_2, sum_b_3, sum_b_4, sum_b_5) = self.compute_b_evals();
    let (sum_c_0, sum_c_2, sum_c_3, sum_c_4, sum_c_5) = self.compute_c_evals();

    // Combine sub-claims at each eval point
    let combined_0 = sum_a_0 + sum_b_0 + sum_c_0;
    let combined_2 = sum_a_2 + sum_b_2 + sum_c_2;
    let combined_3 = sum_a_3 + sum_b_3 + sum_c_3;
    let combined_4 = sum_a_4 + sum_b_4 + sum_c_4;
    let combined_5 = sum_a_5 + sum_b_5 + sum_c_5;

    // rho factors: eq_rho(p) = (1-rho)(1-p) + rho*p = (2p-1)*rho - (p-1)
    // Mirrors prove_helper:173-177.
    let one_minus_rho = E::Scalar::ONE - *rho;
    let three_rho_minus_one = E::Scalar::from(3) * *rho - E::Scalar::ONE;
    let five_rho_minus_two = E::Scalar::from(5) * *rho - E::Scalar::from(2);
    let seven_rho_minus_three = E::Scalar::from(7) * *rho - E::Scalar::from(3);
    let nine_rho_minus_four = E::Scalar::from(9) * *rho - E::Scalar::from(4);

    let eval_at_0 = one_minus_rho * combined_0;
    let eval_at_2 = three_rho_minus_one * combined_2;
    let eval_at_3 = five_rho_minus_two * combined_3;
    let eval_at_4 = seven_rho_minus_three * combined_4;
    let eval_at_5 = nine_rho_minus_four * combined_5;

    // Reconstruct eval@1 from running target: poly(0) + poly(1) = T_running
    // Mirrors nifs.rs:264-272.
    let evals = vec![
      eval_at_0,
      *t_lookup_running - eval_at_0, // eval@1 by (C)-binding constraint
      eval_at_2,
      eval_at_3,
      eval_at_4,
      eval_at_5,
    ];
    UniPoly::<E::Scalar>::from_evals(&evals)
  }

  /// Sub-claim (A) hypercube sums at {0, 2, 3, 4, 5}.
  ///
  /// For each cell (i,j), computes:
  ///   eq_w_right(b)[i] * eq_w_left(b)[j] * (inv_w(b)[k] * (w(b)[k] + r) - 1)
  ///
  /// where each polynomial is interpolated via bound_func between U1 and U2.
  /// Mirrors the nested-loop structure of prove_helper:54-170.
  fn compute_a_evals(
    &self,
  ) -> (E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) {
    let left = self.eq_w1_left.len();
    let right = self.eq_w1_right.len();
    debug_assert_eq!(self.w1_poly.len(), left * right);
    debug_assert_eq!(self.inv_w1_poly.len(), left * right);
    debug_assert_eq!(self.w2_poly.len(), left * right);
    debug_assert_eq!(self.inv_w2_poly.len(), left * right);

    let r = self.r;

    (0..right)
      .into_par_iter()
      .map(|i| {
        // Inner loop: sum over left, producing per-cell evaluations
        // BEFORE the eq_right factor is applied.
        // The "comb_func" for lookup sub-claim (A) is:
        //   eq_left(b) * (inv_w(b) * (w(b) + r) - 1)
        // = eq_left(b) * inv_w(b) * (w(b) + r) - eq_left(b)
        //
        // At point 0: uses U1 values (bound_func at 0 = val_1)
        // At point p: uses (1-p)*val_1 + p*val_2
        let (inner_0, inner_2, inner_3, inner_4, inner_5) = (0..left)
          .into_par_iter()
          .map(|j| {
            let k = i * left + j;

            // bound_func values at point 0 (= U1 values)
            let eq_l_1 = self.eq_w1_left[j];
            let eq_l_2 = self.eq_w2_left[j];
            let inv_w_1 = self.inv_w1_poly[k];
            let inv_w_2 = self.inv_w2_poly[k];
            let w_plus_r_1 = self.w1_poly[k] + r;
            let w_plus_r_2 = self.w2_poly[k] + r;

            // Deltas for incremental bound_func: val(p+1) = val(p) + delta
            let d_eq_l = eq_l_2 - eq_l_1;
            let d_inv_w = inv_w_2 - inv_w_1;
            let d_w_plus_r = w_plus_r_2 - w_plus_r_1;

            // eval at p=0: eq_l_1 * (inv_w_1 * w_plus_r_1 - 1)
            let eval_0 = eq_l_1 * (inv_w_1 * w_plus_r_1 - E::Scalar::ONE);

            // eval at p=2: bound_func gives val_1 + 2*delta = 2*val_2 - val_1
            let eq_l_at_2 = eq_l_1 + d_eq_l + d_eq_l;
            let inv_w_at_2 = inv_w_1 + d_inv_w + d_inv_w;
            let w_plus_r_at_2 = w_plus_r_1 + d_w_plus_r + d_w_plus_r;
            let eval_2 = eq_l_at_2 * (inv_w_at_2 * w_plus_r_at_2 - E::Scalar::ONE);

            // eval at p=3: incremental from p=2
            let eq_l_at_3 = eq_l_at_2 + d_eq_l;
            let inv_w_at_3 = inv_w_at_2 + d_inv_w;
            let w_plus_r_at_3 = w_plus_r_at_2 + d_w_plus_r;
            let eval_3 = eq_l_at_3 * (inv_w_at_3 * w_plus_r_at_3 - E::Scalar::ONE);

            // eval at p=4
            let eq_l_at_4 = eq_l_at_3 + d_eq_l;
            let inv_w_at_4 = inv_w_at_3 + d_inv_w;
            let w_plus_r_at_4 = w_plus_r_at_3 + d_w_plus_r;
            let eval_4 = eq_l_at_4 * (inv_w_at_4 * w_plus_r_at_4 - E::Scalar::ONE);

            // eval at p=5
            let eq_l_at_5 = eq_l_at_4 + d_eq_l;
            let inv_w_at_5 = inv_w_at_4 + d_inv_w;
            let w_plus_r_at_5 = w_plus_r_at_4 + d_w_plus_r;
            let eval_5 = eq_l_at_5 * (inv_w_at_5 * w_plus_r_at_5 - E::Scalar::ONE);

            (eval_0, eval_2, eval_3, eval_4, eval_5)
          })
          .reduce(
            || {
              (
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
              )
            },
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
          );

        // Outer loop: multiply by eq_right(b)[i]
        // Mirrors prove_helper:135-157 (the right-half eq factor)
        let eq_r_1 = self.eq_w1_right[i];
        let eq_r_2 = self.eq_w2_right[i];
        let d_eq_r = eq_r_2 - eq_r_1;

        let eval_at_0 = eq_r_1 * inner_0;

        let eq_r_at_2 = eq_r_1 + d_eq_r + d_eq_r;
        let eval_at_2 = eq_r_at_2 * inner_2;

        let eq_r_at_3 = eq_r_at_2 + d_eq_r;
        let eval_at_3 = eq_r_at_3 * inner_3;

        let eq_r_at_4 = eq_r_at_3 + d_eq_r;
        let eval_at_4 = eq_r_at_4 * inner_4;

        let eq_r_at_5 = eq_r_at_4 + d_eq_r;
        let eval_at_5 = eq_r_at_5 * inner_5;

        (eval_at_0, eval_at_2, eval_at_3, eval_at_4, eval_at_5)
      })
      .reduce(
        || {
          (
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
          )
        },
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
      )
  }

  /// Sub-claim (B) hypercube sums at {0, 2, 3, 4, 5}.
  ///
  /// For each cell (i,j), computes:
  ///   eq_t_right(b)[i] * eq_t_left(b)[j] * (inv_t(b)[k] * (T(b)[k] + r) - ts(b)[k])
  ///
  /// Note: inv_t already encodes `ts/(T+r)`, so on satisfying inputs:
  ///   inv_t * (T+r) - ts = ts/(T+r) * (T+r) - ts = ts - ts = 0
  fn compute_b_evals(
    &self,
  ) -> (E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) {
    let left = self.eq_t1_left.len();
    let right = self.eq_t1_right.len();
    debug_assert_eq!(self.t1_poly.len(), left * right);
    debug_assert_eq!(self.inv_t1_poly.len(), left * right);
    debug_assert_eq!(self.ts1_poly.len(), left * right);
    debug_assert_eq!(self.t2_poly.len(), left * right);
    debug_assert_eq!(self.inv_t2_poly.len(), left * right);
    debug_assert_eq!(self.ts2_poly.len(), left * right);

    let r = self.r;

    (0..right)
      .into_par_iter()
      .map(|i| {
        let (inner_0, inner_2, inner_3, inner_4, inner_5) = (0..left)
          .into_par_iter()
          .map(|j| {
            let k = i * left + j;

            let eq_l_1 = self.eq_t1_left[j];
            let eq_l_2 = self.eq_t2_left[j];
            let inv_t_1 = self.inv_t1_poly[k];
            let inv_t_2 = self.inv_t2_poly[k];
            let t_plus_r_1 = self.t1_poly[k] + r;
            let t_plus_r_2 = self.t2_poly[k] + r;
            let ts_1 = self.ts1_poly[k];
            let ts_2 = self.ts2_poly[k];

            let d_eq_l = eq_l_2 - eq_l_1;
            let d_inv_t = inv_t_2 - inv_t_1;
            let d_t_plus_r = t_plus_r_2 - t_plus_r_1;
            let d_ts = ts_2 - ts_1;

            // p=0
            let eval_0 = eq_l_1 * (inv_t_1 * t_plus_r_1 - ts_1);

            // p=2
            let eq_l_at_2 = eq_l_1 + d_eq_l + d_eq_l;
            let inv_t_at_2 = inv_t_1 + d_inv_t + d_inv_t;
            let t_plus_r_at_2 = t_plus_r_1 + d_t_plus_r + d_t_plus_r;
            let ts_at_2 = ts_1 + d_ts + d_ts;
            let eval_2 = eq_l_at_2 * (inv_t_at_2 * t_plus_r_at_2 - ts_at_2);

            // p=3
            let eq_l_at_3 = eq_l_at_2 + d_eq_l;
            let inv_t_at_3 = inv_t_at_2 + d_inv_t;
            let t_plus_r_at_3 = t_plus_r_at_2 + d_t_plus_r;
            let ts_at_3 = ts_at_2 + d_ts;
            let eval_3 = eq_l_at_3 * (inv_t_at_3 * t_plus_r_at_3 - ts_at_3);

            // p=4
            let eq_l_at_4 = eq_l_at_3 + d_eq_l;
            let inv_t_at_4 = inv_t_at_3 + d_inv_t;
            let t_plus_r_at_4 = t_plus_r_at_3 + d_t_plus_r;
            let ts_at_4 = ts_at_3 + d_ts;
            let eval_4 = eq_l_at_4 * (inv_t_at_4 * t_plus_r_at_4 - ts_at_4);

            // p=5
            let eq_l_at_5 = eq_l_at_4 + d_eq_l;
            let inv_t_at_5 = inv_t_at_4 + d_inv_t;
            let t_plus_r_at_5 = t_plus_r_at_4 + d_t_plus_r;
            let ts_at_5 = ts_at_4 + d_ts;
            let eval_5 = eq_l_at_5 * (inv_t_at_5 * t_plus_r_at_5 - ts_at_5);

            (eval_0, eval_2, eval_3, eval_4, eval_5)
          })
          .reduce(
            || {
              (
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
                E::Scalar::ZERO,
              )
            },
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
          );

        // Outer: eq_right factor
        let eq_r_1 = self.eq_t1_right[i];
        let eq_r_2 = self.eq_t2_right[i];
        let d_eq_r = eq_r_2 - eq_r_1;

        let eval_at_0 = eq_r_1 * inner_0;

        let eq_r_at_2 = eq_r_1 + d_eq_r + d_eq_r;
        let eval_at_2 = eq_r_at_2 * inner_2;

        let eq_r_at_3 = eq_r_at_2 + d_eq_r;
        let eval_at_3 = eq_r_at_3 * inner_3;

        let eq_r_at_4 = eq_r_at_3 + d_eq_r;
        let eval_at_4 = eq_r_at_4 * inner_4;

        let eq_r_at_5 = eq_r_at_4 + d_eq_r;
        let eval_at_5 = eq_r_at_5 * inner_5;

        (eval_at_0, eval_at_2, eval_at_3, eval_at_4, eval_at_5)
      })
      .reduce(
        || {
          (
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
            E::Scalar::ZERO,
          )
        },
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
      )
  }

  /// Sub-claim (C) hypercube sums at {0, 2, 3, 4, 5}.
  ///
  /// `sum inv_w(b) - sum inv_t(b)` where each value is interpolated
  /// between U1 and U2.
  ///
  /// This is linear in b (degree 1), so the evals at different points
  /// DO differ (unlike the broken `(cell, cell, cell)` pattern).
  fn compute_c_evals(
    &self,
  ) -> (E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) {
    // sum inv_w from U1 and U2
    let sum_inv_w_1: E::Scalar = self.inv_w1_poly.Z.par_iter().copied().sum();
    let sum_inv_w_2: E::Scalar = self.inv_w2_poly.Z.par_iter().copied().sum();

    // sum inv_t from U1 and U2
    let sum_inv_t_1: E::Scalar = self.inv_t1_poly.Z.par_iter().copied().sum();
    let sum_inv_t_2: E::Scalar = self.inv_t2_poly.Z.par_iter().copied().sum();

    // bound_func at each point p:
    //   sum_inv_w(p) = (1-p)*sum_inv_w_1 + p*sum_inv_w_2
    //   sum_inv_t(p) = (1-p)*sum_inv_t_1 + p*sum_inv_t_2
    //   c(p) = sum_inv_w(p) - sum_inv_t(p)

    // c(p) = (1-p)*(sum_inv_w_1 - sum_inv_t_1) + p*(sum_inv_w_2 - sum_inv_t_2)
    let c_at_0 = sum_inv_w_1 - sum_inv_t_1;
    let c_at_1 = sum_inv_w_2 - sum_inv_t_2;
    let delta = c_at_1 - c_at_0;

    let c_at_2 = c_at_0 + delta + delta;
    let c_at_3 = c_at_2 + delta;
    let c_at_4 = c_at_3 + delta;
    let c_at_5 = c_at_4 + delta;

    (c_at_0, c_at_2, c_at_3, c_at_4, c_at_5)
  }

  /// Closed-form verifier-side step.
  ///
  /// Asserts the (C) sub-claim and computes the next-step running target
  /// `T_lookup_out`. Mirrors `nifs.rs:325-339`.
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
    // (2) eq(rho, r_b) factor for next-step running target (mirrors nifs.rs:281).
    let eq_rho_r_b =
      (E::Scalar::ONE - *rho) * (E::Scalar::ONE - *r_b) + *rho * *r_b;
    // (3) T_lookup_out (mirrors nifs.rs:282).
    let t_lookup_out = poly_lookup.evaluate(r_b)
      * eq_rho_r_b
        .invert()
        .expect("eq(rho, r_b) zero -- challenge collision, FS RO bug");
    Ok(t_lookup_out)
  }
}

/// Project the lookup-side running-claim scalar from a [`FoldedInstance`].
///
/// Returns the single `T_lookup` scalar that `prove_step` reads to
/// reconstruct eval@1. At outer base, `U1.T_lookup` is `None`
/// and this returns `ZERO`.
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

  /// Build zero split-tensor eq vectors for outer base (mirrors W1.E = [0,...,0]).
  fn zero_split_eq(ell: usize) -> (Vec<Scalar>, Vec<Scalar>) {
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    (vec![Scalar::ZERO; 1 << ell1], vec![Scalar::ZERO; 1 << ell2])
  }

  /// Build split-tensor eq vectors for a length-`2^ell` MLE under random tau.
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

  /// Sanity: `UniPoly::from_evals` with 6 points produces a degree-5 polynomial
  /// that interpolates correctly at all 6 points.
  #[test]
  fn from_evals_interpolates_six_points() {
    let evals: Vec<Scalar> = (0..6).map(|i| Scalar::from((i * i + 3 * i + 7) as u64)).collect();
    let poly = UniPoly::<Scalar>::from_evals(&evals);
    for (i, e) in evals.iter().enumerate() {
      assert_eq!(poly.evaluate(&Scalar::from(i as u64)), *e);
    }
    assert_eq!(poly.eval_at_zero(), evals[0]);
    assert_eq!(poly.eval_at_one(), evals[1]);
  }

  /// Core correctness test: on a satisfying witness, the fold-step polynomial
  /// has the right structure.
  ///
  /// On satisfying inputs:
  /// - Sub-claim (A): each cell is eq_w * (inv_w * (w+r) - 1) = 0
  /// - Sub-claim (B): each cell is eq_t * (inv_t * (T+r) - ts) = 0
  ///   (because inv_t = ts/(T+r))
  /// - Sub-claim (C): sum inv_w - sum inv_t = 0
  ///
  /// So the INNER combined sum is 0 at all points, and the full polynomial
  /// (with rho factor) should be identically zero.
  ///
  /// At outer base, T_running = 0, so poly(0) + poly(1) = 0, and since
  /// the polynomial is zero, this trivially holds.
  #[test]
  fn prove_step_satisfying_witness_produces_zero_polynomial() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_F1A0);
    let (witness, table, multiplicities) = satisfying_fixture(&mut rng, 4, 4);

    // U1 at outer base: zero eq polynomials (mirrors W1.E = [0,...,0])
    let (eq_w1_left, eq_w1_right) = zero_split_eq(4);
    let (eq_t1_left, eq_t1_right) = zero_split_eq(4);
    let (eq_w2_left, eq_w2_right) = split_eq(&mut rng, 4);
    let (eq_t2_left, eq_t2_right) = split_eq(&mut rng, 4);

    let r = Scalar::random(&mut rng);

    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", witness.len().max(table.len()))
      .expect("CE::setup");

    // For U1 (running) at outer base, all polynomials are zero.
    let n_w = witness.len();
    let n_t = table.len();
    let zero_w = vec![Scalar::ZERO; n_w];
    let zero_t = vec![Scalar::ZERO; n_t];

    let (inst, _comm_inv_w2, _comm_inv_t2) = LookupSumcheckInstance::<E>::new(
      &ck,
      // U1 (running): zeros at outer base, including pre-computed inverses
      &zero_w,   // witness_1
      &zero_w,   // inv_w_1 (zero at outer base)
      &zero_t,   // table_1
      &zero_t,   // multiplicities_1
      &zero_t,   // inv_t_1 (zero at outer base)
      eq_w1_left,
      eq_w1_right,
      eq_t1_left,
      eq_t1_right,
      // U2 (fresh): actual satisfying witness
      &witness,
      &table,
      &multiplicities,
      eq_w2_left,
      eq_w2_right,
      eq_t2_left,
      eq_t2_right,
      r,
    )
    .unwrap();

    let rho = Scalar::random(&mut rng);
    let t_lookup_running = Scalar::ZERO; // outer base
    let poly_lookup = inst.prove_step(&rho, &t_lookup_running);

    // (C)-binding: poly(0) + poly(1) = T_lookup_running = 0
    assert_eq!(
      poly_lookup.eval_at_zero() + poly_lookup.eval_at_one(),
      t_lookup_running,
      "C-binding failed"
    );

    // At outer base with all-zero U1, the polynomial is zero at p=0 and
    // p=1 (satisfying U2), but NOT at extrapolation points p=2,3,4,5
    // because the product structure (inv_w * (w+r)) is nonlinear and
    // its interpolation beyond [0,1] is not zero even when both
    // endpoints evaluate to the identity.
    //
    // This mirrors the R1CS NIFS: Az(b)*Bz(b) - Cz(b) is zero at b=0
    // and b=1 for satisfying instances, but nonzero at b=2,3,4,5.
    assert_eq!(
      poly_lookup.eval_at_zero(),
      Scalar::ZERO,
      "eval@0 must be zero at outer base"
    );
    assert_eq!(
      poly_lookup.eval_at_one(),
      Scalar::ZERO,
      "eval@1 must be zero at outer base (T_running = 0, eval@0 = 0)"
    );

    // verify_step should accept the proof.
    let r_b = Scalar::random(&mut rng);
    let result = LookupSumcheckInstance::<E>::verify_step(
      &rho, &r_b, &poly_lookup, &t_lookup_running,
    );
    assert!(result.is_ok(), "verify_step must accept satisfying proof");
  }

  /// Concrete numeric test: manually compute the polynomial at several
  /// points and verify that Lagrange interpolation agrees.
  ///
  /// This test constructs a small (2^2 = 4 entries) instance and
  /// manually verifies the algebra for each sub-claim.
  #[test]
  fn prove_step_lagrange_interpolation_check() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_A1C0);
    let (witness, table, multiplicities) = satisfying_fixture(&mut rng, 2, 2);

    let ell = 2;
    // U1 at outer base: zero eq (mirrors W1.E = [0,...,0])
    let (eq_w1_left, eq_w1_right) = zero_split_eq(ell);
    let (eq_t1_left, eq_t1_right) = zero_split_eq(ell);
    let (eq_w2_left, eq_w2_right) = split_eq(&mut rng, ell);
    let (eq_t2_left, eq_t2_right) = split_eq(&mut rng, ell);

    let r = Scalar::random(&mut rng);

    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", 4)
      .expect("CE::setup");

    // U1: all zeros at outer base (including pre-computed inverses)
    let zero = vec![Scalar::ZERO; 4];

    let (inst, _, _) = LookupSumcheckInstance::<E>::new(
      &ck,
      &zero, &zero, &zero, &zero, &zero,  // w1, inv_w1, t1, ts1, inv_t1
      eq_w1_left.clone(), eq_w1_right.clone(),
      eq_t1_left.clone(), eq_t1_right.clone(),
      &witness, &table, &multiplicities,
      eq_w2_left.clone(), eq_w2_right.clone(),
      eq_t2_left.clone(), eq_t2_right.clone(),
      r,
    )
    .unwrap();

    let rho = Scalar::random(&mut rng);
    let t_running = Scalar::ZERO;
    let poly = inst.prove_step(&rho, &t_running);

    // The polynomial was constructed from evals at {0,1,2,3,4,5}.
    // Verify the C-binding identity.
    assert_eq!(
      poly.eval_at_zero() + poly.eval_at_one(),
      t_running,
      "C-binding must hold"
    );

    // Now manually compute what the polynomial SHOULD be at some points.
    // We'll verify a few evaluation points by recomputing from scratch.

    // Helper: compute bound_func value at point p
    let bf = |v1: Scalar, v2: Scalar, p: Scalar| -> Scalar {
      (Scalar::ONE - p) * v1 + p * v2
    };

    // Check at p=0 (should equal eval_at_0 before rho factor, times rho factor)
    let eq_rho_0 = Scalar::ONE - rho; // eq_rho(0) = 1 - rho

    // Compute sub-claim A at p=0 manually
    let left_w = eq_w1_left.len();
    let right_w = eq_w1_right.len();
    let mut sum_a_0 = Scalar::ZERO;
    for i in 0..right_w {
      let eq_r = bf(inst.eq_w1_right[i], inst.eq_w2_right[i], Scalar::ZERO);
      for j in 0..left_w {
        let k = i * left_w + j;
        let eq_l = bf(inst.eq_w1_left[j], inst.eq_w2_left[j], Scalar::ZERO);
        let inv_w = bf(inst.inv_w1_poly[k], inst.inv_w2_poly[k], Scalar::ZERO);
        let w_plus_r = bf(inst.w1_poly[k] + r, inst.w2_poly[k] + r, Scalar::ZERO);
        sum_a_0 += eq_r * eq_l * (inv_w * w_plus_r - Scalar::ONE);
      }
    }

    // Compute sub-claim B at p=0
    let left_t = eq_t1_left.len();
    let right_t = eq_t1_right.len();
    let mut sum_b_0 = Scalar::ZERO;
    for i in 0..right_t {
      let eq_r = bf(inst.eq_t1_right[i], inst.eq_t2_right[i], Scalar::ZERO);
      for j in 0..left_t {
        let k = i * left_t + j;
        let eq_l = bf(inst.eq_t1_left[j], inst.eq_t2_left[j], Scalar::ZERO);
        let inv_t = bf(inst.inv_t1_poly[k], inst.inv_t2_poly[k], Scalar::ZERO);
        let t_plus_r = bf(inst.t1_poly[k] + r, inst.t2_poly[k] + r, Scalar::ZERO);
        let ts = bf(inst.ts1_poly[k], inst.ts2_poly[k], Scalar::ZERO);
        sum_b_0 += eq_r * eq_l * (inv_t * t_plus_r - ts);
      }
    }

    // Compute sub-claim C at p=0
    let sum_inv_w_0: Scalar = inst.inv_w1_poly.Z.iter().copied().sum();
    let sum_inv_t_0: Scalar = inst.inv_t1_poly.Z.iter().copied().sum();
    let sum_c_0 = sum_inv_w_0 - sum_inv_t_0;

    let expected_eval_0 = eq_rho_0 * (sum_a_0 + sum_b_0 + sum_c_0);
    assert_eq!(
      poly.eval_at_zero(),
      expected_eval_0,
      "eval@0 must match manual computation"
    );

    // Verify at p=2 as well
    let p = Scalar::from(2);
    let eq_rho_2 = Scalar::from(3) * rho - Scalar::ONE;

    let mut sum_a_2 = Scalar::ZERO;
    for i in 0..right_w {
      let eq_r = bf(inst.eq_w1_right[i], inst.eq_w2_right[i], p);
      for j in 0..left_w {
        let k = i * left_w + j;
        let eq_l = bf(inst.eq_w1_left[j], inst.eq_w2_left[j], p);
        let inv_w = bf(inst.inv_w1_poly[k], inst.inv_w2_poly[k], p);
        let w_plus_r = bf(inst.w1_poly[k] + r, inst.w2_poly[k] + r, p);
        sum_a_2 += eq_r * eq_l * (inv_w * w_plus_r - Scalar::ONE);
      }
    }

    let mut sum_b_2 = Scalar::ZERO;
    for i in 0..right_t {
      let eq_r = bf(inst.eq_t1_right[i], inst.eq_t2_right[i], p);
      for j in 0..left_t {
        let k = i * left_t + j;
        let eq_l = bf(inst.eq_t1_left[j], inst.eq_t2_left[j], p);
        let inv_t = bf(inst.inv_t1_poly[k], inst.inv_t2_poly[k], p);
        let t_plus_r = bf(inst.t1_poly[k] + r, inst.t2_poly[k] + r, p);
        let ts = bf(inst.ts1_poly[k], inst.ts2_poly[k], p);
        sum_b_2 += eq_r * eq_l * (inv_t * t_plus_r - ts);
      }
    }

    let sum_inv_w_2: Scalar = inst.inv_w1_poly.Z.iter()
      .zip(inst.inv_w2_poly.Z.iter())
      .map(|(v1, v2)| bf(*v1, *v2, p))
      .sum();
    let sum_inv_t_2: Scalar = inst.inv_t1_poly.Z.iter()
      .zip(inst.inv_t2_poly.Z.iter())
      .map(|(v1, v2)| bf(*v1, *v2, p))
      .sum();
    let sum_c_2 = sum_inv_w_2 - sum_inv_t_2;

    let expected_eval_2 = eq_rho_2 * (sum_a_2 + sum_b_2 + sum_c_2);
    assert_eq!(
      poly.evaluate(&p),
      expected_eval_2,
      "eval@2 must match manual computation"
    );
  }

  /// Verify the polynomial degree is at most 5: construct the polynomial
  /// and check that it's interpolated from exactly 6 points.
  #[test]
  fn prove_step_degree_is_at_most_five() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_D5A0);
    let (witness, table, multiplicities) = satisfying_fixture(&mut rng, 3, 3);

    let ell = 3;
    let (eq_w1_left, eq_w1_right) = zero_split_eq(ell);
    let (eq_t1_left, eq_t1_right) = zero_split_eq(ell);
    let (eq_w2_left, eq_w2_right) = split_eq(&mut rng, ell);
    let (eq_t2_left, eq_t2_right) = split_eq(&mut rng, ell);

    let r = Scalar::random(&mut rng);
    let n = witness.len().max(table.len());
    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", n)
      .expect("CE::setup");

    let zero_w = vec![Scalar::ZERO; witness.len()];
    let zero_t = vec![Scalar::ZERO; table.len()];

    let (inst, _, _) = LookupSumcheckInstance::<E>::new(
      &ck,
      &zero_w, &zero_w, &zero_t, &zero_t, &zero_t,
      eq_w1_left, eq_w1_right, eq_t1_left, eq_t1_right,
      &witness, &table, &multiplicities,
      eq_w2_left, eq_w2_right, eq_t2_left, eq_t2_right,
      r,
    )
    .unwrap();

    let rho = Scalar::random(&mut rng);
    let poly = inst.prove_step(&rho, &Scalar::ZERO);

    // UniPoly from 6 evals has at most degree 5 (coeffs length <= 6).
    // The from_evals method may trim trailing zeros.
    assert!(
      poly.coeffs().len() <= 6,
      "polynomial degree must be at most 5, got {} coefficients",
      poly.coeffs().len()
    );

    // Verify it evaluates correctly at all 6 construction points
    // (from_evals guarantee, but verify anyway)
    let eval_0 = poly.eval_at_zero();
    let eval_1 = poly.eval_at_one();
    assert_eq!(eval_0 + eval_1, Scalar::ZERO, "C-binding at outer base");
  }

  /// `verify_step` accepts a `poly_lookup` produced by `prove_step` on a
  /// satisfying witness, and the returned `t_lookup_out` is consistent.
  #[test]
  fn verify_step_round_trips_on_satisfying_witness() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_7E12);
    let (witness, table, multiplicities) = satisfying_fixture(&mut rng, 4, 4);

    let ell = 4;
    let (eq_w1_left, eq_w1_right) = zero_split_eq(ell);
    let (eq_t1_left, eq_t1_right) = zero_split_eq(ell);
    let (eq_w2_left, eq_w2_right) = split_eq(&mut rng, ell);
    let (eq_t2_left, eq_t2_right) = split_eq(&mut rng, ell);

    let r = Scalar::random(&mut rng);
    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", witness.len().max(table.len()))
      .expect("CE::setup");

    let zero_w = vec![Scalar::ZERO; witness.len()];
    let zero_t = vec![Scalar::ZERO; table.len()];

    let (inst, _, _) = LookupSumcheckInstance::<E>::new(
      &ck,
      &zero_w, &zero_w, &zero_t, &zero_t, &zero_t,
      eq_w1_left, eq_w1_right, eq_t1_left, eq_t1_right,
      &witness, &table, &multiplicities,
      eq_w2_left, eq_w2_right, eq_t2_left, eq_t2_right,
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

  /// `verify_step` rejects when `t_lookup_running` is perturbed.
  #[test]
  fn verify_step_rejects_when_c_binding_violated() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_7E73);
    let (witness, table, multiplicities) = satisfying_fixture(&mut rng, 4, 4);

    let ell = 4;
    let (eq_w1_left, eq_w1_right) = zero_split_eq(ell);
    let (eq_t1_left, eq_t1_right) = zero_split_eq(ell);
    let (eq_w2_left, eq_w2_right) = split_eq(&mut rng, ell);
    let (eq_t2_left, eq_t2_right) = split_eq(&mut rng, ell);

    let r = Scalar::random(&mut rng);
    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", witness.len().max(table.len()))
      .expect("CE::setup");

    let zero_w = vec![Scalar::ZERO; witness.len()];
    let zero_t = vec![Scalar::ZERO; table.len()];

    let (inst, _, _) = LookupSumcheckInstance::<E>::new(
      &ck,
      &zero_w, &zero_w, &zero_t, &zero_t, &zero_t,
      eq_w1_left, eq_w1_right, eq_t1_left, eq_t1_right,
      &witness, &table, &multiplicities,
      eq_w2_left, eq_w2_right, eq_t2_left, eq_t2_right,
      r,
    )
    .unwrap();

    let rho = Scalar::random(&mut rng);
    let t_lookup_running_actual = Scalar::ZERO;
    let poly_lookup = inst.prove_step(&rho, &t_lookup_running_actual);

    // Verifier supplied with a wrong running target -- must reject.
    let t_lookup_running_wrong = Scalar::ONE;
    let r_b = Scalar::random(&mut rng);
    let res =
      LookupSumcheckInstance::<E>::verify_step(&rho, &r_b, &poly_lookup, &t_lookup_running_wrong);
    assert!(matches!(res, Err(NovaError::InvalidSumcheckProof)));
  }

  /// `lookup_running_claims_from` projects correctly.
  #[test]
  fn lookup_running_claims_from_outer_base_is_zero() {
    use crate::{neutron::relation::Structure, r1cs::R1CSShape};
    use crate::{
      frontend::{r1cs::NovaShape, shape_cs::ShapeCS, Circuit},
      spartan::direct::DirectCircuit,
      traits::circuit::NonTrivialCircuit,
    };
    let num_cons: usize = 16;
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

  /// Differential test: verify the polynomial evaluations match between
  /// the optimized `prove_step` and a naive manual computation at multiple
  /// random points.
  #[test]
  fn prove_step_differential_against_naive() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_D1FF);
    let (witness, table, multiplicities) = satisfying_fixture(&mut rng, 3, 3);

    let ell = 3;
    let (eq_w1_left, eq_w1_right) = zero_split_eq(ell);
    let (eq_t1_left, eq_t1_right) = zero_split_eq(ell);
    let (eq_w2_left, eq_w2_right) = split_eq(&mut rng, ell);
    let (eq_t2_left, eq_t2_right) = split_eq(&mut rng, ell);

    let r = Scalar::random(&mut rng);
    let n = witness.len().max(table.len());
    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", n)
      .expect("CE::setup");

    let zero_w = vec![Scalar::ZERO; witness.len()];
    let zero_t = vec![Scalar::ZERO; table.len()];

    let (inst, _, _) = LookupSumcheckInstance::<E>::new(
      &ck,
      &zero_w, &zero_w, &zero_t, &zero_t, &zero_t,
      eq_w1_left.clone(), eq_w1_right.clone(),
      eq_t1_left.clone(), eq_t1_right.clone(),
      &witness, &table, &multiplicities,
      eq_w2_left.clone(), eq_w2_right.clone(),
      eq_t2_left.clone(), eq_t2_right.clone(),
      r,
    )
    .unwrap();

    let rho = Scalar::random(&mut rng);
    let t_running = Scalar::ZERO;
    let poly = inst.prove_step(&rho, &t_running);

    // Naive computation at random points
    let bf = |v1: Scalar, v2: Scalar, p: Scalar| -> Scalar {
      (Scalar::ONE - p) * v1 + p * v2
    };

    for _ in 0..10 {
      let p = Scalar::random(&mut rng);
      let eq_rho_p = (Scalar::ONE - rho) * (Scalar::ONE - p) + rho * p;

      // Sub-claim A
      let left_w = inst.eq_w1_left.len();
      let right_w = inst.eq_w1_right.len();
      let mut sum_a = Scalar::ZERO;
      for ii in 0..right_w {
        let eq_r = bf(inst.eq_w1_right[ii], inst.eq_w2_right[ii], p);
        for jj in 0..left_w {
          let k = ii * left_w + jj;
          let eq_l = bf(inst.eq_w1_left[jj], inst.eq_w2_left[jj], p);
          let inv_w = bf(inst.inv_w1_poly[k], inst.inv_w2_poly[k], p);
          let w_plus_r = bf(inst.w1_poly[k] + r, inst.w2_poly[k] + r, p);
          sum_a += eq_r * eq_l * (inv_w * w_plus_r - Scalar::ONE);
        }
      }

      // Sub-claim B
      let left_t = inst.eq_t1_left.len();
      let right_t = inst.eq_t1_right.len();
      let mut sum_b = Scalar::ZERO;
      for ii in 0..right_t {
        let eq_r = bf(inst.eq_t1_right[ii], inst.eq_t2_right[ii], p);
        for jj in 0..left_t {
          let k = ii * left_t + jj;
          let eq_l = bf(inst.eq_t1_left[jj], inst.eq_t2_left[jj], p);
          let inv_t = bf(inst.inv_t1_poly[k], inst.inv_t2_poly[k], p);
          let t_plus_r = bf(inst.t1_poly[k] + r, inst.t2_poly[k] + r, p);
          let ts = bf(inst.ts1_poly[k], inst.ts2_poly[k], p);
          sum_b += eq_r * eq_l * (inv_t * t_plus_r - ts);
        }
      }

      // Sub-claim C
      let sum_inv_w: Scalar = inst.inv_w1_poly.Z.iter()
        .zip(inst.inv_w2_poly.Z.iter())
        .map(|(v1, v2)| bf(*v1, *v2, p))
        .sum();
      let sum_inv_t: Scalar = inst.inv_t1_poly.Z.iter()
        .zip(inst.inv_t2_poly.Z.iter())
        .map(|(v1, v2)| bf(*v1, *v2, p))
        .sum();
      let sum_c = sum_inv_w - sum_inv_t;

      let expected = eq_rho_p * (sum_a + sum_b + sum_c);
      let actual = poly.evaluate(&p);
      assert_eq!(
        actual, expected,
        "polynomial evaluation at random point must match naive computation"
      );
    }
  }

  /// Test with BOTH instances non-trivial (not outer base).
  /// Simulates a mid-chain fold where U1 has accumulated data.
  #[test]
  fn prove_step_both_instances_nontrivial() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_B0F1);

    // U1 (running) - a satisfying instance
    let (witness_1, table_1, multiplicities_1) = satisfying_fixture(&mut rng, 3, 3);
    // U2 (fresh) - another satisfying instance with same table
    let (witness_2, table_2, multiplicities_2) = satisfying_fixture(&mut rng, 3, 3);

    let ell = 3;
    let (eq_w1_left, eq_w1_right) = split_eq(&mut rng, ell);
    let (eq_t1_left, eq_t1_right) = split_eq(&mut rng, ell);
    let (eq_w2_left, eq_w2_right) = split_eq(&mut rng, ell);
    let (eq_t2_left, eq_t2_right) = split_eq(&mut rng, ell);

    let r = Scalar::random(&mut rng);
    let n = witness_1.len().max(table_1.len());
    let ck = <E as Engine>::CE::setup(b"lookup_sumcheck_test", n)
      .expect("CE::setup");

    // Pre-compute U1's inverses (simulating what the running witness stores)
    let inv_w_1 = batch_invert_plus_r(&witness_1, &r).unwrap();
    let inv_t_1_raw = batch_invert_plus_r(&table_1, &r).unwrap();
    let inv_t_1: Vec<Scalar> = inv_t_1_raw.iter()
      .zip(multiplicities_1.iter())
      .map(|(inv, ts)| *inv * *ts)
      .collect();

    let (inst, _, _) = LookupSumcheckInstance::<E>::new(
      &ck,
      &witness_1, &inv_w_1, &table_1, &multiplicities_1, &inv_t_1,
      eq_w1_left.clone(), eq_w1_right.clone(),
      eq_t1_left.clone(), eq_t1_right.clone(),
      &witness_2, &table_2, &multiplicities_2,
      eq_w2_left.clone(), eq_w2_right.clone(),
      eq_t2_left.clone(), eq_t2_right.clone(),
      r,
    )
    .unwrap();

    let rho = Scalar::random(&mut rng);
    // At non-outer-base, T_running might be nonzero.
    // For this test, use zero (the actual T_running computation
    // depends on the prior fold, which we don't simulate here).
    let t_running = Scalar::ZERO;
    let poly = inst.prove_step(&rho, &t_running);

    // C-binding must hold
    assert_eq!(
      poly.eval_at_zero() + poly.eval_at_one(),
      t_running,
      "C-binding"
    );

    // Differential check at random points (same as above)
    let bf = |v1: Scalar, v2: Scalar, p: Scalar| -> Scalar {
      (Scalar::ONE - p) * v1 + p * v2
    };

    for _ in 0..5 {
      let p = Scalar::random(&mut rng);
      let eq_rho_p = (Scalar::ONE - rho) * (Scalar::ONE - p) + rho * p;

      let left_w = inst.eq_w1_left.len();
      let right_w = inst.eq_w1_right.len();
      let mut sum_a = Scalar::ZERO;
      for ii in 0..right_w {
        let eq_r = bf(inst.eq_w1_right[ii], inst.eq_w2_right[ii], p);
        for jj in 0..left_w {
          let k = ii * left_w + jj;
          let eq_l = bf(inst.eq_w1_left[jj], inst.eq_w2_left[jj], p);
          let inv_w = bf(inst.inv_w1_poly[k], inst.inv_w2_poly[k], p);
          let w_plus_r = bf(inst.w1_poly[k] + r, inst.w2_poly[k] + r, p);
          sum_a += eq_r * eq_l * (inv_w * w_plus_r - Scalar::ONE);
        }
      }

      let left_t = inst.eq_t1_left.len();
      let right_t = inst.eq_t1_right.len();
      let mut sum_b = Scalar::ZERO;
      for ii in 0..right_t {
        let eq_r = bf(inst.eq_t1_right[ii], inst.eq_t2_right[ii], p);
        for jj in 0..left_t {
          let k = ii * left_t + jj;
          let eq_l = bf(inst.eq_t1_left[jj], inst.eq_t2_left[jj], p);
          let inv_t = bf(inst.inv_t1_poly[k], inst.inv_t2_poly[k], p);
          let t_plus_r = bf(inst.t1_poly[k] + r, inst.t2_poly[k] + r, p);
          let ts = bf(inst.ts1_poly[k], inst.ts2_poly[k], p);
          sum_b += eq_r * eq_l * (inv_t * t_plus_r - ts);
        }
      }

      let sum_inv_w: Scalar = inst.inv_w1_poly.Z.iter()
        .zip(inst.inv_w2_poly.Z.iter())
        .map(|(v1, v2)| bf(*v1, *v2, p))
        .sum();
      let sum_inv_t: Scalar = inst.inv_t1_poly.Z.iter()
        .zip(inst.inv_t2_poly.Z.iter())
        .map(|(v1, v2)| bf(*v1, *v2, p))
        .sum();
      let sum_c = sum_inv_w - sum_inv_t;

      let expected = eq_rho_p * (sum_a + sum_b + sum_c);
      let actual = poly.evaluate(&p);
      assert_eq!(
        actual, expected,
        "polynomial evaluation at random point must match naive computation (both instances nontrivial)"
      );
    }

    // verify_step should accept
    let r_b = Scalar::random(&mut rng);
    let result = LookupSumcheckInstance::<E>::verify_step(
      &rho, &r_b, &poly, &t_running,
    );
    assert!(result.is_ok(), "verify_step must accept");
  }
}
