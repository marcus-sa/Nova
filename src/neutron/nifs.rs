//! This module implements a non-interactive folding scheme from NeutronNova.
//!
//! ## Fiat-Shamir transcript pin (Stage I-pri)
//!
//! Three transcript variants live in this module. They share a common
//! head (R1CS-only fold) and diverge for the lookup-fold extension. The
//! transcript byte-stream MUST be byte-identical across native prover,
//! native verifier, and in-circuit verifier (`AllocatedNIFS::verify_*`)
//! at the token level. Future audit firms will compare the three
//! implementations against this pin.
//!
//! ### Variant 1 — `prove` / `verify` (no lookup)
//!
//! ```text
//!   ro.absorb(pp_digest)
//!   U2.absorb_in_ro2
//!   ro.squeeze() -> tau
//!   comm_E.absorb_in_ro2
//!   ro.squeeze() -> rho
//!   poly.absorb_in_ro2
//!   ro.squeeze() -> r_b
//! ```
//!
//! ### Variant 2 — `prove_with_lookup` / `verify_with_lookup` (Stage D/E,
//! single-column lookup)
//!
//! ```text
//!   ro.absorb(pp_digest)
//!   U2.absorb_in_ro2
//!   payload.comm_L.absorb_in_ro2          [single witness column = address ≡ value]
//!   payload.comm_ts.absorb_in_ro2
//!   ro.squeeze() -> tau
//!   comm_E.absorb_in_ro2
//!   ro.squeeze() -> rho
//!   ro.squeeze() -> r_logup
//!   comm_inv_w.absorb_in_ro2
//!   comm_inv_t.absorb_in_ro2
//!   poly.absorb_in_ro2                    [R1CS sumcheck poly]
//!   poly_lookup.absorb_in_ro2             [lookup sumcheck poly]
//!   ro.squeeze() -> r_b
//! ```
//!
//! ### Variant 3 — `prove_with_multi_column_lookup` /
//! `verify_with_multi_column_lookup` (Stage I-pri, multi-column lookup
//! via Lasso §6.2 address-value combine)
//!
//! ```text
//!   ro.absorb(pp_digest)
//!   U2.absorb_in_ro2
//!   payload.comm_L.absorb_in_ro2          [address column]
//!   for cv in payload.comm_values:        [Stage I-pri NEW: value columns,
//!     cv.absorb_in_ro2                     in declaration order]
//!   payload.comm_ts.absorb_in_ro2
//!   ro.squeeze() -> tau
//!   comm_E.absorb_in_ro2
//!   ro.squeeze() -> rho
//!   if !payload.comm_values.is_empty():
//!     ro.squeeze() -> α                   [Stage I-pri NEW: combine challenge,
//!                                          bound to all column commitments]
//!   ro.squeeze() -> r_logup
//!   comm_inv_w.absorb_in_ro2
//!   comm_inv_t.absorb_in_ro2
//!   poly.absorb_in_ro2
//!   poly_lookup.absorb_in_ro2
//!   ro.squeeze() -> r_b
//! ```
//!
//! ### Stage H byte-equivalence pin
//!
//! When `payload.comm_values.is_empty()`, Variant 3 reduces to Variant 2
//! BYTE-FOR-BYTE: no value-column absorptions occur, no α is squeezed,
//! and the combined witness/table degenerate to the single-column case
//! (`combined_W = address`, `combined_T = (0, 1, ..., size-1)`). This
//! is the load-bearing backward-compatibility property — the Stage H
//! `range_check_via_lookup` primitive routes through Variant 3 with an
//! empty value-column vector and produces identical proofs.
//!
//! ### α positioning rationale
//!
//! α is squeezed AFTER `rho` (and therefore after `comm_E`) and BEFORE
//! `r_logup`. This binds α to:
//!
//! 1. All per-step lookup column commitments (`comm_L` + every
//!    `comm_values[i]` + `comm_ts`) — required by the Lasso §6.2
//!    soundness reduction (`(addr, v₁, ..., v_c) ∉ {(i, T₁, ...)}_i`
//!    ⇒ combined witness ∉ combined table except w.h.p. over α).
//! 2. The R1CS-side eq commitment (`comm_E`) and challenge (ρ) — so
//!    α cannot be reused to attack a different R1CS-side state.
//!
//! Squeezing α BEFORE `r_logup` is the load-bearing pin: `r_logup` is
//! used to construct LogUp inverses against the combined witness. If α
//! were squeezed after `r_logup`, a malicious prover could choose
//! witness values targeting a known r_logup, then have α happen to
//! collide them. With α bound first, the combined witness/table are
//! determined before LogUp randomness arrives.
#![allow(non_snake_case)]
use crate::{
  constants::NUM_CHALLENGE_BITS,
  errors::NovaError,
  neutron::relation::{FoldedInstance, FoldedWitness, Structure},
  r1cs::{R1CSInstance, R1CSWitness},
  spartan::polys::{power::PowPolynomial, univariate::UniPoly},
  traits::{commitment::CommitmentEngineTrait, AbsorbInRO2Trait, Engine, RO2Constants, ROTrait},
  Commitment, CommitmentKey, CE,
};
#[cfg(feature = "lookup-fold")]
use crate::neutron::{
  lookup_sumcheck::{lookup_running_claims_from, LookupSumcheckInstance},
  relation::{LookupFreshWitness, LookupPayload, LookupRunningWitness},
};
use ff::Field;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Lasso §6.2 address-value combine — witness side (Stage I-pri).
///
/// Returns `out[k] = address[k] + α·v₁[k] + α²·v₂[k] + ... + α^c·v_c[k]`.
/// When `value_columns.is_empty()`, returns `address.to_vec()` (the
/// single-column degenerate path).
///
/// Pinned by Stage I-pri §I.4 of the implementation outline.
#[cfg(feature = "lookup-fold")]
pub(crate) fn combine_columns_witness<E: Engine>(
  address: &[E::Scalar],
  value_columns: &[Vec<E::Scalar>],
  alpha: &E::Scalar,
) -> Vec<E::Scalar> {
  if value_columns.is_empty() {
    return address.to_vec();
  }
  let n = address.len();
  let mut out = address.to_vec();
  // Horner-style accumulation to avoid recomputing α^i:
  //   out[k] = address[k] + α·(v₁[k] + α·(v₂[k] + ...))
  // implemented by iterating alpha_pow = α, α², ... and adding alpha_pow * v_i.
  let mut alpha_pow = *alpha;
  for col in value_columns {
    debug_assert_eq!(col.len(), n);
    for (k, o) in out.iter_mut().enumerate() {
      *o += alpha_pow * col[k];
    }
    alpha_pow *= *alpha;
  }
  out
}

/// Lasso §6.2 address-value combine — table side (Stage I-pri).
///
/// Returns `out[i] = i + α·T₁[i] + α²·T₂[i] + ... + α^c·T_c[i]` for
/// `i ∈ [0, table_size)`. When `table_columns.is_empty()`, returns
/// the identity vector `(0, 1, ..., table_size-1)` — i.e. the table
/// is the identity table (the range-check special case).
///
/// Pinned by Stage I-pri §I.4 of the implementation outline.
#[cfg(feature = "lookup-fold")]
pub(crate) fn combine_columns_table<E: Engine>(
  table_size: usize,
  table_columns: &[Vec<E::Scalar>],
  alpha: &E::Scalar,
) -> Vec<E::Scalar> {
  let mut out: Vec<E::Scalar> = (0..table_size)
    .map(|i| E::Scalar::from(i as u64))
    .collect();
  if table_columns.is_empty() {
    return out;
  }
  let mut alpha_pow = *alpha;
  for col in table_columns {
    debug_assert_eq!(col.len(), table_size);
    for (i, o) in out.iter_mut().enumerate() {
      *o += alpha_pow * col[i];
    }
    alpha_pow *= *alpha;
  }
  out
}

/// Prover-side per-table bundle for the multi-table fold path
/// (GH-#2 M.3, design pin §5.1).
///
/// One entry per table registered in
/// `Structure::lookups.multi_column_tables`, in `table_id`-canonical
/// order. The prover supplies the bundles in that order; the
/// `prove_with_multi_table_lookup` inner debug-asserts both the count
/// and the ordering.
///
/// For "absent" tables at a given fold step (queries did not touch the
/// table this step), the bundle carries:
/// - `payload`: zero-payload commitments per pin §1.5.3
///   (`comm_L = commit(&[0; n_j], 0)`, `comm_values[i] = commit(&[0; n_j], 0)`,
///   `comm_ts = commit(&[0; n_j], 0)`).
/// - `fresh_witness_address`, `fresh_witness_value_columns[*]`,
///   `fresh_multiplicities`: all-zero vectors of length `n_j`.
/// - `fresh_eq_w_*`, `fresh_eq_t_*`: derived from the step's tau (the
///   step has SOME R1CS witness regardless of whether T_j is queried).
/// - `running_lw`: the per-table running witness; at outer base this is
///   `LookupRunningWitness::default(...)`, mid-fold it's the prior
///   step's folded running witness for table T_j.
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug)]
pub struct PerTableBundle<E: Engine> {
  /// Stable identifier for the table (must match the corresponding
  /// `MultiColumnLookupTable::table_id` registered in the structure).
  pub table_id: u64,
  /// Per-step lookup-side commitment payload for this table.
  pub payload: LookupPayload<E>,
  /// Per-step fresh witness address column for this table (length `n_j`).
  pub fresh_witness_address: Vec<E::Scalar>,
  /// Per-step fresh witness value columns for this table (parallel to
  /// `payload.comm_values`; each of length `n_j`).
  pub fresh_witness_value_columns: Vec<Vec<E::Scalar>>,
  /// Per-step fresh multiplicities for this table (length `n_j`).
  pub fresh_multiplicities: Vec<E::Scalar>,
  /// Per-step witness-side eq vector, left half (split-tensor form).
  pub fresh_eq_w_left: Vec<E::Scalar>,
  /// Per-step witness-side eq vector, right half.
  pub fresh_eq_w_right: Vec<E::Scalar>,
  /// Per-step table-side eq vector, left half.
  pub fresh_eq_t_left: Vec<E::Scalar>,
  /// Per-step table-side eq vector, right half.
  pub fresh_eq_t_right: Vec<E::Scalar>,
  /// Per-table running lookup witness (carried across fold steps).
  pub running_lw: LookupRunningWitness<E>,
}

/// An NIFS message from NeutronNova's folding scheme
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E: Engine> {
  pub(crate) comm_E: Commitment<E>,
  pub(crate) poly: UniPoly<E::Scalar>,

  /// Lookup-fold extension fields (Stage C, C1-beta).
  /// Present only when a lookup payload was supplied to the fold step.
  ///
  /// Multi-table extension (GH-#2, design pin §5.1): widened from
  /// `Option<UniPoly<…>>` / `Option<Commitment<E>>` to `Option<Vec<…>>` so
  /// the multi-table prove path can carry one polynomial / commitment per
  /// registered table in `table_id`-canonical order. Single-table use
  /// (`prove_with_lookup`, `prove_with_multi_column_lookup`) populates a
  /// one-element Vec — the FS transcript byte stream remains identical to
  /// pre-#2 by §5.2 #2.
  #[cfg(feature = "lookup-fold")]
  pub(crate) poly_lookup: Option<Vec<UniPoly<E::Scalar>>>,
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_w: Option<Vec<Commitment<E>>>,
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_t: Option<Vec<Commitment<E>>>,
}

impl<E: Engine> NIFS<E> {
  /// Computes the evaluations of the sum-check polynomial at 0, 2, 3, and 4
  #[inline]
  fn prove_helper(
    rho: &E::Scalar,
    (left, right): (usize, usize),
    e1: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Cz1: &[E::Scalar],
    e2: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
    Cz2: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) {
    // sanity check sizes
    assert_eq!(e1.len(), left + right);
    assert_eq!(Az1.len(), left * right);
    assert_eq!(Bz1.len(), left * right);
    assert_eq!(Cz1.len(), left * right);
    assert_eq!(e2.len(), left + right);
    assert_eq!(Az2.len(), left * right);
    assert_eq!(Bz2.len(), left * right);
    assert_eq!(Cz2.len(), left * right);

    let comb_func = |c1: &E::Scalar, c2: &E::Scalar, c3: &E::Scalar, c4: &E::Scalar| -> E::Scalar {
      *c1 * (*c2 * *c3 - *c4)
    };
    let (eval_at_0, eval_at_2, eval_at_3, eval_at_4, eval_at_5) = (0..right)
      .into_par_iter()
      .map(|i| {
        let (i_eval_at_0, i_eval_at_2, i_eval_at_3, i_eval_at_4, i_eval_at_5) = (0..left)
          .into_par_iter()
          .map(|j| {
            // Turn the two dimensional (i, j) into a single dimension index
            let k = i * left + j;

            // eval 0: bound_func is A(low)
            let eval_point_0 = comb_func(&e1[j], &Az1[k], &Bz1[k], &Cz1[k]);

            // eval 2: bound_func is -A(low) + 2*A(high)
            let poly_e_bound_point = e2[j] + e2[j] - e1[j];
            let poly_Az_bound_point = Az2[k] + Az2[k] - Az1[k];
            let poly_Bz_bound_point = Bz2[k] + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = Cz2[k] + Cz2[k] - Cz1[k];
            let eval_point_2 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
            let poly_e_bound_point = poly_e_bound_point + e2[j] - e1[j];
            let poly_Az_bound_point = poly_Az_bound_point + Az2[k] - Az1[k];
            let poly_Bz_bound_point = poly_Bz_bound_point + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = poly_Cz_bound_point + Cz2[k] - Cz1[k];
            let eval_point_3 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            // eval 4: bound_func is -3A(low) + 4A(high); computed incrementally with bound_func applied to eval(3)
            let poly_e_bound_point = poly_e_bound_point + e2[j] - e1[j];
            let poly_Az_bound_point = poly_Az_bound_point + Az2[k] - Az1[k];
            let poly_Bz_bound_point = poly_Bz_bound_point + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = poly_Cz_bound_point + Cz2[k] - Cz1[k];
            let eval_point_4 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            // eval 5: bound_func is -4A(low) + 5A(high); computed incrementally with bound_func applied to eval(4)
            let poly_e_bound_point = poly_e_bound_point + e2[j] - e1[j];
            let poly_Az_bound_point = poly_Az_bound_point + Az2[k] - Az1[k];
            let poly_Bz_bound_point = poly_Bz_bound_point + Bz2[k] - Bz1[k];
            let poly_Cz_bound_point = poly_Cz_bound_point + Cz2[k] - Cz1[k];
            let eval_point_5 = comb_func(
              &poly_e_bound_point,
              &poly_Az_bound_point,
              &poly_Bz_bound_point,
              &poly_Cz_bound_point,
            );

            (
              eval_point_0,
              eval_point_2,
              eval_point_3,
              eval_point_4,
              eval_point_5,
            )
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

        let f1 = &e1[left..];
        let f2 = &e2[left..];

        // eval 0: bound_func is A(low)
        let eval_at_0 = f1[i] * i_eval_at_0;

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_f_bound_point = f2[i] + f2[i] - f1[i];
        let eval_at_2 = poly_f_bound_point * i_eval_at_2;

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let poly_f_bound_point = poly_f_bound_point + f2[i] - f1[i];
        let eval_at_3 = poly_f_bound_point * i_eval_at_3;

        // eval 4: bound_func is -3A(low) + 4A(high); computed incrementally with bound_func applied to eval(3)
        let poly_f_bound_point = poly_f_bound_point + f2[i] - f1[i];
        let eval_at_4 = poly_f_bound_point * i_eval_at_4;

        // eval 5: bound_func is -4A(low) + 5A(high); computed incrementally with bound_func applied to eval(4)
        let poly_f_bound_point = poly_f_bound_point + f2[i] - f1[i];
        let eval_at_5 = poly_f_bound_point * i_eval_at_5;

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
      );

    // multiply by the common factors
    let one_minus_rho = E::Scalar::ONE - rho;
    let three_rho_minus_one = E::Scalar::from(3) * rho - E::Scalar::ONE;
    let five_rho_minus_two = E::Scalar::from(5) * rho - E::Scalar::from(2);
    let seven_rho_minus_three = E::Scalar::from(7) * rho - E::Scalar::from(3);
    let nine_rho_minus_four = E::Scalar::from(9) * rho - E::Scalar::from(4);

    (
      eval_at_0 * one_minus_rho,
      eval_at_2 * three_rho_minus_one,
      eval_at_3 * five_rho_minus_two,
      eval_at_4 * seven_rho_minus_three,
      eval_at_5 * nine_rho_minus_four,
    )
  }

  /// Takes as input a folded instance-witness tuple `(U1, W1)` and
  /// an R1CS instance-witness tuple `(U2, W2)` with a compatible structure `shape`
  /// and defined with respect to the same `ck`, and outputs
  /// a folded instance-witness tuple `(U, W)` of the same shape `shape`,
  /// with the guarantee that the folded witness `W` satisfies the folded instance `U`
  /// if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  ///
  /// Note that this code is tailored for use with NeutronNova's IVC scheme, which enforces
  /// certain requirements between the two instances that are folded.
  /// In particular, it requires that `U1` and `U2` are such that the hash of `U1` is stored in the public IO of `U2`.
  /// In this particular setting, this means that if `U2` is absorbed in the RO, it implicitly absorbs `U1` as well.
  /// So the code below avoids absorbing `U1` in the RO.
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    W1: &FoldedWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) -> Result<(NIFS<E>, (FoldedInstance<E>, FoldedWitness<E>)), NovaError> {
    // initialize a new RO
    let mut ro = E::RO2::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(*pp_digest);

    // append U2 to transcript
    U2.absorb_in_ro2(&mut ro);

    // generate a challenge for the eq polynomial
    let tau = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute a commitment to the eq polynomial
    let E = PowPolynomial::new(&tau, S.ell).split_evals(S.left, S.right);
    let r_E = E::Scalar::random(&mut OsRng);
    let comm_E = CE::<E>::commit(ck, &E, &r_E);

    comm_E.absorb_in_ro2(&mut ro); // absorb the commitment in the NIFS

    // compute a challenge from the RO
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // We now run a single round of the sum-check protocol to establish
    // T = (1-rho) * T1 + rho * T2, where T1 comes from the running instance and T2 = 0
    let T = (E::Scalar::ONE - rho) * U1.T;

    let (res1, res2) = rayon::join(
      || {
        let z1 = [W1.W.clone(), vec![U1.u], U1.X.clone()].concat();
        S.S.multiply_vec(&z1)
      },
      || {
        let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
        S.S.multiply_vec(&z2)
      },
    );

    let (Az1, Bz1, Cz1) = res1?;
    let (Az2, Bz2, Cz2) = res2?;

    // compute the sum-check polynomial's evaluations at 0, 2, 3
    let (eval_point_0, eval_point_2, eval_point_3, eval_point_4, eval_point_5) = Self::prove_helper(
      &rho,
      (S.left, S.right),
      &W1.E,
      &Az1,
      &Bz1,
      &Cz1,
      &E,
      &Az2,
      &Bz2,
      &Cz2,
    );

    let evals = vec![
      eval_point_0,
      T - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
      eval_point_5,
    ];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals);

    // absorb poly in the RO
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly, &mut ro);

    // squeeze a challenge
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap(); // TODO: remove unwrap

    let U = U1.fold(U2, &comm_E, &r_b, &T_out)?;
    let W = W1.fold(W2, &E, &r_E, &r_b)?;

    // return the folded instance and witness
    Ok((
      Self {
        comm_E,
        poly,
        #[cfg(feature = "lookup-fold")]
        poly_lookup: None,
        #[cfg(feature = "lookup-fold")]
        comm_inv_w: None,
        #[cfg(feature = "lookup-fold")]
        comm_inv_t: None,
      },
      (U, W),
    ))
  }

  /// Takes as input a relaxed R1CS instance `U1` and R1CS instance `U2`
  /// with the same shape and defined with respect to the same parameters,
  /// and outputs a folded instance `U` with the same shape,
  /// with the guarantee that the folded instance `U`
  /// if and only if `U1` and `U2` are satisfiable.
  #[cfg(test)]
  pub fn verify(
    &self,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    U1: &FoldedInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<FoldedInstance<E>, NovaError> {
    // initialize a new RO
    let mut ro = E::RO2::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(*pp_digest);

    // append U2 to transcript
    U2.absorb_in_ro2(&mut ro);

    // generate a challenge for the eq polynomial
    let _tau = ro.squeeze(NUM_CHALLENGE_BITS, false);

    self.comm_E.absorb_in_ro2(&mut ro); // absorb the commitment in the NIFS

    // compute a challenge from the RO
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // T = (1-rho) * T1 + rho * T2, where T1 comes from the running instance and T2 = 0
    let T = (E::Scalar::ONE - rho) * U1.T;

    // check if poly(0) + poly(1) = T
    if self.poly.eval_at_zero() + self.poly.eval_at_one() != T {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // absorb poly in the RO
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&self.poly, &mut ro);

    // squeeze a challenge
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap(); // TODO: remove unwrap

    let U = U1.fold(U2, &self.comm_E, &r_b, &T_out)?;

    // return the folded instance and witness
    Ok(U)
  }

  /// Prove a fold step with lookup data (Stages D, C1-beta).
  ///
  /// Extends the base `prove` with lookup-side FS transcript absorptions:
  ///
  /// 1. (Step 2) Absorb `payload.comm_L` and `payload.comm_ts` BEFORE tau squeeze
  /// 2. (Step 5) After rho squeeze, squeeze `r_logup`
  /// 3. (Step 6) Construct `LookupSumcheckInstance` with dual-instance data,
  ///    absorb `comm_inv_w`, `comm_inv_t`
  /// 4. (Step 7) Call `prove_step`, absorb `poly_lookup` AFTER R1CS `poly`
  /// 5. After `r_b` squeeze, compute `T_lookup_out`
  /// 6. Fold using `fold_with_lookup`
  ///
  /// Returns the NIFS proof, the folded instance+witness pair, the folded
  /// lookup running witness, and the fresh witness data (with computed
  /// inverses populated for downstream use).
  #[cfg(feature = "lookup-fold")]
  #[allow(clippy::too_many_arguments)]
  pub fn prove_with_lookup(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    W1: &FoldedWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    payload: &LookupPayload<E>,
    running_lw: &LookupRunningWitness<E>,
    fresh_witness: &[E::Scalar],
    fresh_table: &[E::Scalar],
    fresh_multiplicities: &[E::Scalar],
    fresh_eq_w_left: Vec<E::Scalar>,
    fresh_eq_w_right: Vec<E::Scalar>,
    fresh_eq_t_left: Vec<E::Scalar>,
    fresh_eq_t_right: Vec<E::Scalar>,
  ) -> Result<
    (
      NIFS<E>,
      (FoldedInstance<E>, FoldedWitness<E>),
      LookupRunningWitness<E>,
    ),
    NovaError,
  > {
    // initialize a new RO
    let mut ro = E::RO2::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(*pp_digest);

    // append U2 to transcript
    U2.absorb_in_ro2(&mut ro);

    // --- Step (2): absorb lookup commitments BEFORE tau squeeze ---
    payload.comm_L.absorb_in_ro2(&mut ro);
    payload.comm_ts.absorb_in_ro2(&mut ro);

    // generate a challenge for the eq polynomial
    let tau = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute a commitment to the eq polynomial
    let E = PowPolynomial::new(&tau, S.ell).split_evals(S.left, S.right);
    let r_E = E::Scalar::random(&mut OsRng);
    let comm_E = CE::<E>::commit(ck, &E, &r_E);

    comm_E.absorb_in_ro2(&mut ro);

    // --- Step (4): squeeze rho ---
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Step (5): squeeze r_logup ---
    let r_logup = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Step (6): Construct LookupSumcheckInstance ---
    let (lookup_inst, comm_inv_w2, comm_inv_t2) = LookupSumcheckInstance::<E>::new(
      ck,
      // U1 (running) data
      &running_lw.witness,
      &running_lw.inv_w,
      &running_lw.table,
      &running_lw.multiplicities,
      &running_lw.inv_t,
      running_lw.eq_w_left.clone(),
      running_lw.eq_w_right.clone(),
      running_lw.eq_t_left.clone(),
      running_lw.eq_t_right.clone(),
      // U2 (fresh) data
      fresh_witness,
      fresh_table,
      fresh_multiplicities,
      fresh_eq_w_left.clone(),
      fresh_eq_w_right.clone(),
      fresh_eq_t_left.clone(),
      fresh_eq_t_right.clone(),
      // shared
      r_logup,
    )?;

    // Absorb the inverse commitments
    comm_inv_w2.absorb_in_ro2(&mut ro);
    comm_inv_t2.absorb_in_ro2(&mut ro);

    // --- R1CS-side sumcheck (same as base prove) ---
    let T = (E::Scalar::ONE - rho) * U1.T;

    let (res1, res2) = rayon::join(
      || {
        let z1 = [W1.W.clone(), vec![U1.u], U1.X.clone()].concat();
        S.S.multiply_vec(&z1)
      },
      || {
        let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
        S.S.multiply_vec(&z2)
      },
    );

    let (Az1, Bz1, Cz1) = res1?;
    let (Az2, Bz2, Cz2) = res2?;

    let (eval_point_0, eval_point_2, eval_point_3, eval_point_4, eval_point_5) =
      Self::prove_helper(
        &rho,
        (S.left, S.right),
        &W1.E,
        &Az1,
        &Bz1,
        &Cz1,
        &E,
        &Az2,
        &Bz2,
        &Cz2,
      );

    let evals = vec![
      eval_point_0,
      T - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
      eval_point_5,
    ];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals);

    // absorb R1CS poly in the RO
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly, &mut ro);

    // --- Step (7): lookup sumcheck ---
    // Multi-table extension (GH-#2, design pin §5.1): `lookup_running_claims_from`
    // returns `Vec<E::Scalar>`. Single-table (`prove_with_lookup`) reads
    // entry [0]; outer-base empty Vec falls back to ZERO. FS transcript
    // unchanged (§5.2 #2).
    let t_lookup_running_vec = lookup_running_claims_from::<E>(U1);
    let t_lookup_running = t_lookup_running_vec
      .first()
      .copied()
      .unwrap_or(E::Scalar::ZERO);
    let poly_lookup = lookup_inst.prove_step(&rho, &t_lookup_running);

    // absorb lookup poly AFTER R1CS poly
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly_lookup, &mut ro);

    // squeeze r_b
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Compute R1CS T_out ---
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();

    // --- Compute T_lookup_out ---
    let T_lookup_out =
      LookupSumcheckInstance::<E>::verify_step(&rho, &r_b, &poly_lookup, &t_lookup_running)?;

    // --- Fold with lookup ---
    // Build the effective payload with the freshly computed inverse commitments
    // from LookupSumcheckInstance::new (the caller's payload may have placeholders).
    let effective_payload = LookupPayload {
      comm_L: payload.comm_L,
      comm_ts: payload.comm_ts,
      comm_inv_w: comm_inv_w2,
      comm_inv_t: comm_inv_t2,
      T2_lookup: payload.T2_lookup,
      comm_values: payload.comm_values.clone(),
    };
    // Multi-table extension (GH-#2 M.2): `fold_with_lookup` takes a slice
    // of per-table running scalars. The single-table caller passes a
    // one-element slice; storage shape is byte-identical to M.1.
    let U = U1.fold_with_lookup(
      U2,
      &comm_E,
      &r_b,
      &T_out,
      &effective_payload,
      std::slice::from_ref(&T_lookup_out),
    )?;
    let W = W1.fold(W2, &E, &r_E, &r_b)?;

    // --- Fold the lookup running witness ---
    // We need the fresh inverse witnesses that LookupSumcheckInstance computed.
    // Reconstruct them from the lookup instance's internal state via
    // batch_invert_plus_r (same computation).
    let fresh_inv_w =
      crate::spartan::logup_inverses::batch_invert_plus_r(fresh_witness, &r_logup)?;
    let fresh_inv_t_raw =
      crate::spartan::logup_inverses::batch_invert_plus_r(fresh_table, &r_logup)?;
    let fresh_inv_t: Vec<E::Scalar> = fresh_inv_t_raw
      .iter()
      .zip(fresh_multiplicities.iter())
      .map(|(inv, ts)| *inv * *ts)
      .collect();

    let fresh_lw = LookupFreshWitness {
      witness: fresh_witness.to_vec(),
      inv_w: fresh_inv_w,
      table: fresh_table.to_vec(),
      multiplicities: fresh_multiplicities.to_vec(),
      inv_t: fresh_inv_t,
      eq_w_left: fresh_eq_w_left,
      eq_w_right: fresh_eq_w_right,
      eq_t_left: fresh_eq_t_left,
      eq_t_right: fresh_eq_t_right,
    };
    let folded_lw = running_lw.fold(&fresh_lw, &r_b);

    // Multi-table extension (GH-#2, design pin §5.1): `poly_lookup`,
    // `comm_inv_w`, `comm_inv_t` are now `Option<Vec<…>>`. Single-table
    // (`prove_with_lookup`) wraps the single value in a one-element Vec.
    // FS transcript byte stream is unchanged — only the in-memory NIFS
    // envelope grows by a Vec length prefix.
    let nifs = NIFS {
      comm_E,
      poly,
      poly_lookup: Some(vec![poly_lookup]),
      comm_inv_w: Some(vec![comm_inv_w2]),
      comm_inv_t: Some(vec![comm_inv_t2]),
    };

    Ok((nifs, (U, W), folded_lw))
  }

  /// Prove a fold step with **multi-column** lookup data (Stage I-pri,
  /// Lasso §6.2 address-value combine).
  ///
  /// Extends [`Self::prove_with_lookup`] with the multi-column transcript
  /// pin:
  ///
  /// 1. (Step 2) Absorb `payload.comm_L` (address column), then each
  ///    `payload.comm_values[i]` (value columns), then `payload.comm_ts`
  ///    BEFORE the tau squeeze. The per-column commitments are bound to
  ///    α before α is sampled.
  /// 2. (Step 5) After `rho` squeeze and BEFORE `r_logup` squeeze:
  ///    squeeze α. This binds α to all column commitments and to ρ
  ///    (which is itself bound to the R1CS-side eq commitment).
  /// 3. Compute the combined witness/table off-circuit:
  ///    `W_combined[k] = address[k] + α·v₁[k] + ... + α^c·v_c[k]`,
  ///    `T_combined[i] = i + α·T₁[i] + ... + α^c·T_c[i]`. The address
  ///    column on the table side is the implicit identity vector
  ///    `(0, 1, ..., size-1)`.
  /// 4. Pass the combined data to `LookupSumcheckInstance::new` (Stage B
  ///    algebra, single-column shape, **unchanged**). The Stage B
  ///    degree-5 dual-instance algebra is preserved — only the witness
  ///    pre-processing changes.
  /// 5. The rest of the path is identical to [`Self::prove_with_lookup`].
  ///
  /// ## Arguments
  ///
  /// - `table_id`: identifier of the multi-column table registered in
  ///   `S.lookups.multi_column_tables` (Stage I-app.2). The table data
  ///   (`size`, `columns`) is resolved from the structure at call time —
  ///   the table identity is bound globally via `pp_digest`, NOT passed
  ///   per-call.
  /// - `fresh_witness_address`: the U2 address column (length = pooled
  ///   query count).
  /// - `fresh_witness_value_columns`: the U2 value columns (one Vec per
  ///   column, parallel to `payload.comm_values`). Must all have the
  ///   same length as `fresh_witness_address`. May be empty (degenerates
  ///   to single-column path).
  /// - `fresh_multiplicities`, `fresh_eq_*`: as in `prove_with_lookup`.
  ///
  /// Returns `(nifs, (folded_U, folded_W), folded_lw)` exactly like
  /// `prove_with_lookup`. The transcript byte-stream is identical to
  /// `prove_with_lookup` when `fresh_witness_value_columns.is_empty()`.
  ///
  /// ## Errors
  ///
  /// Returns [`NovaError::InvalidStructure`] if `table_id` is not registered
  /// in `S.lookups.multi_column_tables` (Stage I-app.2 pp_digest binding).
  #[cfg(feature = "lookup-fold")]
  #[allow(clippy::too_many_arguments)]
  pub fn prove_with_multi_column_lookup(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    W1: &FoldedWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    payload: &LookupPayload<E>,
    running_lw: &LookupRunningWitness<E>,
    table_id: u64,
    fresh_witness_address: &[E::Scalar],
    fresh_witness_value_columns: &[Vec<E::Scalar>],
    fresh_multiplicities: &[E::Scalar],
    fresh_eq_w_left: Vec<E::Scalar>,
    fresh_eq_w_right: Vec<E::Scalar>,
    fresh_eq_t_left: Vec<E::Scalar>,
    fresh_eq_t_right: Vec<E::Scalar>,
  ) -> Result<
    (
      NIFS<E>,
      (FoldedInstance<E>, FoldedWitness<E>),
      LookupRunningWitness<E>,
    ),
    NovaError,
  > {
    // GH-#2 M.3: thin wrapper that draws the eq-polynomial blinding `r_E`
    // from `OsRng` and delegates to the deterministic-`r_E` inner. The
    // inner is also called directly by the M.3 byte-equivalence regression
    // test (pin §5.2 #2) under a seeded `r_E` so the multi-column k=1 and
    // multi-table k=1 paths can be compared at FS-transcript layer with
    // matching commitment outputs.
    let r_E = E::Scalar::random(&mut OsRng);
    Self::prove_with_multi_column_lookup_inner(
      ck,
      ro_consts,
      pp_digest,
      S,
      U1,
      W1,
      U2,
      W2,
      payload,
      running_lw,
      table_id,
      fresh_witness_address,
      fresh_witness_value_columns,
      fresh_multiplicities,
      fresh_eq_w_left,
      fresh_eq_w_right,
      fresh_eq_t_left,
      fresh_eq_t_right,
      r_E,
    )
  }

  /// Inner of [`Self::prove_with_multi_column_lookup`]: takes the
  /// eq-polynomial blinding scalar `r_E` as an explicit parameter so a
  /// test can pin determinism. Production callers should use the public
  /// wrapper, which sources `r_E` from `OsRng`.
  #[cfg(feature = "lookup-fold")]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn prove_with_multi_column_lookup_inner(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    W1: &FoldedWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    payload: &LookupPayload<E>,
    running_lw: &LookupRunningWitness<E>,
    table_id: u64,
    fresh_witness_address: &[E::Scalar],
    fresh_witness_value_columns: &[Vec<E::Scalar>],
    fresh_multiplicities: &[E::Scalar],
    fresh_eq_w_left: Vec<E::Scalar>,
    fresh_eq_w_right: Vec<E::Scalar>,
    fresh_eq_t_left: Vec<E::Scalar>,
    fresh_eq_t_right: Vec<E::Scalar>,
    r_E: E::Scalar,
  ) -> Result<
    (
      NIFS<E>,
      (FoldedInstance<E>, FoldedWitness<E>),
      LookupRunningWitness<E>,
    ),
    NovaError,
  > {
    // Stage I-app.2: resolve the multi-column table from the structure-side
    // registry. The table contents are pinned by `pp_digest` (via serde on
    // `Structure → LookupShape → multi_column_tables`), so the table
    // identity cannot drift across fold steps.
    let lookup_shape = S
      .lookups
      .as_ref()
      .ok_or(NovaError::InvalidStructure)?;
    let multi_table = lookup_shape
      .multi_column_tables
      .iter()
      .find(|t| t.table_id == table_id)
      .ok_or(NovaError::InvalidStructure)?;
    let fresh_table_size = multi_table.size;
    let fresh_table_value_columns: &[Vec<E::Scalar>] = &multi_table.columns;

    // Sanity: column counts must agree across payload, witness, and table.
    let c = payload.comm_values.len();
    if fresh_witness_value_columns.len() != c {
      return Err(NovaError::UnSat {
        reason: format!(
          "prove_with_multi_column_lookup: witness column count ({}) \
           differs from payload.comm_values ({})",
          fresh_witness_value_columns.len(),
          c,
        ),
      });
    }
    if fresh_table_value_columns.len() != c {
      return Err(NovaError::UnSat {
        reason: format!(
          "prove_with_multi_column_lookup: structure-registered table \
           column count ({}) differs from payload.comm_values ({})",
          fresh_table_value_columns.len(),
          c,
        ),
      });
    }
    let w_len = fresh_witness_address.len();
    for (i, col) in fresh_witness_value_columns.iter().enumerate() {
      if col.len() != w_len {
        return Err(NovaError::UnSat {
          reason: format!(
            "prove_with_multi_column_lookup: witness value column {} length ({}) \
             differs from address column ({})",
            i,
            col.len(),
            w_len,
          ),
        });
      }
    }
    for (i, col) in fresh_table_value_columns.iter().enumerate() {
      if col.len() != fresh_table_size {
        return Err(NovaError::UnSat {
          reason: format!(
            "prove_with_multi_column_lookup: structure-registered table \
             value column {} length ({}) differs from registered size ({})",
            i,
            col.len(),
            fresh_table_size,
          ),
        });
      }
    }

    // initialize a new RO
    let mut ro = E::RO2::new(ro_consts.clone());
    ro.absorb(*pp_digest);
    U2.absorb_in_ro2(&mut ro);

    // --- Step (2): Stage I-pri transcript pin ---
    // Order: comm_L (address), comm_values[i] for i=0..c, then comm_ts.
    payload.comm_L.absorb_in_ro2(&mut ro);
    for cv in &payload.comm_values {
      cv.absorb_in_ro2(&mut ro);
    }
    payload.comm_ts.absorb_in_ro2(&mut ro);

    // tau / comm_E / rho
    let tau = ro.squeeze(NUM_CHALLENGE_BITS, false);
    let E = PowPolynomial::new(&tau, S.ell).split_evals(S.left, S.right);
    // GH-#2 M.3: `r_E` arrives via the inner-function parameter so the
    // M.3 byte-equivalence regression test (pin §5.2 #2) can pin a
    // deterministic value across the multi-column and multi-table paths.
    // Production callers reach this via the public wrapper which sources
    // `r_E` from `OsRng`.
    let comm_E = CE::<E>::commit(ck, &E, &r_E);
    comm_E.absorb_in_ro2(&mut ro);
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Stage I-pri: squeeze α IFF value columns are present ---
    // When `c == 0`, α is NOT squeezed — the FS transcript and combined
    // witness/table degenerate to the Stage H single-column path
    // byte-for-byte. This preserves Stage H byte-equivalence as a pin.
    let alpha = if c > 0 {
      ro.squeeze(NUM_CHALLENGE_BITS, false)
    } else {
      // Sentinel — never used. Combined witness/table reduce to the
      // address column when c == 0.
      E::Scalar::ZERO
    };

    // r_logup
    let r_logup = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Combine off-circuit ---
    // W_combined[k] = address[k] + α·v₁[k] + α²·v₂[k] + ... + α^c·v_c[k]
    // T_combined[i] = i + α·T₁[i] + α²·T₂[i] + ... + α^c·T_c[i]
    let combined_witness =
      combine_columns_witness::<E>(fresh_witness_address, fresh_witness_value_columns, &alpha);
    let combined_table = combine_columns_table::<E>(
      fresh_table_size,
      fresh_table_value_columns,
      &alpha,
    );

    // --- Step (6): Construct LookupSumcheckInstance with combined data ---
    let (lookup_inst, comm_inv_w2, comm_inv_t2) = LookupSumcheckInstance::<E>::new(
      ck,
      &running_lw.witness,
      &running_lw.inv_w,
      &running_lw.table,
      &running_lw.multiplicities,
      &running_lw.inv_t,
      running_lw.eq_w_left.clone(),
      running_lw.eq_w_right.clone(),
      running_lw.eq_t_left.clone(),
      running_lw.eq_t_right.clone(),
      &combined_witness,
      &combined_table,
      fresh_multiplicities,
      fresh_eq_w_left.clone(),
      fresh_eq_w_right.clone(),
      fresh_eq_t_left.clone(),
      fresh_eq_t_right.clone(),
      r_logup,
    )?;

    comm_inv_w2.absorb_in_ro2(&mut ro);
    comm_inv_t2.absorb_in_ro2(&mut ro);

    // --- R1CS-side sumcheck (identical to prove_with_lookup) ---
    let T = (E::Scalar::ONE - rho) * U1.T;
    let (res1, res2) = rayon::join(
      || {
        let z1 = [W1.W.clone(), vec![U1.u], U1.X.clone()].concat();
        S.S.multiply_vec(&z1)
      },
      || {
        let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
        S.S.multiply_vec(&z2)
      },
    );
    let (Az1, Bz1, Cz1) = res1?;
    let (Az2, Bz2, Cz2) = res2?;
    let (eval_point_0, eval_point_2, eval_point_3, eval_point_4, eval_point_5) =
      Self::prove_helper(
        &rho,
        (S.left, S.right),
        &W1.E,
        &Az1,
        &Bz1,
        &Cz1,
        &E,
        &Az2,
        &Bz2,
        &Cz2,
      );
    let evals = vec![
      eval_point_0,
      T - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
      eval_point_5,
    ];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals);
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly, &mut ro);

    // Lookup sumcheck
    // Multi-table extension (GH-#2, design pin §5.1): `lookup_running_claims_from`
    // returns `Vec<E::Scalar>`. Single-table (`prove_with_multi_column_lookup`)
    // reads entry [0]; outer-base empty Vec falls back to ZERO.
    let t_lookup_running_vec = lookup_running_claims_from::<E>(U1);
    let t_lookup_running = t_lookup_running_vec
      .first()
      .copied()
      .unwrap_or(E::Scalar::ZERO);
    let poly_lookup = lookup_inst.prove_step(&rho, &t_lookup_running);
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly_lookup, &mut ro);

    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();

    let T_lookup_out =
      LookupSumcheckInstance::<E>::verify_step(&rho, &r_b, &poly_lookup, &t_lookup_running)?;

    let effective_payload = LookupPayload {
      comm_L: payload.comm_L,
      comm_ts: payload.comm_ts,
      comm_inv_w: comm_inv_w2,
      comm_inv_t: comm_inv_t2,
      T2_lookup: payload.T2_lookup,
      comm_values: payload.comm_values.clone(),
    };
    // Multi-table extension (GH-#2 M.2): one-element slice; see commentary
    // at the analogous site in `prove_with_lookup`.
    let U = U1.fold_with_lookup(
      U2,
      &comm_E,
      &r_b,
      &T_out,
      &effective_payload,
      std::slice::from_ref(&T_lookup_out),
    )?;
    let W = W1.fold(W2, &E, &r_E, &r_b)?;

    // Fold the running lookup witness with the combined fresh data.
    let fresh_inv_w =
      crate::spartan::logup_inverses::batch_invert_plus_r(&combined_witness, &r_logup)?;
    let fresh_inv_t_raw =
      crate::spartan::logup_inverses::batch_invert_plus_r(&combined_table, &r_logup)?;
    let fresh_inv_t: Vec<E::Scalar> = fresh_inv_t_raw
      .iter()
      .zip(fresh_multiplicities.iter())
      .map(|(inv, ts)| *inv * *ts)
      .collect();

    let fresh_lw = LookupFreshWitness {
      witness: combined_witness,
      inv_w: fresh_inv_w,
      table: combined_table,
      multiplicities: fresh_multiplicities.to_vec(),
      inv_t: fresh_inv_t,
      eq_w_left: fresh_eq_w_left,
      eq_w_right: fresh_eq_w_right,
      eq_t_left: fresh_eq_t_left,
      eq_t_right: fresh_eq_t_right,
    };
    let folded_lw = running_lw.fold(&fresh_lw, &r_b);

    // Multi-table extension (GH-#2, design pin §5.1): single-table
    // (`prove_with_multi_column_lookup`) wraps in a one-element Vec — see
    // `prove_with_lookup` for the same pattern. The FS transcript byte
    // stream is preserved (§5.2 #2).
    let nifs = NIFS {
      comm_E,
      poly,
      poly_lookup: Some(vec![poly_lookup]),
      comm_inv_w: Some(vec![comm_inv_w2]),
      comm_inv_t: Some(vec![comm_inv_t2]),
    };

    Ok((nifs, (U, W), folded_lw))
  }

  /// Prove a fold step with **multi-table** lookup data (GH-#2 M.3).
  ///
  /// Multi-table extension per design pin §5.1 (API surface) / §2.2
  /// (transcript schedule) / §1.3 (VECTOR (C)-invariant).
  ///
  /// Iterates the existing single-table multi-column structural template
  /// once per registered table in `table_id`-canonical order. The k tables
  /// are resolved from `S.lookups.multi_column_tables` (which has been
  /// canonicalised by ascending `table_id` via
  /// `Structure::new_with_lookups` at `relation.rs:488`); the caller
  /// supplies one [`PerTableBundle`] per registered table in the same
  /// order (debug-asserted).
  ///
  /// Per pin §2.2:
  /// - Step 0: pp_digest + R1CS instance (UNCHANGED).
  /// - Step 2: per-table commitments BEFORE tau, outer loop over j ascending.
  ///   Order: ALL tables' `comm_L` → ALL tables' value columns
  ///   (intra-table order preserved) → ALL tables' `comm_ts`.
  /// - Step 3: tau (UNCHANGED — single tau for R1CS side).
  /// - Step 4: rho (UNCHANGED — single rho across all instances).
  /// - Step 5a: per-table alpha, ascending; tables with empty value
  ///   columns skip the squeeze (preserves single-table c=0 byte-equivalence).
  /// - Step 5b: per-table `r_logup_j`, ascending. Each `r_logup_j` is
  ///   sequentially derived; structural sequencing IS domain separation
  ///   per §2.3 (no per-table domain-separator scalar absorbed).
  /// - Step 6: per-table inverse-witness commitments. Order: ALL tables'
  ///   `comm_inv_w` → ALL tables' `comm_inv_t` (NOT interleaved per-table
  ///   per §2.2 step 6 commentary).
  /// - Step 7: R1CS-side prove + absorb poly (UNCHANGED).
  /// - Step 8: per-table lookup-side prove + absorb `poly_lookup_j`,
  ///   ascending.
  /// - Step 9: r_b (UNCHANGED — single fold randomness shared across ALL
  ///   instances).
  ///
  /// Per pin §1.3, the running-claim invariant is a VECTOR
  /// `T_lookup ∈ E::Scalar^k`. The per-table (C)-binding
  /// `∀ j: poly_lookup_j(0) + poly_lookup_j(1) == T_lookup_running_j` is
  /// enforced at verify time (M.4), NOT here — `prove` threads the
  /// running scalars but does not re-check the binding (mirrors
  /// `fold_with_lookup`'s discipline per `relation.rs:675`).
  ///
  /// Single-table degeneration (k=1) is byte-identical at the
  /// FS-transcript layer to the existing
  /// [`Self::prove_with_multi_column_lookup`] path per pin §5.2 #2 — the
  /// per-table loops collapse to single iterations and the absorb / squeeze
  /// order matches `prove_with_multi_column_lookup` exactly. This is the
  /// load-bearing regression check landed by the M.3 byte-equivalence test
  /// (see `m3_byte_equivalence_k1_multi_table_matches_multi_column` in the
  /// `tests` module). The serialised `NIFS<E>` envelope differs at k=1 by
  /// design (Vec-of-1 vs `Option<UniPoly>` per pin §5.2 #2 reword) — only
  /// the FS-transcript layer is byte-identical.
  ///
  /// Returns `(nifs, (folded_U, folded_W), folded_lw_per_table)`. The
  /// per-table folded running witness is a `Vec<LookupRunningWitness<E>>`
  /// of length k in `table_id`-canonical order, suitable for use as the
  /// `bundles[j].running_lw` input on the next fold step.
  ///
  /// ## Errors
  ///
  /// Returns [`NovaError::InvalidStructure`] if the structure has no
  /// `LookupShape` attached, or if `bundles.len()` does not match
  /// `multi_column_tables.len()`, or if any bundle's `table_id` does not
  /// match the structure-registered table at the same position. Returns
  /// [`NovaError::UnSat`] if any per-bundle column-count or length
  /// invariant is violated.
  #[cfg(feature = "lookup-fold")]
  pub fn prove_with_multi_table_lookup(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    W1: &FoldedWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    bundles: &[PerTableBundle<E>],
  ) -> Result<
    (
      NIFS<E>,
      (FoldedInstance<E>, FoldedWitness<E>),
      Vec<LookupRunningWitness<E>>,
    ),
    NovaError,
  > {
    let r_E = E::Scalar::random(&mut OsRng);
    Self::prove_with_multi_table_lookup_inner(
      ck, ro_consts, pp_digest, S, U1, W1, U2, W2, bundles, r_E,
    )
  }

  /// Inner of [`Self::prove_with_multi_table_lookup`] — see that function
  /// for the contract. Takes `r_E: E::Scalar` (the eq-polynomial blinding)
  /// as an explicit parameter so the M.3 byte-equivalence regression test
  /// (pin §5.2 #2) can pin a deterministic value across the multi-column
  /// k=1 and multi-table k=1 paths. Production callers should use the
  /// public wrapper, which sources `r_E` from `OsRng`.
  #[cfg(feature = "lookup-fold")]
  pub(crate) fn prove_with_multi_table_lookup_inner(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    W1: &FoldedWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    bundles: &[PerTableBundle<E>],
    r_E: E::Scalar,
  ) -> Result<
    (
      NIFS<E>,
      (FoldedInstance<E>, FoldedWitness<E>),
      Vec<LookupRunningWitness<E>>,
    ),
    NovaError,
  > {
    // --- Resolve registry from structure ---
    let lookup_shape = S.lookups.as_ref().ok_or(NovaError::InvalidStructure)?;
    let multi_tables = &lookup_shape.multi_column_tables;
    let k = multi_tables.len();
    if bundles.len() != k {
      return Err(NovaError::InvalidStructure);
    }
    // Pin §5.1: bundles MUST be in ascending `table_id` order matching
    // `multi_column_tables`. Debug-assert ordering and per-position id
    // match; downgrade to InvalidStructure error in release builds when
    // the id mismatch surfaces.
    for (j, b) in bundles.iter().enumerate() {
      debug_assert_eq!(
        b.table_id, multi_tables[j].table_id,
        "PerTableBundle[{}].table_id ({}) must equal multi_column_tables[{}].table_id ({})",
        j, b.table_id, j, multi_tables[j].table_id,
      );
      if b.table_id != multi_tables[j].table_id {
        return Err(NovaError::InvalidStructure);
      }
    }
    // Per-bundle column-count and length invariants (mirrors the
    // single-table inner at `prove_with_multi_column_lookup_inner`).
    for (j, b) in bundles.iter().enumerate() {
      let multi_table = &multi_tables[j];
      let c_j = b.payload.comm_values.len();
      if b.fresh_witness_value_columns.len() != c_j {
        return Err(NovaError::UnSat {
          reason: format!(
            "prove_with_multi_table_lookup: bundle[{}] witness column count ({}) \
             differs from payload.comm_values ({})",
            j,
            b.fresh_witness_value_columns.len(),
            c_j,
          ),
        });
      }
      if multi_table.columns.len() != c_j {
        return Err(NovaError::UnSat {
          reason: format!(
            "prove_with_multi_table_lookup: bundle[{}] structure-registered table \
             column count ({}) differs from payload.comm_values ({})",
            j,
            multi_table.columns.len(),
            c_j,
          ),
        });
      }
      let w_len = b.fresh_witness_address.len();
      for (i, col) in b.fresh_witness_value_columns.iter().enumerate() {
        if col.len() != w_len {
          return Err(NovaError::UnSat {
            reason: format!(
              "prove_with_multi_table_lookup: bundle[{}] witness value column {} \
               length ({}) differs from address column ({})",
              j,
              i,
              col.len(),
              w_len,
            ),
          });
        }
      }
      for (i, col) in multi_table.columns.iter().enumerate() {
        if col.len() != multi_table.size {
          return Err(NovaError::UnSat {
            reason: format!(
              "prove_with_multi_table_lookup: bundle[{}] structure-registered table \
               value column {} length ({}) differs from registered size ({})",
              j,
              i,
              col.len(),
              multi_table.size,
            ),
          });
        }
      }
    }

    // --- Step 0: pp_digest + R1CS instance ---
    let mut ro = E::RO2::new(ro_consts.clone());
    ro.absorb(*pp_digest);
    U2.absorb_in_ro2(&mut ro);

    // --- Step 2: per-table commitments BEFORE tau ---
    // Order per pin §2.2: ALL tables' comm_L → ALL tables' value columns
    // (intra-table order preserved) → ALL tables' comm_ts. Outer loop
    // over j (table-id ascending).
    for b in bundles {
      b.payload.comm_L.absorb_in_ro2(&mut ro);
      for cv in &b.payload.comm_values {
        cv.absorb_in_ro2(&mut ro);
      }
      b.payload.comm_ts.absorb_in_ro2(&mut ro);
    }

    // --- Step 3: tau ---
    let tau = ro.squeeze(NUM_CHALLENGE_BITS, false);
    let E_eq = PowPolynomial::new(&tau, S.ell).split_evals(S.left, S.right);
    let comm_E = CE::<E>::commit(ck, &E_eq, &r_E);
    comm_E.absorb_in_ro2(&mut ro);

    // --- Step 4: rho ---
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Step 5a: per-table alpha (gated on c_j > 0) ---
    // Order: ascending table_id; tables with empty value columns skip the
    // squeeze per pin §2.2 step 5a empty-skip rule. This preserves the
    // single-table c=0 byte-equivalence pin per-table.
    let mut alpha_vec: Vec<E::Scalar> = Vec::with_capacity(k);
    for b in bundles {
      let alpha_j = if b.payload.comm_values.is_empty() {
        E::Scalar::ZERO
      } else {
        ro.squeeze(NUM_CHALLENGE_BITS, false)
      };
      alpha_vec.push(alpha_j);
    }

    // --- Step 5b: per-table r_logup ---
    // Order: ascending table_id. Each r_logup_j is sequentially derived
    // from the running transcript whose state at squeeze-j includes ALL
    // prior table commitments (Step 2 sweep) and ALL prior r_logup_i
    // squeezes (i < j). This sequencing IS domain separation per pin §2.3
    // — no explicit per-table domain-separator scalar is absorbed.
    let mut r_logup_vec: Vec<E::Scalar> = Vec::with_capacity(k);
    for _ in 0..k {
      r_logup_vec.push(ro.squeeze(NUM_CHALLENGE_BITS, false));
    }

    // --- Combine off-circuit (per-table) and construct LookupSumcheckInstance ---
    // For each table, compute the combined witness/table per Lasso §6.2
    // and instantiate a LookupSumcheckInstance under the corresponding
    // r_logup_j.
    let mut lookup_insts: Vec<LookupSumcheckInstance<E>> = Vec::with_capacity(k);
    let mut comm_inv_w_vec: Vec<Commitment<E>> = Vec::with_capacity(k);
    let mut comm_inv_t_vec: Vec<Commitment<E>> = Vec::with_capacity(k);
    let mut combined_witness_per_table: Vec<Vec<E::Scalar>> = Vec::with_capacity(k);
    let mut combined_table_per_table: Vec<Vec<E::Scalar>> = Vec::with_capacity(k);
    for (j, b) in bundles.iter().enumerate() {
      let multi_table = &multi_tables[j];
      let alpha_j = alpha_vec[j];
      let r_logup_j = r_logup_vec[j];

      let combined_witness = combine_columns_witness::<E>(
        &b.fresh_witness_address,
        &b.fresh_witness_value_columns,
        &alpha_j,
      );
      let combined_table =
        combine_columns_table::<E>(multi_table.size, &multi_table.columns, &alpha_j);

      let (lookup_inst, comm_inv_w_j, comm_inv_t_j) = LookupSumcheckInstance::<E>::new(
        ck,
        // U1 (running) data — per-table running witness.
        &b.running_lw.witness,
        &b.running_lw.inv_w,
        &b.running_lw.table,
        &b.running_lw.multiplicities,
        &b.running_lw.inv_t,
        b.running_lw.eq_w_left.clone(),
        b.running_lw.eq_w_right.clone(),
        b.running_lw.eq_t_left.clone(),
        b.running_lw.eq_t_right.clone(),
        // U2 (fresh) data — combined witness/table for this table.
        &combined_witness,
        &combined_table,
        &b.fresh_multiplicities,
        b.fresh_eq_w_left.clone(),
        b.fresh_eq_w_right.clone(),
        b.fresh_eq_t_left.clone(),
        b.fresh_eq_t_right.clone(),
        r_logup_j,
      )?;

      lookup_insts.push(lookup_inst);
      comm_inv_w_vec.push(comm_inv_w_j);
      comm_inv_t_vec.push(comm_inv_t_j);
      combined_witness_per_table.push(combined_witness);
      combined_table_per_table.push(combined_table);
    }

    // --- Step 6: per-table inverse-witness commitments ---
    // Order: ALL tables' comm_inv_w → ALL tables' comm_inv_t (NOT
    // interleaved per-table per pin §2.2 step 6 commentary; matches the
    // `(comm_inv_w2; comm_inv_t2)` pair-grouping at the single-table
    // multi-column path lifted to k tables).
    for c in &comm_inv_w_vec {
      c.absorb_in_ro2(&mut ro);
    }
    for c in &comm_inv_t_vec {
      c.absorb_in_ro2(&mut ro);
    }

    // --- Step 7: R1CS-side sumcheck (UNCHANGED — single R1CS instance) ---
    let T = (E::Scalar::ONE - rho) * U1.T;
    let (res1, res2) = rayon::join(
      || {
        let z1 = [W1.W.clone(), vec![U1.u], U1.X.clone()].concat();
        S.S.multiply_vec(&z1)
      },
      || {
        let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
        S.S.multiply_vec(&z2)
      },
    );
    let (Az1, Bz1, Cz1) = res1?;
    let (Az2, Bz2, Cz2) = res2?;
    let (eval_point_0, eval_point_2, eval_point_3, eval_point_4, eval_point_5) =
      Self::prove_helper(
        &rho,
        (S.left, S.right),
        &W1.E,
        &Az1,
        &Bz1,
        &Cz1,
        &E_eq,
        &Az2,
        &Bz2,
        &Cz2,
      );
    let evals = vec![
      eval_point_0,
      T - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
      eval_point_5,
    ];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals);
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly, &mut ro);

    // --- Step 8: per-table lookup-side prove + absorb poly_lookup_j ---
    // Per-table running scalars from U1.T_lookup (VECTOR per pin §1.3).
    // Single-table degeneration: when k=1, the loop runs once and the
    // running-scalar lookup matches `prove_with_multi_column_lookup_inner`
    // by reading entry [0] / fallback to ZERO on outer-base empty Vec.
    let t_lookup_running_full = lookup_running_claims_from::<E>(U1);
    let mut poly_lookup_vec: Vec<UniPoly<E::Scalar>> = Vec::with_capacity(k);
    let mut t_lookup_running_per_table: Vec<E::Scalar> = Vec::with_capacity(k);
    for (j, lookup_inst) in lookup_insts.iter().enumerate() {
      // Outer base: U1.T_lookup is None → empty Vec → all per-table
      // running scalars are ZERO. Mid-fold: per-table entry [j] in
      // `table_id`-canonical order.
      let t_lookup_running_j = t_lookup_running_full
        .get(j)
        .copied()
        .unwrap_or(E::Scalar::ZERO);
      t_lookup_running_per_table.push(t_lookup_running_j);
      let poly_lookup_j = lookup_inst.prove_step(&rho, &t_lookup_running_j);
      <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly_lookup_j, &mut ro);
      poly_lookup_vec.push(poly_lookup_j);
    }

    // --- Step 9: r_b (UNCHANGED — shared across ALL instances) ---
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Compute R1CS T_out (UNCHANGED) ---
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();

    // --- Compute per-table T_lookup_out_j (VECTOR per pin §1.3) ---
    let mut t_lookup_out_per_table: Vec<E::Scalar> = Vec::with_capacity(k);
    for (j, poly_lookup_j) in poly_lookup_vec.iter().enumerate() {
      let t_lookup_running_j = t_lookup_running_per_table[j];
      let t_lookup_out_j = LookupSumcheckInstance::<E>::verify_step(
        &rho,
        &r_b,
        poly_lookup_j,
        &t_lookup_running_j,
      )?;
      t_lookup_out_per_table.push(t_lookup_out_j);
    }

    // --- Fold the FoldedInstance ---
    // Per pin §5.1 commentary on fold_with_lookup at relation.rs:677, the
    // four `comm_*` fields on FoldedInstance remain
    // `Option<Commitment<E>>` (single per-step value); `T_lookup` is
    // VECTOR. M.2's existing fold_with_lookup signature (taking a
    // per-table running-scalar slice) accommodates this directly.
    //
    // For the multi-table fold, the four single commitments folded into
    // FoldedInstance are aggregated from the per-table bundles via
    // sum-of-commitments (Pedersen homomorphism). Per pin §5.4 / spike
    // conclusion, the multi-table FoldedInstance commitment shape is
    // out-of-scope for M.3 — we use the table-0 (first registered)
    // bundle's commitments as the canonical FoldedInstance position
    // for k=1 byte-equivalence, mirroring how
    // `prove_with_multi_column_lookup_inner` builds its
    // `effective_payload`. M.2's fold_with_lookup takes a full
    // LookupPayload<E>; for k≥2 the fold-shape question is deferred to
    // a later milestone (the per-table comm_inv_w/comm_inv_t for tables
    // j≥1 are still threaded into the NIFS struct's Vec fields and
    // verified per-table at M.4).
    //
    // GH-#2 M.3 single-table-degeneration discipline: at k=1 the
    // effective_payload below is byte-identical to the
    // `prove_with_multi_column_lookup_inner` construction, preserving
    // the byte-equivalence pin §5.2 #2.
    let effective_payload = LookupPayload {
      comm_L: bundles[0].payload.comm_L,
      comm_ts: bundles[0].payload.comm_ts,
      comm_inv_w: comm_inv_w_vec[0],
      comm_inv_t: comm_inv_t_vec[0],
      T2_lookup: bundles[0].payload.T2_lookup,
      comm_values: bundles[0].payload.comm_values.clone(),
    };
    let U = U1.fold_with_lookup(
      U2,
      &comm_E,
      &r_b,
      &T_out,
      &effective_payload,
      &t_lookup_out_per_table,
    )?;
    let W = W1.fold(W2, &E_eq, &r_E, &r_b)?;

    // --- Per-table folded running witness ---
    // For each table, recompute the fresh inverse vectors (the
    // LookupSumcheckInstance computed them internally; we redo to feed
    // the LookupFreshWitness used in fold) and fold against the
    // per-table running_lw under the shared r_b.
    let mut folded_lw_per_table: Vec<LookupRunningWitness<E>> = Vec::with_capacity(k);
    for (j, b) in bundles.iter().enumerate() {
      let r_logup_j = r_logup_vec[j];
      let combined_witness = &combined_witness_per_table[j];
      let combined_table = &combined_table_per_table[j];
      let fresh_inv_w =
        crate::spartan::logup_inverses::batch_invert_plus_r(combined_witness, &r_logup_j)?;
      let fresh_inv_t_raw =
        crate::spartan::logup_inverses::batch_invert_plus_r(combined_table, &r_logup_j)?;
      let fresh_inv_t: Vec<E::Scalar> = fresh_inv_t_raw
        .iter()
        .zip(b.fresh_multiplicities.iter())
        .map(|(inv, ts)| *inv * *ts)
        .collect();
      let fresh_lw = LookupFreshWitness {
        witness: combined_witness.clone(),
        inv_w: fresh_inv_w,
        table: combined_table.clone(),
        multiplicities: b.fresh_multiplicities.clone(),
        inv_t: fresh_inv_t,
        eq_w_left: b.fresh_eq_w_left.clone(),
        eq_w_right: b.fresh_eq_w_right.clone(),
        eq_t_left: b.fresh_eq_t_left.clone(),
        eq_t_right: b.fresh_eq_t_right.clone(),
      };
      let folded_lw_j = b.running_lw.fold(&fresh_lw, &r_b);
      folded_lw_per_table.push(folded_lw_j);
    }

    // Per-table NIFS extension fields. The Vec fields carry the full
    // k-tuple; verifier reads them per-table at M.4.
    let nifs = NIFS {
      comm_E,
      poly,
      poly_lookup: Some(poly_lookup_vec),
      comm_inv_w: Some(comm_inv_w_vec),
      comm_inv_t: Some(comm_inv_t_vec),
    };

    Ok((nifs, (U, W), folded_lw_per_table))
  }

  /// Verify a fold step with **multi-column** lookup data (Stage I-pri).
  ///
  /// Symmetric to [`Self::prove_with_multi_column_lookup`]: re-derives α
  /// from the same transcript order (after rho, before r_logup, gated
  /// on `payload.comm_values.is_empty()`).
  ///
  /// The verifier never reconstructs the combined witness/table — the
  /// per-step LogUp closed-form check operates on `poly_lookup` and the
  /// running target only. The combined-commitment binding is implicit
  /// via the FS transcript: α was bound to all column commitments at
  /// squeeze time, so a malicious prover who supplied wrong column
  /// witnesses would produce inconsistent inverse-witness commitments
  /// and fail the (C)-binding check downstream.
  #[cfg(feature = "lookup-fold")]
  #[allow(clippy::too_many_arguments)]
  pub fn verify_with_multi_column_lookup(
    &self,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    U2: &R1CSInstance<E>,
    payload: &LookupPayload<E>,
    table_id: u64,
  ) -> Result<FoldedInstance<E>, NovaError> {
    // Stage I-app.2: verifier-side check that the supplied `table_id` is
    // registered in the structure. Soundness for the table contents
    // themselves comes from `pp_digest` (which the prover and verifier
    // both absorbed at the head of the transcript). The lookup here
    // is a structural sanity gate — a mismatched id surfaces as
    // `NovaError::InvalidStructure` rather than as a downstream
    // sumcheck rejection.
    let lookup_shape = S
      .lookups
      .as_ref()
      .ok_or(NovaError::InvalidStructure)?;
    let multi_table = lookup_shape
      .multi_column_tables
      .iter()
      .find(|t| t.table_id == table_id)
      .ok_or(NovaError::InvalidStructure)?;
    // Cross-check: the per-step payload's value-column count must match the
    // structurally-pinned table column count. This is a structure-bound
    // check that the verifier can perform without ever reconstructing
    // the combined witness/table.
    if payload.comm_values.len() != multi_table.columns.len() {
      return Err(NovaError::InvalidStructure);
    }

    let mut ro = E::RO2::new(ro_consts.clone());
    ro.absorb(*pp_digest);
    U2.absorb_in_ro2(&mut ro);

    payload.comm_L.absorb_in_ro2(&mut ro);
    for cv in &payload.comm_values {
      cv.absorb_in_ro2(&mut ro);
    }
    payload.comm_ts.absorb_in_ro2(&mut ro);

    let _tau = ro.squeeze(NUM_CHALLENGE_BITS, false);
    self.comm_E.absorb_in_ro2(&mut ro);
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    let _alpha = if !payload.comm_values.is_empty() {
      ro.squeeze(NUM_CHALLENGE_BITS, false)
    } else {
      E::Scalar::ZERO
    };

    let _r_logup = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // Multi-table extension (GH-#2, design pin §5.1): single-table verify
    // path reads entry [0] of the Vec-typed lookup fields. The FS transcript
    // byte stream is preserved (§5.2 #2) — the absorb_in_ro2 call sees the
    // single Commitment/UniPoly value, identical to the pre-#2 emission.
    let comm_inv_w_vec = self
      .comm_inv_w
      .as_ref()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let comm_inv_t_vec = self
      .comm_inv_t
      .as_ref()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let comm_inv_w = comm_inv_w_vec
      .first()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let comm_inv_t = comm_inv_t_vec
      .first()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    comm_inv_w.absorb_in_ro2(&mut ro);
    comm_inv_t.absorb_in_ro2(&mut ro);

    let T = (E::Scalar::ONE - rho) * U1.T;
    if self.poly.eval_at_zero() + self.poly.eval_at_one() != T {
      return Err(NovaError::InvalidSumcheckProof);
    }
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&self.poly, &mut ro);

    let poly_lookup_vec = self
      .poly_lookup
      .as_ref()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let poly_lookup = poly_lookup_vec
      .first()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let t_lookup_running_vec = lookup_running_claims_from::<E>(U1);
    let t_lookup_running = t_lookup_running_vec
      .first()
      .copied()
      .unwrap_or(E::Scalar::ZERO);
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(poly_lookup, &mut ro);

    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();
    let T_lookup_out =
      LookupSumcheckInstance::<E>::verify_step(&rho, &r_b, poly_lookup, &t_lookup_running)?;

    // Multi-table extension (GH-#2 M.2): one-element slice; verify-side
    // path mirrors the prove-side single-table contract.
    let U = U1.fold_with_lookup(
      U2,
      &self.comm_E,
      &r_b,
      &T_out,
      payload,
      std::slice::from_ref(&T_lookup_out),
    )?;
    Ok(U)
  }

  /// Verify a fold step with lookup data (Stage E, C1-beta).
  ///
  /// Symmetric to `prove_with_lookup`. Re-derives the same FS challenges
  /// from the transcript, absorbing commitments in the same order, then
  /// verifies both the R1CS and lookup sumcheck polynomials.
  #[cfg(all(feature = "lookup-fold", test))]
  pub fn verify_with_lookup(
    &self,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    U1: &FoldedInstance<E>,
    U2: &R1CSInstance<E>,
    payload: &LookupPayload<E>,
  ) -> Result<FoldedInstance<E>, NovaError> {
    // initialize a new RO
    let mut ro = E::RO2::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(*pp_digest);

    // append U2 to transcript
    U2.absorb_in_ro2(&mut ro);

    // --- Step (2): absorb lookup commitments BEFORE tau squeeze ---
    payload.comm_L.absorb_in_ro2(&mut ro);
    payload.comm_ts.absorb_in_ro2(&mut ro);

    // generate a challenge for the eq polynomial (tau)
    let _tau = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // absorb comm_E
    self.comm_E.absorb_in_ro2(&mut ro);

    // --- Step (4): squeeze rho ---
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Step (5): squeeze r_logup ---
    let _r_logup = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Step (6): absorb inverse commitments ---
    // Multi-table extension (GH-#2, design pin §5.1): single-table verify
    // path reads entry [0] of the now-Vec lookup fields. FS transcript
    // unchanged (§5.2 #2).
    let comm_inv_w_vec = self
      .comm_inv_w
      .as_ref()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let comm_inv_t_vec = self
      .comm_inv_t
      .as_ref()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let comm_inv_w = comm_inv_w_vec
      .first()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let comm_inv_t = comm_inv_t_vec
      .first()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    comm_inv_w.absorb_in_ro2(&mut ro);
    comm_inv_t.absorb_in_ro2(&mut ro);

    // --- R1CS-side checks ---
    let T = (E::Scalar::ONE - rho) * U1.T;

    if self.poly.eval_at_zero() + self.poly.eval_at_one() != T {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // absorb R1CS poly
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&self.poly, &mut ro);

    // --- Step (7): lookup sumcheck verification ---
    let poly_lookup_vec = self
      .poly_lookup
      .as_ref()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let poly_lookup = poly_lookup_vec
      .first()
      .ok_or(NovaError::InvalidSumcheckProof)?;

    let t_lookup_running_vec = lookup_running_claims_from::<E>(U1);
    let t_lookup_running = t_lookup_running_vec
      .first()
      .copied()
      .unwrap_or(E::Scalar::ZERO);

    // absorb lookup poly AFTER R1CS poly
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(poly_lookup, &mut ro);

    // squeeze r_b
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // --- Compute R1CS T_out ---
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();

    // --- Compute T_lookup_out ---
    let T_lookup_out =
      LookupSumcheckInstance::<E>::verify_step(&rho, &r_b, poly_lookup, &t_lookup_running)?;

    // --- Fold with lookup ---
    // Multi-table extension (GH-#2 M.2): one-element slice for the
    // single-table multi-column verify path.
    let U = U1.fold_with_lookup(
      U2,
      &self.comm_E,
      &r_b,
      &T_out,
      payload,
      std::slice::from_ref(&T_lookup_out),
    )?;

    Ok(U)
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
    provider::{
      hyperkzg::EvaluationEngine as HyperKZGEE, ipa_pc::EvaluationEngine, Bn256EngineKZG,
      PallasEngine, Secp256k1Engine,
    },
    r1cs::R1CSShape,
    spartan::{direct::DirectCircuit, snark::RelaxedR1CSSNARK},
    traits::{circuit::NonTrivialCircuit, snark::RelaxedR1CSSNARKTrait, Engine, RO2Constants},
  };
  use ff::Field;

  fn execute_sequence<E: Engine>(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &<E as Engine>::Scalar,
    shape: &R1CSShape<E>,
    U1: &R1CSInstance<E>,
    W1: &R1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) {
    // produce a default running instance
    let str = Structure::new(shape);
    let mut running_W = FoldedWitness::default(&str);
    let mut running_U = FoldedInstance::default(&str);

    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());

    // produce an NIFS with (W1, U1) as the first incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, &str, &running_U, &running_W, U1, W1,
    );
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify an NIFS with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &running_U, U1);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    running_W = W;
    running_U = U;

    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());

    // produce an NIFS with (W2, U2) as the second incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, &str, &running_U, &running_W, U2, W2,
    );
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify an NIFS with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &running_U, U2);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    running_W = W;
    running_U = U;

    // check if the running instance is satisfiable
    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());
  }

  fn test_tiny_r1cs_bellpepper_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    let ro_consts = RO2Constants::<E>::default();

    // generate a non-trivial circuit
    let num_cons: usize = 32;

    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<E::Scalar>::new(num_cons));

    // synthesize the circuit's shape
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // generate a satisfying instance-witness for the r1cs
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(2)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U1, W1) = cs
      .r1cs_instance_and_witness(&shape, &ck)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })
      .unwrap();

    // generate a satisfying instance-witness for the r1cs
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(3)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U2, W2) = cs
      .r1cs_instance_and_witness(&shape, &ck)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })
      .unwrap();

    // pad the shape and witnesses
    let shape = shape.pad();
    let W1 = W1.pad(&shape);
    let W2 = W2.pad(&shape);

    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<E as Engine>::Scalar::ZERO,
      &shape,
      &U1,
      &W1,
      &U2,
      &W2,
    );
  }

  #[test]
  fn test_tiny_r1cs_bellpepper() {
    test_tiny_r1cs_bellpepper_with::<PallasEngine, RelaxedR1CSSNARK<_, EvaluationEngine<_>>>();
    test_tiny_r1cs_bellpepper_with::<Bn256EngineKZG, RelaxedR1CSSNARK<_, HyperKZGEE<_>>>();
    test_tiny_r1cs_bellpepper_with::<Secp256k1Engine, RelaxedR1CSSNARK<_, EvaluationEngine<_>>>();
  }

  /// Stage F: fold-of-two test exercising the full prove->verify->fold pipeline
  /// with lookup data.
  ///
  /// 1. Creates a small R1CS shape (NonTrivialCircuit with 32 constraints)
  /// 2. Creates a 16-entry lookup table
  /// 3. Fold step 1: default running -> (U1, W1, payload1) with 4 in-table queries
  /// 4. Fold step 2: running -> (U2, W2, payload2) with 4 different queries
  ///    (including a duplicate for multiplicity > 1)
  /// 5. Asserts prove/verify agreement at each step
  /// 6. Asserts running_U.T_lookup is populated after each fold
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn execute_sequence_with_lookup() {
    use crate::{
      neutron::relation::{
        LookupPayload, LookupRunningWitness, LookupShape, LookupTableHandle,
      },
      spartan::polys::power::PowPolynomial,
      traits::commitment::CommitmentEngineTrait,
    };
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, HyperKZGEE<E>>;

    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_CF00);
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = Scalar::ZERO;

    // --- 1. Create a small R1CS shape (32 constraints) ---
    let num_cons: usize = 32;

    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));

    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // --- 2. Create a 16-entry lookup table ---
    let table_size = 16usize;
    let table_log2 = 4usize;
    let table: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 7 + 3) as u64))
      .collect();

    // Commit to table for the LookupTableHandle
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
      // For 4 queries: we need witness_ell such that 2^ell >= 4 => ell = 2.
      // But table_ell = 4 (2^4 = 16).
      // We pad queries to match table size for equal-size polynomials.
      // Actually, witness and table can differ in size. Let's use table_ell=4
      // for both to keep it simple (pad queries to 16).
      witness_ell_cached: table_log2,
    };

    let str = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str.S.clone(); // padded shape

    // --- Generate two R1CS instance-witness pairs ---
    let circuit1: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs1 = SatisfyingAssignment::<E>::new();
    let _ = circuit1.synthesize(&mut cs1);
    let (U1, W1) = cs1
      .r1cs_instance_and_witness(&shape, &ck)
      .unwrap();
    let W1 = W1.pad(&shape);

    let circuit2: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(3)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs2 = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut cs2);
    let (U2, W2) = cs2
      .r1cs_instance_and_witness(&shape, &ck)
      .unwrap();
    let W2 = W2.pad(&shape);

    // --- Initialize running state ---
    let mut running_W = FoldedWitness::default(&str);
    let mut running_U = FoldedInstance::default(&str);
    let mut running_lw = LookupRunningWitness::default(&lookup_shape);

    // Verify default instance satisfies the structure
    let res = str.is_sat(&ck, &running_U, &running_W);
    assert!(res.is_ok(), "default instance should be satisfying: {:?}", res);

    // --- Helper: build a lookup payload from queries ---
    // Queries are indices into the table. We pad to table_size (16).
    let build_payload_and_witness = |query_indices: &[usize], rng: &mut ChaCha20Rng| {
      // Build witness (padded to table_size with table[0])
      let mut witness = vec![Scalar::ZERO; table_size];
      let mut multiplicities = vec![Scalar::ZERO; table_size];
      for (i, &idx) in query_indices.iter().enumerate() {
        witness[i] = table[idx];
        multiplicities[idx] += Scalar::ONE;
      }
      // Pad remaining witness entries with table[0] and count them
      for i in query_indices.len()..table_size {
        witness[i] = table[0];
        multiplicities[0] += Scalar::ONE;
      }

      // Commit
      let comm_L = <E as Engine>::CE::commit(&ck, &witness, &Scalar::ZERO);
      let comm_ts = <E as Engine>::CE::commit(&ck, &multiplicities, &Scalar::ZERO);

      // We need temporary inv commitments -- these will be overwritten by
      // prove_with_lookup, but the payload needs them for fold_with_lookup.
      // At this point we don't have r_logup yet, so we use placeholder zeros.
      // prove_with_lookup will compute the actual inverse commitments.
      // The payload's comm_inv_w and comm_inv_t are the U2 commitments.
      // They get set inside prove_with_lookup too, but the payload struct
      // carries them for fold_with_lookup's instance fold.
      //
      // Actually, prove_with_lookup computes and absorbs them internally,
      // then builds the NIFS with them. The payload's comm_inv_w/t fields
      // are used by fold_with_lookup for the INSTANCE-side fold. So we need
      // to set them to what prove_with_lookup will compute.
      //
      // For the test, we'll set them to default and then prove_with_lookup
      // will use the freshly computed commitments from LookupSumcheckInstance::new.
      // Looking at fold_with_lookup: it uses payload.comm_inv_w and payload.comm_inv_t.
      // But prove_with_lookup doesn't update the payload -- it uses
      // comm_inv_w2 and comm_inv_t2 from LookupSumcheckInstance::new.
      //
      // Actually, re-reading prove_with_lookup: it calls
      //   U1.fold_with_lookup(U2, &comm_E, &r_b, &T_out, payload, &T_lookup_out)
      // where payload is the one passed in. So payload.comm_inv_w must
      // be the commitment to the U2-side inverse witness. But we don't know r_logup
      // yet when constructing the payload outside prove_with_lookup.
      //
      // This is a design tension. Let me fix it: the payload should carry the
      // pre-r commitments (comm_L, comm_ts), and the post-r commitments
      // (comm_inv_w, comm_inv_t) should come from prove_with_lookup itself.
      // Let me update prove_with_lookup to construct the final payload
      // with the computed inverse commitments and use THAT for fold_with_lookup.
      //
      // For now, use placeholder zero commitments. prove_with_lookup will
      // override them via the NIFS struct.

      let payload = LookupPayload {
        comm_L,
        comm_ts,
        comm_inv_w: Commitment::<E>::default(),
        comm_inv_t: Commitment::<E>::default(),
        T2_lookup: Scalar::ZERO,
        // Stage I-pri: single-column path leaves `comm_values` empty.
        comm_values: Vec::new(),
      };

      // Build eq polynomials from a random tau
      let tau = Scalar::random(&mut *rng);
      let pow = PowPolynomial::new(&tau, table_log2);
      let (w_left, w_right) = lookup_shape.witness_split();
      let combined_w = pow.split_evals(w_left, w_right);
      let (eq_w_left, eq_w_right) = combined_w.split_at(w_left);

      let tau_t = Scalar::random(&mut *rng);
      let pow_t = PowPolynomial::new(&tau_t, table_log2);
      let (t_left, t_right) = lookup_shape.table_split();
      let combined_t = pow_t.split_evals(t_left, t_right);
      let (eq_t_left, eq_t_right) = combined_t.split_at(t_left);

      (
        payload,
        witness,
        table.clone(),
        multiplicities,
        eq_w_left.to_vec(),
        eq_w_right.to_vec(),
        eq_t_left.to_vec(),
        eq_t_right.to_vec(),
      )
    };

    // --- 3. Fold step 1: 4 in-table queries ---
    let query_indices_1 = vec![0, 3, 7, 15]; // 4 distinct entries
    let (
      mut payload1,
      witness1,
      table1,
      multiplicities1,
      eq_w1_left,
      eq_w1_right,
      eq_t1_left,
      eq_t1_right,
    ) = build_payload_and_witness(&query_indices_1, &mut rng);

    let res = NIFS::prove_with_lookup(
      &ck,
      &ro_consts,
      &pp_digest,
      &str,
      &running_U,
      &running_W,
      &U1,
      &W1,
      &payload1,
      &running_lw,
      &witness1,
      &table1,
      &multiplicities1,
      eq_w1_left,
      eq_w1_right,
      eq_t1_left,
      eq_t1_right,
    );
    assert!(res.is_ok(), "prove_with_lookup step 1 failed: {:?}", res.err());
    let (nifs1, (folded_U1, folded_W1), folded_lw1) = res.unwrap();

    // Update payload with the computed inverse commitments for verify
    // Multi-table extension (GH-#2): NIFS lookup commitments are now
    // `Option<Vec<Commitment<E>>>`. Single-table tests project to entry [0].
    payload1.comm_inv_w = nifs1.comm_inv_w.as_ref().unwrap()[0];
    payload1.comm_inv_t = nifs1.comm_inv_t.as_ref().unwrap()[0];

    // Verify step 1
    let res = nifs1.verify_with_lookup(&ro_consts, &pp_digest, &running_U, &U1, &payload1);
    assert!(res.is_ok(), "verify_with_lookup step 1 failed: {:?}", res.err());
    let verified_U1 = res.unwrap();

    // Assert prove/verify agreement
    assert_eq!(
      folded_U1, verified_U1,
      "prove and verify must produce the same folded instance (step 1)"
    );

    // Assert T_lookup is populated
    assert!(
      folded_U1.T_lookup.is_some(),
      "T_lookup must be populated after fold step 1"
    );
    println!(
      "Step 1: T_lookup = {:?}",
      folded_U1.T_lookup.as_ref().unwrap()
    );

    // R1CS-side satisfiability check
    let res = str.is_sat(&ck, &folded_U1, &folded_W1);
    assert!(res.is_ok(), "folded instance must be satisfying after step 1: {:?}", res);

    // Update running state
    running_U = folded_U1;
    running_W = folded_W1;
    running_lw = folded_lw1;

    // --- 4. Fold step 2: 4 queries with a duplicate for multiplicity > 1 ---
    let query_indices_2 = vec![1, 5, 5, 10]; // index 5 appears twice (multiplicity 2)
    let (
      mut payload2,
      witness2,
      table2,
      multiplicities2,
      eq_w2_left,
      eq_w2_right,
      eq_t2_left,
      eq_t2_right,
    ) = build_payload_and_witness(&query_indices_2, &mut rng);

    let res = NIFS::prove_with_lookup(
      &ck,
      &ro_consts,
      &pp_digest,
      &str,
      &running_U,
      &running_W,
      &U2,
      &W2,
      &payload2,
      &running_lw,
      &witness2,
      &table2,
      &multiplicities2,
      eq_w2_left,
      eq_w2_right,
      eq_t2_left,
      eq_t2_right,
    );
    assert!(res.is_ok(), "prove_with_lookup step 2 failed: {:?}", res.err());
    let (nifs2, (folded_U2, folded_W2), _folded_lw2) = res.unwrap();

    // Update payload with computed inverse commitments for verify
    payload2.comm_inv_w = nifs2.comm_inv_w.as_ref().unwrap()[0];
    payload2.comm_inv_t = nifs2.comm_inv_t.as_ref().unwrap()[0];

    // Verify step 2
    let res = nifs2.verify_with_lookup(&ro_consts, &pp_digest, &running_U, &U2, &payload2);
    assert!(res.is_ok(), "verify_with_lookup step 2 failed: {:?}", res.err());
    let verified_U2 = res.unwrap();

    // Assert prove/verify agreement
    assert_eq!(
      folded_U2, verified_U2,
      "prove and verify must produce the same folded instance (step 2)"
    );

    // Assert T_lookup is populated
    assert!(
      folded_U2.T_lookup.is_some(),
      "T_lookup must be populated after fold step 2"
    );
    println!(
      "Step 2: T_lookup = {:?}",
      folded_U2.T_lookup.as_ref().unwrap()
    );

    // R1CS-side satisfiability check
    let res = str.is_sat(&ck, &folded_U2, &folded_W2);
    assert!(res.is_ok(), "folded instance must be satisfying after step 2: {:?}", res);

    println!("execute_sequence_with_lookup: both fold steps passed prove/verify/sat checks");
  }

  // ============================================================================
  // Stage I-pri: multi-column lookup (Lasso §6.2 address-value combine) tests
  // ============================================================================

  /// Stage I-pri T1: combine_columns_witness/table determinism — same inputs
  /// + same α produce identical combined vectors, and the empty-columns
  /// path returns the address column / identity table unchanged.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn stage_i_pri_combine_determinism() {
    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;

    let address: Vec<Scalar> = (0..8).map(|i| Scalar::from(100 + i as u64)).collect();
    let v1: Vec<Scalar> = (0..8).map(|i| Scalar::from(200 + i as u64)).collect();
    let v2: Vec<Scalar> = (0..8).map(|i| Scalar::from(300 + i as u64)).collect();
    let alpha = Scalar::from(7u64);

    let combined_a = super::combine_columns_witness::<E>(&address, &[v1.clone(), v2.clone()], &alpha);
    let combined_b = super::combine_columns_witness::<E>(&address, &[v1.clone(), v2.clone()], &alpha);
    assert_eq!(
      combined_a, combined_b,
      "combine_columns_witness must be deterministic given same inputs"
    );

    // Spot-check the formula at index 3: address[3] + α·v1[3] + α²·v2[3].
    let expected =
      address[3] + alpha * v1[3] + alpha * alpha * v2[3];
    assert_eq!(combined_a[3], expected, "combine formula must match");

    // Empty columns -> address column unchanged.
    let combined_empty: Vec<Scalar> =
      super::combine_columns_witness::<E>(&address, &[], &alpha);
    assert_eq!(combined_empty, address);

    // Table side, empty columns -> identity vector.
    let identity: Vec<Scalar> =
      super::combine_columns_table::<E>(8, &[], &alpha);
    let expected_identity: Vec<Scalar> = (0..8).map(|i| Scalar::from(i as u64)).collect();
    assert_eq!(identity, expected_identity);

    // Table side with columns: index i + α·T1[i] + α²·T2[i].
    let t1: Vec<Scalar> = (0..8).map(|i| Scalar::from(50 + i as u64)).collect();
    let t2: Vec<Scalar> = (0..8).map(|i| Scalar::from(60 + i as u64)).collect();
    let combined_t = super::combine_columns_table::<E>(8, &[t1.clone(), t2.clone()], &alpha);
    let expected_t3 = Scalar::from(3u64) + alpha * t1[3] + alpha * alpha * t2[3];
    assert_eq!(combined_t[3], expected_t3);
  }

  /// Stage I-pri T2: prove_with_multi_column_lookup → verify round-trip on
  /// a satisfying multi-column witness. The transcript bytes diverge from
  /// Stage H (α is squeezed) but the prove/verify symmetry must hold.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn stage_i_pri_multi_column_round_trip() {
    use crate::{
      neutron::relation::{
        LookupPayload, LookupRunningWitness, LookupShape, LookupTableHandle,
        MultiColumnLookupTable,
      },
      spartan::polys::power::PowPolynomial,
      traits::commitment::CommitmentEngineTrait,
    };
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, HyperKZGEE<E>>;

    let mut rng = ChaCha20Rng::seed_from_u64(0x1A55_07A7);
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = Scalar::ZERO;

    let num_cons = 32usize;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // 16-row table with 2 value columns. The address column is implicit
    // (i ∈ [0, 16)).
    let table_size = 16usize;
    let table_log2 = 4usize;
    let t1: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 11 + 5) as u64))
      .collect();
    let t2: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 17 + 9) as u64))
      .collect();
    // Table commitments are computed by the prover/verifier; we register
    // them through the LookupShape via the address-side handle commitment
    // below. The per-column commitments live on `MultiColumnLookupTable`,
    // which we don't need on this code path because the Stage I-pri tests
    // exercise prove_with_multi_column_lookup directly with table data
    // arrays.
    let _comm_t1 = <E as Engine>::CE::commit(&ck, &t1, &Scalar::ZERO);
    let _comm_t2 = <E as Engine>::CE::commit(&ck, &t2, &Scalar::ZERO);

    // Address-side LookupTableHandle for `Structure`. We use the
    // identity-vector commitment as the table-side handle (the address
    // column is implicit).
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

    // Satisfying R1CS instance.
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U2, W2) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W2 = W2.pad(&shape);

    // Outer-base running state.
    let running_W = FoldedWitness::default(&str);
    let running_U = FoldedInstance::default(&str);
    let running_lw = LookupRunningWitness::default(&lookup_shape);

    // Build a satisfying multi-column witness pool (4 in-table queries
    // padded to table_size). Each query is `(addr_k, t1[addr_k], t2[addr_k])`.
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
    // Pad with row 0.
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

    // eq polynomials.
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

    // Native prove_with_multi_column_lookup.
    let (mut nifs, (folded_U, folded_W), _folded_lw) =
      NIFS::<E>::prove_with_multi_column_lookup(
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
      .expect("prove_with_multi_column_lookup must succeed on satisfying witness");

    // Construct verify-side payload with the computed inverse commitments.
    // Multi-table extension (GH-#2): NIFS lookup commitments are now Vec.
    let payload_for_verify = LookupPayload::<E> {
      comm_L: payload.comm_L,
      comm_ts: payload.comm_ts,
      comm_inv_w: nifs.comm_inv_w.as_ref().unwrap()[0],
      comm_inv_t: nifs.comm_inv_t.as_ref().unwrap()[0],
      T2_lookup: Scalar::ZERO,
      comm_values: payload.comm_values.clone(),
    };

    let verified_U = nifs
      .verify_with_multi_column_lookup(
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &U2,
        &payload_for_verify,
        0u64,
      )
      .expect("verify_with_multi_column_lookup must succeed");

    assert_eq!(folded_U, verified_U, "prove/verify must agree on folded U");
    assert!(folded_U.T_lookup.is_some());

    // R1CS-side satisfiability.
    str.is_sat(&ck, &folded_U, &folded_W).expect("folded R1CS must satisfy");

    // Suppress unused warning.
    nifs.poly_lookup.take();
  }

  /// Stage I-pri T3: soundness negative — supplying a value tuple NOT in
  /// the table makes the LogUp identity fail. We violate it by changing
  /// `witness_v1[0]` to a value not equal to `t1[query_indices[0]]`. The
  /// verifier must reject.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn stage_i_pri_multi_column_soundness_rejects_off_table() {
    use crate::{
      neutron::relation::{
        LookupPayload, LookupRunningWitness, LookupShape, LookupTableHandle,
        MultiColumnLookupTable,
      },
      spartan::polys::power::PowPolynomial,
      traits::commitment::CommitmentEngineTrait,
    };
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, HyperKZGEE<E>>;

    let mut rng = ChaCha20Rng::seed_from_u64(0x1A55_DEAD);
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = Scalar::ZERO;

    let num_cons = 32usize;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    let table_size = 16usize;
    let table_log2 = 4usize;
    let t1: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 11 + 5) as u64))
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
        columns: vec![t1.clone()],
        value_commitments: vec![<E as Engine>::CE::commit(&ck, &t1, &Scalar::ZERO)],
      }],
      num_addr_columns: 1,
      num_witness_columns: 2,
      witness_ell_cached: table_log2,
    };
    let str = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str.S.clone();

    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U2, W2) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W2 = W2.pad(&shape);

    let running_W = FoldedWitness::default(&str);
    let running_U = FoldedInstance::default(&str);
    let running_lw = LookupRunningWitness::default(&lookup_shape);

    // Malicious witness: query_indices[0] = 3, but witness_v1[0] is set
    // to a value NOT equal to t1[3]. Multiplicities are still
    // consistent with addresses (so the address-side LogUp would close)
    // but the value column is off — combined witness fails.
    let query_indices = [3usize, 7, 11, 15];
    let mut witness_addr = vec![Scalar::ZERO; table_size];
    let mut witness_v1 = vec![Scalar::ZERO; table_size];
    let mut multiplicities = vec![Scalar::ZERO; table_size];
    for (i, &idx) in query_indices.iter().enumerate() {
      witness_addr[i] = Scalar::from(idx as u64);
      witness_v1[i] = t1[idx];
      multiplicities[idx] += Scalar::ONE;
    }
    for i in query_indices.len()..table_size {
      witness_addr[i] = Scalar::from(0u64);
      witness_v1[i] = t1[0];
      multiplicities[0] += Scalar::ONE;
    }

    // Forge: change witness_v1[0] to a value not in t1.
    witness_v1[0] = Scalar::from(0xDEAD_BEEFu64);

    let comm_addr = <E as Engine>::CE::commit(&ck, &witness_addr, &Scalar::ZERO);
    let comm_v1 = <E as Engine>::CE::commit(&ck, &witness_v1, &Scalar::ZERO);
    let comm_ts = <E as Engine>::CE::commit(&ck, &multiplicities, &Scalar::ZERO);

    let payload = LookupPayload::<E> {
      comm_L: comm_addr,
      comm_ts,
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![comm_v1],
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

    // Native prove_with_multi_column_lookup with malicious witness.
    // Per NeutronNova's fold-soundness model: the per-step verify does
    // NOT reject (the (C)-binding `poly_lookup(0)+poly_lookup(1) =
    // t_lookup_running` is satisfied by construction — `prove_step`
    // sets `evals[1] := t_lookup_running - evals[0]` regardless of
    // witness validity). Instead, soundness manifests as a NON-ZERO
    // running target `T_lookup_out` after fold step 1 when the
    // ground-truth target is zero. We assert that signature here.
    let (nifs, (folded_U, _folded_W), _) =
      NIFS::<E>::prove_with_multi_column_lookup(
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
        &[witness_v1.clone()],
        &multiplicities,
        eq_w_left.to_vec(),
        eq_w_right.to_vec(),
        eq_t_left.to_vec(),
        eq_t_right.to_vec(),
      )
      .expect("prove still produces a NIFS for malicious witness — soundness is in the running target divergence");

    // Verify completes without error (per-step (C)-binding holds).
    // Multi-table extension (GH-#2): NIFS lookup commitments are now Vec.
    let payload_for_verify = LookupPayload::<E> {
      comm_L: payload.comm_L,
      comm_ts: payload.comm_ts,
      comm_inv_w: nifs.comm_inv_w.as_ref().unwrap()[0],
      comm_inv_t: nifs.comm_inv_t.as_ref().unwrap()[0],
      T2_lookup: Scalar::ZERO,
      comm_values: payload.comm_values.clone(),
    };
    let verified_U = nifs
      .verify_with_multi_column_lookup(
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &U2,
        &payload_for_verify,
        0u64,
      )
      .expect("per-step verify completes — soundness is downstream");

    // Both prover and verifier produce the same folded instance, but
    // its `T_lookup` is non-zero, signaling the off-table query.
    assert_eq!(folded_U, verified_U);
    // Multi-table extension (GH-#2): `T_lookup` is now `Option<Vec<E::Scalar>>`.
    // For single-table tests we read entry [0].
    let t_lookup = folded_U
      .T_lookup
      .as_ref()
      .expect("T_lookup must be populated after multi-column fold step")[0];
    assert_ne!(
      t_lookup,
      Scalar::ZERO,
      "Stage I-pri soundness: an off-table multi-column witness must \
       produce a non-zero next-step running target T_lookup_out (the \
       NeutronNova fold-soundness signature). This is the witness that \
       a downstream `is_sat` check on the lookup side, or a subsequent \
       fold step's R1CS-lookup consistency, would catch."
    );
  }

  /// Stage I-pri T4: fold-of-two with multi-column lookup. Two consecutive
  /// fold steps using `prove_with_multi_column_lookup`, both with
  /// satisfying multi-column witnesses. Asserts prove/verify symmetry,
  /// `T_lookup` propagation, and R1CS satisfiability after each step.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn stage_i_pri_multi_column_fold_of_two() {
    use crate::{
      neutron::relation::{
        LookupPayload, LookupRunningWitness, LookupShape, LookupTableHandle,
        MultiColumnLookupTable,
      },
      spartan::polys::power::PowPolynomial,
      traits::commitment::CommitmentEngineTrait,
    };
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, HyperKZGEE<E>>;

    let mut rng = ChaCha20Rng::seed_from_u64(0x1A55_F010);
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = Scalar::ZERO;

    let num_cons = 32usize;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

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

    // Two satisfying R1CS pairs.
    let circuit1: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs1 = SatisfyingAssignment::<E>::new();
    let _ = circuit1.synthesize(&mut cs1);
    let (U1_r1cs, W1_r1cs) = cs1.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W1_r1cs = W1_r1cs.pad(&shape);

    let circuit2: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(3)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs2 = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut cs2);
    let (U2_r1cs, W2_r1cs) = cs2.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W2_r1cs = W2_r1cs.pad(&shape);

    let mut running_W = FoldedWitness::default(&str);
    let mut running_U = FoldedInstance::default(&str);
    let mut running_lw = LookupRunningWitness::default(&lookup_shape);

    // Helper closure to build a multi-column payload + witness arrays.
    let build = |query_indices: &[usize], rng: &mut ChaCha20Rng| {
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
      let tau_w = Scalar::random(&mut *rng);
      let pow_w = PowPolynomial::new(&tau_w, table_log2);
      let (w_left, w_right) = lookup_shape.witness_split();
      let combined_w = pow_w.split_evals(w_left, w_right);
      let (eq_w_left, eq_w_right) = combined_w.split_at(w_left);
      let tau_t = Scalar::random(&mut *rng);
      let pow_t = PowPolynomial::new(&tau_t, table_log2);
      let (t_left, t_right) = lookup_shape.table_split();
      let combined_t_eq = pow_t.split_evals(t_left, t_right);
      let (eq_t_left, eq_t_right) = combined_t_eq.split_at(t_left);
      (
        payload,
        witness_addr,
        witness_v1,
        witness_v2,
        multiplicities,
        eq_w_left.to_vec(),
        eq_w_right.to_vec(),
        eq_t_left.to_vec(),
        eq_t_right.to_vec(),
      )
    };

    // Step 1: distinct in-table queries.
    let q1 = vec![0usize, 3, 7, 15];
    let (
      mut payload1,
      witness_addr1,
      witness_v1_1,
      witness_v2_1,
      multiplicities1,
      eq_w1l,
      eq_w1r,
      eq_t1l,
      eq_t1r,
    ) = build(&q1, &mut rng);
    let (nifs1, (folded_U1, folded_W1), folded_lw1) =
      NIFS::<E>::prove_with_multi_column_lookup(
        &ck,
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &running_W,
        &U1_r1cs,
        &W1_r1cs,
        &payload1,
        &running_lw,
        0u64,
        &witness_addr1,
        &[witness_v1_1.clone(), witness_v2_1.clone()],
        &multiplicities1,
        eq_w1l,
        eq_w1r,
        eq_t1l,
        eq_t1r,
      )
      .expect("step 1 prove must succeed");
    payload1.comm_inv_w = nifs1.comm_inv_w.as_ref().unwrap()[0];
    payload1.comm_inv_t = nifs1.comm_inv_t.as_ref().unwrap()[0];
    let verified1 = nifs1
      .verify_with_multi_column_lookup(
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &U1_r1cs,
        &payload1,
        0u64,
      )
      .expect("step 1 verify must succeed");
    assert_eq!(folded_U1, verified1);
    assert!(folded_U1.T_lookup.is_some());
    str.is_sat(&ck, &folded_U1, &folded_W1).expect("step 1 R1CS sat");

    running_U = folded_U1;
    running_W = folded_W1;
    running_lw = folded_lw1;

    // Step 2: queries with a duplicate (multiplicity > 1).
    let q2 = vec![1usize, 5, 5, 10];
    let (
      mut payload2,
      witness_addr2,
      witness_v1_2,
      witness_v2_2,
      multiplicities2,
      eq_w2l,
      eq_w2r,
      eq_t2l,
      eq_t2r,
    ) = build(&q2, &mut rng);
    let (nifs2, (folded_U2, folded_W2), _folded_lw2) =
      NIFS::<E>::prove_with_multi_column_lookup(
        &ck,
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &running_W,
        &U2_r1cs,
        &W2_r1cs,
        &payload2,
        &running_lw,
        0u64,
        &witness_addr2,
        &[witness_v1_2.clone(), witness_v2_2.clone()],
        &multiplicities2,
        eq_w2l,
        eq_w2r,
        eq_t2l,
        eq_t2r,
      )
      .expect("step 2 prove must succeed");
    payload2.comm_inv_w = nifs2.comm_inv_w.as_ref().unwrap()[0];
    payload2.comm_inv_t = nifs2.comm_inv_t.as_ref().unwrap()[0];
    let verified2 = nifs2
      .verify_with_multi_column_lookup(
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &U2_r1cs,
        &payload2,
        0u64,
      )
      .expect("step 2 verify must succeed");
    assert_eq!(folded_U2, verified2);
    assert!(folded_U2.T_lookup.is_some());
    str.is_sat(&ck, &folded_U2, &folded_W2).expect("step 2 R1CS sat");
  }

  // -----------------------------------------------------------------
  // Stage I-app.2 — pp_digest binds multi-column table identity.
  //
  // The three tests below exercise the soundness gain from registering
  // multi-column tables in `LookupShape::multi_column_tables` (instead
  // of passing them per-call):
  //
  // 1. `stage_i_app_2_pp_digest_binds_table_identity` — building two
  //    `Structure`s with different multi-column tables yields different
  //    `Structure` bytes (and hence different pp_digest), pinning the
  //    table at IVC initialisation.
  // 2. `stage_i_app_2_unregistered_table_id_rejects` — passing an
  //    unregistered `table_id` to prove returns
  //    `NovaError::InvalidStructure`.
  // 3. `stage_i_app_2_merged_table_two_logical_partitions` — register
  //    one merged 64-row table simulating two 32-row logical sub-tables
  //    (high bit = sub-table id, low 5 bits = address); two consecutive
  //    fold steps each query a different sub-table and both succeed.
  //
  // Pinned by Stage I-app.2 (pp_digest binding for multi-column tables).
  // -----------------------------------------------------------------

  /// Stage I-app.2 T1: pp_digest binds the multi-column table identity.
  /// Two `Structure`s with different multi-column-table contents must
  /// produce different bincode-serialised bytes — `pp_digest` (which is
  /// SHA3-256 over `bincode(PublicParams)` and `PublicParams` includes
  /// `Structure`) therefore differs whenever the structurally-pinned
  /// multi-column table changes.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn stage_i_app_2_pp_digest_binds_table_identity() {
    use crate::{
      neutron::relation::{LookupShape, LookupTableHandle, MultiColumnLookupTable},
      traits::commitment::CommitmentEngineTrait,
    };

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, HyperKZGEE<E>>;

    let num_cons = 32usize;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    let table_size = 16usize;
    let table_log2 = 4usize;
    let identity: Vec<Scalar> =
      (0..table_size).map(|i| Scalar::from(i as u64)).collect();
    let identity_comm = <E as Engine>::CE::commit(&ck, &identity, &Scalar::ZERO);

    // Two distinct multi-column table contents.
    let t1_a: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 11 + 5) as u64))
      .collect();
    let t1_b: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 13 + 7) as u64))
      .collect();
    let comm_t1_a = <E as Engine>::CE::commit(&ck, &t1_a, &Scalar::ZERO);
    let comm_t1_b = <E as Engine>::CE::commit(&ck, &t1_b, &Scalar::ZERO);

    let mk_shape = |columns: Vec<Scalar>, comm: Commitment<E>| LookupShape::<E> {
      tables: vec![LookupTableHandle {
        table_id: 0,
        size: table_size,
        commitment: identity_comm,
      }],
      multi_column_tables: vec![MultiColumnLookupTable {
        table_id: 0,
        size: table_size,
        columns: vec![columns],
        value_commitments: vec![comm],
      }],
      num_addr_columns: 1,
      num_witness_columns: 2,
      witness_ell_cached: table_log2,
    };

    let str_a = Structure::new_with_lookups(&shape, mk_shape(t1_a.clone(), comm_t1_a));
    let str_b = Structure::new_with_lookups(&shape, mk_shape(t1_b.clone(), comm_t1_b));

    // Bincode-serialise via the same encoder `DigestComputer` uses (legacy
    // little-endian fixed-int — see `src/digest.rs`). Different bytes
    // implies different SHA3-256 hash and therefore different
    // pp_digest field element.
    let cfg = bincode::config::legacy()
      .with_little_endian()
      .with_fixed_int_encoding();
    let bytes_a =
      bincode::serde::encode_to_vec(&str_a, cfg).expect("bincode a must encode");
    let bytes_b =
      bincode::serde::encode_to_vec(&str_b, cfg).expect("bincode b must encode");
    assert_ne!(
      bytes_a, bytes_b,
      "Stage I-app.2: bincode(Structure) must differ when the multi-column \
       table contents change. If this assertion fires, the table identity \
       is NOT bound into pp_digest and the spike's soundness pin is broken."
    );
  }

  /// Stage I-app.2 T2: passing an unregistered `table_id` to
  /// `prove_with_multi_column_lookup` returns `NovaError::InvalidStructure`.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn stage_i_app_2_unregistered_table_id_rejects() {
    use crate::{
      neutron::relation::{
        LookupPayload, LookupRunningWitness, LookupShape, LookupTableHandle,
        MultiColumnLookupTable,
      },
      spartan::polys::power::PowPolynomial,
      traits::commitment::CommitmentEngineTrait,
    };
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, HyperKZGEE<E>>;

    let mut rng = ChaCha20Rng::seed_from_u64(0x1A55_BAD_1D);
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = Scalar::ZERO;

    let num_cons = 32usize;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    let table_size = 16usize;
    let table_log2 = 4usize;
    let t1: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 11 + 5) as u64))
      .collect();
    let identity: Vec<Scalar> =
      (0..table_size).map(|i| Scalar::from(i as u64)).collect();
    let identity_comm = <E as Engine>::CE::commit(&ck, &identity, &Scalar::ZERO);

    // Register table_id=7 (NOT 0).
    let lookup_shape = LookupShape::<E> {
      tables: vec![LookupTableHandle {
        table_id: 0,
        size: table_size,
        commitment: identity_comm,
      }],
      multi_column_tables: vec![MultiColumnLookupTable {
        table_id: 7,
        size: table_size,
        columns: vec![t1.clone()],
        value_commitments: vec![<E as Engine>::CE::commit(&ck, &t1, &Scalar::ZERO)],
      }],
      num_addr_columns: 1,
      num_witness_columns: 2,
      witness_ell_cached: table_log2,
    };
    let str = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str.S.clone();

    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U2, W2) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W2 = W2.pad(&shape);

    let running_W = FoldedWitness::default(&str);
    let running_U = FoldedInstance::default(&str);
    let running_lw = LookupRunningWitness::default(&lookup_shape);

    // Build a satisfying multi-column witness pool (table_id=7 contents).
    let query_indices = [0usize, 3, 7, 15];
    let mut witness_addr = vec![Scalar::ZERO; table_size];
    let mut witness_v1 = vec![Scalar::ZERO; table_size];
    let mut multiplicities = vec![Scalar::ZERO; table_size];
    for (i, &idx) in query_indices.iter().enumerate() {
      witness_addr[i] = Scalar::from(idx as u64);
      witness_v1[i] = t1[idx];
      multiplicities[idx] += Scalar::ONE;
    }
    for i in query_indices.len()..table_size {
      witness_addr[i] = Scalar::from(0u64);
      witness_v1[i] = t1[0];
      multiplicities[0] += Scalar::ONE;
    }
    let comm_addr = <E as Engine>::CE::commit(&ck, &witness_addr, &Scalar::ZERO);
    let comm_v1 = <E as Engine>::CE::commit(&ck, &witness_v1, &Scalar::ZERO);
    let comm_ts = <E as Engine>::CE::commit(&ck, &multiplicities, &Scalar::ZERO);

    let payload = LookupPayload::<E> {
      comm_L: comm_addr,
      comm_ts,
      comm_inv_w: Commitment::<E>::default(),
      comm_inv_t: Commitment::<E>::default(),
      T2_lookup: Scalar::ZERO,
      comm_values: vec![comm_v1],
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

    // Pass UNREGISTERED table_id=99.
    let res = NIFS::<E>::prove_with_multi_column_lookup(
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
      99u64, // unregistered
      &witness_addr,
      &[witness_v1.clone()],
      &multiplicities,
      eq_w_left.to_vec(),
      eq_w_right.to_vec(),
      eq_t_left.to_vec(),
      eq_t_right.to_vec(),
    );
    assert!(matches!(res, Err(NovaError::InvalidStructure)));
  }

  /// Stage I-app.2 T3: merged-table partitioning. Register ONE merged
  /// 32-row multi-column table that simulates two 16-row "logical"
  /// sub-tables — high bit of the address selects which sub-table the
  /// query targets. Two consecutive fold steps each query a different
  /// logical sub-table; both must complete prove/verify.
  ///
  /// This is the BIP-340 chunk-step pattern in miniature: rather than
  /// extending the algebra to handle multi-table-per-step, the four
  /// (g, λg, p, λp) windowed tables are merged into one 64-row table
  /// with a 6-bit address (`table_id_2bit << 4 | window_digit_4bit`).
  /// This test exercises the same merge with two sub-tables for clarity.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn stage_i_app_2_merged_table_two_logical_partitions() {
    use crate::{
      neutron::relation::{
        LookupPayload, LookupRunningWitness, LookupShape, LookupTableHandle,
        MultiColumnLookupTable,
      },
      spartan::polys::power::PowPolynomial,
      traits::commitment::CommitmentEngineTrait,
    };
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, HyperKZGEE<E>>;

    let mut rng = ChaCha20Rng::seed_from_u64(0x1A55_E69_E5);
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = Scalar::ZERO;

    let num_cons = 32usize;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // Merged 32-row table = (sub_table_id=0: 16 rows) || (sub_table_id=1: 16 rows).
    // Address `i ∈ [0, 32)` decomposes into (high_bit=sub_table, low_4_bits=offset).
    let merged_size = 32usize;
    let merged_log2 = 5usize;
    let mut merged_t1: Vec<Scalar> = Vec::with_capacity(merged_size);
    for i in 0..merged_size {
      // Sub-table 0 uses (11i+5); sub-table 1 uses (17i+9). Different formulas
      // for clarity — the merge just concatenates rows under one identity.
      let sub_table_id = (i >> 4) & 0x1;
      let offset = i & 0xF;
      let v = if sub_table_id == 0 {
        offset * 11 + 5
      } else {
        offset * 17 + 9
      };
      merged_t1.push(Scalar::from(v as u64));
    }
    let merged_identity: Vec<Scalar> =
      (0..merged_size).map(|i| Scalar::from(i as u64)).collect();
    let merged_identity_comm =
      <E as Engine>::CE::commit(&ck, &merged_identity, &Scalar::ZERO);
    let merged_t1_comm =
      <E as Engine>::CE::commit(&ck, &merged_t1, &Scalar::ZERO);

    let lookup_shape = LookupShape::<E> {
      tables: vec![LookupTableHandle {
        table_id: 0,
        size: merged_size,
        commitment: merged_identity_comm,
      }],
      multi_column_tables: vec![MultiColumnLookupTable {
        table_id: 0,
        size: merged_size,
        columns: vec![merged_t1.clone()],
        value_commitments: vec![merged_t1_comm],
      }],
      num_addr_columns: 1,
      num_witness_columns: 2,
      witness_ell_cached: merged_log2,
    };
    let str = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str.S.clone();

    // Two satisfying R1CS pairs.
    let circuit1: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs1 = SatisfyingAssignment::<E>::new();
    let _ = circuit1.synthesize(&mut cs1);
    let (U1_r1cs, W1_r1cs) = cs1.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W1_r1cs = W1_r1cs.pad(&shape);

    let circuit2: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(3)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs2 = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut cs2);
    let (U2_r1cs, W2_r1cs) = cs2.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W2_r1cs = W2_r1cs.pad(&shape);

    let mut running_W = FoldedWitness::default(&str);
    let mut running_U = FoldedInstance::default(&str);
    let mut running_lw = LookupRunningWitness::default(&lookup_shape);

    // Helper closure to build a payload+witness for an arbitrary in-range
    // set of merged-table addresses.
    let build = |query_indices: &[usize], rng: &mut ChaCha20Rng| {
      let mut witness_addr = vec![Scalar::ZERO; merged_size];
      let mut witness_v1 = vec![Scalar::ZERO; merged_size];
      let mut multiplicities = vec![Scalar::ZERO; merged_size];
      for (i, &idx) in query_indices.iter().enumerate() {
        witness_addr[i] = Scalar::from(idx as u64);
        witness_v1[i] = merged_t1[idx];
        multiplicities[idx] += Scalar::ONE;
      }
      for i in query_indices.len()..merged_size {
        witness_addr[i] = Scalar::from(0u64);
        witness_v1[i] = merged_t1[0];
        multiplicities[0] += Scalar::ONE;
      }
      let comm_addr = <E as Engine>::CE::commit(&ck, &witness_addr, &Scalar::ZERO);
      let comm_v1 = <E as Engine>::CE::commit(&ck, &witness_v1, &Scalar::ZERO);
      let comm_ts = <E as Engine>::CE::commit(&ck, &multiplicities, &Scalar::ZERO);
      let payload = LookupPayload::<E> {
        comm_L: comm_addr,
        comm_ts,
        comm_inv_w: Commitment::<E>::default(),
        comm_inv_t: Commitment::<E>::default(),
        T2_lookup: Scalar::ZERO,
        comm_values: vec![comm_v1],
      };
      let tau_w = Scalar::random(&mut *rng);
      let pow_w = PowPolynomial::new(&tau_w, merged_log2);
      let (w_left, w_right) = lookup_shape.witness_split();
      let combined_w = pow_w.split_evals(w_left, w_right);
      let (eq_w_left, eq_w_right) = combined_w.split_at(w_left);
      let tau_t = Scalar::random(&mut *rng);
      let pow_t = PowPolynomial::new(&tau_t, merged_log2);
      let (t_left, t_right) = lookup_shape.table_split();
      let combined_t_eq = pow_t.split_evals(t_left, t_right);
      let (eq_t_left, eq_t_right) = combined_t_eq.split_at(t_left);
      (
        payload,
        witness_addr,
        witness_v1,
        multiplicities,
        eq_w_left.to_vec(),
        eq_w_right.to_vec(),
        eq_t_left.to_vec(),
        eq_t_right.to_vec(),
      )
    };

    // Step 1 queries SUB-TABLE 0 (high bit = 0; addresses [0, 16)).
    let q1 = vec![0usize, 3, 7, 15];
    let (mut p1, wa1, wv1, m1, e1wl, e1wr, e1tl, e1tr) = build(&q1, &mut rng);
    let (nifs1, (folded_U1, folded_W1), folded_lw1) =
      NIFS::<E>::prove_with_multi_column_lookup(
        &ck,
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &running_W,
        &U1_r1cs,
        &W1_r1cs,
        &p1,
        &running_lw,
        0u64,
        &wa1,
        &[wv1.clone()],
        &m1,
        e1wl,
        e1wr,
        e1tl,
        e1tr,
      )
      .expect("step 1 (sub-table 0) prove must succeed");
    p1.comm_inv_w = nifs1.comm_inv_w.as_ref().unwrap()[0];
    p1.comm_inv_t = nifs1.comm_inv_t.as_ref().unwrap()[0];
    let v1 = nifs1
      .verify_with_multi_column_lookup(
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &U1_r1cs,
        &p1,
        0u64,
      )
      .expect("step 1 (sub-table 0) verify must succeed");
    assert_eq!(folded_U1, v1);
    str.is_sat(&ck, &folded_U1, &folded_W1).expect("step 1 R1CS sat");
    running_U = folded_U1;
    running_W = folded_W1;
    running_lw = folded_lw1;

    // Step 2 queries SUB-TABLE 1 (high bit = 1; addresses [16, 32)).
    let q2 = vec![16usize, 19, 23, 31];
    let (mut p2, wa2, wv2, m2, e2wl, e2wr, e2tl, e2tr) = build(&q2, &mut rng);
    let (nifs2, (folded_U2, folded_W2), _) =
      NIFS::<E>::prove_with_multi_column_lookup(
        &ck,
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &running_W,
        &U2_r1cs,
        &W2_r1cs,
        &p2,
        &running_lw,
        0u64,
        &wa2,
        &[wv2.clone()],
        &m2,
        e2wl,
        e2wr,
        e2tl,
        e2tr,
      )
      .expect("step 2 (sub-table 1) prove must succeed");
    p2.comm_inv_w = nifs2.comm_inv_w.as_ref().unwrap()[0];
    p2.comm_inv_t = nifs2.comm_inv_t.as_ref().unwrap()[0];
    let v2 = nifs2
      .verify_with_multi_column_lookup(
        &ro_consts,
        &pp_digest,
        &str,
        &running_U,
        &U2_r1cs,
        &p2,
        0u64,
      )
      .expect("step 2 (sub-table 1) verify must succeed");
    assert_eq!(folded_U2, v2);
    str.is_sat(&ck, &folded_U2, &folded_W2).expect("step 2 R1CS sat");
  }

  /// GH-#2 M.3: design pin §5.2 #2 byte-equivalence regression check
  /// at the FS-transcript layer.
  ///
  /// At k=1 the multi-table prove path
  /// (`prove_with_multi_table_lookup_inner`) MUST produce a byte-identical
  /// FS transcript to the existing single-table multi-column path
  /// (`prove_with_multi_column_lookup_inner`). The pin is explicit that
  /// the comparison is at the **FS-transcript layer**, NOT serialized
  /// `NIFS<E>` bytes — the multi-table struct expansion (`Vec<UniPoly>`,
  /// `Vec<Commitment>`) changes the bincode envelope by design at k=1.
  ///
  /// We verify byte-equivalence at the FS-transcript layer by asserting
  /// that the tuple of values that flow through the transcript at every
  /// absorb / squeeze step are equal across the two paths under
  /// identical inputs and a pinned-deterministic `r_E`. Specifically:
  ///
  /// - `nifs.comm_E` (Step 3 absorb) — depends on `tau` (squeeze 0) and
  ///   `r_E` (input).
  /// - `nifs.poly` (Step 7 absorb) — depends on `tau`, `comm_E`, `rho`
  ///   (squeeze 1), and the R1CS state.
  /// - `nifs.poly_lookup[0]` (Step 8 absorb) — depends on the entire
  ///   transcript prefix through `r_logup`.
  /// - `nifs.comm_inv_w[0]` and `nifs.comm_inv_t[0]` (Step 6 absorb)
  ///   — depend on `r_logup` (squeeze 4 in single-column).
  /// - The final folded `U` — depends on `r_b` (Step 9 squeeze).
  ///
  /// Any divergence in transcript byte order between the two paths would
  /// propagate Fiat-Shamir-deterministically to one of these values
  /// breaking the equality assertion. The chain of squeezes (`tau`,
  /// `rho`, `α` (gated), `r_logup`, `r_b`) is exhaustive — if all
  /// post-squeeze values agree, every intermediate transcript byte
  /// agreed too.
  ///
  /// This test deliberately uses `_inner` variants of both prove
  /// functions so the eq-polynomial blinding `r_E` is pinned to a
  /// `ChaCha20Rng`-derived deterministic scalar. Production-facing
  /// (`prove_with_multi_*_lookup`) callers go through `OsRng` and never
  /// expose this seam.
  #[cfg(feature = "lookup-fold")]
  #[test]
  fn m3_byte_equivalence_k1_multi_table_matches_multi_column() {
    use crate::neutron::relation::{
      LookupPayload, LookupRunningWitness, LookupShape, LookupTableHandle,
      MultiColumnLookupTable,
    };
    use crate::spartan::polys::power::PowPolynomial;
    use crate::traits::commitment::CommitmentEngineTrait;
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    type E = Bn256EngineKZG;
    type Scalar = <E as Engine>::Scalar;
    type S = RelaxedR1CSSNARK<E, HyperKZGEE<E>>;

    // Pinned-seed RNG so the test is reviewer-reproducible per
    // `.claude/rules/cryptography.md`'s determinism requirement.
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1B2_F003);
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = Scalar::ZERO;

    // R1CS shape (mirrors `stage_i_pri_multi_column_round_trip`).
    let num_cons = 32usize;
    let circuit: DirectCircuit<E, NonTrivialCircuit<Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<Scalar>::new(num_cons));
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // Single multi-column table (k = 1) — the regression target. 16
    // entries, 2 value columns.
    let table_size = 16usize;
    let table_log2 = 4usize;
    let t1: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 11 + 5) as u64))
      .collect();
    let t2: Vec<Scalar> = (0..table_size)
      .map(|i| Scalar::from((i * 17 + 9) as u64))
      .collect();
    let identity: Vec<Scalar> = (0..table_size).map(|i| Scalar::from(i as u64)).collect();
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
    let str_local = Structure::new_with_lookups(&shape, lookup_shape.clone());
    let shape = str_local.S.clone();

    // Satisfying R1CS instance.
    let circuit2: DirectCircuit<E, NonTrivialCircuit<Scalar>> = DirectCircuit::new(
      Some(vec![Scalar::from(2)]),
      NonTrivialCircuit::<Scalar>::new(num_cons),
    );
    let mut cs2 = SatisfyingAssignment::<E>::new();
    let _ = circuit2.synthesize(&mut cs2);
    let (U2, W2) = cs2.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let W2 = W2.pad(&shape);

    // Outer-base running state.
    let running_W = FoldedWitness::default(&str_local);
    let running_U = FoldedInstance::default(&str_local);
    let running_lw = LookupRunningWitness::default(&lookup_shape);

    // Build a satisfying multi-column witness pool — 4 in-table queries
    // padded to table_size.
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

    // eq polynomials.
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

    // Pin the eq-polynomial blinding `r_E` deterministically — both
    // paths receive the SAME scalar, so any FS-transcript divergence
    // surfaces in the post-squeeze values rather than being masked by
    // distinct OsRng draws.
    let r_E_pinned = Scalar::random(&mut rng);

    // --- Path A: existing single-table multi-column inner ---
    let (nifs_single, (folded_U_single, folded_W_single), folded_lw_single) =
      NIFS::<E>::prove_with_multi_column_lookup_inner(
        &ck,
        &ro_consts,
        &pp_digest,
        &str_local,
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
        r_E_pinned,
      )
      .expect("multi-column inner must succeed on satisfying witness");

    // --- Path B: new multi-table inner (k=1) ---
    let bundle = super::PerTableBundle::<E> {
      table_id: 0,
      payload: payload.clone(),
      fresh_witness_address: witness_addr.clone(),
      fresh_witness_value_columns: vec![witness_v1.clone(), witness_v2.clone()],
      fresh_multiplicities: multiplicities.clone(),
      fresh_eq_w_left: eq_w_left.to_vec(),
      fresh_eq_w_right: eq_w_right.to_vec(),
      fresh_eq_t_left: eq_t_left.to_vec(),
      fresh_eq_t_right: eq_t_right.to_vec(),
      running_lw: running_lw.clone(),
    };
    let (nifs_multi, (folded_U_multi, folded_W_multi), folded_lw_multi_per_table) =
      NIFS::<E>::prove_with_multi_table_lookup_inner(
        &ck,
        &ro_consts,
        &pp_digest,
        &str_local,
        &running_U,
        &running_W,
        &U2,
        &W2,
        &[bundle],
        r_E_pinned,
      )
      .expect("multi-table inner must succeed on satisfying witness at k=1");

    // --- Pin §5.2 #2: FS-transcript layer byte-equivalence ---
    //
    // Each assertion is gated on a value that absorbs into / squeezes
    // out of the transcript. Equality across all of them is sufficient
    // and necessary for byte-equivalent FS transcripts at k=1.
    //
    // (Step 3) `comm_E` is the absorbed eq-commitment — equal here
    // implies tau (squeeze 0) was equal, and `r_E` was equal (pinned).
    assert_eq!(
      nifs_multi.comm_E, nifs_single.comm_E,
      "Step 3: comm_E must be byte-identical at k=1 (pin §5.2 #2)"
    );

    // Pin §5.1: multi-table NIFS extends to Vec; at k=1 the Vec carries
    // exactly one element which must equal the single-table singleton.
    let multi_inv_w = nifs_multi
      .comm_inv_w
      .as_ref()
      .expect("multi-table NIFS must carry comm_inv_w");
    let single_inv_w = nifs_single
      .comm_inv_w
      .as_ref()
      .expect("single-table NIFS must carry comm_inv_w");
    assert_eq!(
      multi_inv_w.len(),
      1,
      "Step 6: multi-table comm_inv_w Vec length must be 1 at k=1"
    );
    assert_eq!(
      single_inv_w.len(),
      1,
      "Step 6: single-table comm_inv_w Vec length must be 1 (post-M.1)"
    );
    assert_eq!(
      multi_inv_w[0], single_inv_w[0],
      "Step 6: comm_inv_w[0] must be byte-identical at k=1 (pin §5.2 #2)"
    );

    let multi_inv_t = nifs_multi
      .comm_inv_t
      .as_ref()
      .expect("multi-table NIFS must carry comm_inv_t");
    let single_inv_t = nifs_single
      .comm_inv_t
      .as_ref()
      .expect("single-table NIFS must carry comm_inv_t");
    assert_eq!(
      multi_inv_t[0], single_inv_t[0],
      "Step 6: comm_inv_t[0] must be byte-identical at k=1 (pin §5.2 #2)"
    );

    // (Step 7) `poly` is the absorbed R1CS sumcheck poly — equal here
    // implies rho (squeeze 1) was equal AND alpha (squeeze 5a, gated)
    // was equal — both sides squeeze alpha because comm_values is
    // non-empty here.
    assert_eq!(
      nifs_multi.poly, nifs_single.poly,
      "Step 7: R1CS sumcheck poly must be byte-identical at k=1 (pin §5.2 #2)"
    );

    // (Step 8) `poly_lookup_j` for j=0 is the absorbed lookup sumcheck
    // poly — equal here implies r_logup (squeeze 5b) and the running
    // scalar threading at outer base agreed.
    let multi_poly_lookup = nifs_multi
      .poly_lookup
      .as_ref()
      .expect("multi-table NIFS must carry poly_lookup");
    let single_poly_lookup = nifs_single
      .poly_lookup
      .as_ref()
      .expect("single-table NIFS must carry poly_lookup");
    assert_eq!(
      multi_poly_lookup.len(),
      1,
      "Step 8: multi-table poly_lookup Vec length must be 1 at k=1"
    );
    assert_eq!(
      single_poly_lookup.len(),
      1,
      "Step 8: single-table poly_lookup Vec length must be 1 (post-M.1)"
    );
    assert_eq!(
      multi_poly_lookup[0], single_poly_lookup[0],
      "Step 8: poly_lookup[0] must be byte-identical at k=1 (pin §5.2 #2)"
    );

    // (Step 9) The folded FoldedInstance — depends on r_b (final
    // squeeze). Equal folded U implies r_b agreed, AND every
    // intermediate transcript prefix agreed.
    assert_eq!(
      folded_U_multi, folded_U_single,
      "Step 9: folded U must be byte-identical at k=1 (pin §5.2 #2)"
    );

    // The folded W must also agree (depends on the same r_b).
    assert_eq!(
      folded_W_multi, folded_W_single,
      "Step 9: folded W must be byte-identical at k=1 (pin §5.2 #2)"
    );

    // The per-table running witness vector at k=1 carries one entry,
    // which must equal the single-table folded running witness.
    assert_eq!(
      folded_lw_multi_per_table.len(),
      1,
      "M.3 contract: per-table folded running witness Vec length must be 1 at k=1"
    );
    assert_eq!(
      folded_lw_multi_per_table[0], folded_lw_single,
      "M.3 contract: folded running witness must be byte-identical at k=1"
    );

    // Sanity: T_lookup VECTOR is one-element at k=1 and matches the
    // single-table M.1+M.2 storage shape.
    let t_lookup_multi = folded_U_multi
      .T_lookup
      .as_ref()
      .expect("k=1 multi-table fold must populate T_lookup");
    let t_lookup_single = folded_U_single
      .T_lookup
      .as_ref()
      .expect("k=1 single-table-multi-column fold must populate T_lookup");
    assert_eq!(t_lookup_multi.len(), 1, "k=1: T_lookup must have length 1");
    assert_eq!(t_lookup_single.len(), 1, "k=1: T_lookup must have length 1");
    assert_eq!(
      t_lookup_multi[0], t_lookup_single[0],
      "k=1: per-table T_lookup_0 must be byte-identical (pin §5.2 #2)"
    );
  }
}

#[cfg(test)]
mod benchmarks {
  use super::*;
  use crate::{
    frontend::{
      gadgets::{
        boolean::{AllocatedBit, Boolean},
        num::AllocatedNum,
        sha256::sha256,
      },
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      ConstraintSystem, SynthesisError,
    },
    nova::nifs::NIFS as NovaNIFS,
    provider::Bn256EngineKZG,
    r1cs::{R1CSShape, SparseMatrix},
    traits::{snark::default_ck_hint, ROConstants},
  };
  use core::marker::PhantomData;
  use criterion::Criterion;
  use ff::PrimeField;
  use num_integer::Integer;
  use num_traits::ToPrimitive;
  use rand::Rng;

  /// generates a satisfying R1CS with small witness values
  fn generate_sample_r1cs<E: Engine>(
    num_cons: usize,
  ) -> (
    R1CSShape<E>,
    CommitmentKey<E>,
    R1CSWitness<E>,
    Vec<u8>,
    Vec<E::Scalar>,
  ) {
    let num_vars = num_cons;
    let num_io = 1;

    // we will just generate constraints of the form x * x = x, checking Booleanity
    // generate the constraints by creating sparse matrices
    let A = SparseMatrix::new(
      &(0..num_cons)
        .map(|i| (i, i, E::Scalar::ONE))
        .collect::<Vec<_>>(),
      num_cons,
      num_vars + 1 + num_io,
    );
    let B = A.clone();
    let C = A.clone();

    let S: R1CSShape<E> = R1CSShape::new(num_cons, num_vars, num_io, A, B, C).unwrap();

    let S = S.pad();

    // sample a ck
    let ck = R1CSShape::commitment_key(&[&S], &[&*default_ck_hint()]).unwrap();

    // let witness be randomly generated booleans
    let w = (0..S.num_cons)
      .into_par_iter()
      .map(|_| {
        let mut rng = rand::thread_rng();
        rng.gen::<u8>() % 2
      })
      .collect::<Vec<_>>();

    let W = {
      // convert W to field elements
      let W = (0..S.num_cons)
        .into_par_iter()
        .map(|i| <E as Engine>::Scalar::from(w[i] as u64))
        .collect::<Vec<_>>();
      R1CSWitness::new(&S, &W).unwrap()
    };

    let x = vec![E::Scalar::from(0)];
    (S, ck, W, w, x)
  }

  struct Sha256Circuit<E: Engine> {
    preimage: Vec<u8>,
    _p: PhantomData<E>,
  }

  impl<E: Engine> Sha256Circuit<E> {
    pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
    ) -> Result<(), SynthesisError> {
      // we write a circuit that checks if the input is a SHA256 preimage
      let bit_values: Vec<_> = self
        .preimage
        .clone()
        .into_iter()
        .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
        .map(Some)
        .collect();
      assert_eq!(bit_values.len(), self.preimage.len() * 8);

      let preimage_bits = bit_values
        .into_iter()
        .enumerate()
        .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
        .map(|b| b.map(Boolean::from))
        .collect::<Result<Vec<_>, _>>()?;

      let _ = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::ZERO))?;
      x.inputize(cs.namespace(|| "inputize x"))?;

      Ok(())
    }
  }

  fn generarate_sha_r1cs<E: Engine>(
    len: usize,
  ) -> (
    R1CSShape<E>,
    CommitmentKey<E>,
    R1CSWitness<E>,
    Vec<u8>,
    Vec<E::Scalar>,
  ) {
    let circuit = Sha256Circuit::<E> {
      preimage: vec![0u8; len],
      _p: Default::default(),
    };

    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let S = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&S], &[&*default_ck_hint()]).unwrap();

    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U, W) = cs.r1cs_instance_and_witness(&S, &ck).unwrap();

    let S = S.pad();
    let W = W.pad(&S);

    let w = W
      .W
      .iter()
      .map(|e| {
        // map field element to u8
        // this assumes little-endian representation
        e.to_repr().as_ref()[0]
      })
      .collect::<Vec<_>>();

    // sanity check by recommiting to w
    let comm_W = <E as Engine>::CE::commit_small(&ck, &w, &W.r_W);
    assert_eq!(comm_W, U.comm_W);

    let X = U.X.clone();
    (S, ck, W, w, X)
  }

  fn bench_nifs_inner<E: Engine, T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    c: &mut Criterion,
    name: &str,
    S: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    W: &R1CSWitness<E>,
    w: &[T],
    x: &[E::Scalar],
  ) {
    let num_cons = S.num_cons;

    // generate a default running instance
    let str = Structure::new(S);
    let f_W = FoldedWitness::default(&str);
    let f_U = FoldedInstance::default(&str);
    let res = str.is_sat(ck, &f_U, &f_W);
    assert!(res.is_ok());

    // generate default values
    let pp_digest = E::Scalar::ZERO;
    let ro_consts = RO2Constants::<E>::default();

    // produce an NIFS with (W, U) as the first incoming witness-instance pair
    c.bench_function(&format!("neutron_nifs_{name}_{num_cons}"), |b| {
      b.iter(|| {
        // commit with the specialized method
        let comm_W = E::CE::commit_small(ck, w, &W.r_W);

        // make an R1CS instance
        let U = R1CSInstance::new(S, &comm_W, x).unwrap();

        let res = NIFS::prove(ck, &ro_consts, &pp_digest, &str, &f_U, &f_W, &U, W);
        assert!(res.is_ok());
      })
    });

    // generate a random relaxed R1CS instance-witness pair
    let (r_U, r_W) = R1CSShape::<E>::sample_random_instance_witness(S, ck).unwrap();
    let ro_consts = ROConstants::<E>::default();

    // produce an NIFS with (r_W, r_U) as the second incoming witness-instance pair
    c.bench_function(&format!("nova_nifs_{name}_{num_cons}"), |b| {
      b.iter(|| {
        // commit to R1CS witness
        let comm_W = W.commit(ck);

        // make an R1CS instance
        let U = R1CSInstance::new(S, &comm_W, x).unwrap();

        let res = NovaNIFS::prove(ck, &ro_consts, &pp_digest, S, &r_U, &r_W, &U, W);
        assert!(res.is_ok());
      })
    });
  }

  #[test]
  fn bench_nifs_simple() {
    type E = Bn256EngineKZG;

    let mut criterion = Criterion::default();
    let num_cons = 1024;
    let (S, ck, W, w, x) = generate_sample_r1cs::<E>(num_cons); // W is R1CSWitness, w is a vector of u8, x is a vector of field elements
    bench_nifs_inner(&mut criterion, "simple", &S, &ck, &W, &w, &x);
  }

  #[test]
  fn bench_nifs_sha256() {
    type E = Bn256EngineKZG;

    let mut criterion = Criterion::default();
    for len in [32, 64].iter() {
      let (S, ck, W, w, x) = generarate_sha_r1cs::<E>(*len);
      bench_nifs_inner(&mut criterion, "sha256", &S, &ck, &W, &w, &x);
    }
  }
}
