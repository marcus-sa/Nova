//! This module implements a non-interactive folding scheme from NeutronNova
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

/// An NIFS message from NeutronNova's folding scheme
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E: Engine> {
  pub(crate) comm_E: Commitment<E>,
  pub(crate) poly: UniPoly<E::Scalar>,

  /// Lookup-fold extension fields (Stage C, C1-beta).
  /// Present only when a lookup payload was supplied to the fold step.
  #[cfg(feature = "lookup-fold")]
  pub(crate) poly_lookup: Option<UniPoly<E::Scalar>>,
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_w: Option<Commitment<E>>,
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_t: Option<Commitment<E>>,
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
    let t_lookup_running = lookup_running_claims_from::<E>(U1);
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
    };
    let U = U1.fold_with_lookup(U2, &comm_E, &r_b, &T_out, &effective_payload, &T_lookup_out)?;
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

    let nifs = NIFS {
      comm_E,
      poly,
      poly_lookup: Some(poly_lookup),
      comm_inv_w: Some(comm_inv_w2),
      comm_inv_t: Some(comm_inv_t2),
    };

    Ok((nifs, (U, W), folded_lw))
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
    let comm_inv_w = self
      .comm_inv_w
      .as_ref()
      .ok_or(NovaError::InvalidSumcheckProof)?;
    let comm_inv_t = self
      .comm_inv_t
      .as_ref()
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
    let poly_lookup = self
      .poly_lookup
      .as_ref()
      .ok_or(NovaError::InvalidSumcheckProof)?;

    let t_lookup_running = lookup_running_claims_from::<E>(U1);

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
    let U = U1.fold_with_lookup(U2, &self.comm_E, &r_b, &T_out, payload, &T_lookup_out)?;

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
    payload1.comm_inv_w = nifs1.comm_inv_w.unwrap();
    payload1.comm_inv_t = nifs1.comm_inv_t.unwrap();

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
      folded_U1.T_lookup.unwrap()
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
    payload2.comm_inv_w = nifs2.comm_inv_w.unwrap();
    payload2.comm_inv_t = nifs2.comm_inv_t.unwrap();

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
      folded_U2.T_lookup.unwrap()
    );

    // R1CS-side satisfiability check
    let res = str.is_sat(&ck, &folded_U2, &folded_W2);
    assert!(res.is_ok(), "folded instance must be satisfying after step 2: {:?}", res);

    println!("execute_sequence_with_lookup: both fold steps passed prove/verify/sat checks");
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
