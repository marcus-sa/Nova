//! Circuit representation of NeutronNova's NIFS
use crate::{
  constants::NUM_CHALLENGE_BITS,
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{ecc::AllocatedNonnativePoint, utils::le_bits_to_num},
  neutron::{
    circuit::{
      r1cs::AllocatedNonnativeR1CSInstance, relation::AllocatedFoldedInstance,
      univariate::AllocatedUniPoly,
    },
    nifs::NIFS,
  },
  traits::{commitment::CommitmentTrait, Engine, RO2ConstantsCircuit, ROCircuitTrait},
};
use ff::Field;

/// An in-circuit representation of NeutronNova's NIFS
pub struct AllocatedNIFS<E: Engine> {
  pub(crate) comm_E: AllocatedNonnativePoint<E>,
  pub(crate) poly: AllocatedUniPoly<E>,
}

impl<E: Engine> AllocatedNIFS<E> {
  /// Allocates the given `NIFS` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    nifs: Option<&NIFS<E>>,
    degree: usize,
  ) -> Result<Self, SynthesisError> {
    let comm_E = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_E"),
      nifs.map(|nifs| nifs.comm_E.to_coordinates()),
    )?;

    // Allocate the polynomial
    let poly = AllocatedUniPoly::alloc(
      cs.namespace(|| "allocate poly"),
      degree,
      nifs.map(|nifs| &nifs.poly),
    )?;

    Ok(Self { comm_E, poly })
  }

  /// verify the provided NIFS inside the circuit
  pub fn verify<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>, // verifier key
    U1: &AllocatedFoldedInstance<E>,     // folded instance
    U2: &AllocatedNonnativeR1CSInstance<E>,
    comm_W_fold: &AllocatedNonnativePoint<E>, // untrusted hint
    comm_E_fold: &AllocatedNonnativePoint<E>, // untrusted hint
    ro_consts: RO2ConstantsCircuit<E>,
  ) -> Result<AllocatedFoldedInstance<E>, SynthesisError> {
    // Compute r:
    let mut ro = E::RO2Circuit::new(ro_consts);
    ro.absorb(pp_digest);

    // running instance `U1` does not need to absorbed since U2.X[0] = Hash(vk, U1, i, z0, zi)
    U2.absorb_in_ro(cs.namespace(|| "absorb U2"), &mut ro)?;

    // generate a challenge for the eq polynomial
    let _tau = ro.squeeze(cs.namespace(|| "tau"), NUM_CHALLENGE_BITS, false);

    // TODO: We will check the power-check instance contains tau as public IO later

    // absorb the commitment in the NIFS
    self
      .comm_E
      .absorb_in_ro(cs.namespace(|| "absorb comm_E"), &mut ro)?;

    // squeeze a challenge from the RO
    let rho_bits = ro.squeeze(cs.namespace(|| "rho_bits"), NUM_CHALLENGE_BITS, false)?;
    let rho = le_bits_to_num(cs.namespace(|| "rho"), &rho_bits)?;

    // T = (1-rho) * U1.T + rho * U2.T, but U2.T = 0
    let T = AllocatedNum::alloc(cs.namespace(|| "allocate T"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U1_T = U1.T.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(U1_T * (E::Scalar::ONE - rho))
    })?;
    cs.enforce(
      || "enforce T = (1-rho) * U1.T",
      |lc| lc + U1.T.get_variable(),
      |lc| lc + CS::one() - rho.get_variable(),
      |lc| lc + T.get_variable(),
    );

    self
      .poly
      .check_poly_zero_poly_one_with(cs.namespace(|| "poly_at_zero + poly_at_one = T"), &T)?;

    // absorb poly in the RO
    self.poly.absorb_in_ro(&mut ro);

    // squeeze a challenge
    let r_b_bits = ro.squeeze(cs.namespace(|| "r_b_bits"), NUM_CHALLENGE_BITS, false)?;
    let r_b = le_bits_to_num(cs.namespace(|| "r_b"), &r_b_bits)?;

    // compute the sum-check polynomial's evaluations at r_b
    // let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let eq_rho_r_b_one = AllocatedNum::alloc(cs.namespace(|| "allocate eq_rho_r_b_one"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b))
    })?;
    cs.enforce(
      || "check eq_rho_r_b_one = (1 - rho) * (1 - r_b)",
      |lc| lc + CS::one() - rho.get_variable(),
      |lc| lc + CS::one() - r_b.get_variable(),
      |lc| lc + eq_rho_r_b_one.get_variable(),
    );

    let eq_rho_r_b = AllocatedNum::alloc(cs.namespace(|| "allocate eq_rho_r_b"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b)
    })?;

    // check eq_rho_r_b = (1 - rho) * (1 - r_b) + rho * r_b
    cs.enforce(
      || "check eq_rho_r_b = (1 - rho) * (1 - r_b) + rho * r_b",
      |lc| lc + rho.get_variable(),
      |lc| lc + r_b.get_variable(),
      |lc| lc + eq_rho_r_b.get_variable() - eq_rho_r_b_one.get_variable(),
    );

    // let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();
    let eval = { self.poly.evaluate(cs.namespace(|| "eval"), &r_b)? };
    let T_out = AllocatedNum::alloc(cs.namespace(|| "allocate T_out"), || {
      let eval = eval.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let eq_rho_r_b_inv = eq_rho_r_b
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?
        .invert()
        .unwrap();
      Ok(eval * eq_rho_r_b_inv)
    })?;
    cs.enforce(
      || "enforce T_out * eq_rho_r_b = eval",
      |lc| lc + T_out.get_variable(),
      |lc| lc + eq_rho_r_b.get_variable(),
      |lc| lc + eval.get_variable(),
    );

    // let U = U1.fold(U2, &self.comm_E, &r_b, &T_out)?;
    let U = U1.fold(
      cs.namespace(|| "fold"),
      U2,
      &r_b,
      &T_out,
      comm_W_fold,
      comm_E_fold,
    )?;

    // return the folded instance and witness
    Ok(U)
  }
}

#[cfg(feature = "lookup-fold")]
#[allow(dead_code)]
mod lookup_verify {
  use super::*;
  use crate::neutron::circuit::lookup::{
    AllocatedLookupNIFS, AllocatedLookupNIFSMultiTable,
    AllocatedLookupPayloadPublicMultiTable,
  };

  /// Result of `AllocatedNIFS::verify_with_lookup`: the folded R1CS-zero
  /// instance and the next-step lookup-zero running target. The augmented
  /// circuit attaches `T_lookup_out` to its public-input hash (Stage G+H).
  pub struct LookupVerifyOutput<E: Engine> {
    pub U_fold: AllocatedFoldedInstance<E>,
    pub T_lookup_out: AllocatedNum<E::Scalar>,
  }

  impl<E: Engine> AllocatedNIFS<E> {
    /// In-circuit lookup-aware verifier. Mirrors the native
    /// `NIFS::verify_with_lookup` (`src/neutron/nifs.rs:591-672`) with the
    /// transcript order spelled out in §G.2 of the C1-β implementation
    /// outline:
    ///
    /// ```text
    ///   ro.absorb(pp_digest)
    ///   U2.absorb_in_ro
    ///   payload.comm_L.absorb_in_ro       [lookup-fold]
    ///   payload.comm_ts.absorb_in_ro      [lookup-fold]
    ///   ro.squeeze() -> tau
    ///   comm_E.absorb_in_ro
    ///   ro.squeeze() -> rho
    ///   ro.squeeze() -> r_logup           [lookup-fold]
    ///   comm_inv_w.absorb_in_ro           [lookup-fold]
    ///   comm_inv_t.absorb_in_ro           [lookup-fold]
    ///   poly.check_poly_zero_poly_one_with(T)
    ///   poly_lookup.check_poly_zero_poly_one_with(t_lookup_running)  [lookup-fold]
    ///   poly.absorb_in_ro
    ///   poly_lookup.absorb_in_ro          [lookup-fold]
    ///   ro.squeeze() -> r_b
    /// ```
    ///
    /// `comm_L_pub` / `comm_ts_pub` are absorbed-only commitments carried as
    /// public inputs (`LookupPayloadPublic`); the augmented circuit does not
    /// fold them in-circuit (the native fold-of-commitments lives in
    /// `FoldedInstance::fold_with_lookup`, which the augmented circuit
    /// trusts via the standard Nova hash check on the next-step running
    /// instance).
    ///
    /// `t_lookup_running` is the lookup-side running target projected from
    /// the running instance (cf. native `lookup_running_claims_from`). At
    /// outer base, this is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_with_lookup<CS: ConstraintSystem<E::Scalar>>(
      &self,
      mut cs: CS,
      pp_digest: &AllocatedNum<E::Scalar>,
      U1: &AllocatedFoldedInstance<E>,
      U2: &AllocatedNonnativeR1CSInstance<E>,
      lookup: &AllocatedLookupNIFS<E>,
      comm_L_pub: &AllocatedNonnativePoint<E>,
      comm_ts_pub: &AllocatedNonnativePoint<E>,
      t_lookup_running: &AllocatedNum<E::Scalar>,
      comm_W_fold: &AllocatedNonnativePoint<E>,
      comm_E_fold: &AllocatedNonnativePoint<E>,
      ro_consts: RO2ConstantsCircuit<E>,
    ) -> Result<LookupVerifyOutput<E>, SynthesisError> {
      let mut ro = E::RO2Circuit::new(ro_consts);
      ro.absorb(pp_digest);

      // U1 absorption is implicit via U2.X[0] = Hash(vk, U1, i, z0, zi);
      // mirror non-lookup verify (see comment at line 60).
      U2.absorb_in_ro(cs.namespace(|| "absorb U2"), &mut ro)?;

      // --- Step (2): absorb lookup-side public commitments BEFORE tau squeeze ---
      comm_L_pub.absorb_in_ro(cs.namespace(|| "absorb comm_L"), &mut ro)?;
      comm_ts_pub.absorb_in_ro(cs.namespace(|| "absorb comm_ts"), &mut ro)?;

      // squeeze tau (R1CS-side eq challenge)
      let _tau = ro.squeeze(cs.namespace(|| "tau"), NUM_CHALLENGE_BITS, false)?;

      // absorb comm_E from the NIFS message
      self
        .comm_E
        .absorb_in_ro(cs.namespace(|| "absorb comm_E"), &mut ro)?;

      // squeeze rho (R1CS- and lookup-side fold challenge)
      let rho_bits = ro.squeeze(cs.namespace(|| "rho_bits"), NUM_CHALLENGE_BITS, false)?;
      let rho = le_bits_to_num(cs.namespace(|| "rho"), &rho_bits)?;

      // --- Step (5): squeeze r_logup BEFORE absorbing inverse commitments ---
      let _r_logup_bits =
        ro.squeeze(cs.namespace(|| "r_logup_bits"), NUM_CHALLENGE_BITS, false)?;

      // --- Step (6): absorb inverse-witness commitments ---
      lookup.absorb_inv_comms_in_ro(cs.namespace(|| "absorb inv comms"), &mut ro)?;

      // --- R1CS-side (C)-binding: poly(0)+poly(1) = T = (1-rho)*U1.T ---
      let T = AllocatedNum::alloc(cs.namespace(|| "allocate R1CS T"), || {
        let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let U1_T = U1.T.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(U1_T * (E::Scalar::ONE - rho_v))
      })?;
      cs.enforce(
        || "enforce R1CS T = (1-rho) * U1.T",
        |lc| lc + U1.T.get_variable(),
        |lc| lc + CS::one() - rho.get_variable(),
        |lc| lc + T.get_variable(),
      );

      self
        .poly
        .check_poly_zero_poly_one_with(cs.namespace(|| "R1CS poly(0)+poly(1) = T"), &T)?;

      // --- Lookup-side (C)-binding: poly_lookup(0)+poly_lookup(1) = t_lookup_running ---
      lookup.poly_lookup.check_poly_zero_poly_one_with(
        cs.namespace(|| "lookup poly(0)+poly(1) = t_lookup_running"),
        t_lookup_running,
      )?;

      // --- Absorb polynomials in transcript: R1CS first, then lookup ---
      self.poly.absorb_in_ro(&mut ro);
      lookup.absorb_poly_in_ro(&mut ro);

      // squeeze r_b (shared sumcheck challenge)
      let r_b_bits = ro.squeeze(cs.namespace(|| "r_b_bits"), NUM_CHALLENGE_BITS, false)?;
      let r_b = le_bits_to_num(cs.namespace(|| "r_b"), &r_b_bits)?;

      // --- R1CS T_out: T_out * eq(rho, r_b) = poly(r_b) ---
      let eq_rho_r_b_one = AllocatedNum::alloc(
        cs.namespace(|| "allocate R1CS eq_rho_r_b_one"),
        || {
          let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          let r_b_v = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok((E::Scalar::ONE - rho_v) * (E::Scalar::ONE - r_b_v))
        },
      )?;
      cs.enforce(
        || "R1CS eq_rho_r_b_one = (1-rho)(1-r_b)",
        |lc| lc + CS::one() - rho.get_variable(),
        |lc| lc + CS::one() - r_b.get_variable(),
        |lc| lc + eq_rho_r_b_one.get_variable(),
      );

      let eq_rho_r_b = AllocatedNum::alloc(cs.namespace(|| "allocate R1CS eq_rho_r_b"), || {
        let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let r_b_v = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok((E::Scalar::ONE - rho_v) * (E::Scalar::ONE - r_b_v) + rho_v * r_b_v)
      })?;
      cs.enforce(
        || "R1CS eq_rho_r_b = (1-rho)(1-r_b) + rho*r_b",
        |lc| lc + rho.get_variable(),
        |lc| lc + r_b.get_variable(),
        |lc| lc + eq_rho_r_b.get_variable() - eq_rho_r_b_one.get_variable(),
      );

      let r1cs_eval = self.poly.evaluate(cs.namespace(|| "R1CS eval(r_b)"), &r_b)?;
      let T_out = AllocatedNum::alloc(cs.namespace(|| "allocate R1CS T_out"), || {
        let eval = r1cs_eval
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let eq_inv = eq_rho_r_b
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?
          .invert()
          .unwrap();
        Ok(eval * eq_inv)
      })?;
      cs.enforce(
        || "enforce R1CS T_out * eq_rho_r_b = eval(r_b)",
        |lc| lc + T_out.get_variable(),
        |lc| lc + eq_rho_r_b.get_variable(),
        |lc| lc + r1cs_eval.get_variable(),
      );

      // --- Lookup T_lookup_out: t_lookup_out * eq(rho, r_b) = poly_lookup(r_b) ---
      // Reuse the same eq_rho_r_b (FS-rebinding requires same rho, same r_b).
      let lookup_eval = lookup
        .poly_lookup
        .evaluate(cs.namespace(|| "lookup eval(r_b)"), &r_b)?;
      let T_lookup_out =
        AllocatedNum::alloc(cs.namespace(|| "allocate T_lookup_out"), || {
          let eval = lookup_eval
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          let eq_inv = eq_rho_r_b
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?
            .invert()
            .unwrap();
          Ok(eval * eq_inv)
        })?;
      cs.enforce(
        || "enforce T_lookup_out * eq_rho_r_b = lookup eval(r_b)",
        |lc| lc + T_lookup_out.get_variable(),
        |lc| lc + eq_rho_r_b.get_variable(),
        |lc| lc + lookup_eval.get_variable(),
      );

      // --- Fold the R1CS-zero instance (lookup-side instance fold is the
      // augmented circuit's responsibility via the next-step Nova hash). ---
      let U_fold = U1.fold(
        cs.namespace(|| "fold R1CS"),
        U2,
        &r_b,
        &T_out,
        comm_W_fold,
        comm_E_fold,
      )?;

      Ok(LookupVerifyOutput {
        U_fold,
        T_lookup_out,
      })
    }

    /// In-circuit **multi-column** lookup-aware verifier (Stage I-pri).
    ///
    /// Mirrors the native [`crate::neutron::nifs::NIFS::verify_with_multi_column_lookup`]
    /// transcript:
    ///
    /// ```text
    ///   ro.absorb(pp_digest)
    ///   U2.absorb_in_ro
    ///   comm_L_pub.absorb_in_ro                    [address column]
    ///   for cv in comm_values_pub: cv.absorb_in_ro [Stage I-pri]
    ///   comm_ts_pub.absorb_in_ro
    ///   ro.squeeze() -> tau
    ///   comm_E.absorb_in_ro
    ///   ro.squeeze() -> rho
    ///   if !comm_values_pub.is_empty():
    ///     ro.squeeze() -> α                        [Stage I-pri]
    ///   ro.squeeze() -> r_logup
    ///   comm_inv_w / comm_inv_t / poly / poly_lookup absorptions
    ///   ro.squeeze() -> r_b
    /// ```
    ///
    /// When `comm_values_pub.is_empty()` the FS transcript is byte-
    /// identical to [`Self::verify_with_lookup`] (Stage H byte-equivalence
    /// pin).
    ///
    /// The combined-witness commitment is NEVER reconstructed in-circuit
    /// — soundness flows through α's binding to all column commitments
    /// at squeeze time and the (C)-binding on `poly_lookup` against the
    /// running target.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_with_multi_column_lookup<CS: ConstraintSystem<E::Scalar>>(
      &self,
      mut cs: CS,
      pp_digest: &AllocatedNum<E::Scalar>,
      U1: &AllocatedFoldedInstance<E>,
      U2: &AllocatedNonnativeR1CSInstance<E>,
      lookup: &AllocatedLookupNIFS<E>,
      comm_L_pub: &AllocatedNonnativePoint<E>,
      comm_values_pub: &[AllocatedNonnativePoint<E>],
      comm_ts_pub: &AllocatedNonnativePoint<E>,
      t_lookup_running: &AllocatedNum<E::Scalar>,
      comm_W_fold: &AllocatedNonnativePoint<E>,
      comm_E_fold: &AllocatedNonnativePoint<E>,
      ro_consts: RO2ConstantsCircuit<E>,
    ) -> Result<LookupVerifyOutput<E>, SynthesisError> {
      let mut ro = E::RO2Circuit::new(ro_consts);
      ro.absorb(pp_digest);

      U2.absorb_in_ro(cs.namespace(|| "absorb U2"), &mut ro)?;

      // --- Step (2): absorb address, value columns, ts ---
      comm_L_pub.absorb_in_ro(cs.namespace(|| "absorb comm_L"), &mut ro)?;
      for (i, cv) in comm_values_pub.iter().enumerate() {
        cv.absorb_in_ro(cs.namespace(|| format!("absorb comm_values[{}]", i)), &mut ro)?;
      }
      comm_ts_pub.absorb_in_ro(cs.namespace(|| "absorb comm_ts"), &mut ro)?;

      let _tau = ro.squeeze(cs.namespace(|| "tau"), NUM_CHALLENGE_BITS, false)?;
      self
        .comm_E
        .absorb_in_ro(cs.namespace(|| "absorb comm_E"), &mut ro)?;

      let rho_bits = ro.squeeze(cs.namespace(|| "rho_bits"), NUM_CHALLENGE_BITS, false)?;
      let rho = le_bits_to_num(cs.namespace(|| "rho"), &rho_bits)?;

      // Stage I-pri: squeeze α IFF value columns present.
      if !comm_values_pub.is_empty() {
        let _alpha_bits =
          ro.squeeze(cs.namespace(|| "alpha_bits"), NUM_CHALLENGE_BITS, false)?;
      }

      let _r_logup_bits =
        ro.squeeze(cs.namespace(|| "r_logup_bits"), NUM_CHALLENGE_BITS, false)?;

      lookup.absorb_inv_comms_in_ro(cs.namespace(|| "absorb inv comms"), &mut ro)?;

      // R1CS (C)-binding
      let T = AllocatedNum::alloc(cs.namespace(|| "allocate R1CS T"), || {
        let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let U1_T = U1.T.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(U1_T * (E::Scalar::ONE - rho_v))
      })?;
      cs.enforce(
        || "enforce R1CS T = (1-rho) * U1.T",
        |lc| lc + U1.T.get_variable(),
        |lc| lc + CS::one() - rho.get_variable(),
        |lc| lc + T.get_variable(),
      );

      self
        .poly
        .check_poly_zero_poly_one_with(cs.namespace(|| "R1CS poly(0)+poly(1) = T"), &T)?;

      lookup.poly_lookup.check_poly_zero_poly_one_with(
        cs.namespace(|| "lookup poly(0)+poly(1) = t_lookup_running"),
        t_lookup_running,
      )?;

      self.poly.absorb_in_ro(&mut ro);
      lookup.absorb_poly_in_ro(&mut ro);

      let r_b_bits = ro.squeeze(cs.namespace(|| "r_b_bits"), NUM_CHALLENGE_BITS, false)?;
      let r_b = le_bits_to_num(cs.namespace(|| "r_b"), &r_b_bits)?;

      // R1CS T_out
      let eq_rho_r_b_one = AllocatedNum::alloc(
        cs.namespace(|| "allocate R1CS eq_rho_r_b_one"),
        || {
          let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          let r_b_v = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok((E::Scalar::ONE - rho_v) * (E::Scalar::ONE - r_b_v))
        },
      )?;
      cs.enforce(
        || "R1CS eq_rho_r_b_one = (1-rho)(1-r_b)",
        |lc| lc + CS::one() - rho.get_variable(),
        |lc| lc + CS::one() - r_b.get_variable(),
        |lc| lc + eq_rho_r_b_one.get_variable(),
      );

      let eq_rho_r_b = AllocatedNum::alloc(cs.namespace(|| "allocate R1CS eq_rho_r_b"), || {
        let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let r_b_v = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok((E::Scalar::ONE - rho_v) * (E::Scalar::ONE - r_b_v) + rho_v * r_b_v)
      })?;
      cs.enforce(
        || "R1CS eq_rho_r_b = (1-rho)(1-r_b) + rho*r_b",
        |lc| lc + rho.get_variable(),
        |lc| lc + r_b.get_variable(),
        |lc| lc + eq_rho_r_b.get_variable() - eq_rho_r_b_one.get_variable(),
      );

      let r1cs_eval = self.poly.evaluate(cs.namespace(|| "R1CS eval(r_b)"), &r_b)?;
      let T_out = AllocatedNum::alloc(cs.namespace(|| "allocate R1CS T_out"), || {
        let eval = r1cs_eval
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let eq_inv = eq_rho_r_b
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?
          .invert()
          .unwrap();
        Ok(eval * eq_inv)
      })?;
      cs.enforce(
        || "enforce R1CS T_out * eq_rho_r_b = eval(r_b)",
        |lc| lc + T_out.get_variable(),
        |lc| lc + eq_rho_r_b.get_variable(),
        |lc| lc + r1cs_eval.get_variable(),
      );

      // Lookup T_lookup_out
      let lookup_eval = lookup
        .poly_lookup
        .evaluate(cs.namespace(|| "lookup eval(r_b)"), &r_b)?;
      let T_lookup_out =
        AllocatedNum::alloc(cs.namespace(|| "allocate T_lookup_out"), || {
          let eval = lookup_eval
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
          let eq_inv = eq_rho_r_b
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?
            .invert()
            .unwrap();
          Ok(eval * eq_inv)
        })?;
      cs.enforce(
        || "enforce T_lookup_out * eq_rho_r_b = lookup eval(r_b)",
        |lc| lc + T_lookup_out.get_variable(),
        |lc| lc + eq_rho_r_b.get_variable(),
        |lc| lc + lookup_eval.get_variable(),
      );

      let U_fold = U1.fold(
        cs.namespace(|| "fold R1CS"),
        U2,
        &r_b,
        &T_out,
        comm_W_fold,
        comm_E_fold,
      )?;

      Ok(LookupVerifyOutput {
        U_fold,
        T_lookup_out,
      })
    }

    /// In-circuit **multi-table** lookup-aware verifier (GH-#2 M.6).
    ///
    /// Multi-table extension of [`Self::verify_with_multi_column_lookup`]
    /// per design pin §5.1 — the in-circuit half of the (M.4) native
    /// `verify_with_multi_table_lookup` pair. Mirrors the native
    /// transcript schedule of pin §2.2 byte-for-byte at the scalar-
    /// sequence layer per pin §2.4 (the underlying RO2 Poseidon byte
    /// encoding differs across the BN256/Grumpkin cycle, so equivalence
    /// is asserted on the ordered (op, scalar) tuple sequence, not on
    /// raw bytes).
    ///
    /// ```text
    ///   ro.absorb(pp_digest)
    ///   U2.absorb_in_ro
    ///   for j in 0..k:                           // pin §2.2 Step 2
    ///     comm_L_pub[j].absorb_in_ro
    ///     for cv in comm_values_pub[j]: cv.absorb_in_ro
    ///     comm_ts_pub[j].absorb_in_ro
    ///   ro.squeeze() -> tau                      // Step 3
    ///   comm_E.absorb_in_ro                      // Step 3
    ///   ro.squeeze() -> rho                      // Step 4
    ///   for j in 0..k:                           // Step 5a (gated)
    ///     if !comm_values_pub[j].is_empty():
    ///       ro.squeeze() -> alpha_j
    ///   for j in 0..k:                           // Step 5b
    ///     ro.squeeze() -> r_logup_j
    ///   for j in 0..k: comm_inv_w[j].absorb_in_ro    // Step 6
    ///   for j in 0..k: comm_inv_t[j].absorb_in_ro    // Step 6
    ///   poly.check_poly_zero_poly_one_with(T)        // Step 7 (R1CS-side)
    ///   for j in 0..k:                               // Step 8
    ///     poly_lookup[j].check_poly_zero_poly_one_with(t_lookup_running[j])
    ///   poly.absorb_in_ro                            // Step 7
    ///   for j in 0..k: poly_lookup[j].absorb_in_ro   // Step 8
    ///   ro.squeeze() -> r_b                          // Step 9
    /// ```
    ///
    /// Constant-shape verifier per pin §1.5.4 (P1): always synthesises
    /// `k` tables' worth of FS absorptions, commitment-decode
    /// constraints, `r_logup_j` squeezes, and `poly_lookup_j`
    /// (C)-binding checks. There is no `present_mask` to branch on —
    /// absent tables are structurally indistinguishable from "queried
    /// with all-zero values" at the FS-transcript and (C)-binding
    /// layer.
    ///
    /// Per-table (C)-binding `poly_lookup_j(0) + poly_lookup_j(1) ==
    /// t_lookup_running_j` is enforced per-table per pin §1.3
    /// (load-bearing soundness check; closes the cross-table
    /// cancellation forgery vector).
    ///
    /// `t_lookup_running_per_table` is the lookup-side running target
    /// VECTOR projected from the running instance (cf. native
    /// `lookup_running_claims_from`). Its length MUST equal `k =
    /// lookups.k() = public_bundles.len()`. At outer base, all entries
    /// are zero.
    ///
    /// `comm_values_pub` lengths per-table MUST match the structurally
    /// pinned `multi_column_tables[j].columns.len()` — the caller is
    /// responsible for that allocation; this method does not
    /// branch on the size (it simply iterates the supplied slice).
    ///
    /// **GH-#2 M.7 / pin §3.2 + §3.3 wire-in.** Before
    /// `pp_digest.absorb(ro)` (which seeds the FS transcript), this
    /// method invokes
    /// [`crate::shape_registry::assert_pp_digest_matches_registry`]
    /// to enforce the per-position binding `pp_digest_in ==
    /// shape_registry[chunk_index_in_z]`. This closes the cross-
    /// position witness substitution attack (pin §3.4) by reducing
    /// any cross-position forgery to a `pp_digest` collision under
    /// the underlying `RO2` instance. Inputs:
    ///
    /// - `chunk_index_in_z` — the augmented circuit's running-instance
    ///   chunk-position index, carried in public IO `X` per ADR-0021.
    /// - `shape_registry` — the per-position list of
    ///   per-`Structure<E>` `pp_digest`s in chunk-position-canonical
    ///   order (NOT `table_id` order — pin §3.1). Construction lives
    ///   in `inumbra-spend-circuits` public-params per pin §6.4; this
    ///   method is the consumption site.
    /// - `index_n_bits` — the bit-width used for the range-check on
    ///   `chunk_index_in_z`. Must be ≥ `ceil(log2(shape_registry.len()))`.
    ///   At production arity (`shape_registry.len() ≤ 30`) this is `5`;
    ///   the M.6 k=2 regression test passes `4` (16-entry main-loop
    ///   subset).
    #[allow(clippy::too_many_arguments)]
    pub fn verify_with_multi_table_lookup<CS: ConstraintSystem<E::Scalar>>(
      &self,
      mut cs: CS,
      pp_digest: &AllocatedNum<E::Scalar>,
      chunk_index_in_z: &AllocatedNum<E::Scalar>,
      shape_registry: &[AllocatedNum<E::Scalar>],
      index_n_bits: usize,
      U1: &AllocatedFoldedInstance<E>,
      U2: &AllocatedNonnativeR1CSInstance<E>,
      lookups: &AllocatedLookupNIFSMultiTable<E>,
      public_bundles: &[AllocatedLookupPayloadPublicMultiTable<E>],
      t_lookup_running_per_table: &[AllocatedNum<E::Scalar>],
      comm_W_fold: &AllocatedNonnativePoint<E>,
      comm_E_fold: &AllocatedNonnativePoint<E>,
      ro_consts: RO2ConstantsCircuit<E>,
    ) -> Result<LookupVerifyOutputMultiTable<E>, SynthesisError> {
      // --- Sanity: per-table count must agree across all four inputs ---
      let k = lookups.k();
      if public_bundles.len() != k
        || t_lookup_running_per_table.len() != k
        || lookups.comm_inv_w.len() != k
        || lookups.comm_inv_t.len() != k
      {
        return Err(SynthesisError::Unsatisfiable(format!(
          "verify_with_multi_table_lookup: per-table count mismatch \
           (k={k}, public_bundles={}, t_lookup_running={}, comm_inv_w={}, comm_inv_t={})",
          public_bundles.len(),
          t_lookup_running_per_table.len(),
          lookups.comm_inv_w.len(),
          lookups.comm_inv_t.len(),
        )));
      }

      // --- M.7 / pin §3.2 + §3.3: per-position shape-registry assertion ---
      //
      // Sited BEFORE the `pp_digest.absorb(ro)` step: if `pp_digest`
      // is a wrong value (cross-position witness substitution), the
      // FS transcript would diverge from the prover's at byte 0,
      // making every downstream squeeze meaningless. Asserting the
      // shape-registry binding here fails fast and cleanly.
      crate::shape_registry::assert_pp_digest_matches_registry::<E, _>(
        cs.namespace(|| "M.7 shape-registry assertion"),
        pp_digest,
        chunk_index_in_z,
        shape_registry,
        index_n_bits,
      )?;

      let mut ro = E::RO2Circuit::new(ro_consts);
      ro.absorb(pp_digest);

      U2.absorb_in_ro(cs.namespace(|| "absorb U2"), &mut ro)?;

      // --- Step 2: per-table commitments BEFORE tau (pin §2.2) ---
      // Order: ALL tables' comm_L → ALL tables' value columns (intra-
      // table order preserved) → ALL tables' comm_ts. Outer loop over
      // j (table-id ascending; caller-pinned).
      for (j, b) in public_bundles.iter().enumerate() {
        b.comm_L.absorb_in_ro(
          cs.namespace(|| format!("absorb comm_L[{}]", j)),
          &mut ro,
        )?;
        for (i, cv) in b.comm_values.iter().enumerate() {
          cv.absorb_in_ro(
            cs.namespace(|| format!("absorb comm_values[{}][{}]", j, i)),
            &mut ro,
          )?;
        }
        b.comm_ts.absorb_in_ro(
          cs.namespace(|| format!("absorb comm_ts[{}]", j)),
          &mut ro,
        )?;
      }

      // --- Step 3: tau (single shared R1CS-side challenge) ---
      let _tau = ro.squeeze(cs.namespace(|| "tau"), NUM_CHALLENGE_BITS, false)?;

      // Absorb comm_E from the NIFS message.
      self
        .comm_E
        .absorb_in_ro(cs.namespace(|| "absorb comm_E"), &mut ro)?;

      // --- Step 4: rho (single shared R1CS / lookup batching challenge) ---
      let rho_bits = ro.squeeze(cs.namespace(|| "rho_bits"), NUM_CHALLENGE_BITS, false)?;
      let rho = le_bits_to_num(cs.namespace(|| "rho"), &rho_bits)?;

      // --- Step 5a: per-table alpha_j (gated on c_j > 0 per pin §2.2) ---
      // Per pin §2.2 step 5a, tables with empty value columns skip the
      // squeeze, preserving Stage I-pri's per-table c=0 byte-equivalence.
      // The verifier does not USE alpha_j directly (the combined-witness
      // identity is reconstructed implicitly via the FS-bound transcript);
      // what matters is that the squeeze fires per-bundle to advance the
      // RO state in lockstep with the prover.
      for (j, b) in public_bundles.iter().enumerate() {
        if !b.comm_values.is_empty() {
          let _alpha_j = ro.squeeze(
            cs.namespace(|| format!("alpha_bits[{}]", j)),
            NUM_CHALLENGE_BITS,
            false,
          )?;
        }
      }

      // --- Step 5b: per-table r_logup_j (one squeeze per table) ---
      // The verifier doesn't use r_logup_j directly (the LogUp identity
      // is proven by the prover-supplied `poly_lookup_j` and the
      // (C)-binding check below); the squeezes advance the RO state.
      for j in 0..k {
        let _r_logup_j = ro.squeeze(
          cs.namespace(|| format!("r_logup_bits[{}]", j)),
          NUM_CHALLENGE_BITS,
          false,
        )?;
      }

      // --- Step 6: per-table inverse-witness commitments ---
      // Order: ALL tables' comm_inv_w → ALL tables' comm_inv_t (NOT
      // interleaved per-table; matches native M.4 step 6 pair-grouping).
      for (j, c) in lookups.comm_inv_w.iter().enumerate() {
        c.absorb_in_ro(
          cs.namespace(|| format!("absorb comm_inv_w[{}]", j)),
          &mut ro,
        )?;
      }
      for (j, c) in lookups.comm_inv_t.iter().enumerate() {
        c.absorb_in_ro(
          cs.namespace(|| format!("absorb comm_inv_t[{}]", j)),
          &mut ro,
        )?;
      }

      // --- Step 7: R1CS-side (C)-binding poly(0)+poly(1) = T ---
      let T = AllocatedNum::alloc(cs.namespace(|| "allocate R1CS T"), || {
        let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let U1_T = U1.T.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(U1_T * (E::Scalar::ONE - rho_v))
      })?;
      cs.enforce(
        || "enforce R1CS T = (1-rho) * U1.T",
        |lc| lc + U1.T.get_variable(),
        |lc| lc + CS::one() - rho.get_variable(),
        |lc| lc + T.get_variable(),
      );

      self
        .poly
        .check_poly_zero_poly_one_with(cs.namespace(|| "R1CS poly(0)+poly(1) = T"), &T)?;

      // --- Step 8: per-table lookup-side (C)-binding (pin §1.3) ---
      // Per pin §1.3 the per-table (C)-binding
      // `poly_lookup_j(0) + poly_lookup_j(1) == t_lookup_running_j` is
      // the VECTOR running-claim invariant. Cross-table cancellation
      // forgery is closed STRUCTURALLY here: any single table's binding
      // failure rejects the whole fold step regardless of whether some
      // hypothetical SUM-aggregate would have cancelled.
      for (j, poly_lookup_j) in lookups.poly_lookup.iter().enumerate() {
        poly_lookup_j.check_poly_zero_poly_one_with(
          cs.namespace(|| format!("lookup poly(0)+poly(1) = t_lookup_running[{}]", j)),
          &t_lookup_running_per_table[j],
        )?;
      }

      // --- Absorb polynomials in transcript: R1CS first, then per-table lookup ---
      self.poly.absorb_in_ro(&mut ro);
      for poly_lookup_j in &lookups.poly_lookup {
        poly_lookup_j.absorb_in_ro(&mut ro);
      }

      // --- Step 9: r_b (single fold randomness shared across all instances) ---
      let r_b_bits = ro.squeeze(cs.namespace(|| "r_b_bits"), NUM_CHALLENGE_BITS, false)?;
      let r_b = le_bits_to_num(cs.namespace(|| "r_b"), &r_b_bits)?;

      // --- R1CS T_out: T_out * eq(rho, r_b) = poly(r_b) ---
      let eq_rho_r_b_one = AllocatedNum::alloc(
        cs.namespace(|| "allocate R1CS eq_rho_r_b_one"),
        || {
          let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          let r_b_v = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok((E::Scalar::ONE - rho_v) * (E::Scalar::ONE - r_b_v))
        },
      )?;
      cs.enforce(
        || "R1CS eq_rho_r_b_one = (1-rho)(1-r_b)",
        |lc| lc + CS::one() - rho.get_variable(),
        |lc| lc + CS::one() - r_b.get_variable(),
        |lc| lc + eq_rho_r_b_one.get_variable(),
      );

      let eq_rho_r_b = AllocatedNum::alloc(cs.namespace(|| "allocate R1CS eq_rho_r_b"), || {
        let rho_v = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let r_b_v = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok((E::Scalar::ONE - rho_v) * (E::Scalar::ONE - r_b_v) + rho_v * r_b_v)
      })?;
      cs.enforce(
        || "R1CS eq_rho_r_b = (1-rho)(1-r_b) + rho*r_b",
        |lc| lc + rho.get_variable(),
        |lc| lc + r_b.get_variable(),
        |lc| lc + eq_rho_r_b.get_variable() - eq_rho_r_b_one.get_variable(),
      );

      let r1cs_eval = self.poly.evaluate(cs.namespace(|| "R1CS eval(r_b)"), &r_b)?;
      let T_out = AllocatedNum::alloc(cs.namespace(|| "allocate R1CS T_out"), || {
        let eval = r1cs_eval
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        let eq_inv = eq_rho_r_b
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?
          .invert()
          .unwrap();
        Ok(eval * eq_inv)
      })?;
      cs.enforce(
        || "enforce R1CS T_out * eq_rho_r_b = eval(r_b)",
        |lc| lc + T_out.get_variable(),
        |lc| lc + eq_rho_r_b.get_variable(),
        |lc| lc + r1cs_eval.get_variable(),
      );

      // --- Per-table T_lookup_out_j (VECTOR per pin §1.3) ---
      // Reuse the same eq_rho_r_b across all per-table evaluations
      // (FS-rebinding requires same rho, same r_b — pin §2.2 step 9).
      let mut t_lookup_out_per_table: Vec<AllocatedNum<E::Scalar>> = Vec::with_capacity(k);
      for (j, poly_lookup_j) in lookups.poly_lookup.iter().enumerate() {
        let lookup_eval_j = poly_lookup_j.evaluate(
          cs.namespace(|| format!("lookup eval[{}](r_b)", j)),
          &r_b,
        )?;
        let t_lookup_out_j = AllocatedNum::alloc(
          cs.namespace(|| format!("allocate T_lookup_out[{}]", j)),
          || {
            let eval = lookup_eval_j
              .get_value()
              .ok_or(SynthesisError::AssignmentMissing)?;
            let eq_inv = eq_rho_r_b
              .get_value()
              .ok_or(SynthesisError::AssignmentMissing)?
              .invert()
              .unwrap();
            Ok(eval * eq_inv)
          },
        )?;
        cs.enforce(
          || format!("enforce T_lookup_out[{}] * eq_rho_r_b = lookup eval[{}](r_b)", j, j),
          |lc| lc + t_lookup_out_j.get_variable(),
          |lc| lc + eq_rho_r_b.get_variable(),
          |lc| lc + lookup_eval_j.get_variable(),
        );
        t_lookup_out_per_table.push(t_lookup_out_j);
      }

      // --- Fold the R1CS-zero instance ---
      // (lookup-side instance fold is the augmented circuit's
      // responsibility via the next-step Nova hash; the per-table
      // T_lookup_out vector flows through the augmented public IO.)
      let U_fold = U1.fold(
        cs.namespace(|| "fold R1CS"),
        U2,
        &r_b,
        &T_out,
        comm_W_fold,
        comm_E_fold,
      )?;

      Ok(LookupVerifyOutputMultiTable {
        U_fold,
        T_lookup_out_per_table: t_lookup_out_per_table,
      })
    }
  }

  /// Result of [`AllocatedNIFS::verify_with_multi_table_lookup`]: the
  /// folded R1CS-zero instance and the next-step lookup-zero running
  /// target VECTOR (one entry per registered table). The augmented
  /// circuit attaches `T_lookup_out_per_table` to its public-input hash
  /// per pin §1.3 (T_lookup is a Vec<E::Scalar> in `FoldedInstance`).
  pub struct LookupVerifyOutputMultiTable<E: Engine> {
    pub U_fold: AllocatedFoldedInstance<E>,
    pub T_lookup_out_per_table: Vec<AllocatedNum<E::Scalar>>,
  }
}

#[cfg(feature = "lookup-fold")]
#[allow(unused_imports)]
pub use lookup_verify::{LookupVerifyOutput, LookupVerifyOutputMultiTable};
