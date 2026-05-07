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
  use crate::neutron::circuit::lookup::AllocatedLookupNIFS;

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
  }
}

#[cfg(feature = "lookup-fold")]
#[allow(unused_imports)]
pub use lookup_verify::LookupVerifyOutput;
