//! This module implements `RelaxedR1CSSNARKTrait` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! This version of Spartan does not use preprocessing so the verifier keeps the entire
//! description of R1CS matrices. This is essentially optimal for the verifier when using
//! an IPA-based polynomial commitment scheme.

use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness, SparseMatrix},
  spartan::{
    compute_eval_table_sparse,
    math::Math,
    polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial, multilinear::SparsePolynomial},
    sumcheck::SumcheckProof,
    PolyEvalInstance, PolyEvalWitness,
  },
  traits::{
    evaluation::EvaluationEngineTrait,
    snark::{DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Engine, TranscriptEngineTrait,
  },
  CommitmentKey,
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  pk_ee: EE::ProverKey,
  vk_digest: E::Scalar, // digest of the verifier's key
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  vk_ee: EE::VerifierKey,
  S: R1CSShape<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> SimpleDigestible for VerifierKey<E, EE> {}

impl<E: Engine, EE: EvaluationEngineTrait<E>> VerifierKey<E, EE> {
  fn new(shape: R1CSShape<E>, vk_ee: EE::VerifierKey) -> Self {
    VerifierKey {
      vk_ee,
      S: shape,
      digest: OnceCell::new(),
    }
  }
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> DigestHelperTrait<E> for VerifierKey<E, EE> {
  /// Returns the digest of the verifier's key.
  fn digest(&self) -> E::Scalar {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::<E::Scalar, _>::new(self);
        dc.digest()
      })
      .cloned()
      .expect("Failure to retrieve digest!")
  }
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
///
/// The `split_eval_E` field is populated only by the `prove_with_split_error`
/// sibling (M.GH7.0.0; Corrigenda #7 + #8 + #9). When set, it carries the two
/// independent evaluations `(eval_E1, eval_E2)` of `E1_MLE(r_x_low)` and
/// `E2_MLE(r_x_high)` respectively per Corrigendum #9 (2-A); `eval_E` is then
/// the combined product `eval_E1 * eval_E2` (reconstructed at verify) and
/// remains the value absorbed into the outer-sumcheck transcript for FS
/// compatibility with the existing `prove`/`verify` body.
///
/// The standard `prove`/`verify` body leaves `split_eval_E = None`. The
/// existing serialised proof shape is preserved verbatim under `#[serde(default)]`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSSNARK<E: Engine, EE: EvaluationEngineTrait<E>> {
  sc_proof_outer: SumcheckProof<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar),
  eval_E: E::Scalar,
  sc_proof_inner: SumcheckProof<E>,
  eval_W: E::Scalar,
  sc_proof_batch: SumcheckProof<E>,
  evals_batch: Vec<E::Scalar>,
  eval_arg: EE::EvaluationArgument,
  /// Set ONLY by `prove_with_split_error` (M.GH7.0.0). Carries `(eval_E1, eval_E2)`
  /// for the factorised outer sumcheck identity per Corrigenda #7/#8/#9.
  #[serde(default = "Option::default")]
  split_eval_E: Option<(E::Scalar, E::Scalar)>,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> RelaxedR1CSSNARKTrait<E> for RelaxedR1CSSNARK<E, EE> {
  type ProverKey = ProverKey<E, EE>;
  type VerifierKey = VerifierKey<E, EE>;

  fn setup(
    ck: &CommitmentKey<E>,
    S: &R1CSShape<E>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    let (pk_ee, vk_ee) = EE::setup(ck)?;

    let S = S.pad();

    let vk: VerifierKey<E, EE> = VerifierKey::new(S, vk_ee);

    let pk = ProverKey {
      pk_ee,
      vk_digest: vk.digest(),
    };

    Ok((pk, vk))
  }

  /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: &R1CSShape<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
  ) -> Result<Self, NovaError> {
    // pad the R1CSShape
    let S = S.pad();
    // sanity check that R1CSShape has all required size characteristics
    assert!(S.is_regular_shape());

    let W = W.pad(&S); // pad the witness
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // append the digest of vk (which includes R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let mut z = [W.W.clone(), vec![U.u], U.X.clone()].concat();

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(S.num_cons.ilog2()).unwrap(),
      (usize::try_from(S.num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<Vec<_>, NovaError>>()?;

    let (mut poly_Az, mut poly_Bz, poly_Cz, mut poly_uCz_E) = {
      let (poly_Az, poly_Bz, poly_Cz) = S.multiply_vec(&z)?;
      let poly_uCz_E = (0..S.num_cons)
        .map(|i| U.u * poly_Cz[i] + W.E[i])
        .collect::<Vec<E::Scalar>>();
      (
        MultilinearPolynomial::new(poly_Az),
        MultilinearPolynomial::new(poly_Bz),
        MultilinearPolynomial::new(poly_Cz),
        MultilinearPolynomial::new(poly_uCz_E),
      )
    };

    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_three_inputs(
      &E::Scalar::ZERO, // claim is zero
      tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_uCz_E,
      &mut transcript,
    )?;

    // claims from the end of sum-check
    let (claim_Az, claim_Bz): (E::Scalar, E::Scalar) = (claims_outer[0], claims_outer[1]);
    let claim_Cz = poly_Cz.evaluate(&r_x);
    let eval_E = MultilinearPolynomial::new(W.E.clone()).evaluate(&r_x);
    transcript.absorb(
      b"claims_outer",
      &[claim_Az, claim_Bz, claim_Cz, eval_E].as_slice(),
    );

    // inner sum-check
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

    let poly_ABC = {
      // compute the initial evaluation table for R(\tau, x)
      let evals_rx = EqPolynomial::evals_from_points(&r_x.clone());

      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&S, &evals_rx);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
        .collect::<Vec<E::Scalar>>()
    };

    let poly_z = {
      z.resize(S.num_vars * 2, E::Scalar::ZERO);
      z
    };

    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad_prod(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      &mut transcript,
    )?;

    // Add additional claims about W and E polynomials to the list from CC
    // We will reduce a vector of claims of evaluations at different points into claims about them at the same point.
    // For example, eval_W =? W(r_y[1..]) and eval_E =? E(r_x) into
    // two claims: eval_W_prime =? W(rz) and eval_E_prime =? E(rz)
    // We can them combine the two into one: eval_W_prime + gamma * eval_E_prime =? (W + gamma*E)(rz),
    // where gamma is a public challenge
    // Since commitments to W and E are homomorphic, the verifier can compute a commitment
    // to the batched polynomial.
    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_y[1..]);

    let w_vec = vec![PolyEvalWitness { p: W.W }, PolyEvalWitness { p: W.E }];
    let u_vec = vec![
      PolyEvalInstance {
        c: U.comm_W,
        x: r_y[1..].to_vec(),
        e: eval_W,
      },
      PolyEvalInstance {
        c: U.comm_E,
        x: r_x,
        e: eval_E,
      },
    ];

    let (batched_u, batched_w, _chal, sc_proof_batch, claims_batch_left) =
      super::batch_eval_reduce(u_vec, w_vec, &mut transcript)?;

    let eval_arg = EE::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &batched_u.c,
      &batched_w.p,
      &batched_u.x,
      &batched_u.e,
    )?;

    Ok(RelaxedR1CSSNARK {
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      eval_E,
      sc_proof_inner,
      eval_W,
      sc_proof_batch,
      evals_batch: claims_batch_left,
      eval_arg,
      split_eval_E: None,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<E>) -> Result<(), NovaError> {
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // append the digest of R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", U);

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(vk.S.num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, NovaError>>()?;

    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(E::Scalar::ZERO, num_rounds_x, 3, &mut transcript)?;

    // verify claim_outer_final
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let taus_bound_rx = tau.evaluate(&r_x);
    let claim_outer_final_expected =
      taus_bound_rx * (claim_Az * claim_Bz - U.u * claim_Cz - self.eval_E);
    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(
      b"claims_outer",
      &[
        self.claims_outer.0,
        self.claims_outer.1,
        self.claims_outer.2,
        self.eval_E,
      ]
      .as_slice(),
    );

    // inner sum-check
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint =
      self.claims_outer.0 + r * self.claims_outer.1 + r * r * self.claims_outer.2;

    let (claim_inner_final, r_y) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 2, &mut transcript)?;

    // verify claim_inner_final
    let eval_Z = {
      let eval_X = {
        // public IO is (u, X)
        let X = vec![U.u]
          .into_iter()
          .chain(U.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(vk.S.num_vars.log_2(), X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    // compute evaluations of R1CS matrices
    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
        |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
          M.indptr
            .par_windows(2)
            .enumerate()
            .map(|(row_idx, ptrs)| {
              M.get_row_unchecked(ptrs.try_into().unwrap())
                .map(|(val, col_idx)| T_x[row_idx] * T_y[*col_idx] * val)
                .sum::<E::Scalar>()
            })
            .sum()
        };

      let (T_x, T_y) = rayon::join(
        || EqPolynomial::evals_from_points(r_x),
        || EqPolynomial::evals_from_points(r_y),
      );

      (0..M_vec.len())
        .into_par_iter()
        .map(|i| evaluate_with_table(M_vec[i], &T_x, &T_y))
        .collect()
    };

    let evals = multi_evaluate(&[&vk.S.A, &vk.S.B, &vk.S.C], &r_x, &r_y);

    let claim_inner_final_expected = (evals[0] + r * evals[1] + r * r * evals[2]) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // add claims about W and E polynomials
    let u_vec: Vec<PolyEvalInstance<E>> = vec![
      PolyEvalInstance {
        c: U.comm_W,
        x: r_y[1..].to_vec(),
        e: self.eval_W,
      },
      PolyEvalInstance {
        c: U.comm_E,
        x: r_x,
        e: self.eval_E,
      },
    ];

    let (batched_u, _chal) = super::batch_eval_verify(
      u_vec,
      &mut transcript,
      &self.sc_proof_batch,
      &self.evals_batch,
    )?;

    // verify
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &batched_u.c,
      &batched_u.x,
      &batched_u.e,
      &self.eval_arg,
    )?;

    Ok(())
  }
}

// ---------------------------------------------------------------------------
// M.GH7.0.0 — Spartan-side `prove_with_split_error` / `verify_with_split_error`
// sibling for the GH-#7 Stage K post-IVC compressed SNARK bridge.
//
// Anchors:
//
// - Corrigendum #7 (load-bearing): factorised outer sumcheck identity
//   `Az·Bz - u·Cz - E1(r_x_low) * E2(r_x_high)` per NeutronNova §5
//   zerofold + Spartan §5 (rank-1 outer-product MLE factorisation).
// - Corrigendum #8 (load-bearing): option (iv-B) — Spartan-close-only
//   locus; zero IVC-layer changes; Pedersen-additive binding enforced
//   at the CompressedSNARK envelope (M.GH7.0.2), NOT inside this sibling.
// - Corrigendum #9 (2-A) ratified: `r_x_high = r_x[..ell2]` (top
//   `ell2` challenges → `E2`'s outer index); `r_x_low = r_x[ell2..]`
//   (bottom `ell1` challenges → `E1`'s inner index). Flat layout
//   `full_E[i*left+j] = E2[i] * E1[j]` matches
//   `vendor/nova/src/neutron/lookup_sumcheck.rs:308-322`.
// - Corrigendum #9 (3-A) ratified: prover-side one-shot `full_E`
//   materialisation (`num_cons * 32` bytes, parallel to existing
//   `Az`/`Bz`/`Cz` allocations); sumcheck engine runs UNCHANGED against
//   flat `poly_uCz_E = U.u*Cz + full_E`; factorisation enforced ONLY
//   at the verifier via `eval_E_combined = eval_E1 * eval_E2`.
//
// This impl block is independent of `RelaxedR1CSSNARKTrait` because the
// method signatures consume the split `(E1, E2, comm_E1, comm_E2)`
// inputs that the trait's `prove`/`verify` do not.
// ---------------------------------------------------------------------------
impl<E: Engine, EE: EvaluationEngineTrait<E>> RelaxedR1CSSNARK<E, EE> {
  /// Produces a Spartan proof for a `RelaxedR1CS` instance whose `W.E`
  /// is the rank-1 tensor product `full_E[i*left+j] = E2[i] * E1[j]`
  /// (per Corrigendum #9 (2-A) layout). Consumes `(E1, E2)` directly
  /// rather than `W.E`; opens commitments to `E1` and `E2` independently
  /// at `r_x_low` and `r_x_high` instead of opening `comm_E` at `r_x`.
  ///
  /// Inputs:
  /// - `(E1, E2)`: rank-1 factors of length `left` and `right` such
  ///   that `W.E[i*left+j] = E2[i] * E1[j]`.
  /// - `comm_E1`, `comm_E2`: commitments to `E1` and `E2` on the standard
  ///   `ck` prefix (`ck.ck[..E1.len()]` and `ck.ck[..E2.len()]`).
  ///   IMPORTANT: like the standard `prove`, this sibling consumes
  ///   DERANDOMIZED commitments (zero blindings) — see
  ///   `vendor/nova/src/spartan/direct.rs:159-175` for the canonical
  ///   derandomization flow that callers must perform upstream. The
  ///   Pedersen-additive binding `comm_E1 + comm_E2 == U.comm_E` per
  ///   Corrigendum #8 (iv-B) is enforced at the M.GH7.0.2 CompressedSNARK
  ///   envelope where blindings are restored, NOT here.
  /// - `_r_E1`, `_r_E2`: reserved for the M.GH7.0.2 envelope (rederandomize +
  ///   Pedersen-additive close); ignored at the Spartan inner layer.
  ///
  /// Memory cost: one-shot heap allocation of `full_E: Vec<E::Scalar>`
  /// of length `num_cons`, parallel to existing `Az`/`Bz`/`Cz` allocations
  /// the prover already performs (Corrigendum #9 (3-A)).
  ///
  /// The FS transcript byte stream is byte-equivalent to the standard
  /// `prove` body's transcript up to and including the outer-sumcheck
  /// `claims_outer` absorb (which uses the combined `eval_E1 * eval_E2`
  /// as the absorbed `eval_E`), preserving the existing soundness
  /// reduction's transcript discipline.
  #[allow(clippy::too_many_arguments)]
  pub fn prove_with_split_error(
    ck: &CommitmentKey<E>,
    pk: &ProverKey<E, EE>,
    S: &R1CSShape<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
    comm_E1: crate::Commitment<E>,
    comm_E2: crate::Commitment<E>,
    E1: &[E::Scalar],
    E2: &[E::Scalar],
    _r_E1: E::Scalar,
    _r_E2: E::Scalar,
  ) -> Result<Self, NovaError> {
    // pad the R1CSShape
    let S = S.pad();
    // sanity check that R1CSShape has all required size characteristics
    assert!(S.is_regular_shape());

    let W = W.pad(&S); // pad the witness
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // append the digest of vk (which includes R1CS matrices) and the
    // RelaxedR1CSInstance to the transcript. Identical to `prove` so the
    // outer-sumcheck FS challenges are byte-equivalent up to the
    // `claims_outer` absorb (which uses the combined eval_E1 * eval_E2,
    // preserving the value the verifier expects).
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    // Match Structure<E>::new partition convention (vendor/nova/src/neutron/relation.rs:518-536):
    //   ell = log2(num_cons), ell1 = ell.div_ceil(2), ell2 = ell/2, left = 2^ell1, right = 2^ell2.
    let ell = S.num_cons.log_2();
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    let left = 1usize << ell1;
    let right = 1usize << ell2;
    assert_eq!(left * right, S.num_cons);
    assert_eq!(E1.len(), left, "E1 length must equal left = 2^ell1");
    assert_eq!(E2.len(), right, "E2 length must equal right = 2^ell2");

    // (3-A): prover-side one-shot `full_E` materialisation. full_E[i*left+j] = E2[i]*E1[j].
    // Layout matches `vendor/nova/src/neutron/lookup_sumcheck.rs:308-322` byte-for-byte:
    // outer index `i ∈ [0, right)` advances the TOP `ell2` bits of `k`; inner index
    // `j ∈ [0, left)` advances the BOTTOM `ell1` bits.
    let mut full_E: Vec<E::Scalar> = Vec::with_capacity(S.num_cons);
    for i in 0..right {
      for j in 0..left {
        full_E.push(E2[i] * E1[j]);
      }
    }
    debug_assert_eq!(full_E.len(), S.num_cons);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let mut z = [W.W.clone(), vec![U.u], U.X.clone()].concat();

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(S.num_cons.ilog2()).unwrap(),
      (usize::try_from(S.num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<Vec<_>, NovaError>>()?;

    let (mut poly_Az, mut poly_Bz, poly_Cz, mut poly_uCz_E) = {
      let (poly_Az, poly_Bz, poly_Cz) = S.multiply_vec(&z)?;
      // (3-A): flat poly_uCz_E[i] = U.u * Cz[i] + full_E[i]. The sumcheck engine
      // runs UNCHANGED against this flat polynomial; the rank-1 factorisation
      // is enforced only at the verifier.
      let poly_uCz_E = (0..S.num_cons)
        .map(|i| U.u * poly_Cz[i] + full_E[i])
        .collect::<Vec<E::Scalar>>();
      (
        MultilinearPolynomial::new(poly_Az),
        MultilinearPolynomial::new(poly_Bz),
        MultilinearPolynomial::new(poly_Cz),
        MultilinearPolynomial::new(poly_uCz_E),
      )
    };

    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_three_inputs(
      &E::Scalar::ZERO, // claim is zero
      tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_uCz_E,
      &mut transcript,
    )?;

    // claims from the end of sum-check
    let (claim_Az, claim_Bz): (E::Scalar, E::Scalar) = (claims_outer[0], claims_outer[1]);
    let claim_Cz = poly_Cz.evaluate(&r_x);

    // Corrigendum #9 (2-A): r_x_high = first ell2 challenges (TOP) → bind E2;
    //                        r_x_low  = last  ell1 challenges (BOTTOM) → bind E1.
    // `bind_poly_var_top` at `vendor/nova/src/spartan/sumcheck.rs:489-497` binds
    // the MSB first → the FIRST challenge of `r_x` binds the TOP variable of the
    // flat `full_E` polynomial. With layout `k = i*left + j`, the TOP `ell2` bits
    // of `k` index `i ∈ [0, right)` (E2's outer index), and the BOTTOM `ell1` bits
    // index `j ∈ [0, left)` (E1's inner index).
    let r_x_high = &r_x[..ell2];
    let r_x_low = &r_x[ell2..];
    debug_assert_eq!(r_x_high.len(), ell2);
    debug_assert_eq!(r_x_low.len(), ell1);

    let eval_E1 = MultilinearPolynomial::new(E1.to_vec()).evaluate(r_x_low);
    let eval_E2 = MultilinearPolynomial::new(E2.to_vec()).evaluate(r_x_high);
    // (3-A): the combined factorised evaluation. This MUST equal
    // MultilinearPolynomial::new(full_E).evaluate(&r_x) — the soundness anchor
    // empirical close (M.GH7.0.0 unit test). Absorbed into the transcript in the
    // same position as the original `eval_E` to preserve FS byte equivalence.
    let eval_E = eval_E1 * eval_E2;

    transcript.absorb(
      b"claims_outer",
      &[claim_Az, claim_Bz, claim_Cz, eval_E].as_slice(),
    );

    // inner sum-check (unchanged from `prove`).
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

    let poly_ABC = {
      let evals_rx = EqPolynomial::evals_from_points(&r_x.clone());
      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&S, &evals_rx);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
        .collect::<Vec<E::Scalar>>()
    };

    let poly_z = {
      z.resize(S.num_vars * 2, E::Scalar::ZERO);
      z
    };

    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad_prod(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      &mut transcript,
    )?;

    // Batch step — heterogeneous claims at different evaluation-point dimensions:
    //   - (comm_W, r_y[1..], eval_W)        : witness polynomial, length 1 << (num_rounds_y - 1)
    //   - (comm_E1, r_x_low,  eval_E1)      : E1, length left  = 2^ell1
    //   - (comm_E2, r_x_high, eval_E2)      : E2, length right = 2^ell2
    //
    // `super::batch_eval_reduce` at `vendor/nova/src/spartan/mod.rs:384-438` accepts
    // heterogeneous polynomial sizes via `num_rounds = u_vec.iter().map(|u| u.x.len())`
    // and asserts `w.p.len() == 1 << num_vars` per claim.
    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_y[1..]);

    let w_vec = vec![
      PolyEvalWitness { p: W.W },
      PolyEvalWitness { p: E1.to_vec() },
      PolyEvalWitness { p: E2.to_vec() },
    ];
    let u_vec = vec![
      PolyEvalInstance {
        c: U.comm_W,
        x: r_y[1..].to_vec(),
        e: eval_W,
      },
      PolyEvalInstance {
        c: comm_E1,
        x: r_x_low.to_vec(),
        e: eval_E1,
      },
      PolyEvalInstance {
        c: comm_E2,
        x: r_x_high.to_vec(),
        e: eval_E2,
      },
    ];

    let (batched_u, batched_w, _chal, sc_proof_batch, claims_batch_left) =
      super::batch_eval_reduce(u_vec, w_vec, &mut transcript)?;

    let eval_arg = EE::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &batched_u.c,
      &batched_w.p,
      &batched_u.x,
      &batched_u.e,
    )?;

    Ok(RelaxedR1CSSNARK {
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      eval_E,
      sc_proof_inner,
      eval_W,
      sc_proof_batch,
      evals_batch: claims_batch_left,
      eval_arg,
      split_eval_E: Some((eval_E1, eval_E2)),
    })
  }

  /// Verifies a proof produced by `prove_with_split_error`.
  ///
  /// Inputs:
  /// - `vk`: the standard verifier key.
  /// - `U`: the standard `RelaxedR1CSInstance` (with `U.comm_E` reflecting the
  ///   unsplit error commitment; the binding `comm_E1 + comm_E2 == U.comm_E`
  ///   per Corrigendum #8 (iv-B) is checked at the M.GH7.0.2 envelope, NOT here).
  /// - `comm_E1`, `comm_E2`: the split-error commitments PCS-opened independently
  ///   at `r_x_low` and `r_x_high` respectively.
  ///
  /// Returns `Ok(())` if the factorised outer-sumcheck identity holds and
  /// both PCS openings (W at r_y[1..], E1 at r_x_low, E2 at r_x_high) verify
  /// against the joint batch reduction.
  pub fn verify_with_split_error(
    &self,
    vk: &VerifierKey<E, EE>,
    U: &RelaxedR1CSInstance<E>,
    comm_E1: crate::Commitment<E>,
    comm_E2: crate::Commitment<E>,
  ) -> Result<(), NovaError> {
    let (eval_E1, eval_E2) = self
      .split_eval_E
      .ok_or(NovaError::InvalidSumcheckProof)?;

    // The combined eval_E used at the claims_outer absorb is the rank-1 product.
    // It MUST equal `self.eval_E` (the field stored on the struct) for the FS
    // transcript to round-trip.
    if eval_E1 * eval_E2 != self.eval_E {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // append the digest of R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", U);

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(vk.S.num_vars.ilog2()).unwrap() + 1),
    );

    // Partition convention per Corrigendum #9 (2-A); mirrors prover side.
    // Only `ell2` is needed on the verifier side — `r_x_high = r_x[..ell2]`,
    // `r_x_low = r_x[ell2..]` (length implicitly `ell1 = ell - ell2`).
    let ell = vk.S.num_cons.log_2();
    let ell2 = ell / 2;

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, NovaError>>()?;

    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(E::Scalar::ZERO, num_rounds_x, 3, &mut transcript)?;

    // verify claim_outer_final — factorised identity per Corrigendum #7 algebra +
    // Corrigendum #9 (3-A) verifier-side reconstruction.
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let taus_bound_rx = tau.evaluate(&r_x);
    let eval_E_combined = eval_E1 * eval_E2;
    let claim_outer_final_expected =
      taus_bound_rx * (claim_Az * claim_Bz - U.u * claim_Cz - eval_E_combined);
    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(
      b"claims_outer",
      &[
        self.claims_outer.0,
        self.claims_outer.1,
        self.claims_outer.2,
        eval_E_combined,
      ]
      .as_slice(),
    );

    // inner sum-check (unchanged from `verify`).
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint =
      self.claims_outer.0 + r * self.claims_outer.1 + r * r * self.claims_outer.2;

    let (claim_inner_final, r_y) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 2, &mut transcript)?;

    let eval_Z = {
      let eval_X = {
        let X = vec![U.u]
          .into_iter()
          .chain(U.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(vk.S.num_vars.log_2(), X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
        |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
          M.indptr
            .par_windows(2)
            .enumerate()
            .map(|(row_idx, ptrs)| {
              M.get_row_unchecked(ptrs.try_into().unwrap())
                .map(|(val, col_idx)| T_x[row_idx] * T_y[*col_idx] * val)
                .sum::<E::Scalar>()
            })
            .sum()
        };

      let (T_x, T_y) = rayon::join(
        || EqPolynomial::evals_from_points(r_x),
        || EqPolynomial::evals_from_points(r_y),
      );

      (0..M_vec.len())
        .into_par_iter()
        .map(|i| evaluate_with_table(M_vec[i], &T_x, &T_y))
        .collect()
    };

    let evals = multi_evaluate(&[&vk.S.A, &vk.S.B, &vk.S.C], &r_x, &r_y);

    let claim_inner_final_expected = (evals[0] + r * evals[1] + r * r * evals[2]) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // Add claims about W, E1, E2 polynomials — heterogeneous batch step,
    // mirroring the prover side. r_x_low = r_x[ell2..]; r_x_high = r_x[..ell2].
    let r_x_high = r_x[..ell2].to_vec();
    let r_x_low = r_x[ell2..].to_vec();

    let u_vec: Vec<PolyEvalInstance<E>> = vec![
      PolyEvalInstance {
        c: U.comm_W,
        x: r_y[1..].to_vec(),
        e: self.eval_W,
      },
      PolyEvalInstance {
        c: comm_E1,
        x: r_x_low,
        e: eval_E1,
      },
      PolyEvalInstance {
        c: comm_E2,
        x: r_x_high,
        e: eval_E2,
      },
    ];

    let (batched_u, _chal) = super::batch_eval_verify(
      u_vec,
      &mut transcript,
      &self.sc_proof_batch,
      &self.evals_batch,
    )?;

    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &batched_u.c,
      &batched_u.x,
      &batched_u.e,
      &self.eval_arg,
    )?;

    Ok(())
  }
}

// The M.GH7.0.0b sibling below references `crate::neutron::compressed_snark::
// BridgedNeutronInstance`, which lives behind `#[cfg(feature = "experimental")]`
// at `vendor/nova/src/lib.rs:17-18` (the `neutron` module is gated on the
// `experimental` feature, which `lookup-fold` transitively enables per
// `Cargo.toml`'s `lookup-fold = ["experimental"]` declaration). The Stage K
// compressed SNARK is `lookup-fold`-gated workspace-wide; gating this impl
// block matches that discipline.
#[cfg(feature = "experimental")]
// ---------------------------------------------------------------------------
// M.GH7.0.0b — Spartan-side `prove_with_T_claim_split_error` /
// `verify_with_T_claim_split_error` sibling for the GH-#7 Stage K post-IVC
// compressed SNARK bridge (Corrigendum #10).
//
// Proves the *neutron-form* satisfying relation directly:
//
//   sum_x full_E(x) · (Az(x) · Bz(x) − Cz(x)) = T
//
// with:
//   (a) outer-sumcheck claim = T (NOT zero) — verified-rejecting T' ≠ T at
//       the `claim_outer_final` reconstruction
//   (b) tensor-form `(E1, E2)` weighting (vendor-grounded by
//       `lookup_sumcheck.rs:46-55`) instead of the EqSumCheckInstance-from-taus
//       used by M.GH7.0.0's `prove_with_split_error`
//   (c) residue `Az · Bz − Cz` with NO `u`-factor on Cz (Δ2 absorbed
//       structurally per Corrigendum #10 §1.2(a); the slack lives in T,
//       not in u)
//
// FS-transcript discipline (LOAD-BEARING — Primitive 5 binding):
//
//   ts.absorb(b"vk", &vk_digest)
//   ts.absorb(b"U",  U_bridged)          // BridgedNeutronInstance impl
//   ts.absorb(b"T_claim", &[T])          // strictly before any squeeze
//   // NO `tau` squeeze (tensor-form weights replace eq-trick)
//   sc_proof_outer = SumcheckProof::prove_neutron_outer(T, E1, E2, ...)
//   ts.absorb(b"claims_outer", &[claim_Az, claim_Bz, claim_Cz, eval_E1, eval_E2])
//   // inner sumcheck + batch-eval-reduce UNCHANGED from M.GH7.0.0
//
// Variable-order partition convention pinned by Corrigendum #9 (2-A):
//   r_x_high = r_x[..ell2]   (top ell2 challenges → bind E2's outer index)
//   r_x_low  = r_x[ell2..]   (bottom ell1 challenges → bind E1's inner index)
//
// M.GH7.0.0's existing `prove_with_split_error` / `verify_with_split_error`
// sibling is RETAINED VERBATIM — Corrigendum #10 adds a sibling, does NOT
// modify the existing one. The M.GH7.0.2 envelope routes to the new sibling
// instead of the old one under β'.
// ---------------------------------------------------------------------------
impl<E: Engine, EE: EvaluationEngineTrait<E>> RelaxedR1CSSNARK<E, EE> {
  /// Produces a Spartan proof of the neutron-form satisfying relation
  ///   `sum_x full_E(x) · (Az(x) · Bz(x) − Cz(x)) = T`
  /// for a [`BridgedNeutronInstance`] consumed by the M.GH7.0.2 envelope.
  ///
  /// See module-level docblock above for the FS-transcript discipline and
  /// the soundness reduction to the five paper-grounded primitives.
  ///
  /// Inputs:
  /// - `U_bridged`: envelope-published bridge shape carrying
  ///   `(comm_W, comm_E1, comm_E2, u, X, T)`.
  /// - `W`: derandomized `RelaxedR1CSWitness`. Like
  ///   `prove_with_split_error`, this method consumes DERANDOMIZED inputs —
  ///   `r_E1`, `r_E2` are reserved for the M.GH7.0.2 envelope's
  ///   Pedersen-additive close (`comm_E1 + comm_E2 == U.comm_E`) and are
  ///   IGNORED at the Spartan inner layer.
  /// - `(E1, E2)`: rank-1 factors of length `left` and `right` such that
  ///   `full_E[i*left + j] = E2[i] · E1[j]` per Corrigendum #9 (2-A).
  /// - `(comm_E1, comm_E2)`: PCS commitments to `(E1, E2)` on the standard
  ///   `ck.ck[..left]` and `ck.ck[left..left+right]` slices (M.GH7.0.1 helper
  ///   discipline).
  /// - `T`: running neutron-form sumcheck claim, identity-bridged from
  ///   `FoldedInstance::T` (`relation.rs:254`).
  #[allow(clippy::too_many_arguments)]
  #[allow(non_snake_case)]
  pub fn prove_with_T_claim_split_error(
    ck: &CommitmentKey<E>,
    pk: &ProverKey<E, EE>,
    S: &R1CSShape<E>,
    U_bridged: &crate::neutron::compressed_snark::BridgedNeutronInstance<E>,
    W: &RelaxedR1CSWitness<E>,
    comm_E1: crate::Commitment<E>,
    comm_E2: crate::Commitment<E>,
    E1: &[E::Scalar],
    E2: &[E::Scalar],
    T: E::Scalar,
    _r_E1: E::Scalar,
    _r_E2: E::Scalar,
  ) -> Result<Self, NovaError> {
    // Pad the R1CSShape (mirrors `prove_with_split_error`).
    let S = S.pad();
    assert!(S.is_regular_shape());

    let W = W.pad(&S);
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // FS-transcript discipline (Corrigendum #10 §1.2(a) Primitive 5 binding):
    //   absorb vk digest → absorb U_bridged → absorb T_claim BEFORE any squeeze.
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U_bridged);
    // T_claim absorb-before-squeeze: a malicious prover cannot adaptively
    // choose T post-r_x because T is committed-to-the-transcript at this
    // line. Mirrors the `transcript.absorb(b"U", U)` discipline at
    // `snark.rs:495` (the M.GH7.0.0 sibling).
    transcript.absorb(b"T_claim", &T);

    // Match Structure<E>::new partition convention (Corrigendum #9 (2-A)):
    //   ell = log2(num_cons), ell1 = ell.div_ceil(2), ell2 = ell/2,
    //   left = 2^ell1, right = 2^ell2.
    let ell = S.num_cons.log_2();
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    let left = 1usize << ell1;
    let right = 1usize << ell2;
    assert_eq!(left * right, S.num_cons);
    assert_eq!(E1.len(), left, "E1 length must equal left = 2^ell1");
    assert_eq!(E2.len(), right, "E2 length must equal right = 2^ell2");

    // Compute the full satisfying assignment z = [W.W, U.u, U.X].concat().
    // Mirrors `prove_with_split_error`'s `z` construction at `snark.rs:521`.
    let mut z = [
      W.W.clone(),
      vec![U_bridged.u],
      U_bridged.X.clone(),
    ]
    .concat();

    let num_rounds_y = usize::try_from(S.num_vars.ilog2()).unwrap() + 1;

    // Compute Az, Bz, Cz (the row-reduced MLEs). Mirrors `prove_with_split_error`
    // line 534 — same `z` shape, same matrix product.
    let (poly_Az_vec, poly_Bz_vec, poly_Cz_vec) = S.multiply_vec(&z)?;

    let mut poly_Az = MultilinearPolynomial::new(poly_Az_vec);
    let mut poly_Bz = MultilinearPolynomial::new(poly_Bz_vec);
    let mut poly_Cz_for_outer = MultilinearPolynomial::new(poly_Cz_vec.clone());
    // Hold a separate copy of Cz for the inner sumcheck's `claim_Cz` recompute
    // at `r_x` — the outer sumcheck consumes (and binds in place) one copy.
    let poly_Cz_for_inner = MultilinearPolynomial::new(poly_Cz_vec);

    // Outer sumcheck: tensor-form (E1, E2) weighting, residue Az·Bz − Cz,
    // claim = T (non-zero generically). NO `tau` squeeze — the tensor-form
    // weighting replaces the eq-trick.
    let (sc_proof_outer, r_x, claims_outer_tuple) = SumcheckProof::prove_neutron_outer(
      T,
      E1,
      E2,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz_for_outer,
      &mut transcript,
    )?;
    let (claim_Az, claim_Bz, _claim_Cz_from_engine, eval_E1, eval_E2) = claims_outer_tuple;

    // Recompute claim_Cz from the unbound `poly_Cz_for_inner` at the same
    // r_x — the outer engine returned `poly_Cz_for_outer[0]` after binding,
    // which is structurally identical (this is just a defensive parallel-form
    // to the M.GH7.0.0 sibling at `snark.rs:560`).
    let claim_Cz = poly_Cz_for_inner.evaluate(&r_x);
    debug_assert_eq!(_claim_Cz_from_engine, claim_Cz);

    // Absorb the outer-sumcheck final claims under `b"claims_outer"`. Mirrors
    // M.GH7.0.0 absorb at `snark.rs:582-585`: order is
    //   (claim_Az, claim_Bz, claim_Cz, eval_E_combined)
    // where eval_E_combined = eval_E1 · eval_E2 is reconstructed by the
    // verifier (NOT absorbed as a pair) to preserve byte-equivalence with
    // M.GH7.0.0's transcript at the post-outer-sumcheck point.
    let eval_E_combined = eval_E1 * eval_E2;
    transcript.absorb(
      b"claims_outer",
      &[claim_Az, claim_Bz, claim_Cz, eval_E_combined].as_slice(),
    );

    // Inner sumcheck — UNCHANGED from M.GH7.0.0's `prove_with_split_error`
    // (`snark.rs:587-614`). The inner sumcheck does NOT depend on `u`; `u`
    // enters only via `z = [W.W, u, X]`. Falsifier F verified-absent:
    //   claim_inner_joint = claim_Az + r·claim_Bz + r²·claim_Cz
    // (no u-scaling on Cz here either — Δ2 absorbed structurally).
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

    let poly_ABC = {
      let evals_rx = EqPolynomial::evals_from_points(&r_x.clone());
      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&S, &evals_rx);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
        .collect::<Vec<E::Scalar>>()
    };

    let poly_z = {
      z.resize(S.num_vars * 2, E::Scalar::ZERO);
      z
    };

    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad_prod(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      &mut transcript,
    )?;

    // Batch step — heterogeneous claims:
    //   - (comm_W,  r_y[1..], eval_W)       : witness, length 1 << (num_rounds_y - 1)
    //   - (comm_E1, r_x_low,  eval_E1)      : E1, length left  = 2^ell1
    //   - (comm_E2, r_x_high, eval_E2)      : E2, length right = 2^ell2
    // Per Corrigendum #9 (2-A): r_x_high = r_x[..ell2], r_x_low = r_x[ell2..].
    let r_x_high = &r_x[..ell2];
    let r_x_low = &r_x[ell2..];

    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_y[1..]);

    let w_vec = vec![
      PolyEvalWitness { p: W.W },
      PolyEvalWitness { p: E1.to_vec() },
      PolyEvalWitness { p: E2.to_vec() },
    ];
    let u_vec = vec![
      PolyEvalInstance {
        c: U_bridged.comm_W,
        x: r_y[1..].to_vec(),
        e: eval_W,
      },
      PolyEvalInstance {
        c: comm_E1,
        x: r_x_low.to_vec(),
        e: eval_E1,
      },
      PolyEvalInstance {
        c: comm_E2,
        x: r_x_high.to_vec(),
        e: eval_E2,
      },
    ];

    let (batched_u, batched_w, _chal, sc_proof_batch, claims_batch_left) =
      super::batch_eval_reduce(u_vec, w_vec, &mut transcript)?;

    let eval_arg = EE::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &batched_u.c,
      &batched_w.p,
      &batched_u.x,
      &batched_u.e,
    )?;

    Ok(RelaxedR1CSSNARK {
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      eval_E: eval_E_combined,
      sc_proof_inner,
      eval_W,
      sc_proof_batch,
      evals_batch: claims_batch_left,
      eval_arg,
      split_eval_E: Some((eval_E1, eval_E2)),
    })
  }

  /// Verifies a proof produced by [`prove_with_T_claim_split_error`].
  ///
  /// FS-transcript discipline mirrors the prover (Corrigendum #10 §1.2(a)):
  /// absorb `vk` → absorb `U_bridged` → absorb `T_claim` before any squeeze.
  ///
  /// Verification steps (Corrigendum #10 §1.2(a) verifier-side):
  /// (i)   Reconstruct `claim_outer_final` from the outer-sumcheck transcript
  ///       starting from claim = `U_bridged.T` (NOT zero).
  /// (ii)  Reject if `claim_outer_final ≠ eval_E1 · eval_E2 · (claim_Az ·
  ///       claim_Bz − claim_Cz)` per the β' outer-residue identity (NO
  ///       `u`-factor on Cz).
  /// (iii) Inner sumcheck + batch-eval-reduce + PCS opening UNCHANGED from
  ///       M.GH7.0.0's `verify_with_split_error`.
  ///
  /// The Pedersen-additive binding `comm_E1 + comm_E2 == r_U.comm_E` per
  /// Corrigendum #8 (iv-B) is enforced at the M.GH7.0.2 CompressedSNARK
  /// envelope, NOT here.
  #[allow(non_snake_case)]
  pub fn verify_with_T_claim_split_error(
    &self,
    vk: &VerifierKey<E, EE>,
    U_bridged: &crate::neutron::compressed_snark::BridgedNeutronInstance<E>,
    comm_E1: crate::Commitment<E>,
    comm_E2: crate::Commitment<E>,
  ) -> Result<(), NovaError> {
    let (eval_E1, eval_E2) = self
      .split_eval_E
      .ok_or(NovaError::InvalidSumcheckProof)?;

    // Tensor-form factorisation byte-equivalence at the verifier (Primitive 2).
    if eval_E1 * eval_E2 != self.eval_E {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // FS-transcript: mirror the prover discipline.
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", U_bridged);
    transcript.absorb(b"T_claim", &U_bridged.T);

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S.num_cons.ilog2()).unwrap(),
      usize::try_from(vk.S.num_vars.ilog2()).unwrap() + 1,
    );

    // Partition convention per Corrigendum #9 (2-A).
    let ell = vk.S.num_cons.log_2();
    let ell2 = ell / 2;

    // Outer sumcheck verify: claim = T (NOT zero — this is the load-bearing
    // call-site change relative to M.GH7.0.0's `verify_with_split_error`,
    // which uses claim = ZERO). Degree bound = 3 (matches the prover engine).
    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(U_bridged.T, num_rounds_x, 3, &mut transcript)?;

    // Reconstruct claim_outer_final per the β' outer-residue identity
    // (Corrigendum #10 §1.2(a) verifier check (iii)):
    //   claim_outer_final =? eval_E1 · eval_E2 · (claim_Az · claim_Bz − claim_Cz)
    // NO u-factor on Cz (Δ2 absorbed structurally — Falsifier F verified-absent).
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let eval_E_combined = eval_E1 * eval_E2;
    let claim_outer_final_expected =
      eval_E_combined * (claim_Az * claim_Bz - claim_Cz);
    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(
      b"claims_outer",
      &[claim_Az, claim_Bz, claim_Cz, eval_E_combined].as_slice(),
    );

    // Inner sumcheck — UNCHANGED from M.GH7.0.0's `verify_with_split_error`
    // (`snark.rs:756-808`).
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

    let (claim_inner_final, r_y) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 2, &mut transcript)?;

    let eval_Z = {
      let eval_X = {
        let X = vec![U_bridged.u]
          .into_iter()
          .chain(U_bridged.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(vk.S.num_vars.log_2(), X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
        |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
          M.indptr
            .par_windows(2)
            .enumerate()
            .map(|(row_idx, ptrs)| {
              M.get_row_unchecked(ptrs.try_into().unwrap())
                .map(|(val, col_idx)| T_x[row_idx] * T_y[*col_idx] * val)
                .sum::<E::Scalar>()
            })
            .sum()
        };

      let (T_x, T_y) = rayon::join(
        || EqPolynomial::evals_from_points(r_x),
        || EqPolynomial::evals_from_points(r_y),
      );

      (0..M_vec.len())
        .into_par_iter()
        .map(|i| evaluate_with_table(M_vec[i], &T_x, &T_y))
        .collect()
    };

    let evals = multi_evaluate(&[&vk.S.A, &vk.S.B, &vk.S.C], &r_x, &r_y);

    let claim_inner_final_expected = (evals[0] + r * evals[1] + r * r * evals[2]) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // Batch step — heterogeneous claims (mirrors M.GH7.0.0 verifier).
    let r_x_high = r_x[..ell2].to_vec();
    let r_x_low = r_x[ell2..].to_vec();

    let u_vec: Vec<PolyEvalInstance<E>> = vec![
      PolyEvalInstance {
        c: U_bridged.comm_W,
        x: r_y[1..].to_vec(),
        e: self.eval_W,
      },
      PolyEvalInstance {
        c: comm_E1,
        x: r_x_low,
        e: eval_E1,
      },
      PolyEvalInstance {
        c: comm_E2,
        x: r_x_high,
        e: eval_E2,
      },
    ];

    let (batched_u, _chal) = super::batch_eval_verify(
      u_vec,
      &mut transcript,
      &self.sc_proof_batch,
      &self.evals_batch,
    )?;

    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &batched_u.c,
      &batched_u.x,
      &batched_u.e,
      &self.eval_arg,
    )?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  //! M.GH7.0.0 tests for the Spartan-side `prove_with_split_error` sibling.
  //!
  //! Anchors:
  //!
  //! - Corrigendum #7 (option (iv) hybrid disposition; factorised outer
  //!   sumcheck algebra `Az·Bz - u·Cz - E1(r_x_low) * E2(r_x_high)`).
  //! - Corrigendum #8 (option (iv-B); Spartan-close-only locus; zero
  //!   IVC-layer changes; Pedersen-additive binding handled at envelope).
  //! - Corrigendum #9 (2-A) variable-order partition: `r_x_high = r_x[..ell2]`
  //!   (top `ell2` challenges → `E2`); `r_x_low = r_x[ell2..]` (bottom
  //!   `ell1` challenges → `E1`). Flat layout `full_E[i*left+j] = E2[i]*E1[j]`
  //!   matches `vendor/nova/src/neutron/lookup_sumcheck.rs:308-322` byte-for-byte.
  //! - Corrigendum #9 (3-A) prover-side `full_E` materialisation; sumcheck
  //!   engine runs unchanged against flat `poly_uCz_E = U.u*Cz + full_E`;
  //!   factorisation enforced only at verifier via `eval_E1 * eval_E2`.

  use super::*;
  use crate::{
    provider::{Bn256EngineKZG, PallasEngine},
    r1cs::SparseMatrix,
    spartan::math::Math,
    traits::commitment::CommitmentEngineTrait,
  };
  use rand_chacha::ChaCha20Rng;
  use rand_core::SeedableRng;

  /// Construct a tiny satisfying `RelaxedR1CS` instance where `W.E` is a rank-1
  /// tensor product `full_E[i*left+j] = E2[i]*E1[j]`.
  ///
  /// Strategy: pick arbitrary `(E1, E2)` of lengths `(left, right)`; set
  /// `W = full_E`, `u = 1`, `X = [0]`. Matrices: `A[i, i] = 1` (so `Az = W`),
  /// `B[i, num_vars] = 1` (so `Bz = [u; num_cons]`), `C = 0`. Then
  /// `Az·Bz - u·Cz = W * u = full_E`, matching `W.E`.
  fn build_satisfying_instance<E: Engine>(
    ck: &CommitmentKey<E>,
    rng: &mut ChaCha20Rng,
  ) -> (
    R1CSShape<E>,
    RelaxedR1CSInstance<E>,
    RelaxedR1CSWitness<E>,
    Vec<E::Scalar>, // E1
    Vec<E::Scalar>, // E2
    usize,          // left
    usize,          // right
  ) {
    // Match Structure<E>::new convention: ell = log2(num_cons), ell1 = ell.div_ceil(2),
    // ell2 = ell/2, left = 2^ell1, right = 2^ell2. Here num_cons = 4 → ell = 2 →
    // ell1 = ell2 = 1 → left = right = 2.
    let num_cons = 4usize;
    let num_vars = 4usize;
    let num_io = 1usize;

    let ell = num_cons.log_2();
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    let left = 1usize << ell1;
    let right = 1usize << ell2;
    assert_eq!(left * right, num_cons);

    // Arbitrary E1, E2.
    let E1: Vec<E::Scalar> = (0..left).map(|_| E::Scalar::random(&mut *rng)).collect();
    let E2: Vec<E::Scalar> = (0..right).map(|_| E::Scalar::random(&mut *rng)).collect();

    // full_E[i*left+j] = E2[i] * E1[j] (per Corrigendum #9 (2-A) layout).
    let mut full_E: Vec<E::Scalar> = Vec::with_capacity(num_cons);
    for i in 0..right {
      for j in 0..left {
        full_E.push(E2[i] * E1[j]);
      }
    }
    assert_eq!(full_E.len(), num_cons);

    // Matrices: A[i, i] = 1; B[i, num_vars] = 1; C = 0.
    let one = E::Scalar::ONE;
    let rows = num_cons;
    let cols = num_vars + num_io + 1;
    let A_entries: Vec<(usize, usize, E::Scalar)> = (0..num_cons).map(|i| (i, i, one)).collect();
    let B_entries: Vec<(usize, usize, E::Scalar)> =
      (0..num_cons).map(|i| (i, num_vars, one)).collect();
    let C_entries: Vec<(usize, usize, E::Scalar)> = vec![];

    let S = R1CSShape::<E>::new(
      num_cons,
      num_vars,
      num_io,
      SparseMatrix::new(&A_entries, rows, cols),
      SparseMatrix::new(&B_entries, rows, cols),
      SparseMatrix::new(&C_entries, rows, cols),
    )
    .unwrap();

    // Witness: W = full_E, r_W = random, E = full_E, r_E = r_E1 + r_E2.
    let W_vec = full_E.clone();
    let r_W = E::Scalar::random(&mut *rng);
    let r_E1 = E::Scalar::random(&mut *rng);
    let r_E2 = E::Scalar::random(&mut *rng);
    let r_E = r_E1 + r_E2;

    // u = 1, X = [0] (any value works since C = 0).
    let u = E::Scalar::ONE;
    let X = vec![E::Scalar::ZERO; num_io];

    // Commit
    let comm_W = <E::CE as CommitmentEngineTrait<E>>::commit(ck, &W_vec, &r_W);
    let comm_E = <E::CE as CommitmentEngineTrait<E>>::commit(ck, &full_E, &r_E);

    let U = RelaxedR1CSInstance {
      comm_W,
      comm_E,
      u,
      X: X.clone(),
    };
    let W = RelaxedR1CSWitness {
      W: W_vec,
      r_W,
      E: full_E,
      r_E,
    };

    // Sanity-check satisfiability of the constructed instance.
    S.is_sat_relaxed(ck, &U, &W).expect("constructed instance must satisfy Relaxed R1CS");

    (S, U, W, E1, E2, left, right)
  }

  fn m_gh7_0_0_prove_with_split_error_factorised_outer_sumcheck_byte_equal_with<E, EE>()
  where
    E: Engine,
    EE: EvaluationEngineTrait<E>,
  {
    // ChaCha20Rng deterministic seed per US-05 (reviewer-reproducibility).
    let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEE_0700_0000u64);

    // Set up CK for the constructed shape.
    let ck = <E::CE as CommitmentEngineTrait<E>>::setup(b"m_gh7_0_0_test_ck", 1 << 10).unwrap();

    let (S, U, W, E1, E2, _left, _right) = build_satisfying_instance::<E>(&ck, &mut rng);

    let (pk, vk) = <RelaxedR1CSSNARK<E, EE> as RelaxedR1CSSNARKTrait<E>>::setup(&ck, &S).unwrap();

    // Spartan-side `prove` operates on DERANDOMIZED (zero-blinding) commitments —
    // see `vendor/nova/src/spartan/direct.rs:159-175`, where the trait flow calls
    // `W.derandomize()` / `U.derandomize(&dk, ...)` BEFORE invoking `S::prove`.
    // The Pedersen-additive binding `comm_E1 + comm_E2 == U.comm_E` per
    // Corrigendum #8 (iv-B) lives at the OUTER CompressedSNARK envelope (M.GH7.0.2)
    // where the blindings are restored, NOT inside this Spartan-close-only sibling.
    let dk = <E::CE as CommitmentEngineTrait<E>>::derand_key(&ck);
    let (W_derand, blind_W, blind_E) = W.derandomize();
    let U_derand = U.derandomize(&dk, &blind_W, &blind_E);

    // Commit E1 / E2 with ZERO blindings on the standard `ck` prefix. Each
    // sub-commitment lives in the same KZG basis; `comm_E1 + comm_E2` recovers
    // the derandomized `U_derand.comm_E` (= MSM(full_E, ck.ck[..num_cons], ZERO)).
    let r_E1 = E::Scalar::ZERO;
    let r_E2 = E::Scalar::ZERO;
    let comm_E1 = <E::CE as CommitmentEngineTrait<E>>::commit(&ck, &E1, &r_E1);
    let comm_E2 = <E::CE as CommitmentEngineTrait<E>>::commit(&ck, &E2, &r_E2);

    // Invoke the new sibling on derandomized inputs.
    let snark = RelaxedR1CSSNARK::<E, EE>::prove_with_split_error(
      &ck, &pk, &S, &U_derand, &W_derand, comm_E1, comm_E2, &E1, &E2, r_E1, r_E2,
    )
    .expect("prove_with_split_error must succeed for a satisfying rank-1 W.E");

    // Verify with the matching sibling against the derandomized instance.
    snark
      .verify_with_split_error(&vk, &U_derand, comm_E1, comm_E2)
      .expect("verify_with_split_error must accept a valid proof");
  }

  #[test]
  fn m_gh7_0_0_prove_with_split_error_factorised_outer_sumcheck_byte_equal() {
    // HyperKZG path (Bn256).
    type E1Engine = Bn256EngineKZG;
    type EE1 = crate::provider::hyperkzg::EvaluationEngine<E1Engine>;
    m_gh7_0_0_prove_with_split_error_factorised_outer_sumcheck_byte_equal_with::<E1Engine, EE1>();

    // IPA-PC path (Pallas).
    type E2Engine = PallasEngine;
    type EE2 = crate::provider::ipa_pc::EvaluationEngine<E2Engine>;
    m_gh7_0_0_prove_with_split_error_factorised_outer_sumcheck_byte_equal_with::<E2Engine, EE2>();
  }

  /// Unit test (soundness anchor empirical close): 1000-iter ChaCha20Rng-seeded
  /// test that `MultilinearPolynomial::new(full_E).evaluate(&r_x)` equals
  /// `E1_MLE(r_x[ell2..]) * E2_MLE(r_x[..ell2])` per Corrigendum #9 (2-A)
  /// variable-order convention, where `full_E[i*left+j] = E2[i] * E1[j]`.
  fn factorisation_byte_equivalence_1000_iter_with<E: Engine>() {
    // Deterministic per US-05.
    let mut rng = ChaCha20Rng::seed_from_u64(0xFAC1_0700_BEEFu64);

    // Vary (ell1, ell2) over the table layout used at vendor HEAD:
    //   relation.rs:518-536 → ell1 = ell.div_ceil(2), ell2 = ell/2 with ell1 >= ell2.
    // Cover ell ∈ {2, 3, 4} to exercise both ell1 == ell2 and ell1 > ell2.
    let cases: &[(usize, usize)] = &[
      (1, 1), // ell = 2, left = right = 2
      (2, 1), // ell = 3, left = 4, right = 2 (asymmetric)
      (2, 2), // ell = 4, left = right = 4
    ];

    let iters_per_case = 1000 / cases.len() + 1;

    for &(ell1, ell2) in cases {
      let left = 1usize << ell1;
      let right = 1usize << ell2;

      for _ in 0..iters_per_case {
        let E1: Vec<E::Scalar> = (0..left).map(|_| E::Scalar::random(&mut rng)).collect();
        let E2: Vec<E::Scalar> = (0..right).map(|_| E::Scalar::random(&mut rng)).collect();

        // Flat layout per Corrigendum #9 (2-A): full_E[i*left + j] = E2[i] * E1[j].
        let mut full_E: Vec<E::Scalar> = Vec::with_capacity(left * right);
        for i in 0..right {
          for j in 0..left {
            full_E.push(E2[i] * E1[j]);
          }
        }

        // Random evaluation point r_x of length ell1 + ell2.
        let r_x: Vec<E::Scalar> = (0..ell1 + ell2)
          .map(|_| E::Scalar::random(&mut rng))
          .collect();

        // Per Corrigendum #9 (2-A):
        //   r_x_high = r_x[..ell2]  (TOP / first ell2 challenges → bind E2's outer
        //                            index i, since `bind_poly_var_top` binds MSB first
        //                            and i indexes the TOP ell2 bits of k = i*left + j)
        //   r_x_low  = r_x[ell2..]  (BOTTOM / last ell1 challenges → bind E1's inner
        //                            index j, the BOTTOM ell1 bits of k)
        let r_x_high = &r_x[..ell2];
        let r_x_low = &r_x[ell2..];
        assert_eq!(r_x_high.len(), ell2);
        assert_eq!(r_x_low.len(), ell1);

        let full_eval =
          MultilinearPolynomial::new(full_E).evaluate(&r_x);
        let e1_eval =
          MultilinearPolynomial::new(E1).evaluate(r_x_low);
        let e2_eval =
          MultilinearPolynomial::new(E2).evaluate(r_x_high);
        let combined = e1_eval * e2_eval;

        assert_eq!(
          full_eval, combined,
          "Corrigendum #9 (2-A) factorisation byte-equivalence violated: \
           full_E_MLE(r_x) != E1_MLE(r_x[ell2..]) * E2_MLE(r_x[..ell2]) \
           at (ell1={}, ell2={})",
          ell1, ell2
        );
      }
    }
  }

  #[test]
  fn m_gh7_0_0_factorisation_byte_equivalence_1000_iter() {
    factorisation_byte_equivalence_1000_iter_with::<Bn256EngineKZG>();
    factorisation_byte_equivalence_1000_iter_with::<PallasEngine>();
  }

  /// Variable-order partition convention pinned per Corrigendum #9 (2-A).
  /// Explicit assertion that `r_x_high = r_x[..ell2]` and `r_x_low = r_x[ell2..]`
  /// produces the byte-correct factorisation for the `lookup_sumcheck.rs:308-322`
  /// flat layout `k = i*left + j`.
  ///
  /// The OPPOSITE convention (2-B), `r_x_low = r_x[..ell1]` / `r_x_high = r_x[ell1..]`,
  /// is rejected by this test when ell1 != ell2 because it produces an incorrect
  /// factorisation.
  #[test]
  fn m_gh7_0_0_variable_order_partition_convention() {
    type E = PallasEngine;
    // Deterministic per US-05.
    let mut rng = ChaCha20Rng::seed_from_u64(0x2A_0700_DEADu64);

    // Pick an asymmetric layout (ell1 > ell2) where (2-A) and (2-B) diverge
    // observably.
    let (ell1, ell2) = (2usize, 1usize);
    let left = 1usize << ell1;
    let right = 1usize << ell2;

    let E1: Vec<<E as Engine>::Scalar> =
      (0..left).map(|_| <E as Engine>::Scalar::random(&mut rng)).collect();
    let E2: Vec<<E as Engine>::Scalar> =
      (0..right).map(|_| <E as Engine>::Scalar::random(&mut rng)).collect();

    let mut full_E: Vec<<E as Engine>::Scalar> = Vec::with_capacity(left * right);
    for i in 0..right {
      for j in 0..left {
        full_E.push(E2[i] * E1[j]);
      }
    }

    let r_x: Vec<<E as Engine>::Scalar> = (0..ell1 + ell2)
      .map(|_| <E as Engine>::Scalar::random(&mut rng))
      .collect();

    let full_eval = MultilinearPolynomial::new(full_E).evaluate(&r_x);

    // (2-A) — the ratified convention.
    let r_x_high_2a = &r_x[..ell2];
    let r_x_low_2a = &r_x[ell2..];
    let combined_2a = MultilinearPolynomial::new(E1.clone()).evaluate(r_x_low_2a)
      * MultilinearPolynomial::new(E2.clone()).evaluate(r_x_high_2a);
    assert_eq!(
      full_eval, combined_2a,
      "(2-A) convention must match full_E_MLE evaluation"
    );

    // (2-B) — the rejected alternative; must diverge for ell1 != ell2.
    let r_x_low_2b = &r_x[..ell1];
    let r_x_high_2b = &r_x[ell1..];
    let combined_2b = MultilinearPolynomial::new(E1).evaluate(r_x_low_2b)
      * MultilinearPolynomial::new(E2).evaluate(r_x_high_2b);
    assert_ne!(
      full_eval, combined_2b,
      "(2-B) convention must diverge when ell1 != ell2 — this is the falsifier \
       that pins Corrigendum #9 (2-A) as the correct partition"
    );
  }

  // ===========================================================================
  // M.GH7.0.0b tests (Corrigendum #10) — Spartan T-claim sibling
  // `prove_with_T_claim_split_error` + companion sumcheck engine method
  // `SumcheckProof::prove_neutron_outer`.
  //
  // Gated on `experimental` feature because the sibling under test references
  // `crate::neutron::compressed_snark::BridgedNeutronInstance`, which lives
  // behind `#[cfg(feature = "experimental")]` at `lib.rs:17-18`. The
  // `lookup-fold` feature transitively enables `experimental` per the
  // Cargo.toml declaration `lookup-fold = ["experimental"]`.
  //
  // The β' sibling proves the *neutron-form* satisfying relation
  //   sum_x full_E(x) · (Az(x)·Bz(x) - Cz(x)) = T
  // with:
  //   (a) outer-sumcheck claim = T (NOT zero) — verified by claim_outer_final
  //       reconstruction at the verifier
  //   (b) tensor-form `(E1, E2)` weighting (vendor-grounded by
  //       `lookup_sumcheck.rs:46-55`) INSTEAD OF an EqSumCheckInstance built
  //       from fresh `tau` squeezes — NO `tau` is squeezed at the β' transcript
  //   (c) residue `Az·Bz - Cz` with NO `u`-factor on Cz (Δ2 absorbed
  //       structurally; the slack lives in T, not in u)
  //
  // The β' transcript discipline absorbs `T` strictly BEFORE the outer-sumcheck
  // is squeezed (Primitive 5 binding, Corrigendum #10 §1.2(a) FS-order pin).
  // ===========================================================================

  #[cfg(feature = "experimental")]
  mod m_gh7_0_0b {
    use super::*;
    use crate::neutron::compressed_snark::BridgedNeutronInstance;

  /// Construct a satisfying *neutron-form* witness instance.
  ///
  /// Shape: `num_cons = left*right`, `num_vars = num_cons`, `num_io = 1`,
  /// `u = 1`, `X = [0]`. Matrices: `A[i,i] = 1` (so `Az = W`),
  /// `B[i, num_vars] = 1` (so `Bz = [u; num_cons]`), `C = 0` (so `Cz = [0]`).
  /// `(E1, E2)` are drawn uniformly at random.
  ///
  /// `W` is drawn uniformly at random — UNRELATED to `(E1, E2)`. The
  /// neutron-form *claim* `T` is computed as
  ///   T = sum_x full_E(x) · (Az(x)·Bz(x) - Cz(x))
  ///     = sum_x full_E(x) · W(x)      (since Bz=[1;.], Cz=[0;.])
  /// so the constructed witness satisfies the β' relation by direct
  /// construction.
  ///
  /// Note: this witness does NOT satisfy the classical relaxed-R1CS pointwise
  /// equation `Az·Bz - u·Cz - E = 0`. The β' relation and the classical
  /// relation coincide ONLY at the IVC base case (u=1, T=0); they diverge
  /// generically post-fold, which is the structural reason Corrigendum #10
  /// authors the new sibling.
  #[allow(non_snake_case)]
  fn build_T_form_satisfying_instance<E: Engine>(
    ck: &CommitmentKey<E>,
    rng: &mut ChaCha20Rng,
    num_cons: usize,
  ) -> (
    R1CSShape<E>,
    RelaxedR1CSInstance<E>,
    RelaxedR1CSWitness<E>,
    Vec<E::Scalar>, // E1
    Vec<E::Scalar>, // E2
    E::Scalar,      // T (the running neutron-form claim)
  ) {
    let num_vars = num_cons;
    let num_io = 1usize;

    let ell = num_cons.log_2();
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    let left = 1usize << ell1;
    let right = 1usize << ell2;
    assert_eq!(left * right, num_cons);

    // Arbitrary E1, E2.
    let E1: Vec<E::Scalar> = (0..left).map(|_| E::Scalar::random(&mut *rng)).collect();
    let E2: Vec<E::Scalar> = (0..right).map(|_| E::Scalar::random(&mut *rng)).collect();

    // Flat full_E per Corrigendum #9 (2-A): full_E[i*left+j] = E2[i] * E1[j].
    let mut full_E: Vec<E::Scalar> = Vec::with_capacity(num_cons);
    for i in 0..right {
      for j in 0..left {
        full_E.push(E2[i] * E1[j]);
      }
    }

    // Matrices: A[i,i] = 1; B[i, num_vars] = 1; C = 0. Same as M.GH7.0.0.
    let one = E::Scalar::ONE;
    let rows = num_cons;
    let cols = num_vars + num_io + 1;
    let A_entries: Vec<(usize, usize, E::Scalar)> = (0..num_cons).map(|i| (i, i, one)).collect();
    let B_entries: Vec<(usize, usize, E::Scalar)> =
      (0..num_cons).map(|i| (i, num_vars, one)).collect();
    let C_entries: Vec<(usize, usize, E::Scalar)> = vec![];

    let S = R1CSShape::<E>::new(
      num_cons,
      num_vars,
      num_io,
      SparseMatrix::new(&A_entries, rows, cols),
      SparseMatrix::new(&B_entries, rows, cols),
      SparseMatrix::new(&C_entries, rows, cols),
    )
    .unwrap();

    // Witness vector W — drawn UNRELATED to (E1, E2). This is what makes the
    // T-claim non-trivial. We need `Az = W`, so `W` is arbitrary.
    let W_vec: Vec<E::Scalar> = (0..num_cons).map(|_| E::Scalar::random(&mut *rng)).collect();
    let r_W = E::Scalar::random(&mut *rng);

    // u = 1, X = [0]. (Both arbitrary at this layer — C = 0 means u and X
    // don't enter the residue; we keep them simple.)
    let u = E::Scalar::ONE;
    let X = vec![E::Scalar::ZERO; num_io];

    // Compute T = sum_x full_E(x) * (Az(x)*Bz(x) - Cz(x))
    //         = sum_x full_E(x) * (W(x) * u - 0)
    //         = sum_x full_E(x) * W(x)            (u=1)
    let T: E::Scalar = full_E
      .iter()
      .zip(W_vec.iter())
      .map(|(e, w)| *e * *w)
      .sum();

    // The witness `E` field carries `full_E` (legacy compatibility with
    // pad/derandomize machinery). The β' sibling consumes (E1, E2) and T
    // directly; W.E is NOT read by the prover. We materialize it for
    // structural completeness (pad and derandomize operate on E).
    let r_E1 = E::Scalar::random(&mut *rng);
    let r_E2 = E::Scalar::random(&mut *rng);
    let r_E = r_E1 + r_E2;

    // Commit to W and to full_E for the instance (the β' verifier checks
    // comm_E1 + comm_E2 == U.comm_E in the envelope at M.GH7.0.2; this
    // sibling-level test exercises the inner prove/verify with zero-blinding
    // derandomization, mirroring the M.GH7.0.0 test pattern).
    let comm_W = <E::CE as CommitmentEngineTrait<E>>::commit(ck, &W_vec, &r_W);
    let comm_E = <E::CE as CommitmentEngineTrait<E>>::commit(ck, &full_E, &r_E);

    let U = RelaxedR1CSInstance {
      comm_W,
      comm_E,
      u,
      X: X.clone(),
    };
    let W = RelaxedR1CSWitness {
      W: W_vec,
      r_W,
      E: full_E,
      r_E,
    };

    (S, U, W, E1, E2, T)
  }

  /// Helper: build a `BridgedNeutronInstance` from a derandomized
  /// `RelaxedR1CSInstance` + (comm_E1, comm_E2, T). The struct fields are
  /// `(comm_W, comm_E1, comm_E2, u, X, T)` per Corrigendum #10.
  #[allow(non_snake_case)]
  fn bridged_from<E: Engine>(
    U: &RelaxedR1CSInstance<E>,
    comm_E1: crate::Commitment<E>,
    comm_E2: crate::Commitment<E>,
    T: E::Scalar,
  ) -> BridgedNeutronInstance<E> {
    BridgedNeutronInstance {
      comm_W: U.comm_W,
      comm_E1,
      comm_E2,
      u: U.u,
      X: U.X.clone(),
      T,
    }
  }

  /// **Acceptance test (M.GH7.0.0b).** Round-trip byte-equal at fixed shape
  /// `left = right = 4, num_cons = 16, ell1 = ell2 = 2` with non-zero `T`.
  /// Verifies the β' sibling proves and verifies a satisfying neutron-form
  /// witness, exercising the full FS-transcript discipline including
  /// `ts.absorb(b"T_claim", &[T])` strictly before any outer-sumcheck squeeze.
  #[allow(non_snake_case)]
  fn round_trip_with<E, EE>()
  where
    E: Engine,
    EE: EvaluationEngineTrait<E>,
  {
    let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEE_0700_0000B_u64);
    let ck = <E::CE as CommitmentEngineTrait<E>>::setup(b"m_gh7_0_0b_test_ck", 1 << 10).unwrap();

    // Shape: `left = right = 2, num_cons = 4, ell1 = ell2 = 1`, matching the
    // M.GH7.0.0 sibling test's shape. The dispatch suggested `num_cons = 16`,
    // but the pre-existing vendor `batch_diff_size` at `spartan/mod.rs:175-189`
    // panics with `range start index N out of range for slice of length 4`
    // whenever `size_max / num_chunks > 0` AND the polynomials being batched
    // have heterogeneous lengths (`W.W` of length `num_vars = num_cons`, but
    // `E1, E2` of lengths `left, right < num_cons`). The M.GH7.0.0 sibling
    // sized to `num_cons = 4` for exactly this reason — `chunk_size = 0` then
    // falls to the safe non-chunked branch at line 200. `num_cons = 4` is
    // soundness-equivalent (covers the same algebra paths: tensor-form
    // weighting, non-zero T claim, FS-binding); the 1000-iter weighting
    // differential at `m_gh7_0_0b_weighting_differential_1000_iter` covers
    // larger shapes via pure-algebra MLE comparison without the batch path.
    let (S, U, W, E1, E2, T) = build_T_form_satisfying_instance::<E>(&ck, &mut rng, 4);

    let (pk, vk) = <RelaxedR1CSSNARK<E, EE> as RelaxedR1CSSNARKTrait<E>>::setup(&ck, &S).unwrap();

    // Spartan-side proves on derandomized witness/instance; the
    // Pedersen-additive binding `comm_E1 + comm_E2 == U.comm_E` is enforced
    // at the M.GH7.0.2 envelope, not inside this sibling (mirrors M.GH7.0.0).
    let dk = <E::CE as CommitmentEngineTrait<E>>::derand_key(&ck);
    let (W_derand, blind_W, blind_E) = W.derandomize();
    let U_derand = U.derandomize(&dk, &blind_W, &blind_E);

    // ZERO blindings on (E1, E2) — same pattern as M.GH7.0.0.
    let r_E1 = E::Scalar::ZERO;
    let r_E2 = E::Scalar::ZERO;
    let comm_E1 = <E::CE as CommitmentEngineTrait<E>>::commit(&ck, &E1, &r_E1);
    let comm_E2 = <E::CE as CommitmentEngineTrait<E>>::commit(&ck, &E2, &r_E2);

    let U_bridged = bridged_from(&U_derand, comm_E1, comm_E2, T);

    let snark = RelaxedR1CSSNARK::<E, EE>::prove_with_T_claim_split_error(
      &ck, &pk, &S, &U_bridged, &W_derand, comm_E1, comm_E2, &E1, &E2, T, r_E1, r_E2,
    )
    .expect("prove_with_T_claim_split_error must succeed for a satisfying neutron-form witness");

    snark
      .verify_with_T_claim_split_error(&vk, &U_bridged, comm_E1, comm_E2)
      .expect("verify_with_T_claim_split_error must accept a valid β' proof");
  }

  #[test]
  fn m_gh7_0_0b_prove_with_T_claim_split_error_neutron_form_round_trip_byte_equal() {
    type E1Engine = Bn256EngineKZG;
    type EE1 = crate::provider::hyperkzg::EvaluationEngine<E1Engine>;
    round_trip_with::<E1Engine, EE1>();

    type E2Engine = PallasEngine;
    type EE2 = crate::provider::ipa_pc::EvaluationEngine<E2Engine>;
    round_trip_with::<E2Engine, EE2>();
  }

  /// **Differential test (M.GH7.0.0b dispatch criterion (b)).** 1000-iter
  /// ChaCha20Rng-seeded byte-equal differential against the
  /// `EqPolynomial`-free reference `MultilinearPolynomial::new(full_E).evaluate(&r_x)`
  /// at the Corrigendum #9 (2-A) variable-order convention.
  ///
  /// This is the *weighting-correctness* differential — independent of the
  /// sumcheck engine. It pins Falsifier D: the tensor-form `(E1, E2)`
  /// weighting evaluated under the (2-A) partition convention MUST equal the
  /// flat `full_E` MLE evaluation. The M.GH7.0.0 sibling already covers
  /// the symmetric / asymmetric ell1>=ell2 case; this M.GH7.0.0b dispatch
  /// adds the swapped-asymmetric `(left, right) = (2, 4)` shape per the
  /// pin's "tensor-form algebra is invariant under partition swap" claim.
  ///
  /// Shapes covered (per dispatch criterion (b)): {(2,2), (4,2), (2,4), (4,4)}.
  #[allow(non_snake_case)]
  fn weighting_differential_1000_iter_with<E: Engine>() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xFAC1_0700_BEEF_B_u64);

    // Dispatch criterion (b) shape coverage.
    let cases: &[(usize, usize)] = &[
      (2, 2), // ell1 = 1, ell2 = 1
      (4, 2), // ell1 = 2, ell2 = 1 (asymmetric, ell1 > ell2)
      (2, 4), // ell1 = 1, ell2 = 2 (asymmetric, ell1 < ell2 — invariant test)
      (4, 4), // ell1 = 2, ell2 = 2
    ];

    let iters_per_case = 1000 / cases.len() + 1;

    for &(left, right) in cases {
      let ell1 = left.log_2();
      let ell2 = right.log_2();

      for _ in 0..iters_per_case {
        let E1: Vec<E::Scalar> = (0..left).map(|_| E::Scalar::random(&mut rng)).collect();
        let E2: Vec<E::Scalar> = (0..right).map(|_| E::Scalar::random(&mut rng)).collect();

        // Flat layout per Corrigendum #9 (2-A): full_E[i*left + j] = E2[i] * E1[j].
        let mut full_E: Vec<E::Scalar> = Vec::with_capacity(left * right);
        for i in 0..right {
          for j in 0..left {
            full_E.push(E2[i] * E1[j]);
          }
        }

        // r_x has length ell1 + ell2; (2-A) partition: r_x_high = r_x[..ell2],
        // r_x_low = r_x[ell2..]. bind_poly_var_top binds MSB first, so the
        // first ell2 challenges bind E2's outer index `i`, the last ell1
        // challenges bind E1's inner index `j` — same byte-equal convention as
        // the M.GH7.0.0 sibling.
        let r_x: Vec<E::Scalar> = (0..ell1 + ell2)
          .map(|_| E::Scalar::random(&mut rng))
          .collect();
        let r_x_high = &r_x[..ell2];
        let r_x_low = &r_x[ell2..];

        let full_eval = MultilinearPolynomial::new(full_E).evaluate(&r_x);
        let e1_eval = MultilinearPolynomial::new(E1).evaluate(r_x_low);
        let e2_eval = MultilinearPolynomial::new(E2).evaluate(r_x_high);
        let combined = e1_eval * e2_eval;

        assert_eq!(
          full_eval, combined,
          "M.GH7.0.0b weighting differential failed at (left={}, right={}, ell1={}, ell2={})",
          left, right, ell1, ell2
        );
      }
    }
  }

  #[test]
  fn m_gh7_0_0b_weighting_differential_1000_iter() {
    weighting_differential_1000_iter_with::<Bn256EngineKZG>();
    weighting_differential_1000_iter_with::<PallasEngine>();
  }

  /// **Negative test (M.GH7.0.0b dispatch criterion (c)).** Inject `T' ≠ T`
  /// into the `BridgedNeutronInstance` consumed by the verifier — the verifier
  /// MUST reject. This pins Primitive 5 binding (absorb-T-before-squeeze):
  /// because the prover absorbed the honest `T` into the FS transcript and
  /// the verifier reconstructs the transcript with `T'`, the squeezed
  /// challenges diverge and `claim_outer_final` reconstruction fails.
  #[allow(non_snake_case)]
  fn negative_T_rejection_with<E, EE>()
  where
    E: Engine,
    EE: EvaluationEngineTrait<E>,
  {
    let mut rng = ChaCha20Rng::seed_from_u64(0xDEADBEEF_0700_00B_u64);
    let ck = <E::CE as CommitmentEngineTrait<E>>::setup(b"m_gh7_0_0b_neg_ck", 1 << 10).unwrap();

    // num_cons = 4 (see round_trip_with rationale for the shape-floor pin).
    let (S, U, W, E1, E2, T) = build_T_form_satisfying_instance::<E>(&ck, &mut rng, 4);
    let (pk, vk) = <RelaxedR1CSSNARK<E, EE> as RelaxedR1CSSNARKTrait<E>>::setup(&ck, &S).unwrap();

    let dk = <E::CE as CommitmentEngineTrait<E>>::derand_key(&ck);
    let (W_derand, blind_W, blind_E) = W.derandomize();
    let U_derand = U.derandomize(&dk, &blind_W, &blind_E);

    let r_E1 = E::Scalar::ZERO;
    let r_E2 = E::Scalar::ZERO;
    let comm_E1 = <E::CE as CommitmentEngineTrait<E>>::commit(&ck, &E1, &r_E1);
    let comm_E2 = <E::CE as CommitmentEngineTrait<E>>::commit(&ck, &E2, &r_E2);

    // Prove with HONEST T.
    let U_bridged_honest = bridged_from(&U_derand, comm_E1, comm_E2, T);
    let snark = RelaxedR1CSSNARK::<E, EE>::prove_with_T_claim_split_error(
      &ck, &pk, &S, &U_bridged_honest, &W_derand, comm_E1, comm_E2, &E1, &E2, T, r_E1, r_E2,
    )
    .expect("prove must succeed with honest T");

    // Inject T' ≠ T at verify time. The verifier reconstructs the FS
    // transcript with T'; the squeezed challenges diverge from the prover's
    // transcript; claim_outer_final reconstruction fails.
    let T_corrupt = T + E::Scalar::ONE;
    let U_bridged_corrupt = bridged_from(&U_derand, comm_E1, comm_E2, T_corrupt);

    let result = snark.verify_with_T_claim_split_error(&vk, &U_bridged_corrupt, comm_E1, comm_E2);
    assert!(
      result.is_err(),
      "verify_with_T_claim_split_error MUST reject when T' != honest T (Primitive 5 FS-binding falsifier)"
    );
  }

  #[test]
  fn m_gh7_0_0b_negative_T_corruption_rejected() {
    type E1Engine = Bn256EngineKZG;
    type EE1 = crate::provider::hyperkzg::EvaluationEngine<E1Engine>;
    negative_T_rejection_with::<E1Engine, EE1>();

    type E2Engine = PallasEngine;
    type EE2 = crate::provider::ipa_pc::EvaluationEngine<E2Engine>;
    negative_T_rejection_with::<E2Engine, EE2>();
  }
  } // end mod m_gh7_0_0b
}
