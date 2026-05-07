//! This module defines relations used in the Neutron folding scheme
use crate::{
  errors::NovaError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
  spartan::math::Math,
  traits::{commitment::CommitmentEngineTrait, AbsorbInRO2Trait, Engine, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A handle to a lookup table that has been registered with a commitment key.
///
/// Carried in [`Structure::lookups`] (via [`LookupShape`]) and absorbed into the
/// pp-digest as part of the running `Structure<E>`. The commitment is computed
/// once at table-registration time and pins the table's identity for every
/// subsequent fold step on the IVC chain.
///
/// Pinned by `c1-beta-lookup-fold-soundness-sketch-addendum-spike-design-pins.md`
/// §A.1.1 property 4 ("table commitment provenance — public input, not per-step
/// witness").
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupTableHandle<E: Engine> {
  /// Stable identifier for the table; ordering on `LookupShape::tables` is
  /// canonicalised by sorting on this id (addendum §A.1.1 property 4 line "sorted by table-id").
  pub table_id: u64,
  /// Number of entries in the table.
  pub size: usize,
  /// Commitment to the table contents under the table-derivation generator.
  pub commitment: Commitment<E>,
}

/// Lookup-side shape for a [`Structure`] that uses the lookup-fold extension.
///
/// Pinned by addendum §A.2.1: `comm_L`, `comm_ts`, `comm_inv_w`, `comm_inv_t`
/// per-step witness shapes are described here, and the table identities are
/// fixed via `tables` (which is what binds the LogUp identity's table side).
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupShape<E: Engine> {
  /// Tables registered for this `Structure`. Order canonicalised by ascending
  /// `table_id` for deterministic `pp_digest` derivation (addendum §A.1.1
  /// property 4).
  pub tables: Vec<LookupTableHandle<E>>,
  /// Number of address columns in the per-step lookup witness layout.
  pub num_addr_columns: usize,
  /// Number of witness columns in the per-step lookup witness layout.
  pub num_witness_columns: usize,
  /// Number of variables in the witness-side `eq` polynomial; the pooled
  /// per-step witness vector is padded to length `2^witness_ell_cached`.
  ///
  /// Populated by `LookupConstraintSystem::finalize()` at end-of-step (Stage 3,
  /// §H.3); set by callers explicitly at construction time for the spike
  /// scope. Pinned by §B.4.3 of the implementation outline (amendment
  /// 2026-05-07).
  pub(crate) witness_ell_cached: usize,
}

#[cfg(feature = "lookup-fold")]
impl<E: Engine> LookupShape<E> {
  /// Number of variables in the witness-side `eq` polynomial. The pooled
  /// per-step witness vector is padded to length `2^witness_ell()`.
  ///
  /// Populated by `LookupConstraintSystem::finalize()` at end-of-step
  /// (Stage 3); for direct callers (Stage 1 spike), supplied via the
  /// `witness_ell_cached` field at construction time.
  ///
  /// Pinned by §B.4.3 of the implementation outline (amendment 2026-05-07).
  pub fn witness_ell(&self) -> usize {
    self.witness_ell_cached
  }

  /// Number of variables in the table-side `eq` polynomial.
  ///
  /// Defined as `ceil(log2(Σ table.size for table in tables))` for the
  /// merged-table pool (addendum §A.3.2). For single-table use (the spike
  /// default), this is `log2(self.tables[0].size)`.
  ///
  /// Pinned by §B.4.3 of the implementation outline (amendment 2026-05-07).
  pub fn table_ell(&self) -> usize {
    let total = self.tables.iter().map(|h| h.size).sum::<usize>();
    if total == 0 {
      0
    } else {
      total.next_power_of_two().trailing_zeros() as usize
    }
  }

  /// Split of `witness_ell()` into `(left, right)` matching `Structure`'s split
  /// for tensor-form `eq` reuse with `evaluation_points_cubic_with_two_inputs`
  /// (mirrors the R1CS-side split at `Structure::new`).
  ///
  /// Returns `(2^ell1, 2^ell2)` with `ell1 = ⌈ell/2⌉`, `ell2 = ⌊ell/2⌋`,
  /// and `ell1 + ell2 = witness_ell()`.
  ///
  /// Pinned by §B.4.3 of the implementation outline (amendment 2026-05-07).
  pub fn witness_split(&self) -> (usize, usize) {
    let ell = self.witness_ell();
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    (1 << ell1, 1 << ell2)
  }

  /// Split of `table_ell()` into `(left, right)` for tensor-form `eq` reuse.
  ///
  /// Pinned by §B.4.3 of the implementation outline (amendment 2026-05-07).
  pub fn table_split(&self) -> (usize, usize) {
    let ell = self.table_ell();
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    (1 << ell1, 1 << ell2)
  }
}

/// A type that holds structure information for a zero-fold relation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Structure<E: Engine> {
  /// Shape of the R1CS relation
  pub(crate) S: R1CSShape<E>,

  /// the number of variables in the Eq polynomial
  pub(crate) ell: usize,
  pub(crate) left: usize,
  pub(crate) right: usize,

  /// Lookup-side shape, present iff the `Structure` participates in the
  /// lookup-fold extension. `None` for the legacy zero-fold path so non-lookup
  /// `StepCircuit`s pay zero serde / per-step cost. Pinned by addendum §A.2.1
  /// (the `Option<>` wrapper is the backward-compatibility pin).
  ///
  /// Because `Structure<E>` participates in `SimpleDigestible` (see
  /// `vendor/nova/src/neutron/mod.rs:54`), the `Serialize`/`Deserialize` derive
  /// on this field makes the table commitments and sizes part of the
  /// `pp_digest` automatically. This satisfies addendum §A.1.1 property 4
  /// without a separate `Structure::digest()` method.
  #[cfg(feature = "lookup-fold")]
  pub(crate) lookups: Option<LookupShape<E>>,
}

/// A type that holds witness information for a zero-fold relation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedWitness<E: Engine> {
  /// Running witness of the main relation
  pub(crate) W: Vec<E::Scalar>,
  r_W: E::Scalar,

  /// eq polynomial in tensor form
  pub(crate) E: Vec<E::Scalar>,
  r_E: E::Scalar,
}

/// A type that holds instance information for a zero-fold relation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedInstance<E: Engine> {
  pub(crate) comm_W: Commitment<E>,
  pub(crate) comm_E: Commitment<E>,
  pub(crate) T: E::Scalar,
  pub(crate) u: E::Scalar,
  pub(crate) X: Vec<E::Scalar>,

  /// Folded per-step lookup-witness commitment. `None` until a non-default
  /// running instance has been produced via a fold step that carried
  /// lookup data. Pinned by addendum §A.2.1.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_L: Option<Commitment<E>>,
  /// Folded multiplicity-vector commitment.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_ts: Option<Commitment<E>>,
  /// Folded inverse-witness commitment for `1/(w_i + r)`.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_w: Option<Commitment<E>>,
  /// Folded inverse-table commitment for `1/(T_j + r)`.
  #[cfg(feature = "lookup-fold")]
  pub(crate) comm_inv_t: Option<Commitment<E>>,
  /// Running target for the lookup-zero side. Mirrors the existing `T` for
  /// the R1CS-zero side (§A.2.1).
  #[cfg(feature = "lookup-fold")]
  pub(crate) T_lookup: Option<E::Scalar>,
}

/// Per-step lookup-side payload delivered to [`NIFS::prove`] alongside the
/// incoming [`R1CSInstance`].
///
/// Carries the per-step lookup commitments and the per-step `T2_lookup` evaluation
/// target. `NIFS::prove` absorbs the `comm_L` / `comm_ts` fields at addendum
/// §A.1.1 step (2), squeezes `r`, and absorbs `comm_inv_w` / `comm_inv_t` at
/// §A.1.1 step (6).
#[cfg(feature = "lookup-fold")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupPayload<E: Engine> {
  /// Per-step lookup-witness commitment (the pooled query vector under the
  /// `LookupConstraintSystem` collector — addendum §A.3.2).
  pub comm_L: Commitment<E>,
  /// Per-step multiplicity-vector commitment.
  pub comm_ts: Commitment<E>,
  /// Per-step inverse-witness commitment for `1/(w_i + r)`. Computed AFTER
  /// `r` is squeezed in §A.1.1 step (5); supplied at this point as a fully
  /// committed value so `NIFS::prove` can absorb it at step (6).
  pub comm_inv_w: Commitment<E>,
  /// Per-step inverse-table commitment for `1/(T_j + r)`.
  pub comm_inv_t: Commitment<E>,
  /// Per-step lookup-zero evaluation target. For a fresh per-step `R1CSInstance`
  /// satisfying the lookup relation, this is the per-step LogUp residual
  /// evaluated at the bound randomness (the analogue of `T2 = 0` for the
  /// R1CS-zero side at line 233 of `vendor/nova/src/neutron/nifs.rs`).
  pub T2_lookup: E::Scalar,
}

impl<E: Engine> Structure<E> {
  /// Create a new structure using the provided shape
  pub fn new(S: &R1CSShape<E>) -> Self {
    // pad to the regular shape
    let S = S.pad();

    let ell = S.num_cons.next_power_of_two().log_2();

    // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
    let ell1 = ell.div_ceil(2); // This ensures ell1 >= ell2
    let ell2 = ell / 2;

    Structure {
      S: S.clone(),
      ell,
      left: 1 << ell1,
      right: 1 << ell2,
      #[cfg(feature = "lookup-fold")]
      lookups: None,
    }
  }

  /// Create a new structure with a lookup-side shape attached.
  ///
  /// Constructs a `Structure<E>` whose `lookups` field is `Some(shape)`. The
  /// `tables` vec is canonicalised by sorting on `table_id` in place so the
  /// `pp_digest` derivation is deterministic regardless of caller-side
  /// insertion order (addendum §A.1.1 property 4).
  #[cfg(feature = "lookup-fold")]
  pub fn new_with_lookups(S: &R1CSShape<E>, mut shape: LookupShape<E>) -> Self {
    let mut s = Self::new(S);
    shape.tables.sort_by_key(|h| h.table_id);
    s.lookups = Some(shape);
    s
  }

  /// Check if the witness is satisfying
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<E>,
    U: &FoldedInstance<E>,
    W: &FoldedWitness<E>,
  ) -> Result<(), NovaError> {
    // check if the witness is satisfying
    let z = [W.W.clone(), vec![U.u], U.X.clone()].concat();
    let (Az, Bz, Cz) = self.S.multiply_vec(&z)?;

    // full_E is the outer product of E1 and E2
    // E1 and E2 are splits of E
    let (E1, E2) = W.E.split_at(self.left);
    let mut full_E = vec![E::Scalar::ONE; self.left * self.right];
    for i in 0..self.right {
      for j in 0..self.left {
        full_E[i * self.left + j] = E2[i] * E1[j];
      }
    }

    let sum = full_E
      .par_iter()
      .zip(Az.par_iter())
      .zip(Bz.par_iter())
      .zip(Cz.par_iter())
      .map(|(((e, a), b), c)| *e * ((*a) * (*b) - *c))
      .reduce(|| E::Scalar::ZERO, |acc, x| acc + x);

    if sum != U.T {
      return Err(NovaError::UnSat {
        reason: format!("sum != U.T\n    sum: {:?}\n    U.T: {:?}", sum, U.T),
      });
    }

    // check the validity of the commitments
    let comm_W = E::CE::commit(ck, &W.W, &W.r_W);
    let comm_E = E::CE::commit(ck, &W.E, &W.r_E);

    if comm_W != U.comm_W || comm_E != U.comm_E {
      return Err(NovaError::UnSat {
        reason: "comm_W != U.comm_W || comm_E != U.comm_E".to_string(),
      });
    }

    Ok(())
  }
}

impl<E: Engine> FoldedWitness<E> {
  /// Create a default witness
  pub fn default(S: &Structure<E>) -> Self {
    FoldedWitness {
      W: vec![E::Scalar::ZERO; S.S.num_vars],
      r_W: E::Scalar::ZERO,
      E: vec![E::Scalar::ZERO; S.left + S.right],
      r_E: E::Scalar::ZERO,
    }
  }

  /// Fold the witness with another witness
  pub fn fold(
    &self,
    W2: &R1CSWitness<E>,
    E2: &Vec<E::Scalar>,
    r_E2: &E::Scalar,
    r_b: &E::Scalar,
  ) -> Result<Self, NovaError> {
    // we need to compute the weighted sum using weights of (1-r_b) and r_b
    let W = self
      .W
      .par_iter()
      .zip(W2.W.par_iter())
      .map(|(w1, w2)| *w1 + *r_b * (*w2 - *w1))
      .collect::<Vec<_>>();
    let r_W = (E::Scalar::ONE - r_b) * self.r_W + *r_b * W2.r_W;

    let E = self
      .E
      .par_iter()
      .zip(E2.par_iter())
      .map(|(e1, e2)| *e1 + *r_b * (*e2 - *e1))
      .collect::<Vec<_>>();
    let r_E = (E::Scalar::ONE - r_b) * self.r_E + *r_b * r_E2;

    Ok(Self { W, r_W, E, r_E })
  }
}

impl<E: Engine> FoldedInstance<E> {
  /// Create a default instance
  pub fn default(S: &Structure<E>) -> Self {
    FoldedInstance {
      comm_W: Commitment::<E>::default(),
      comm_E: Commitment::<E>::default(),
      T: E::Scalar::ZERO,
      u: E::Scalar::ZERO,
      X: vec![E::Scalar::ZERO; S.S.num_io],
      // Pinned by addendum §A.2.1: defaults to `None` so non-lookup `StepCircuit`s
      // pay zero per-step cost. The lookup-side fields become `Some` only after
      // a fold step that carried a lookup payload.
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
    }
  }

  /// Fold the instance with another instance
  pub fn fold(
    &self,
    U2: &R1CSInstance<E>,
    comm_E: &Commitment<E>,
    r_b: &E::Scalar,
    T_out: &E::Scalar,
  ) -> Result<Self, NovaError> {
    // we need to compute the weighted sum using weights of (1-r_b) and r_b
    let comm_W = self.comm_W * (E::Scalar::ONE - r_b) + U2.comm_W * *r_b;
    let comm_E = self.comm_E * (E::Scalar::ONE - r_b) + *comm_E * *r_b;
    let X = self
      .X
      .par_iter()
      .zip(U2.X.par_iter())
      .map(|(x1, x2)| (E::Scalar::ONE - r_b) * x1 + *r_b * x2)
      .collect::<Vec<_>>();
    let u = (E::Scalar::ONE - r_b) * self.u + r_b;

    Ok(Self {
      comm_W,
      comm_E,
      T: *T_out,
      u,
      X,
      #[cfg(feature = "lookup-fold")]
      comm_L: self.comm_L,
      #[cfg(feature = "lookup-fold")]
      comm_ts: self.comm_ts,
      #[cfg(feature = "lookup-fold")]
      comm_inv_w: self.comm_inv_w,
      #[cfg(feature = "lookup-fold")]
      comm_inv_t: self.comm_inv_t,
      #[cfg(feature = "lookup-fold")]
      T_lookup: self.T_lookup,
    })
  }

  /// Fold the instance together with a per-step lookup payload.
  ///
  /// This is the lookup-fold extension's analogue of [`Self::fold`]. The
  /// R1CS-zero side is folded identically to [`Self::fold`]; the lookup-zero
  /// side folds the four per-step lookup commitments into the running ones
  /// using the same `r_b` weight (Path β common-challenge composition,
  /// addendum §A.2.4).
  ///
  /// On the FIRST fold of a default running instance, `self.comm_L` is `None`
  /// and the lookup-side weighted sum degenerates to `0 * (1 - r_b) +
  /// payload * r_b = payload * r_b` (treating absent commitments as the
  /// identity / zero element). This matches the invariant that a default
  /// running instance carries no committed lookup data.
  #[cfg(feature = "lookup-fold")]
  pub fn fold_with_lookup(
    &self,
    U2: &R1CSInstance<E>,
    comm_E: &Commitment<E>,
    r_b: &E::Scalar,
    T_out: &E::Scalar,
    payload: &LookupPayload<E>,
    T_lookup_out: &E::Scalar,
  ) -> Result<Self, NovaError> {
    // R1CS-zero side: identical to `Self::fold`.
    let comm_W = self.comm_W * (E::Scalar::ONE - r_b) + U2.comm_W * *r_b;
    let comm_E = self.comm_E * (E::Scalar::ONE - r_b) + *comm_E * *r_b;
    let X = self
      .X
      .par_iter()
      .zip(U2.X.par_iter())
      .map(|(x1, x2)| (E::Scalar::ONE - r_b) * x1 + *r_b * x2)
      .collect::<Vec<_>>();
    let u = (E::Scalar::ONE - r_b) * self.u + r_b;

    // Lookup-zero side: fold each commitment independently under the SAME r_b
    // (the common Fiat-Shamir challenge — Path β, addendum §A.2.4). Default
    // (`None`) running commitments contribute zero to the (1 - r_b) weight.
    let one_minus_rb = E::Scalar::ONE - r_b;
    let zero = Commitment::<E>::default();
    let fold_one = |running: Option<Commitment<E>>, fresh: Commitment<E>| -> Commitment<E> {
      let r = running.unwrap_or(zero);
      r * one_minus_rb + fresh * *r_b
    };
    let comm_L_new = fold_one(self.comm_L, payload.comm_L);
    let comm_ts_new = fold_one(self.comm_ts, payload.comm_ts);
    let comm_inv_w_new = fold_one(self.comm_inv_w, payload.comm_inv_w);
    let comm_inv_t_new = fold_one(self.comm_inv_t, payload.comm_inv_t);

    Ok(Self {
      comm_W,
      comm_E,
      T: *T_out,
      u,
      X,
      comm_L: Some(comm_L_new),
      comm_ts: Some(comm_ts_new),
      comm_inv_w: Some(comm_inv_w_new),
      comm_inv_t: Some(comm_inv_t_new),
      T_lookup: Some(*T_lookup_out),
    })
  }
}

impl<E: Engine> AbsorbInRO2Trait<E> for FoldedInstance<E> {
  fn absorb_in_ro2(&self, ro: &mut E::RO2) {
    self.comm_W.absorb_in_ro2(ro);
    self.comm_E.absorb_in_ro2(ro);

    ro.absorb(self.T);
    ro.absorb(self.u);
    for x in &self.X {
      ro.absorb(*x);
    }
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
    spartan::{direct::DirectCircuit, snark::RelaxedR1CSSNARK},
    spartan::{math::Math, polys::eq::EqPolynomial},
    traits::{circuit::NonTrivialCircuit, snark::RelaxedR1CSSNARKTrait},
  };
  use rand::rngs::OsRng;

  fn test_sat_inner<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() -> Result<(), NovaError> {
    // generate a non-trivial circuit
    let num_cons: usize = 16;
    let log_num_cons = num_cons.log_2();

    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<E::Scalar>::new(num_cons));

    // synthesize the circuit's shape
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();
    let S = Structure::new(&shape);

    // test default instance-witness pair under the structure
    let W = FoldedWitness::default(&S);
    let U = FoldedInstance::default(&S);
    S.is_sat(&ck, &U, &W)?;

    // generate a satisfying instance-witness for the r1cs
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(2)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (u, w) = cs
      .r1cs_instance_and_witness(&shape, &ck)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })?;

    // generate a random eq polynomial
    let coords = (0..log_num_cons)
      .map(|_| E::Scalar::random(&mut OsRng))
      .collect::<Vec<_>>();
    let E = EqPolynomial::new(coords).evals();

    // pad witness
    let mut W = w.W.clone();
    W.resize(S.S.num_vars, E::Scalar::ZERO);

    let W = FoldedWitness {
      W,
      r_W: w.r_W,
      E: E.clone(),
      r_E: E::Scalar::random(&mut OsRng),
    };

    let U = FoldedInstance {
      comm_W: u.comm_W,
      comm_E: E::CE::commit(&ck, &E, &W.r_E),
      T: E::Scalar::ZERO,
      X: u.X.clone(),
      u: E::Scalar::ONE,
      // Stage 1.A added these `Option<>` lookup-fold fields. The legacy
      // `test_sat_inner` exercises the R1CS-zero side only; the lookup-side
      // is `None` to mirror a default running instance with no
      // `LookupShape` attached (cf. `FoldedInstance::default`). Without
      // these initialisers the test fails to compile under
      // `--features lookup-fold`.
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

    S.is_sat(&ck, &U, &W)
  }

  #[test]
  fn test_sat() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;
    let res = test_sat_inner::<E, S>();
    assert!(res.is_ok());
  }
}
