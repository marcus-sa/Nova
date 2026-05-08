//! This module implements various gadgets necessary for folding R1CS types with NeutronNova folding scheme.
use crate::{
  frontend::{num::AllocatedNum, Boolean, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedNonnativePoint,
    utils::{alloc_zero, conditionally_select},
  },
  neutron::{circuit::r1cs::AllocatedNonnativeR1CSInstance, relation::FoldedInstance},
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};
use ff::Field;

/// An in-circuit representation of NeutronNova's FoldedInstance
/// In our context, public IO of circuits folded will have only one entry, so we have X
#[derive(Clone, Debug)]
pub struct AllocatedFoldedInstance<E: Engine> {
  pub(crate) comm_W: AllocatedNonnativePoint<E>,
  pub(crate) comm_E: AllocatedNonnativePoint<E>,
  pub(crate) T: AllocatedNum<E::Scalar>,

  /// Per-table running lookup scalar VECTOR. `None` only at outer base
  /// when no prior fold step has carried lookup data; for fold-depth >= 1
  /// always `Some` (length pinned by the shape registry's
  /// `multi_column_tables.len()`). At outer base each entry that does
  /// exist is `alloc_zero()`.
  ///
  /// GH-#5 design pin §1.4 / §3.2 (W1) binding-via-hash: this field is
  /// absorbed in `absorb_in_ro` between `T` and `u` in `table_id`-canonical
  /// order. The absorption binds the per-table running scalar VECTOR
  /// through the FS hash that becomes `u.X[0]`, preserving the single-IO
  /// `AllocatedNonnativeR1CSInstance` absorption pattern as sound.
  ///
  /// Mirrors the native-side `FoldedInstance::T_lookup: Option<Vec<E::Scalar>>`
  /// at `vendor/nova/src/neutron/relation.rs:265`.
  #[cfg(feature = "lookup-fold")]
  pub(crate) T_lookup_per_table: Option<Vec<AllocatedNum<E::Scalar>>>,

  pub(crate) u: AllocatedNum<E::Scalar>,
  pub(crate) X: AllocatedNum<E::Scalar>,
}

impl<E: Engine> AllocatedFoldedInstance<E> {
  /// Allocates the given `FoldedInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    inst: Option<&FoldedInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    // We do not need to check that W or E are well-formed (e.g., on the curve) as we do a hash check
    // in the Nova augmented circuit, which ensures that the relaxed instance
    // came from a prior iteration of Nova.
    let comm_W = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate W"),
      inst.map(|inst| inst.comm_W.to_coordinates()),
    )?;

    let comm_E = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate E"),
      inst.map(|inst| inst.comm_E.to_coordinates()),
    )?;

    let T = AllocatedNum::alloc(cs.namespace(|| "allocate T"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |inst| inst.T))
    })?;

    let u = AllocatedNum::alloc(cs.namespace(|| "allocate u"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |inst| inst.u))
    })?;

    // Allocate X. If the input instance is None, then allocate default values 0.
    let X = AllocatedNum::alloc(cs.namespace(|| "allocate X"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |inst| inst.X[0]))
    })?;

    // GH-#5 design pin §3.2: allocate `T_lookup_per_table` from `inst.t_lookup()`.
    // Length is implicit (whatever the prior running U1 carried). At outer
    // base (`inst == None` OR `inst.t_lookup() == None`), this is `None`.
    #[cfg(feature = "lookup-fold")]
    let T_lookup_per_table = {
      let t_lookup_slice: Option<Vec<E::Scalar>> = inst
        .and_then(|inst| inst.t_lookup())
        .map(|s| s.to_vec());
      match t_lookup_slice {
        None => None,
        Some(slice) => {
          let allocated = slice
            .into_iter()
            .enumerate()
            .map(|(j, t_j)| {
              AllocatedNum::alloc(
                cs.namespace(|| format!("allocate T_lookup_per_table[{j}]")),
                || Ok(t_j),
              )
            })
            .collect::<Result<Vec<_>, _>>()?;
          Some(allocated)
        }
      }
    };

    Ok(Self {
      comm_W,
      comm_E,
      T,
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table,
      u,
      X,
    })
  }

  /// Allocates a default `RelaxedR1CSInstance` with a `lookup-fold`-active
  /// per-table running scalar vector of length `k`.
  ///
  /// W = E = 0, T = 0, T_lookup_per_table = [0; k], u = 0, X = 0.
  ///
  /// GH-#5 design pin §2.3 / §3.2: pinned by the shape registry's
  /// `multi_column_tables.len()`. Used at the augmented-circuit's
  /// `synthesize_base_case` site (M.GH5.3 wire-up); also exercised by the
  /// M.GH5.0 STAGE 0 spike harness in
  /// `crates/inumbra-spend-harness/tests/gh5_m0_stage0_*` to construct
  /// fold-depth-≥1 fixtures with non-empty `T_lookup_per_table` in a way
  /// that survives the `cfg`-gate without depending on the native
  /// multi-table prover's running U. `dead_code` allowed until M.GH5.3
  /// lands; all consumers are external (inumbra-harness integration
  /// tests) at this milestone.
  #[allow(dead_code)] // until M.GH5.3 wires `synthesize_base_case` through this constructor
  #[cfg(feature = "lookup-fold")]
  pub fn default_with_lookup_k<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    k: usize,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::default(cs.namespace(|| "allocate W"))?;
    let comm_E = comm_W.clone();
    let T = alloc_zero(cs.namespace(|| "allocate T"));
    let u = T.clone();
    let X = T.clone();

    let T_lookup_per_table = Some(
      (0..k)
        .map(|j| {
          AllocatedNum::alloc(
            cs.namespace(|| format!("allocate T_lookup_per_table[{j}] = 0")),
            || Ok(E::Scalar::ZERO),
          )
        })
        .collect::<Result<Vec<_>, _>>()?,
    );

    Ok(Self {
      comm_W,
      comm_E,
      T,
      T_lookup_per_table,
      u,
      X,
    })
  }

  /// Allocates the hardcoded default `RelaxedR1CSInstance` in the circuit.
  /// W = E = 0, T = 0, u = 0, X = 0
  ///
  /// For `lookup-fold` builds, `T_lookup_per_table = None`. This represents
  /// the structurally-empty outer base where no prior fold step has carried
  /// lookup data. The augmented-circuit's `synthesize_base_case` (under
  /// M.GH5.3) calls [`Self::default_with_lookup_k`] instead to allocate a
  /// k-typed running instance bound by the shape registry; this method is
  /// retained for non-`lookup-fold` builds and for fixtures that
  /// intentionally model the structurally-empty outer base.
  pub fn default<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::default(cs.namespace(|| "allocate W"))?;
    let comm_E = comm_W.clone();

    // Allocate T = 0. Similar to X0 and X1, we do not need to check that T is well-formed
    let T = alloc_zero(cs.namespace(|| "allocate T"));

    let u = T.clone();

    // X is allocated and set to zero
    let X = T.clone();

    Ok(Self {
      comm_W,
      comm_E,
      T,
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table: None,
      u,
      X,
    })
  }

  /// Absorb the provided instance in the RO
  ///
  /// GH-#5 design pin §1.4 (W1) binding-via-hash: under `lookup-fold`, each
  /// `T_lookup_per_table[j]` is absorbed in `table_id`-canonical order
  /// BETWEEN the existing `T` absorption and the `u`/`X` absorptions. This
  /// pins the per-table running scalar VECTOR through the FS hash that
  /// becomes `u.X[0]`, closing the (W1) cross-step substitution attack at
  /// fold-depth ≥ 1. At outer base where `T_lookup_per_table == None`,
  /// no scalars are absorbed (the sequence is empty, not skipped).
  ///
  /// Native-side mirror: [`crate::neutron::relation::FoldedInstance`]
  /// `absorb_in_ro2` (impl of [`crate::traits::AbsorbInRO2Trait`]) absorbs
  /// `T_lookup` between `T` and `u` in the same `table_id`-canonical order.
  /// Byte-equivalence is the M.GH5.0 STAGE 0 acceptance criterion.
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    ro: &mut E::RO2Circuit,
  ) -> Result<(), SynthesisError> {
    self
      .comm_W
      .absorb_in_ro(cs.namespace(|| "absorb W in RO"), ro)?;
    self
      .comm_E
      .absorb_in_ro(cs.namespace(|| "absorb E in RO"), ro)?;
    ro.absorb(&self.T);

    // GH-#5 (W1) BINDING: per-table running lookup scalar VECTOR is
    // absorbed in `table_id`-canonical order, BEFORE u/X. Pinned by the
    // shape registry's `multi_column_tables.len()`. At outer base
    // (`T_lookup_per_table == None`) no scalars are absorbed (the
    // sequence is empty, not skipped).
    #[cfg(feature = "lookup-fold")]
    if let Some(t_lookup) = &self.T_lookup_per_table {
      for t_j in t_lookup {
        ro.absorb(t_j);
      }
    }

    ro.absorb(&self.u);
    ro.absorb(&self.X);
    Ok(())
  }

  /// Folds self with an r1cs instance and returns the result
  pub fn fold<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    U2: &AllocatedNonnativeR1CSInstance<E>,
    r_b: &AllocatedNum<E::Scalar>,
    T_out: &AllocatedNum<E::Scalar>,
    comm_W_fold: &AllocatedNonnativePoint<E>,
    comm_E_fold: &AllocatedNonnativePoint<E>,
  ) -> Result<Self, SynthesisError> {
    // u_fold = (1-r_b) * self.u + r_b * U2.u
    // u_fold = self.u - r_b * self.u + r_b * U2.u
    // u_fold = self.u + r_b (U2.u - self.u)
    // In our context U2.u = 1, so u_fold = self.u + r_b (1 - self.u)
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "allocate u_fold"), || {
      let u = self
        .u
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U2_u = E::Scalar::ONE;
      Ok(u + r_b * (U2_u - u))
    })?;

    cs.enforce(
      || "enforce u_fold -self.u  = r_b (U2.u - self.u)",
      |lc| lc + r_b.get_variable(),
      |lc| lc + CS::one() - self.u.get_variable(),
      |lc| lc + u_fold.get_variable() - self.u.get_variable(),
    );

    // Fold the IO:
    // X_fold = self.X + r_b (U2.X - self.X)
    let X_fold = AllocatedNum::alloc(cs.namespace(|| "allocate X_fold"), || {
      let X = self
        .X
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U2_X = U2.X.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(X + r_b * (U2_X - X))
    })?;
    cs.enforce(
      || "enforce X_fold - self.X = r_b (U2.X - self.X)",
      |lc| lc + r_b.get_variable(),
      |lc| lc + U2.X.get_variable() - self.X.get_variable(),
      |lc| lc + X_fold.get_variable() - self.X.get_variable(),
    );

    // GH-#5 design pin §3.2: `fold` passes `T_lookup_per_table` through
    // unchanged from `self`. Post-fold update of the per-table running
    // scalar vector is the responsibility of the lookup verifier path
    // (`verify_with_multi_table_lookup`); the augmented-circuit caller at
    // M.GH5.3 wires the post-fold T_lookup values into the post-fold
    // `AllocatedFoldedInstance` via a sibling helper. Mirrors the native
    // `FoldedInstance::fold` passthrough at
    // `vendor/nova/src/neutron/relation.rs:695`.
    Ok(Self {
      comm_W: comm_W_fold.clone(),
      comm_E: comm_E_fold.clone(),
      T: T_out.clone(),
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table: self.T_lookup_per_table.clone(),
      u: u_fold,
      X: X_fold,
    })
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::conditionally_select(
      cs.namespace(|| "W = cond ? self.W : other.W"),
      &self.comm_W,
      &other.comm_W,
      condition,
    )?;

    let comm_E = AllocatedNonnativePoint::conditionally_select(
      cs.namespace(|| "E = cond ? self.E : other.E"),
      &self.comm_E,
      &other.comm_E,
      condition,
    )?;

    let T = conditionally_select(
      cs.namespace(|| "T = cond ? self.T : other.T"),
      &self.T,
      &other.T,
      condition,
    )?;

    let u = conditionally_select(
      cs.namespace(|| "u = cond ? self.u : other.u"),
      &self.u,
      &other.u,
      condition,
    )?;

    let X = conditionally_select(
      cs.namespace(|| "X[0] = cond ? self.X[0] : other.X[0]"),
      &self.X,
      &other.X,
      condition,
    )?;

    // GH-#5 design pin §2.3 / §3.2: under `lookup-fold`, both branches of
    // `conditionally_select` must carry `T_lookup_per_table` of the same
    // shape (`None`/`None`, or `Some(len=k)` / `Some(len=k)`). The
    // augmented-circuit's `synthesize_base_case` allocates via
    // `default_with_lookup_k(cs, k)` and `synthesize_non_base_case`
    // produces the post-fold instance from `verify_with_multi_table_lookup`'s
    // length-k output, so honest synthesis preserves the invariant. A
    // shape mismatch indicates a wire-up bug at the caller; this method
    // returns `SynthesisError::Unsatisfiable` rather than panicking so
    // the constraint system fails-closed at synthesis time.
    #[cfg(feature = "lookup-fold")]
    let T_lookup_per_table = match (&self.T_lookup_per_table, &other.T_lookup_per_table) {
      (None, None) => None,
      (Some(self_v), Some(other_v)) if self_v.len() == other_v.len() => {
        let selected = self_v
          .iter()
          .zip(other_v.iter())
          .enumerate()
          .map(|(j, (s, o))| {
            conditionally_select(
              cs.namespace(|| {
                format!("T_lookup_per_table[{j}] = cond ? self.T_lookup_per_table[{j}] : other.T_lookup_per_table[{j}]")
              }),
              s,
              o,
              condition,
            )
          })
          .collect::<Result<Vec<_>, _>>()?;
        Some(selected)
      }
      _ => {
        return Err(SynthesisError::Unsatisfiable(
          "AllocatedFoldedInstance::conditionally_select: T_lookup_per_table shape mismatch (None/Some or unequal lengths) — invariant violated by caller".to_string(),
        ));
      }
    };

    Ok(Self {
      comm_W,
      comm_E,
      T,
      #[cfg(feature = "lookup-fold")]
      T_lookup_per_table,
      u,
      X,
    })
  }
}
