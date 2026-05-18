//! This module defines traits that a step function must implement
use crate::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::PrimeField;

#[cfg(feature = "lookup-fold")]
use crate::{lookup::LookupConstraintSystem, traits::Engine};

/// A helper trait for a step of the incremental computation (i.e., circuit for F)
pub trait StepCircuit<F: PrimeField>: Send + Sync + Clone {
  /// Return the number of inputs or outputs of each step
  /// (this method is called only at circuit synthesis time)
  /// `synthesize` and `output` methods are expected to take as
  /// input a vector of size equal to arity and output a vector of size equal to arity
  fn arity(&self) -> usize;

  /// Synthesize the circuit for a computation step and return variable
  /// that corresponds to the output of the step `z_{i+1}`
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>;
}

/// A trivial step circuit that simply returns the input
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrivialCircuit<F: PrimeField> {
  _p: PhantomData<F>,
}

impl<F: PrimeField> StepCircuit<F> for TrivialCircuit<F> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    Ok(z.to_vec())
  }
}

/// A non-trivial step circuit that repeats the squaring operation `num_cons` times
#[derive(Clone, Debug, Default)]
pub struct NonTrivialCircuit<F: PrimeField> {
  num_cons: usize,
  _p: PhantomData<F>,
}

impl<F: PrimeField> NonTrivialCircuit<F> {
  /// Create a new non-trivial circuit that repeats the squaring operation `num_cons` times
  pub fn new(num_cons: usize) -> Self {
    Self {
      num_cons,
      _p: PhantomData,
    }
  }
}
impl<F: PrimeField> StepCircuit<F> for NonTrivialCircuit<F> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    // Consider an equation: `x^2 = y`, where `x` and `y` are respectively the input and output.
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..self.num_cons {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok(vec![y])
  }
}

/// C1-β BIP-340 witness/ck threading sub-corrigendum §4.1 (Halpert,
/// 2026-05-18): a parallel-trait extension of [`StepCircuit`] whose
/// synthesize-site CS bound is widened to admit
/// `LookupConstraintSystem<F>` calls (`register_lookup_query`,
/// `register_chunk_lookup_table::<E>(...)`) from inside the
/// step-circuit body.
///
/// Composition contract:
///
/// - Implementors of `StepCircuitWithAux<F, E>` MUST also implement
///   [`StepCircuit<F>`] (enforced by the super-trait bound). The
///   shape pass (under `setup_with_ptau_dir_aux`) and the prove pass
///   (under `prove_step_with_lookup_fold_aux`) both invoke
///   `synthesize_with_aux`; the non-aux `synthesize` is reserved for
///   non-lookup-aware code paths (vendor upstream-tracking
///   `prove_step` + non-`lookup-fold` builds), where the
///   `StepCircuit` impl carries the witness-independent body alone.
/// - The trait carries ONLY the CS-trait-bound widening. Auxiliary
///   inputs (BIP-340 witnesses, prove-time `ck`) remain on `&self`
///   via the parent corrigendum §3 struct extension.
///
/// Vendor-side consumer chain:
///
/// - [`NeutronAugmentedCircuit::with_step_circuit_aux`] opts the
///   augmented circuit into lookup-aware step synthesis: when set,
///   the augmented-circuit `synthesize` body wraps `cs` in
///   `CSWithLookups` before invoking `synthesize_with_aux`. The
///   local wrapper is intentionally NOT flushed — the chunk-lookup
///   binding is carried by `LookupStepCircuit::per_table_bundles_at_step`
///   off-circuit (sub-corrigendum §4.2).
/// - [`RecursiveSNARK::prove_step_with_lookup_fold_aux`] mirrors
///   `prove_step_with_lookup_fold` byte-for-byte except it sets
///   `with_step_circuit_aux(true)` on the augmented-circuit builder
///   (sub-corrigendum §4.3).
/// - [`PublicParams::setup_with_ptau_dir_aux`] mirrors
///   `setup_with_ptau_dir` byte-for-byte except it also sets
///   `with_step_circuit_aux(true)` so the shape pass and the prove
///   pass exercise the same per-PC synthesize body (sub-corrigendum
///   §6).
///
/// HG-A1.4-7 (trait-bound mismatch) DISCHARGED on this trait landing.
#[cfg(feature = "lookup-fold")]
pub trait StepCircuitWithAux<F: PrimeField, E: Engine<Scalar = F>>: StepCircuit<F> {
  /// Synthesize the step circuit with auxiliary inputs threaded from
  /// the prover. Distinct from [`StepCircuit::synthesize`] by:
  ///
  ///   1. CS bound widened to `ConstraintSystem<F> + LookupConstraintSystem<F>`,
  ///      admitting `register_lookup_query` emission from inside the
  ///      body.
  ///   2. The implementation reads auxiliary inputs (e.g., the
  ///      prove-time `ck: &CommitmentKey<E>` and per-input BIP-340
  ///      witnesses) from `&self` per the parent corrigendum's
  ///      struct extension.
  fn synthesize_with_aux<CS>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
  where
    CS: ConstraintSystem<F> + LookupConstraintSystem<F>;
}
