//! `LookupConstraintSystem` trait + `QueryCollector` per-step accumulator
//! + `CSWithLookups` constraint-system wrapper. Stage H §H.4.
//!
//! The collector captures lookup queries emitted by user circuits during
//! synthesis. At the end of a step the collector is flushed and its
//! contents become input to per-step `LookupPayload` construction.
//!
//! ## Why a wrapper instead of a default trait method
//!
//! The `frontend::ConstraintSystem` trait in this crate is bellpepper-style
//! and does NOT expose the wrapped value of an `AllocatedNum` to the
//! constraint-system itself (only the variable index). A user-callable
//! `range_check_via_lookup` therefore needs the witness side-channel that
//! `LookupConstraintSystem` provides. `CSWithLookups` is the bridge: it
//! delegates every `ConstraintSystem` method to the inner CS and adds the
//! `register_lookup_query` channel.
#![cfg(feature = "lookup-fold")]

use crate::frontend::{
  num::AllocatedNum, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use ff::PrimeField;
use std::cell::RefCell;
use std::marker::PhantomData;

/// A single recorded lookup query.
///
/// `address` is the table-side index (or whatever the table's domain is);
/// `value` is the witness-side query value. For `range_check_via_lookup`
/// over a `[0, 2^n_bits)` table both fields are the same `AllocatedNum`,
/// but we keep them separate so the same collector can serve more general
/// table-lookup primitives in future stages.
#[derive(Debug, Clone)]
pub struct LookupQuery<F: PrimeField> {
  /// Stable identifier of the table this query targets. Matches
  /// `LookupTableHandle::table_id`.
  pub table_id: u64,
  /// In-circuit address (lookup-side index).
  pub address: AllocatedNum<F>,
  /// In-circuit value (witness-side query value).
  pub value: AllocatedNum<F>,
}

/// Per-step accumulator of lookup queries. Carried by `CSWithLookups`.
///
/// Interior mutability via `RefCell` — synthesis sites typically take
/// `&mut CS`, so a non-interior collector would force the user to pass
/// `&mut QueryCollector` everywhere.
#[derive(Debug, Default)]
pub struct QueryCollector<F: PrimeField> {
  queries: RefCell<Vec<LookupQuery<F>>>,
}

impl<F: PrimeField> QueryCollector<F> {
  /// Construct an empty collector.
  pub fn new() -> Self {
    Self {
      queries: RefCell::new(Vec::new()),
    }
  }

  /// Record a query.
  pub fn push(&self, q: LookupQuery<F>) {
    self.queries.borrow_mut().push(q);
  }

  /// Drain all collected queries. The collector is empty after this call.
  pub fn flush(&self) -> Vec<LookupQuery<F>> {
    std::mem::take(&mut *self.queries.borrow_mut())
  }

  /// Number of queries currently recorded.
  pub fn len(&self) -> usize {
    self.queries.borrow().len()
  }

  /// `true` if no queries have been recorded.
  pub fn is_empty(&self) -> bool {
    self.queries.borrow().is_empty()
  }
}

/// Trait extending `ConstraintSystem<F>` with a lookup-query side channel.
///
/// Implementors carry a `QueryCollector` and forward queries to it. The
/// canonical implementor is [`CSWithLookups`].
pub trait LookupConstraintSystem<F: PrimeField>: ConstraintSystem<F> {
  /// Record a lookup query against the table identified by `table_id`.
  fn register_lookup_query(
    &mut self,
    table_id: u64,
    address: AllocatedNum<F>,
    value: AllocatedNum<F>,
  );
}

/// Constraint-system wrapper that adds a `QueryCollector` to any
/// `ConstraintSystem` implementor.
///
/// All `ConstraintSystem` methods delegate to the inner CS, so the wrapper
/// is a transparent overlay for circuits that don't use lookup queries.
pub struct CSWithLookups<'a, F: PrimeField, CS: ConstraintSystem<F>> {
  inner: &'a mut CS,
  collector: QueryCollector<F>,
  _p: PhantomData<F>,
}

impl<'a, F: PrimeField, CS: ConstraintSystem<F>> CSWithLookups<'a, F, CS> {
  /// Wrap an existing constraint system. The inner CS is borrowed for the
  /// lifetime of the wrapper.
  pub fn new(inner: &'a mut CS) -> Self {
    Self {
      inner,
      collector: QueryCollector::new(),
      _p: PhantomData,
    }
  }

  /// Drain the collected queries, returning ownership of the contents.
  pub fn flush(&self) -> Vec<LookupQuery<F>> {
    self.collector.flush()
  }

  /// Borrow the underlying collector. Useful for callers that want to
  /// inspect the queue without draining.
  pub fn collector(&self) -> &QueryCollector<F> {
    &self.collector
  }
}

impl<F: PrimeField, CS: ConstraintSystem<F>> ConstraintSystem<F> for CSWithLookups<'_, F, CS> {
  type Root = CS::Root;

  fn new() -> Self {
    unimplemented!("CSWithLookups must wrap an existing CS via CSWithLookups::new(&mut cs)");
  }

  fn one() -> Variable {
    Variable::new_unchecked(Index::Input(0))
  }

  fn alloc<FN, A, AR>(&mut self, annotation: A, f: FN) -> Result<Variable, SynthesisError>
  where
    FN: FnOnce() -> Result<F, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.inner.alloc(annotation, f)
  }

  fn alloc_input<FN, A, AR>(&mut self, annotation: A, f: FN) -> Result<Variable, SynthesisError>
  where
    FN: FnOnce() -> Result<F, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.inner.alloc_input(annotation, f)
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<F>) -> LinearCombination<F>,
    LB: FnOnce(LinearCombination<F>) -> LinearCombination<F>,
    LC: FnOnce(LinearCombination<F>) -> LinearCombination<F>,
  {
    self.inner.enforce(annotation, a, b, c)
  }

  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    self.inner.push_namespace(name_fn)
  }

  fn pop_namespace(&mut self) {
    self.inner.pop_namespace()
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self.inner.get_root()
  }

  fn is_witness_generator(&self) -> bool {
    self.inner.is_witness_generator()
  }
}

impl<F: PrimeField, CS: ConstraintSystem<F>> LookupConstraintSystem<F> for CSWithLookups<'_, F, CS> {
  fn register_lookup_query(
    &mut self,
    table_id: u64,
    address: AllocatedNum<F>,
    value: AllocatedNum<F>,
  ) {
    self.collector.push(LookupQuery {
      table_id,
      address,
      value,
    });
  }
}
