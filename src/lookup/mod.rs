//! User-callable lookup primitives for the C1-β lookup-fold extension.
//!
//! This module exposes:
//!
//! - [`register_table`] — registers a table with a commitment key, returning
//!   a [`LookupTableHandle`] that can be carried in a [`LookupShape`] and
//!   consumed by `range_check_via_lookup`.
//! - [`range_check_via_lookup`] — the user-facing range-check primitive.
//!   Records a lookup query against a table without enforcing native R1CS
//!   constraints; the LogUp identity is enforced cumulatively at fold time
//!   via `NIFS::prove_with_lookup`.
//! - [`range_check_native`] — the off-circuit reference (matches the
//!   `Gadget` trait contract from `.claude/rules/cryptography.md`).
//! - [`LookupConstraintSystem`] / [`QueryCollector`] / [`CSWithLookups`] —
//!   the collector trait and per-step accumulator that gather queries
//!   emitted during synthesis. The collector's `flush()` output feeds into
//!   the per-step `LookupPayload` (Stage 3 wires this into the `RecursiveSNARK`
//!   prover loop).
//!
//! All items are gated behind `--features lookup-fold`.
#![cfg(feature = "lookup-fold")]

pub mod circuit;
mod collector;
pub mod reference;

pub use crate::neutron::relation::{LookupTableHandle, MultiColumnLookupTable};
pub use circuit::{lookup_via_address, range_check_via_lookup};
pub use collector::{CSWithLookups, LookupConstraintSystem, LookupQuery, QueryCollector};
pub use reference::range_check_native;

use crate::{
  traits::{commitment::CommitmentEngineTrait, Engine},
  CommitmentKey, CE,
};
use ff::Field;

/// Register a table with the commitment key, returning a
/// [`LookupTableHandle`] that pins the table's identity.
///
/// The commitment is computed once at registration time using zero blinding
/// (the table is public data) and is bound to the `Structure<E>`'s
/// `pp_digest` through `LookupShape::tables` — see addendum §A.1.1
/// property 4 ("table commitment provenance — public input, not per-step
/// witness").
///
/// `table_id` should be a stable identifier chosen by the caller; in the
/// per-step `LookupShape`, the tables are sorted by `table_id` for
/// deterministic `pp_digest` derivation regardless of insertion order.
pub fn register_table<E: Engine>(
  ck: &CommitmentKey<E>,
  table_id: u64,
  values: &[E::Scalar],
) -> LookupTableHandle<E> {
  let commitment = CE::<E>::commit(ck, values, &E::Scalar::ZERO);
  LookupTableHandle {
    table_id,
    size: values.len(),
    commitment,
  }
}

/// Register a multi-column lookup table (Stage I-pri).
///
/// Each column is committed independently (zero blinding — table data is
/// public). The address column is implicit (`i ∈ [0, size)`), so it has
/// no separate commitment. All columns must have the same length.
///
/// The verifier reconstructs the combined-table commitment by linear
/// homomorphism over a fresh FS challenge α once per fold step:
///
/// ```text
/// comm_T_combined  =  comm_address  +  α·comm_columns[0]  +  α²·comm_columns[1] + ...
/// ```
///
/// where `comm_address` is the publicly-known commitment to
/// `(0, 1, ..., size-1)`. This is computed by the prover/verifier at
/// fold time, not stored on the handle.
pub fn register_multi_column_table<E: Engine>(
  ck: &CommitmentKey<E>,
  table_id: u64,
  columns: &[Vec<E::Scalar>],
) -> MultiColumnLookupTable<E> {
  if columns.is_empty() {
    return MultiColumnLookupTable {
      table_id,
      size: 0,
      columns: Vec::new(),
      value_commitments: Vec::new(),
    };
  }
  let size = columns[0].len();
  for (i, c) in columns.iter().enumerate() {
    assert_eq!(
      c.len(),
      size,
      "register_multi_column_table: column {} has length {} but column 0 has length {}",
      i,
      c.len(),
      size,
    );
  }
  let value_commitments: Vec<_> = columns
    .iter()
    .map(|col| CE::<E>::commit(ck, col, &E::Scalar::ZERO))
    .collect();
  MultiColumnLookupTable {
    table_id,
    size,
    columns: columns.to_vec(),
    value_commitments,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      num::AllocatedNum, util_cs::test_cs::TestConstraintSystem, ConstraintSystem,
    },
    provider::Bn256EngineKZG,
    r1cs::R1CSShape,
    traits::{snark::default_ck_hint, Engine},
  };

  type E = Bn256EngineKZG;
  type Scalar = <E as Engine>::Scalar;

  /// `register_table` produces a handle with the expected fields and a
  /// non-default (deterministic) commitment.
  #[test]
  fn register_table_produces_valid_handle() {
    // tiny commitment key — we just need at least `table_size` generators.
    let table_size = 8usize;
    // Build a minimal R1CS shape so we can derive a CK.
    let dummy_shape = R1CSShape::<E>::new(
      1,
      table_size,
      1,
      crate::r1cs::SparseMatrix::new(&[], 1, table_size + 2),
      crate::r1cs::SparseMatrix::new(&[], 1, table_size + 2),
      crate::r1cs::SparseMatrix::new(&[], 1, table_size + 2),
    )
    .unwrap();
    let ck = R1CSShape::commitment_key(&[&dummy_shape], &[&*default_ck_hint()]).unwrap();

    let values: Vec<Scalar> = (0..table_size as u64).map(Scalar::from).collect();
    let h = register_table::<E>(&ck, 42, &values);
    assert_eq!(h.table_id, 42);
    assert_eq!(h.size, table_size);

    // The same inputs must produce the same commitment (deterministic).
    let h2 = register_table::<E>(&ck, 42, &values);
    assert_eq!(h.commitment, h2.commitment);

    // Different table_ids leave the commitment intact (only label changes).
    let h3 = register_table::<E>(&ck, 99, &values);
    assert_eq!(h.commitment, h3.commitment);
    assert_ne!(h.table_id, h3.table_id);
  }

  /// `CSWithLookups` correctly delegates `ConstraintSystem` synthesis: a
  /// trivial `x * x = x^2` circuit synthesised through the wrapper produces
  /// the same satisfaction status as on the inner CS.
  #[test]
  fn cs_with_lookups_delegates_constraint_system() {
    let mut inner = TestConstraintSystem::<Scalar>::new();
    {
      let mut wrapped = CSWithLookups::new(&mut inner);
      let x = AllocatedNum::alloc(wrapped.namespace(|| "x"), || Ok(Scalar::from(7))).unwrap();
      let xx =
        AllocatedNum::alloc(wrapped.namespace(|| "xx"), || Ok(Scalar::from(49))).unwrap();
      wrapped.enforce(
        || "x * x = xx",
        |lc| lc + x.get_variable(),
        |lc| lc + x.get_variable(),
        |lc| lc + xx.get_variable(),
      );
      // No lookups recorded.
      assert!(wrapped.collector().is_empty());
    }
    assert!(
      inner.is_satisfied(),
      "delegated synthesis must produce a satisfied CS"
    );
  }

  /// `range_check_via_lookup` records the expected query via `flush()`.
  #[test]
  fn range_check_via_lookup_records_query() {
    let n_bits = 4usize;
    let table_size = 1usize << n_bits;
    let dummy_shape = R1CSShape::<E>::new(
      1,
      table_size,
      1,
      crate::r1cs::SparseMatrix::new(&[], 1, table_size + 2),
      crate::r1cs::SparseMatrix::new(&[], 1, table_size + 2),
      crate::r1cs::SparseMatrix::new(&[], 1, table_size + 2),
    )
    .unwrap();
    let ck = R1CSShape::commitment_key(&[&dummy_shape], &[&*default_ck_hint()]).unwrap();
    let values: Vec<Scalar> = (0..table_size as u64).map(Scalar::from).collect();
    let handle = register_table::<E>(&ck, /*table_id=*/ 7, &values);

    let mut inner = TestConstraintSystem::<Scalar>::new();
    let queries = {
      let mut wrapped = CSWithLookups::new(&mut inner);
      let v = AllocatedNum::alloc(wrapped.namespace(|| "v"), || Ok(Scalar::from(11))).unwrap();
      range_check_via_lookup::<_, E>(&mut wrapped, &v, n_bits, &handle).unwrap();
      let v2 = AllocatedNum::alloc(wrapped.namespace(|| "v2"), || Ok(Scalar::from(3))).unwrap();
      range_check_via_lookup::<_, E>(&mut wrapped, &v2, n_bits, &handle).unwrap();
      wrapped.flush()
    };
    assert_eq!(queries.len(), 2);
    assert_eq!(queries[0].table_id, 7);
    assert_eq!(queries[1].table_id, 7);
    // Stage I-pri: range_check_via_lookup is the empty-values special case;
    // the address carries the queried value.
    assert!(queries[0].values.is_empty());
    assert!(queries[1].values.is_empty());
    assert_eq!(queries[0].address.get_value(), Some(Scalar::from(11)));
    assert_eq!(queries[1].address.get_value(), Some(Scalar::from(3)));
    assert!(inner.is_satisfied());
  }

  /// `range_check_via_lookup` panics if `table.size` doesn't match `2^n_bits`.
  #[test]
  #[should_panic(expected = "range_check_via_lookup: handle.size")]
  fn range_check_via_lookup_size_mismatch_panics() {
    let dummy_shape = R1CSShape::<E>::new(
      1,
      8,
      1,
      crate::r1cs::SparseMatrix::new(&[], 1, 10),
      crate::r1cs::SparseMatrix::new(&[], 1, 10),
      crate::r1cs::SparseMatrix::new(&[], 1, 10),
    )
    .unwrap();
    let ck = R1CSShape::commitment_key(&[&dummy_shape], &[&*default_ck_hint()]).unwrap();
    // Table claims size 8 (2^3) but caller asks for n_bits=4 (2^4 = 16).
    let values: Vec<Scalar> = (0..8u64).map(Scalar::from).collect();
    let handle = register_table::<E>(&ck, 0, &values);

    let mut inner = TestConstraintSystem::<Scalar>::new();
    let mut wrapped = CSWithLookups::new(&mut inner);
    let v = AllocatedNum::alloc(wrapped.namespace(|| "v"), || Ok(Scalar::from(5))).unwrap();
    let _ = range_check_via_lookup::<_, E>(&mut wrapped, &v, /*n_bits=*/ 4, &handle);
  }

  /// `range_check_native` rejects values >= 2^n_bits and accepts values
  /// < 2^n_bits (round-trip on the in-circuit primitive's contract).
  #[test]
  fn range_check_native_round_trip_against_in_circuit_contract() {
    for n_bits in [1usize, 2, 3, 8, 16] {
      let limit = 1u64 << n_bits;
      for v in 0..limit {
        assert!(range_check_native::<Scalar>(Scalar::from(v), n_bits).is_ok());
      }
      // boundary
      assert!(range_check_native::<Scalar>(Scalar::from(limit), n_bits).is_err());
      // far above
      if n_bits < 32 {
        assert!(
          range_check_native::<Scalar>(Scalar::from(limit + 17), n_bits).is_err()
        );
      }
    }
  }
}
