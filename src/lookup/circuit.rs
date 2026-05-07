//! In-circuit user-facing lookup primitives. Stage H §H.3.
//!
//! Currently only `range_check_via_lookup` is exposed; future stages may
//! add multi-column queries (e.g., XOR tables, S-box lookups) by extending
//! the `LookupConstraintSystem` trait protocol.
//!
//! ## Soundness model
//!
//! `range_check_via_lookup` does NOT enforce native R1CS bit-decomposition
//! constraints on `value`. Soundness comes from the cumulative LogUp
//! identity that `NIFS::prove_with_lookup` proves at fold-time:
//!
//! ```text
//!   sum_i 1/(value_i + r) = sum_j ts_j/(table_j + r)
//! ```
//!
//! If a malicious prover supplies a `value` that is NOT in the table, the
//! per-step LogUp identity fails to hold and the fold step verifier
//! rejects. This shifts the soundness obligation to the lookup-fold
//! mechanism (which has its own proof — see addendum §A.4).
//!
//! ## Usage example
//!
//! ```rust,ignore
//! use nova_snark::lookup::{
//!   range_check_via_lookup, register_table, CSWithLookups,
//! };
//!
//! // 1. Register a table at setup time (off-circuit):
//! let table: Vec<_> = (0..1u64 << 8).map(|i| Scalar::from(i)).collect();
//! let handle = register_table::<E>(&ck, /*table_id=*/ 0, &table);
//!
//! // 2. At synthesis time, wrap the CS:
//! let mut cs_with_lookups = CSWithLookups::new(&mut cs);
//!
//! // 3. Call the primitive:
//! let value = AllocatedNum::alloc(&mut cs_with_lookups, || Ok(scalar))?;
//! range_check_via_lookup::<_, E>(&mut cs_with_lookups, &value, 8, &handle)?;
//!
//! // 4. After synthesis, drain the collector:
//! let queries = cs_with_lookups.flush();
//! // Stage 3 wires this into LookupPayload construction for prove_with_lookup.
//! ```
#![cfg(feature = "lookup-fold")]

use super::collector::LookupConstraintSystem;
use crate::{
  frontend::{num::AllocatedNum, SynthesisError},
  neutron::relation::{LookupTableHandle, MultiColumnLookupTable},
  traits::Engine,
};

/// Range-check `value` against the `[0, 2^n_bits)` table identified by
/// `table`.
///
/// Records the query into the `LookupConstraintSystem`'s collector. The
/// query is `(address = value, value = value)` because the
/// `[0, 2^n_bits)` range table is the identity table — a witness `v` is in
/// range iff there exists a row `(v, v)` in the table.
///
/// Asserts at synthesis time that `table.size == 2^n_bits` (a misconfigured
/// handle is a programming error, not a witness-time failure).
///
/// Stage H scope: this is a query-collection primitive only — the LogUp
/// identity is enforced cumulatively by `NIFS::prove_with_lookup`. See the
/// module-level docs for the soundness model.
pub fn range_check_via_lookup<CS, E>(
  cs: &mut CS,
  value: &AllocatedNum<E::Scalar>,
  n_bits: usize,
  table: &LookupTableHandle<E>,
) -> Result<(), SynthesisError>
where
  CS: LookupConstraintSystem<E::Scalar>,
  E: Engine,
{
  assert_eq!(
    table.size,
    1usize << n_bits,
    "range_check_via_lookup: handle.size ({}) != 2^n_bits (2^{} = {})",
    table.size,
    n_bits,
    1usize << n_bits,
  );

  // For the `[0, 2^n_bits)` identity table, address ≡ value and there are
  // no separate value columns: a witness `v` is in range iff the address
  // `v` itself is a row of the identity table. Stage I-pri's multi-column
  // lookup reduces to single-column when `values.is_empty()` — the LogUp
  // combine becomes `combined[k] = address[k]`, no α is squeezed (the
  // multi-column transcript path is gated on `num_value_columns > 0`),
  // and the existing Stage B/D/E single-column algebra is exercised
  // byte-for-byte.
  cs.register_lookup_query(table.table_id, value.clone(), Vec::new());
  Ok(())
}

/// Multi-column lookup primitive (Stage I-pri, Lasso §6.2 address-value
/// combine).
///
/// Records a query of the form `(address, v₁, v₂, ..., v_c)` against the
/// table identified by `table`. The query asserts that the row at index
/// `address` of the table equals `(v₁, v₂, ..., v_c)`.
///
/// ## Soundness model
///
/// Soundness is via Lasso's address-value combine: a fresh FS challenge
/// `α` is squeezed AFTER all column commitments are absorbed and BEFORE
/// the LogUp randomness. The combined witness/table:
///
/// ```text
/// W_combined[k] = address[k] + α·v₁[k] + α²·v₂[k] + ... + α^c·v_c[k]
/// T_combined[i] = i           + α·T₁[i]  + α²·T₂[i]  + ... + α^c·T_c[i]
/// ```
///
/// is then proved against single-column LogUp via the existing Stage B
/// `LookupSumcheckInstance`. Stage B's degree-5 algebra is preserved —
/// only the witness pre-processing changes.
///
/// ## Compatibility with `range_check_via_lookup`
///
/// `range_check_via_lookup` is the `values.is_empty()` special case of
/// this primitive: the table is the identity `{0, ..., 2^n-1}`, the
/// combined witness reduces to `address[k]`, and α is unused (and not
/// squeezed in the FS transcript — Stage H byte-equivalence pin).
pub fn lookup_via_address<CS, E>(
  cs: &mut CS,
  address: &AllocatedNum<E::Scalar>,
  values: &[AllocatedNum<E::Scalar>],
  table: &MultiColumnLookupTable<E>,
) -> Result<(), SynthesisError>
where
  CS: LookupConstraintSystem<E::Scalar>,
  E: Engine,
{
  assert_eq!(
    values.len(),
    table.columns.len(),
    "lookup_via_address: number of value columns supplied ({}) must match \
     table column count ({})",
    values.len(),
    table.columns.len(),
  );
  assert_eq!(
    table.columns.len(),
    table.value_commitments.len(),
    "lookup_via_address: table internal invariant — columns ({}) must equal \
     value_commitments ({})",
    table.columns.len(),
    table.value_commitments.len(),
  );
  cs.register_lookup_query(table.table_id, address.clone(), values.to_vec());
  Ok(())
}
