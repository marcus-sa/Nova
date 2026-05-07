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
  neutron::relation::LookupTableHandle,
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

  // For the identity table, address == value.
  cs.register_lookup_query(table.table_id, value.clone(), value.clone());
  Ok(())
}
