//! Off-circuit reference for `range_check_via_lookup`. Stage H §H.5.
//!
//! Per `.claude/rules/cryptography.md`, every primitive ships an in-circuit
//! half AND an off-circuit reference, structurally parallel. The reference
//! is production code (the wallet derives the actual range-check witnesses
//! through it), not test scaffolding.
#![cfg(feature = "lookup-fold")]

use ff::PrimeField;

/// Off-circuit range check: returns `Ok(())` iff `value` represents an
/// integer in `[0, 2^n_bits)`.
///
/// The reference compares the canonical big-endian byte representation of
/// `value` (via `PrimeField::to_repr`) against `2^n_bits`. A value is in
/// range iff:
///
/// - all bytes above the `n_bits`-th bit are zero, AND
/// - the partial byte at the boundary has zero bits above bit position
///   `n_bits % 8`.
///
/// This intentionally does NOT use `u128`-only fast paths so the same
/// reference works for `n_bits > 128` (e.g., 254-bit Bn256 / Bls12-381
/// scalar field elements).
///
/// Returns `Err(&'static str)` with a short reason on rejection.
pub fn range_check_native<F: PrimeField>(
  value: F,
  n_bits: usize,
) -> Result<(), &'static str> {
  // Special-case n_bits == 0: only zero is in range.
  if n_bits == 0 {
    return if value == F::ZERO {
      Ok(())
    } else {
      Err("range_check_native: n_bits == 0 only accepts zero")
    };
  }

  let repr = value.to_repr();
  // ff's `to_repr` returns a `Repr: AsRef<[u8]>`; the byte order depends on
  // the field implementation but is canonical (little-endian for all
  // RustCrypto primefields used in this workspace).
  let bytes = repr.as_ref();

  let n_full_bytes = n_bits / 8;
  let partial_bits = n_bits % 8;

  // Locate the byte that contains the boundary bit (if any).
  for (i, b) in bytes.iter().enumerate() {
    if i < n_full_bytes {
      // Whole byte allowed — no constraint.
      continue;
    }
    if i == n_full_bytes && partial_bits > 0 {
      // Boundary byte: only the lower `partial_bits` bits may be set.
      let mask = (1u8 << partial_bits) - 1;
      if (*b & !mask) != 0 {
        return Err("range_check_native: value exceeds 2^n_bits in boundary byte");
      }
      continue;
    }
    // Beyond boundary — must be zero.
    if *b != 0 {
      return Err("range_check_native: value exceeds 2^n_bits in high bytes");
    }
  }

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::Bn256EngineKZG;
  use crate::traits::Engine;
  use ff::Field;
  type Scalar = <Bn256EngineKZG as Engine>::Scalar;

  #[test]
  fn zero_in_range_for_any_n_bits() {
    for n in 0..=64 {
      assert!(range_check_native::<Scalar>(Scalar::ZERO, n).is_ok());
    }
  }

  #[test]
  fn exact_boundary_rejected() {
    // 2^8 must be rejected for n_bits=8.
    assert!(range_check_native::<Scalar>(Scalar::from(1u64 << 8), 8).is_err());
    // 2^64 - 1 in range for n_bits=64.
    assert!(range_check_native::<Scalar>(Scalar::from(u64::MAX), 64).is_ok());
    // 2^64 rejected for n_bits=64.
    let too_big = Scalar::from(u64::MAX) + Scalar::from(1u64);
    assert!(range_check_native::<Scalar>(too_big, 64).is_err());
  }

  #[test]
  fn small_values_in_range() {
    for v in 0u64..256 {
      assert!(range_check_native::<Scalar>(Scalar::from(v), 8).is_ok(),
        "value {v} should be in [0, 2^8)");
    }
  }

  #[test]
  fn n_bits_zero() {
    assert!(range_check_native::<Scalar>(Scalar::ZERO, 0).is_ok());
    assert!(range_check_native::<Scalar>(Scalar::from(1u64), 0).is_err());
  }

  #[test]
  fn partial_boundary_byte() {
    // n_bits = 5 → max value is 31 (0b11111).
    for v in 0..32u64 {
      assert!(range_check_native::<Scalar>(Scalar::from(v), 5).is_ok());
    }
    for v in 32..64u64 {
      assert!(range_check_native::<Scalar>(Scalar::from(v), 5).is_err());
    }
  }
}
