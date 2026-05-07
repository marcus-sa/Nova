//! LogUp denominator-cleared inversions, shared between PPSNARK's row-half
//! `MemorySumcheckInstance` and the C1-β lookup-fold extension's
//! `LookupSumcheckInstance`.
//!
//! Soundness-pin-neutral; mirrors Lasso 2023/1216 v3 §6.2 + Haböck eprint
//! 2022/1530 (multivariate logarithmic derivatives).
//!
//! Pinned by §B.7 of the C1-β spike implementation outline:
//! `docs/research/cryptography/c1-beta-spike-implementation-outline-stages-b-f.md`.
//!
//! At spike scope, only `batch_invert_plus_r` is exposed — the single-side
//! inverse-witness builder used by `LookupSumcheckInstance::new`. The full
//! `batch_invert_logup` extraction (which would supersede the
//! `MemorySumcheckInstance::compute_oracles` inner closure at
//! `vendor/nova/src/spartan/ppsnark.rs:411-444`) is deferred — that migration
//! is independent of the spike measurement gate and the row-half differential
//! test would be the regression check for it. Doing it now would risk a
//! soundness-relevant divergence between PPSNARK's helper and the new
//! function under test.

// Stage 1.B lands `batch_invert_plus_r` in isolation; the in-crate consumer
// (`LookupSumcheckInstance::new`) is partially wired (Stage 1.B test
// fixtures invoke it, but the production wiring through `NIFS::prove`
// arrives in Stages C-E). Until the production wiring lands, the helper
// would otherwise trip `#[deny(unused)]`. The `cfg(test)` test module
// below exercises it directly to keep coverage attached.
#![allow(dead_code)]

use crate::{errors::NovaError, spartan::batch_invert};
use ff::PrimeField;
use rayon::prelude::*;

/// Compute `1/(values[i] + r)` for every entry, in batch, via Montgomery's
/// trick (re-using `spartan::batch_invert`).
///
/// Returns an error if any `values[i] + r` is zero (which would make the
/// LogUp identity ill-defined for the offending entry; in practice the
/// caller should have ensured `r` is squeezed AFTER all `values` are
/// committed, so a colliding `r` is cryptographically negligible).
///
/// Soundness: this is the LogUp denominator-clearing primitive (Lasso
/// 2023/1216 v3 §6.2). Mirrors the inner `helper` closure in
/// `MemorySumcheckInstance::compute_oracles` at
/// `vendor/nova/src/spartan/ppsnark.rs:411-444`, specialised to the
/// single-side (no `ts` multiplication) case used by the lookup-fold
/// extension.
pub(crate) fn batch_invert_plus_r<Scalar: PrimeField>(
  values: &[Scalar],
  r: &Scalar,
) -> Result<Vec<Scalar>, NovaError> {
  let plus_r: Vec<Scalar> = values.par_iter().map(|v| *v + *r).collect();
  batch_invert(&plus_r)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::Bn256EngineKZG;
  use crate::traits::Engine;
  use ff::Field;
  use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

  type Scalar = <Bn256EngineKZG as Engine>::Scalar;

  #[test]
  fn batch_invert_plus_r_round_trips() {
    // Seed-suffix taxonomy: 0xC1BE_xxxx; INV = 0x10F0.
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_10F0);
    let r = Scalar::random(&mut rng);
    let values: Vec<Scalar> = (0..1024).map(|_| Scalar::random(&mut rng)).collect();
    let inverses = batch_invert_plus_r(&values, &r).unwrap();
    assert_eq!(inverses.len(), values.len());
    for (v, inv) in values.iter().zip(inverses.iter()) {
      assert_eq!((*v + r) * inv, Scalar::ONE);
    }
  }

  #[test]
  fn batch_invert_plus_r_rejects_zero_denominator() {
    // Choose r = -values[0] so values[0] + r = 0.
    // INV_ZERO = 0x10F1.
    let mut rng = ChaCha20Rng::seed_from_u64(0xC1BE_10F1);
    let v0 = Scalar::random(&mut rng);
    let r = -v0;
    let values = vec![v0, Scalar::random(&mut rng)];
    assert!(batch_invert_plus_r(&values, &r).is_err());
  }
}
