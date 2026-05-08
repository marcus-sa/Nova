//! Per-position shape-registry assertion (GH-#2 M.7 / pin §3 / spike-
//! conclusion auditor obligation #4).
//!
//! ADR-0021's chunked Strauss-Shamir produces 16 main-loop
//! `Structure<E>` instances (plus ~9 chunk-band shapes; ~25-30 total
//! at production arity). The augmented circuit per IVC step must
//! assert `pp_digest_in == registry[chunk_index_in_z]` to close the
//! cross-position witness substitution attack (pin §3.4 / SuperNova
//! eprint 2022/1758 v3 §3 step-kind discipline).
//!
//! M.7 ships the in-circuit (R1CS-emitting) half:
//!
//! - [`circuit::assert_pp_digest_matches_registry`] — range-check
//!   (pin §3.2) + constant-time conditional-select chain + equality
//!   assert. Wired into
//!   [`crate::neutron::circuit::nifs::AllocatedNIFS::verify_with_multi_table_lookup`]
//!   at the surface specified by pin §3.3 (BEFORE
//!   `pp_digest.absorb(ro)`).
//!
//! M.7.5 adds the off-circuit reference and the differential coverage
//! that satisfies the Gadget contract from
//! `.claude/rules/cryptography.md`.
#![cfg(feature = "lookup-fold")]

pub mod circuit;

pub use circuit::assert_pp_digest_matches_registry;
