//! This library implements Nova, a high-speed recursive SNARK.
#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![forbid(unsafe_code)]
#![cfg_attr(not(test), warn(clippy::print_stdout, clippy::print_stderr))]

// main APIs exposed by this library
pub mod nova;

#[cfg(feature = "experimental")]
pub mod neutron;

/// User-callable lookup gadgets and the `LookupConstraintSystem` collector
/// trait. Stage H of the C1-β lookup-fold extension.
#[cfg(feature = "lookup-fold")]
pub mod lookup;

/// Per-position shape-registry assertion gadget (in-circuit + off-circuit
/// pair). GH-#2 M.7 + M.7.5 of the C1-β multi-table lookup-fold
/// extension; closes the cross-position witness substitution attack
/// per the design pin §3.
#[cfg(feature = "lookup-fold")]
pub mod shape_registry;

// public modules
pub mod constants;
pub mod digest;
pub mod errors;
pub mod frontend;
pub mod gadgets;
pub mod provider;
pub mod r1cs;
pub mod spartan;
pub mod traits;

use traits::{commitment::CommitmentEngineTrait, Engine};

// some type aliases
type CommitmentKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey;
type DerandKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::DerandKey;
type Commitment<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment;
type CE<E> = <E as Engine>::CE;
