//! Benchmark harness.
//!
//! This exists as a crate with its own tests, rather than a script, because the M4 Air
//! is fanless and throttles: a naive "run A, then run B" comparison measures thermal
//! state as much as it measures code.
//!
//! The protocol: each run pairs one variant with the ruler, alternates them in slices of
//! the same examples with the order randomized per cycle, records the thermal state
//! around every slice, and checks every logit against rung 0 in the timed run itself.
//! Workers speak the protocol in `PROTOCOL.md`. The harness records raw samples only;
//! the statistics live in `python/`.
//!
//! See `PLAN.md` for the full protocol and its changelog.

pub mod bench;
pub mod integrity;
pub mod npy;
pub mod power;
pub mod rng;
pub mod schedule;
pub mod thermal;
pub mod worker;
