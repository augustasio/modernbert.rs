//! Benchmark harness.
//!
//! This exists as a crate with its own tests, rather than a script, because the M4 Air
//! is fanless and throttles: a naive "run A, then run B" comparison measures thermal
//! state as much as it measures code.
//!
//! The protocol: round-robin variants in ~10-second slices with the order randomized
//! within each cycle, re-measure a ruler variant every cycle and report other variants
//! as ratios to it, and capture `CPU_Speed_Limit` alongside every sample. Burst and
//! sustained numbers are reported separately. Memory is a second pass, in fresh
//! processes, because allocator high-water marks never shrink.
//!
//! See `PLAN.md` for the full protocol and the tier-4 integrity gates this enforces.
