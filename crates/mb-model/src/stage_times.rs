//! Analysis only, under the `stage-times` feature: wall time per stage of the
//! threaded layer, summed over a run, so a profile can say where the time
//! outside oneDNN's matmuls goes. Never part of a benchmarked rung: a timed
//! build reads the clock about twenty times a layer.
//!
//! A matmul stage here is the whole `linear_into` call, so it holds the int8
//! quantize and dequantize loops as well as oneDNN's own time. oneDNN's
//! verbose log gives the second, and the difference is the first.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The stages, in the order a layer runs them.
pub const STAGES: [&str; 10] = [
    "attn_norm",
    "wqkv",
    "attention",
    "gather_heads",
    "attn_wo",
    "add_mlp_norm",
    "wi",
    "gelu_gate",
    "mlp_wo",
    "add",
];

static NANOS: [AtomicU64; STAGES.len()] = [const { AtomicU64::new(0) }; STAGES.len()];
static CALLS: [AtomicU64; STAGES.len()] = [const { AtomicU64::new(0) }; STAGES.len()];

/// Run `work` and add its wall time to stage `index` of [`STAGES`].
pub fn timed<T>(index: usize, work: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = work();
    NANOS[index].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    CALLS[index].fetch_add(1, Ordering::Relaxed);
    out
}

/// One line per stage: total milliseconds per example and calls per example.
pub fn report(examples: usize) -> String {
    let per = examples.max(1) as f64;
    let mut out = String::new();
    let mut total = 0.0;
    for (i, name) in STAGES.iter().enumerate() {
        let ms = NANOS[i].load(Ordering::Relaxed) as f64 / 1e6 / per;
        total += ms;
        let calls = CALLS[i].load(Ordering::Relaxed) as f64 / per;
        out.push_str(&format!(
            "stage {name:13} {ms:7.2} ms/example  {calls:4.0} calls\n"
        ));
    }
    out.push_str(&format!(
        "stage {:13} {total:7.2} ms/example\n",
        "layers_total"
    ));
    out
}
