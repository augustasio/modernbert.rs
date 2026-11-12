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

/// The stages, in the order a layer runs them. The `q8_` lines are inside the
/// four matmul stages (int8 builds only) and the `att_` lines inside
/// attention, and the `gate_` lines inside `gelu_gate` (the GELU gate's
/// arithmetic, then the rounding of each gated row for the MLP's Wo), so the
/// total leaves them out. The `att_` and `gate_` lines run on every
/// thread at once and sum over the threads: divide by the thread count for
/// wall time.
pub const STAGES: [&str; 19] = [
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
    "q8_quantize",
    "q8_onednn",
    "q8_dequantize",
    "att_scatter",
    "att_scores",
    "att_softmax",
    "att_pv",
    "gate_math",
    "gate_quant",
];

/// How many of [`STAGES`] make up a layer; the rest are inside them.
const LAYER_STAGES: usize = 10;

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
        if i < LAYER_STAGES {
            total += ms;
        }
        let calls = CALLS[i].load(Ordering::Relaxed) as f64 / per;
        out.push_str(&format!(
            "stage {name:13} {ms:7.2} ms/example  {calls:4.0} calls\n"
        ));
    }
    out.push_str(&format!(
        "stage {:13} {total:7.2} ms/example (without the q8_ and att_ lines)\n",
        "layers_total"
    ));
    out
}
