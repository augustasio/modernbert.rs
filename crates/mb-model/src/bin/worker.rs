//! The benchmark worker for rungs 6, 7 and 8. See `crates/mb-harness/PROTOCOL.md`.
//!
//! One binary serves all three. Built normally it is rung 6, the naive loop;
//! built with `--features accelerate` it is rung 7; with `--features fused`
//! it is rung 8, the fused forward pass. It names itself from the backend it
//! was compiled with, so the harness records which one it ran.

use std::io::{BufRead, Write};
use std::time::Instant;

use mb_model::fused::Workspace;
use mb_model::model::ModernBert;
use mb_model::ops::backend;
use serde_json::{json, Value};

/// The four verbalizers, in rung 0's order: Business, World, Sports, Tech.
const VERBALIZER_IDS: [u32; 4] = [10518, 3645, 15000, 18447];

fn main() {
    let weights = std::env::var("MB_WEIGHTS").unwrap_or_else(|_| fail("MB_WEIGHTS is not set"));
    let threads: u64 = std::env::var("MB_THREADS")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| fail("MB_THREADS is not set to a number"));

    let start = Instant::now();
    let model = ModernBert::load(&weights).unwrap_or_else(|e| fail(&format!("load: {e}")));
    let load_seconds = start.elapsed().as_secs_f64();

    let variant = match backend() {
        "accelerate-fused" => "rung8",
        "accelerate" => "rung7",
        _ => "rung6",
    };
    // Rung 8's buffers. Built once here, outside any timed call, and sized
    // past the longest AG News prompt so no request grows them.
    let mut workspace = Workspace::new(model.config(), 512);
    let mut out = std::io::stdout().lock();
    send(
        &mut out,
        &json!({
            "ready": true,
            "variant": variant,
            "backend": backend(),
            "threads": threads,
            "load_seconds": load_seconds,
        }),
    );

    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap_or_else(|e| fail(&format!("stdin: {e}")));
        let request: Value =
            serde_json::from_str(&line).unwrap_or_else(|e| fail(&format!("bad request: {e}")));

        match request["cmd"].as_str() {
            Some("run") => {
                let example = request["example"]
                    .as_u64()
                    .unwrap_or_else(|| fail("no example"));
                let mask = request["mask"].as_u64().unwrap_or_else(|| fail("no mask")) as usize;
                let ids: Vec<u32> = request["ids"]
                    .as_array()
                    .unwrap_or_else(|| fail("no ids"))
                    .iter()
                    .map(|v| v.as_u64().unwrap_or_else(|| fail("a non-integer id")) as u32)
                    .collect();

                // The forward pass alone. Parsing and replying are outside it.
                let t = Instant::now();
                let logits = forward(&model, &mut workspace, &ids, mask)
                    .unwrap_or_else(|e| fail(&format!("example {example}: {e}")));
                let seconds = t.elapsed().as_secs_f64();

                send(
                    &mut out,
                    &json!({ "example": example, "seconds": seconds, "logits": logits }),
                );
            }
            Some("quit") => return,
            other => fail(&format!("unknown command {other:?}")),
        }
    }
}

/// The four verbalizer logits, through rung 8's fused path when this build is
/// rung 8 and through the plain `logits_for_tokens` otherwise.
fn forward(
    model: &ModernBert,
    workspace: &mut Workspace,
    ids: &[u32],
    mask: usize,
) -> Result<[f32; 4], mb_model::safetensors::TensorError> {
    let mut logits = [0.0f32; 4];
    if cfg!(feature = "fused") {
        model.logits_for_tokens_fused(ids, mask, &VERBALIZER_IDS, workspace, &mut logits)?;
    } else {
        logits.copy_from_slice(&model.logits_for_tokens(ids, mask, &VERBALIZER_IDS)?);
    }
    Ok(logits)
}

fn send(out: &mut impl Write, message: &Value) {
    writeln!(out, "{message}").unwrap_or_else(|e| fail(&format!("stdout: {e}")));
    out.flush()
        .unwrap_or_else(|e| fail(&format!("stdout: {e}")));
}

/// Report an error the way the protocol asks, then exit non-zero.
fn fail(message: &str) -> ! {
    println!("{}", json!({ "error": message }));
    eprintln!("worker: {message}");
    std::process::exit(1);
}
