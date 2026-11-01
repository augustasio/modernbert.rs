//! The benchmark worker for rungs 6 and 7. See `crates/mb-harness/PROTOCOL.md`.
//!
//! One binary serves both rungs. Built normally it is rung 6, the naive loop;
//! built with `--features accelerate` it is rung 7. It names itself from the
//! backend it was compiled with, so the harness records which one it ran.

use std::io::{BufRead, Write};
use std::time::Instant;

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

    let variant = if backend() == "accelerate" {
        "rung7"
    } else {
        "rung6"
    };
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
                let logits = model
                    .logits_for_tokens(&ids, mask, &VERBALIZER_IDS)
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
