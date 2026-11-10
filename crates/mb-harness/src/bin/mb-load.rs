//! `mb-load --addr HOST:PORT --rates 10,20,... [--seconds S] [--warmup S]
//!  [--seed N] [--senders N] [--budget-ms MS] [--gate-logit X --gate-share Y]
//!  [--prompt-ids TSV] [--reference NPY]`
//!
//! Rung 21's load test (`load.rs`). Prints one JSON line per arrival rate, in
//! rising order, then a summary line with the highest rate whose p99 latency
//! met the budget. It stops after two rates in a row over the budget, since
//! higher rates only queue more. Exits 1 if any reply broke the logit bound, or the
//! share of changed predictions over the whole test is past the gate's.

use mb_harness::bench::read_examples;
use mb_harness::integrity::Gate;
use mb_harness::load::{run_rate, LoadConfig};
use mb_harness::rng::Rng;
use serde_json::json;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|f| f == name).and_then(|i| args.get(i + 1));
    let number = |name: &str, default: f64| -> f64 {
        flag(name).map_or(default, |v| {
            v.parse().unwrap_or_else(|_| fail(&format!("{name} {v} is not a number")))
        })
    };
    let addr = flag("--addr").cloned().unwrap_or_else(|| "127.0.0.1:8080".to_string());
    let rates: Vec<f64> = flag("--rates")
        .unwrap_or_else(|| fail("--rates is required, for example 10,20,30"))
        .split(',')
        .map(|r| r.parse().unwrap_or_else(|_| fail(&format!("rate {r} is not a number"))))
        .collect();
    let budget_ms = number("--budget-ms", 200.0);
    let seed = number("--seed", 21000.0) as u64;
    let gate = Gate {
        max_logit_abs: number("--gate-logit", 1e-2) as f32,
        max_disagreement_rate: number("--gate-share", 0.0),
    };
    let ids = flag("--prompt-ids").map_or("data/agnews_test_prompt_ids.tsv", |s| s.as_str());
    let reference = flag("--reference").map_or("results/rung0_logits.npy", |s| s.as_str());
    let examples = read_examples(Path::new(ids)).unwrap_or_else(|e| fail(&e));
    let reference = mb_harness::npy::read(Path::new(reference)).unwrap_or_else(|e| fail(&e));

    // One seeded shuffle of every example; each rate starts where the last stopped.
    let mut order: Vec<usize> = (0..examples.len()).collect();
    Rng::new(seed).shuffle(&mut order);

    let mut best: Option<f64> = None;
    let mut over = 0;
    let mut valid = true;
    let (mut checked, mut changed) = (0u64, 0u64);
    for (k, &rate) in rates.iter().enumerate() {
        let config = LoadConfig {
            addr: addr.clone(),
            rate,
            warmup_seconds: number("--warmup", 5.0),
            seconds: number("--seconds", 40.0),
            seed: seed + 1 + k as u64,
            senders: number("--senders", 64.0) as usize,
            gate,
        };
        let shift = (k * 7919) % order.len();
        let rotated: Vec<usize> = order[shift..].iter().chain(&order[..shift]).copied().collect();
        let record = run_rate(&config, &examples, &reference, &rotated).unwrap_or_else(|e| fail(&e));
        println!("{record}");
        valid &= record["integrity"]["within_logit_bound"].as_bool() == Some(true);
        checked += record["integrity"]["checked"].as_u64().unwrap_or(0);
        changed += record["integrity"]["changed_predictions"].as_u64().unwrap_or(0);
        let p99 = record["latency_ms"]["p99"].as_f64().unwrap_or(f64::INFINITY);
        if p99 <= budget_ms {
            best = Some(best.map_or(rate, |b: f64| b.max(rate)));
            over = 0;
        } else {
            over += 1;
            if over == 2 {
                break;
            }
        }
    }
    // The share of changed predictions, over every reply of the whole test.
    let share = changed as f64 / checked.max(1) as f64;
    valid &= share <= gate.max_disagreement_rate;
    println!("{}", json!({ "summary": true, "budget_ms": budget_ms,
                           "best_rate_within_budget": best, "checked": checked,
                           "changed_predictions": changed, "changed_share": share,
                           "valid": valid }));
    if !valid {
        std::process::exit(1);
    }
}

fn fail(message: &str) -> ! {
    eprintln!("mb-load: {message}");
    std::process::exit(2);
}
