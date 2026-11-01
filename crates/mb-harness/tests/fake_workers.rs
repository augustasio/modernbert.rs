//! The harness's rules, tested end to end with fake workers and no model.
//!
//! Each fake is a few lines of the system `python3` speaking the worker
//! protocol. The reference logits are the same for every example, so a fake
//! that echoes them is correct and one that does not is wrong. The thermal
//! reader is faked too, so its gate can be tested.

use std::path::{Path, PathBuf};

use mb_harness::bench::{run, Config};
use serde_json::{json, Value};

const REFERENCE: [f32; 4] = [1.0, 3.0, 2.0, 0.5];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mb-harness-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Six examples, each with exactly one [MASK].
fn write_inputs(dir: &Path) -> (PathBuf, PathBuf) {
    let ids = dir.join("ids.tsv");
    let mut text = String::new();
    for i in 0..6 {
        text.push_str(&format!(
            "{i}\tWorld\t50281 100 {} 50285 50284 50282\n",
            200 + i
        ));
    }
    std::fs::write(&ids, text).expect("ids");

    let npy = dir.join("ref.npy");
    let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (6, 4), }";
    let padding = (64 - (10 + dict.len() + 1) % 64) % 64;
    let header = format!("{dict}{}\n", " ".repeat(padding));
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for _ in 0..6 {
        for v in REFERENCE {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(&npy, bytes).expect("npy");
    (ids, npy)
}

/// A protocol-speaking fake. `answer` is a Python expression for the logits.
fn fake_worker(dir: &Path, name: &str, answer: &str, fail_on_run: bool) -> Vec<String> {
    let script = dir.join(format!("{name}.py"));
    let body = format!(
        r#"import json, sys
print(json.dumps({{"ready": True, "variant": "{name}", "backend": "fake", "threads": 1, "load_seconds": 0.0}}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    if r["cmd"] == "quit":
        sys.exit(0)
    if {fail_on_run}:
        print(json.dumps({{"error": "deliberate failure"}}), flush=True)
        sys.exit(1)
    print(json.dumps({{"example": r["example"], "seconds": 0.001, "logits": {answer}}}), flush=True)
"#,
        fail_on_run = if fail_on_run { "True" } else { "False" },
    );
    std::fs::write(&script, body).expect("script");
    vec!["python3".to_string(), script.display().to_string()]
}

fn fake_thermal(dir: &Path, output: &str) -> PathBuf {
    let path = dir.join(format!("thermal-{output}.sh"));
    std::fs::write(&path, format!("#!/bin/sh\necho {output}\n")).expect("script");
    let status = std::process::Command::new("chmod")
        .args(["+x", &path.display().to_string()])
        .status()
        .expect("chmod");
    assert!(status.success());
    path
}

fn config(dir: &Path, ruler: Vec<String>, variant: Vec<String>, thermal: PathBuf) -> Config {
    let (ids, npy) = write_inputs(dir);
    Config::from_json(&json!({
        "label": "test",
        "ruler": { "name": "ruler", "command": ruler },
        "variant": { "name": "variant", "command": variant },
        "threads": 1, "cycles": 4, "examples_per_slice": 3, "warmup_examples": 1, "seed": 9,
        "weights": "unused",
        "prompt_ids": ids.display().to_string(),
        "reference_logits": npy.display().to_string(),
        "thermal_helper": thermal.display().to_string(),
    }))
    .expect("config parses")
}

const CORRECT: &str = "[1.0, 3.0, 2.0, 0.5]";

#[test]
fn a_correct_pair_gives_a_valid_run_with_every_slice_recorded() {
    let dir = scratch("valid");
    let cfg = config(
        &dir,
        fake_worker(&dir, "ruler", CORRECT, false),
        fake_worker(&dir, "variant", CORRECT, false),
        fake_thermal(&dir, "nominal"),
    );
    let result = run(&cfg).expect("the run completes");

    assert_eq!(
        result["valid"],
        json!(true),
        "{}",
        result["invalid_because"]
    );
    // 2 warm-up slices plus 2 per cycle.
    let slices = result["slices"].as_array().expect("slices");
    assert_eq!(slices.len(), 2 + 2 * 4);
    assert_eq!(
        slices.iter().filter(|s| s["warmup"] == json!(true)).count(),
        2
    );
    // 1 warm-up example per worker plus 3 per slice per cycle.
    assert_eq!(result["integrity"]["checked"], json!(2 + 2 * 4 * 3));

    for s in slices.iter().filter(|s| s["warmup"] == json!(false)) {
        assert_eq!(s["worker_seconds"].as_array().map(Vec::len), Some(3));
        assert_eq!(s["thermal_before"], json!("nominal"));
    }

    // In each cycle the two slices ran the same examples: a paired design.
    for c in 0..4 {
        let pair: Vec<&Value> = slices.iter().filter(|s| s["cycle"] == json!(c)).collect();
        assert_eq!(pair.len(), 2);
        assert_eq!(pair[0]["examples"], pair[1]["examples"], "cycle {c}");
        assert_ne!(pair[0]["role"], pair[1]["role"], "cycle {c}");
    }
}

#[test]
fn a_variant_that_computes_something_else_makes_the_run_invalid() {
    let dir = scratch("wrong");
    let cfg = config(
        &dir,
        fake_worker(&dir, "ruler", CORRECT, false),
        // Fast and wrong: the prediction moves from index 1 to index 0.
        fake_worker(&dir, "variant", "[9.0, 3.0, 2.0, 0.5]", false),
        fake_thermal(&dir, "nominal"),
    );
    let result = run(&cfg).expect("the run completes, so the evidence is kept");

    assert_eq!(result["valid"], json!(false));
    let failures = result["integrity"]["failures"]
        .as_array()
        .expect("failures");
    // Every variant reply is wrong: 1 warm-up plus 4 cycles of 3.
    assert_eq!(failures.len(), 1 + 4 * 3);
    assert!(failures.iter().all(|f| f["worker"] == json!("variant")));
    assert!(failures.iter().all(|f| f["argmax_agrees"] == json!(false)));
}

#[test]
fn a_small_logit_drift_with_the_same_prediction_still_fails_past_the_tolerance() {
    let dir = scratch("drift");
    let cfg = config(
        &dir,
        fake_worker(&dir, "ruler", CORRECT, false),
        fake_worker(&dir, "variant", "[1.0, 3.01, 2.0, 0.5]", false),
        fake_thermal(&dir, "nominal"),
    );
    let result = run(&cfg).expect("the run completes");
    assert_eq!(
        result["valid"],
        json!(false),
        "a 1e-2 drift is ten times the tolerance"
    );
}

#[test]
fn a_worker_error_stops_the_run_with_its_message() {
    let dir = scratch("error");
    let cfg = config(
        &dir,
        fake_worker(&dir, "ruler", CORRECT, false),
        fake_worker(&dir, "variant", CORRECT, true),
        fake_thermal(&dir, "nominal"),
    );
    let err = run(&cfg).expect_err("a worker error must stop the run");
    assert!(err.contains("deliberate failure"), "got {err}");
}

#[test]
fn an_unreadable_thermal_state_stops_the_run_before_anything_loads() {
    let dir = scratch("thermal");
    let cfg = config(
        &dir,
        fake_worker(&dir, "ruler", CORRECT, false),
        fake_worker(&dir, "variant", CORRECT, false),
        fake_thermal(&dir, "lukewarm"),
    );
    let err = run(&cfg).expect_err("a bad thermal reading must stop the run");
    assert!(err.contains("lukewarm"), "got {err}");
}
