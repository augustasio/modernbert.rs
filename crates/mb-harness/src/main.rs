//! `bench <config.json> <out.json>`: run one benchmark and write its result.
//!
//! The exit status is 0 only for a valid run. A run that broke a rule still
//! writes its result file, with `valid: false` and the reasons, and exits 1.

use mb_harness::bench::{run, Config};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <config.json> <out.json>", args[0]);
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {e}", args[1]);
        std::process::exit(2);
    });
    let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("{} is not JSON: {e}", args[1]);
        std::process::exit(2);
    });
    let config = Config::from_json(&json).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });

    let result = run(&config).unwrap_or_else(|e| {
        eprintln!("the run could not complete: {e}");
        std::process::exit(1);
    });

    let out = std::path::Path::new(&args[2]);
    if let Some(dir) = out.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let pretty = serde_json::to_string_pretty(&result).expect("a JSON value serializes");
    std::fs::write(out, pretty + "\n").unwrap_or_else(|e| {
        eprintln!("cannot write {}: {e}", out.display());
        std::process::exit(1);
    });

    let valid = result["valid"].as_bool() == Some(true);
    eprintln!(
        "wrote {}. valid: {valid}. checked {} replies, worst logit difference {}",
        out.display(),
        result["integrity"]["checked"],
        result["integrity"]["worst_max_abs"]
    );
    if !valid {
        eprintln!("invalid because: {}", result["invalid_because"]);
        std::process::exit(1);
    }
}
