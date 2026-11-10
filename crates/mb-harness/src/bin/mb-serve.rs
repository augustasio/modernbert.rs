//! `mb-serve --workers N --threads T [--port P] [--max-batch B] [--max-wait-ms W]
//!  [--smt] --weights DIR -- <worker command...>`: rung 21's HTTP server. See `serve.rs`.

use mb_harness::serve::{serve, ServeConfig};
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let split = args
        .iter()
        .position(|a| a == "--")
        .unwrap_or_else(|| usage(&args[0]));
    let (flags, command) = (&args[1..split], args[split + 1..].to_vec());
    let flag = |name: &str| {
        flags
            .iter()
            .position(|f| f == name)
            .and_then(|i| flags.get(i + 1))
    };
    let number = |name: &str, default: Option<f64>| -> f64 {
        match flag(name) {
            Some(v) => v
                .parse()
                .unwrap_or_else(|_| fail(&format!("{name} {v} is not a number"))),
            None => default.unwrap_or_else(|| fail(&format!("{name} is required"))),
        }
    };
    let config = ServeConfig {
        port: number("--port", Some(8080.0)) as u16,
        workers: number("--workers", None) as usize,
        threads: number("--threads", None) as usize,
        max_batch: number("--max-batch", Some(1.0)) as usize,
        max_wait: Duration::from_secs_f64(number("--max-wait-ms", Some(0.0)) / 1e3),
        weights: flag("--weights")
            .cloned()
            .unwrap_or_else(|| fail("--weights is required")),
        command,
        smt: flags.iter().any(|f| f == "--smt"),
    };
    if config.command.is_empty() {
        usage(&args[0]);
    }
    if let Err(e) = serve(&config) {
        fail(&e);
    }
}

fn usage(program: &str) -> ! {
    fail(&format!(
        "usage: {program} --workers N --threads T [--port P] [--max-batch B] [--max-wait-ms W] [--smt] \
         --weights DIR -- <worker command...>"
    ))
}

fn fail(message: &str) -> ! {
    eprintln!("mb-serve: {message}");
    std::process::exit(2);
}
