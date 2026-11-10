//! Rung 21's load test: open-loop Poisson arrivals against the server.
//!
//! Open loop means a request goes out at its scheduled time whether or not
//! earlier ones have been answered, as real users arrive. A closed loop (send
//! the next request when the last reply comes back) slows itself down exactly
//! when the server falls behind, and so hides the queue the budget must count.
//!
//! Trap (coordinated omission): latency runs from the request's *scheduled*
//! arrival, not from when a sender thread got round to it. If every sender is
//! busy, the request's wait still counts against the server's budget, and
//! `late_send_ms` reports how far behind the client itself fell, so a run
//! limited by the client shows it.
//!
//! Every reply is checked against rung 0's logits (hard rule 5), with the same
//! gate as the paired harness.

use crate::bench::Example;
use crate::http::{read_message, write_message};
use crate::integrity::{check, Gate};
use crate::npy::Array2;
use crate::rng::Rng;
use serde_json::{json, Value};
use std::io::BufReader;
use std::net::TcpStream;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

pub struct LoadConfig {
    pub addr: String,
    pub rate: f64,
    pub warmup_seconds: f64,
    pub seconds: f64,
    pub seed: u64,
    pub senders: usize,
    pub gate: Gate,
}

/// What one request saw.
struct Sample {
    example: usize,
    scheduled: f64,
    late_send: f64,
    latency: f64,
    queue_seconds: f64,
    worker_seconds: f64,
    batch: usize,
    logits: Vec<f32>,
}

/// The arrival times, in seconds from the start, of a Poisson process at
/// `rate` per second up to `until`: exponential gaps of mean 1 / rate.
pub fn poisson_arrivals(rate: f64, until: f64, rng: &mut Rng) -> Vec<f64> {
    let mut times = Vec::new();
    let mut t = 0.0;
    loop {
        // A uniform draw in (0, 1]: 53 random bits, never 0, so ln is finite.
        let u = ((rng.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64;
        t += -u.ln() / rate;
        if t >= until {
            return times;
        }
        times.push(t);
    }
}

/// Run one arrival rate and return its record. `order` is the example order;
/// request i uses example `order[i % len]`.
pub fn run_rate(
    config: &LoadConfig,
    examples: &[Example],
    reference: &Array2,
    order: &[usize],
) -> Result<Value, String> {
    let mut rng = Rng::new(config.seed);
    let arrivals = poisson_arrivals(config.rate, config.warmup_seconds + config.seconds, &mut rng);

    let (send, receive) = mpsc::channel::<(usize, f64)>();
    let receive = Arc::new(Mutex::new(receive));
    let samples = Arc::new(Mutex::new(Vec::<Sample>::new()));
    let start = Instant::now() + Duration::from_millis(200);
    let bodies: Arc<Vec<Vec<u8>>> = Arc::new(
        examples
            .iter()
            .map(|e| json!({ "ids": e.ids }).to_string().into_bytes())
            .collect(),
    );

    let mut senders = Vec::new();
    for _ in 0..config.senders {
        let stream = TcpStream::connect(&config.addr)
            .map_err(|e| format!("cannot connect to {}: {e}", config.addr))?;
        stream.set_nodelay(true).map_err(|e| format!("nodelay: {e}"))?;
        let (receive, samples, bodies) = (Arc::clone(&receive), Arc::clone(&samples), Arc::clone(&bodies));
        senders.push(std::thread::spawn(move || -> Result<(), String> {
            let mut writer = stream.try_clone().map_err(|e| format!("clone: {e}"))?;
            let mut reader = BufReader::new(stream);
            loop {
                let next = receive.lock().expect("receiver lock").recv();
                let Ok((example, scheduled)) = next else { return Ok(()) };
                let sent = start.elapsed().as_secs_f64();
                write_message(&mut writer, "POST /classify HTTP/1.1", &bodies[example])?;
                let reply = read_message(&mut reader)?.ok_or("the server closed the connection")?;
                let latency = start.elapsed().as_secs_f64() - scheduled;
                let body: Value = serde_json::from_slice(&reply.body)
                    .map_err(|e| format!("reply is not JSON: {e}"))?;
                if !reply.start_line.contains(" 200 ") {
                    return Err(format!("{}: {body}", reply.start_line));
                }
                let logits = body["logits"]
                    .as_array()
                    .ok_or_else(|| format!("no logits in {body}"))?
                    .iter()
                    .map(|v| v.as_f64().map(|x| x as f32))
                    .collect::<Option<Vec<f32>>>()
                    .ok_or("a logit is not a number")?;
                samples.lock().expect("samples lock").push(Sample {
                    example,
                    scheduled,
                    late_send: sent - scheduled,
                    latency,
                    queue_seconds: body["queue_seconds"].as_f64().unwrap_or(f64::NAN),
                    worker_seconds: body["worker_seconds"].as_f64().unwrap_or(f64::NAN),
                    batch: body["batch"].as_u64().unwrap_or(0) as usize,
                    logits,
                });
            }
        }));
    }

    // The schedule: sleep until each arrival, then hand it to a free sender.
    for (i, &t) in arrivals.iter().enumerate() {
        let due = start + Duration::from_secs_f64(t);
        let now = Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        }
        send.send((order[i % order.len()], t))
            .map_err(|_| "every sender stopped".to_string())?;
    }
    drop(send);
    for s in senders {
        s.join().map_err(|_| "a sender panicked".to_string())??;
    }

    let samples = std::mem::take(&mut *samples.lock().expect("samples lock"));
    summarize(config, &samples, reference)
}

/// The record of one rate: percentiles only (hard rule 4), and the check.
fn summarize(config: &LoadConfig, samples: &[Sample], reference: &Array2) -> Result<Value, String> {
    let mut changed = 0usize;
    let mut worst = 0.0f32;
    let mut failures = Vec::new();
    for s in samples {
        let c = check(reference.row(s.example), &s.logits);
        worst = worst.max(c.max_abs);
        changed += usize::from(!c.argmax_agrees);
        if !c.within(&config.gate) {
            failures.push(json!({ "example": s.example, "max_abs": c.max_abs }));
        }
    }
    // Changed predictions are judged against the gate's share over the whole
    // load test, not per rate: one rate sees about 1,500 requests, where a
    // single known close call is already past a share set from 7,600.
    let valid = failures.is_empty();

    let measured: Vec<&Sample> = samples
        .iter()
        .filter(|s| s.scheduled >= config.warmup_seconds)
        .collect();
    let ms = |f: &dyn Fn(&Sample) -> f64| -> Value {
        let mut v: Vec<f64> = measured.iter().map(|s| f(s) * 1e3).collect();
        v.sort_by(f64::total_cmp);
        json!({ "p50": percentile(&v, 0.50), "p90": percentile(&v, 0.90),
                "p99": percentile(&v, 0.99), "max": v.last().copied() })
    };
    let mut batches = std::collections::BTreeMap::<usize, usize>::new();
    for s in &measured {
        *batches.entry(s.batch).or_default() += 1;
    }
    Ok(json!({
        "rate_per_s": config.rate,
        "seconds": config.seconds,
        "requests": measured.len(),
        "served_per_s": measured.len() as f64 / config.seconds,
        "latency_ms": ms(&|s| s.latency),
        "queue_ms": ms(&|s| s.queue_seconds),
        "worker_ms": ms(&|s| s.worker_seconds),
        "late_send_ms": ms(&|s| s.late_send),
        "requests_by_batch_size": batches,
        "integrity": { "checked": samples.len(), "changed_predictions": changed,
                       "worst_max_abs": worst, "failures": failures, "within_logit_bound": valid },
        "load_average": std::fs::read_to_string("/proc/loadavg").ok()
            .and_then(|l| l.split(' ').next().map(str::to_string)),
    }))
}

/// The nearest-rank percentile of sorted values: the smallest value with at
/// least `q` of the values at or below it.
pub fn percentile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, sorted.len()) - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_arrivals_have_the_asked_rate() {
        let mut rng = Rng::new(7);
        let times = poisson_arrivals(50.0, 200.0, &mut rng);
        let rate = times.len() as f64 / 200.0;
        assert!((rate - 50.0).abs() < 1.5, "rate {rate}");
        assert!(times.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn nearest_rank_percentiles() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 0.99), Some(99.0));
        assert_eq!(percentile(&v, 0.50), Some(50.0));
        assert_eq!(percentile(&[3.0], 0.99), Some(3.0));
        assert_eq!(percentile(&[], 0.5), None);
    }
}
