//! Rung 21: an HTTP server in front of the model workers.
//!
//! `POST /classify` with `{"ids": [...]}` (the prompt's token ids, so every rung
//! still reads byte-identical ids, hard rule 1) returns the verbalizer logits.
//! Requests wait in one queue. Each worker process has one dispatcher thread
//! here, which takes the oldest request (or up to `max_batch` of them), sends
//! it over the worker protocol of `PROTOCOL.md`, and answers the connection.
//!
//! The workers are separate processes, not threads of this one, because
//! libgomp keeps one thread pool per process: N workers of T threads each need
//! their own pool, bound to their own cores. The rough check before this rung
//! found 4 workers of 4 threads beat 1 of 16 by 1.45x within the budget, and
//! that binding threads through `OMP_PLACES` removed 1.5 s stalls that
//! `taskset` caused.

use crate::bench::MASK_ID;
use crate::http::{read_message, write_message};
use crate::worker::Worker;
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub struct ServeConfig {
    pub port: u16,
    pub workers: usize,
    pub threads: usize,
    /// At most this many queued requests go to a worker as one batch.
    pub max_batch: usize,
    /// How long a worker waits for a batch to fill once one request is queued.
    pub max_wait: Duration,
    pub weights: String,
    pub command: Vec<String>,
}

/// One request, from its arrival to its answer.
struct Job {
    ids: Vec<u32>,
    mask: usize,
    arrived: Instant,
    answer: mpsc::Sender<Result<Done, String>>,
}

struct Done {
    logits: Vec<f32>,
    queue_seconds: f64,
    worker_seconds: f64,
    batch: usize,
}

#[derive(Default)]
struct Queue {
    jobs: Mutex<VecDeque<Job>>,
    changed: Condvar,
}

/// Start the workers, then serve until the process is killed.
pub fn serve(config: &ServeConfig) -> Result<(), String> {
    let cores = physical_cores();
    let bind = cores.len() >= config.workers * config.threads;
    let queue = Arc::new(Queue::default());
    let next_example = Arc::new(AtomicUsize::new(0));

    for k in 0..config.workers {
        let mut env = BTreeMap::new();
        let threads = config.threads.to_string();
        env.insert("MB_WEIGHTS".to_string(), config.weights.clone());
        env.insert("MB_THREADS".to_string(), threads.clone());
        env.insert("OMP_NUM_THREADS".to_string(), threads.clone());
        env.insert("MKL_NUM_THREADS".to_string(), threads.clone());
        env.insert("VECLIB_MAXIMUM_THREADS".to_string(), threads);
        env.insert("MKL_THREADING_LAYER".to_string(), "GNU".to_string());
        if bind {
            // One OpenMP thread per physical core, on this worker's cores only.
            let mine = &cores[k * config.threads..(k + 1) * config.threads];
            let places: Vec<String> = mine.iter().map(|c| format!("{{{c}}}")).collect();
            env.insert("OMP_PLACES".to_string(), places.join(","));
            env.insert("OMP_PROC_BIND".to_string(), "close".to_string());
        }
        let worker = Worker::start(&format!("worker {k}"), &config.command, &env)?;
        let queue = Arc::clone(&queue);
        let next_example = Arc::clone(&next_example);
        let (max_batch, max_wait) = (config.max_batch.max(1), config.max_wait);
        std::thread::spawn(move || dispatch(worker, &queue, &next_example, max_batch, max_wait));
    }

    let listener = TcpListener::bind(("0.0.0.0", config.port))
        .map_err(|e| format!("cannot listen on port {}: {e}", config.port))?;
    println!(
        "{}",
        json!({ "ready": true, "port": config.port, "workers": config.workers,
                "threads": config.threads, "max_batch": config.max_batch,
                "max_wait_ms": config.max_wait.as_secs_f64() * 1e3, "bound_to_cores": bind })
    );
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let queue = Arc::clone(&queue);
        std::thread::spawn(move || {
            if let Err(e) = connection(stream, &queue) {
                eprintln!("connection: {e}");
            }
        });
    }
    Ok(())
}

/// One kept-alive client connection: read a request, queue it, wait, answer.
fn connection(stream: TcpStream, queue: &Queue) -> Result<(), String> {
    stream
        .set_nodelay(true)
        .map_err(|e| format!("nodelay: {e}"))?;
    let mut writer = stream.try_clone().map_err(|e| format!("clone: {e}"))?;
    let mut reader = BufReader::new(stream);
    while let Some(request) = read_message(&mut reader)? {
        // The latency the budget counts starts here, so it includes the queue.
        let arrived = Instant::now();
        let (status, body) = match request.start_line.split(' ').take(2).collect::<Vec<_>>()[..] {
            ["POST", "/classify"] => classify(&request.body, arrived, queue),
            ["GET", "/health"] => ("200 OK", json!({ "ok": true })),
            _ => ("404 Not Found", json!({ "error": "unknown path" })),
        };
        write_message(
            &mut writer,
            &format!("HTTP/1.1 {status}"),
            body.to_string().as_bytes(),
        )?;
    }
    Ok(())
}

fn classify(body: &[u8], arrived: Instant, queue: &Queue) -> (&'static str, Value) {
    let ids: Option<Vec<u32>> = serde_json::from_slice::<Value>(body).ok().and_then(|v| {
        v["ids"]
            .as_array()?
            .iter()
            .map(|id| id.as_u64().map(|x| x as u32))
            .collect()
    });
    let Some(ids) = ids else {
        return (
            "400 Bad Request",
            json!({ "error": "expected {\"ids\": [integers]}" }),
        );
    };
    // The last [MASK], as every other rung reads it.
    let Some(mask) = ids.iter().rposition(|&id| id == MASK_ID) else {
        return (
            "400 Bad Request",
            json!({ "error": "the prompt has no [MASK]" }),
        );
    };
    let (answer, wait) = mpsc::channel();
    queue.jobs.lock().expect("queue lock").push_back(Job {
        ids,
        mask,
        arrived,
        answer,
    });
    queue.changed.notify_all();
    match wait.recv() {
        Ok(Ok(done)) => (
            "200 OK",
            json!({ "logits": done.logits, "queue_seconds": done.queue_seconds,
                    "worker_seconds": done.worker_seconds, "batch": done.batch }),
        ),
        Ok(Err(e)) => ("500 Internal Server Error", json!({ "error": e })),
        Err(_) => (
            "500 Internal Server Error",
            json!({ "error": "no worker answered" }),
        ),
    }
}

/// One worker's loop: take queued requests, run them, answer each.
fn dispatch(
    mut worker: Worker,
    queue: &Queue,
    next_example: &AtomicUsize,
    max_batch: usize,
    max_wait: Duration,
) {
    loop {
        let jobs = take(queue, max_batch, max_wait);
        let started = Instant::now();
        let first = next_example.fetch_add(jobs.len(), Ordering::Relaxed);
        let examples: Vec<usize> = (first..first + jobs.len()).collect();
        let replies = if jobs.len() == 1 {
            worker
                .run(first, jobs[0].mask, &jobs[0].ids)
                .map(|r| vec![r])
        } else {
            let masks: Vec<usize> = jobs.iter().map(|j| j.mask).collect();
            let ids: Vec<&[u32]> = jobs.iter().map(|j| j.ids.as_slice()).collect();
            worker.run_batch(&examples, &masks, &ids)
        };
        match replies {
            Ok(replies) => {
                for (job, reply) in jobs.into_iter().zip(replies) {
                    let _ = job.answer.send(Ok(Done {
                        logits: reply.logits,
                        queue_seconds: (started - job.arrived).as_secs_f64(),
                        worker_seconds: reply.worker_seconds,
                        batch: examples.len(),
                    }));
                }
            }
            // A broken worker must fail loudly, not leave the server half up.
            Err(e) => {
                for job in jobs {
                    let _ = job.answer.send(Err(e.clone()));
                }
                eprintln!("worker failed: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// Wait for a request, then for up to `max_wait` more while the batch is not
/// full, and take up to `max_batch` of the oldest.
fn take(queue: &Queue, max_batch: usize, max_wait: Duration) -> Vec<Job> {
    let mut jobs = queue.jobs.lock().expect("queue lock");
    loop {
        while jobs.is_empty() {
            jobs = queue.changed.wait(jobs).expect("queue lock");
        }
        let deadline = jobs[0].arrived + max_wait;
        while jobs.len() < max_batch && !jobs.is_empty() {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            jobs = queue
                .changed
                .wait_timeout(jobs, deadline - now)
                .expect("queue lock")
                .0;
        }
        // Another worker may have taken them while this one waited.
        if !jobs.is_empty() {
            let n = jobs.len().min(max_batch);
            return jobs.drain(..n).collect();
        }
    }
}

/// The CPUs this process may use, one per physical core (the first hardware
/// thread of each), in order. Empty where Linux's `/proc` and `/sys` are not.
pub fn physical_cores() -> Vec<usize> {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return Vec::new();
    };
    let Some(list) = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
    else {
        return Vec::new();
    };
    let mut cores = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for cpu in parse_cpu_list(list.trim()) {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let siblings = std::fs::read_to_string(path).unwrap_or_else(|_| cpu.to_string());
        if seen.insert(siblings.trim().to_string()) {
            cores.push(cpu);
        }
    }
    cores
}

/// "4,7,18-19" -> [4, 7, 18, 19].
fn parse_cpu_list(list: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in list.split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    cpus.extend(a..=b);
                }
            }
            None => cpus.extend(part.parse::<usize>().ok()),
        }
    }
    cpus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_expand_ranges() {
        assert_eq!(parse_cpu_list("4,7,18-20"), vec![4, 7, 18, 19, 20]);
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
    }

    fn job(arrived: Instant) -> (Job, mpsc::Receiver<Result<Done, String>>) {
        let (answer, wait) = mpsc::channel();
        (
            Job {
                ids: vec![MASK_ID],
                mask: 0,
                arrived,
                answer,
            },
            wait,
        )
    }

    #[test]
    fn take_returns_at_most_a_batch_of_the_oldest() {
        let queue = Queue::default();
        let now = Instant::now();
        let mut keep = Vec::new();
        for _ in 0..5 {
            let (j, w) = job(now);
            keep.push(w);
            queue.jobs.lock().unwrap().push_back(j);
        }
        assert_eq!(take(&queue, 2, Duration::ZERO).len(), 2);
        assert_eq!(take(&queue, 8, Duration::ZERO).len(), 3);
    }

    #[test]
    fn take_waits_no_longer_than_max_wait_for_a_batch_to_fill() {
        let queue = Queue::default();
        let (j, _w) = job(Instant::now());
        queue.jobs.lock().unwrap().push_back(j);
        let t = Instant::now();
        assert_eq!(take(&queue, 4, Duration::from_millis(20)).len(), 1);
        let waited = t.elapsed();
        assert!(waited >= Duration::from_millis(15) && waited < Duration::from_millis(200));
    }
}
