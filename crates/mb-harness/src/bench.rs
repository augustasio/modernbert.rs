//! One benchmark run: a variant against the ruler, interleaved.
//!
//! The harness records raw samples and nothing else. Percentiles, intervals
//! and ratios are computed afterwards in `python/`, in one place, from the
//! result file, so the raw data is always there to recompute them from.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{json, Value};

use crate::integrity::{self, MAX_LOGIT_ABS};
use crate::npy;
use crate::power::{self, Sampler};
use crate::schedule::{self, Role};
use crate::thermal;
use crate::worker::Worker;

/// `[MASK]`, where the verbalizer logits are read.
const MASK_ID: u32 = 50284;

#[derive(Debug, Clone)]
pub struct WorkerSpec {
    pub name: String,
    pub command: Vec<String>,
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub label: String,
    pub ruler: WorkerSpec,
    pub variant: WorkerSpec,
    pub threads: u32,
    pub cycles: usize,
    pub examples_per_slice: usize,
    /// Examples per request. 1, the default, sends one `run` per example;
    /// more sends `run_batch` requests of this many, to both workers alike, so
    /// each request is still a pair. `examples_per_slice` must be a multiple.
    pub batch_size: usize,
    pub warmup_examples: usize,
    pub seed: u64,
    pub weights: PathBuf,
    pub prompt_ids: PathBuf,
    pub reference_logits: PathBuf,
    pub thermal_helper: PathBuf,
    /// The command that runs `powermetrics` as root, for example
    /// `["sudo", "-n", "/usr/bin/powermetrics"]`. Absent means energy and
    /// frequency are not measured, and the result says so.
    pub powermetrics: Option<Vec<String>>,
}

impl Config {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let text = |k: &str| {
            v[k].as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("config: {k} must be a string"))
        };
        let number = |k: &str| {
            v[k].as_u64()
                .ok_or_else(|| format!("config: {k} must be a non-negative integer"))
        };
        let spec = |k: &str| -> Result<WorkerSpec, String> {
            let w = &v[k];
            let command = w["command"]
                .as_array()
                .ok_or_else(|| format!("config: {k}.command must be a list"))?
                .iter()
                .map(|s| s.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| format!("config: {k}.command must hold strings"))?;
            let mut env = BTreeMap::new();
            if let Some(map) = w["env"].as_object() {
                for (key, val) in map {
                    let val = val
                        .as_str()
                        .ok_or_else(|| format!("config: {k}.env.{key} must be a string"))?;
                    env.insert(key.clone(), val.to_string());
                }
            }
            Ok(WorkerSpec {
                name: w["name"]
                    .as_str()
                    .ok_or_else(|| format!("config: {k}.name must be a string"))?
                    .to_string(),
                command,
                env,
            })
        };
        Ok(Config {
            label: text("label")?,
            ruler: spec("ruler")?,
            variant: spec("variant")?,
            threads: number("threads")? as u32,
            cycles: number("cycles")? as usize,
            examples_per_slice: number("examples_per_slice")? as usize,
            batch_size: match &v["batch_size"] {
                Value::Null => 1,
                b => b
                    .as_u64()
                    .filter(|&b| b >= 1)
                    .ok_or("config: batch_size must be a positive integer")?
                    as usize,
            },
            warmup_examples: number("warmup_examples")? as usize,
            seed: number("seed")?,
            weights: text("weights")?.into(),
            prompt_ids: text("prompt_ids")?.into(),
            reference_logits: text("reference_logits")?.into(),
            thermal_helper: text("thermal_helper")?.into(),
            powermetrics: match &v["powermetrics"] {
                Value::Null => None,
                Value::Array(parts) => Some(
                    parts
                        .iter()
                        .map(|s| s.as_str().map(str::to_string))
                        .collect::<Option<Vec<_>>>()
                        .ok_or("config: powermetrics must hold strings")?,
                ),
                _ => return Err("config: powermetrics must be a list".to_string()),
            },
        })
    }
}

/// One test example: its token ids and where the mask sits.
struct Example {
    ids: Vec<u32>,
    mask: usize,
}

fn read_examples(path: &Path) -> Result<Vec<Example>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let ids_field = line
            .split('\t')
            .nth(2)
            .ok_or_else(|| format!("{}:{line_no}: expected three fields", path.display()))?;
        let ids: Vec<u32> = ids_field
            .split(' ')
            .map(|t| {
                t.parse()
                    .map_err(|_| format!("line {line_no}: bad id {t:?}"))
            })
            .collect::<Result<_, _>>()?;
        let mask = ids
            .iter()
            .rposition(|&id| id == MASK_ID)
            .ok_or_else(|| format!("line {line_no}: no [MASK]"))?;
        out.push(Example { ids, mask });
    }
    Ok(out)
}

/// Run a command and return its trimmed output, or a note saying why not.
fn probe(program: &str, args: &[&str]) -> String {
    match Command::new(program).args(args).output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Ok(o) => format!("({program} exited with {})", o.status),
        Err(e) => format!("({program}: {e})"),
    }
}

/// What the machine was doing, recorded with every run.
fn environment() -> Value {
    json!({
        "started_utc": probe("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "macos": probe("sw_vers", &["-productVersion"]),
        "model": probe("sysctl", &["-n", "hw.model"]),
        "cpus": probe("sysctl", &["-n", "hw.ncpu"]),
        "perf_cores": probe("sysctl", &["-n", "hw.perflevel0.physicalcpu"]),
        "efficiency_cores": probe("sysctl", &["-n", "hw.perflevel1.physicalcpu"]),
        "memory_bytes": probe("sysctl", &["-n", "hw.memsize"]),
        "power_source": probe("pmset", &["-g", "ps"]).lines().next().unwrap_or("").to_string(),
        "low_power_mode": probe("pmset", &["-g"]).lines()
            .find(|l| l.contains("lowpowermode")).unwrap_or("(not reported)").trim().to_string(),
        "git_commit": probe("git", &["rev-parse", "HEAD"]),
        // Tracked files only: the session writes its own results while it runs.
        "git_dirty": !probe("git", &["status", "--porcelain", "--untracked-files=no"]).is_empty(),
    })
}

fn worker_env(config: &Config, spec: &WorkerSpec) -> BTreeMap<String, String> {
    let threads = config.threads.to_string();
    let mut env = BTreeMap::from([
        ("MB_THREADS".to_string(), threads.clone()),
        ("VECLIB_MAXIMUM_THREADS".to_string(), threads.clone()),
        ("OMP_NUM_THREADS".to_string(), threads),
        (
            "MB_WEIGHTS".to_string(),
            config.weights.display().to_string(),
        ),
    ]);
    // Per-worker settings win, but a worker may not quietly change the
    // thread count, since equal threads is what makes the pair comparable.
    for (k, v) in &spec.env {
        env.insert(k.clone(), v.clone());
    }
    env
}

/// Run the benchmark and return the result document.
///
/// `Err` only for a run that could not happen. A run that happened but broke
/// a rule, such as a wrong logit, returns `Ok` with `valid: false` and the
/// reasons, so the evidence is kept and the failure is loud.
pub fn run(config: &Config) -> Result<Value, String> {
    let examples = read_examples(&config.prompt_ids)?;
    let reference = npy::read(&config.reference_logits)?;
    if reference.rows != examples.len() || reference.cols != 4 {
        return Err(format!(
            "reference logits are {}x{}, expected {}x4",
            reference.rows,
            reference.cols,
            examples.len()
        ));
    }
    // Fail before loading anything if the thermal reader is not there.
    thermal::read(&config.thermal_helper)?;

    // Every timestamp in the result is seconds since this instant, so slices
    // and power samples share one clock.
    let origin = Instant::now();
    let sampler = match &config.powermetrics {
        Some(command) => Some(Sampler::start(command, origin)?),
        None => None,
    };

    if config.batch_size > 1 && config.examples_per_slice % config.batch_size != 0 {
        return Err(format!(
            "examples_per_slice {} is not a multiple of batch_size {}",
            config.examples_per_slice, config.batch_size
        ));
    }
    let environment = environment();
    let cycles = schedule::plan(
        examples.len(),
        config.cycles,
        config.examples_per_slice,
        config.seed,
    );

    eprintln!("starting {} and {}", config.ruler.name, config.variant.name);
    let mut ruler = Worker::start(
        &config.ruler.name,
        &config.ruler.command,
        &worker_env(config, &config.ruler),
    )?;
    let mut variant = Worker::start(
        &config.variant.name,
        &config.variant.command,
        &worker_env(config, &config.variant),
    )?;

    let mut slices = Vec::new();
    let mut checked = 0usize;
    let mut worst_abs = 0.0f32;
    let mut failures: Vec<Value> = Vec::new();

    // Warm-up: the same examples on both, recorded and marked, never mixed
    // in with the measured slices.
    let warmup: Vec<usize> = (0..config.warmup_examples)
        .map(|i| (examples.len() - 1 - i) % examples.len())
        .collect();

    let mut run_slice = |worker: &mut Worker,
                         role: Role,
                         cycle: Option<usize>,
                         picks: &[usize],
                         slices: &mut Vec<Value>|
     -> Result<(), String> {
        let thermal_before = thermal::read(&config.thermal_helper)?;
        let start_seconds = origin.elapsed().as_secs_f64();
        let mut worker_seconds = Vec::with_capacity(picks.len());
        let mut roundtrip_seconds = Vec::with_capacity(picks.len());
        // One request per group: one example, or a batch of `batch_size`.
        // The groups are the same for both workers, so their timings pair.
        for group in picks.chunks(config.batch_size) {
            let replies = if config.batch_size == 1 {
                let e = group[0];
                vec![worker.run(e, examples[e].mask, &examples[e].ids)?]
            } else {
                let masks: Vec<usize> = group.iter().map(|&e| examples[e].mask).collect();
                let ids: Vec<&[u32]> = group.iter().map(|&e| examples[e].ids.as_slice()).collect();
                worker.run_batch(group, &masks, &ids)?
            };
            for (&e, reply) in group.iter().zip(&replies) {
                let c = integrity::check(reference.row(e), &reply.logits);
                checked += 1;
                worst_abs = worst_abs.max(c.max_abs);
                if !c.passes() {
                    failures.push(json!({
                        "worker": worker.name, "example": e,
                        "argmax_agrees": c.argmax_agrees, "max_abs": c.max_abs,
                        "logits": reply.logits, "reference": reference.row(e),
                    }));
                }
            }
            // A batch is one request with one time.
            worker_seconds.push(replies[0].worker_seconds);
            roundtrip_seconds.push(replies[0].roundtrip_seconds);
        }
        let end_seconds = origin.elapsed().as_secs_f64();
        let thermal_after = thermal::read(&config.thermal_helper)?;
        slices.push(json!({
            "cycle": cycle,
            "warmup": cycle.is_none(),
            "role": role.name(),
            "worker": worker.name,
            "examples": picks,
            "worker_seconds": worker_seconds,
            "roundtrip_seconds": roundtrip_seconds,
            "thermal_before": thermal_before,
            "thermal_after": thermal_after,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
        }));
        Ok(())
    };

    run_slice(&mut ruler, Role::Ruler, None, &warmup, &mut slices)?;
    run_slice(&mut variant, Role::Variant, None, &warmup, &mut slices)?;

    for c in &cycles {
        for role in c.order {
            let worker = match role {
                Role::Ruler => &mut ruler,
                Role::Variant => &mut variant,
            };
            run_slice(worker, role, Some(c.index), &c.examples, &mut slices)?;
        }
        eprintln!("cycle {} of {} done", c.index + 1, cycles.len());
    }

    let workers = json!({ "ruler": ruler.ready.clone(), "variant": variant.ready.clone() });
    ruler.quit()?;
    variant.quit()?;
    let power = match sampler {
        Some(s) => {
            let samples = s.stop()?;
            json!({
                "measured": true,
                "command": config.powermetrics,
                "interval_ms": power::INTERVAL_MS,
                "samples": samples.iter().map(power::Sample::to_json).collect::<Vec<_>>(),
            })
        }
        None => json!({
            "measured": false,
            "reason": "no powermetrics command in the config",
        }),
    };

    let mut reasons = Vec::new();
    if !failures.is_empty() {
        reasons.push(format!(
            "{} replies failed the in-run correctness check against rung 0",
            failures.len()
        ));
    }
    let thermal_states: Vec<&str> = slices
        .iter()
        .flat_map(|s| [s["thermal_before"].as_str(), s["thermal_after"].as_str()])
        .flatten()
        .collect();
    if thermal_states.is_empty() {
        reasons.push("no thermal state was recorded".to_string());
    }

    Ok(json!({
        "label": config.label,
        "valid": reasons.is_empty(),
        "invalid_because": reasons,
        "config": {
            "ruler": config.ruler.name, "variant": config.variant.name,
            "ruler_command": config.ruler.command, "variant_command": config.variant.command,
            "threads": config.threads, "cycles": config.cycles,
            "examples_per_slice": config.examples_per_slice,
            "batch_size": config.batch_size,
            "warmup_examples": config.warmup_examples, "seed": config.seed,
            "prompt_ids": config.prompt_ids, "reference_logits": config.reference_logits,
        },
        "environment": environment,
        "workers": workers,
        "integrity": {
            "checked": checked,
            "tolerance_max_abs": MAX_LOGIT_ABS,
            "worst_max_abs": worst_abs,
            "failures": failures,
        },
        "slices": slices,
        "power": power,
    }))
}
