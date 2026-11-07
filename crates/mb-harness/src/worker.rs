//! One worker process, spoken to over the protocol in `PROTOCOL.md`.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use serde_json::{json, Value};

pub struct Worker {
    pub name: String,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// The worker's `ready` line, kept for the result file.
    pub ready: Value,
}

/// One answer to a `run` request.
pub struct Reply {
    /// The worker's own timing of the forward pass alone.
    pub worker_seconds: f64,
    /// The harness's timing of the whole request, including the pipe and the
    /// JSON. The gap between the two shows what the protocol itself costs.
    pub roundtrip_seconds: f64,
    pub logits: Vec<f32>,
}

impl Worker {
    pub fn start(
        name: &str,
        command: &[String],
        env: &BTreeMap<String, String>,
    ) -> Result<Self, String> {
        let (program, args) = command
            .split_first()
            .ok_or_else(|| format!("{name}: empty command"))?;
        let mut child = Command::new(program)
            .args(args)
            .envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("{name}: cannot start {program}: {e}"))?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout was piped"));

        let mut worker = Worker {
            name: name.to_string(),
            child,
            stdin,
            stdout,
            ready: Value::Null,
        };
        let ready = worker.receive()?;
        if ready["ready"] != json!(true) {
            return Err(format!("{name}: expected a ready line, got {ready}"));
        }
        worker.ready = ready;
        Ok(worker)
    }

    pub fn run(&mut self, example: usize, mask: usize, ids: &[u32]) -> Result<Reply, String> {
        let request = json!({ "cmd": "run", "example": example, "mask": mask, "ids": ids });
        let t = Instant::now();
        self.send(&request)?;
        let reply = self.receive()?;
        let roundtrip_seconds = t.elapsed().as_secs_f64();

        if reply["example"].as_u64() != Some(example as u64) {
            return Err(format!(
                "{}: asked for example {example}, got {reply}",
                self.name
            ));
        }
        let worker_seconds = reply["seconds"]
            .as_f64()
            .ok_or_else(|| format!("{}: no seconds in {reply}", self.name))?;
        let logits = reply["logits"]
            .as_array()
            .ok_or_else(|| format!("{}: no logits in {reply}", self.name))?
            .iter()
            .map(|v| v.as_f64().map(|x| x as f32))
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| format!("{}: a logit is not a number in {reply}", self.name))?;
        Ok(Reply {
            worker_seconds,
            roundtrip_seconds,
            logits,
        })
    }

    /// One `run_batch` request: one reply per example, all carrying the
    /// batch's one timing.
    pub fn run_batch(
        &mut self,
        examples: &[usize],
        masks: &[usize],
        ids: &[&[u32]],
    ) -> Result<Vec<Reply>, String> {
        let request =
            json!({ "cmd": "run_batch", "examples": examples, "masks": masks, "ids": ids });
        let t = Instant::now();
        self.send(&request)?;
        let reply = self.receive()?;
        let roundtrip_seconds = t.elapsed().as_secs_f64();

        if reply["examples"] != json!(examples) {
            return Err(format!(
                "{}: asked for examples {examples:?}, got {}",
                self.name, reply["examples"]
            ));
        }
        let worker_seconds = reply["seconds"]
            .as_f64()
            .ok_or_else(|| format!("{}: no seconds in the batch reply", self.name))?;
        let rows = reply["logits"]
            .as_array()
            .filter(|rows| rows.len() == examples.len())
            .ok_or_else(|| format!("{}: not one logit row per example", self.name))?;
        rows.iter()
            .map(|row| {
                let logits = row
                    .as_array()
                    .ok_or_else(|| format!("{}: a logit row is not a list", self.name))?
                    .iter()
                    .map(|v| v.as_f64().map(|x| x as f32))
                    .collect::<Option<Vec<f32>>>()
                    .ok_or_else(|| format!("{}: a logit is not a number", self.name))?;
                Ok(Reply {
                    worker_seconds,
                    roundtrip_seconds,
                    logits,
                })
            })
            .collect()
    }

    pub fn quit(mut self) -> Result<(), String> {
        self.send(&json!({ "cmd": "quit" }))?;
        let status = self
            .child
            .wait()
            .map_err(|e| format!("{}: wait: {e}", self.name))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("{}: exited with {status}", self.name))
        }
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    fn send(&mut self, message: &Value) -> Result<(), String> {
        writeln!(self.stdin, "{message}")
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("{}: cannot write to the worker: {e}", self.name))
    }

    fn receive(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        let n = self
            .stdout
            .read_line(&mut line)
            .map_err(|e| format!("{}: cannot read from the worker: {e}", self.name))?;
        if n == 0 {
            return Err(format!("{}: the worker closed its output", self.name));
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|e| format!("{}: not JSON ({e}): {line:?}", self.name))?;
        if let Some(message) = value.get("error") {
            return Err(format!("{}: the worker reported {message}", self.name));
        }
        Ok(value)
    }
}

impl Drop for Worker {
    /// A worker left behind holds 1.6 GB, so never leave one.
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
