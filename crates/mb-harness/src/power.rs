//! CPU energy and clock frequency, sampled by `powermetrics` during a run.
//!
//! `powermetrics` needs root. The README's "Machine setup" section installs
//! a sudo rule for exactly `/usr/bin/powermetrics`, so the config names the
//! command as `["sudo", "-n", "/usr/bin/powermetrics"]`. `-n` makes sudo
//! fail at once instead of waiting for a password nobody will type.
//!
//! One sampler runs for the whole benchmark, one sample every
//! `INTERVAL_MS`. Each sample is the energy used since the one before, so the
//! harness records when each sample arrived and how long it covers, and the
//! analysis in `python/` attributes energy to slices by overlap. The harness
//! records raw samples only, as for timings.
//!
//! The figure is the whole CPU's energy, not one process's. Only one worker
//! runs at a time, so a slice's energy is that worker's plus whatever else
//! the machine does, including `powermetrics` itself. The A/A run measures
//! that floor the same way it measures the timing floor.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

pub const INTERVAL_MS: u64 = 100;

/// How long to wait for the first sample before deciding the sampler is
/// broken (for example, no sudo rule).
const FIRST_SAMPLE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// When the sample arrived, in seconds since the run's origin.
    pub end_seconds: f64,
    /// The span the sample covers, ending at `end_seconds`.
    pub elapsed_seconds: f64,
    /// CPU energy over that span, in millijoules (`processor.cpu_energy`).
    pub cpu_energy_mj: f64,
    /// CPU, GPU and ANE power together, in milliwatts.
    pub combined_power_mw: f64,
    /// Mean active frequency per cluster, e.g. `"P-Cluster" -> 4.04e9`.
    pub cluster_hz: BTreeMap<String, f64>,
}

impl Sample {
    pub fn to_json(&self) -> Value {
        json!({
            "end_seconds": self.end_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "cpu_energy_mj": self.cpu_energy_mj,
            "combined_power_mw": self.combined_power_mw,
            "cluster_hz": self.cluster_hz,
        })
    }
}

pub struct Sampler {
    child: Child,
    /// `None` once `stop` has taken it.
    reader: Option<JoinHandle<Result<Vec<Sample>, String>>>,
    stopped: bool,
}

impl Sampler {
    /// Start sampling, and return once the first sample has arrived.
    pub fn start(command: &[String], origin: Instant) -> Result<Self, String> {
        let (program, prefix) = command
            .split_first()
            .ok_or("the powermetrics command is empty")?;
        let interval = INTERVAL_MS.to_string();
        let mut child = Command::new(program)
            .args(prefix)
            .args(["--samplers", "cpu_power", "-i", &interval, "-f", "plist"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("cannot start {command:?}: {e}"))?;
        let stdout = child.stdout.take().ok_or("powermetrics has no stdout")?;

        let (first, arrived) = mpsc::channel();
        let reader = std::thread::spawn(move || read_samples(stdout, origin, first));

        if arrived.recv_timeout(FIRST_SAMPLE_TIMEOUT).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            let detail = match reader.join() {
                Ok(Err(e)) => e,
                _ => "no sample arrived".to_string(),
            };
            return Err(format!(
                "{command:?} gave no sample within {FIRST_SAMPLE_TIMEOUT:?}: {detail}. \
                 Is the sudo rule from the README installed?"
            ));
        }
        Ok(Self {
            child,
            reader: Some(reader),
            stopped: false,
        })
    }

    /// Stop sampling and return every sample.
    ///
    /// SIGTERM, not SIGINT: sudo did not pass SIGINT on to powermetrics when
    /// tried, and SIGKILL would kill sudo alone and leave powermetrics
    /// running as root. `kill` is the system tool, because the standard
    /// library only sends SIGKILL.
    ///
    /// It first waits two intervals, so a sample arrives whose span covers
    /// the end of the last slice. Without that, the last slice's tail would
    /// have no energy recorded against it.
    pub fn stop(mut self) -> Result<Vec<Sample>, String> {
        std::thread::sleep(Duration::from_millis(2 * INTERVAL_MS));
        self.terminate()?;
        self.reader
            .take()
            .ok_or("the sampler was already stopped")?
            .join()
            .map_err(|_| "the powermetrics reader panicked".to_string())?
    }

    fn terminate(&mut self) -> Result<(), String> {
        if self.stopped {
            return Ok(());
        }
        self.stopped = true;
        let pid = self.child.id().to_string();
        let status = Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .map_err(|e| format!("cannot run kill: {e}"))?;
        if !status.success() {
            return Err(format!("kill -TERM {pid} failed: {status}"));
        }
        self.child
            .wait()
            .map_err(|e| format!("waiting for powermetrics: {e}"))?;
        Ok(())
    }
}

/// A run that fails part-way returns early with `?`. Without this, the
/// sampler would keep running as root after the harness exits.
impl Drop for Sampler {
    fn drop(&mut self) {
        if let Err(e) = self.terminate() {
            eprintln!("could not stop powermetrics: {e}");
        }
    }
}

/// powermetrics separates plist samples with a NUL byte.
fn read_samples(
    stdout: impl std::io::Read,
    origin: Instant,
    first: mpsc::Sender<()>,
) -> Result<Vec<Sample>, String> {
    let mut input = BufReader::new(stdout);
    let mut samples = Vec::new();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let n = input
            .read_until(b'\0', &mut buf)
            .map_err(|e| format!("reading powermetrics: {e}"))?;
        if n == 0 {
            return Ok(samples);
        }
        let end_seconds = origin.elapsed().as_secs_f64();
        let text = String::from_utf8_lossy(&buf);
        // The last read before exit can be a partial sample with no NUL.
        if !buf.ends_with(b"\0") {
            return Ok(samples);
        }
        samples.push(parse(&text, end_seconds)?);
        if samples.len() == 1 {
            let _ = first.send(());
        }
    }
}

/// Pull the few fields we use out of one plist sample.
///
/// A plist parser would be a dependency for five numbers. The fields sit at
/// fixed places in powermetrics' output: a `<key>` followed directly by its
/// value element.
pub fn parse(plist: &str, end_seconds: f64) -> Result<Sample, String> {
    let elapsed_ns = value_after(plist, "elapsed_ns")?;
    let cpu_energy_mj = value_after(plist, "cpu_energy")?;
    let combined_power_mw = value_after(plist, "combined_power")?;

    // Each cluster dict starts with its name; its own `freq_hz` is the
    // first one after the name (the per-CPU ones come later, inside `cpus`).
    let mut cluster_hz = BTreeMap::new();
    let name_key = "<key>name</key><string>";
    let mut rest = plist;
    while let Some(at) = rest.find(name_key) {
        rest = &rest[at + name_key.len()..];
        let end = rest.find('<').ok_or("an unterminated cluster name")?;
        let name = rest[..end].to_string();
        cluster_hz.insert(name, value_after(rest, "freq_hz")?);
    }
    if cluster_hz.is_empty() {
        return Err("a powermetrics sample with no clusters".to_string());
    }

    Ok(Sample {
        end_seconds,
        elapsed_seconds: elapsed_ns / 1e9,
        cpu_energy_mj,
        combined_power_mw,
        cluster_hz,
    })
}

/// The number in the element right after `<key>{key}</key>`, either
/// `<integer>` or `<real>`.
fn value_after(plist: &str, key: &str) -> Result<f64, String> {
    let marker = format!("<key>{key}</key>");
    let at = plist
        .find(&marker)
        .ok_or_else(|| format!("powermetrics sample has no {key}"))?;
    let rest = plist[at + marker.len()..].trim_start();
    let open = rest
        .strip_prefix("<integer>")
        .or_else(|| rest.strip_prefix("<real>"))
        .ok_or_else(|| format!("{key} is not a number"))?;
    let end = open
        .find('<')
        .ok_or_else(|| format!("{key} is unterminated"))?;
    open[..end]
        .parse()
        .map_err(|e| format!("{key} = {:?}: {e}", &open[..end]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape of a real M4 sample, trimmed to the fields `parse` reads
    /// and a few around them.
    const SAMPLE: &str = "<?xml version=\"1.0\"?><plist version=\"1.0\"><dict>\n\
        <key>is_delta</key><true/>\n\
        <key>elapsed_ns</key><integer>204042250</integer>\n\
        <key>processor</key><dict>\n\
        <key>clusters</key><array>\n\
        <dict><key>name</key><string>E-Cluster</string>\n\
        <key>hw_resid_counters</key><true/>\n\
        <key>freq_hz</key><real>1368740000.0</real>\n\
        <key>cpus</key><array><dict><key>cpu</key><integer>0</integer>\n\
        <key>freq_hz</key><real>999.0</real></dict></array></dict>\n\
        <dict><key>name</key><string>P-Cluster</string>\n\
        <key>hw_resid_counters</key><true/>\n\
        <key>freq_hz</key><real>4042570000.0</real></dict>\n\
        </array>\n\
        <key>cpu_energy</key><integer>663</integer>\n\
        <key>combined_power</key><real>3667.58</real>\n\
        </dict></dict></plist>\n";

    #[test]
    fn parses_a_sample() {
        let s = parse(SAMPLE, 1.5).expect("parses");
        assert_eq!(s.end_seconds, 1.5);
        assert!((s.elapsed_seconds - 0.204042250).abs() < 1e-12);
        assert_eq!(s.cpu_energy_mj, 663.0);
        assert_eq!(s.combined_power_mw, 3667.58);
        // The cluster's own frequency, not the per-CPU one after it.
        assert_eq!(s.cluster_hz["E-Cluster"], 1368740000.0);
        assert_eq!(s.cluster_hz["P-Cluster"], 4042570000.0);
    }

    #[test]
    fn a_missing_field_is_an_error() {
        let broken = SAMPLE.replace("cpu_energy", "cpu_enrgy");
        assert!(parse(&broken, 0.0).unwrap_err().contains("cpu_energy"));
    }
}
