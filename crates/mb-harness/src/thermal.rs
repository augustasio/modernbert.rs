//! macOS's thermal state, read before and after every timed slice.
//!
//! On Apple silicon the unprivileged signal is `ProcessInfo.thermalState`,
//! read by the small Swift program in `tools/thermal_state.swift`. A timing
//! without its thermal state is not a measurement (CLAUDE.md hard rule 3), so
//! a missing or failing reader stops the run instead of being skipped.
//!
//! On a Linux cloud host there is no thermal state. There the helper is
//! `tools/cpu_state_linux.sh`, which prints `linux mhz=.. steal=.. total=..`:
//! the CPU clock and the steal counters that PLAN.md's rough checks record in
//! its place. Such a slice is never `nominal`, so it can never be mixed into
//! the Mac's nominal-only figures.

use std::path::Path;
use std::process::Command;

pub fn read(helper: &Path) -> Result<String, String> {
    let output = Command::new(helper).output().map_err(|e| {
        format!(
            "cannot run {}: {e}. Build it with: swiftc -O tools/thermal_state.swift -o {}",
            helper.display(),
            helper.display()
        )
    })?;
    if !output.status.success() {
        return Err(format!(
            "{} exited with {}",
            helper.display(),
            output.status
        ));
    }
    let state = String::from_utf8_lossy(&output.stdout).trim().to_string();
    match state.as_str() {
        "nominal" | "fair" | "serious" | "critical" => Ok(state),
        linux if linux.starts_with("linux mhz=") && linux.contains(" steal=") => Ok(state),
        other => Err(format!("unexpected thermal state {other:?}")),
    }
}
