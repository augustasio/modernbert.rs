//! macOS's thermal state, read before and after every timed slice.
//!
//! On Apple silicon the unprivileged signal is `ProcessInfo.thermalState`,
//! read by the small Swift program in `tools/thermal_state.swift`. A timing
//! without its thermal state is not a measurement (CLAUDE.md hard rule 3), so
//! a missing or failing reader stops the run instead of being skipped.

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
        other => Err(format!("unexpected thermal state {other:?}")),
    }
}
