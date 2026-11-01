// Print macOS's thermal state: nominal, fair, serious or critical.
//
// On Apple silicon this is the unprivileged thermal signal. `pmset -g therm`
// records nothing here, and `powermetrics` needs root. The harness runs this
// before and after every timed slice, so every timing carries its thermal
// state (CLAUDE.md hard rule 3).
//
// Build once: swiftc -O tools/thermal_state.swift -o target/tools/thermal_state

import Foundation

switch ProcessInfo.processInfo.thermalState {
case .nominal: print("nominal")
case .fair: print("fair")
case .serious: print("serious")
case .critical: print("critical")
@unknown default: print("unknown")
}
