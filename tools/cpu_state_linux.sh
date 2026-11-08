#!/bin/sh
# Linux stand-in for tools/thermal_state.swift, for rough checks on cloud hosts
# (PLAN.md, phase 2). A cloud VM has no thermalState, so the harness records this
# line before and after every timed slice instead:
#
#     linux mhz=<mean clock of all CPUs> steal=<steal jiffies> total=<all jiffies>
#
# steal and total are /proc/stat's cumulative counters, so the analysis takes the
# steal share of a slice from the difference of its two readings. Steal is time the
# hypervisor gave this VM's CPUs to another renter.
#
# Install where the configs look for it: cp tools/cpu_state_linux.sh target/tools/thermal_state
mhz=$(awk -F: '/^cpu MHz/ { s += $2; n++ } END { if (n) printf "%.0f", s / n; else print "unknown" }' /proc/cpuinfo)
awk -v mhz="$mhz" '$1 == "cpu" { t = 0; for (i = 2; i <= NF; i++) t += $i; print "linux mhz=" mhz " steal=" $9 " total=" t; exit }' /proc/stat
