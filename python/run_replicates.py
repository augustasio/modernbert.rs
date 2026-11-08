"""Run one benchmark configuration as N replicates, ready for analyze_bench.py.

    python python/run_replicates.py bench/configs/<name>.json 5 results/bench/<name>

Each replicate is a separate harness run, so a fresh pair of worker processes, with its
own seed (the config's seed plus the replicate number), so the replicates run different
example schedules. Before each one the script waits until macOS reports the thermal state
as nominal, so every replicate starts from the same thermal condition, and records how
long that took. A replicate that comes back invalid stops the sequence: an invalid run
must be looked at, not averaged in.

The output folder gets replicate_<k>.json for each run and manifest.json, which records
the seeds, the start times and the cool-down waits.

It is resumable: run the same command again and it skips every replicate the manifest
records as finished and valid, with its result file present (plain or gzipped). A
replicate that was cut off is run again from the start. A replicate during which the Mac
slept is discarded and run again, because its slices mix two thermal states; the
manifest lists it under "discarded".

On a Linux cloud host (rung 16 on) the thermal helper is tools/cpu_state_linux.sh, which
reports a CPU clock and steal counters, not a thermal state. There is nothing to cool
down to, so the wait is skipped.
"""

import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = ["cargo", "run", "-q", "--release", "-p", "mb-harness", "--bin", "mb-harness", "--"]
THERMAL = os.path.join(REPO, "target", "tools", "thermal_state")

POLL_SECONDS = 30
MAX_COOLDOWN_SECONDS = 15 * 60
# The wall clock runs on through sleep and CLOCK_UPTIME_RAW does not, so a gap between
# them means the Mac slept. A few seconds of slack covers clock adjustments.
SLEEP_SLACK_SECONDS = 10
MAX_ATTEMPTS = 3


def thermal_state() -> str:
    return subprocess.run([THERMAL], capture_output=True, text=True, check=True).stdout.strip()


def wait_for_nominal() -> float:
    """Wait until the thermal state is nominal. Returns the seconds waited."""
    started = time.monotonic()
    if thermal_state().startswith("linux "):
        return 0.0
    while (state := thermal_state()) != "nominal":
        waited = time.monotonic() - started
        if waited > MAX_COOLDOWN_SECONDS:
            raise SystemExit(
                f"still {state} after {waited / 60:.0f} minutes; is something else running?"
            )
        print(f"  thermal state {state}, waiting", flush=True)
        time.sleep(POLL_SECONDS)
    return time.monotonic() - started


# A clock that stops while the machine sleeps: CLOCK_UPTIME_RAW on macOS, CLOCK_MONOTONIC
# on Linux (which, unlike CLOCK_BOOTTIME, does not count suspend).
AWAKE_CLOCK = getattr(time, "CLOCK_UPTIME_RAW", time.CLOCK_MONOTONIC)


def run_harness(config_file: str, result_file: str) -> tuple[int, float]:
    """Run one replicate. Returns the exit status and the seconds the Mac slept meanwhile."""
    wall, awake = time.time(), time.clock_gettime(AWAKE_CLOCK)
    status = subprocess.run([*HARNESS, config_file, result_file], cwd=REPO, check=False).returncode
    slept = (time.time() - wall) - (time.clock_gettime(AWAKE_CLOCK) - awake)
    return status, slept


def load_manifest(path: str, config_path: str) -> dict:
    if not os.path.exists(path):
        return {"config": config_path, "replicates": []}
    with open(path) as handle:
        manifest = json.load(handle)
    manifest.setdefault("discarded", [])
    return manifest


def is_done(manifest: dict, k: int, out_dir: str) -> bool:
    """A replicate is done when the manifest says it passed and its result file exists."""
    entry = next((r for r in manifest["replicates"] if r["replicate"] == k), None)
    if entry is None:
        return False
    if entry["exit_status"] != 0:
        # Never rerun over an invalid result: it must be looked at first.
        raise SystemExit(
            f"replicate {k} in {out_dir} failed or was invalid; look at it, then delete "
            "its manifest entry to run it again"
        )
    result = os.path.join(out_dir, entry["result"])
    return os.path.exists(result) or os.path.exists(result + ".gz")


def save_manifest(manifest: dict, path: str) -> None:
    manifest["replicates"].sort(key=lambda r: r["replicate"])
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    config_path, n, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(config_path) as handle:
        base = json.load(handle)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = load_manifest(manifest_path, config_path)

    for k in range(n):
        if is_done(manifest, k, out_dir):
            print(f"replicate {k + 1} of {n} already done, skipping", flush=True)
            continue
        manifest["replicates"] = [r for r in manifest["replicates"] if r["replicate"] != k]
        config = dict(base, seed=base["seed"] + k, label=f"{base['label']} (replicate {k})")
        config_file = os.path.join(out_dir, f"replicate_{k}.config.json")
        with open(config_file, "w") as handle:
            json.dump(config, handle, indent=2)
        result_file = os.path.join(out_dir, f"replicate_{k}.json")

        for attempt in range(MAX_ATTEMPTS):
            waited = wait_for_nominal()
            print(f"replicate {k + 1} of {n}, seed {config['seed']}", flush=True)
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            status, slept = run_harness(config_file, result_file)
            entry = {
                "replicate": k,
                "seed": config["seed"],
                "started_utc": started,
                "cooldown_seconds": round(waited, 1),
                "exit_status": status,
                "result": os.path.basename(result_file),
            }
            if slept <= SLEEP_SLACK_SECONDS:
                break
            print(f"  the Mac slept {slept:.0f} s during replicate {k}; running it again")
            manifest.setdefault("discarded", []).append(dict(entry, slept_seconds=round(slept)))
            if os.path.exists(result_file):
                os.remove(result_file)
            save_manifest(manifest, manifest_path)
        else:
            raise SystemExit(f"replicate {k}: the Mac slept during {MAX_ATTEMPTS} attempts")

        manifest["replicates"].append(entry)
        save_manifest(manifest, manifest_path)
        if status != 0:
            raise SystemExit(f"replicate {k} failed or was invalid; see {result_file}")

    print(f"done: {n} replicates in {out_dir}")


if __name__ == "__main__":
    main()
