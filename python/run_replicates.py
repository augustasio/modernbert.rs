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


def thermal_state() -> str:
    return subprocess.run([THERMAL], capture_output=True, text=True, check=True).stdout.strip()


def wait_for_nominal() -> float:
    """Wait until the thermal state is nominal. Returns the seconds waited."""
    started = time.monotonic()
    while (state := thermal_state()) != "nominal":
        waited = time.monotonic() - started
        if waited > MAX_COOLDOWN_SECONDS:
            raise SystemExit(
                f"still {state} after {waited / 60:.0f} minutes; is something else running?"
            )
        print(f"  thermal state {state}, waiting", flush=True)
        time.sleep(POLL_SECONDS)
    return time.monotonic() - started


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    config_path, n, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(config_path) as handle:
        base = json.load(handle)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {"config": config_path, "replicates": []}
    for k in range(n):
        config = dict(base, seed=base["seed"] + k, label=f"{base['label']} (replicate {k})")
        config_file = os.path.join(out_dir, f"replicate_{k}.config.json")
        with open(config_file, "w") as handle:
            json.dump(config, handle, indent=2)

        waited = wait_for_nominal()
        result_file = os.path.join(out_dir, f"replicate_{k}.json")
        print(f"replicate {k + 1} of {n}, seed {config['seed']}", flush=True)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status = subprocess.run(
            [*HARNESS, config_file, result_file], cwd=REPO, check=False
        ).returncode

        manifest["replicates"].append(
            {
                "replicate": k,
                "seed": config["seed"],
                "started_utc": started,
                "cooldown_seconds": round(waited, 1),
                "exit_status": status,
                "result": os.path.basename(result_file),
            }
        )
        with open(os.path.join(out_dir, "manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2)
        if status != 0:
            raise SystemExit(f"replicate {k} failed or was invalid; see {result_file}")

    print(f"done: {n} replicates in {out_dir}")


if __name__ == "__main__":
    main()
