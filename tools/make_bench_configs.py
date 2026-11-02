"""Write the benchmark configs for one thread count: the A/A floor, then every rung.

    python tools/make_bench_configs.py 4

Each config pairs the rung 0 ruler with one variant (PLAN.md, Benchmark protocol). Slices
hold enough examples for about 10 s of the ruler (rung 0 takes about 0.15 s an example at 4 threads),
so slices are long enough for the energy figures too. Rung 6 takes about 30 s an
example, so it gets 2 examples a slice and 8 cycles, which is still about 8 minutes a
replicate.

Writes bench/configs/t<threads>_<name>.json. The Rust workers are release builds, one
target folder per rung, because mb-model and mb-candle both name their binary `worker`.
"""

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PY = ["python/.venv/bin/python", "python/worker.py", "--rung"]

VARIANTS = {
    "aa": ("rung0-copy", [*PY, "0"]),
    "rung1": ("rung1", [*PY, "1"]),
    "rung2": ("rung2", [*PY, "2"]),
    "rung3": ("rung3", [*PY, "3"]),
    "rung4": ("rung4", ["target/bench/rung4/release/worker"]),
    "rung5": ("rung5", ["target/bench/rung5/release/worker"]),
    "rung6": ("rung6", ["target/bench/rung6/release/worker"]),
    "rung7": ("rung7", ["target/bench/rung7/release/worker"]),
}


def config(threads: int, name: str) -> dict:
    variant_name, command = VARIANTS[name]
    slow = name == "rung6"
    return {
        "label": f"{threads} threads: rung 0 ruler against {variant_name}",
        "ruler": {"name": "rung0", "command": [*PY, "0"]},
        "variant": {"name": variant_name, "command": command},
        "threads": threads,
        "cycles": 8 if slow else 30,
        "examples_per_slice": 2 if slow else 64,
        "warmup_examples": 1 if slow else 3,
        "seed": 1000 * threads,
        "weights": "weights/ModernBERT-Large-Instruct",
        "prompt_ids": "data/agnews_test_prompt_ids.tsv",
        "reference_logits": "results/rung0_logits.npy",
        "thermal_helper": "target/tools/thermal_state",
        "powermetrics": ["sudo", "-n", "/usr/bin/powermetrics"],
    }


def main() -> None:
    threads = int(sys.argv[1])
    for name in VARIANTS:
        path = REPO / "bench" / "configs" / f"t{threads}_{name}.json"
        path.write_text(json.dumps(config(threads, name), indent=2) + "\n")
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
