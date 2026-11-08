"""Write the benchmark configs for one thread count: the A/A floor, then every rung.

    python tools/make_bench_configs.py 4

Each config pairs the rung 0 ruler with one variant (PLAN.md, Benchmark protocol). Slices
hold enough examples for about 10 s of the ruler (rung 0 takes about 0.15 s an example at 4 threads),
so slices are long enough for the energy figures too. Rung 6 takes about 30 s an
example, so it gets 2 examples a slice and 8 cycles, which is still about 8 minutes a
replicate.

Writes bench/configs/t<threads>_<name>.json. The Rust workers are release builds, one
target folder per rung, because mb-model and mb-candle both name their binary `worker`:

    cargo build --release -p mb-candle --bin worker --target-dir target/bench/rung4
    cargo build --release -p mb-candle --features accelerate --bin worker --target-dir target/bench/rung5
    cargo build --release -p mb-model --bin worker --target-dir target/bench/rung6
    cargo build --release -p mb-model --features accelerate --bin worker --target-dir target/bench/rung7
    cargo build --release -p mb-model --features fused --bin worker --target-dir target/bench/rung8
    cargo build --release -p mb-model --features expf --bin worker --target-dir target/bench/rung9
    cargo build --release -p mb-model --features attention-sgemm --bin worker --target-dir target/bench/rung10
    cargo build --release -p mb-model --features threads --bin worker --target-dir target/bench/rung11
    cargo build --release -p mb-model --features f32-math --bin worker --target-dir target/bench/rung12
    cargo build --release -p mb-model --features owned-weights --bin worker --target-dir target/bench/rung13
    cargo build --release -p mb-model --features batch --bin worker --target-dir target/bench/rung14
    cargo build --release -p mb-model --features mask-rows --bin worker --target-dir target/bench/rung15

Rung 16 is rung 15's build on Linux x86 with MKL, so its configs are for a cloud host
only: bench/configs/cloud_t<threads>_<name>.json, for the A/A floor, rung 16 at batch 1
and rung 16 at batch 16. They have no powermetrics, and the thermal helper they name is
tools/cpu_state_linux.sh installed at target/tools/thermal_state. Build it there with:

    cargo build --release -p mb-model --features mask-rows --bin worker --target-dir target/bench/rung16

Rungs 14 and 15 also get the batch sweep: t<threads>_b<size>_<rung>.json for each size
in BATCH_SIZES, with rung 0 as the ruler at the same batch size (padded, as HF batches).
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
    "rung8": ("rung8", ["target/bench/rung8/release/worker"]),
    "rung9": ("rung9", ["target/bench/rung9/release/worker"]),
    "rung10": ("rung10", ["target/bench/rung10/release/worker"]),
    "rung11": ("rung11", ["target/bench/rung11/release/worker"]),
    "rung12": ("rung12", ["target/bench/rung12/release/worker"]),
    "rung13": ("rung13", ["target/bench/rung13/release/worker"]),
    "rung14": ("rung14", ["target/bench/rung14/release/worker"]),
    "rung15": ("rung15", ["target/bench/rung15/release/worker"]),
}

# Rough checks on a Linux cloud host (PLAN.md, phase 2). Not in VARIANTS, so the Mac
# session never runs them.
CLOUD = {
    "aa": VARIANTS["aa"],
    "rung16": ("rung16", ["target/bench/rung16/release/worker"]),
}
CLOUD_BATCH_SIZES = (16,)

# The rungs that accept batches.
BATCHED = ("rung14", "rung15")

# Each divides the 64 examples of a slice.
BATCH_SIZES = (4, 16, 64)


def config(threads: int, name: str, batch_size: int = 1, cloud: bool = False) -> dict:
    variant_name, command = (CLOUD if cloud else VARIANTS)[name]
    slow = name == "rung6"
    batched = {}
    if batch_size > 1:
        # The warm-up is one full batch, so the first measured batch finds its
        # buffers already grown.
        batched = {"batch_size": batch_size, "warmup_examples": batch_size}
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
        # Linux has no powermetrics, so a cloud run reports energy as not measured.
        "powermetrics": None if cloud else ["sudo", "-n", "/usr/bin/powermetrics"],
    } | batched


def main() -> None:
    threads = int(sys.argv[1])
    for name in VARIANTS:
        path = REPO / "bench" / "configs" / f"t{threads}_{name}.json"
        path.write_text(json.dumps(config(threads, name), indent=2) + "\n")
        print(path.relative_to(REPO))
    for name in BATCHED:
        for size in BATCH_SIZES:
            path = REPO / "bench" / "configs" / f"t{threads}_b{size}_{name}.json"
            body = config(threads, name, size)
            body["label"] = f"{threads} threads, batches of {size}: rung 0 ruler against {name}"
            path.write_text(json.dumps(body, indent=2) + "\n")
            print(path.relative_to(REPO))
    for name in CLOUD:
        path = REPO / "bench" / "configs" / f"cloud_t{threads}_{name}.json"
        body = config(threads, name, cloud=True)
        body["label"] = f"cloud, {body['label']}"
        path.write_text(json.dumps(body, indent=2) + "\n")
        print(path.relative_to(REPO))
    for size in CLOUD_BATCH_SIZES:
        path = REPO / "bench" / "configs" / f"cloud_t{threads}_b{size}_rung16.json"
        body = config(threads, "rung16", size, cloud=True)
        body["label"] = f"cloud, {threads} threads, batches of {size}: rung 0 ruler against rung16"
        path.write_text(json.dumps(body, indent=2) + "\n")
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
