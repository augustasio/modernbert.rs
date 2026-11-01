"""Rung 2: rung 1 plus torch.compile (inductor backend, CPU).

The one variable against rung 1 is `torch.compile(model, backend="inductor",
dynamic=True)`. Same module, same weights, same inputs.

`dynamic=True`, because AG News prompts have about 230 distinct lengths (119 to 348
tokens). Without it, inductor specialises on the first shape and recompiles for each new
length. Each compile of a 395M-parameter graph takes tens of seconds, so that would turn
compile time into most of the run.

Compile time belongs to cold start, so it is recorded separately here, not hidden:
  * the wall time of the first call, which includes tracing and inductor codegen;
  * every later call that takes over RECOMPILE_SECONDS, which at this model size means a
    recompile, since a warm forward pass is a fraction of a second;
  * dynamo's own counters: graphs, graph breaks and recompiles.
None of this is a latency measurement. The harness measures latency later.

If inductor fails, the exception is written to results/rung2.json with status "failed",
and the script exits non-zero. A failed rung is a result.

Usage:
    python/.venv/bin/python python/rung2_compile.py
"""

import os

# One inductor compile worker. The default forks a pool of workers, one per core, and
# each holds its own memory. On a 16 GB machine shared with a Rust build, that is the
# fastest way to run out of memory (CLAUDE.md hard rule 9). Must be set before torch is
# imported.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import time
import traceback

import torch
import torch._dynamo
from transformers import AutoTokenizer

import ladder
from agnews import MODEL_DIR, resolve_verbalizer_ids

RUNG = 2
RECOMPILE_SECONDS = 5.0


def dynamo_counters() -> dict:
    """A plain-dict copy of dynamo's counters: graphs, graph breaks, recompiles."""
    return {
        group: {str(key): int(value) for key, value in counter.items()}
        for group, counter in torch._dynamo.utils.counters.items()
    }


def main() -> None:
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    verbalizer_ids = resolve_verbalizer_ids(tokenizer)
    inputs, ids_sha256 = ladder.encode_test_set(tokenizer)
    content_free_inputs = ladder.encode_content_free(tokenizer)

    model = ladder.load_mask_head_model(verbalizer_ids)
    record = {
        "rung": RUNG,
        "description": "rung 1 plus torch.compile(backend='inductor', dynamic=True), CPU",
        "torch_version": torch.__version__,
        "compile_settings": {
            "backend": "inductor",
            "dynamic": True,
            "mode": "default",
            "TORCHINDUCTOR_COMPILE_THREADS": os.environ["TORCHINDUCTOR_COMPILE_THREADS"],
        },
        "input_ids_sha256": ids_sha256,
    }

    try:
        compiled = torch.compile(model, backend="inductor", dynamic=True)
        compiled_score = ladder.torch_score(compiled)

        # The first call compiles. Timed on its own, because it is cold-start cost.
        started = time.perf_counter()
        compiled_score(inputs[0])
        record["compile"] = {
            "first_call_seconds": time.perf_counter() - started,
            "first_call_seq_len": int(inputs[0]["input_ids"].shape[1]),
        }

        slow_calls = []

        def watched_score(item):
            call_started = time.perf_counter()
            out = compiled_score(item)
            seconds = time.perf_counter() - call_started
            if seconds > RECOMPILE_SECONDS:
                slow_calls.append({"seq_len": int(item["input_ids"].shape[1]), "seconds": seconds})
            return out

        logits = ladder.run(watched_score, inputs, f"rung {RUNG}")
        content_free_logits = ladder.run(
            watched_score, content_free_inputs, f"rung {RUNG} content-free"
        )
    except Exception as error:  # A failed compile is a result: record it in full.
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)[:4000]
        record["traceback"] = traceback.format_exc()[-8000:]
        record["dynamo_counters"] = dynamo_counters()
        ladder.write_result(RUNG, record, None)
        raise

    record["compile"]["calls_over_recompile_threshold"] = slow_calls
    record["compile"]["recompile_threshold_seconds"] = RECOMPILE_SECONDS
    record["dynamo_counters"] = dynamo_counters()
    graph_breaks = sum(record["dynamo_counters"].get("graph_break", {}).values())
    record["graph_breaks"] = graph_breaks
    record["status"] = "ok" if graph_breaks == 0 else "ok_with_graph_breaks"

    comparison = ladder.compare_with_rung0(logits, content_free_logits)
    record.update(comparison)
    ladder.write_result(RUNG, record, logits)
    ladder.print_summary(RUNG, comparison)
    print(f"  compile, first call: {record['compile']['first_call_seconds']:.1f} s")
    print(f"  calls over {RECOMPILE_SECONDS} s (recompiles): {len(slow_calls)}")
    print(f"  graph breaks: {graph_breaks}")


if __name__ == "__main__":
    main()
