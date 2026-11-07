"""memory_run.py measures real processes: a fake worker that holds known amounts of memory."""

import sys
import textwrap

import numpy as np
import pytest

import memory_run

MiB = 2**20

# Speaks the worker protocol. Holds 200 MiB after load, and 100 MiB more for 0.3 s while
# it answers, so the 10 ms sampler must see it. Its logits are the example index.
FAKE_WORKER = textwrap.dedent("""
    import json, os, sys, time
    if os.environ.get("MB_STOP_BEFORE_LOAD"):
        print(json.dumps({"floor": True, "variant": "fake"}), flush=True)
        sys.exit(0)
    weights = bytearray(200 * 2**20)
    print(json.dumps({"ready": True, "variant": "fake", "load_seconds": 0.0}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["cmd"] == "quit":
            sys.exit(0)
        activations = bytearray(100 * 2**20)
        time.sleep(0.3)
        del activations
        logits = [float(request["example"]) + float(os.environ.get("FAKE_OFFSET", "0"))] * 4
        print(json.dumps({"example": request["example"], "seconds": 0.3,
                          "logits": logits}), flush=True)
""")


@pytest.fixture
def worker(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text(FAKE_WORKER)
    return [sys.executable, str(path)]


EXAMPLES = [(3, ([1, 50284, 2], 1)), (5, ([1, 50284, 2], 1))]
REFERENCE = np.array([[i] * 4 for i in range(8)], dtype=np.float32)


def test_the_floor_process_loads_nothing(worker):
    floor = memory_run.floor_process(worker, 1, {})
    assert floor["variant"] == "fake"
    assert floor["exit_code"] == 0
    # A bare interpreter, far below the 200 MiB the full process holds.
    assert floor["peak_rss_bytes"] < 100 * MiB


def test_the_full_process_sees_load_and_activations(worker):
    full = memory_run.full_process(worker, 1, {}, EXAMPLES, REFERENCE)
    assert full["post_load_rss_bytes"] > 200 * MiB
    assert full["activation_delta_bytes"] > 80 * MiB
    assert full["peak_rss_bytes"] >= full["post_load_rss_bytes"] + 80 * MiB
    assert full["cold_start_seconds"] >= 0.3
    assert full["worst_logit_difference"] == 0.0
    assert full["examples"] == 2


def test_wrong_logits_fail_the_run(worker, monkeypatch):
    monkeypatch.setenv("FAKE_OFFSET", "0.5")
    with pytest.raises(RuntimeError, match="above the gate"):
        memory_run.full_process(worker, 1, {}, EXAMPLES, REFERENCE)


def test_the_workload_starts_with_the_longest_prompt():
    examples = [([0], 0)] * 7600
    chosen = memory_run.workload("rung12", examples)
    assert chosen[0][0] == memory_run.LONGEST_EXAMPLE
    assert len(chosen) == memory_run.EXAMPLES
    assert len({i for i, _ in chosen}) == len(chosen)
    assert len(memory_run.workload("rung6", examples)) == memory_run.RUNG6_EXAMPLES


def test_the_memory_builds_replace_only_the_folder():
    assert memory_run.command_for("rung12") == ["target/mem/rung12/release/worker"]
    assert memory_run.command_for("rung2-warm") == memory_run.command_for("rung2")
    assert memory_run.command_for("rung0")[-1] == "0"


def test_system_memory_reads_vm_stat():
    reading = memory_run.system_memory()
    assert reading["free_bytes"] > 0
    assert reading["swap_used_bytes"] >= 0
