"""Where OpenVINO's static int8 model spends a request: analysis, not a rung.

Runs the OpenVINO baseline (baseline_int8.py) on the first N test prompts with the CPU
plugin's per-node counters on, and prints the time per example by node type and by the
kernel the plugin chose (its "exec type", e.g. brgemm_avx512_core_vnni_I8). It also saves
the compiled ("runtime") graph, which shows what the plugin fused into what.

The same cores and thread count as the rung's `stage-times` run, so the two tables can be
read side by side:

    OMP_PLACES="{4},{5}" python/.venv/bin/python python/profile_openvino.py 2 64 out_dir
"""

import collections
import json
import os
import sys
import time

import numpy as np

import baseline_int8
from agnews import REPO_ROOT


def items(n: int) -> list[dict]:
    """The first n test prompts, from the same ids file the Rust rungs read."""
    path = os.path.join(REPO_ROOT, "data", "agnews_test_prompt_ids.tsv")
    out = []
    with open(path) as handle:
        for line in handle:
            ids = [int(t) for t in line.rstrip("\n").split("\t")[2].split()]
            input_ids = np.array([ids], dtype=np.int64)
            mask = ids.index(50284)  # [MASK]
            out.append({
                "input_ids": input_ids,
                "attention_mask": np.ones_like(input_ids),
                "mask_position": np.array([mask], dtype=np.int64),
            })
            if len(out) == n:
                break
    return out


def main() -> None:
    threads, n, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    import openvino as ov  # noqa: PLC0415

    baseline_int8.bind_to_worker_cores()
    os.makedirs(out_dir, exist_ok=True)
    compiled = ov.Core().compile_model(
        baseline_int8.OPENVINO_INT8_PATH,
        "CPU",
        {
            "INFERENCE_NUM_THREADS": threads,
            "PERFORMANCE_HINT": "LATENCY",
            "ENABLE_CPU_PINNING": False,
            "PERF_COUNT": True,
        },
    )
    ov.save_model(compiled.get_runtime_model(), os.path.join(out_dir, "runtime_model.xml"))
    request = compiled.create_infer_request()
    batch = items(n)
    for item in batch[:4]:  # warm-up, not counted
        request.infer(item)

    by_type = collections.Counter()
    by_exec = collections.Counter()
    walls = []
    for item in batch:
        start = time.perf_counter()
        request.infer(item)
        walls.append(time.perf_counter() - start)
        for node in request.get_profiling_info():
            ms = node.real_time.total_seconds() * 1e3
            by_type[node.node_type] += ms
            by_exec[(node.node_type, node.exec_type)] += ms

    walls_ms = np.array(walls) * 1e3
    report = {
        "threads": threads,
        "examples": n,
        "wall_ms_p50": float(np.percentile(walls_ms, 50)),
        "wall_ms_p90": float(np.percentile(walls_ms, 90)),
        "nodes_ms_per_example": sum(by_type.values()) / n,
        "by_node_type": {k: v / n for k, v in by_type.most_common()},
        "by_exec_type": [[t, e, v / n] for (t, e), v in by_exec.most_common()],
    }
    with open(os.path.join(out_dir, f"openvino_t{threads}.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wall p50 {report['wall_ms_p50']:.1f} ms, p90 {report['wall_ms_p90']:.1f} ms, "
          f"nodes {report['nodes_ms_per_example']:.1f} ms per example")
    for (t, e), v in by_exec.most_common(25):
        print(f"  {v / n:8.2f} ms  {t:28s} {e}")


if __name__ == "__main__":
    main()
