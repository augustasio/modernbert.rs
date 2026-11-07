"""The markdown table for one memory session, from memory_run.py's files.

    python/.venv/bin/python python/memory_table.py results/memory/t4

Memory figures are nearly deterministic, so each cell is the median over the fresh
processes with the range [min, max], not an interval (PLAN.md, Statistics, Memory).
Sizes are MiB.

A process that macOS squeezed under memory pressure reads low on resident size and on
ru_maxrss. So a process whose kernel peak is more than 10% below its rung's median is
left out of the row and counted in the last column; its record stays in the file.
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_run import ORDER

MiB = 2**20
SQUEEZED = 0.9

LABELS = {
    "rung0": "0: HF pipeline (ruler)",
    "rung1": "1: HF, head at the mask only",
    "rung2": "2: + torch.compile, empty cache",
    "rung2-warm": "2: + torch.compile, warm cache",
    "rung3": "3: ONNX Runtime",
    "rung4": "4: candle, CPU matmul",
    "rung5": "5: candle + Accelerate",
    "rung6": "6: our Rust, naive matmul",
    "rung7": "7: our Rust + Accelerate",
    "rung8": "8: + fusion, no allocation",
    "rung9": "9: + f32 exp",
    "rung10": "10: + attention in sgemm",
    "rung11": "11: + threads outside matmul",
    "rung12": "12: + f32 erf, norms, softmax sum",
}


def cell(values, scale=MiB, digits=0):
    values = [v / scale for v in values]
    fmt = f"{{:.{digits}f}}"
    return (
        f"{fmt.format(statistics.median(values))} "
        f"[{fmt.format(min(values))}, {fmt.format(max(values))}]"
    )


def row(result):
    records = result["records"]
    median_peak = statistics.median(r["full"]["peak_rss_bytes"] for r in records)
    kept = [r for r in records if r["full"]["peak_rss_bytes"] >= SQUEEZED * median_peak]
    floor = [r["floor"]["peak_rss_bytes"] for r in kept]
    full = [r["full"] for r in kept]
    return [
        LABELS[result["name"]],
        cell(floor),
        cell([f["post_load_rss_bytes"] for f in full]),
        cell([f["peak_rss_bytes"] for f in full]),
        cell([f["activation_delta_bytes"] for f in full]),
        cell([f["major_faults"] for f in full], scale=1),
        cell([f["cold_start_seconds"] for f in full], scale=1, digits=2),
        f"{len(records) - len(kept)} of {len(records)}",
    ]


def main():
    folder = sys.argv[1]
    header = [
        "Rung",
        "Floor RSS",
        "Post-load RSS",
        "Peak RSS",
        "Activation delta",
        "Major faults",
        "Cold start s",
        "Squeezed, left out",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name in ORDER:
        path = os.path.join(folder, f"{name}.json")
        if os.path.exists(path):
            with open(path) as handle:
                print("| " + " | ".join(row(json.load(handle))) + " |")


if __name__ == "__main__":
    main()
