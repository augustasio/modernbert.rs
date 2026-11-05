"""Print one session's analyses as a Markdown table, so no figure is typed by hand.

    python python/bench_table.py results/bench/t4

Reads <session>/<name>/analysis.json for the A/A floor and every rung. Speedup is rung 0
time over the variant's time, so above 1 is faster than rung 0. A speedup whose interval
overlaps the A/A interval is "no detectable difference" (PLAN.md, The A/A floor).
"""

import json
import os
import sys

ROWS = [
    ("aa", "A/A: rung 0 against itself"),
    ("rung1", "1: HF, head at the mask only"),
    ("rung2", "2: + torch.compile"),
    ("rung3", "3: ONNX Runtime"),
    ("rung4", "4: candle, CPU matmul"),
    ("rung5", "5: candle + Accelerate"),
    ("rung6", "6: our Rust, naive matmul"),
    ("rung7", "7: our Rust + Accelerate"),
    ("rung8", "8: + fusion, no allocation"),
    ("rung9", "9: + f32 exp"),
    ("rung10", "10: + attention in sgemm"),
    ("rung11", "11: + threads outside matmul"),
]


def interval(figure: dict | None, digits: int = 3) -> str:
    if figure is None:
        return "n/a"
    low, high = figure["ci95"]
    return f"{figure['point']:.{digits}g} [{low:.{digits}g}, {high:.{digits}g}]"


def main() -> None:
    session = sys.argv[1]
    analyses = {}
    for name, _ in ROWS:
        path = os.path.join(session, name, "analysis.json")
        if os.path.exists(path):  # a rung added after this session ran
            with open(path) as handle:
                analyses[name] = json.load(handle)
    floor = analyses["aa"]["all_slices"]["figures"]["speedup"]["ci95"]

    print(
        "| Rung | Speedup vs rung 0 | Variant p50 ms | Variant p99 ms | Energy ratio "
        "| P-cluster GHz | Speedup, nominal slices only (cycles) |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, label in ROWS:
        if name not in analyses:
            continue
        a = analyses[name]
        f = a["all_slices"]["figures"]
        nominal = a["nominal_slices_only"]
        low, high = f["speedup"]["ci95"]
        speedup = interval(f["speedup"])
        if name != "aa" and low <= floor[1] and high >= floor[0]:
            speedup += " (no detectable difference)"
        nominal_speedup = (
            f"{interval(nominal['figures']['speedup'])} ({nominal['cycles_used']})"
            if nominal.get("figures")
            else "none nominal"
        )
        print(
            f"| {label} | {speedup} | {interval(f['variant_p50_ms'])} "
            f"| {interval(f['variant_p99_ms'])} | {interval(f.get('energy_ratio'))} "
            f"| {interval(f.get('variant_p_cluster_ghz'))} | {nominal_speedup} |"
        )


if __name__ == "__main__":
    main()
