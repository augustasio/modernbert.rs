"""Turn benchmark result files into numbers with intervals.

The harness records raw samples only. This script is the one place where they become
latencies, speedups and intervals, following the Statistics section of PLAN.md.

    python python/analyze_bench.py results/bench/<label>/replicate_*.json

Each input file is one **replicate**: one harness run, in its own pair of worker
processes. The design is nested. Examples sit inside cycles, and cycles inside
replicates:

- Inside a cycle, the ruler and the variant run the same examples in the same order, so
  each example gives one paired ratio, ruler time over variant time. Above 1 means the
  variant is faster.
- Inside a replicate, cycles share warm caches and allocator state, so they are not
  independent of each other across replicates.
- Replicates are separate processes, and they are the unit of replication.

So the intervals come from a **two-stage bootstrap**: resample replicates, then resample
cycles inside each chosen replicate, then recompute the statistic. That respects both
levels of the nesting. Resampling single examples would treat correlated samples as
independent and give intervals that are too narrow.

The speedup is a geometric mean of the paired per-example ratios, because ratios
multiply: a 2x speedup and a 0.5x slowdown should average to 1, not to 1.25.

Energy, when the run sampled it, comes from `powermetrics` samples, each the CPU energy
since the one before. A sample that straddles a slice edge mixes two workers' power, so
only samples wholly inside a slice count: their mean power, times the slice's length, is
the slice's energy. That assumes steady power inside a slice, which a loop of forward
passes gives. Energy adds up, so energy per example is the total
over the total, not a mean of per-example figures, and hard rule 4 is about latency, not
this. The energy ratio is, like the speedup, a geometric mean of paired ratios, here one
per cycle, because powermetrics cannot see single examples.

Every figure is reported twice: over all measured slices, and over the slices whose
thermal state was nominal before and after. If the two disagree, the machine's
temperature moved the result.
"""

import json
import sys

import numpy as np

from stats import N_RESAMPLES, SEED, percentile_interval

# A configuration whose spread across replicates exceeds this is reported as too noisy,
# not summarized into a number (PLAN.md, Statistics).
MAX_CV = 0.05
MIN_REPLICATES = 5

# A slice that samples wholly inside it cover less than this has too little data for a
# power figure (with 100 ms samples, the two edges lose up to 0.2 s), and the analysis
# stops rather than guess. Slices for energy runs must be several seconds long.
MIN_POWER_COVERAGE = 0.5


def load(paths: list[str]) -> list[dict]:
    runs = []
    for p in paths:
        with open(p) as handle:
            runs.append(json.load(handle))
    first = runs[0]["config"]
    same = ("ruler", "variant", "threads", "cycles", "examples_per_slice")
    for path, run in zip(paths, runs, strict=True):
        if not run["valid"]:
            raise SystemExit(f"{path} is not a valid run: {run['invalid_because']}")
        for key in same:
            if run["config"][key] != first[key]:
                raise SystemExit(f"{path}: {key} is {run['config'][key]}, expected {first[key]}")
    return runs


def is_nominal(s: dict) -> bool:
    return s["thermal_before"] == "nominal" and s["thermal_after"] == "nominal"


def slice_power(power: dict, start: float, end: float) -> dict:
    """The CPU energy and mean P-cluster clock of [start, end], from the samples inside it."""
    samples = power["samples"]
    ends = np.array([p["end_seconds"] for p in samples])
    spans = np.array([p["elapsed_seconds"] for p in samples])
    energy = np.array([p["cpu_energy_mj"] for p in samples])
    p_hz = np.array([p["cluster_hz"]["P-Cluster"] for p in samples])
    inside = (ends - spans >= start) & (ends <= end)
    covered = spans[inside].sum()
    if covered < MIN_POWER_COVERAGE * (end - start):
        raise SystemExit(
            f"power samples cover {covered:.3f} s of a {end - start:.3f} s slice at {start:.3f} s"
        )
    mean_mw = energy[inside].sum() / covered
    return {
        "energy_mj": float(mean_mw * (end - start)),
        "p_cluster_hz": float((p_hz[inside] * spans[inside]).sum() / covered),
    }


def cycles_of(run: dict, nominal_only: bool) -> list[dict]:
    """Each measured cycle as {'ruler': [...], 'variant': [...]}, paired by example.

    With power measured, each cycle also holds both roles' energy per example and mean
    P-cluster clock.
    """
    power = run.get("power", {})
    measured = power.get("measured", False)
    by_cycle: dict[int, dict] = {}
    for s in run["slices"]:
        if s["warmup"]:
            continue
        by_cycle.setdefault(s["cycle"], {})[s["role"]] = s
    out = []
    for c in sorted(by_cycle):
        pair = by_cycle[c]
        if set(pair) != {"ruler", "variant"}:
            raise SystemExit(f"cycle {c} is missing a role: {sorted(pair)}")
        if pair["ruler"]["examples"] != pair["variant"]["examples"]:
            raise SystemExit(f"cycle {c}: the two slices ran different examples")
        if nominal_only and not (is_nominal(pair["ruler"]) and is_nominal(pair["variant"])):
            continue
        cycle = {
            "ruler": np.array(pair["ruler"]["worker_seconds"]),
            "variant": np.array(pair["variant"]["worker_seconds"]),
        }
        if measured:
            for role in ("ruler", "variant"):
                s = pair[role]
                p = slice_power(power, s["start_seconds"], s["end_seconds"])
                cycle[f"{role}_mj"] = p["energy_mj"] / len(s["examples"])
                cycle[f"{role}_p_hz"] = p["p_cluster_hz"]
        out.append(cycle)
    return out


def statistics(cycles_per_replicate: list[list[dict]]) -> dict:
    """The point figures, from replicates that each hold a list of cycles."""
    log_ratios, ruler, variant = [], [], []
    for cycles in cycles_per_replicate:
        for c in cycles:
            log_ratios.append(np.log(c["ruler"] / c["variant"]))
            ruler.append(c["ruler"])
            variant.append(c["variant"])
    ruler = np.concatenate(ruler)
    variant = np.concatenate(variant)
    out = {"speedup": float(np.exp(np.concatenate(log_ratios).mean()))}
    for name, values in (("ruler", ruler), ("variant", variant)):
        for q in (50, 90, 99):
            out[f"{name}_p{q}_ms"] = float(np.percentile(values, q) * 1000)

    cycles = [c for cs in cycles_per_replicate for c in cs]
    if all("ruler_mj" in c for c in cycles):
        for name in ("ruler", "variant"):
            # Every slice holds the same number of examples, so the mean of per-slice
            # energy per example is the total over the total.
            out[f"{name}_mj_per_example"] = float(np.mean([c[f"{name}_mj"] for c in cycles]))
            out[f"{name}_p_cluster_ghz"] = float(np.mean([c[f"{name}_p_hz"] for c in cycles]) / 1e9)
        ratios = [c["ruler_mj"] / c["variant_mj"] for c in cycles]
        out["energy_ratio"] = float(np.exp(np.mean(np.log(ratios))))
    return out


def two_stage_bootstrap(cycles_per_replicate: list[list[dict]]) -> dict:
    """Resample replicates, then cycles inside each, then recompute everything."""
    rng = np.random.default_rng(SEED)
    r = len(cycles_per_replicate)
    samples: dict[str, list[float]] = {}
    for _ in range(N_RESAMPLES):
        chosen = []
        for k in rng.integers(0, r, size=r):
            cycles = cycles_per_replicate[k]
            picks = rng.integers(0, len(cycles), size=len(cycles))
            chosen.append([cycles[i] for i in picks])
        for key, value in statistics(chosen).items():
            samples.setdefault(key, []).append(value)
    return {key: percentile_interval(np.array(v)) for key, v in samples.items()}


def per_replicate(cycles_per_replicate: list[list[dict]]) -> list[dict]:
    return [statistics([cycles]) for cycles in cycles_per_replicate]


def coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    a = np.array(values)
    return float(a.std(ddof=1) / a.mean())


def summarize(runs: list[dict], nominal_only: bool) -> dict:
    cycles_per_replicate = [cycles_of(run, nominal_only) for run in runs]
    kept = [c for c in cycles_per_replicate if c]
    dropped = sum(len(cycles_of(run, False)) for run in runs) - sum(len(c) for c in kept)
    if not kept:
        return {"cycles_used": 0, "note": "no cycle was nominal throughout"}

    point = statistics(kept)
    intervals = two_stage_bootstrap(kept)
    replicates = per_replicate(kept)

    figures = {}
    noisy = []
    for key, value in point.items():
        cv = coefficient_of_variation([rep[key] for rep in replicates])
        figures[key] = {
            "point": value,
            "ci95": intervals[key],
            "per_replicate": [rep[key] for rep in replicates],
            "cv_across_replicates": cv,
        }
        if cv is not None and cv > MAX_CV:
            noisy.append(key)

    return {
        "replicates_used": len(kept),
        "cycles_used": sum(len(c) for c in kept),
        "cycles_dropped": dropped,
        "figures": figures,
        "too_noisy": noisy,
    }


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    runs = load(paths)
    config = runs[0]["config"]

    report = {
        "ruler": config["ruler"],
        "variant": config["variant"],
        "threads": config["threads"],
        "replicates": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "bootstrap": {"seed": SEED, "resamples": N_RESAMPLES, "design": "two-stage"},
        "speedup_definition": "ruler time / variant time, geometric mean over paired examples",
        "energy_ratio_definition": "ruler energy / variant energy, geometric mean over cycles",
        "power_measured": all(run.get("power", {}).get("measured", False) for run in runs),
        "all_slices": summarize(runs, nominal_only=False),
        "nominal_slices_only": summarize(runs, nominal_only=True),
        "warnings": [],
    }
    if len(runs) < MIN_REPLICATES:
        report["warnings"].append(
            f"{len(runs)} replicate(s), fewer than {MIN_REPLICATES}: the interval reflects "
            "variation between cycles, not between processes, and is too narrow to publish"
        )
    if len(set(report["seeds"])) < len(runs):
        report["warnings"].append("replicates share a seed, so they ran the same examples")
    if report["all_slices"]["too_noisy"]:
        report["warnings"].append(
            f"too noisy across replicates: {report['all_slices']['too_noisy']}"
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
