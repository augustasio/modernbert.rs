"""analyze_bench.py on data where the right answer is known in advance."""

import numpy as np
import pytest

from analyze_bench import slice_power, summarize


def make_run(seed: int, variant_factor: float, noise: float, thermal: str = "nominal") -> dict:
    """One fake replicate: 30 cycles of 20 paired examples.

    Each example has its own base cost, shared by both roles, as real examples do (a long
    prompt is slow on every rung). The variant takes `variant_factor` of the ruler's time,
    times independent log-normal noise of size `noise`.
    """
    rng = np.random.default_rng(seed)
    slices = []
    for c in range(30):
        base = rng.uniform(0.3, 0.7, size=20)
        examples = list(range(c * 20, c * 20 + 20))
        for role, factor in (("ruler", 1.0), ("variant", variant_factor)):
            t = base * factor * np.exp(rng.normal(0.0, noise, size=20))
            slices.append(
                {
                    "cycle": c,
                    "warmup": False,
                    "role": role,
                    "examples": examples,
                    "worker_seconds": t.tolist(),
                    "thermal_before": thermal,
                    "thermal_after": thermal,
                }
            )
    return {"valid": True, "config": {"seed": seed}, "slices": slices}


def test_a_known_two_times_speedup_is_recovered():
    runs = [make_run(s, 0.5, 0.02) for s in range(5)]
    speedup = summarize(runs, nominal_only=False)["figures"]["speedup"]
    assert abs(speedup["point"] - 2.0) < 0.01
    low, high = speedup["ci95"]
    assert low < 2.0 < high
    assert high - low < 0.02, "with 2% noise and 3,000 pairs the interval should be tight"


def test_an_aa_comparison_has_an_interval_containing_one():
    runs = [make_run(s, 1.0, 0.05) for s in range(5)]
    low, high = summarize(runs, nominal_only=False)["figures"]["speedup"]["ci95"]
    assert low < 1.0 < high


def test_pairing_removes_the_spread_between_examples():
    # Examples differ by more than 2x in cost, but the ratio of each pair does not,
    # so the speedup interval stays far narrower than the spread of latencies.
    runs = [make_run(s, 0.5, 0.01) for s in range(5)]
    figures = summarize(runs, nominal_only=False)["figures"]
    low, high = figures["speedup"]["ci95"]
    p50 = figures["ruler_p50_ms"]["point"]
    p99 = figures["ruler_p99_ms"]["point"]
    assert (high - low) / 2.0 < 0.01
    assert p99 / p50 > 1.2


def test_hot_slices_are_dropped_from_the_nominal_view():
    runs = [make_run(s, 0.5, 0.02) for s in range(4)] + [make_run(9, 0.5, 0.02, "serious")]
    nominal = summarize(runs, nominal_only=True)
    assert nominal["replicates_used"] == 4
    assert nominal["cycles_dropped"] == 30


def with_power(run: dict, watts: dict[str, float], interval: float = 0.1) -> dict:
    """Lay the slices end to end on one clock and sample a constant power per role.

    Each slice's span is the sum of its worker times, and the sampler sees `watts[role]`
    while that role runs. Samples fall on a fixed grid, so most straddle a slice boundary,
    which is what the overlap attribution has to handle.
    """
    t = 0.0
    bounds = []
    for s in run["slices"]:
        s["start_seconds"] = t
        t += sum(s["worker_seconds"])
        s["end_seconds"] = t
        bounds.append((s["start_seconds"], t, watts[s["role"]]))

    samples = []
    end = interval
    while end < t + interval:
        start = end - interval
        mj = sum(w * max(0.0, min(end, b) - max(start, a)) * 1000 for a, b, w in bounds)
        samples.append(
            {
                "end_seconds": end,
                "elapsed_seconds": interval,
                "cpu_energy_mj": mj,
                "combined_power_mw": 0.0,
                "cluster_hz": {"P-Cluster": 4.0e9},
            }
        )
        end += interval
    run["power"] = {"measured": True, "samples": samples}
    return run


def test_slice_power_uses_only_samples_inside_the_slice():
    def sample(end, mj, hz):
        return {
            "end_seconds": end,
            "elapsed_seconds": 1.0,
            "cpu_energy_mj": mj,
            "cluster_hz": {"P-Cluster": hz},
        }

    # The first and last samples straddle the edges and belong partly to other slices,
    # so the 1000 mJ and 9000 mJ must not leak in.
    power = {
        "samples": [
            sample(1.0, 1000.0, 1e9),
            sample(2.0, 100.0, 4e9),
            sample(3.0, 300.0, 2e9),
            sample(4.0, 9000.0, 1e9),
        ]
    }
    p = slice_power(power, 0.5, 3.5)
    # Mean power inside is 200 mJ/s, over the slice's 3 s.
    assert abs(p["energy_mj"] - 600.0) < 1e-9
    assert abs(p["p_cluster_hz"] - 3e9) < 1.0


def test_a_known_energy_ratio_is_recovered():
    # The variant is twice as fast at 1.5x the power, so it uses 1.5 / 2 = 0.75 of the
    # ruler's energy per example, and the ruler / variant ratio is 1 / 0.75.
    runs = [with_power(make_run(s, 0.5, 0.02), {"ruler": 4.0, "variant": 6.0}) for s in range(5)]
    figures = summarize(runs, nominal_only=False)["figures"]
    ratio = figures["energy_ratio"]
    assert abs(ratio["point"] - 1 / 0.75) < 0.02
    low, high = ratio["ci95"]
    assert low < 1 / 0.75 < high
    assert abs(figures["variant_p_cluster_ghz"]["point"] - 4.0) < 1e-9


def test_a_gap_in_the_power_samples_stops_the_analysis():
    run = with_power(make_run(0, 0.5, 0.02), {"ruler": 4.0, "variant": 4.0})
    # The sampler died as the last slice began, so that slice has no samples at all.
    last_start = run["slices"][-1]["start_seconds"]
    samples = run["power"]["samples"]
    run["power"]["samples"] = [p for p in samples if p["end_seconds"] <= last_start]
    with pytest.raises(SystemExit, match="power samples cover"):
        summarize([run], nominal_only=False)


def test_slices_too_short_for_energy_report_no_energy_and_say_so():
    run = with_power(make_run(0, 0.5, 0.02), {"ruler": 4.0, "variant": 4.0})
    for s in run["slices"]:
        s["end_seconds"] = s["start_seconds"] + 0.3
    summary = summarize([run], nominal_only=False)
    assert "energy_ratio" not in summary["figures"]
    assert summary["energy"].startswith("not reported")
