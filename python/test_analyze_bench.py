"""analyze_bench.py on data where the right answer is known in advance."""

import numpy as np

from analyze_bench import summarize


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
