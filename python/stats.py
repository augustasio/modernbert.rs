"""Confidence intervals and the raw-against-calibrated comparison for rung 0.

Reads saved files only, and never loads the model:
    results/rung0.json         gold labels
    results/rung0_logits.npy   raw verbalizer logits, shape (7600, 4)
    results/calibration.json   content-free calibration params (calibration.py)

Writes results/rung0_stats.json. Method notes are in STATISTICS.md.

Every interval is 95%. Bootstrap intervals use the percentile method, 10,000 resamples
of the 7,600 examples with replacement, and a fixed seed. Raw and calibrated metrics are
computed on the SAME resampled indices each time, so the difference between them is a
paired bootstrap.

Usage:
    python/.venv/bin/python python/stats.py
"""

import json
import math
import os

import numpy as np

import calibration
from agnews import RESULTS_DIR, VERBALIZERS

SEED = 20261101
N_RESAMPLES = 10_000
CONFIDENCE = 0.95
Z_95 = 1.959963984540054  # two-sided 95% normal quantile, used by the Wilson interval

# Resamples are drawn in batches so that the index matrix stays small: 500 x 7600 int32
# is 15 MB, while all 10,000 resamples at once would be 300 MB.
BATCH = 500

N_CLASSES = len(VERBALIZERS)


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. A closed-form cross-check on the
    bootstrap accuracy interval."""
    p = successes / n
    denominator = 1 + Z_95**2 / n
    centre = (p + Z_95**2 / (2 * n)) / denominator
    half_width = Z_95 * math.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n * n)) / denominator
    return centre - half_width, centre + half_width


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs.

    b = examples raw gets right and calibrated gets wrong; c = the reverse. Under the null
    hypothesis that the two decision rules are equally accurate, each discordant pair is a
    fair coin, so min(b, c) follows Binomial(b + c, 1/2). Python integers make the tail
    sum exact.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def metrics_for_batch(gold: np.ndarray, predicted: np.ndarray) -> dict:
    """Accuracy, per-class recall and macro-F1, for each row of a batch of resamples.

    gold, predicted: shape (batch, n). Each row is one resample.

    F1 for a class that is never predicted has precision 0/0. It is set to 0, as in
    scikit-learn's zero_division=0. That case is real here: raw rung 0 never predicts
    Tech.
    """
    accuracy = (gold == predicted).mean(axis=1)
    recalls = np.zeros((gold.shape[0], N_CLASSES))
    f1s = np.zeros((gold.shape[0], N_CLASSES))
    for k in range(N_CLASSES):
        true_positive = ((gold == k) & (predicted == k)).sum(axis=1)
        actual = (gold == k).sum(axis=1)
        predicted_k = (predicted == k).sum(axis=1)
        recall = np.divide(true_positive, actual, out=np.zeros(len(actual)), where=actual > 0)
        precision = np.divide(
            true_positive, predicted_k, out=np.zeros(len(actual)), where=predicted_k > 0
        )
        denominator = precision + recall
        f1 = np.divide(
            2 * precision * recall, denominator, out=np.zeros(len(actual)), where=denominator > 0
        )
        recalls[:, k] = recall
        f1s[:, k] = f1
    return {"accuracy": accuracy, "recall": recalls, "macro_f1": f1s.mean(axis=1)}


def percentile_interval(samples: np.ndarray) -> list[float]:
    """Percentile bootstrap interval: the 2.5th and 97.5th percentiles."""
    alpha = (1 - CONFIDENCE) / 2
    return [float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))]


def bootstrap(gold: np.ndarray, raw_pred: np.ndarray, cal_pred: np.ndarray) -> dict:
    """10,000 paired resamples. Each resample scores raw and calibrated on the same
    indices, so the accuracy difference keeps the pairing."""
    rng = np.random.default_rng(SEED)
    n = len(gold)
    collected = {
        "raw": {"accuracy": [], "recall": [], "macro_f1": []},
        "calibrated": {"accuracy": [], "recall": [], "macro_f1": []},
    }
    for start in range(0, N_RESAMPLES, BATCH):
        size = min(BATCH, N_RESAMPLES - start)
        indices = rng.integers(0, n, size=(size, n), dtype=np.int32)
        gold_sample = gold[indices]
        for name, predicted in (("raw", raw_pred), ("calibrated", cal_pred)):
            batch = metrics_for_batch(gold_sample, predicted[indices])
            for key, values in batch.items():
                collected[name][key].append(values)
    return {
        name: {key: np.concatenate(parts) for key, parts in metrics.items()}
        for name, metrics in collected.items()
    }


def point_and_interval(point: float, samples: np.ndarray) -> dict:
    low, high = percentile_interval(samples)
    return {"point": point, "ci95": [low, high], "half_width": (high - low) / 2}


def recall_entry(k: int, gold, predicted, point: dict, samples: dict) -> dict:
    """Recall for class k: bootstrap interval plus a Wilson interval.

    The Wilson interval matters when recall is 0. A bootstrap cannot resample successes
    that do not exist, so it returns the degenerate interval [0, 0]. Wilson still gives
    an upper bound. Raw Tech recall is exactly this case.
    """
    entry = point_and_interval(float(point["recall"][0, k]), samples["recall"][:, k])
    in_class = gold == k
    entry["n_class"] = int(in_class.sum())
    entry["wilson_ci95"] = list(
        wilson_interval(int((predicted[in_class] == k).sum()), int(in_class.sum()))
    )
    return entry


def main() -> None:
    with open(os.path.join(RESULTS_DIR, "rung0.json")) as handle:
        rung0 = json.load(handle)
    with open(os.path.join(RESULTS_DIR, "calibration.json")) as handle:
        calibration_record = json.load(handle)
    logits = np.load(os.path.join(RESULTS_DIR, "rung0_logits.npy"))

    gold = np.array(rung0["gold"], dtype=np.int64)
    raw_pred = logits.argmax(axis=1)
    if not np.array_equal(raw_pred, np.array(rung0["predictions"])):
        raise ValueError("argmax of rung0_logits.npy disagrees with rung0.json predictions")
    cal_pred = calibration.apply(logits, calibration_record["params"]).argmax(axis=1)
    n = len(gold)

    # Point estimates on the full set, with the same function the bootstrap uses.
    point = {
        "raw": metrics_for_batch(gold[None, :], raw_pred[None, :]),
        "calibrated": metrics_for_batch(gold[None, :], cal_pred[None, :]),
    }
    samples = bootstrap(gold, raw_pred, cal_pred)

    report = {"n": n, "seed": SEED, "n_resamples": N_RESAMPLES, "confidence": CONFIDENCE}
    report["bootstrap_method"] = "percentile, resampling examples with replacement, paired"
    report["calibration"] = calibration_record["method"]

    for name, predicted in (("raw", raw_pred), ("calibrated", cal_pred)):
        correct = int((predicted == gold).sum())
        wilson_low, wilson_high = wilson_interval(correct, n)
        accuracy = point_and_interval(float(point[name]["accuracy"][0]), samples[name]["accuracy"])
        accuracy["n_correct"] = correct
        accuracy["wilson_ci95"] = [wilson_low, wilson_high]
        report[name] = {
            "accuracy": accuracy,
            "macro_f1": point_and_interval(
                float(point[name]["macro_f1"][0]), samples[name]["macro_f1"]
            ),
            "recall": {
                VERBALIZERS[k]: recall_entry(k, gold, predicted, point[name], samples[name])
                for k in range(N_CLASSES)
            },
            "predicted_counts": {
                VERBALIZERS[k]: int((predicted == k).sum()) for k in range(N_CLASSES)
            },
        }

    raw_correct = raw_pred == gold
    cal_correct = cal_pred == gold
    b = int((raw_correct & ~cal_correct).sum())
    c = int((~raw_correct & cal_correct).sum())
    difference = samples["calibrated"]["accuracy"] - samples["raw"]["accuracy"]
    report["calibrated_minus_raw"] = {
        "accuracy_difference": point_and_interval(
            float(cal_correct.mean() - raw_correct.mean()), difference
        ),
        "mcnemar_exact": {
            "raw_right_calibrated_wrong": b,
            "raw_wrong_calibrated_right": c,
            "both_right": int((raw_correct & cal_correct).sum()),
            "both_wrong": int((~raw_correct & ~cal_correct).sum()),
            "p_value_two_sided": exact_mcnemar_p(b, c),
        },
    }

    path = os.path.join(RESULTS_DIR, "rung0_stats.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print_report(report)
    print(f"\nwrote {path}")


def pct(x: float) -> str:
    return f"{100 * x:6.2f}"


def print_report(report: dict) -> None:
    print(f"n = {report['n']}, {report['n_resamples']} paired resamples, seed {report['seed']}")
    for name in ("raw", "calibrated"):
        r = report[name]
        a = r["accuracy"]
        print(f"\n{name}")
        print(
            f"  accuracy  {pct(a['point'])}%  bootstrap [{pct(a['ci95'][0])}, {pct(a['ci95'][1])}]"
            f"  wilson [{pct(a['wilson_ci95'][0])}, {pct(a['wilson_ci95'][1])}]"
            f"  (+/-{100 * a['half_width']:.2f} pp)"
        )
        f = r["macro_f1"]
        print(
            f"  macro-F1  {pct(f['point'])}%  bootstrap [{pct(f['ci95'][0])}, {pct(f['ci95'][1])}]"
            f"  (+/-{100 * f['half_width']:.2f} pp)"
        )
        for cls, rec in r["recall"].items():
            print(
                f"  recall {cls:9s} {pct(rec['point'])}%  [{pct(rec['ci95'][0])}, "
                f"{pct(rec['ci95'][1])}]  (+/-{100 * rec['half_width']:.2f} pp)"
                f"  wilson [{pct(rec['wilson_ci95'][0])}, {pct(rec['wilson_ci95'][1])}]"
            )
        print(f"  predicted counts {r['predicted_counts']}")
    d = report["calibrated_minus_raw"]
    diff = d["accuracy_difference"]
    m = d["mcnemar_exact"]
    print("\ncalibrated minus raw")
    print(
        f"  accuracy difference {100 * diff['point']:+.2f} pp  paired bootstrap "
        f"[{100 * diff['ci95'][0]:+.2f}, {100 * diff['ci95'][1]:+.2f}]"
    )
    print(
        f"  McNemar exact: b (raw right, cal wrong) = {m['raw_right_calibrated_wrong']}, "
        f"c (raw wrong, cal right) = {m['raw_wrong_calibrated_right']}, "
        f"p = {m['p_value_two_sided']:.3g}"
    )


if __name__ == "__main__":
    main()
