"""Report a rung that ran on the fixed 400-example sample, against rung 0 on the same rows.

Rung 6, the naive matmul, is too slow for all 7,600 examples (about 30 s each), so it
runs on the seeded sample from `tools/export_subset.py`. This script checks that the ids
it read are the full set's rows at the recorded indices (hard rule 1), then compares its
logits with rung 0 and rung 7 on exactly those rows.

    python python/subset_rung.py 6

Accuracy is not reported: 400 examples give an interval about 4.4x wider than the full
set, and agreement with rung 0 is the claim this rung needs. With no disagreement, the
rule of three bounds the disagreement rate below 3/400.
"""

import hashlib
import json
import os
import sys

import numpy as np

from stats import wilson_interval

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
RESULTS = os.path.join(REPO, "results")


def check_ids(indices: list[int]) -> str:
    sample_tsv = os.path.join(DATA, "agnews_subset400_prompt_ids.tsv")
    with open(sample_tsv, "rb") as handle:
        sample = handle.read()
    with open(sample_tsv.replace(".tsv", ".sha256")) as handle:
        recorded = handle.read().strip()
    digest = hashlib.sha256(sample).hexdigest()
    if digest != recorded:
        raise SystemExit(f"{sample_tsv} has sha256 {digest}, its sidecar says {recorded}")

    with open(os.path.join(DATA, "agnews_test_prompt_ids.tsv")) as handle:
        full = [line.split("\t", 1)[1] for line in handle.read().splitlines()]
    for line, index in zip(sample.decode().splitlines(), indices, strict=True):
        if line.split("\t", 1)[1] != full[index]:
            raise SystemExit(f"sample row for example {index} differs from the full set")
    return digest


def compare(logits: np.ndarray, other: np.ndarray) -> dict:
    agree = int((logits.argmax(1) == other.argmax(1)).sum())
    diff = np.abs(logits - other).max(1)
    return {
        "agree": agree,
        "n": len(logits),
        "agreement_wilson95": list(wilson_interval(agree, len(logits))),
        "disagreement_rule_of_three_upper": 3 / len(logits) if agree == len(logits) else None,
        "max_abs_logit_diff": {
            "median": float(np.median(diff)),
            "p99": float(np.percentile(diff, 99)),
            "max": float(diff.max()),
            "argmax_example": int(np.argmax(diff)),
        },
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    rung = int(sys.argv[1])
    with open(os.path.join(DATA, "agnews_subset400_indices.json")) as handle:
        indices = json.load(handle)["indices"]
    digest = check_ids(indices)

    logits_path = os.path.join(RESULTS, f"rung{rung}_subset400_logits.npy")
    with open(logits_path.replace(".npy", ".json")) as handle:
        sidecar = json.load(handle)
    if sidecar["prompt_ids_sha256_expected"] != digest:
        raise SystemExit("the rung read a different sample than the one on disk")
    logits = np.load(logits_path)

    record = {
        "rung": rung,
        "backend": sidecar["backend"],
        "sample": {
            "n": len(indices),
            "indices_file": "data/agnews_subset400_indices.json",
            "prompt_ids_sha256": digest,
        },
        "eval_wall_clock_seconds_not_a_benchmark": sidecar["wall_clock_seconds"],
        "vs_rung0": compare(logits, np.load(os.path.join(RESULTS, "rung0_logits.npy"))[indices]),
        "vs_rung7": compare(logits, np.load(os.path.join(RESULTS, "rung7_logits.npy"))[indices]),
    }
    for key in ("vs_rung0", "vs_rung7"):
        record[key]["max_abs_logit_diff"]["argmax_example"] = indices[
            record[key]["max_abs_logit_diff"]["argmax_example"]
        ]
    out = os.path.join(RESULTS, f"rung{rung}.json")
    with open(out, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
