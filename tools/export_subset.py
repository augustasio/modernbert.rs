"""Write a fixed random subset of the AG News prompt ids, for a rung too slow to run in full.

Rung 6, the naive matmul, takes about 40 s per example, so all 7,600 would take days. It
is checked on a seeded random sample instead. The sample is drawn once, recorded, and
reused, so the result can be reproduced and compared with the other rungs on exactly the
same examples.

    python tools/export_subset.py 400

Writes `data/agnews_subset<N>_prompt_ids.tsv` (renumbered from 0, as eval_agnews
expects), its `.sha256`, and `data/agnews_subset<N>_indices.json` mapping each line back
to its index in the full test set.
"""

import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
FULL = REPO / "data/agnews_test_prompt_ids.tsv"
SEED = 20261101  # the same seed python/stats.py uses


def main() -> None:
    n = int(sys.argv[1])
    lines = FULL.read_text().splitlines()
    picks = sorted(np.random.default_rng(SEED).choice(len(lines), size=n, replace=False))

    out = []
    for new_index, original in enumerate(picks):
        _, label, ids = lines[original].split("\t")
        out.append(f"{new_index}\t{label}\t{ids}\n")
    body = "".join(out).encode()

    stem = REPO / f"data/agnews_subset{n}"
    tsv = stem.with_name(stem.name + "_prompt_ids.tsv")
    tsv.write_bytes(body)
    tsv.with_suffix(".sha256").write_text(hashlib.sha256(body).hexdigest() + "\n")
    indices = stem.with_name(stem.name + "_indices.json")
    indices.write_text(json.dumps({"seed": SEED, "n": n, "indices": [int(i) for i in picks]}))
    print(f"wrote {n} examples to {tsv.relative_to(REPO)} and {indices.relative_to(REPO)}")


if __name__ == "__main__":
    main()
