"""Report a Rust rung against rung 0, in the same format as the Python rungs.

The Rust binary `eval_agnews` scores all 7,600 examples and writes the four verbalizer
logits per example. This script does the rest, so every rung is compared by the same
code in `ladder.compare_with_rung0`:

1. **Hard rule 1.** It checks that the token ids the Rust binary read are the ones the
   Python rungs feed their models, element by element, and that their SHA-256, computed
   the way `ladder.encode_test_set` computes it, matches the digest recorded in
   `results/rung1.json`.
2. **Calibration.** It runs the three content-free prompts through the same Rust binary,
   so the rung's calibration is fitted from its own logits, as for every other rung.
3. **The comparison.** Agreement, the logit-difference distribution, and raw and
   calibrated accuracy with intervals, written to `results/rungN.json`.

    python python/rust_rung.py 7 target/eval/eval_agnews

The binary must be built with the features of the rung named, for example
`cargo build --profile eval -p mb-model --features accelerate --bin eval_agnews` for
rung 7. Its sidecar `.json` records the backend, and the script refuses a mismatch.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from transformers import AutoTokenizer

import ladder

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS_TSV = os.path.join(REPO, "data", "agnews_test_prompt_ids.tsv")
CONTENT_FREE_TSV = os.path.join(REPO, "data", "content_free_prompt_ids.tsv")
WEIGHTS = os.path.join(REPO, "weights", "ModernBERT-Large-Instruct")

EXPECTED_BACKEND = {
    4: "candle-cpu",
    5: "candle-accelerate",
    6: "naive",
    7: "accelerate",
    8: "accelerate-fused",
    9: "accelerate-fused-expf",
}
DESCRIPTION = {
    4: "candle-transformers encoder with a mask-only head, candle CPU (gemm) matmul",
    5: "candle-transformers encoder with a mask-only head, candle Accelerate matmul",
    6: "hand-written Rust forward pass, naive matmul",
    7: "hand-written Rust forward pass, Accelerate matmul",
    8: "hand-written Rust forward pass, Accelerate matmul, fused and allocation-free",
    9: "rung 8 with f32 polynomial exp in the softmax and erf, instead of f64 libm exp",
}


def read_tsv_ids(path: str) -> list[np.ndarray]:
    rows = []
    with open(path) as handle:
        for line in handle:
            ids = line.rstrip("\n").split("\t")[2]
            rows.append(np.array([int(t) for t in ids.split(" ")], dtype=np.int64))
    return rows


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def check_inputs(tokenizer) -> str:
    """Hard rule 1: the Rust rung read exactly the ids the Python rungs use."""
    with open(IDS_TSV.replace(".tsv", ".sha256")) as handle:
        recorded = handle.read().strip()
    actual = sha256_file(IDS_TSV)
    if actual != recorded:
        raise SystemExit(f"{IDS_TSV} has sha256 {actual}, its sidecar says {recorded}")

    python_inputs, python_digest = ladder.encode_test_set(tokenizer)
    rust_ids = read_tsv_ids(IDS_TSV)
    if len(rust_ids) != len(python_inputs):
        raise SystemExit(f"{len(rust_ids)} examples in the tsv, {len(python_inputs)} in Python")
    digest = hashlib.sha256()
    for i, (theirs, ours) in enumerate(zip(python_inputs, rust_ids, strict=True)):
        if not np.array_equal(theirs["input_ids"][0], ours):
            raise SystemExit(f"example {i}: the Rust ids differ from the Python ids")
        digest.update(ours.reshape(1, -1).tobytes())

    with open(os.path.join(ladder.RESULTS_DIR, "rung1.json")) as handle:
        rung1_digest = json.load(handle)["input_ids_sha256"]
    if digest.hexdigest() != rung1_digest or python_digest != rung1_digest:
        raise SystemExit("the ids digest does not match the one rung 1 recorded")
    return digest.hexdigest()


def content_free_logits(binary: str, tokenizer) -> np.ndarray:
    """Run the content-free prompts through the Rust binary itself."""
    items = ladder.encode_content_free(tokenizer)
    lines = "".join(
        f"{i}\tcontent-free\t{' '.join(str(t) for t in item['input_ids'][0])}\n"
        for i, item in enumerate(items)
    )
    with open(CONTENT_FREE_TSV, "w") as handle:
        handle.write(lines)
    with open(CONTENT_FREE_TSV.replace(".tsv", ".sha256"), "w") as handle:
        handle.write(hashlib.sha256(lines.encode()).hexdigest() + "\n")

    out = os.path.join(REPO, "data", "content_free_logits.npy")
    subprocess.run([binary, WEIGHTS, CONTENT_FREE_TSV, out], check=True)
    return np.load(out)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    rung, binary = int(sys.argv[1]), sys.argv[2]

    logits_path = os.path.join(ladder.RESULTS_DIR, f"rung{rung}_logits.npy")
    with open(logits_path.replace(".npy", ".json")) as handle:
        sidecar = json.load(handle)
    if sidecar["backend"] != EXPECTED_BACKEND[rung]:
        raise SystemExit(
            f"rung {rung} needs the {EXPECTED_BACKEND[rung]} backend, the logits came from "
            f"{sidecar['backend']}"
        )

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS)
    ids_digest = check_inputs(tokenizer)
    logits = np.load(logits_path)
    comparison = ladder.compare_with_rung0(logits, content_free_logits(binary, tokenizer))

    record = {
        "rung": rung,
        "description": DESCRIPTION[rung],
        "status": "ok",
        "backend": sidecar["backend"],
        "input_ids_sha256": ids_digest,
        "eval_wall_clock_seconds_not_a_benchmark": sidecar["wall_clock_seconds"],
        **comparison,
    }
    ladder.write_result(rung, record, logits)
    ladder.print_summary(rung, comparison)


if __name__ == "__main__":
    main()
