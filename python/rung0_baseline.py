"""Rung 0: HF transformers, stock pipeline, fp32.

The baseline, and the correctness oracle every other rung is compared against. So this
file optimises for being obviously right rather than for being fast. It runs the stock
`ModernBertForMaskedLM` forward pass with nothing removed: the MLM head still projects
1024 to 50368 at every position, even though only the mask position is read. Stripping
that is rung 1's job, and doing it here would quietly move rung 1's win into the baseline.

Batch size is 1 and nothing is padded. As the oracle this matters: padded keys are masked
with `finfo(dtype).min` rather than true `-inf`, so batch composition perturbs the logits
in the last bits, and an oracle whose answers depend on how you grouped the inputs is not
an oracle. The throughput sweep over batch sizes is the harness's job, not this file's.

Usage:
    python/.venv/bin/python python/rung0_baseline.py
    python/.venv/bin/python python/rung0_baseline.py --limit 64    # smoke test
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from agnews import (
    MODEL_DIR,
    RESULTS_DIR,
    VERBALIZERS,
    build_prompt,
    find_mask_position,
    load_agnews_test,
    resolve_verbalizer_ids,
)

# PLAN.md's sanity gate. There is no published zero-shot AG News number, so "better than
# chance by a clear margin" is the only early protection against a wrong prompt. Chance
# on four balanced classes is 25%.
CHANCE_ACCURACY = 0.25
SANITY_GATE_ACCURACY = 0.40


def load_model():
    """Stock fp32 model, eval mode.

    Trap: the checkpoint is bf16 and there is no fp32 download, so fp32 means upcasting on
    load. Every rung must upcast identically or the numerical comparison is meaningless
    (CLAUDE.md hard rule 7). `dtype=torch.float32` is the non-deprecated spelling of
    transformers' old `torch_dtype=`; it is the same upcast.
    """
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    actual = next(model.parameters()).dtype
    if actual != torch.float32:
        raise ValueError(f"expected fp32 parameters after upcast, got {actual}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="only run the first N examples (smoke test)"
    )
    parser.add_argument("--threads", type=int, default=4, help="torch intra-op threads (default 4)")
    parser.add_argument(
        "--tag",
        default=None,
        help="suffix for the output filenames, so a partial run does not overwrite the "
        "full-set baseline; --limit 64 --tag first64 writes results/rung0_first64.json",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    verbalizer_ids = resolve_verbalizer_ids(tokenizer)
    print(
        f"verbalizer ids resolved by encoding: {dict(zip(VERBALIZERS, verbalizer_ids, strict=True))}"
    )

    texts, gold = load_agnews_test()
    if args.limit is not None:
        texts, gold = texts[: args.limit], gold[: args.limit]
    print(f"AG News test examples: {len(texts)}")

    model = load_model()
    print(f"model loaded, fp32, {sum(p.numel() for p in model.parameters()):,} parameters")

    verbalizer_id_tensor = torch.tensor(verbalizer_ids)

    predictions = np.zeros(len(texts), dtype=np.int64)
    verbalizer_logits = np.zeros((len(texts), len(VERBALIZERS)), dtype=np.float32)
    token_counts = np.zeros(len(texts), dtype=np.int64)

    started = time.perf_counter()
    with torch.inference_mode():
        for i, text in enumerate(texts):
            encoded = tokenizer(build_prompt(text), return_tensors="pt")
            input_ids = encoded["input_ids"]
            mask_position = find_mask_position(input_ids[0])

            logits = model(**encoded).logits

            # Read the four verbalizer logits at the single mask position, and take the
            # argmax over only those four. An unrestricted argmax over the whole vocab can
            # return one of the 83 unused ids inside the 50368-wide padded vocabulary.
            row = logits[0, mask_position]
            picked = row[verbalizer_id_tensor]

            predictions[i] = int(torch.argmax(picked))
            verbalizer_logits[i] = picked.numpy()
            token_counts[i] = input_ids.shape[1]

            if (i + 1) % 200 == 0 or i + 1 == len(texts):
                elapsed = time.perf_counter() - started
                running = float((predictions[: i + 1] == np.array(gold[: i + 1])).mean())
                rate = (i + 1) / elapsed
                remaining = (len(texts) - i - 1) / rate
                print(
                    f"  {i + 1}/{len(texts)}  acc {running:.4f}  "
                    f"{rate:.2f} ex/s  eta {remaining / 60:.1f} min",
                    flush=True,
                )
    elapsed = time.perf_counter() - started

    gold_array = np.array(gold, dtype=np.int64)
    accuracy = float((predictions == gold_array).mean())

    # Per-class confusion, rows gold and columns predicted. PLAN.md tier 3 wants this
    # because a verbalizer-ordering bug can leave overall accuracy looking reasonable.
    confusion = np.zeros((len(VERBALIZERS), len(VERBALIZERS)), dtype=np.int64)
    for g, p in zip(gold_array, predictions, strict=True):
        confusion[g, p] += 1

    print()
    print(f"accuracy: {accuracy:.4f}  ({int((predictions == gold_array).sum())}/{len(texts)})")
    print(f"wall clock: {elapsed / 60:.1f} min for {len(texts)} examples")
    print(
        f"prompt tokens: min {token_counts.min()} max {token_counts.max()} mean {token_counts.mean():.1f}"
    )
    print()
    print("confusion matrix (rows gold, columns predicted):")
    print(f"  {'':>10}" + "".join(f"{v:>10}" for v in VERBALIZERS))
    for row_index, name in enumerate(VERBALIZERS):
        print(f"  {name:>10}" + "".join(f"{c:>10}" for c in confusion[row_index]))
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # A tagged run writes its own files, so a partial run can never overwrite the
    # full-set baseline that every other rung is compared against.
    stem = "rung0" if args.tag is None else f"rung0_{args.tag}"
    logits_name = f"{stem}_logits.npy"
    logits_path = os.path.join(RESULTS_DIR, logits_name)
    np.save(logits_path, verbalizer_logits)

    summary = {
        "rung": 0,
        "description": "HF transformers, stock pipeline, fp32 (upcast from bf16 on load)",
        "model": MODEL_DIR,
        "dataset": "fancyzhx/ag_news test split",
        "n_examples": len(texts),
        "batch_size": 1,
        "torch_threads": args.threads,
        "verbalizers": VERBALIZERS,
        "verbalizer_ids": verbalizer_ids,
        "accuracy": accuracy,
        "n_correct": int((predictions == gold_array).sum()),
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_layout": "rows gold, columns predicted, indexed by verbalizers",
        "wall_clock_seconds": elapsed,
        "prompt_tokens": {
            "min": int(token_counts.min()),
            "max": int(token_counts.max()),
            "mean": float(token_counts.mean()),
        },
        "predictions": predictions.tolist(),
        "gold": gold_array.tolist(),
        "verbalizer_logits_file": logits_name,
        "verbalizer_logits_shape": list(verbalizer_logits.shape),
        "note": (
            "No published zero-shot AG News number exists for this model; the 95.24 in "
            "the paper is a two-epoch full fine-tune. This figure is not comparable to "
            "it. The correctness criterion is agreement with this run, not this number."
        ),
    }

    summary_path = os.path.join(RESULTS_DIR, f"{stem}.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"wrote {summary_path}")
    print(f"wrote {logits_path}")

    if accuracy <= SANITY_GATE_ACCURACY:
        raise SystemExit(
            f"SANITY GATE FAILED: accuracy {accuracy:.4f} is not clearly above chance "
            f"({CHANCE_ACCURACY}). The prompt is probably wrong, or the verbalizer "
            f"logits are being read at the wrong position."
        )
    print(f"sanity gate passed: {accuracy:.4f} is clearly above chance ({CHANCE_ACCURACY})")


if __name__ == "__main__":
    main()
