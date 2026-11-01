"""Contextual calibration of the four verbalizer logits.

Method: contextual calibration, from Zhao, Wallace, Feng, Klein and Singh (2021),
"Calibrate Before Use: Improving Few-Shot Performance of Language Models", ICML 2021,
arXiv:2102.09690.

Why: zero-shot, the model prefers `World`, `Business` and `Sports` over `Tech` by about
6 logits before it reads any article (see TECH_VERBALIZER.md). Zhao et al. measure that
preference by scoring the prompt with a content-free input, then divide it out.

The transform, in their notation, restricted to the four verbalizers:

    p      = softmax(logits)                    per example, over the 4 verbalizers
    p_cf   = mean of softmax(cf_logits)         over the content-free inputs
    q      = softmax(W p + b),  W = diag(p_cf)^-1,  b = 0

With b = 0, q is p divided elementwise by p_cf and renormalised. That is what `apply`
computes. Note p_cf is a mean of PROBABILITIES, not a mean of logits; this follows the
paper. (TECH_VERBALIZER.md probed with a plain mean of logits instead; the two agree
closely but not exactly.)

Leakage: `fit` sees only the content-free logits. It never sees a test example or a
label, so the calibrated accuracy is still a zero-shot number.

`fit` and `apply` are pure numpy functions, so a Rust rung can reproduce the transform
exactly from `results/calibration.json`, which this script writes. Rung agreement is
always measured on the raw logits; calibration is post-processing only.

Usage (loads the model once, about 1.7 GB, three forward passes):
    python/.venv/bin/python python/calibration.py
"""

import json
import os

import numpy as np

from agnews import MASK_TOKEN_ID, MODEL_DIR, RESULTS_DIR, VERBALIZERS

# The content-free inputs from Zhao et al. are "N/A", "" and "[MASK]". "[MASK]" cannot
# be used here: it would put a second [MASK] in the prompt, and the prompt would then have
# no single answer slot. "none" replaces it.
CONTENT_FREE_STRINGS = ["N/A", "", "none"]

METHOD = "contextual calibration (Zhao et al. 2021, Calibrate Before Use, arXiv:2102.09690)"


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis, in float64. Subtracting the row maximum keeps exp()
    from overflowing at logits near 25."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit(content_free_logits: np.ndarray) -> dict:
    """Fit the calibration from content-free logits only.

    content_free_logits: shape (n_content_free, 4), the four verbalizer logits at the
    mask position for each content-free prompt.

    Returns {"p_cf": [4 floats]}, the mean content-free probability of each verbalizer.
    """
    content_free_logits = np.asarray(content_free_logits, dtype=np.float64)
    if content_free_logits.ndim != 2 or content_free_logits.shape[1] != len(VERBALIZERS):
        raise ValueError(
            f"content_free_logits must have shape (n, {len(VERBALIZERS)}), "
            f"got {content_free_logits.shape}"
        )
    p_cf = softmax(content_free_logits).mean(axis=0)
    return {"p_cf": p_cf.tolist()}


def apply(logits: np.ndarray, params: dict) -> np.ndarray:
    """Calibrated probabilities q, shape (n, 4). Each row sums to 1.

    q = p / p_cf, renormalised, where p = softmax(logits). The prediction is argmax(q).
    """
    p = softmax(logits)
    p_cf = np.asarray(params["p_cf"], dtype=np.float64)
    unnormalised = p / p_cf
    return unnormalised / unnormalised.sum(axis=-1, keepdims=True)


def content_free_logits_from_model() -> np.ndarray:
    """The four verbalizer logits for each content-free prompt. Loads one model instance.

    Heavy imports are local so `fit` and `apply` can be imported without loading torch.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: PLC0415

    from agnews import build_prompt, find_mask_position, resolve_verbalizer_ids  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    verbalizer_ids = torch.tensor(resolve_verbalizer_ids(tokenizer))

    # Same load as rung 0: fp32 upcast, default attention (SDPA), eval mode, batch 1.
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    torch.set_num_threads(4)

    rows = []
    with torch.inference_mode():
        for content in CONTENT_FREE_STRINGS:
            encoded = tokenizer(build_prompt(content), return_tensors="pt")
            mask_position = find_mask_position(encoded["input_ids"][0])
            if int(encoded["input_ids"][0, mask_position]) != MASK_TOKEN_ID:
                raise ValueError("mask position does not hold [MASK]")
            row = model(**encoded).logits[0, mask_position]
            rows.append(row[verbalizer_ids].numpy())
    return np.array(rows, dtype=np.float32)


def main() -> None:
    content_free = content_free_logits_from_model()
    params = fit(content_free)

    print(f"method: {METHOD}")
    for content, row in zip(CONTENT_FREE_STRINGS, content_free, strict=True):
        print(
            f"  content {content!r:8s} logits: "
            + ", ".join(f"{n} {v:.4f}" for n, v in zip(VERBALIZERS, row, strict=True))
        )
    print(
        "  p_cf: "
        + ", ".join(f"{n} {v:.6g}" for n, v in zip(VERBALIZERS, params["p_cf"], strict=True))
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "calibration.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "method": METHOD,
                "transform": (
                    "p = softmax(logits over the 4 verbalizers); "
                    "q = (p / p_cf) / sum(p / p_cf); prediction = argmax(q). "
                    "Computed in float64."
                ),
                "fitted_on": "content-free prompts only; no test examples, no labels",
                "verbalizers": VERBALIZERS,
                "content_free_strings": CONTENT_FREE_STRINGS,
                "content_free_logits": content_free.tolist(),
                "params": params,
            },
            handle,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
