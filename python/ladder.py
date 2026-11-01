"""Shared pieces for rungs 1 to 3: the mask-only model, the runner, and the comparison
against rung 0.

Each rung script supplies one thing, a function that maps one encoded prompt to its four
verbalizer logits. Everything else is here, so the rungs differ only in that function:
the same token ids, the same content-free prompts, and the same checks against rung 0.

Correctness is measured against rung 0's raw logits (results/rung0_logits.npy), which are
the oracle. Calibration is post-processing: each rung fits it from its own content-free
logits, and the result is reported, but agreement is always measured on raw logits.
"""

import hashlib
import json
import os
import time

import numpy as np
import torch
from torch import nn

import calibration
import stats
from agnews import (
    MODEL_DIR,
    RESULTS_DIR,
    VERBALIZERS,
    build_prompt,
    find_mask_position,
    load_agnews_test,
)

# --------------------------------------------------------------------------------------
# The rung 1 model
# --------------------------------------------------------------------------------------


def verbalizer_head(hidden_at_mask, head, decoder_rows, decoder_bias):
    """The MLM head at the mask position, for the four verbalizers only.

    hidden_at_mask: (1, 1024), the final_norm output at the mask position.
    head:           ModernBertPredictionHead (dense, then GELU, then LayerNorm).
    decoder_rows:   (4, 1024), the decoder weight rows of the four verbalizer ids.
    decoder_bias:   (4,), the decoder bias entries of the same ids.

    Returns (1, 4) logits. This is what the Rust head mirrors.

    The stock head projects 1024 to 50368 at every position. Only 4 numbers at one
    position are ever read, so this computes only those.

    Traps:
      * head.dense has no bias, but the decoder does: decoder.bias is the only bias in the
        checkpoint, and dropping it shifts every logit.
      * head.act is GELU in its exact erf form, not the tanh approximation.
      * head.norm is LayerNorm with weight only, not RMSNorm: it subtracts the mean.
    """
    hidden = head.dense(hidden_at_mask)
    hidden = head.act(hidden)
    hidden = head.norm(hidden)
    return hidden @ decoder_rows.T + decoder_bias


class MaskHeadModel(nn.Module):
    """Rung 1: the stock encoder, then the head at the mask position for 4 ids only.

    forward(input_ids (1, seq), attention_mask (1, seq), mask_position (1,)) -> (1, 4)

    mask_position is a tensor rather than an int so the same module can be compiled
    (rung 2) and exported (rung 3) with the mask position as a runtime input.
    """

    def __init__(self, masked_lm, verbalizer_ids: list[int]):
        super().__init__()
        self.encoder = masked_lm.model
        self.head = masked_lm.head
        ids = torch.tensor(verbalizer_ids)
        # Trap: decoder.weight is absent from the checkpoint; it is tied to
        # tok_embeddings.weight. Slicing it picks the tied embedding rows, as the stock
        # decoder does. clone() keeps the 4 rows as a separate small buffer.
        self.register_buffer("decoder_rows", masked_lm.decoder.weight[ids].detach().clone())
        self.register_buffer("decoder_bias", masked_lm.decoder.bias[ids].detach().clone())

    def forward(self, input_ids, attention_mask, mask_position):
        # last_hidden_state is the final_norm output (ModernBertModel applies final_norm).
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        hidden_at_mask = hidden[0].index_select(0, mask_position)
        return verbalizer_head(hidden_at_mask, self.head, self.decoder_rows, self.decoder_bias)


def load_mask_head_model(verbalizer_ids: list[int]) -> MaskHeadModel:
    """The same load as rung 0: fp32 upcast from bf16, default attention (SDPA), eval."""
    from transformers import AutoModelForMaskedLM  # noqa: PLC0415

    masked_lm = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    masked_lm.eval()
    model = MaskHeadModel(masked_lm, verbalizer_ids)
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# Inputs: identical token ids for every rung
# --------------------------------------------------------------------------------------


def encode(tokenizer, prompt: str) -> dict:
    """One prompt as int64 numpy arrays, so torch and ONNX Runtime get the same ids."""
    encoded = tokenizer(prompt, return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64)
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"].astype(np.int64),
        "mask_position": np.array(
            [find_mask_position(torch.from_numpy(input_ids[0]))], dtype=np.int64
        ),
    }


def encode_test_set(tokenizer) -> tuple[list[dict], str]:
    """All 7,600 test prompts, plus a SHA-256 of their token ids.

    The hash is recorded in every rung's json. CLAUDE.md hard rule 1: every rung must
    consume byte-identical token ids, and the hash is how that is checked later.
    """
    texts, _ = load_agnews_test()
    inputs = [encode(tokenizer, build_prompt(text)) for text in texts]
    digest = hashlib.sha256()
    for item in inputs:
        digest.update(item["input_ids"].tobytes())
    return inputs, digest.hexdigest()


def encode_content_free(tokenizer) -> list[dict]:
    """The content-free prompts used to fit calibration (same strings as rung 0)."""
    return [encode(tokenizer, build_prompt(text)) for text in calibration.CONTENT_FREE_STRINGS]


# --------------------------------------------------------------------------------------
# Running a rung
# --------------------------------------------------------------------------------------


def run(score, inputs: list[dict], label: str) -> np.ndarray:
    """Apply score(item) -> 4 logits to every input. Returns (n, 4) float32.

    Prints progress only. No timing is reported: the benchmark harness times rungs under
    the interleaved protocol, and a wall clock from here would not be a measurement.
    """
    logits = np.zeros((len(inputs), len(VERBALIZERS)), dtype=np.float32)
    started = time.perf_counter()
    for i, item in enumerate(inputs):
        logits[i] = np.asarray(score(item), dtype=np.float32).reshape(-1)
        if (i + 1) % 500 == 0 or i + 1 == len(inputs):
            elapsed = time.perf_counter() - started
            print(f"  {label}: {i + 1}/{len(inputs)}  ({elapsed / 60:.1f} min)", flush=True)
    return logits


def torch_score(model):
    """Wrap a torch module as score(item) -> 4 logits, under inference_mode."""

    def score(item):
        with torch.inference_mode():
            out = model(
                torch.from_numpy(item["input_ids"]),
                torch.from_numpy(item["attention_mask"]),
                torch.from_numpy(item["mask_position"]),
            )
        return out.numpy()

    return score


# --------------------------------------------------------------------------------------
# Comparison against rung 0
# --------------------------------------------------------------------------------------


def accuracy_entry(gold, predicted, samples) -> dict:
    """Accuracy with the stats.py bootstrap interval and a Wilson interval."""
    correct = int((predicted == gold).sum())
    entry = stats.point_and_interval(float((predicted == gold).mean()), samples)
    entry["n_correct"] = correct
    entry["wilson_ci95"] = list(stats.wilson_interval(correct, len(gold)))
    return entry


def compare_with_rung0(logits: np.ndarray, content_free_logits: np.ndarray) -> dict:
    """Everything the coordinator asked for, computed from this rung's own logits."""
    with open(os.path.join(RESULTS_DIR, "rung0.json")) as handle:
        rung0 = json.load(handle)
    with open(os.path.join(RESULTS_DIR, "calibration.json")) as handle:
        rung0_calibration = json.load(handle)
    rung0_logits = np.load(os.path.join(RESULTS_DIR, "rung0_logits.npy"))

    gold = np.array(rung0["gold"], dtype=np.int64)
    n = len(gold)
    if logits.shape != rung0_logits.shape:
        raise ValueError(f"logits shape {logits.shape}, rung 0 has {rung0_logits.shape}")

    rung0_pred = rung0_logits.argmax(axis=1)
    predicted = logits.argmax(axis=1)
    agree = predicted == rung0_pred
    n_agree = int(agree.sum())

    agreement = {
        "n_agree": n_agree,
        "n": n,
        "rate": n_agree / n,
        "wilson_ci95": list(stats.wilson_interval(n_agree, n)),
        "n_disagree": n - n_agree,
    }
    if n_agree == n:
        # Rule of three: with zero events in n trials, 3/n is an approximate 95% upper
        # bound on the true event rate. Here the event is a disagreement with rung 0.
        agreement["rule_of_three_upper_bound_on_disagreement_rate"] = 3 / n

    # Disagreements are listed in full and never explained away.
    disagreements = [
        {
            "index": int(i),
            "gold": VERBALIZERS[gold[i]],
            "rung0_prediction": VERBALIZERS[rung0_pred[i]],
            "this_prediction": VERBALIZERS[predicted[i]],
            "rung0_logits": rung0_logits[i].tolist(),
            "this_logits": logits[i].tolist(),
        }
        for i in np.where(~agree)[0]
    ]

    difference = np.abs(logits.astype(np.float64) - rung0_logits.astype(np.float64)).ravel()
    logit_difference = {
        "over": f"all {n} x {len(VERBALIZERS)} verbalizer logits",
        "median": float(np.median(difference)),
        "p99": float(np.quantile(difference, 0.99)),
        "max": float(difference.max()),
    }

    # Calibration fitted from THIS rung's own content-free logits.
    params = calibration.fit(content_free_logits)
    calibrated_pred = calibration.apply(logits, params).argmax(axis=1)
    rung0_p_cf = np.array(rung0_calibration["params"]["p_cf"])
    p_cf = np.array(params["p_cf"])
    rung0_cf_logits = np.array(rung0_calibration["content_free_logits"])
    rung0_calibrated_pred = calibration.apply(rung0_logits, rung0_calibration["params"]).argmax(
        axis=1
    )

    calibration_report = {
        "params": params,
        "content_free_logits": content_free_logits.tolist(),
        "content_free_logits_max_abs_diff_vs_rung0": float(
            np.abs(content_free_logits - rung0_cf_logits).max()
        ),
        "p_cf_max_abs_diff_vs_rung0": float(np.abs(p_cf - rung0_p_cf).max()),
        "p_cf_max_rel_diff_vs_rung0": float((np.abs(p_cf - rung0_p_cf) / rung0_p_cf).max()),
        # Extra, labelled as such: agreement after calibration is NOT the correctness
        # criterion, but it shows whether a small p_cf shift moves any prediction.
        "calibrated_prediction_agreement_with_rung0_calibrated": int(
            (calibrated_pred == rung0_calibrated_pred).sum()
        ),
    }

    samples = stats.bootstrap(gold, predicted, calibrated_pred)
    return {
        "agreement_with_rung0": agreement,
        "disagreements": disagreements,
        "abs_logit_difference_vs_rung0": logit_difference,
        "raw_accuracy": accuracy_entry(gold, predicted, samples["raw"]["accuracy"]),
        "calibrated_accuracy": accuracy_entry(
            gold, calibrated_pred, samples["calibrated"]["accuracy"]
        ),
        "bootstrap": {
            "seed": stats.SEED,
            "n_resamples": stats.N_RESAMPLES,
            "method": "percentile, paired, stats.bootstrap",
        },
        "calibration": calibration_report,
        "predictions": predicted.tolist(),
    }


def write_result(rung: int, record: dict, logits: np.ndarray | None) -> None:
    """results/rungN.json, plus results/rungN_logits.npy when there are logits."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if logits is not None:
        np.save(os.path.join(RESULTS_DIR, f"rung{rung}_logits.npy"), logits)
        record["verbalizer_logits_file"] = f"rung{rung}_logits.npy"
    path = os.path.join(RESULTS_DIR, f"rung{rung}.json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {path}")


def print_summary(rung: int, comparison: dict) -> None:
    a = comparison["agreement_with_rung0"]
    d = comparison["abs_logit_difference_vs_rung0"]
    raw = comparison["raw_accuracy"]
    cal = comparison["calibrated_accuracy"]
    c = comparison["calibration"]
    print(f"\nrung {rung} against rung 0")
    print(f"  agreement      {a['n_agree']}/{a['n']}  wilson {a['wilson_ci95']}")
    if "rule_of_three_upper_bound_on_disagreement_rate" in a:
        print(f"  rule of three  disagreement rate < {3 / a['n']:.2e}")
    print(f"  |logit diff|   median {d['median']:.2e}  p99 {d['p99']:.2e}  max {d['max']:.2e}")
    print(f"  raw acc        {raw['point']:.4f}  {raw['ci95']}")
    print(f"  calibrated acc {cal['point']:.4f}  {cal['ci95']}")
    print(
        f"  p_cf vs rung 0 max abs {c['p_cf_max_abs_diff_vs_rung0']:.2e}, "
        f"max rel {c['p_cf_max_rel_diff_vs_rung0']:.2e}"
    )
