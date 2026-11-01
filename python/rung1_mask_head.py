"""Rung 1: rung 0 with the MLM head computed only where it is read.

What changes against rung 0, and nothing else:
  * The head runs at the mask position only, for the four verbalizer ids only
    (`ladder.verbalizer_head`). The stock pipeline projects 1024 to 50368 at every one of
    roughly 155 positions and then reads 4 numbers at one of them.

What does not change: the same weights, the same fp32 upcast, the same SDPA encoder, the
same token ids, batch 1, `torch.inference_mode`.

Note on PLAN.md's "eval/no-grad hygiene": rung 0 already runs `model.eval()` under
`torch.inference_mode()`, as HF's own pipelines do. So that hygiene is already in the
baseline, and the only variable between rung 0 and rung 1 is the head.

No timing here; the benchmark harness times rungs later.

Usage:
    python/.venv/bin/python python/rung1_mask_head.py
"""

import torch
from transformers import AutoTokenizer

import ladder
from agnews import MODEL_DIR, resolve_verbalizer_ids

RUNG = 1


def main() -> None:
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    verbalizer_ids = resolve_verbalizer_ids(tokenizer)
    inputs, ids_sha256 = ladder.encode_test_set(tokenizer)
    content_free_inputs = ladder.encode_content_free(tokenizer)

    model = ladder.load_mask_head_model(verbalizer_ids)
    score = ladder.torch_score(model)

    logits = ladder.run(score, inputs, f"rung {RUNG}")
    content_free_logits = ladder.run(score, content_free_inputs, f"rung {RUNG} content-free")

    comparison = ladder.compare_with_rung0(logits, content_free_logits)
    record = {
        "rung": RUNG,
        "status": "ok",
        "description": "rung 0 with the MLM head at the mask position, for the 4 ids only",
        "input_ids_sha256": ids_sha256,
        **comparison,
    }
    ladder.write_result(RUNG, record, logits)
    ladder.print_summary(RUNG, comparison)


if __name__ == "__main__":
    main()
