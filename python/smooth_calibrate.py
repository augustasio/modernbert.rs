"""Rung 23's smoothing scales (SmoothQuant, Xiao et al. 2023), from the AG News *train* split.

Rung 22 left Wqkv and the MLP's Wo in bf16, because s8 rounding of their inputs moved
the logits most. Their inputs have a few channels with much larger values than the rest,
and one scale per row has to cover those, so the other channels lose their precision.
SmoothQuant moves that range into the weight: for input channel j,

    s_j = max|X_j| ** alpha / max|W_j| ** (1 - alpha),      alpha = 0.5 (the paper's default)

and the product is unchanged, X W^T = (X / s) (W s)^T, but X / s has flatter rows.

The maxima of X come from rung 0 (PyTorch, fp32) on 512 prompts drawn from the train
split with a fixed seed: never the test split that every rung is judged on, and no
labels read. The prompts are built and tokenized here with the reference template and
HF's tokenizer, which is fine offline: these ids never reach a timed run.

Output: `<weights dir>/smooth_scales.safetensors`, f32 tensors `layers.{i}.wqkv` [1024]
and `layers.{i}.mlp_wo` [2624], and a JSON record beside it.

    python python/smooth_calibrate.py weights/ModernBERT-Large-Instruct
"""

import hashlib
import json
import pathlib
import random
import sys

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from agnews import AGNEWS_REPO, build_prompt  # noqa: E402

ALPHA = 0.5
EXAMPLES = 512
SEED = 23


def main() -> None:
    model_dir = pathlib.Path(sys.argv[1])
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    train = load_dataset(AGNEWS_REPO, split="train")
    picks = random.Random(SEED).sample(range(len(train)), EXAMPLES)

    layers = model.model.layers
    seen: dict[str, torch.Tensor] = {}

    def record(name: str):
        def hook(_module, args):
            x = args[0].detach().abs().reshape(-1, args[0].shape[-1]).amax(dim=0)
            seen[name] = torch.maximum(seen[name], x) if name in seen else x

        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.attn.Wqkv.register_forward_pre_hook(record(f"layers.{i}.wqkv")))
        handles.append(layer.mlp.Wo.register_forward_pre_hook(record(f"layers.{i}.mlp_wo")))
    with torch.inference_mode():
        for n, index in enumerate(picks):
            encoded = tokenizer(build_prompt(train[index]["text"]), return_tensors="pt")
            model(**encoded)
            if (n + 1) % 128 == 0:
                print(f"  {n + 1} / {EXAMPLES}", flush=True)
    for h in handles:
        h.remove()

    scales = {}
    for i, layer in enumerate(layers):
        for name, weight in (("wqkv", layer.attn.Wqkv.weight), ("mlp_wo", layer.mlp.Wo.weight)):
            key = f"layers.{i}.{name}"
            w_max = weight.detach().abs().amax(dim=0)  # over output rows: one per input channel
            x_max = seen[key]
            s = (x_max.clamp(min=1e-5) ** ALPHA) / (w_max.clamp(min=1e-5) ** (1 - ALPHA))
            scales[key] = s.float().contiguous()

    out = model_dir / "smooth_scales.safetensors"
    save_file(scales, str(out))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    spread = {k: float(v.max() / v.min()) for k, v in scales.items()}
    record_json = {
        "alpha": ALPHA,
        "examples": EXAMPLES,
        "seed": SEED,
        "split": "train",
        "train_indices_first_10": picks[:10],
        "sha256": digest,
        "widest_scale_spread": max(spread.values()),
    }
    (model_dir / "smooth_scales.json").write_text(json.dumps(record_json, indent=2) + "\n")
    print(f"wrote {out} ({digest[:12]}), widest max/min scale ratio {record_json['widest_scale_spread']:.1f}")


if __name__ == "__main__":
    main()
