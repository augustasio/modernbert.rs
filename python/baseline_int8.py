"""The int8 baselines: what PyTorch and ONNX Runtime give with their own int8 paths.

Not rungs. The ladder's int8 rungs (20 to 29) are compared with rung 0, which is fp32, so
by themselves they cannot say how much of their gain is int8 and how much is the code
around it. These two answer that. Each is rung 1's model (the head at the mask position
only, the same fp32 upcast) with its linear layers made int8 by the library's own
dynamic-quantization recipe, per-channel weights, nothing tuned by hand:

  * torch: `torch.ao.quantization.quantize_dynamic` on every `nn.Linear`, with
    `per_channel_dynamic_qconfig`. On x86 this runs on FBGEMM's VNNI kernels.
  * onnx: `onnxruntime.quantization.quantize_dynamic` on rung 3's export, QInt8
    weights, per-channel. ONNX Runtime runs it with its own int8 kernels.

Each run writes results/baseline_<name>.json and _logits.npy, scored against rung 0 the
way the rungs are (`ladder.compare_with_rung0`).

    python/.venv/bin/python python/baseline_int8.py torch
    python/.venv/bin/python python/baseline_int8.py quantize-onnx   # needs rung 3's export
    python/.venv/bin/python python/baseline_int8.py onnx
"""

import argparse
import json
import os

import numpy as np

import ladder
from agnews import MODEL_DIR, REPO_ROOT, RESULTS_DIR, resolve_verbalizer_ids

ONNX_FP32_PATH = os.path.join(REPO_ROOT, "weights", "onnx", "rung1_mask_head.onnx")
ONNX_INT8_PATH = os.path.join(REPO_ROOT, "weights", "onnx", "rung1_mask_head_int8.onnx")


def torch_int8_model(verbalizer_ids: list[int], model_dir: str = MODEL_DIR):
    """Rung 1's model with every nn.Linear dynamically quantized to int8, per channel."""
    import torch  # noqa: PLC0415
    from torch.ao.quantization import per_channel_dynamic_qconfig, quantize_dynamic  # noqa: PLC0415

    # On x86 torch picks its x86 (FBGEMM) engine by itself. On Arm it picks none, so the
    # Mac smoke test asks for QNNPACK; the measured runs are all on x86.
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "qnnpack"
    model = ladder.load_mask_head_model(verbalizer_ids, model_dir)
    return quantize_dynamic(model, {torch.nn.Linear: per_channel_dynamic_qconfig}, inplace=False)


def quantize_onnx() -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    if not os.path.exists(ONNX_FP32_PATH):
        raise SystemExit(f"no fp32 export at {ONNX_FP32_PATH}; run `rung3_onnx.py export`")
    quantize_dynamic(
        ONNX_FP32_PATH,
        ONNX_INT8_PATH,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"wrote {ONNX_INT8_PATH}")


def onnx_int8_score(path: str = ONNX_INT8_PATH, threads: int = 4):
    """score(item) -> 4 logits through an ONNX Runtime session on the int8 model."""
    import onnxruntime as ort  # noqa: PLC0415

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    names = [value.name for value in session.get_inputs()]

    def score(item):
        return session.run(None, {name: item[name] for name in names})[0]

    return score


def write(name: str, record: dict, logits: np.ndarray) -> None:
    np.save(os.path.join(RESULTS_DIR, f"baseline_{name}_logits.npy"), logits)
    record["verbalizer_logits_file"] = f"baseline_{name}_logits.npy"
    path = os.path.join(RESULTS_DIR, f"baseline_{name}.json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {path}")


def evaluate(name: str, description: str) -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    threads = int(os.environ.get("MB_THREADS", "4"))
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    verbalizer_ids = resolve_verbalizer_ids(tokenizer)
    inputs, ids_sha256 = ladder.encode_test_set(tokenizer)
    content_free_inputs = ladder.encode_content_free(tokenizer)

    if name == "torch_int8":
        score = ladder.torch_score(torch_int8_model(verbalizer_ids))
    else:
        score = onnx_int8_score(threads=threads)
    logits = ladder.run(score, inputs, name)
    content_free_logits = ladder.run(score, content_free_inputs, f"{name} content-free")

    comparison = ladder.compare_with_rung0(logits, content_free_logits)
    record = {
        "baseline": name,
        "status": "ok",
        "description": description,
        "torch_version": torch.__version__,
        "input_ids_sha256": ids_sha256,
        **comparison,
    }
    if name == "onnx_int8":
        import onnxruntime  # noqa: PLC0415

        record["onnxruntime_version"] = onnxruntime.__version__
    write(name, record, logits)
    a = comparison["agreement_with_rung0"]
    d = comparison["abs_logit_difference_vs_rung0"]
    print(
        f"{name}: agreement {a['n_agree']}/{a['n_agree'] + a['n_disagree']}, "
        f"calibrated {comparison['calibrated_accuracy']['point']:.4f}, "
        f"max |diff| {d['max']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["torch", "quantize-onnx", "onnx"])
    args = parser.parse_args()
    if args.step == "torch":
        evaluate("torch_int8", "rung 1 with every nn.Linear in PyTorch dynamic int8, per channel")
    elif args.step == "quantize-onnx":
        quantize_onnx()
    else:
        evaluate("onnx_int8", "rung 3's export in ONNX Runtime dynamic int8 (QInt8, per channel)")


if __name__ == "__main__":
    main()
