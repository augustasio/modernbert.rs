"""Rung 3: rung 1's graph exported to ONNX and run in ONNX Runtime on CPU.

The one variable against rung 1 is the runtime: the same MaskHeadModel, traced once by
torch.onnx.export, then executed by ONNX Runtime's CPU execution provider instead of
PyTorch eager.

Two steps, run as two separate processes, so that the torch model and the ONNX Runtime
session are never resident together (about 3.2 GB; CLAUDE.md hard rule 9):

    python/.venv/bin/python python/rung3_onnx.py export   # writes weights/onnx/
    python/.venv/bin/python python/rung3_onnx.py run      # writes results/rung3*.json/npy

The exporter is torch's default (dynamo-based) exporter, with a dynamic sequence axis.
If the export fails, the error and the opset are written to results/rung3_export.json and
the script exits non-zero. The model is never rewritten to force an export.
"""

import argparse
import json
import os
import traceback

import numpy as np
import torch

import ladder
from agnews import MODEL_DIR, REPO_ROOT, RESULTS_DIR, build_prompt, resolve_verbalizer_ids

RUNG = 3
ONNX_DIR = os.path.join(REPO_ROOT, "weights", "onnx")
ONNX_PATH = os.path.join(ONNX_DIR, "rung1_mask_head.onnx")
EXPORT_RECORD = os.path.join(RESULTS_DIR, "rung3_export.json")
INPUT_NAMES = ["input_ids", "attention_mask", "mask_position"]


def write_export_record(record: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(EXPORT_RECORD, "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {EXPORT_RECORD}")


def export() -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = ladder.load_mask_head_model(resolve_verbalizer_ids(tokenizer))

    # The example input fixes nothing but dtypes and rank: the sequence axis is dynamic.
    # A prompt longer than the 129-key local window is used so the trace takes the
    # sliding-window path, the one most likely to fight the exporter.
    example = ladder.encode(tokenizer, build_prompt("example " * 60))
    args = tuple(torch.from_numpy(example[name]) for name in INPUT_NAMES)
    seq = torch.export.Dim("seq", min=2, max=8192)
    dynamic_shapes = {"input_ids": {1: seq}, "attention_mask": {1: seq}, "mask_position": None}

    record = {
        "exporter": "torch.onnx.export(dynamo=True)",
        "torch_version": torch.__version__,
        "dynamic_shapes": "sequence axis 1 of input_ids and attention_mask, 2..8192",
        "example_seq_len": int(example["input_ids"].shape[1]),
        "path": os.path.relpath(ONNX_PATH, REPO_ROOT),
    }
    os.makedirs(ONNX_DIR, exist_ok=True)
    try:
        with torch.inference_mode(False), torch.no_grad():
            torch.onnx.export(
                model,
                args,
                ONNX_PATH,
                dynamo=True,
                input_names=INPUT_NAMES,
                output_names=["verbalizer_logits"],
                dynamic_shapes=dynamic_shapes,
                external_data=True,
            )
    except Exception as error:  # A failed export is a result: record it in full.
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)[:4000]
        record["traceback"] = traceback.format_exc()[-8000:]
        write_export_record(record)
        raise

    import onnx  # noqa: PLC0415

    exported = onnx.load(ONNX_PATH, load_external_data=False)
    record["status"] = "ok"
    record["opset"] = {entry.domain or "ai.onnx": entry.version for entry in exported.opset_import}
    record["graph_inputs"] = [value.name for value in exported.graph.input]
    record["n_nodes"] = len(exported.graph.node)
    record["files"] = {
        name: os.path.getsize(os.path.join(ONNX_DIR, name)) for name in sorted(os.listdir(ONNX_DIR))
    }
    write_export_record(record)
    print(f"exported, opset {record['opset']}, {record['n_nodes']} nodes")


def run() -> None:
    import onnxruntime as ort  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    resolve_verbalizer_ids(tokenizer)  # asserts the ids, as every rung does
    inputs, ids_sha256 = ladder.encode_test_set(tokenizer)
    content_free_inputs = ladder.encode_content_free(tokenizer)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4  # same thread count as the torch rungs
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(ONNX_PATH, options, providers=["CPUExecutionProvider"])
    session_inputs = [value.name for value in session.get_inputs()]
    print(f"session inputs: {session_inputs}, providers: {session.get_providers()}")

    def score(item):
        feed = {name: item[name] for name in session_inputs}
        return session.run(None, feed)[0]

    logits = ladder.run(score, inputs, f"rung {RUNG}")
    content_free_logits = ladder.run(score, content_free_inputs, f"rung {RUNG} content-free")

    with open(EXPORT_RECORD) as handle:
        export_record = json.load(handle)
    comparison = ladder.compare_with_rung0(logits, content_free_logits)
    record = {
        "rung": RUNG,
        "status": "ok",
        "description": "rung 1's graph exported to ONNX, run in ONNX Runtime, CPU provider",
        "onnxruntime_version": ort.__version__,
        "providers": session.get_providers(),
        "session_inputs": session_inputs,
        "graph_optimization_level": str(options.graph_optimization_level),
        "intra_op_num_threads": 4,
        "export": export_record,
        "input_ids_sha256": ids_sha256,
        **comparison,
    }
    ladder.write_result(RUNG, record, np.asarray(logits))
    ladder.print_summary(RUNG, comparison)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["export", "run"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.step == "export":
        export()
    else:
        run()


if __name__ == "__main__":
    main()
