"""The benchmark worker for the Python rungs 0 to 3. See crates/mb-harness/PROTOCOL.md.

This mirrors crates/mb-model/src/bin/worker.rs. The harness starts it, it loads one rung,
prints one `ready` line, then answers `run` requests on stdin until `quit`.

    python/.venv/bin/python python/worker.py --rung {0,1,2,3}

Rules from the protocol, and where they are kept here:
  * Thread counts are explicit: MB_THREADS goes to torch.set_num_threads, and for rung 3
    to ONNX Runtime's intra-op threads.
  * The worker never tokenizes. The ids arrive in the request (hard rule 1).
  * `seconds` covers the forward pass alone, from the built tensors to the four logits,
    measured with time.perf_counter. Parsing and replying are outside it.
  * stdout carries protocol lines only. Everything else goes to stderr; see
    `protocol_stdout`.
  * `backend` is the BLAS torch reports for rungs 0 to 2, and `onnxruntime-cpu` for rung 3.
"""

import argparse
import json
import os
import re
import sys
import time

# The verbalizer ids in rung 0's order: Business, World, Sports, Tech. The worker never
# tokenizes, so it cannot resolve them by encoding. These are the values agnews.py
# resolves by encoding and asserts, and the Rust worker uses the same four.
VERBALIZER_IDS = [10518, 3645, 15000, 18447]
MASK_TOKEN_ID = 50284

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(REPO_ROOT, "weights", "onnx", "rung1_mask_head.onnx")
ONNX_EXPORT_RECORD = os.path.join(REPO_ROOT, "results", "rung3_export.json")

# Rung 2 compiles during start-up, before `ready`, so compile time stays out of the timed
# slices. One warm-up length on each side of the 129-key local window, because real
# prompts (119 to 348 tokens) fall on both sides. The values are synthetic: inductor
# compiles for shapes and dtypes, not for token values.
RUNG2_WARMUP_LENGTHS = [120, 300]
WARMUP_FILLER_ID = 3645
CLS_ID, SEP_ID = 50281, 50282


def protocol_stdout():
    """Reserve the real stdout for protocol lines and send everything else to stderr.

    Libraries print: transformers warnings, inductor's compiler, progress bars. Pointing
    Python's sys.stdout at stderr is not enough, because a subprocess or C code can write
    to file descriptor 1 directly. So fd 1 is duplicated for the protocol, and then fd 1
    itself is pointed at stderr.
    """
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "w", buffering=1)


PROTOCOL = protocol_stdout()


def send(message: dict) -> None:
    PROTOCOL.write(json.dumps(message) + "\n")
    PROTOCOL.flush()


def fail(message: str):
    """Report an error the way the protocol asks, then exit non-zero."""
    send({"error": message})
    print(f"worker: {message}", file=sys.stderr)
    sys.exit(1)


def torch_backend_name() -> str:
    """The BLAS torch reports, e.g. BLAS_INFO=accelerate gives "accelerate"."""
    import torch  # noqa: PLC0415

    match = re.search(r"BLAS_INFO=(\w+)", torch.__config__.show())
    return match.group(1) if match else "unknown"


# --------------------------------------------------------------------------------------
# One builder per rung. Each returns forward(ids, mask) -> (seconds, 4 logits).
# --------------------------------------------------------------------------------------


def build_rung0(weights: str):
    """The stock pipeline, scored exactly as rung0_baseline.py scores one example: the
    full MLM head over every position, then the four ids read at the mask."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForMaskedLM  # noqa: PLC0415

    model = AutoModelForMaskedLM.from_pretrained(weights, dtype=torch.float32)
    model.eval()
    verbalizer_ids = torch.tensor(VERBALIZER_IDS)

    def forward(ids, mask):
        input_ids = torch.tensor([ids], dtype=torch.int64)
        # All ones, as the tokenizer gives for one unpadded sequence in rung 0.
        attention_mask = torch.ones_like(input_ids)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            picked = logits[0, mask][verbalizer_ids]
        seconds = time.perf_counter() - started
        return seconds, picked.tolist()

    return forward


def torch_forward(model):
    """forward(ids, mask) for a MaskHeadModel, compiled or not."""
    import torch  # noqa: PLC0415

    def forward(ids, mask):
        input_ids = torch.tensor([ids], dtype=torch.int64)
        attention_mask = torch.ones_like(input_ids)
        mask_position = torch.tensor([mask], dtype=torch.int64)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids, attention_mask, mask_position)
        seconds = time.perf_counter() - started
        return seconds, logits.reshape(-1).tolist()

    return forward


def build_rung1(weights: str):
    """MaskHeadModel from ladder.py: the head at the mask position, for 4 ids only."""
    import ladder  # noqa: PLC0415

    return torch_forward(ladder.load_mask_head_model(VERBALIZER_IDS, weights))


def build_rung2(weights: str):
    """Rung 1 under torch.compile, configured exactly as rung2_compile.py configures it.

    Returns (forward, compile_seconds). The compile happens here, before `ready`.
    """
    import torch  # noqa: PLC0415

    import ladder  # noqa: PLC0415

    model = ladder.load_mask_head_model(VERBALIZER_IDS, weights)
    compiled = torch.compile(model, backend="inductor", dynamic=True)
    forward = torch_forward(compiled)

    started = time.perf_counter()
    for length in RUNG2_WARMUP_LENGTHS:
        ids = [CLS_ID] + [WARMUP_FILLER_ID] * (length - 3) + [MASK_TOKEN_ID, SEP_ID]
        forward(ids, length - 2)
    compile_seconds = time.perf_counter() - started
    return forward, compile_seconds


def build_rung3(threads: int):
    """The ONNX Runtime session, configured exactly as `rung3_onnx.py run` configures it,
    except that intra-op threads come from MB_THREADS."""
    if os.path.exists(ONNX_EXPORT_RECORD):
        with open(ONNX_EXPORT_RECORD) as handle:
            export_record = json.load(handle)
        if export_record.get("status") != "ok":
            fail(f"rung 3 export failed: {export_record.get('error', 'no error recorded')}")
    if not os.path.exists(ONNX_PATH):
        fail(f"rung 3: no ONNX model at {ONNX_PATH}; run `rung3_onnx.py export` first")

    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(ONNX_PATH, options, providers=["CPUExecutionProvider"])
    input_names = [value.name for value in session.get_inputs()]

    def forward(ids, mask):
        input_ids = np.array([ids], dtype=np.int64)
        arrays = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),
            "mask_position": np.array([mask], dtype=np.int64),
        }
        feed = {name: arrays[name] for name in input_names}
        started = time.perf_counter()
        logits = session.run(None, feed)[0]
        seconds = time.perf_counter() - started
        return seconds, logits.reshape(-1).tolist()

    return forward


# --------------------------------------------------------------------------------------
# The protocol loop
# --------------------------------------------------------------------------------------


def stop_before_load(rung: int, threads: int) -> None:
    """The memory runner's floor measurement (python/memory_run.py): import everything the
    rung imports, set its threads, then report and exit before reading any weights. So the
    process's peak RSS is the runtime alone. Timing runs never set MB_STOP_BEFORE_LOAD."""
    if rung == 3:
        import numpy  # noqa: F401, PLC0415
        import onnxruntime  # noqa: F401, PLC0415
    else:
        import torch  # noqa: PLC0415
        import transformers  # noqa: F401, PLC0415

        torch.set_num_threads(threads)
        if rung in (1, 2):
            import ladder  # noqa: F401, PLC0415
    send({"floor": True, "variant": f"rung{rung}"})
    sys.exit(0)


def parse_run(request: dict) -> tuple[int, int, list[int]]:
    """example, mask and ids from a run request, checked before anything is timed."""
    example, mask, ids = request.get("example"), request.get("mask"), request.get("ids")
    if not isinstance(example, int):
        fail(f"no example in {request}")
    if not isinstance(mask, int):
        fail(f"example {example}: no mask")
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        fail(f"example {example}: ids must be a list of integers")
    if not 0 <= mask < len(ids) or ids[mask] != MASK_TOKEN_ID:
        fail(f"example {example}: position {mask} does not hold [MASK] ({MASK_TOKEN_ID})")
    return example, mask, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", type=int, choices=[0, 1, 2, 3], required=True)
    args = parser.parse_args()

    weights = os.environ.get("MB_WEIGHTS") or fail("MB_WEIGHTS is not set")
    try:
        threads = int(os.environ["MB_THREADS"])
    except (KeyError, ValueError):
        fail("MB_THREADS is not set to a number")

    if args.rung == 2:
        # One inductor compile worker, as in rung2_compile.py. Set before torch is imported.
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    ready = {"ready": True, "variant": f"rung{args.rung}", "threads": threads}
    if os.environ.get("MB_STOP_BEFORE_LOAD"):
        stop_before_load(args.rung, threads)
    started = time.perf_counter()
    try:
        if args.rung == 3:
            forward = build_rung3(threads)
            ready["backend"] = "onnxruntime-cpu"
        else:
            import torch  # noqa: PLC0415

            torch.set_num_threads(threads)
            ready["backend"] = torch_backend_name()
            if args.rung == 0:
                forward = build_rung0(weights)
            elif args.rung == 1:
                forward = build_rung1(weights)
            else:
                forward, compile_seconds = build_rung2(weights)
                ready["compile_seconds"] = compile_seconds
    # Deliberately broad: the protocol turns ANY failure into an error line and a
    # non-zero exit, so nothing is swallowed. SystemExit from fail() is not an Exception.
    except Exception as error:  # noqa: BLE001
        fail(f"rung {args.rung} load: {type(error).__name__}: {error}")

    total = time.perf_counter() - started
    # load_seconds is load only; for rung 2 the compile is reported on its own.
    ready["load_seconds"] = total - ready.get("compile_seconds", 0.0)
    send(ready)

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"bad request: {error}")
        command = request.get("cmd")
        if command == "run":
            example, mask, ids = parse_run(request)
            try:
                seconds, logits = forward(ids, mask)
            except Exception as error:  # noqa: BLE001  (same reason as at load)
                fail(f"example {example}: {type(error).__name__}: {error}")
            send({"example": example, "seconds": seconds, "logits": logits})
        elif command == "quit":
            return
        else:
            fail(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
