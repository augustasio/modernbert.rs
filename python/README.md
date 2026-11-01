# Python side

Rungs 0 to 3 of the ladder, plus the reference activation dumps that the Rust tests
verify against.

- Rung 0: HF transformers, stock pipeline. The correctness oracle for everything else.
- Rung 1: MLM head at the mask position only.
- Rung 2: `torch.compile`, inductor, CPU.
- Rung 3: ONNX Runtime.

Also here: the forward hooks that dump per-layer activations to safetensors for tier-2
verification, and the plotting for the p99-against-throughput frontier.

Before anything else, run the BLAS check. If torch is linked against bundled OpenBLAS
rather than Accelerate, rungs 0 to 3 are running on a different BLAS from rungs 5 and 7
and the headline comparison is indefensible.
