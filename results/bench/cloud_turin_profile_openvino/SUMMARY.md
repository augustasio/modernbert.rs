# Rung 28 and OpenVINO's static int8, op by op (2026-11-12)

RunPod cpu5c, AMD EPYC 9575F. Commit 52a8970. Both stacks on the same two CPUs (two
physical cores), two threads, batch 1, the first 64 test prompts, two interleaved
rounds at host load 9 to 11. Rung 28: a `stage-times` build. OpenVINO 2026.4.1: the
CPU plugin's per-node counters (`python/profile_openvino.py`), and its compiled graph
(`ov_round*/runtime_model.xml`, weights removed), which names what was fused into what.
The rounds agree within 2 ms. Round 2 below.

| | rung 28 | OpenVINO |
|---|---|---|
| wall per example | 82 ms (layers 81.9) | 83.2 ms p50 (nodes 77.8, the rest is Python) |
| int8 linear layers | 59.2 (oneDNN 55.7, quantize 1.1, dequantize 2.4) | 59.0 (FullyConnected brgemm i8, with dequantize, bias and the next quantize fused) |
| GELU gate (+ quantize for the MLP's Wo) | 7.3 | 2.5 ("Subtract" node: dequantize, GELU, gate, smoothing and FakeQuantize fused, bf16) |
| attention | 11.3 | about 10 (int8 `Q K^T` and `P V` 2.1, bf16 softmax 2.4, transposes, RoPE, mask) |
| norms and adds | 3.0 | 1.7 (MVN, Add, bf16) |
| layout copies | none | 4.3 (Reorder, bf16 and i8) |

What it says:

- **The int8 matmuls tie.** Both use oneDNN's VNNI brgemm kernels and spend 59 ms.
- **OpenVINO is ahead on the GELU gate by about 4.8 ms.** It runs it as one fused bf16
  kernel with the activation quantize, at a fixed (static) scale. Ours computes the gate
  in f32 and then searches each row for its largest value to set the scale. 11.8 million
  gate values per request in 14.6 thread-ms is about 1.2 ns each, which suggests the loop
  does not vectorise (to check in the disassembly before any change: rung 25's lesson).
- **OpenVINO is a little ahead on the softmax** (2.4 ms in bf16, against our 9.3
  thread-ms, about 4.7 ms of wall). NNCF also made its attention products int8.
- **We are ahead on layout:** OpenVINO spends 4.3 ms on reorders that our layout avoids.
- Net, the two are within about 2% per request, from different places.
