# Inside the GELU gate, and the rounding candidate (2026-11-12)

RunPod cpu5c, AMD EPYC 9575F. Commit 4d38970. `stage-times` builds at two threads on two
physical cores (CPUs 15 and 18), batch 1, the first 64 test prompts, three rounds with
the two builds interleaved. The host was busy (load 53 to 58 against 8 to 11 in
`../cloud_turin_profile_openvino`), so every time here is about a third higher than
there; the comparison holds within a round. `gate_` lines are thread-ms (two threads).

| Stage (ms/example) | rung 28, rounds 1/2/3 | round-simd, rounds 1/2/3 |
|---|---|---|
| gelu_gate (wall) | 8.27 / 8.18 / 8.42 | 6.82 / 6.97 / 6.90 |
| gate_math (thread-ms) | 9.72 / 9.64 / 10.11 | 10.09 / 10.39 / 10.23 |
| gate_quant (thread-ms) | 5.16 / 5.10 / 5.15 | 1.30 / 1.28 / 1.28 |
| q8_quantize | 1.29 / 1.31 / 1.28 | 0.55 / 0.59 / 0.59 |
| layers total | 111.5 / 114.1 / 115.5 | 108.7 / 110.9 / 110.4 |

- **The rounding was a third of the gate.** The saturating `round_ties_even() as i8`
  costs 5.1 thread-ms over 11.8 million gated values; the 1.5 * 2^23 shift costs 1.3.
  The separate quantize passes fall from 1.3 to 0.6 ms. The layers total falls by 2.8 to
  5.1 ms per round (about 3%).
- **The logits are bit-identical** on the 64 examples, and the exhaustive test (every f32
  in [-300, 300]) passes on the pod.
- **The gate's arithmetic is the bigger half:** about 10 thread-ms, 5 ms of wall. The
  final binary's `gelu_gate_row` has a vector loop (74 zmm instructions). At about 0.8
  thread-ns per value it is still far from OpenVINO's fused bf16 gate (2.5 ms wall for
  the whole stage). That is the next candidate, and it would change the numbers.
