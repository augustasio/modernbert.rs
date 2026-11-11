# Rung 29 (softmax writes bf16) against rung 28

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 2026-11-11, 14:54Z to 15:23Z, one pod. Session
log: `session.log`. This host was slow: at two threads the layers took about 116 ms per
request, against 87.5 ms on rung 28's pod, and the worker p50 behind the server was about
100 ms against 85. Both rungs ran on it, interleaved, so the comparison between them
holds; the absolute numbers are not comparable with earlier tables.

## Bound

Rung 29's logits equal rung 28's, bit for bit, over all 7,600 examples (compared as
`u32`): agreement with rung 0 7565/7600, calibrated accuracy 84.92%. The 57 tests on the
pod passed, and so did the in-run gate in every serve run.

## Stage times at two threads (batch 1, 64 examples, two interleaved rounds)

| Stage | rung 28 | rung 29 |
|---|---|---|
| attention, wall ms | 16.54, 16.76 | 16.47, 18.73 |
| softmax, thread-ms | 9.50, 9.55 | 10.52, 10.37 |
| `P V`, thread-ms | 4.47, 4.50 | 3.99, 4.07 |
| layers total, wall ms | 116.98, 115.91 | 115.28, 117.81 |

`P V` loses its rounding pass (about 0.45 thread-ms), and the softmax gains about 0.9:
rounding in its last loop costs more there than the separate pass did. No gain; the work
moved, as in rung 26.

## Behind the server (8x2, open loop, three interleaved rounds)

Best rate within 200 ms: rung 29 65, 45, none; rung 28 40, 45, none. Rung 29's first
round ran at load 20 to 27 while rung 28's ran at 28 to 40, and the third round missed the
budget at 40/s for both. At matched load (round 2) the two are equal. The `*.jsonl` lines
hold the p99, the worker p50 and the load at each rate.

Read: rung 29 computes the same thing at the same speed. Later rungs build on rung 28.
