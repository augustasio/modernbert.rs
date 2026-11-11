# Rung 28 (scatter in bf16) against rung 27

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 2026-11-11, 11:50Z to 12:23Z, one pod. Session
log: `session.log`.

## Bound

Rung 28's logits equal rung 27's, bit for bit, over all 7,600 examples (compared as
`u32`). So the Tier 3 numbers are rung 27's: agreement with rung 0 7565/7600,
calibrated accuracy 84.92% [84.12, 85.75], worst logit difference 0.813. The tests on
the pod passed (56), and so did the in-run gate in every serve run.

## Stage times at two threads (batch 1, 64 examples, two interleaved rounds)

| Stage | rung 27 | rung 28 |
|---|---|---|
| attention, wall ms | 15.37, 15.21 | 13.34, 13.18 |
| scatter, thread-ms | 7.91, 7.89 | 8.67, 8.44 |
| `Q K^T`, thread-ms | 6.42, 6.37 | 4.11, 4.10 |
| softmax, thread-ms | 9.53, 9.41 | 9.40, 9.40 |
| `P V`, thread-ms | 5.99, 5.97 | 4.02, 4.00 |
| layers total, wall ms | 90.20, 89.68 | 87.81, 87.26 |

The second pass is gone from the two products (about 4.3 thread-ms), and the scatter
pays about 0.7 thread-ms of it back: it now rounds every value and writes V transposed.
Net: about 2 ms less per request, 2.3%, at the low end of the 1 to 3 ms expected.

## Behind the server (8x2, open loop, three interleaved rounds)

p99 ms by rate, worker p50 in brackets. Host load is in each line of the `*.jsonl`.

| Rate/s | 28 r1 | 27 r1 | 28 r2 | 27 r2 | 28 r3 | 27 r3 |
|---|---|---|---|---|---|---|
| 40 | 128 (84) | 197 (101) | 137 (85) | 132 (88) | 137 (86) | 135 (87) |
| 45 | 133 (84) | 156 (92) | 150 (85) | 169 (89) | 142 (85) | 750 (111) |
| 50 | 142 (84) | 161 (92) | 152 (86) | 164 (92) | 158 (87) | 4311 (167) |
| 55 | 182 (89) | 177 (90) | 3912 (134) | 190 (92) | 183 (90) | |
| 60 | 171 (87) | 181 (92) | 1680 (98) | 212 (93) | 171 (88) | |
| 65 | 163 (87) | 192 (93) | | 253 (92) | 610 (101) | |
| 70 | 1745 (109) | 283 (96) | | | 224 (88) | |

Best rate within 200 ms: rung 28 65, 50, 60; rung 27 65, 55, 40.

The host was noisy. Each collapse lines up with a neighbour's burst (load 60 to 143 against
a usual 26 to 40): rung 27 r1 at 40/s (load 117), rung 27 r3 at 45 and 50 (68, 130), rung
28 r1 at 70 and 80 (63, 143). Rung 28 r2 at 55 (load 38) is the one collapse without a
clear burst. So the best-rate line here measures the neighbour more than the rung. The
cleaner signal is the worker p50 at the same rate and similar load: rung 28 is 3 to 6 ms
lower, matching the stage times.

Read: rung 28 is about 2 ms faster per request and computes the same thing; within the
noise, it serves the same 60 to 65/s as rung 27, about 193,000 to 209,000 examples per
dollar.
