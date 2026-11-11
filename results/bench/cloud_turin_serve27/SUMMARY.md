# Rung 27 (bf16 attention products): passes its bound, about 10% faster

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 2026-11-11, 09:18Z to 09:56Z, host load 13 to 92.
Rungs 26 and 27 built and run on the same pod. Unit tests: 55 passed, including the new
check that the bf16 products are the exact sums of the rounded inputs, from four threads
at once and across shape changes.

## Bound: met

| | Rung 27 | Bound |
|---|---|---|
| Agreement with rung 0 | 7565/7600 (99.54%) | at least 7524 |
| Calibrated accuracy | 84.92% [84.12, 85.75] | at least 83.78% (rung 0: 84.78%) |
| Logit difference vs rung 0, median / p99 / max | 0.116 / 0.450 / 0.813 | reported, no bound |

Rung 26 also disagrees with rung 0 on 35 examples. Against rung 26, rung 27 moves the
logits by at most 0.79 and changes 40 raw (uncalibrated) predictions. In-run gate, by
rung 17's rule: logit bound 3.253, changed share 2 x 35/7600. `results/rung27.json`,
`results/rung27_logits.npy`.

## Where the time moved (stage-times, 2 threads, batch 1, 64 examples)

Thread-ms per example, summed over both threads, except the totals (wall ms).

| Stage | 26 r1 | 27 r1 | 26 r2 | 27 r2 |
|---|---|---|---|---|
| scores `Q K^T` | 15.32 | 6.32 | 15.42 | 6.42 |
| `P V` | 14.17 | 6.11 | 14.14 | 6.20 |
| softmax | 9.55 | 9.47 | 9.67 | 9.50 |
| scatter | 8.46 | 8.11 | 8.03 | 8.42 |
| **attention (wall)** | **26.28** | **16.49** | **26.05** | **16.87** |
| **layers total (wall)** | **119.75** | **110.01** | **121.40** | **111.31** |

The two products drop from 29.5 to 12.5 thread-ms, about 2.4x, so the bf16 rounding on
the way in and the cached primitives cost little. This host was slower for the timed
build than earlier ones (rung 26 at 120 ms here against 92 to 94 ms on the last pod),
so read the differences, not the totals.

## Behind the server (8x2, interleaved 27 then 26, three rounds)

p99 latency (ms); worker p50 in brackets; host load at the first rate in the last row.

| Rate/s | 27 r1 | 26 r1 | 27 r2 | 26 r2 | 27 r3 | 26 r3 |
|---|---|---|---|---|---|---|
| 40 | 137 (90) | 151 (97) | 153 (91) | 187 (102) | 168 (96) | 159 (98) |
| 45 | 146 | 158 | 154 | 193 | 176 | 167 |
| 50 | 172 | 194 | 167 | 211 | 180 | 176 |
| 55 | 154 | 175 | 263 | 370 | 219 | 193 |
| 60 | 183 | 212 | 252 | | 190 | 187 |
| 65 | 299 | 489 | | | 223 | 296 |
| Best within 200 ms (mb-load) | 60 | 55 | 50 | 45 | 60 | 60 |
| Host load | 18 | 41 | 76 | 30 | 21 | 20 |

The worker p50 is 5 to 11 ms lower for rung 27 in every round. Rung 27 holds the budget
5/s further in two rounds and ties in the third. In round 2 it did so at a host load of
54 to 92 against rung 26's 25 to 30. At 60/s, rung 27 serves about 193,000 examples per
dollar at $1.12/h, on this host where rung 26 reached 55 to 60/s.

**Rung 27 is the new base.** The next largest stages that are not matmuls are the
softmax (about 4.7 ms wall at two threads) and the scatter (about 4.1 ms).

Every serve run passed its in-run gate. Raw lines: `*.jsonl`; session log: `session.log`.
