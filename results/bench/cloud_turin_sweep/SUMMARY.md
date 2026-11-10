# Thread and batch sweep on Zen 5 (rough check)

Host: RunPod `cpu5c`, AMD EPYC 9575F (Zen 5, Turin), 32 vCPUs = 16 cores with SMT, on
CPUs scattered across the socket. $1.12 an hour. Host load average about 170 to 200 from
other renters, so this is a noisy host: read the A/A floor first. Dates: 2026-11-09 and
2026-11-10. Each job: 5 replicates x 30 cycles x 16 examples, every slice paired with rung
0 (PyTorch, bf16 off, MKL) in the same process. No energy (Linux has no powermetrics).

Every job passed the in-run gate (rung 17's: worst logit difference 0.72, changed share
2 x 2/7600). Latencies are per request (one batch), worker side, in ms.

## Batch 1

| Job | Speedup vs rung 0 | p50 | p99 [95% CI] | Examples/s [95% CI] | Rung 0 p99 |
|---|---|---|---|---|---|
| A/A, 16 threads | 1.0 [1.0, 1.0] | 234 | 322 | 4.2 | 321 |
| rung 17, 8 threads | 0.4 | 879 | 1401 | 1.1 | 459 |
| rung 17, 16 threads | 0.2 | 1258 | 1939 | 0.8 | 379 |
| rung 18, 4 threads | 3.8 [3.7, 3.9] | 118 | 190 [177, 196] | 8.1 | 721 |
| rung 18, 8 threads | 2.8 [2.6, 3.0] | 127 | 175 [164, 198] | 7.9 | 490 |
| rung 18, 16 threads | 1.7 [1.6, 1.9] | 155 | 267 [247, 286] | 6.2 | 358 |
| rung 19, 4 threads | 4.3 [4.1, 4.6] | 114 | 173 [160, 178] | 8.6 [8.3, 8.9] | 800 |
| rung 19, 8 threads | 3.5 [3.3, 3.8] | 100 | 140 [132, 147] | 10.1 [9.7, 10.7] | 479 |
| **rung 19, 16 threads** | **4.4 [4.2, 4.5]** | **63** | **115 [86, 139]** | **15.4 [14.8, 16.0]** | 366 |

- Rung 17 gets slower with more threads: each attention thread opened its own OpenMP team
  through MKL. Rung 18 fixes that; rung 19 also runs our threads on libgomp's pool.
- Rung 0 itself misses the 200 ms budget at every thread count (p99 366 to 800 ms).
- Rung 19 at 16 threads meets it with room: p99 115 ms.
- 32 threads (both SMT siblings) was pathological in a 24-example probe (2.6 s an
  example for rung 19) and was not swept.

## Batches of 2, 4 and 8 (rung 19, 16 threads)

| Batch | Speedup | p50 | p99 [95% CI] | Examples/s | Calls over 1 s |
|---|---|---|---|---|---|
| 1 | 4.4 | 63 | 115 [86, 139] | 15.4 | 1/2400 |
| 2 | 2.1 [2.0, 2.3] | 167 | 1408 [1205, 1521] | 7.8 | 36/1200 |
| 4 | 2.8 [2.7, 2.9] | 266 | 1202 [667, 1408] | 12.8 | 9/600 |
| 8 | 3.6 [3.5, 3.8] | 440 | 926 [571, 1588] | 17.4 | 3/300 |

- Batching does not pay here. Batch 8 gives 17.4 examples/s, 13% over batch 1, at a p50 of
  440 ms. Every batch above 1 misses the budget.
- A few calls still stall for about 1.5 s at batch 2 and above, after the fix below, and
  never at batch 1. Cause not found yet. It is the next thing to look at before rung 21's
  server batches requests.
- The first runs of these jobs (and of rung 19 at 4 and 8 threads) had a bug: the
  OpenMP team was one thread per attention block, up to 128. They are kept in `buggy/`.
  Capping the team at the pool width cut the stalls (87 to 36 of 1200 at batch 2) and gave
  9% more throughput at 4 threads, but did not remove them.

## Per dollar, within the 200 ms budget

Measured: rung 19, batch 1, 16 threads on the whole 32-vCPU pod: 15.4 examples/s for
$1.12/h, about **49,500 examples per dollar**. Rung 0 on the same pod cannot meet the
budget at all.

Not measured, but the next question: 4 threads give 8.6 examples/s on a quarter of the
cores. If four 4-thread workers side by side kept that rate (memory bandwidth may not
allow it), the pod would serve about 34 examples/s, about 110,000 examples per dollar,
with p99 near 175 ms. Rung 21 (the server) should test several workers per host
before it tests batching.

## Precision rungs on this host (Tier 3, all 7,600 examples)

- Rungs 18 and 19: 7598/7600 agreement, calibrated accuracy 84.72% (rung 0: 84.78%).
- Rung 20 (int8): fails its bound. 99.55% agreement, but calibrated accuracy 83.12%, 1.66
  points down against 1.5 allowed. Also slower than rung 19. Not benchmarked.

## Cost

This pod ran about 7.4 hours (2026-11-09 21:20Z to 2026-11-10 04:43Z): about $8.30.
