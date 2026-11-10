# Rung 24 (u8 input to the s8 matmuls) against rung 23

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 32 vCPUs = 16 cores, 2026-11-10, 22:13Z to 23:49Z,
host load 8 to 72. Both rungs built and run on the same pod.

## The bound: bit-identical logits

Tier 3, all 7,600 examples, batches of 16 on 16 threads: rung 24's logits equal rung
23's exactly (max difference 0.0). The bound passes. Tier 3 time: 0.0255 s per example
against 0.0272 for rung 23 (one run each, so 6% is a hint, not a result).

## Behind the server (same method as `../cloud_turin_serve23/`)

4 workers of 4 threads bound by `OMP_PLACES`, batch 1, Poisson arrivals, 40 s per rate,
latency from the scheduled arrival, rungs alternating, three rounds. Every run passed the
in-run gate (rung 23's: logit bound 3.416, at most 2 x 35/7600 of predictions changed).
The changed-prediction counts are identical for the two rungs in each round, as they must
be with identical logits.

p99 latency (ms) by arrival rate; worker p50 in brackets.

| Rate/s | rung 24 r1 | r2 | r3 | rung 23 r1 | r2 | r3 |
|---|---|---|---|---|---|---|
| 20 | 110 (69) | 114 (71) | 109 (67) | 112 (69) | 106 (68) | 120 (72) |
| 25 | 127 | 125 | 107 | 122 | 123 | 116 |
| 30 | 148 | 120 | 153 | 140 | 122 | 164 |
| 35 | 142 (69) | 141 (68) | 131 (67) | 150 (70) | 140 (69) | 135 (69) |
| 40 | 220 | 226 | 157 | 224 | 220 | 170 |
| 45 | 207 | 215 | 202 | 329 | 221 | 227 |
| 50 | | | 377 | | | 467 |

Highest rate with p99 at or under 200 ms: 35, 35, 40 for both rungs. Worker p50 is 1 to
2 ms lower for rung 24 (67 to 72 ms against 68 to 72). **Rung 24 is exact and about even
with rung 23 in serving**: about 112,000 examples per dollar, unchanged. Rung 24 stays on
the ladder because it is free (same numbers) and its kernel path is the one VNNI wants,
but it is not a speedup at batch 1.

## Batch-1 speed check: not valid on this pod

The eval binary at batch 1 on CPUs 0 to 3 took 8.4 s per example for both rungs, about 60x
the 0.11 to 0.13 s seen on other pods and the 69 ms the server's workers measured on this
same pod. CPUs 0 to 3 were likely shared with something else on the host. Both rungs saw
the same, so it says nothing about either; the server's worker times above stand in.

Raw lines: `rung2*_4x4_round*.jsonl`; session log: `session.log`.
