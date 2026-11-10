# Rung 22 behind rung 21's server, against rung 19 (rough check)

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 32 vCPUs = 16 cores, 2026-11-10, 20:27Z to 20:59Z.
The pod was kept because its host load was 27 at start (16 to 54 during the run); the
first try at 20:15Z, at load 113 to 144, is in `../cloud_turin_serve22/` and not read.

Same method as `../cloud_turin_serve/`: `mb-serve` with 4 workers of 4 threads bound by
`OMP_PLACES`, batch 1; `mb-load` with Poisson arrivals, 5 s warm-up and 40 s per rate,
latency from the scheduled arrival. Rung 22 and rung 19 alternate, three rounds. Every
run passed its in-run gate (rung 22: logit bound 1.707, changed predictions within its
Tier 3 share; rung 19: 0.72).

## p99 latency (ms) by arrival rate; worker p99 in brackets

| Rate/s | rung 22 r1 | r2 | r3 | rung 19 r1 | r2 | r3 |
|---|---|---|---|---|---|---|
| 10 | 114 (113) | 109 (109) | 111 (111) | 142 (133) | 136 (129) | 129 (123) |
| 15 | 124 | 113 | 114 | 152 | 154 | 131 |
| 20 | 152 | 139 | 135 | 186 | 171 | 160 |
| 25 | 166 | 142 | 211 | 208 | 186 | 241 |
| 30 | 199 | 188 | 241 | 320 | 343 | 326 |
| 35 | 184 | 213 | | | 338 | |
| 40 | 326 | 386 | | | | |

Highest rate tested with p99 at or under 200 ms: rung 22 35, 30, 20; rung 19 20, 25, 20.
Read between the 5/s steps, the crossing is near 24 to 36/s for rung 22 and 22 to 26/s
for rung 19: **about 1.45x more traffic within the budget**, and a lower p99 at every
rate in every round pair.

Per dollar at $1.12/h, using the median best rate: rung 22 about 30/s, **about 96,000
examples per dollar**; rung 19 about 20/s, about 64,000.

## Batch 1, 4 threads, 64 examples (eval binary, same rounds)

rung 19: 0.155, 0.158, 0.148 s; rung 22: 0.134, 0.132, 0.127 s, so 1.17x.
