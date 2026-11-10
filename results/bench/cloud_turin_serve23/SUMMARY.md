# Rung 23 behind rung 21's server, against rung 22 (rough check)

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 32 vCPUs = 16 cores, 2026-11-10, 21:03Z to 21:43Z,
host load 16 to 36. Same method as `../cloud_turin_serve22b/`: 4 workers of 4 threads
bound by `OMP_PLACES`, batch 1, Poisson arrivals, 40 s per rate, latency from the
scheduled arrival; rung 23 and rung 22 alternate, three rounds. Every run passed its
in-run gate (rung 23: logit bound 3.416, at most 2 x 35/7600 of predictions changed).

## p99 latency (ms) by arrival rate; worker p99 in brackets

| Rate/s | rung 23 r1 | r2 | r3 | rung 22 r1 | r2 | r3 |
|---|---|---|---|---|---|---|
| 15 | 101 (97) | 116 (100) | 97 (95) | 118 (112) | 134 (117) | 139 (115) |
| 20 | 145 | 125 | 115 | 175 | 154 | 152 |
| 25 | 154 | 133 | 128 | 178 | 171 | 172 |
| 30 | 146 | 132 | 130 | 249 | 186 | 207 |
| 35 | 157 | 160 | 159 | 246 | 230 | 250 |
| 40 | 278 | 172 | 284 | | 386 | |
| 45 | 251 | 280 | 1104 | | | |

Highest rate tested with p99 at or under 200 ms: rung 23 35, 40, 35; rung 22 25, 30, 25.
**About 1.35x rung 22**, and lower p99 at every rate in every round pair. At about 35/s
on a $1.12/h pod: **about 112,000 examples per dollar** (rung 22 about 87,000 on this
host, rung 19 about 64,000 on the previous one).

## Batch 1, 4 threads, 64 examples (eval binary, same pod)

rung 22: 0.132, 0.132, 0.131 s; rung 23: 0.110, 0.112, 0.110 s, so 1.19x.

## Changed predictions in serving

Rung 23 changed 0.53% to 0.65% of predictions in the load tests, against 0.46% (35 of
7,600) in its Tier 3 run at batches of 16. Batch 1 rounds differently from batch 16, so a
few more close calls move; it stays inside the gate's twice-the-Tier-3 share.
