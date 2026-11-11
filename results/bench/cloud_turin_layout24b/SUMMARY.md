# Rung 24: 8x2 against 16x1 and 16x2 on SMT

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 32 vCPUs = 16 cores (a scattered subset of the
host's 128 hardware threads), 503 GB RAM, 2026-11-11, 01:00Z to 01:15Z, host load 17 to
36. Same method as `../cloud_turin_layout24/`: open loop, batch 1, 40 s per rate, two
rounds, rates from 40/s.

p99 latency (ms) by arrival rate; worker p50 in brackets.

| Rate/s | 8x2 r1 | r2 | 16x1 r1 | r2 | 16x2 on SMT r1 | r2 |
|---|---|---|---|---|---|---|
| 40 | 164 (110) | 164 (105) | 265 (179) | 273 (182) | 210 (135) | 266 (143) |
| 45 | 178 | 172 | 259 | 280 | 238 | 290 |
| 50 | 193 | 183 | | | | |
| 55 | 208 | 207 | | | | |
| 60 | 370 | 212 | | | | |

- **16x1:** one request on one thread takes about 180 ms, so the p99 is past the budget
  before any queue forms.
- **16x2 on SMT** (workers k and k + 8 share cores): about 140 ms per request, also past the
  budget at 40/s.
- **8x2:** holds the budget to 50/s in both rounds, as on the previous host (50 to 60/s).

**8x2 is the best layout for rung 24: about 50 to 55/s within the budget, about 160,000 to
177,000 examples per dollar.** With fewer threads per request, a request gets slower than
the queue saves. Two threads per request is the point where one request still fits in
about half the budget.

Every run passed the in-run gate. Raw lines: `*.jsonl`; session log: `session.log`.
