# Rung 24: where a request's time goes, and three server layouts

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5, 64 cores, 128 hardware threads on the host),
2026-11-11, 00:00Z to 00:49Z, host load 16 to 42. The pod gets 32 of the 128 hardware
threads: both threads of 16 cores, scattered (`Cpus_allowed_list` 2, 6, 9, 16-17, ...,
with siblings n and n + 64).

## CPUs 0 to 3 are not ours

Batch 1 on 4 threads, 16 examples, two rounds each:

| Binding | s per example |
|---|---|
| `OMP_PLACES={0},{1},{2},{3}` | 8.41, 8.41 |
| `OMP_PLACES=cores` (the first 4 allowed cores) | 0.083, 0.083 |

CPUs 0 to 3 are outside this pod's set, so binding there fails and the threads crawl.
This explains the invalid batch-1 check in `../cloud_turin_serve24/`. Earlier pods
happened to own CPUs 0 to 3. The server was never affected: `mb-serve` binds to the
allowed set it reads from `/proc/self/status`. From now on, batch-1 checks use
`OMP_PLACES=cores`.

## Where a batch-1 request's time goes (4 threads, 64 examples, two rounds)

Wall time per stage of the 28 layers (`stage-times` build), against oneDNN's own
execution time from its verbose log (a separate run):

| Stage | ms/example | oneDNN exec | Rest (quantize, dequantize, calls) |
|---|---|---|---|
| Wi (1024 -> 5248) | 27.2 | 22.7 | 4.5 |
| Wqkv (1024 -> 3072) | 13.5 | 10.4 | 3.1 |
| MLP Wo (2624 -> 1024) | 10.4 | 7.7 | 2.7 |
| attention Wo (1024 -> 1024) | 5.1 | 3.6 | 1.5 |
| attention (scatter, RoPE, scores, softmax, P V) | 13.4 | | |
| GELU gate | 2.6 | | |
| gather heads, norms, adds | 2.6 | | |
| **layers total** | **74.9** | 44.4 | |

The four s8 matmuls are 56 ms. oneDNN computes for 44 of them, and our quantize and
dequantize loops add 12. Attention is the largest stage that is not a matmul, at 13 ms
(18%). Rung 24's whole forward pass is 75 to 83 ms, against rung 0's roughly 200.

## Server layouts (open loop, batch 1, two rounds, rung 24)

p99 latency (ms) by arrival rate; worker p50 in brackets.

| Rate/s | 4x4 r1 | r2 | 8x4 on SMT r1 | r2 | 8x2 r1 | r2 |
|---|---|---|---|---|---|---|
| 25 | 129 (70) | 105 (67) | 132 (82) | 125 (75) | 152 (106) | 151 (100) |
| 35 | 157 | 153 | 165 | 148 | 147 | 197 |
| 40 | 185 | 207 | 188 | 182 | 157 | 215 |
| 45 | 206 | 194 | 195 | 168 | 156 | 184 |
| 50 | 618 | 337 | 400 | 199 | 181 | 172 |
| 55 | | 533 | 348 | 228 | 197 | 183 |
| 60 | | | | 356 | 195 | 257 |
| 70 | | | | | 268 | 1629 |

- **4x4** (4 workers of 4 threads, one thread per core): holds the budget to about 40 to 45/s.
- **8x4 on SMT** (workers k and k + 4 share cores): to about 45 to 50/s. The second
  hardware thread adds about 10%, and each worker slows from 68 to about 90 ms.
- **8x2** (8 workers of 2 threads, one thread per core): to about 50 to 60/s. One
  noisy step in round 2 (215 ms at 40/s) is the exception. Each request is slower (about
  100 ms against 68), but twice as many run at once.

**8x2 serves about 1.3x 4x4 within the budget: about 55/s, about 177,000 examples per
dollar** at $1.12/h, against about 130,000 for 4x4 on this host. More, smaller workers
win while one request's time stays well under the budget. The next check: 16x1 and 16x2
on SMT.

Every run passed the in-run gate (logit bound 3.416, changed predictions within 2 x
35/7600 over a full pass). Raw lines: `*.jsonl`; session log: `session.log`.
