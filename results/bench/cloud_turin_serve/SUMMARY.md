# Rung 21: open-loop load test of the server (rough check)

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 32 vCPUs = 16 cores. 2026-11-10, 08:45Z to
09:14Z. Host load 39 to 120 from other renters. `mb-serve` and `mb-load` on the same pod;
workers bound to the 16 first hardware threads, so the server and load generator ran on
the SMT siblings. Poisson arrivals, 5 s warm-up then 40 s per rate. Latency runs from
each request's scheduled arrival, so it includes the queue. Every reply passed the
check against rung 0 (no changed predictions in any run). Raw lines: `*.jsonl`.

## Highest rate with p99 at or under 200 ms

| Layout | Round 1 | Round 2 |
|---|---|---|
| rung 19, 4 workers x 4 threads, batch 1 | 15/s | 20/s |
| rung 19, 1 x 16, batch 1 | 10/s | 10/s |
| rung 19, 4 x 4, batch up to 2 | 15/s | none (load 120) |
| rung 19, 2 x 8, batch up to 2 | 10/s | none (load 108) |
| rung 0, 1 x 16, batch 1 | none, even at 1/s | none |

So the result is about **15 to 20 examples/s within the budget**, on 4 bound workers of 4
threads: about 48,000 to 64,000 examples per dollar at $1.12/h. Rung 0 misses the budget at
1 request per second (p99 415 ms).

## What the queue costs

The closed-loop check served 36.7/s on 4 x 4 at p99 160 ms. Open loop, p99 passes 200 ms
near 20 to 25/s. The worker time stays near 140 ms at p99; the queue wait grows: at
25/s its p99 is 145 to 162 ms. With Poisson arrivals, requests bunch, and a 100 ms
service time leaves little room for a wait inside 200 ms. One 16-thread worker is faster
per request (p50 48 ms) but serves one at a time, so its queue grows sooner (p99 205 ms at
15/s).

The load generator kept up: requests left at most 0.2 ms after their scheduled time.

Noise: the host load doubled between rounds for the batch-2 layouts, and their results
moved from 15/s to none. Read single numbers here as rough.

## Next

- Shorter requests help the queue the most, so the remaining precision and kernel
  work still pays twice.
- 4 x 4 is the right shape on 16 cores: fewer, wider workers queue sooner, and narrower
  ones (8 x 2, p50 162 ms closed loop) leave no room for any wait.
