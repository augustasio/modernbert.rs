# Rung 24 on 8x2: dynamic batching does not help

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 32 vCPUs = 16 cores, 2026-11-11, 01:49Z to 02:01Z,
host load 46 to 75 (a busy host; every number here is worse than on the last two pods).
Open loop, 40 s per rate, two rounds, rates from 40/s. Three server settings, 8 workers of
2 threads each: batch 1 (as before); up to 4 queued requests per worker call with no
wait (`--max-batch 4 --max-wait-ms 0`); and the same with up to 5 ms of wait for the
batch to fill.

p99 latency (ms); worker p50 and mean batch size in brackets.

| Rate/s | batch 1 r1 | r2 | up to 4, no wait r1 | r2 | up to 4, 5 ms r1 | r2 |
|---|---|---|---|---|---|---|
| 40 | 171 (114, 1.0) | 234 (122) | 431 (116, 1.1) | 676 (128, 1.2) | 507 (137, 1.4) | 636 (145, 1.5) |
| 45 | 187 | 247 | 506 (120, 1.2) | 1983 (582, 2.7) | 677 (154, 1.7) | 673 (160, 1.7) |
| 50 | 299 | | | | | |

Batching more than doubles the p99 at the same rate in both rounds. At 2 threads a
worker's batch costs about its size times one request, so a batch saves no compute, and
every request in it waits for the longest. The int8 path pads nothing, but the work is
still per row. **Batch 1 stays.** This host was busy enough that batch 1 itself held the
budget only to 45/s in round 1, against 50 on the last two hosts.

Every run passed the in-run gate. Raw lines: `*.jsonl`; session log: `session.log`.
