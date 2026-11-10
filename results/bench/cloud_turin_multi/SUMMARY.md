# Several workers per host, and thread binding (rough check)

Host: RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 32 vCPUs = 16 cores with SMT. $1.12 an
hour. 2026-11-10, 08:03Z to 08:39Z. Host load 16 to 30 for grids 1 and 2, 72 to 95 for
grid 3.

Not the paired harness. `tools/multi_worker_check.py` starts N workers of T threads,
each fed its own random examples in a closed loop (the next request goes in as soon as
the reply comes back), for 3 warm-up calls and then a fixed count. Throughput is all
examples over the wall time. Latency is each worker's own forward-pass time, so it
has **no queue wait**: an open-loop load test (rung 21) will add it. Raw lines are in
the `.jsonl` files, three rounds each (two for grid 3).

`pin=2` sets `OMP_PLACES` to that worker's own cores, one thread per physical core, and
`OMP_PROC_BIND=close`. `pin=1` uses `taskset` on the same cores.

## Rung 19 at batch 1, by layout (pin=2, mean of 3 rounds)

| Workers x threads | Examples/s | p50 ms | p99 ms | Within 200 ms? |
|---|---|---|---|---|
| 1 x 16 | 25.5 | 38 | 78 | yes |
| 2 x 8 | 27.6 | 70 | 99 | yes |
| **4 x 4** | **36.7** | **102** | **160** | **yes** |
| 8 x 2 | 44.1 | 162 | 257 | no |

Batches above 1 do not beat 4 x 4 within the budget: 2 x 8 at batch 2 gives 30.8/s at
p99 181 ms, 1 x 16 at batch 4 gives 25.3/s at p99 212 ms.

## Rung 0 (PyTorch) in the same layouts (pin=2, 2 rounds, load about 75 to 85)

| Workers x threads | Examples/s | p50 ms | p99 ms |
|---|---|---|---|
| 1 x 16 | 5.8 | 168 | 216 |
| 2 x 8 | 8.2 | 241 | 314 |
| 4 x 4 | 9.0 | 410 | 607 |
| 8 x 2 | 8.5 | 800 | 1160 |
| 16 x 1 | 10.8 | 1310 | 2110 |

Rung 0 meets the budget in no layout; its best near it is 1 x 16 at 5.8/s. Rung 19 at
4 x 4 in the same rounds: 36.9/s and 35.0/s, p99 150 and 160 ms.

## Per dollar within the 200 ms budget

| | Examples/s | Examples per dollar |
|---|---|---|
| Rung 0, 1 x 16 (p99 216 ms, just over) | 5.8 | about 18,600 |
| Rung 19, 1 x 16 | 25.5 | about 82,000 |
| Rung 19, 4 x 4 | 36.7 | about 118,000 |

## The 1.5 s stalls

- With `taskset` (pin=1), rung 19 stalls for about 1.49 s on some calls, at batch 1 and
  batch 2 (8 of 400 and 13 of 300 in the worst rounds). The stall time is nearly the same
  every time.
- With OpenMP's own binding (pin=2, or `OMP_PROC_BIND=close OMP_PLACES=cores`), no call
  in grids 2 and 3 went over 1 s, in any layout or batch.
- Unbound, this quieter host showed none at batch 2, but the noisy host of the sweep
  showed 36 of 1200. `GOMP_SPINCOUNT=0` and `OMP_WAIT_POLICY=passive` also removed them,
  but cost about 20% of throughput.
- So bind threads through OpenMP. The exact mechanism behind the 1.49 s is not known.

Binding also helps on its own: 1 x 16 at batch 1 goes from 20.4/s (p99 102 ms) unbound to
25.5/s (p99 78 ms), and batch 2 from 18.3/s to 23.4/s.

## Cost

About 40 minutes of pod: about $0.80.
