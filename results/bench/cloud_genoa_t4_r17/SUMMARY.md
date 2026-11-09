# Cloud rough check: rung 17 on Genoa, 4 threads

A rough check (PLAN.md, phase 2), not a Mac measurement: never mix these figures into
the Mac tables.

Run on 2026-11-09, 06:46 to 08:29 UTC, at commit 815efac, on a RunPod secure pod (L4 GPU
type, GPU unused). It was a different pod from rung 16's, with the same host CPU: AMD EPYC
9254 (Zen 4, Genoa), the container capped at 5.1 CPUs. Ubuntu 22.04, Python 3.13,
PyTorch 2.15.0 CPU wheel (MKL 2024.2, GNU OpenMP). Rung 17 is linked to oneDNN 3.13.3,
built from source (OpenMP runtime, matmul and reorder primitives only), and to pip `mkl`
2024.2.2 for its remaining f32 matmuls.

Each row is 5 replicates (separate processes) of 30 cycles of 16 examples, rung 0 as the
ruler in every cycle. Intervals are 95%, from the two-stage bootstrap in
`python/analyze_bench.py`.

**In-run correctness check:** all 15 replicates passed. Rung 17's gate is 0.72 on any
logit and a share of 2 x 2/7600 changed predictions, both from its Tier 3 run. The worst
logit difference seen was 0.140, and no prediction changed in 4,790 checked replies.

| Config | Speedup vs rung 0 | Rung 0 examples/s | Variant examples/s | Variant p50 ms | Variant p99 ms |
|---|---|---|---|---|---|
| A/A: rung 0 against itself | 0.993 [0.988, 0.998] | 1.99 [1.97, 2.01] | 1.97 [1.95, 1.99] | 499 [495, 503] | 693 [674, 715] |
| 17, batch 1 | 2.04 [2.02, 2.06] | 2.00 [1.98, 2.01] | 4.07 [4.04, 4.10] | 243 [242, 244] | 339 [324, 353] |
| 17, batch 16 | 3.10 [3.00, 3.21] | 1.79 [1.72, 1.87] | 5.61 [5.56, 5.66] | 2830 [2810, 2870] | 3220 [3110, 3290] |

At batch 16 the latencies are per batch.

## Notes

- **bf16 doubles rung 0's speed at batch 1 and triples it at batch 16.** Rung 16 (fp32,
  the same code otherwise) scored 1.10x and 1.51x on the same CPU. So bf16 alone is worth
  about 1.9x at batch 1 and 2.1x at batch 16, which is close to the 2x that halving the
  bytes of every weight and activation predicts.
- **It still misses the 200 ms budget on 4 cores.** Batch-1 p50 is 243 ms and p99 339 ms.
  Under the phase 2 target (p99 at or under 200 ms per request) this host slice cannot
  serve even one request at a time. The next host needs more cores per request.
- **Batch 16 is a ceiling only.** 2.8 s a batch is far outside the budget. It shows the
  throughput a 4-core slice could reach with no latency limit: 5.6 examples/s.
- **The A/A floor is 0.993 [0.988, 0.998] on this pod**, against 1.004 [1.000, 1.008] on
  rung 16's pod. Both are within 1% of 1, but this one excludes 1. So on these shared
  hosts, read a ratio within about 1% of 1 as no difference. Both rung 17 figures are far
  past it.
- **The host was busy.** The mean host clock was 2.50 to 3.02 GHz between configs, and
  rung 0's batch-16 p50 was 8.4 s here against 6.0 s on rung 16's pod. The paired design
  absorbs this; the absolute figures carry it.
- **Steal time was 0 in every slice**, as on rung 16's pod.
- **Cost:** the pod ran about 7 h 13 min at $0.59/h, about $4.26. About 5 h of that was
  idle: a wait loop matched its own command line and never saw the setup finish at
  01:25 UTC. Waits now poll for a marker line in the job's log.
