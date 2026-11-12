# Rung 30 against rung 28, serving at 8x2 (2026-11-12)

RunPod cpu5c, AMD EPYC 9575F. Commit ff282bb. Both rungs on one pod, interleaved, three
rounds, 8 workers of 2 threads, open-loop rates from 50/s in steps of 5, 40 s per rate,
p99 budget 200 ms. The pod was deleted after the run.

## Tier 3: the bound holds

Rung 30's logits equal rung 28's, bit for bit, on all 7,600 examples. Rung 28 was rerun
on this pod and matches the committed `rung28_logits.npy`. Agreement with rung 0 is
7565/7600; calibrated accuracy is 84.92%, the same as rung 28's. Logits:
`results/rung30_logits.npy`.

## Serving: not resolved

| Round | Rung, start time | Host load | Best rate within 200 ms | Worker p50 at 50/s |
|---|---|---|---|---|
| 1 | 30, 11:01Z | 23 to 36 | 65/s | 80 ms |
| 1 | 28, 11:06Z | 44 to 52 | none | 108 ms |
| 2 | 30, 11:16Z | 96 to 107 | none | 100 ms |
| 2 | 28, 11:25Z | 123 to 156 | none | 119 ms |
| 3 | 30, 11:35Z | 89 to 133 | 55/s | 102 ms |
| 3 | 28, 11:46Z | 75 to 81 | none | 141 ms |

- **The host got busy after the first test.** Load rose from 23 to between 44 and 156,
  against 8 to 24 in the quiet runs. Rung 28's three tests all ran on the busy host, so
  the rates cannot be compared.
- **The one quiet test agrees with the earlier quiet runs.** Rung 30 held 65/s (p99 173
  ms at 65/s, 236 ms at 70/s). Rung 28 held 70, 80 and 70/s on the quiet host in
  `../cloud_turin_baseline_int8_quiet2`. The two are within one rate step, which is the
  A/A floor.
- **The expected gain is under one rate step.** The gate split
  (`../cloud_turin_gate_split`) puts rung 30's gain at about 3% of the layers' time. A
  rate step of 5/s is about 7% at 70/s. So a serving test at this resolution would show
  a tie even on a quiet host. The stage times are the measurement of record for rung 30.
