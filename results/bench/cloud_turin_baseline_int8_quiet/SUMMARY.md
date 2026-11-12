# Rung 28 against the stock int8 baselines, quiet host (2026-11-12)

RunPod cpu5c, AMD EPYC 9575F, 16 cores. Commit a02d844. Three rounds, the same order in
each. Before each test the pod waited up to 8 minutes for the host's 1-minute load to
fall under 24 (our own server adds about 16 while it runs); `session.log` logs the load
at each start. Gates per stack from its own Tier 3 run (`../cloud_turin_baseline_int8`):
4x its worst logit difference and 2x its disagreement rate. Every run passed its gate.

Each cell: best rate (requests/s) with p99 at or under 200 ms, per round.

| Stack | Layout | Round 1 | Round 2 | Round 3 | Worker p50 at 40/s |
|---|---|---|---|---|---|
| rung 28 | 8x2 | 70 | none (host load 34) | 70 | 80 ms |
| rung 28, A/A repeat | 8x2 | 70 | 80 | 70 | 81 ms |
| OpenVINO static int8 | 8x2 | 50+ | 50+ | 50+ | 85 ms |
| rung 28 | 4x4 | 40 | 40 | 40 | 62 ms |
| OpenVINO static int8 | 4x4 | 40 | 40 | 50 | 61 ms |
| ONNX Runtime dynamic int8 | 4x4 | 40+ | 40+ | 40+ | 68 ms |
| ONNX Runtime, spinning off | 4x4 | 30 | 25 | 20 | 80 ms |
| PyTorch dynamic int8 | 4x4 | 35 | 40 | 40 | 71 ms |
| PyTorch dynamic int8 | 2x8 | 25 | 20 | 25 | 51 ms (at 15/s) |
| rung 28 | 1x16 | 20 | 20 | 20 | 30 ms (at 10/s) |
| OpenVINO static int8 | 1x16 | none (load 33) | 10 | 15 | 34 ms (at 10/s) |

"50+" and "40+" mean the stack still met the budget at the highest rate tried: the sweep
was capped too low for it. `../cloud_turin_baseline_int8_quiet2` reruns those at higher
rates.

What it says:

- **OpenVINO's static int8 is about as fast per request as rung 28.** At 8x2 its worker
  p50 is 84 to 85 ms against rung 28's 80 to 81; at 4x4 both are 60 to 62 ms. The
  difference that is left is small and needs the higher-rate rerun.
- **The A/A floor is one rate step.** Rung 28 against itself at 8x2: 70 and 70, 80 and
  none (the second round's first test ran at host load 34), 70 and 70. Below the limit,
  p99 at the same rate differs by up to 10 ms between the two (139 vs 139 at 40/s,
  146 vs 154 at 60/s), within the bootstrap intervals.
- **ONNX Runtime's spinning pool helps here.** With `session.intra_op.allow_spinning=0`
  its p50 rises from about 67 ms to 110 to 130 ms at low rates. Its default is the fair
  setting. On the busy host (`../cloud_turin_baseline_int8`) spinning was what hurt it.
- **PyTorch's dynamic int8 serves 35 to 40/s at 4x4, but it fails accuracy.** It loses
  13.7 points of calibrated accuracy (Tier 3), so it is not a usable result.
- **The lowest p99 under load is rung 28 at 1x16.** At 5/s: p50 37 to 38 ms, p99 66 to
  79 ms; at 10/s, p99 81 to 108 ms. One worker on 16 threads answers in about 30 ms,
  but it serves only up to 20/s within 200 ms. For throughput, 8x2 wins: 70/s with p99
  139 to 155 ms at 40 to 60/s.
