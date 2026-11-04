# Session t4: batch 1, 4 threads

Run on 2026-11-02, 10:08 to 18:08 UTC, at commit 3fa98ab, on AC power, Low Power Mode
off. Each row is 5 replicates (separate processes) of 30 cycles of 64 examples, rung 0
as the ruler in every cycle. Rung 6 is 5 replicates of 8 cycles of 2 examples. Rung 8
was added later, on 2026-11-02, 21:01 to 22:05 UTC, at commit 693078e, in the same
setup. Rung 9 was added on 2026-11-04 in two sittings, 08:02 to 08:36 and 19:21 to
19:43 UTC, at commit 3133da8, with the resumable runner. All 50 replicates passed the
in-run correctness check. Intervals are 95%, from the two-stage
bootstrap in `python/analyze_bench.py`. The table comes from `python/bench_table.py`.

Speedup is rung 0 time over the variant's time: above 1 is faster than rung 0. The energy
ratio is rung 0 energy over the variant's, per example: above 1 uses less energy.

| Rung | Speedup vs rung 0 | Variant p50 ms | Variant p99 ms | Energy ratio | P-cluster GHz | Speedup, nominal slices only (cycles) |
|---|---|---|---|---|---|---|
| A/A: rung 0 against itself | 1 [0.998, 1.01] | 137 [132, 139] | 207 [197, 218] | 1 [0.995, 1.01] | 3.67 [3.54, 3.82] | 0.998 [0.99, 1.01] (37) |
| 1: HF, head at the mask only | 1.12 [1.12, 1.12] | 117 [115, 119] | 188 [184, 198] | 1.12 [1.11, 1.12] | 3.77 [3.7, 3.83] | 1.12 [1.11, 1.12] (21) |
| 2: + torch.compile | 1.12 [1.12, 1.13] | 118 [114, 119] | 181 [171, 189] | 1.12 [1.11, 1.12] | 3.66 [3.58, 3.74] | 1.12 [1.11, 1.13] (21) |
| 3: ONNX Runtime | 0.982 [0.975, 0.988] | 140 [139, 141] | 247 [222, 267] | 0.917 [0.881, 0.953] | 3.37 [3.23, 3.52] | 0.965 [0.955, 0.975] (14) |
| 4: candle, CPU matmul | 0.243 [0.242, 0.245] | 561 [557, 565] | 811 [795, 836] | 0.222 [0.22, 0.224] | 2.85 [2.82, 2.88] | 0.272 [0.268, 0.275] (5) |
| 5: candle + Accelerate | 0.782 [0.775, 0.79] | 167 [167, 168] | 243 [234, 248] | 0.918 [0.899, 0.936] | 3.96 [3.94, 3.97] | 0.77 [0.763, 0.774] (24) |
| 6: our Rust, naive matmul | 0.00803 [0.00732, 0.00877] | 2.38e+04 [2.31e+04, 2.44e+04] | 3.14e+04 [2.78e+04, 3.33e+04] | n/a | n/a | 0.00803 [0.00732, 0.00877] (40) |
| 7: our Rust + Accelerate | 0.408 [0.397, 0.421] | 338 [336, 341] | 601 [581, 630] | 0.476 [0.452, 0.5] | 3.97 [3.94, 3.99] | 0.389 [0.385, 0.392] (13) |
| 8: + fusion, no allocation | 0.681 [0.677, 0.685] | 220 [216, 223] | 339 [324, 348] | 0.668 [0.661, 0.676] | 3.68 [3.62, 3.75] | 0.651 [0.646, 0.657] (11) |
| 9: + f32 exp | 0.734 [0.715, 0.756] | 189 [185, 199] | 320 [272, 374] | 0.838 [0.817, 0.859] | 3.9 [3.76, 3.98] | 0.725 [0.71, 0.742] (123) |

## Notes

- **The A/A floor is about ±1%.** Rung 0 against itself gives 1.00 [0.998, 1.01]. No
  rung's interval overlaps it.
- **Most slices ran at "fair", not "nominal".** The Mac cools to nominal before each
  replicate and leaves it within minutes. The nominal-only column agrees with the main
  column to within a few percent, but on few cycles (5 for rung 4).
- **Rung 2 equals rung 1.** torch.compile adds nothing measurable at batch 1.
- **Rung 7 is 2.1x slower than rung 5.** Both use Accelerate for matmul. So the gap is
  in our non-matmul code (attention, softmax, norms, GELU), which rung 8 targets.
- **Rung 8 is 1.67x faster than rung 7, with the same logits.** p50 drops from 338 ms
  to 220 ms, and energy per example falls 1.40x. The arithmetic is unchanged: every
  logit row is bit-identical to rung 7's. Rung 8 still takes 15% longer than rung 5,
  and 1.47x as long as rung 0. Its profile points next at the f64 `exp` in
  softmax and at the attention products, which do not use Accelerate yet.
- **Rung 9 is 1.08x faster than rung 8.** p50 drops from 220 ms to 189 ms. The only
  change is f32 `exp` in the softmax and `erf` (`PLAN.md`, changelog). Energy per example
  falls 1.25x, more than the time, perhaps because the vector `exp` does less work per
  element. Replicate 2 was disturbed (ruler p50 165 ms against 126 to 142 ms, 4 nominal
  cycles of 30), which widens the interval; its speedup, 0.772, is the highest of the
  five, and the other four give 0.703 to 0.748. Rung 9 still takes 1.36x as long as
  rung 0.
- **Rungs 6 and 7 do not use 4 threads.** The naive loop is single-threaded, and
  rung 7's P-cluster clock (3.97 GHz, near the maximum) suggests Accelerate ran it on
  about one core. Rung 4's 2.85 GHz is the one rung loading all 4 cores.
- **p99 is noisy.** Several p99 figures vary more than 5% across replicates, so
  `analyze_bench.py` flags them. Trust p50 and the speedups more than p99.
- **Rung 6 has no energy figure.** Its ruler slices are 0.3 s, too short to hold whole
  power samples.
- **The raw replicates are gzipped.** `analyze_bench.py` reads `.json.gz` directly.
