# Session t4: batch 1, 4 threads

Run on 2026-11-02, 10:08 to 18:08 UTC, at commit 3fa98ab, on AC power, Low Power Mode
off. Each row is 5 replicates (separate processes) of 30 cycles of 64 examples, rung 0
as the ruler in every cycle. Rung 6 is 5 replicates of 8 cycles of 2 examples. Rung 8
was added later, on 2026-11-02, 21:01 to 22:05 UTC, at commit 693078e, in the same
setup. Rung 9 was added on 2026-11-04 in two sittings, 08:02 to 08:36 and 19:21 to
19:43 UTC, at commit 3133da8, with the resumable runner. Rung 10 was added on
2026-11-05, 00:31 to 01:26 UTC, at commit e3b9a66, and rung 11 the same day, 08:04 to
08:50 UTC, at commit 9f46fbd. Rung 12 was added on 2026-11-05, 19:12 to 19:58 UTC,
at commit 465c39e. Rung 13 was added on 2026-11-07, 14:02 to 14:42 UTC, at commit
e05f371. Rung 14 and its batched configs were added on 2026-11-07 and 08, 22:29 to
01:09 UTC, at commit 5da6d1d (one restart at 23:28 resumed batch 4 from replicate 2).
All 90 replicates passed the in-run correctness check. Intervals are 95%, from the two-stage
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
| 10: + attention in sgemm | 0.969 [0.951, 0.997] | 146 [144, 151] | 364 [219, 682] | 1.14 [1.1, 1.17] | 3.88 [3.74, 3.95] | 0.954 [0.942, 0.967] (18) |
| 11: + threads outside matmul | 1.14 [1.11, 1.16] | 116 [111, 120] | 323 [184, 486] | 1.13 [1.09, 1.16] | 3.81 [3.7, 3.91] | 1.12 [0.989, 1.16] (25) |
| 12: + f32 erf, norms, softmax sum | 1.22 [1.2, 1.25] | 110 [107, 117] | 292 [180, 387] | 1.23 [1.21, 1.26] | 3.8 [3.7, 3.88] | 1.22 [1.2, 1.24] (30) |
| 13: + weights moved, not copied | 1.22 [1.21, 1.23] | 105 [102, 110] | 139 [132, 142] | 1.21 [1.2, 1.22] | 3.86 [3.8, 3.92] | 1.22 [1.21, 1.23] (135) |
| 14: batched forward pass, batch 1 | 1.22 [1.22, 1.22] | 105 [104, 106] | 133 [132, 139] | 1.22 [1.22, 1.22] | 3.92 [3.91, 3.93] | 1.22 [1.21, 1.22] (92) |

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
- **Rung 10 is 1.32x faster than rung 9, and nearly level with rung 0.** The speedup
  is 0.969 [0.951, 0.997], p50 146 ms against the ruler's 141 ms. Energy per example
  is 1.14x lower than rung 0's, so rung 10 already uses less energy. The change moves
  the attention products into `cblas_sgemm` (`PLAN.md`, changelog). Replicate 1 ran
  while the Mac was still busy (ruler p50 154 ms against 133 to 146 ms); it gives 1.00,
  and the other four give 0.948 to 0.983. Its p99 interval, [219, 682] ms, is wide for
  the same reason.
- **Rung 11 is faster than rung 0: 1.14 [1.11, 1.16].** p50 drops from 146 ms to
  116 ms, against the ruler's 125 to 140 ms. Rung 11 is rung 10 with attention, the
  norms, the residual adds and the GELU gate split across the 4 threads, and its logits
  are bit-identical to rung 10's (`PLAN.md`, changelog). So threading that work is worth
  1.17x over rung 10. Rung 11 is now level with rungs 1 and 2 (1.12), the best PyTorch
  rungs. Energy per example is 1.13x lower than rung 0's, about the same as rung 10: the
  extra cores finish sooner but draw more power while they run.
- **Rung 12 is 1.22 [1.20, 1.25], now ahead of every PyTorch rung.** p50 drops from
  116 ms to 110 ms. The change moves GELU's `erf`, the LayerNorm mean and variance,
  and the softmax total from f64 to f32 (`PLAN.md`, changelog). Energy per example is
  1.23x lower than rung 0's. Replicate 0 ran while the Mac was still busy (ruler p50
  145 ms against 127 to 135 ms); it gives 1.27, and the other four give 1.20 to 1.21.
  The gain over rung 11 is smaller here than with 1 thread (session t1), because at 4
  threads the work it speeds up is already split 4 ways.
- **Rung 13 is 1.22 [1.21, 1.23], the same as rung 12, as expected.** It changes only
  how the weights are loaded, so the forward pass is rung 12's and all 7,600 Tier 3
  logits are bit-identical. Its gain is memory: see `results/memory/t4_quiet/`. Its p99
  of 139 ms against rung 12's 292 ms is not from the change: 135 of its 150 cycles ran
  at nominal, against 30 for rung 12, and its ruler's p50 fell from 137 to 123 ms
  across the replicates. The paired speedup is steady through that, from 1.21 to 1.23.
- **Rung 14 at batch 1 is 1.22 [1.22, 1.22], the same as rung 13.** The batched code
  path with one sequence gives logits bit-identical to rung 13's.
- **Rungs 6 and 7 do not use 4 threads.** The naive loop is single-threaded, and
  rung 7's P-cluster clock (3.97 GHz, near the maximum) suggests Accelerate ran it on
  about one core. Rung 4's 2.85 GHz is the one rung loading all 4 cores.
- **p99 is noisy.** Several p99 figures vary more than 5% across replicates, so
  `analyze_bench.py` flags them. Trust p50 and the speedups more than p99.
- **Rung 6 has no energy figure.** Its ruler slices are 0.3 s, too short to hold whole
  power samples.
- **The raw replicates are gzipped.** `analyze_bench.py` reads `.json.gz` directly.

## Batched requests

Each batch config sends the same groups of B examples to both workers, so each slice is
a pair. Rung 0 pads each batch on the right to its longest prompt. Rung 14 packs the
prompts into one set of rows with no padding: linears, norms and GELU run over all rows,
and attention runs per sequence. Latency here is per batch, not per example. Throughput
is examples over the summed slice time. The warmup is one batch.

| Batch | Speedup vs rung 0 | Rung 0 examples/s | Rung 14 examples/s | Rung 14 batch p50 ms | Rung 14 batch p99 ms | Energy ratio | Speedup, nominal slices only (cycles) |
|---|---|---|---|---|---|---|---|
| 1 | 1.22 [1.22, 1.22] | 7.6 [7.55, 7.67] | 9.26 [9.2, 9.33] | 105 [104, 106] | 133 [132, 139] | 1.22 [1.22, 1.22] | 1.22 [1.21, 1.22] (92) |
| 4 | 1.36 [1.35, 1.37] | 8.65 [8.34, 8.88] | 11.8 [11.4, 12.2] | 331 [325, 349] | 435 [406, 500] | 1.38 [1.37, 1.39] | 1.36 [1.35, 1.38] (30) |
| 16 | 1.62 [1.59, 1.64] | 7.59 [7.43, 7.78] | 12.4 [12.3, 12.5] | 1.29e+03 [1.27e+03, 1.3e+03] | 1.42e+03 [1.4e+03, 1.43e+03] | 1.65 [1.62, 1.69] | 1.6 [1.53, 1.68] (22) |
| 64 | 1.98 [1.92, 2.05] | 6.5 [6.17, 6.81] | 13.1 [12.6, 13.6] | 4.73e+03 [4.69e+03, 5.12e+03] | 6.31e+03 [5.36e+03, 6.88e+03] | 1.97 [1.9, 2.05] | 2.01 [1.75, 2.38] (18) |

- **Rung 14 throughput rises with the batch: 9.3, 11.8, 12.4 and 13.1 examples/s** at
  batch 1, 4, 16 and 64, so batch 64 gives 1.42x over batch 1. Most of the gain is
  already there at batch 16.
- **Rung 0 does not gain from batching.** It peaks at 8.6 examples/s at batch 4 and
  falls to 6.5 at batch 64, because padding to the longest prompt (up to 348 tokens,
  median 152) wastes more work as the batch grows. So the speedup rises to 1.98x at
  batch 64, mostly from the padding that rung 14 skips.
- **Batch latency grows about linearly with the batch.** At batch 16, p50 is 1.29 s;
  at 64, 4.7 s. For a server, batch 16 keeps most of the throughput at a quarter of the
  latency.
- **Energy per example falls with the batch** for rung 14, from 1027 mJ at batch 1 to
  743 mJ at batch 64.
- **Few cycles ran at nominal thermal state** at the larger batches (30, 22 and 18 of
  150). The nominal-only speedups agree with the all-slice figures, with wider
  intervals.
