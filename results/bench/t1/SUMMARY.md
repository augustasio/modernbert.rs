# Session t1: batch 1, 1 thread

Run in six parts, on AC power, Low Power Mode off, with the worker builds of session t4.
Part 1: 2026-11-02 22:38 to 2026-11-03 07:02 UTC, rungs 7, 5, 4, and one replicate of
rung 1. It was stopped because the Mac had to sleep. Part 2: 2026-11-03 23:03 to
2026-11-04 02:15 UTC, the rest of rung 1 and rungs 3, 2, 6, with the resumable runner
(f5dd712). Part 3: 2026-11-05 00:25 to 04:16 UTC, overnight with the Mac idle, the A/A
floor and rungs 8, 9, 10. Part 4: 2026-11-05 08:50 to 09:41 UTC, rung 11. Part 5:
2026-11-05 19:58 to 20:49 UTC, rung 12, at commit 465c39e. Part 6: 2026-11-07 18:03 to 18:50 UTC, rung 13, at commit
c653da9. The A/A floor and rung 8 from part 1 ran while the Mac was
disturbed, so part 3 reran them; the old runs are in `results/bench/t1_disturbed/`. All
75 replicates in the table passed the in-run correctness check. No replicate was
discarded for sleep. Each row is 5 replicates of 30 cycles of 64 examples,
rung 0 as the ruler in every cycle; rung 6 is 5 replicates of 8 cycles of 2 examples.
Intervals are 95%, from the two-stage bootstrap in `python/analyze_bench.py`. The table
comes from `python/bench_table.py`.

Speedup is rung 0 time over the variant's time: above 1 is faster than rung 0. The energy
ratio is rung 0 energy over the variant's, per example: above 1 uses less energy.

| Rung | Speedup vs rung 0 | Variant p50 ms | Variant p99 ms | Energy ratio | P-cluster GHz | Speedup, nominal slices only (cycles) |
|---|---|---|---|---|---|---|
| A/A: rung 0 against itself | 1 [0.993, 1.01] | 157 [156, 158] | 347 [299, 397] | 0.999 [0.985, 1.01] | 3.95 [3.94, 3.95] | 1 [0.998, 1.01] (129) |
| 1: HF, head at the mask only | 1.12 [1.1, 1.14] | 140 [137, 145] | 354 [247, 491] | 1.14 [1.1, 1.18] | 3.89 [3.81, 3.94] | 1.13 [1.11, 1.15] (79) |
| 2: + torch.compile | 1.11 [1.11, 1.11] | 140 [139, 140] | 211 [206, 220] | 1.11 [1.11, 1.12] | 3.95 [3.95, 3.95] | 1.11 [1.11, 1.11] (150) |
| 3: ONNX Runtime | 0.832 [0.826, 0.838] | 188 [186, 190] | 327 [302, 371] | 1.05 [1.04, 1.07] | 3.96 [3.96, 3.96] | 0.832 [0.826, 0.838] (148) |
| 4: candle, CPU matmul | 0.136 [0.136, 0.137] | 1.17e+03 [1.16e+03, 1.17e+03] | 1.65e+03 [1.64e+03, 1.68e+03] | 0.157 [0.155, 0.159] | 4.03 [4.03, 4.03] | 0.136 [0.136, 0.137] (148) |
| 5: candle + Accelerate | 0.879 [0.877, 0.881] | 175 [174, 175] | 263 [250, 269] | 0.926 [0.922, 0.93] | 3.97 [3.97, 3.97] | 0.879 [0.877, 0.881] (148) |
| 6: our Rust, naive matmul | 0.0099 [0.00881, 0.0112] | 2.39e+04 [2.34e+04, 2.47e+04] | 3.11e+04 [2.7e+04, 3.55e+04] | n/a | n/a | 0.0099 [0.00881, 0.0112] (40) |
| 7: our Rust + Accelerate | 0.454 [0.452, 0.456] | 341 [339, 343] | 593 [585, 608] | 0.492 [0.488, 0.496] | 4 [3.99, 4] | 0.454 [0.452, 0.456] (134) |
| 8: + fusion, no allocation | 0.758 [0.751, 0.765] | 211 [209, 214] | 367 [337, 402] | 0.737 [0.726, 0.748] | 3.97 [3.95, 3.98] | 0.76 [0.752, 0.767] (113) |
| 9: + f32 exp | 0.827 [0.824, 0.83] | 191 [191, 192] | 303 [292, 319] | 0.837 [0.832, 0.843] | 3.97 [3.97, 3.97] | 0.827 [0.824, 0.83] (150) |
| 10: + attention in sgemm | 1.07 [1.07, 1.07] | 147 [146, 147] | 224 [217, 240] | 1.13 [1.12, 1.14] | 3.95 [3.95, 3.95] | 1.07 [1.07, 1.07] (136) |
| 11: + threads outside matmul | 1.04 [1.04, 1.05] | 144 [143, 144] | 215 [213, 227] | 1.13 [1.12, 1.14] | 3.96 [3.96, 3.96] | 1.04 [1.04, 1.05] (150) |
| 12: + f32 erf, norms, softmax sum | 1.22 [1.19, 1.24] | 126 [124, 128] | 337 [212, 517] | 1.21 [1.16, 1.24] | 3.9 [3.84, 3.94] | 1.22 [1.2, 1.25] (101) |
| 13: + weights moved, not copied | 1.22 [1.22, 1.23] | 124 [124, 124] | 169 [168, 170] | 1.21 [1.2, 1.21] | 3.95 [3.95, 3.95] | 1.22 [1.22, 1.23] (150) |

## Notes

- **Rung 10 is faster than rung 0 with 1 thread: 1.07 [1.07, 1.07].** p50 is 147 ms,
  against 157 ms for the ruler, and energy per example is 1.13x lower. All 5 replicates
  give 1.067 to 1.073. Rung 1 (HF with the head at the mask only) is still 1.12, so
  rung 10 is about 4% behind it.
- **Rung 11 on 1 thread is rung 10, as it should be.** Its speedup is 1.04
  [1.04, 1.05] against rung 10's 1.07, but its own p50 is 144 ms against rung 10's
  147 ms. The ratio is lower because the ruler ran faster in part 4 (149 to 151 ms p50)
  than in part 3 (156 to 158 ms). So a cross-session gap of about 3% in a speedup can
  come from the ruler alone. Compare rungs from different parts by their own p50 too.
- **Rung 12 is 1.22 [1.19, 1.24] with 1 thread, ahead of rung 1 (1.12).** p50 is
  125 ms, against 144 ms for rung 11, so f32 arithmetic outside the matmuls is worth
  1.15x on one core. Four replicates give 1.22; replicate 2 gives 1.20. Its p99 is
  noisy (211 to 536 ms across replicates); p50 is steady.
- **Rung 13 is 1.22 [1.22, 1.23] with 1 thread, the same as rung 12.** It changes only
  the load path, so this is expected. Its p50 is 124 ms; every cycle ran at nominal, and
  the five replicates agree to within 0.003.
- **Rung 12 gains only 1.14x from 4 threads (125 ms to 110 ms); rung 0 gains 1.15x.**
  So rung 12 now scales with threads about as well as PyTorch does.
- **Rung 9 is 1.09x faster than rung 8, and rung 10 is 1.29x faster than rung 9.**
  Moving the attention products into sgemm is the biggest single step since rung 8.
- **The A/A floor is about ±1% again**: 1.00 [0.993, 1.01], with the ruler at 156 to
  158 ms p50 in every replicate. The disturbed part 1 run gave [0.952, 1.04].
- **The rerun moves rung 8 a little**: 0.758 [0.751, 0.765], against 0.745 [0.74, 0.75]
  while the Mac was disturbed. The pairing held most of the speedup, but not all.
- **Rung 10 gains nothing from 4 threads; rung 0 gains 1.15x.** Rung 10's p50 is 147 ms
  with 1 thread and 146 ms with 4. Rung 0's is 157 ms and 137 ms. So at 4 threads our
  remaining gap to rung 0 is threading, not per-core work. The non-matmul loops are
  single-threaded, and the per-head sgemm calls are too small for Accelerate to split.
- **Rung 7 to rung 8 is 1.67x**, the same as in session t4.
- **Rung 5 is closer to rung 0 here (0.88 against 0.78 at 4 threads)**, and ONNX
  Runtime falls further behind (0.83 against 0.98). Candle CPU (rung 4) is 0.14, against
  0.24 at 4 threads, so it is the rung that uses the threads most.
- **Rung 6 is 0.0099, against 0.0080 at 4 threads.** It is single-threaded either way;
  only the ruler gets slower here.
- **Most slices ran at "nominal" in this session**, unlike t4, because one core makes
  less heat. The nominal-only column matches the main one.
- **The raw replicates are gzipped.** `analyze_bench.py` reads `.json.gz` directly.
