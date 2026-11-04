# Session t1: batch 1, 1 thread

Run in two parts, on AC power, Low Power Mode off, with the worker builds of session t4.
Part 1: 2026-11-02 22:38 to 2026-11-03 07:02 UTC, the A/A floor and rungs 8, 7, 5, 4,
and one replicate of rung 1. It was stopped because the Mac had to sleep. Part 2:
2026-11-03 23:03 to 2026-11-04 02:15 UTC, the rest of rung 1 and rungs 3, 2, 6, with the
resumable runner (f5dd712). All 45 replicates passed the in-run correctness check. No
replicate was discarded for sleep. Each row is 5 replicates of 30 cycles of 64 examples,
rung 0 as the ruler in every cycle; rung 6 is 5 replicates of 8 cycles of 2 examples.
Intervals are 95%, from the two-stage bootstrap in `python/analyze_bench.py`. The table
comes from `python/bench_table.py`.

Speedup is rung 0 time over the variant's time: above 1 is faster than rung 0. The energy
ratio is rung 0 energy over the variant's, per example: above 1 uses less energy.

| Rung | Speedup vs rung 0 | Variant p50 ms | Variant p99 ms | Energy ratio | P-cluster GHz | Speedup, nominal slices only (cycles) |
|---|---|---|---|---|---|---|
| A/A: rung 0 against itself | 0.996 [0.952, 1.04] | 171 [160, 185] | 668 [375, 1.06e+03] | 1.01 [0.963, 1.05] | 3.51 [3.24, 3.77] | 0.999 [0.972, 1.02] (36) |
| 1: HF, head at the mask only | 1.12 [1.1, 1.14] | 140 [137, 145] | 354 [247, 491] | 1.14 [1.1, 1.18] | 3.89 [3.81, 3.94] | 1.13 [1.11, 1.15] (79) |
| 2: + torch.compile | 1.11 [1.11, 1.11] | 140 [139, 140] | 211 [206, 220] | 1.11 [1.11, 1.12] | 3.95 [3.95, 3.95] | 1.11 [1.11, 1.11] (150) |
| 3: ONNX Runtime | 0.832 [0.826, 0.838] | 188 [186, 190] | 327 [302, 371] | 1.05 [1.04, 1.07] | 3.96 [3.96, 3.96] | 0.832 [0.826, 0.838] (148) |
| 4: candle, CPU matmul | 0.136 [0.136, 0.137] | 1.17e+03 [1.16e+03, 1.17e+03] | 1.65e+03 [1.64e+03, 1.68e+03] | 0.157 [0.155, 0.159] | 4.03 [4.03, 4.03] | 0.136 [0.136, 0.137] (148) |
| 5: candle + Accelerate | 0.879 [0.877, 0.881] | 175 [174, 175] | 263 [250, 269] | 0.926 [0.922, 0.93] | 3.97 [3.97, 3.97] | 0.879 [0.877, 0.881] (148) |
| 6: our Rust, naive matmul | 0.0099 [0.00881, 0.0112] | 2.39e+04 [2.34e+04, 2.47e+04] | 3.11e+04 [2.7e+04, 3.55e+04] | n/a | n/a | 0.0099 [0.00881, 0.0112] (40) |
| 7: our Rust + Accelerate | 0.454 [0.452, 0.456] | 341 [339, 343] | 593 [585, 608] | 0.492 [0.488, 0.496] | 4 [3.99, 4] | 0.454 [0.452, 0.456] (134) |
| 8: + fusion, no allocation | 0.745 [0.74, 0.75] | 230 [211, 247] | 405 [354, 461] | 0.727 [0.72, 0.736] | 3.7 [3.42, 3.96] | 0.743 [0.734, 0.752] (12) |

## Notes

- **The A/A floor and rung 8 ran while the Mac was disturbed.** From 22:49 to 00:55 UTC
  on 2026-11-02, the rung 0 ruler slowed from 152 ms to 168 to 192 ms p50, and almost no
  cycle started at "nominal" (2 to 4 of 30 per replicate, against about 30 elsewhere).
  The PyTorch source-reading session ran until 23:01, and something unknown ran after
  that. So the A/A interval, [0.952, 1.04], is
  about 4 times wider than session t4's, and the A/A and rung 8 p50, p99 and GHz figures
  are inflated. The rung 8 **speedup** holds: its 5 replicates give 0.741 to 0.751,
  because each example is paired with the ruler in the same cycle. From rung 7 on the
  ruler is steady at 151 to 156 ms. Both rows should be rerun with the next batch of
  1-thread runs.
- **Threading is a small part of rung 8's gap.** With 1 thread, rung 0 takes about
  151 ms p50, against 137 ms with 4 threads, so its 4 threads gain only about 1.1x. Rung 8
  is 0.745 of rung 0 with 1 thread and 0.681 with 4. So with both on one core, rung 8 is
  still 1.34x slower. Most of the gap is per-core work: the f64 `exp` and `erf`, and
  the attention products outside sgemm (`notes/pytorch_cpu_research.md`, rungs 10 to 12).
- **Rung 7 to rung 8 is 1.64x**, close to the 1.67x of session t4.
- **Rung 5 is closer to rung 0 here (0.88 against 0.78 at 4 threads)**, and ONNX
  Runtime falls further behind (0.83 against 0.98). Candle CPU (rung 4) is 0.14, against
  0.24 at 4 threads, so it is the rung that uses the threads most.
- **Rung 6 is 0.0099, against 0.0080 at 4 threads.** It is single-threaded either way;
  only the ruler gets slower here.
- **Most slices ran at "nominal" in this session**, unlike t4, because one core makes
  less heat. The nominal-only column matches the main one.
- **The raw replicates are gzipped.** `analyze_bench.py` reads `.json.gz` directly.
