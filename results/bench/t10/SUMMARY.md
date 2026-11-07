# Session t10: batch 1, 10 threads

Run in two parts, on AC power, Low Power Mode off, with the worker builds of session t4.
Part 1: 2026-11-05 22:19 to 2026-11-06 08:08 UTC, the A/A floor and rungs 12, 11, 10,
9, 8. It was stopped during rung 7 because the Mac was needed; that replicate was
discarded. Part 2: 2026-11-06 19:28 to 2026-11-07 03:41 UTC, rungs 7, 5, 4, 1, 3, 2, 6,
with the resumable runner. Both at commit 289161d. All 65 replicates passed the in-run
correctness check. No replicate was discarded for sleep. Each row is 5 replicates of 30
cycles of 64 examples, rung 0 as the ruler in every cycle; rung 6 is 5 replicates of 8
cycles of 2 examples. Intervals are 95%, from the two-stage bootstrap in
`python/analyze_bench.py`. The table comes from `python/bench_table.py`.

Speedup is rung 0 time over the variant's time: above 1 is faster than rung 0. The energy
ratio is rung 0 energy over the variant's, per example: above 1 uses less energy.

| Rung | Speedup vs rung 0 | Variant p50 ms | Variant p99 ms | Energy ratio | P-cluster GHz | Speedup, nominal slices only (cycles) |
|---|---|---|---|---|---|---|
| A/A: rung 0 against itself | 1 [0.973, 1.03] | 150 [140, 165] | 716 [395, 1.04e+03] | 1.01 [0.972, 1.05] | 3.65 [3.35, 3.9] | 1 [0.997, 1.01] (52) |
| 1: HF, head at the mask only | 1.12 [1.1, 1.12] | 128 [125, 137] | 280 [200, 327] | 1.11 [1.1, 1.12] | 3.77 [3.6, 3.88] | 1.12 [1.11, 1.13] (21) |
| 2: + torch.compile | 1.12 [1.12, 1.13] | 122 [121, 126] | 192 [181, 228] | 1.1 [1.09, 1.11] | 3.86 [3.79, 3.9] | 1.13 [1.11, 1.14] (30) |
| 3: ONNX Runtime | 0.869 [0.826, 0.915] | 169 [159, 179] | 520 [452, 568] | 0.721 [0.695, 0.748] | 2.7 [2.54, 2.86] | 1.01 [0.976, 1.05] (12) |
| 4: candle, CPU matmul | 0.288 [0.281, 0.297] | 595 [553, 661] | 1.63e+03 [1.06e+03, 2.56e+03] | 0.287 [0.279, 0.297] | 2.33 [2.12, 2.53] | 0.348 [0.323, 0.376] (2) |
| 5: candle + Accelerate | 0.889 [0.874, 0.907] | 199 [191, 207] | 421 [365, 459] | 0.893 [0.88, 0.911] | 3.28 [3.17, 3.4] | 0.835 [0.819, 0.857] (7) |
| 6: our Rust, naive matmul | 0.00963 [0.00871, 0.0105] | 2.41e+04 [2.28e+04, 2.48e+04] | 2.86e+04 [2.72e+04, 2.86e+04] | n/a | n/a | 0.00963 [0.00871, 0.0105] (40) |
| 7: our Rust + Accelerate | 0.464 [0.45, 0.477] | 364 [357, 373] | 704 [665, 739] | 0.465 [0.451, 0.481] | 3.72 [3.62, 3.83] | 0.426 [0.416, 0.435] (18) |
| 8: + fusion, no allocation | 0.855 [0.805, 0.908] | 589 [565, 609] | 867 [839, 898] | 0.882 [0.836, 0.93] | 3.63 [3.57, 3.68] | 0.687 [0.634, 0.728] (6) |
| 9: + f32 exp | 0.909 [0.854, 0.968] | 459 [373, 507] | 754 [733, 779] | 0.926 [0.874, 0.977] | 3.74 [3.73, 3.76] | 0.78 [0.732, 0.832] (12) |
| 10: + attention in sgemm | 1.06 [1.03, 1.1] (no detectable difference) | 156 [154, 159] | 588 [548, 616] | 1.2 [1.14, 1.26] | 3.9 [3.89, 3.92] | 1.04 [1, 1.1] (43) |
| 11: + threads outside matmul | 1.19 [1.14, 1.23] | 137 [134, 140] | 619 [586, 640] | 1.19 [1.13, 1.24] | 3.79 [3.77, 3.82] | 1.19 [1.13, 1.28] (23) |
| 12: + f32 erf, norms, softmax sum | 1.24 [1.2, 1.28] | 133 [131, 137] | 620 [567, 1.09e+03] | 1.26 [1.2, 1.31] | 3.68 [3.49, 3.8] | 1.21 [1.16, 1.27] (15) |

## Notes

- **Rungs 8 and 9 were disturbed; do not quote them.** In their replicates the ruler's
  p50 is 258 to 603 ms, against 135 to 214 ms in every other row, and the P-cluster
  clock is normal (3.6 to 3.8 GHz). So something else was using the cores, not heat.
  Later, at 04:43 UTC, `mediaanalysisd` (the macOS photo-analysis daemon, which runs
  when the Mac is idle) was using 115% CPU, and it is the likely cause. With 10 threads
  every core is busy, so any other load lands on the benchmark. The pairing keeps the
  ratio closer than the absolute times, but not close enough: rung 8 gives 0.79 to 0.93
  across replicates. These two rows are to be rerun.
- **The A/A floor is wider here: 1.00 [0.973, 1.03].** Replicates 3 and 4 ran warmer
  and slower (ruler p50 183 and 165 ms, P-cluster 3.16 and 3.45 GHz); the other three
  give 138 to 149 ms. The nominal-only column is 1.00 [0.997, 1.01].
- **Rung 12 is the fastest rung at 10 threads too: 1.24 [1.20, 1.28].** p50 is 133 ms.
  Rung 11 is 1.19, and rungs 1 and 2 are 1.12. Energy per example is 1.26x lower than
  rung 0's.
- **10 threads is slower than 4 for every fast rung.** Rung 12's p50 is 110 ms at 4
  threads and 133 ms at 10; rung 0's ruler is about 130 ms at 4 and 140 to 170 ms at 10.
  The 6 extra cores are efficiency cores, and the work waits for the slowest thread.
  `PLAN.md` predicted this.
- **p99 is high for most rungs** (about 600 ms for rungs 10 to 12), for the same reason:
  a slice whose threads land on efficiency cores, or behind another process, is slow.
  Trust p50 and the speedups more than p99.
- **Rung 3 (ONNX Runtime) loses most from 10 threads**: 0.87, against 0.98 at 4
  threads, with the P-cluster at 2.7 GHz. Its nominal-only figure is 1.01, on 12 cycles.
- **Rung 4 (candle CPU) is 0.29**, against 0.24 at 4 threads: it is the rung that uses
  extra cores best, but it is still the slowest real rung.
- **Rung 6 has no energy figure.** Its ruler slices are 0.3 s, too short to hold whole
  power samples.
- **The raw replicates are gzipped.** `analyze_bench.py` reads `.json.gz` directly.
