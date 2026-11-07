# Memory session t4_quiet: batch 1, 4 threads, other apps closed

Run on 2026-11-07, 12:23 to 12:37 UTC, at commit b6f742b (the memory code is unchanged
since afe4c38), on AC power, Low Power Mode off, with `python/memory_run.py`. This is a
rerun of session `t4`, which ran under memory pressure. Before this run Docker Desktop,
Firefox and Cursor were quit: 3.6 GB was free, 4.2 GB inactive, the compressor held
1.9 GB and 1.2 GB of swap was in use. No process was squeezed.

Each row is 5 rounds. Each round is one fresh floor process, which exits before reading
weights, and one fresh full process, which loads and answers 16 examples: the longest
prompt (348 tokens) and 15 seeded others. Rung 6 answers 2. Every answer passed the
logit gate of 1e-2. Cells are the median over the processes, with the range [min, max].
Sizes are MiB. The table comes from `python/memory_table.py`.

| Rung | Floor RSS | Post-load RSS | Peak RSS | Activation delta | Major faults | Cold start s | Squeezed, left out |
|---|---|---|---|---|---|---|---|
| 0: HF pipeline (ruler) | 230 [229, 230] | 1840 [1840, 1841] | 2595 [2595, 2595] | 160 [160, 160] | 41 [41, 48971] | 1.93 [1.87, 3.40] | 0 of 5 |
| 1: HF, head at the mask only | 300 [300, 300] | 1914 [1913, 1914] | 2668 [2668, 2668] | 26 [26, 26] | 41 [41, 48] | 2.18 [2.11, 2.22] | 0 of 5 |
| 2: + torch.compile, empty cache | 300 [300, 300] | 2212 [2211, 2212] | 2668 [2668, 2668] | 9 [9, 10] | 3893 [3885, 11326] | 23.50 [23.05, 25.29] | 0 of 5 |
| 2: + torch.compile, warm cache | 300 [300, 300] | 2120 [2120, 2120] | 2668 [2668, 2668] | 9 [9, 13] | 442 [384, 48802] | 10.14 [10.02, 11.83] | 0 of 5 |
| 3: ONNX Runtime | 45 [45, 45] | 1384 [1384, 1384] | 1484 [1484, 1484] | 100 [100, 100] | 40 [40, 85070] | 0.79 [0.76, 6.57] | 0 of 5 |
| 4: candle, CPU matmul | 2 [2, 2] | 2618 [2618, 2618] | 2690 [2689, 2692] | 72 [72, 74] | 0 [0, 56] | 1.27 [1.26, 1.32] | 0 of 5 |
| 5: candle + Accelerate | 6 [6, 6] | 2622 [2622, 2622] | 2691 [2690, 2691] | 69 [68, 69] | 0 [0, 55] | 0.61 [0.61, 0.62] | 0 of 5 |
| 6: our Rust, naive matmul | 2 [2, 2] | 2441 [2278, 2486] | 3158 [3157, 3163] | 12 [12, 12] | 0 [0, 5] | 52.86 [52.47, 53.89] | 0 of 5 |
| 7: our Rust + Accelerate | 6 [6, 6] | 2486 [2276, 2523] | 3162 [3159, 3166] | 17 [17, 17] | 0 [0, 5] | 1.67 [1.67, 1.67] | 0 of 5 |
| 8: + fusion, no allocation | 6 [6, 6] | 2432 [2251, 2520] | 3162 [3161, 3166] | 4 [4, 4] | 0 [0, 6] | 0.91 [0.91, 0.91] | 0 of 5 |
| 9: + f32 exp | 6 [6, 6] | 2334 [2309, 2515] | 3164 [3161, 3169] | 4 [4, 4] | 0 [0, 7] | 0.85 [0.84, 0.86] | 0 of 5 |
| 10: + attention in sgemm | 6 [6, 6] | 2365 [2339, 2495] | 3160 [3158, 3165] | 3 [3, 4] | 0 [0, 5] | 0.69 [0.68, 0.70] | 0 of 5 |
| 11: + threads outside matmul | 6 [6, 6] | 2466 [2334, 2524] | 3162 [3160, 3168] | 4 [4, 4] | 0 [0, 8] | 0.61 [0.60, 0.61] | 0 of 5 |
| 12: + f32 erf, norms, softmax sum | 6 [6, 6] | 2344 [2272, 2550] | 3164 [3157, 3169] | 4 [4, 4] | 0 [0, 8] | 0.60 [0.59, 0.60] | 0 of 5 |

## Notes

- **This session replaces `t4` for every column.** In `t4` our Rust rungs read as low as
  433 MiB after load, because macOS had compressed their pages. Here every Rust process
  reads 2,251 to 2,550 MiB after load, and every activation delta is 3 to 17 MiB. The
  floor, the peak and the Python rows agree with `t4` to within a few MiB.
- **The floor: a Rust worker starts at 2 to 6 MiB.** Python with torch and transformers
  imported starts at 230 to 300 MiB, and ONNX Runtime at 45 MiB.
- **Our Rust rungs have the highest peak: about 3,160 MiB, against 2,595 for rung 0.**
  The peak comes during load, not while answering. The loader holds two f32 copies of
  the weights at once: `Checkpoint::from_file` decodes the whole file to f32 (1,507 MiB),
  and `ModernBert::from_checkpoint` then clones every tensor into the model before the
  checkpoint is dropped. PLAN.md names this hazard: "accidental weight duplication,
  which transiently doubles peak". Candle peaks at 2,690 MiB, and ONNX Runtime at 1,484.
- **After load, our Rust rungs still hold 750 to 1,040 MiB more than the weights.** The
  weights are 1,507 MiB and the floor is 6, but post-load is 2,251 to 2,550 MiB, and the
  amount differs between processes of the same binary. The physical footprint matches
  the resident size to within 4 MiB, so this is memory the process still owns, not freed
  pages that macOS has yet to reclaim. The cause is not verified. One candidate is that
  the allocator keeps part of the dropped checkpoint copy instead of returning it to the
  OS. Candle also holds 2,618 MiB after load, about 1.1 GiB above the weights.
- **Our Rust rungs need the least memory while answering: 3 to 4 MiB for rungs 8 to 12.**
  Rung 8 preallocates every buffer, so answering allocates almost nothing. Rungs 6 and 7
  allocate per pass and need 12 to 17 MiB. Candle needs 69 to 72 MiB, ONNX Runtime 100
  MiB, and rung 0 160 MiB.
- **Rung 1's activation delta is 26 MiB; rung 0's is 160 MiB.** Rung 0 runs the full MLM
  head over every position, so it builds a [seq, 50,368] logit matrix that rung 1 skips.
- **Cold start: rung 12 is 0.60 s and rung 0 is 1.93 s.** That is process start to the
  first answer, with the weights already in the file cache (the cache cannot be emptied
  without `sudo purge`). Rung 5 is 0.61 s and ONNX Runtime 0.79 s. Rung 6 is 53 s,
  because its first answer takes most of a minute. Each Rust rung starts faster than the
  one before it, because the cold start includes the first answer.
- **torch.compile costs 23.5 s of cold start with an empty cache, and 10.1 s with a warm
  one.** Its major faults are the compiler reading and writing its cache.
- **Single slow first processes.** The first ONNX Runtime process took 6.57 s (85,070
  major faults), and one rung 0 process had 48,971 major faults and a 3.40 s cold start.
  Both read files from disk that the later processes found in the file cache. The
  medians are not affected.
