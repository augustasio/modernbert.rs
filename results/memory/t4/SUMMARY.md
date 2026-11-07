# Memory session t4: batch 1, 4 threads

Run on 2026-11-07, 03:51 to 04:07 UTC, at commit afe4c38, on AC power, Low Power Mode
off, with `python/memory_run.py`. Each row is 5 rounds. Each round is one fresh floor
process, which exits before reading weights, and one fresh full process, which loads and
answers 16 examples: the longest prompt (348 tokens) and 15 seeded others. Rung 6 answers
2. Every answer passed the logit gate of 1e-2. Cells are the median over the processes,
with the range [min, max]. Sizes are MiB. The table comes from `python/memory_table.py`.

| Rung | Floor RSS | Post-load RSS | Peak RSS | Activation delta | Major faults | Cold start s | Squeezed, left out |
|---|---|---|---|---|---|---|---|
| 0: HF pipeline (ruler) | 230 [229, 230] | 1840 [1840, 1840] | 2595 [2595, 2595] | 160 [145, 160] | 46 [41, 50] | 2.05 [1.93, 2.11] | 1 of 5 |
| 1: HF, head at the mask only | 300 [300, 300] | 1913 [1913, 1914] | 2668 [2668, 2668] | 26 [26, 26] | 44 [41, 48] | 2.18 [2.17, 2.23] | 0 of 5 |
| 2: + torch.compile, empty cache | 300 [300, 300] | 1968 [1902, 1973] | 2668 [2587, 2668] | 18 [15, 18] | 54206 [7662, 54245] | 27.16 [26.44, 27.58] | 0 of 5 |
| 2: + torch.compile, warm cache | 300 [300, 300] | 1832 [1768, 1884] | 2668 [2543, 2668] | 18 [18, 21] | 4098 [1318, 48950] | 11.49 [11.43, 12.26] | 0 of 5 |
| 3: ONNX Runtime | 45 [45, 45] | 1384 [1320, 1384] | 1484 [1484, 1491] | 100 [100, 171] | 40 [40, 85262] | 0.78 [0.76, 7.73] | 0 of 5 |
| 4: candle, CPU matmul | 2 [2, 2] | 2618 [2618, 2618] | 2691 [2690, 2692] | 73 [73, 74] | 0 [0, 0] | 1.34 [1.31, 1.42] | 1 of 5 |
| 5: candle + Accelerate | 6 [6, 6] | 2622 [2622, 2622] | 2691 [2690, 2691] | 69 [69, 69] | 2 [0, 59] | 0.65 [0.64, 0.69] | 0 of 5 |
| 6: our Rust, naive matmul | 2 [2, 2] | 575 [519, 2439] | 2059 [1882, 3153] | 866 [12, 885] | 0 [0, 5] | 56.17 [55.98, 56.54] | 0 of 5 |
| 7: our Rust + Accelerate | 6 [6, 6] | 679 [433, 2148] | 3102 [2777, 3145] | 1023 [20, 1256] | 2 [1, 2] | 2.22 [1.85, 2.38] | 1 of 5 |
| 8: + fusion, no allocation | 6 [6, 6] | 2364 [539, 2505] | 3161 [3009, 3168] | 4 [4, 1194] | 2 [0, 8] | 1.10 [1.08, 1.52] | 0 of 5 |
| 9: + f32 exp | 6 [6, 6] | 2382 [2272, 2554] | 3167 [3159, 3170] | 4 [4, 4] | 2 [0, 7] | 1.00 [0.99, 1.03] | 0 of 5 |
| 10: + attention in sgemm | 6 [6, 6] | 2300 [856, 2541] | 3161 [3069, 3163] | 4 [3, 1226] | 2 [0, 9] | 0.85 [0.82, 1.24] | 0 of 5 |
| 11: + threads outside matmul | 6 [6, 6] | 1989 [496, 2332] | 3127 [3064, 3165] | 224 [4, 1283] | 2 [0, 8] | 0.83 [0.74, 1.24] | 0 of 5 |
| 12: + f32 erf, norms, softmax sum | 6 [6, 6] | 2292 [1598, 2471] | 3168 [3147, 3173] | 4 [4, 781] | 2 [0, 8] | 0.75 [0.72, 0.94] | 0 of 5 |

## This session ran under memory pressure; read the columns differently

Other apps (Firefox, Cursor, a virtual machine) held most of the 16 GB. Before the
session, 7.6 GB of swap was in use and 6 GB was compressed, and 1.5 to 3 GB was free
before each process. Under pressure macOS compresses a process's pages, and they then
drop out of its resident size. So:

- **Floor RSS, peak RSS and cold start are sound.** The floor processes are small. The
  peak is the kernel's own high-water mark (`ru_maxrss`), and it agrees across the 5
  processes to within a few percent for every rung except 6 and 7.
- **Post-load RSS and the activation delta are sound for the Python rungs and rungs 4
  and 5, and not for rungs 6 to 12.** Our Rust rungs read as low as 433 MiB after
  loading, which is less than the 1,507 MiB of f32 weights they hold. That is only
  possible if pages were compressed. Their activation deltas are wrong for the same
  reason: a process whose pages come back from the compressor while it answers shows a
  large "delta" (781 to 1,283 MiB), and one that stays squeezed shows 4 MiB.
  The "Squeezed, left out" column drops only processes whose peak is low, so it does not
  catch this. These two columns need a rerun with the other apps closed.

## Notes

- **The floor is the most quotable number.** A Rust worker starts at 2 to 6 MiB. Python
  with torch and transformers imported starts at 230 to 300 MiB, and ONNX Runtime at
  45 MiB.
- **Our Rust rungs have the highest peak: about 3.1 GiB, against 2.5 GiB for rung 0.**
  The loader holds two f32 copies of the weights at once. `Checkpoint::from_file`
  decodes the whole file to f32 (1.5 GiB), and `ModernBert::from_checkpoint` then clones
  every tensor into the model (another 1.5 GiB) before the checkpoint is dropped. PLAN.md
  names this hazard: "accidental weight duplication, which transiently doubles peak".
  Candle (rungs 4 and 5) peaks at 2.6 GiB, and ONNX Runtime at 1.45 GiB.
- **Cold start: rung 12 is 0.75 s and rung 0 is 2.05 s.** That is process start to the
  first answer, with the weights already in the file cache (the cache cannot be emptied
  without `sudo purge`). Rung 5 is 0.65 s and ONNX Runtime 0.78 s. Rung 6 is 56 s,
  because its first answer takes most of a minute.
- **torch.compile costs 27 s of cold start with an empty cache, and 11.5 s with a warm
  one.** Its many major faults (7,662 to 54,245) are the compiler reading and writing
  its cache.
- **Rung 1's activation delta is 26 MiB; rung 0's is 160 MiB.** Rung 0 runs the full MLM
  head over every position, so it builds a [seq, 50,368] logit matrix that rung 1 skips.
- **The ONNX Runtime row has one slow cold start (7.7 s, 85,262 major faults)**, the
  first process, which read its model from disk. The other four took 0.76 to 0.85 s.
