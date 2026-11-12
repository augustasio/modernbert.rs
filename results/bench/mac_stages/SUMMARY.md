# Stage profile on the Mac: rungs 7 to 15, batch 1, 4 threads

Where one request's time goes, stage by stage, for every Mac rung from 7 to 15. Run on
2026-11-12, 03:57 to 04:13 UTC, at commit a02d844 (as logged in `session.log`), on AC power.

**This is analysis, not a benchmark.** Each rung is built with its own feature plus
`stage-times` (`cargo build --profile eval -p mb-model --features "<rung> stage-times" --bin
eval_agnews`). That build reads the clock around every stage of every layer
(`crates/mb-model/src/stage_times.rs`), so it runs a little slower than the benchmarked
build. Rung 0 is not run and nothing is paired. The latency figures of record stay in
`results/bench/t4/SUMMARY.md`.

How it was run:

- `eval_agnews` on the first 64 test prompts (`first64.tsv`, the first 64 lines of
  `data/agnews_test_prompt_ids.tsv`, sha256 in `first64.sha256`; 127 to 286 tokens,
  median 160.5), one example at a time (batch 1), with `MB_THREADS=4` and
  `VECLIB_MAXIMUM_THREADS=4`.
- Three rounds. In each round the eight rungs run one after another, each in a fresh
  process, with 20 s idle between runs and 120 s idle after the builds.
- Each cell is a stage's total wall time over the 64 examples divided by 64, in ms. This
  is a breakdown of where the time went, not a latency statistic. It includes the first,
  colder examples of each process.
- The tables give the **median over the three rounds**. Category rows are the median of
  the per-round category sums, so a column of medians need not add up exactly to the
  median total.
- `other` is the per-example wall time that `eval_agnews` prints (three significant
  digits, so about ±0.5 ms) minus the sum of the ten layer stages. It holds the
  embeddings, the final norm, the head and, for rung 15, the last layer's mask-row
  attention, which has no timer of its own.

## Thermal state

All 24 runs started and ended at **nominal**, on AC power (`session.log`). No run was at
"fair" or "serious".

## Median ms per example

| Stage | Rung 7 | Rung 8 | Rung 9 | Rung 10 | Rung 11 | Rung 12 | Rung 13 | Rung 15 |
|---|---|---|---|---|---|---|---|---|
| attn_norm | 5.41 | 5.17 | 5.07 | 5.04 | 2.47 | 1.22 | 1.33 | 1.22 |
| wqkv | 30.51 | 29.65 | 28.88 | 29.62 | 29.58 | 29.38 | 28.79 | 28.65 |
| attention | 241.44 | 81.39 | 72.00 | 23.66 | 9.47 | 8.24 | 7.91 | 7.59 |
| gather_heads | 0.00 | 0.00 | 0.00 | 0.00 | 0.41 | 0.41 | 0.41 | 0.39 |
| attn_wo | 10.78 | 9.93 | 9.71 | 9.92 | 9.99 | 9.97 | 9.62 | 9.34 |
| add_mlp_norm | 6.12 | 5.95 | 5.94 | 5.70 | 2.71 | 1.48 | 1.55 | 1.42 |
| wi | 43.11 | 42.62 | 41.74 | 42.40 | 42.19 | 42.43 | 41.44 | 40.05 |
| gelu_gate | 31.89 | 31.72 | 21.63 | 21.53 | 7.18 | 4.57 | 4.49 | 4.22 |
| mlp_wo | 25.46 | 25.70 | 25.00 | 25.39 | 25.54 | 25.63 | 24.50 | 23.88 |
| add | 0.46 | 0.47 | 0.47 | 0.47 | 1.28 | 1.27 | 1.37 | 1.18 |
| other | 3.57 | 0.71 | 0.91 | 0.82 | 0.18 | 0.79 | 0.17 | 0.31 |
| per_example | 398.00 | 233.00 | 211.00 | 165.00 | 132.00 | 127.00 | 122.00 | 118.00 |

| Category | Rung 7 | Rung 8 | Rung 9 | Rung 10 | Rung 11 | Rung 12 | Rung 13 | Rung 15 |
|---|---|---|---|---|---|---|---|---|
| Matmuls (Wqkv, Wo, Wi, MLP Wo) | 109.9 | 107.9 | 105.2 | 107.4 | 107.2 | 107.3 | 104.3 | 101.8 |
| Attention (RoPE, scores, softmax, P·V) | 241.4 | 81.4 | 72.0 | 23.7 | 9.9 | 8.7 | 8.3 | 8.0 |
| Norms and residual adds | 12.0 | 11.6 | 11.5 | 11.2 | 6.4 | 4.0 | 4.3 | 3.8 |
| GELU gate | 31.9 | 31.7 | 21.6 | 21.5 | 7.2 | 4.6 | 4.5 | 4.2 |
| Outside the layers | 3.6 | 0.7 | 0.9 | 0.8 | 0.2 | 0.8 | 0.2 | 0.3 |

Per-example time in each round (ms): rung 7: 398, 412, 397; rung 8: 233, 240, 230; rung 9: 210, 218, 211; rung 10: 165, 170, 163; rung 11: 131, 137, 132; rung 12: 127, 128, 125; rung 13: 122, 121, 122; rung 15: 129, 118, 118

Stage meanings (the indices of `stage_times::STAGES`, called from `model.rs`,
`attention.rs` and `mlp.rs` for rung 7, from `fused.rs` for rungs 8 to 10, and from
`fused/threads.rs` for rungs 11 to 15):

- `attn_norm`: the LayerNorm before attention (absent in layer 0).
- `wqkv`, `attn_wo`, `wi`, `mlp_wo`: the four `cblas_sgemm` calls of a layer.
- `attention`: everything between Wqkv and Wo. Rung 7: RoPE on Q and K, then the
  per-head score, softmax and `P V` loops. Rungs 8 to 10: the per-head scatter with RoPE,
  then attention. Rungs 11 to 15: the same, split by head over the threads.
- `gather_heads`: rungs 11 to 15 copy each head's `[seq, 64]` output into the context.
- `add_mlp_norm`: the residual add after attention and the LayerNorm before the MLP.
- `gelu_gate`: GELU on the first half of Wi's output, times the second half.
- `add`: the residual add after the MLP.

## Notes

- **Rung 7 spends 241 of 398 ms (61%) in attention, and 110 ms (28%) in the four
  matmuls.** Attention is under 3% of the request's multiply-adds. The GELU gate takes
  another 32 ms, and the norms and adds 12.
- **Rung 15 spends 102 of 118 ms (86%) in the four matmuls.** Attention is 8 ms, the GELU
  gate 4, and the norms and adds 4.
- **The matmuls barely move from rung 7 to rung 15: 110 to 102 ms.** Every rung calls the
  same `cblas_sgemm`. Over these 64 prompts the four linear layers cost about 56.6
  billion multiply-adds per example (55.1 billion for rung 15, whose last layer runs Wo
  and the MLP on the mask row only). So the matmuls run at about 1.03 TFLOP/s (rung 7)
  to 1.08 TFLOP/s (rung 15). `python/BLAS.md` measured 1.17 TFLOP/s for a one-thread
  2048^3 sgemm on this Mac.
- **Attention: 241, 81, 72, 24, 9.9, 8.7, 8.3, 8.0 ms** from rung 7 to rung 15. Layout
  (rung 8) removes 160 ms, f32 `exp` (rung 9) 9, `sgemm` for the products (rung 10) 48,
  and threads (rung 11) 14.
- **GELU gate: 32, 32, 22, 22, 7.2, 4.6, 4.5, 4.2 ms.** f32 `exp` inside `erf` saves 10
  ms (rung 9), threads 14 (rung 11), and f32 `erf` 2.6 (rung 12).
- **Norms and adds: about 12 ms up to rung 10, 6.4 at rung 11, 4.0 at rung 12.** The
  plain residual `add` grows from 0.47 to 1.28 ms at rung 11: split across threads, so
  small a step costs more than it saves.
- **Rung 15's round 1 took 129 ms per example, against 118 in rounds 2 and 3.** The
  extra 10 ms is inside the matmuls (Wi 46.1 against 40.1 ms, Wqkv 30.4 against 28.6),
  although the run was at nominal. The median is not affected.
- `o.json` is the logits' sidecar from the last run; the script deleted the logits.
