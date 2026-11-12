# Rung 31 against rung 30 (2026-11-12)

RunPod cpu5c, AMD EPYC 9575F. Commit 5cca642. One pod, both rungs, interleaved. Host
load 12 to 30 for the Tier 3 and stage runs, and 15 to 41 during serving (the earlier
quiet runs: 8 to 24). The pod was deleted after the run.

## Tier 3: the bound holds

Rung 31's logits equal rung 30's, bit for bit, on all 7,600 examples. Rung 30 was rerun
on this pod and matches the committed `rung30_logits.npy`. Agreement with rung 0 is
7565/7600; calibrated accuracy is 84.92%. The 58 library tests pass on the pod, among
them `two_to_the_equals_the_cast`.

## Stage times: the gate and the softmax halve

Two threads on two physical cores, batch 1, the first 64 test prompts, three rounds,
the two builds interleaved (`../cloud_turin_stages31/`). Times in ms per example; the
`att_` and `gate_` lines are thread-ms (two threads).

| Stage | rung 30, rounds 1/2/3 | rung 31, rounds 1/2/3 |
|---|---|---|
| gelu_gate (wall) | 5.45 / 5.40 / 5.40 | 2.80 / 2.79 / 2.81 |
| gate_math | 9.51 / 9.44 / 9.42 | 4.23 / 4.24 / 4.24 |
| att_softmax | 9.34 / 9.31 / 9.28 | 4.72 / 4.72 / 4.72 |
| attention (wall) | 11.31 / 11.57 / 11.26 | 9.00 / 9.20 / 8.99 |
| q8_onednn (int8 matmuls) | 55.50 / 55.45 / 55.28 | 55.39 / 55.58 / 55.41 |
| layers total | 77.49 / 77.65 / 77.02 | 72.33 / 72.80 / 72.34 |

- **The layers get 6.6% faster** (5.1 to 5.3 ms per example in every round). Everything
  else is unchanged, the matmuls included.
- **The gate now matches OpenVINO's.** 2.8 ms wall at two threads, against OpenVINO's
  2.5 ms for its fused bf16 gate (`../cloud_turin_profile_openvino/`, another pod).
- **The machine code shows why** (`../cloud_turin_stages31/vector_code.txt`). In rung 30,
  `gelu_gate_row` has 17 scalar float-to-int converts and 36 conditional moves, and
  `softmax_in_place` 21 and 42. In rung 31 both have none of the converts, and 2 and 0
  conditional moves.
- The 64 logits of the stage builds are bit-identical too.

## Serving at 8x2: one to two rate steps

Open loop, 40 s per rate from 55/s in steps of 5, p99 budget 200 ms, three rounds.

| Round | Rung 31 best rate | p99 at that rate | Rung 30 best rate | p99 at that rate |
|---|---|---|---|---|
| 1 | 75/s | 159 ms | 70/s | 175 ms |
| 2 | 75/s | 186 ms | 70/s | 175 ms |
| 3 | 80/s | 165 ms | 70/s | 152 ms |

- **Rung 31 wins every round**, by one or two rate steps. The A/A floor of rung 28
  against itself is one step, so the gain is real but small.
- **Worker time per request falls**: a p50 of 73 to 82 ms at 55 to 75/s (two tests at
  load 34 and 40 excluded: 97 and 100 ms), against 78 to 94 ms for rung 30 at 55 to 70/s.
- **Per dollar:** 75 to 80/s is 241,000 to 257,000 examples per dollar, against 225,000
  for rung 30 at 70/s.
- Rung 31's round 2 missed the budget at 70/s (load 40, p99 1,269 ms), then held 75/s.
  The best rate is the highest rate that passes, as in every earlier run.
