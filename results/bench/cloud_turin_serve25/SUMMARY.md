# Rung 25 (16 running maxima): exact, and slower

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 2026-11-11, 05:58Z to 06:34Z, host load 20 to 41.
Rungs 24 and 25 built and run on the same pod.

**Bound: met.** Tier 3 logits equal rung 24's bit for bit on all 7,600 examples (batch
16), and on 64 examples on the batch-1 path. `results/rung25_logits.npy`.

**Speed: worse.** A `stage-times` build at 2 threads, batch 1, 64 examples, two rounds
each (they agree within 0.2 ms):

| Stage | rung 24 | rung 25 |
|---|---|---|
| q8_quantize (ms) | 6.0 | 10.9 |
| att_softmax (thread-ms) | 9.4 | 11.4 |
| layers total (ms) | 94.2 | 99.8 |

Behind the server, 8x2, interleaved 25 then 24, three rounds. p99 latency (ms); worker
p50 in brackets.

| Rate/s | 25 r1 | 24 r1 | 25 r2 | 24 r2 | 25 r3 | 24 r3 |
|---|---|---|---|---|---|---|
| 40 | 176 (107) | 164 (100) | 170 (104) | 158 (100) | 173 (106) | 164 (98) |
| 45 | 184 | 164 | 205 | 176 | 178 | 166 |
| 50 | 209 | 187 | 213 | 199 | 202 | 179 |
| 55 | 224 | 189 | | 215 | 220 | 193 |
| 60 | | 226 | | 208 | | 234 |
| Best within 200 ms | 45 | 55 | 40 | 50 | 45 | 55 |

Rung 25 loses one to two rate steps in every round. The hypothesis was wrong: the compiler
already vectorised the single-maximum fold, and the hand-written 16-lane array compiles
to worse code. The softmax's 1 ns per value has another cause. Rung 25 stays in the
ladder as a recorded negative result; the next rungs build on rung 24, not on it.

Every serve run passed the in-run gate. Raw lines: `*.jsonl`; session log: `session.log`.
