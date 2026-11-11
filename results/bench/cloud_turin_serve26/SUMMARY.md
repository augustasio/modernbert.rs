# Rung 26 (rounding inside the row loops): exact, about the same speed

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 2026-11-11, 07:56Z to 08:27Z, host load 22 to 42.
Rungs 24 and 26 built and run on the same pod.

**Bound: met.** Tier 3 logits equal rung 24's bit for bit on all 7,600 examples (batch
16), and on 64 examples on the batch-1 path. The refactored rung 24 also equals the
committed `results/rung24_logits.npy` bit for bit. `results/rung26_logits.npy`.

## Where the time moved (stage-times, 2 threads, batch 1, 64 examples)

| Stage (ms/example) | 24 r1 | 26 r1 | 24 r2 | 26 r2 |
|---|---|---|---|---|
| attention norm | 0.34 | 1.44 | 0.36 | 1.40 |
| add and MLP norm | 0.45 | 1.52 | 0.49 | 1.48 |
| GELU gate | 4.81 | 7.63 | 4.87 | 7.36 |
| Wqkv + Wi + MLP Wo (with any separate rounding) | 59.55 | 55.42 | 60.18 | 54.51 |
| separate rounding (inside the line above) | 5.97 | 1.12 | 5.98 | 1.10 |
| **layers total** | **93.02** | **94.33** | **94.00** | **92.40** |

The separate pass drops from 6.0 to 1.1 ms (the attention Wo and layer 0's Wqkv keep
it), and the norms and the GELU gate grow by about 5 ms. The rounding arithmetic itself
was the cost, not the extra pass over memory or the pool dispatch. At batch 1 the net
is within 1 ms either way.

## Behind the server (8x2, interleaved 26 then 24, three rounds)

p99 latency (ms); worker p50 in brackets.

| Rate/s | 26 r1 | 24 r1 | 26 r2 | 24 r2 | 26 r3 | 24 r3 |
|---|---|---|---|---|---|---|
| 40 | 166 (102) | 166 (105) | 175 (104) | 175 (105) | 169 (105) | 181 (107) |
| 45 | 222 (104) | 240 (110) | 182 (105) | 181 (109) | 202 (106) | 198 (109) |
| 50 | 199 | 212 | 203 | 228 | 215 | 230 |
| 55 | 222 | | 219 | 249 | | 236 |
| Best within 200 ms | 50 | 40 | 45 | 45 | 40 | 45 |

The worker p50 is 1 to 6 ms lower for rung 26 at every rate in all three rounds; with
eight workers sharing the caches, the row still in cache helps a little. The best rate
within the budget is a tie (135 against 130 summed over the rounds), inside this host's
noise. This host was busier than the final table's: rung 24 held 40 to 45/s here
against 55 to 60/s there.

**Rung 26 is exact and at best a few percent faster.** It stays the base for later rungs:
it is no slower, and its worker time is lower in every round.

Every serve run passed the in-run gate. Raw lines: `*.jsonl`; session log: `session.log`.
