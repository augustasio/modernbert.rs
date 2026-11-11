# The ladder behind the server, at its best layout (8x2)

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 32 vCPUs = 16 cores, scattered over the host's
128 hardware threads, 2026-11-11, 04:12Z to 04:46Z. Open loop, batch 1, 8 workers of 2
threads (one thread per core), 40 s per rate, two rounds in the order 24, 19, 23, 22.
The orchestrator accepted the host only at load under 40 and model 9575F (see the trap
below). Host load was 12 to 47 in round 1 and 37 to 53 in round 2.

p99 latency (ms) by arrival rate; worker p50 in brackets.

| Rate/s | 19 r1 | r2 | 22 r1 | r2 | 23 r1 | r2 | 24 r1 | r2 |
|---|---|---|---|---|---|---|---|---|
| 15 | 197 (133) | 212 (143) | | | | | | |
| 20 | 198 | 215 | 174 (115) | 183 (124) | | | | |
| 25 | 212 | | 173 | 176 | | | | |
| 30 | 216 | | 188 | 217 | 151 (91) | 149 (98) | 147 (90) | 140 (93) |
| 35 | | | 197 | 254 | 146 | 153 | 142 | 150 |
| 40 | | | 223 | | 148 | 169 | 151 | 151 |
| 45 | | | 211 | | 174 | 204 | 156 | 171 |
| 50 | | | | | 181 | 215 | 162 | 237 |
| 55 | | | | | 175 | | 162 | 382 |
| 60 | | | | | 220 | | 187 | |
| 65 | | | | | 246 | | 219 | |
| 70 | | | | | | | 266 | |

Best rate within the 200 ms budget, and examples per dollar at $1.12/h (rate x 3600 /
1.12):

| Rung | r1 | r2 | Examples per dollar (r1) |
|---|---|---|---|
| 19 (omp-pool, fp32) | 20/s | none | about 64,000 |
| 22 (int8-partial) | 35/s | 25/s | about 112,000 |
| 23 (int8-smooth) | 55/s | 40/s | about 177,000 |
| 24 (int8-u8) | 60/s | 45/s | about 193,000 |

Round 1 ran at host load 12 to 47; round 2 at 37 to 53, above the orchestrator's
threshold, and every rung lost 5 to 15/s there. Read round 1 as the result and round 2
as what a busier neighbour costs. Rungs 23 and 24 differ by one rate step here, inside
the host noise; the earlier interleaved runs found them equal.

- **Rung 19** (fp32) takes about 133 ms per request on 2 threads, so the p99 is at the
  budget before a queue forms. 8x2 does not help it: 4x4 also gave about 20/s.
- **Int8 is what makes 8x2 work.** At about 90 ms per request, a request fits in half
  the budget, so eight of them at once can absorb the queue.

## The EPYC 4564P trap

The `cpu5c` flavor sometimes hands out an AMD EPYC 4564P (Zen 4, 16 cores, CPUs 0 to 31)
instead of the 9575F. It is idle (load near 0) but about 1.5x slower per request, and
every rung misses the budget on it. Two earlier attempts of this table landed there and
were discarded. The orchestrator now reads `lscpu` and deletes any pod that is not a
9575F. All earlier committed cloud results are from 128-thread hosts, so from the 9575F.

Every run passed the in-run gate. Raw lines: `*.jsonl`; session log: `session.log`.
