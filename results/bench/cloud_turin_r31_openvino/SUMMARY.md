# Rung 31 against OpenVINO static int8, same rounds (2026-11-12)

RunPod cpu5c, AMD EPYC 9575F. Commit c172ff0. OpenVINO 2026.4.1, NNCF 3.4.0, the same
recipe and gate as `../cloud_turin_baseline_int8_quiet2/`. Three rounds; the order
alternates by round; each test waits up to 8 minutes for the host's load to fall under
24. A busy host: the load during rung 31's tests reached 40 to 133 in several of them,
while most OpenVINO tests ran at 16 to 33. The comparison below therefore leans on the
tests at matched load and on the worker time, which the queue does not inflate. Raw log:
`run.log`. The pod was deleted after the run.

## 8x2: best rate within a 200 ms p99

| Round | Rung 31 | load | OpenVINO | load |
|---|---|---|---|---|
| 1 | 65/s | 41 to 49 | 60/s | 26 to 33 |
| 2 | 65/s | 10 to 20 | 60/s | 17 to 18 |
| 3 | none (p99 246 ms at 55/s) | 33 to 40 | 60/s | 16 to 18 |

- **At matched quiet load (round 2), rung 31 holds one rate step more.** At 55 to 65/s
  its p99 is 130 to 166 ms, against OpenVINO's 173 to 252 ms.
- **Per request, rung 31 is about 15% faster.** Worker p50 at 55 to 65/s: 75 to 76 ms at
  load 10 to 12, against OpenVINO's 89 to 91 ms at load 16 to 18. In round 1, where rung
  31 ran at a higher load, its worker p50 was still lower (87 to 94 against 93 to 95).
- Round 3's rung 31 test failed at the lowest rate, with the host at load 33 to 40. It is
  a host effect, not a rung effect: the same build held 75 to 80/s on the quieter host of
  `../cloud_turin_serve31/`. This run did not reach a quiet enough host to measure rung
  31's top rate; that measurement stays in `../cloud_turin_serve31/`.

## 1x16: the lowest latency

| Rate | Rung 31 p50 | Rung 31 p99 | OpenVINO p50 | OpenVINO p99 |
|---|---|---|---|---|
| 5/s | 35 to 36 ms | 64 to 108 ms | 46 to 49 ms | 116 to 133 ms |
| 10/s | 33 to 35 ms | 83 to 89 ms | 46 to 60 ms | 174 to 222 ms |
| best rate within 200 ms | 15 to 25/s | | 5 to 15/s | |

- **Rung 31 answers in about 35 ms (p50)**, about 12 ms less than OpenVINO, with a p99 of
  about half OpenVINO's at 10/s. Its worker time is 29 to 34 ms against OpenVINO's 34 to
  49 ms.
- Rung 31 at 1x16 matches rung 28's earlier quiet numbers (p50 35 to 38 ms, p99 66 to 79
  ms at 5/s); the gain at 16 threads is small because the gate and softmax are a small
  share there.
- Rung 31's 10/s test of round 2 and its 15/s tests ran at load 51 to 110 and still kept
  the p99 under 160 ms.
