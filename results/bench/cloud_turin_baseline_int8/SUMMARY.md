# The int8 baselines on a busy host (2026-11-12)

RunPod cpu5c, AMD EPYC 9575F, 16 cores. Commit 1d0e5a8. `session.log` is the pod's log.

**The serving numbers here are not clean.** The host passed the load check when the pod
started (load 8), but from the second test on its 1-minute load was 60 to 90, against
about 16 from our own server. Neighbours held cores for the whole run, and round 2 was
cut short at 03:54Z by the orchestrator's time limit (rung 28 at 4x4 in round 2 and all
round-2 baselines are missing). The clean comparison is in `cloud_turin_baseline_int8_quiet`.

What still stands from this run:

## Tier 3 (all 7,600 test examples, against rung 0)

| Baseline | Agrees with rung 0 | Calibrated accuracy | Max abs logit diff |
|---|---|---|---|
| PyTorch dynamic int8, per channel | 6770 (89.1%) | 71.09% | 12.2 |
| ONNX Runtime dynamic int8, per channel | 7218 (95.0%) | 85.01% | 9.38 |
| OpenVINO NNCF static int8 (transformer, 300 train prompts) | 7493 (98.6%) | 86.12% | 9.06 |
| Rung 28 (ours) | 7565 (99.5%) | 84.92% | 0.81 |

Rung 0 is 84.78%. PyTorch's stock dynamic int8 loses 13.7 points: it would fail every
int8 rung's bound by far. ONNX Runtime and OpenVINO keep accuracy; their logits move
about ten times as far as rung 28's, so the serving gate for each baseline is set from
its own Tier 3 result (4x its max difference, 2x its disagreements), not from rung 28's.

## Serving, same conditions within the round (round 1, load 60 to 90)

Best rate with p99 at or under 200 ms:

| Layout | rung 28 | OpenVINO | PyTorch | ONNX Runtime |
|---|---|---|---|---|
| 1x16 | 10 | 10 | 5 | none |
| 2x8 | 20 | 15 | 30 | none |
| 4x4 | 30 | 30 | 25 | none |
| 8x2 | 45 to 55 | 25 | 5 | none |

ONNX Runtime's worker time is bimodal at every layout (p50 60 to 145 ms, p90 above
300 ms), so its p99 never met 200 ms, even at 2 requests per second. Its pool spins
between ops; on a busy host that is the likely cause. The quiet run also measures it with
spinning off.

The A/A pair (rung 28 at 8x2 twice in a row) gave 55 and 45 in round 1, as the load
rose from 8 to 34 and beyond, and 45 and 45 in round 2. On a busy host the floor is
one or two rate steps.
