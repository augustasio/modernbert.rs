# Rung 28 against the int8 baselines at their limits (2026-11-12)

The follow-up to `../cloud_turin_baseline_int8_quiet`, whose sweeps stopped at 50/s for
OpenVINO and 40/s for ONNX Runtime while both still met the budget. RunPod cpu5c, AMD
EPYC 9575F, commit a02d844, three rounds, same order each round, same gates. This host
was slower than the quiet run's: rung 28's worker p50 is 96 to 99 ms here against 80 to
81 there, and round 2 started at host load 114 (`session.log`). So compare within the
round, not with the quiet run's 70/s.

Best rate with p99 at or under 200 ms, and p99 at 60/s (ms):

| Stack, 8x2 | Round 1 | Round 2 | Round 3 | p99 at 60/s, r1/r2/r3 | Worker p50 at 60/s |
|---|---|---|---|---|---|
| rung 28 | 60 | 60 | 60 | 162 / 197 / 195 | 96 to 99 ms |
| OpenVINO static int8 | 60 | 55 | 60 | 182 / 229 / 191 | 101 to 105 ms |
| ONNX Runtime dynamic int8 | 20 | 30 | 40 | | 99 to 120 ms |
| PyTorch dynamic int8 | 40 | 30 | 30 | | 112 to 117 ms |

ONNX Runtime at 4x4 (its better layout in the quiet run): 40, 40, none.

What it says, with `../cloud_turin_baseline_int8_quiet`:

- **Rung 28 and OpenVINO's static int8 are within one rate step**, the A/A floor. Rung 28
  is ahead by about 4% in worker time and has the lower p99 at 60/s in two rounds of
  three. That is the honest size of the gap: small, and in our favour.
- **ONNX Runtime's dynamic int8 tops out near 40/s**, at 4x4 or 8x2.
- **PyTorch's dynamic int8 serves 30 to 40/s but fails accuracy** (Tier 3: 71.1% against
  84.8%), so it is not a usable configuration.
