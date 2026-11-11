# Rung 24 at two threads: inside the int8 matmuls and inside attention

RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 2026-11-11, two pods: 04:50Z at host load 34 to 41
(`session.log`), and 05:32Z at host load 13 to 23 (`session_attention.log`). A
`stage-times` build (analysis only), batch 1, 64 examples, `OMP_PLACES=cores`, two rounds
each. The rounds agree within 0.2 ms.

## Where a request goes at 2 threads (the 8x2 serving layout), and at 4

| Stage | t2 ms/example | t4 ms/example |
|---|---|---|
| Wqkv, attention Wo, Wi, MLP Wo (int8) | 66.4 | 51.9 |
| of which oneDNN | 56.6 | 42.3 |
| of which quantize | 6.0 | 3.8 |
| of which dequantize | 2.9 | 3.9 |
| attention | 21.1 | 12.3 |
| GELU gate | 4.9 | 2.5 |
| norms, adds, gather heads | 1.8 | 2.5 |
| **layers total** | **94.2** | **69.1** |

oneDNN's s8 matmuls are near the hardware's limit. They do about 110 billion int8
operations per example in 56.6 ms on two cores, about 77% of the VNNI peak. So the
levers left are outside them.

## Inside attention at 2 threads (thread-ms, summed over both threads)

| Part | thread-ms/example | per head call |
|---|---|---|
| scatter (RoPE, transpose K) | 5.2 | 12 us |
| scores, `Q K^T` (MKL sgemm) | 14.0 | 32 us |
| softmax | 9.4 | 22 us |
| `P V` (MKL sgemm) | 13.1 | 30 us |
| **total (about 2x the 21 ms wall)** | **41.7** | |

The softmax runs at about 1 ns per value. A vectorised exp should take a tenth of that.
The suspect is its row maximum, `fold(f32::max)`: one running maximum is a chain where
each step waits for the one before, so it cannot vectorise. The int8 row scales use the
same fold. Rung 25 tests this (`../cloud_turin_serve25/`).

Banding the local layers' attention (window 128) was considered and set aside. With
prompts of about 160 tokens, query tiles of 32 rows skip only about a quarter of the two
sgemms on two thirds of the layers, about 2.4 ms per request.

## Follow-up: the x86 code (2026-11-11, 06:41Z, a 9575F pod)

Rung 25 (16 running maxima) was exact but slower (`../cloud_turin_serve25/`), so the
code was read instead of guessed. `objdump` of rung 24's `eval_agnews` built with
`target-cpu=native`:

- no calls to `roundf`, `expf`, `fmaxf` or any other libm function anywhere;
- `ops::softmax_in_place`: 101 zmm (512-bit) instructions, so the exp, the maximum and
  the lane sum are already AVX-512;
- `fused::gelu_gate_row` and the int8 quantize closure: also zmm.

So the softmax's cost is the exp polynomial and three passes over rows of about 160
values, not a missed vectorisation. What is left there is a few percent of a request.
